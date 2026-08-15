"""Run the analytic PCA head-wise K2/V16 study from ``LongBenchAnalytic.md``.

The runner is stage-resumable.  It constructs three deterministic row-vector
rotations, evaluates held-out WikiText attention distortion, then reuses the
verified LongBench-E subset generation and official scoring implementation.

Typical invocation from the TurboQuant+ checkout::

    python -m experiments.spin_turboquant.longbench_analytic \
      --stage orchestrate \
      --model /path/to/Meta-Llama-3.1-8B-Instruct/snapshot \
      --longbench-repo ../LongBench_official \
      --data-dir ../LongBench_data/<revision>/data \
      --reference-subset-manifest experiments/spin_turboquant/results/longbench_subset/subset_manifest.json \
      --output-dir experiments/spin_turboquant/results/longbench_analytic
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import gc
import hashlib
import json
import math
import os
import random
import re
import statistics
import subprocess
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

import numpy as np
import torch
from datasets import DownloadConfig, load_dataset
from transformers import AutoTokenizer

from experiments.spin_turboquant import longbench as lb
from experiments.spin_turboquant.analytic_core import (
    apply_row_rope,
    attention_query_second_moment,
    normalized_hadamard,
    normalized_key_second_moment,
    principal_subspace_similarity,
    relative_frobenius_difference,
    rotation_checks,
    spectral_rotation,
    tensor_sha256,
    to_codec_rotations,
    weight_second_moment,
)
from experiments.spin_turboquant.core import (
    apply_codec,
    codebook_tensor,
    install_key_codec_hooks,
)
from experiments.spin_turboquant.run import capture_activations


SPECIFICATION_PATH = Path(__file__).resolve().parents[3] / "LongBenchAnalytic.md"
WIKITEXT_DATASET = "Salesforce/wikitext"
WIKITEXT_CONFIG = "wikitext-2-raw-v1"
WIKITEXT_REVISION = "b08601e04326c79dfdd32d625aee71d232d685c3"
CALIBRATION_SEED = 20_020_305
SEQUENCE_COUNT = 8
SEQUENCE_LENGTH = 512
CALIBRATION_TOKENS = SEQUENCE_COUNT * SEQUENCE_LENGTH
KEY_BITS = 2
VALUE_BITS = 16
SUBSPACE_RANKS = (16, 32, 64)


@dataclass(frozen=True)
class AnalyticCondition:
    method: str
    condition_id: str
    label: str
    artifact_filename: str

    @property
    def bit_width(self) -> int:
        return KEY_BITS

    @property
    def seed(self) -> None:
        return None


CONDITIONS = (
    AnalyticCondition(
        "wk_pca_h",
        "wk_pca_h_K2_V16",
        "Wk-PCA+H",
        "wk_pca_rotations.pt",
    ),
    AnalyticCondition(
        "activation_k_pca_h",
        "activation_k_pca_h_K2_V16",
        "Activation-K-PCA+H",
        "activation_k_pca_rotations.pt",
    ),
    AnalyticCondition(
        "attention_q_pca_h",
        "attention_q_pca_h_K2_V16",
        "Attention-aware-Q-PCA+H",
        "attention_q_pca_rotations.pt",
    ),
)
CONDITION_BY_ID = {condition.condition_id: condition for condition in CONDITIONS}
METHOD_TO_CONDITION = {condition.method: condition for condition in CONDITIONS}
PAIRWISE_COMPARISONS = (
    ("activation_k_pca_h", "wk_pca_h"),
    ("attention_q_pca_h", "wk_pca_h"),
    ("attention_q_pca_h", "activation_k_pca_h"),
)


@dataclass(frozen=True)
class WikiDocument:
    document_index: int
    row_start: int
    row_end: int
    title: str
    text: str
    document_id: str


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage",
        choices=(
            "validate",
            "rotations",
            "diagnostics",
            "smoke",
            "full",
            "report",
            "orchestrate",
        ),
        default="orchestrate",
    )
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--longbench-repo", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--reference-subset-manifest", type=Path, required=True)
    parser.add_argument("--reference-study-dir", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--condition", choices=tuple(CONDITION_BY_ID))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-context-length", type=int)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--batch-token-budget", type=int, default=32_768)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--dataset-revision", default=lb.LONG_BENCH_DATASET_REVISION)
    parser.add_argument("--longbench-commit", default=lb.LONG_BENCH_COMMIT)
    parser.add_argument("--wikitext-revision", default=WIKITEXT_REVISION)
    return parser.parse_args(argv)


def condition_by_id(condition_id: str) -> AnalyticCondition:
    try:
        return CONDITION_BY_ID[condition_id]
    except KeyError as error:
        raise ValueError(f"unknown analytic condition: {condition_id}") from error


def atomic_torch_save(payload: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> None:
    lb.write_jsonl(path, records)


def canonical_token_hash(token_ids: Sequence[int]) -> str:
    return lb.sha256_json([int(value) for value in token_ids])


_TOP_LEVEL_TITLE = re.compile(r"^\s*=\s+[^=].*?\s+=\s*$")


def group_wikitext_documents(rows: Sequence[str]) -> list[WikiDocument]:
    """Group raw WikiText rows into stable top-level article documents."""

    grouped: list[tuple[int, int, str, str]] = []
    current: list[str] = []
    row_start = 0
    title = ""
    for row_index, text in enumerate(rows):
        if _TOP_LEVEL_TITLE.match(text):
            if current:
                grouped.append((row_start, row_index - 1, title, "".join(current)))
            current = [text]
            row_start = row_index
            title = text.strip()
        elif current:
            current.append(text)
    if current:
        grouped.append((row_start, len(rows) - 1, title, "".join(current)))
    documents: list[WikiDocument] = []
    for document_index, (start, end, heading, text) in enumerate(grouped):
        documents.append(
            WikiDocument(
                document_index=document_index,
                row_start=start,
                row_end=end,
                title=heading,
                text=text,
                document_id=hashlib.sha256(text.encode("utf-8")).hexdigest(),
            )
        )
    return documents


def load_wikitext_split(
    split: str,
    *,
    revision: str,
) -> Any:
    download = DownloadConfig(local_files_only=True)
    return load_dataset(
        WIKITEXT_DATASET,
        WIKITEXT_CONFIG,
        split=split,
        revision=revision,
        download_config=download,
    )


def select_wikitext_sequences(
    tokenizer: Any,
    rows: Sequence[str],
    *,
    split: str,
    dataset_fingerprint: str,
    revision: str,
    seed: int = CALIBRATION_SEED,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Select eight unique documents and one deterministic 512-token span each."""

    tokenized: list[tuple[WikiDocument, list[int]]] = []
    for document in group_wikitext_documents(rows):
        tokens = tokenizer.encode(document.text, add_special_tokens=False)
        if len(tokens) >= SEQUENCE_LENGTH:
            tokenized.append((document, [int(value) for value in tokens]))
    if len(tokenized) < SEQUENCE_COUNT:
        raise RuntimeError(
            f"{split} has only {len(tokenized)} documents with {SEQUENCE_LENGTH} tokens"
        )
    generator = random.Random(seed)
    selected = generator.sample(tokenized, SEQUENCE_COUNT)
    sequences: list[torch.Tensor] = []
    sequence_rows: list[dict[str, Any]] = []
    for sequence_index, (document, tokens) in enumerate(selected):
        start = generator.randrange(0, len(tokens) - SEQUENCE_LENGTH + 1)
        stop = start + SEQUENCE_LENGTH
        selected_tokens = tokens[start:stop]
        sequences.append(torch.tensor(selected_tokens, dtype=torch.long))
        sequence_rows.append(
            {
                "sequence_index": sequence_index,
                "document_index": document.document_index,
                "document_id": document.document_id,
                "document_title": document.title,
                "dataset_row_start": document.row_start,
                "dataset_row_end": document.row_end,
                "document_token_count": len(tokens),
                "token_start_inclusive": start,
                "token_end_exclusive": stop,
                "token_count": len(selected_tokens),
                "token_sha256": canonical_token_hash(selected_tokens),
            }
        )
    stacked = torch.stack(sequences)
    manifest = {
        "schema_version": 1,
        "dataset": WIKITEXT_DATASET,
        "dataset_config": WIKITEXT_CONFIG,
        "dataset_revision": revision,
        "dataset_fingerprint": dataset_fingerprint,
        "split": split,
        "sampling_seed": seed,
        "sampling_method": (
            "group raw rows by top-level article title; retain documents with at "
            "least 512 tokenizer tokens; random.sample eight unique documents; "
            "draw one uniform contiguous 512-token span per sampled document with "
            "the same Python random.Random stream"
        ),
        "sequence_count": SEQUENCE_COUNT,
        "tokens_per_sequence": SEQUENCE_LENGTH,
        "total_tokens": int(stacked.numel()),
        "token_tensor_sha256": tensor_sha256(stacked),
        "sequences": sequence_rows,
    }
    return stacked, manifest


def ensure_input_artifacts(args: argparse.Namespace, assets: dict[str, Any]) -> dict[str, Any]:
    """Create or immutably verify calibration, held-out, and subset manifests."""

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    reference = lb.read_json(args.reference_subset_manifest)
    expected_subset = lb.build_subset_manifest(
        args.data_dir,
        dataset_revision=args.dataset_revision,
        data_hashes=assets["longbench"]["data_hashes"],
    )
    if reference != expected_subset:
        raise RuntimeError(
            "reference LongBench subset manifest does not match the pinned seed/data"
        )
    subset_path = output_dir / "longbench_subset_manifest.json"
    existing_subset = lb.read_json(subset_path)
    if existing_subset is None:
        lb.write_json(subset_path, expected_subset)
    elif existing_subset != expected_subset:
        raise RuntimeError(f"existing analytic subset manifest differs: {subset_path}")

    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    selected_payload: dict[str, torch.Tensor] = {}
    manifests: dict[str, dict[str, Any]] = {}
    for name, split in (("calibration", "train"), ("heldout", "validation")):
        dataset = load_wikitext_split(split, revision=args.wikitext_revision)
        rows = [str(row["text"]) for row in dataset]
        tokens, manifest = select_wikitext_sequences(
            tokenizer,
            rows,
            split=split,
            dataset_fingerprint=str(dataset._fingerprint),
            revision=args.wikitext_revision,
        )
        manifest["tokenizer_revision"] = assets["model"]["revision"]
        manifest["tokenizer_files"] = assets["model"]["tokenizer_files"]
        selected_payload[name] = tokens
        manifests[name] = manifest

    calibration_path = output_dir / "pca_calibration_manifest.json"
    heldout_path = output_dir / "heldout_manifest.json"
    for path, expected in (
        (calibration_path, manifests["calibration"]),
        (heldout_path, manifests["heldout"]),
    ):
        existing = lb.read_json(path)
        if existing is None:
            lb.write_json(path, expected)
        elif existing != expected:
            raise RuntimeError(f"existing data manifest differs from protocol: {path}")

    token_path = output_dir / "pca_sequences.pt"
    if token_path.exists():
        existing_tokens = torch.load(token_path, map_location="cpu", weights_only=True)
        if set(existing_tokens) != set(selected_payload) or any(
            not torch.equal(existing_tokens[key], value)
            for key, value in selected_payload.items()
        ):
            raise RuntimeError(f"existing token artifact differs from manifests: {token_path}")
    else:
        atomic_torch_save(selected_payload, token_path)

    return {
        "calibration": manifests["calibration"],
        "heldout": manifests["heldout"],
        "subset": expected_subset,
        "paths": {
            "calibration_manifest": str(calibration_path.resolve()),
            "heldout_manifest": str(heldout_path.resolve()),
            "longbench_subset_manifest": str(subset_path.resolve()),
            "tokens": str(token_path.resolve()),
        },
        "hashes": {
            "calibration_manifest": lb.sha256_json(manifests["calibration"]),
            "heldout_manifest": lb.sha256_json(manifests["heldout"]),
            "longbench_subset_manifest": lb.sha256_json(expected_subset),
            "tokens_file": lb.sha256_file(token_path),
        },
    }


def validate_static_assets(args: argparse.Namespace) -> dict[str, Any]:
    for label, path in {
        "model": args.model,
        "LongBench repository": args.longbench_repo,
        "LongBench data": args.data_dir,
        "reference subset manifest": args.reference_subset_manifest,
    }.items():
        if not path.exists():
            raise FileNotFoundError(f"{label} does not exist: {path}")
    if not SPECIFICATION_PATH.is_file():
        raise FileNotFoundError(SPECIFICATION_PATH)
    if str(args.device).startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    if args.batch_size < 1 or args.batch_token_budget < 1:
        raise ValueError("batch size and token budget must be positive")
    if args.bootstrap_samples < 100:
        raise ValueError("at least 100 bootstrap samples are required")

    actual_commit = lb.git_output(args.longbench_repo, "rev-parse", "HEAD")
    if actual_commit != args.longbench_commit:
        raise RuntimeError(
            f"LongBench checkout is {actual_commit}, expected {args.longbench_commit}"
        )
    prompts, maximum_generation_lengths = lb.load_official_configs(args.longbench_repo)
    dimensions = lb.model_dimensions(args.model)
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    if not tokenizer.chat_template:
        raise RuntimeError("the instruct tokenizer has no chat template")
    max_context_length = args.max_context_length or dimensions["max_position_embeddings"]
    if max_context_length > dimensions["max_position_embeddings"]:
        raise ValueError("max context length exceeds model configuration")
    if max_context_length <= max(maximum_generation_lengths.values()):
        raise ValueError("max context length leaves no generation room")

    data_counts: dict[str, int] = {}
    data_hashes: dict[str, str] = {}
    for task in lb.TASKS:
        path = args.data_dir / f"{task}_e.jsonl"
        if not path.is_file():
            raise FileNotFoundError(path)
        data_counts[task] = len(lb.load_examples(args.data_dir, task))
        data_hashes[task] = lb.sha256_file(path)

    tokenizer_files = [
        args.model / name
        for name in ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json")
        if (args.model / name).is_file()
    ]
    repository_root = Path(__file__).resolve().parents[2]
    implementation_files = {
        "analytic_runner": Path(__file__).resolve(),
        "analytic_core": Path(__file__).resolve().with_name("analytic_core.py"),
        "longbench_shared_runner": Path(__file__).resolve().with_name("longbench.py"),
        "spin_turboquant_core": Path(__file__).resolve().with_name("core.py"),
        "turboquant_codebook": repository_root / "turboquant" / "codebook.py",
    }
    centroids = codebook_tensor(KEY_BITS, dimensions["head_dim"], device="cpu").numpy()
    v1_dir = lb.official_v1_dir(args.longbench_repo)
    config_dir = v1_dir / "config"
    return {
        "specification": {
            "path": str(SPECIFICATION_PATH),
            "sha256": lb.sha256_file(SPECIFICATION_PATH),
        },
        "model": {
            "path": str(args.model),
            "revision": args.model.name,
            **dimensions,
            "tokenizer_files": {
                path.name: lb.sha256_file(path) for path in tokenizer_files
            },
        },
        "longbench": {
            "repository": str(args.longbench_repo),
            "commit": actual_commit,
            "dataset_revision": args.dataset_revision,
            "data_counts": data_counts,
            "data_hashes": data_hashes,
            "prompt_config_sha256": lb.sha256_file(config_dir / "dataset2prompt.json"),
            "max_generation_config_sha256": lb.sha256_file(
                config_dir / "dataset2maxlen.json"
            ),
        },
        "wikitext": {
            "dataset": WIKITEXT_DATASET,
            "config": WIKITEXT_CONFIG,
            "revision": args.wikitext_revision,
        },
        "codebook_sha256": hashlib.sha256(centroids.tobytes()).hexdigest(),
        "implementation_hashes": {
            name: lb.sha256_file(path) for name, path in implementation_files.items()
        },
        "max_context_length": int(max_context_length),
        "prompts": prompts,
        "maximum_generation_lengths": maximum_generation_lengths,
    }


def rotation_artifact_paths(output_dir: Path) -> dict[str, Path]:
    return {
        condition.method: output_dir / condition.artifact_filename
        for condition in CONDITIONS
    }


def validate_rotation_artifacts(
    args: argparse.Namespace,
    assets: dict[str, Any],
    *,
    deep: bool,
) -> dict[str, str]:
    expected_shape = (
        int(assets["model"]["num_hidden_layers"]),
        int(assets["model"]["num_key_value_heads"]),
        int(assets["model"]["head_dim"]),
        int(assets["model"]["head_dim"]),
    )
    hashes: dict[str, str] = {}
    for method, path in rotation_artifact_paths(args.output_dir).items():
        if not path.is_file():
            raise FileNotFoundError(path)
        hashes[method] = lb.sha256_file(path)
        if deep:
            artifact = torch.load(path, map_location="cpu", weights_only=True)
            rotations = artifact.get("rotations")
            if not isinstance(rotations, torch.Tensor) or tuple(rotations.shape) != expected_shape:
                raise RuntimeError(f"invalid rotation tensor in {path}")
            if rotations.dtype != torch.float64:
                raise RuntimeError(f"analytic rotation is not float64 in {path}")
            gram = rotations.transpose(-1, -2) @ rotations
            identity = torch.eye(expected_shape[-1], dtype=torch.float64)
            if float((gram - identity).abs().max()) > 1e-10:
                raise RuntimeError(f"non-orthogonal rotations in {path}")
    metadata_path = args.output_dir / "rotation_metadata.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(metadata_path)
    metadata = lb.read_json(metadata_path)
    if metadata.get("artifact_hashes") != hashes:
        raise RuntimeError("rotation metadata artifact hashes do not match files")
    return hashes


def write_study_config(
    args: argparse.Namespace,
    assets: dict[str, Any],
    inputs: dict[str, Any],
    status: str,
    **updates: Any,
) -> None:
    path = args.output_dir / "study_config.json"
    existing = lb.read_json(path, {})
    payload = {
        "created_at": existing.get("created_at", lb.utc_now()),
        "updated_at": lb.utc_now(),
        "status": status,
        "specification": assets["specification"],
        "model": assets["model"],
        "longbench": assets["longbench"],
        "wikitext": assets["wikitext"],
        "input_artifacts": inputs["paths"],
        "input_hashes": inputs["hashes"],
        "codebook_sha256": assets["codebook_sha256"],
        "implementation_hashes": assets["implementation_hashes"],
        "conditions": [condition.condition_id for condition in CONDITIONS],
        "condition_matrix": [
            {
                "condition_id": condition.condition_id,
                "method": condition.method,
                "spectral_target": condition.label,
                "key_bit_width": KEY_BITS,
                "value_bit_width": VALUE_BITS,
                "calibration_tokens": (
                    0 if condition.method == "wk_pca_h" else CALIBRATION_TOKENS
                ),
                "post_mixing": "fixed normalized Sylvester Hadamard H_128",
            }
            for condition in CONDITIONS
        ],
        "execution_order": [condition.condition_id for condition in CONDITIONS],
        "calibration_seed": CALIBRATION_SEED,
        "calibration_tokens": CALIBRATION_TOKENS,
        "heldout_tokens": CALIBRATION_TOKENS,
        "examples_per_condition": int(inputs["subset"]["example_count"]),
        "expected_total_predictions": int(inputs["subset"]["example_count"])
        * len(CONDITIONS),
        "bootstrap_samples": args.bootstrap_samples,
        "batch_size": args.batch_size,
        "batch_token_budget": args.batch_token_budget,
        **updates,
    }
    lb.write_json(path, payload)


def cleanup_cuda() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def compute_weight_covariances(model: Any, assets: dict[str, Any]) -> torch.Tensor:
    layers = int(assets["model"]["num_hidden_layers"])
    heads = int(assets["model"]["num_key_value_heads"])
    head_dim = int(assets["model"]["head_dim"])
    result = torch.empty((layers, heads, head_dim, head_dim), dtype=torch.float64)
    for layer_index, layer in enumerate(model.model.layers):
        result[layer_index] = weight_second_moment(
            layer.self_attn.k_proj.weight,
            num_kv_heads=heads,
        ).cpu()
        print(
            f"[{lb.utc_now()}] Wk covariance layer={layer_index + 1}/{layers}",
            flush=True,
        )
    return result


@torch.inference_mode()
def collect_calibration_sequence(
    model: Any,
    input_ids: torch.Tensor,
    *,
    half_index: int,
    activation_sums: torch.Tensor,
    attention_sums: torch.Tensor,
    assets: dict[str, Any],
    device: torch.device,
) -> None:
    """Stream one sequence through all layers and update float64 moments."""

    layers = model.model.layers
    kv_heads = int(assets["model"]["num_key_value_heads"])
    query_heads = int(model.config.num_attention_heads)
    head_dim = int(assets["model"]["head_dim"])
    rope: dict[str, torch.Tensor] = {}
    handles: list[Any] = []

    def rope_hook(
        _module: torch.nn.Module,
        _inputs: tuple[torch.Tensor, ...],
        output: tuple[torch.Tensor, torch.Tensor],
    ) -> None:
        rope["cos"] = output[0].detach()[0].to(dtype=torch.float64)
        rope["sin"] = output[1].detach()[0].to(dtype=torch.float64)

    handles.append(model.model.rotary_emb.register_forward_hook(rope_hook))
    for layer_index, layer in enumerate(layers):

        def key_hook(
            _module: torch.nn.Module,
            _inputs: tuple[torch.Tensor, ...],
            output: torch.Tensor,
            *,
            selected_layer: int = layer_index,
        ) -> None:
            keys = output.detach()[0].reshape(output.shape[1], kv_heads, head_dim)
            moment, _ = normalized_key_second_moment(keys)
            activation_sums[half_index, selected_layer].add_(moment)

        def query_hook(
            _module: torch.nn.Module,
            _inputs: tuple[torch.Tensor, ...],
            output: torch.Tensor,
            *,
            selected_layer: int = layer_index,
        ) -> None:
            if "cos" not in rope:
                raise RuntimeError("rotary embeddings were not captured before Q projection")
            queries = output.detach()[0].reshape(output.shape[1], query_heads, head_dim)
            moment, _ = attention_query_second_moment(
                queries,
                rope["cos"],
                rope["sin"],
                num_kv_heads=kv_heads,
            )
            attention_sums[half_index, selected_layer].add_(moment)

        handles.append(layer.self_attn.k_proj.register_forward_hook(key_hook))
        handles.append(layer.self_attn.q_proj.register_forward_hook(query_hook))

    try:
        model.model(input_ids=input_ids.unsqueeze(0).to(device), use_cache=False)
    finally:
        for handle in handles:
            handle.remove()


def write_covariance_archive(
    path: Path,
    covariances: dict[str, torch.Tensor],
    eigenvalues: dict[str, torch.Tensor],
) -> None:
    arrays: dict[str, np.ndarray] = {}
    for key, value in covariances.items():
        arrays[f"{key}_covariance"] = value.cpu().numpy()
    for key, value in eigenvalues.items():
        arrays[f"{key}_eigenvalues"] = value.cpu().numpy()
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    temporary.replace(path)


def _batched_rotation_checks(rotations: torch.Tensor) -> dict[str, torch.Tensor]:
    identity = torch.eye(rotations.shape[-1], dtype=torch.float64)
    error = rotations.transpose(-1, -2) @ rotations - identity
    sign, logabsdet = torch.linalg.slogdet(rotations)
    return {
        "orthogonality_frobenius": torch.linalg.matrix_norm(error, ord="fro"),
        "orthogonality_max_abs": error.abs().amax(dim=(-1, -2)),
        "determinant_sign": sign,
        "determinant_log_abs": logabsdet,
    }


def build_rotation_artifacts(
    args: argparse.Namespace,
    assets: dict[str, Any],
    inputs: dict[str, Any],
    *,
    wk_covariance: torch.Tensor,
    activation_sums: torch.Tensor,
    attention_sums: torch.Tensor,
    counts: torch.Tensor,
) -> None:
    if counts.tolist() != [CALIBRATION_TOKENS // 2, CALIBRATION_TOKENS // 2]:
        raise RuntimeError(f"calibration half counts are {counts.tolist()}")
    activation_halves = activation_sums / counts[:, None, None, None, None]
    attention_halves = attention_sums / counts[:, None, None, None, None]
    covariances = {
        "wk_pca_h": wk_covariance,
        "activation_k_pca_h": activation_sums.sum(dim=0) / counts.sum(),
        "attention_q_pca_h": attention_sums.sum(dim=0) / counts.sum(),
        "activation_k_pca_h_half1": activation_halves[0],
        "activation_k_pca_h_half2": activation_halves[1],
        "attention_q_pca_h_half1": attention_halves[0],
        "attention_q_pca_h_half2": attention_halves[1],
    }
    eigenvalues: dict[str, torch.Tensor] = {}
    eigenvectors: dict[str, torch.Tensor] = {}
    rotations: dict[str, torch.Tensor] = {}
    for name, covariance in covariances.items():
        print(f"[{lb.utc_now()}] eigendecomposition {name}", flush=True)
        values, vectors, matrix = spectral_rotation(covariance)
        eigenvalues[name] = values.cpu()
        eigenvectors[name] = vectors.cpu()
        rotations[name] = matrix.cpu()

    artifact_paths = rotation_artifact_paths(args.output_dir)
    for condition in CONDITIONS:
        method = condition.method
        payload = {
            "schema_version": 1,
            "condition_id": condition.condition_id,
            "method": method,
            "key_bit_width": KEY_BITS,
            "value_bit_width": VALUE_BITS,
            "row_vector_convention": "z = k @ R; reconstructed = z_hat @ R.T",
            "spectral_construction": "C = U diag(lambda) U.T; R = U @ H_128",
            "eigenvalue_order": "descending",
            "eigenvector_sign": "largest absolute component is positive",
            "hadamard": "normalized Sylvester H_128; no random signs or permutations",
            "calibration_manifest_sha256": (
                None
                if method == "wk_pca_h"
                else inputs["hashes"]["calibration_manifest"]
            ),
            "covariance_sha256": tensor_sha256(covariances[method]),
            "eigenvalues_sha256": tensor_sha256(eigenvalues[method]),
            "rotations_sha256": tensor_sha256(rotations[method]),
            "rotations": rotations[method],
        }
        atomic_torch_save(payload, artifact_paths[method])

    half_path = args.output_dir / "calibration_half_rotations.pt"
    atomic_torch_save(
        {
            "activation_k_pca_h": torch.stack(
                (
                    rotations["activation_k_pca_h_half1"],
                    rotations["activation_k_pca_h_half2"],
                )
            ),
            "attention_q_pca_h": torch.stack(
                (
                    rotations["attention_q_pca_h_half1"],
                    rotations["attention_q_pca_h_half2"],
                )
            ),
            "row_vector_convention": "z = k @ R",
        },
        half_path,
    )
    archive_path = args.output_dir / "covariance_eigenvalues.npz"
    write_covariance_archive(archive_path, covariances, eigenvalues)

    artifact_hashes = {
        method: lb.sha256_file(path) for method, path in artifact_paths.items()
    }
    hadamard_hash = tensor_sha256(normalized_hadamard(128))
    metadata_rows: list[dict[str, Any]] = []
    for condition in CONDITIONS:
        method = condition.method
        checks = _batched_rotation_checks(rotations[method])
        for layer in range(rotations[method].shape[0]):
            for head in range(rotations[method].shape[1]):
                row = {
                    "condition_id": condition.condition_id,
                    "method": method,
                    "layer": layer,
                    "kv_head": head,
                    "construction_method": condition.label,
                    "calibration_manifest_sha256": (
                        None
                        if method == "wk_pca_h"
                        else inputs["hashes"]["calibration_manifest"]
                    ),
                    "covariance_sha256": tensor_sha256(covariances[method][layer, head]),
                    "eigenvalues_sha256": tensor_sha256(eigenvalues[method][layer, head]),
                    "eigenvectors_sha256": tensor_sha256(eigenvectors[method][layer, head]),
                    "hadamard_convention": "normalized Sylvester H_128",
                    "hadamard_sha256": hadamard_hash,
                    "rotation_sha256": tensor_sha256(rotations[method][layer, head]),
                    "final_artifact_sha256": artifact_hashes[method],
                    "orthogonality_frobenius": float(
                        checks["orthogonality_frobenius"][layer, head]
                    ),
                    "orthogonality_max_abs": float(
                        checks["orthogonality_max_abs"][layer, head]
                    ),
                    "determinant_sign": float(checks["determinant_sign"][layer, head]),
                    "determinant_log_abs": float(
                        checks["determinant_log_abs"][layer, head]
                    ),
                    "shape": [128, 128],
                    "dtype": "torch.float64",
                }
                if row["orthogonality_max_abs"] > 1e-10:
                    raise RuntimeError(
                        f"orthogonality gate failed: {method} layer={layer} head={head}"
                    )
                if abs(row["determinant_log_abs"]) > 1e-9:
                    raise RuntimeError(
                        f"determinant gate failed: {method} layer={layer} head={head}"
                    )
                metadata_rows.append(row)
    metadata = {
        "schema_version": 1,
        "generated_at": lb.utc_now(),
        "specification": assets["specification"],
        "model": assets["model"],
        "calibration_manifest_sha256": inputs["hashes"]["calibration_manifest"],
        "covariance_archive_sha256": lb.sha256_file(archive_path),
        "half_rotation_artifact_sha256": lb.sha256_file(half_path),
        "artifact_hashes": artifact_hashes,
        "row_vector_convention": "z = k @ R; reconstructed = z_hat @ R.T",
        "eigendecomposition_dtype": "torch.float64",
        "rows": metadata_rows,
    }
    lb.write_json(args.output_dir / "rotation_metadata.json", metadata)

    stability_rows: list[dict[str, Any]] = []
    for method in ("activation_k_pca_h", "attention_q_pca_h"):
        first_covariance = covariances[f"{method}_half1"]
        second_covariance = covariances[f"{method}_half2"]
        first_vectors = eigenvectors[f"{method}_half1"]
        second_vectors = eigenvectors[f"{method}_half2"]
        head_rows: list[dict[str, Any]] = []
        for layer in range(first_covariance.shape[0]):
            for head in range(first_covariance.shape[1]):
                row = {
                    "method": method,
                    "scope": "head",
                    "layer": layer,
                    "kv_head": head,
                    "covariance_relative_frobenius_difference": relative_frobenius_difference(
                        first_covariance[layer, head], second_covariance[layer, head]
                    ),
                    **{
                        f"principal_subspace_similarity_top{rank}": principal_subspace_similarity(
                            first_vectors[layer, head], second_vectors[layer, head], rank
                        )
                        for rank in SUBSPACE_RANKS
                    },
                    "heldout_half1_normalized_key_mse": "",
                    "heldout_half2_normalized_key_mse": "",
                    "heldout_half2_minus_half1_normalized_key_mse": "",
                }
                head_rows.append(row)
        stability_rows.extend(head_rows)
        stability_rows.append(
            {
                "method": method,
                "scope": "overall",
                "layer": "",
                "kv_head": "",
                "covariance_relative_frobenius_difference": statistics.fmean(
                    float(row["covariance_relative_frobenius_difference"])
                    for row in head_rows
                ),
                **{
                    f"principal_subspace_similarity_top{rank}": statistics.fmean(
                        float(row[f"principal_subspace_similarity_top{rank}"])
                        for row in head_rows
                    )
                    for rank in SUBSPACE_RANKS
                },
                "heldout_half1_normalized_key_mse": "",
                "heldout_half2_normalized_key_mse": "",
                "heldout_half2_minus_half1_normalized_key_mse": "",
            }
        )
    lb.write_csv(args.output_dir / "calibration_stability.csv", stability_rows)


def rotation_stage(
    args: argparse.Namespace,
    assets: dict[str, Any],
    inputs: dict[str, Any],
) -> None:
    required = [
        *rotation_artifact_paths(args.output_dir).values(),
        args.output_dir / "covariance_eigenvalues.npz",
        args.output_dir / "rotation_metadata.json",
        args.output_dir / "calibration_stability.csv",
        args.output_dir / "calibration_half_rotations.pt",
    ]
    if all(path.is_file() for path in required):
        validate_rotation_artifacts(args, assets, deep=True)
        print("analytic rotation artifacts already complete", flush=True)
        return

    device = torch.device(args.device)
    tokens = torch.load(
        args.output_dir / "pca_sequences.pt", map_location="cpu", weights_only=True
    )["calibration"]
    if tuple(tokens.shape) != (SEQUENCE_COUNT, SEQUENCE_LENGTH):
        raise RuntimeError(f"unexpected calibration token shape: {tuple(tokens.shape)}")
    model = lb.load_model(args.model, device)
    checkpoint_path = args.output_dir / "calibration_moments.pt"
    moment_protocol = lb.sha256_json(
        {
            "model": assets["model"],
            "calibration_manifest": inputs["hashes"]["calibration_manifest"],
            "implementation_hashes": assets["implementation_hashes"],
            "dtype": "torch.float64",
            "attention_target": "future-query-normalized pre-RoPE effective query second moment",
        }
    )
    if checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        if checkpoint.get("protocol_sha256") != moment_protocol:
            raise RuntimeError("calibration moment checkpoint uses a different protocol")
        processed = int(checkpoint["processed_sequences"])
        wk_covariance = checkpoint["wk_covariance"].double()
        activation_sums = checkpoint["activation_sums"].to(device=device, dtype=torch.float64)
        attention_sums = checkpoint["attention_sums"].to(device=device, dtype=torch.float64)
        counts = checkpoint["counts"].long()
    else:
        processed = 0
        wk_covariance = compute_weight_covariances(model, assets)
        shape = (
            2,
            int(assets["model"]["num_hidden_layers"]),
            int(assets["model"]["num_key_value_heads"]),
            int(assets["model"]["head_dim"]),
            int(assets["model"]["head_dim"]),
        )
        activation_sums = torch.zeros(shape, dtype=torch.float64, device=device)
        attention_sums = torch.zeros(shape, dtype=torch.float64, device=device)
        counts = torch.zeros(2, dtype=torch.long)

    try:
        for sequence_index in range(processed, SEQUENCE_COUNT):
            half_index = 0 if sequence_index < SEQUENCE_COUNT // 2 else 1
            started = time.perf_counter()
            collect_calibration_sequence(
                model,
                tokens[sequence_index],
                half_index=half_index,
                activation_sums=activation_sums,
                attention_sums=attention_sums,
                assets=assets,
                device=device,
            )
            counts[half_index] += SEQUENCE_LENGTH
            checkpoint = {
                "protocol_sha256": moment_protocol,
                "processed_sequences": sequence_index + 1,
                "wk_covariance": wk_covariance,
                "activation_sums": activation_sums.cpu(),
                "attention_sums": attention_sums.cpu(),
                "counts": counts,
            }
            atomic_torch_save(checkpoint, checkpoint_path)
            activation_sums = checkpoint["activation_sums"].to(device)
            attention_sums = checkpoint["attention_sums"].to(device)
            print(
                f"[{lb.utc_now()}] calibration sequence={sequence_index + 1}/{SEQUENCE_COUNT} "
                f"half={half_index + 1} seconds={time.perf_counter() - started:.3f}",
                flush=True,
            )
    finally:
        del model
        cleanup_cuda()

    build_rotation_artifacts(
        args,
        assets,
        inputs,
        wk_covariance=wk_covariance,
        activation_sums=activation_sums.cpu(),
        attention_sums=attention_sums.cpu(),
        counts=counts,
    )
    validate_rotation_artifacts(args, assets, deep=True)
    print(f"[{lb.utc_now()}] analytic rotations complete", flush=True)


def new_diagnostic_accumulator(assets: dict[str, Any], protocol_hash: str) -> dict[str, Any]:
    methods = len(CONDITIONS)
    layers = int(assets["model"]["num_hidden_layers"])
    heads = int(assets["model"]["num_key_value_heads"])
    head_dim = int(assets["model"]["head_dim"])
    head_shape = (methods, layers, heads)
    return {
        "protocol_sha256": protocol_hash,
        "processed_sequences": 0,
        "token_count": 0,
        "logit_values_per_head": 0,
        "query_rows_per_head": 0,
        "output_values_per_head": 0,
        "normalized_key_sse": torch.zeros(head_shape, dtype=torch.float64),
        "norm_restored_key_sse": torch.zeros(head_shape, dtype=torch.float64),
        "key_energy": torch.zeros(head_shape, dtype=torch.float64),
        "rotated_channel_sumsq": torch.zeros(
            (*head_shape, head_dim), dtype=torch.float64
        ),
        "maximum_channel_magnitude": torch.zeros(head_shape, dtype=torch.float64),
        "attention_logit_sse": torch.zeros(head_shape, dtype=torch.float64),
        "attention_kl_sum": torch.zeros(head_shape, dtype=torch.float64),
        "attention_output_sse": torch.zeros(head_shape, dtype=torch.float64),
        "half_normalized_key_sse": torch.zeros(
            (2, 2, layers, heads), dtype=torch.float64
        ),
    }


def load_row_rotations(args: argparse.Namespace) -> dict[str, torch.Tensor]:
    result: dict[str, torch.Tensor] = {}
    for condition in CONDITIONS:
        artifact = torch.load(
            args.output_dir / condition.artifact_filename,
            map_location="cpu",
            weights_only=True,
        )
        result[condition.method] = artifact["rotations"].double()
    return result


@torch.inference_mode()
def accumulate_diagnostic_sequence(
    accumulator: dict[str, Any],
    capture: dict[str, torch.Tensor],
    rotations: dict[str, torch.Tensor],
    half_rotations: dict[str, torch.Tensor],
    *,
    centroids: torch.Tensor,
    device: torch.device,
) -> None:
    queries = capture["q"]
    keys = capture["k"]
    values = capture["v"]
    cos = capture["cos"]
    sin = capture["sin"]
    layers, tokens, query_heads, head_dim = queries.shape
    kv_heads = keys.shape[2]
    if keys.shape != values.shape or keys.shape[:2] != (layers, tokens):
        raise RuntimeError("held-out Q/K/V capture shapes are inconsistent")
    if query_heads % kv_heads:
        raise RuntimeError("query/KV head ratio is not integral")
    group_size = query_heads // kv_heads
    causal = torch.ones((tokens, tokens), dtype=torch.bool, device=device).tril()
    causal_float = causal.to(dtype=torch.float32).unsqueeze(0)
    scale = 1.0 / math.sqrt(head_dim)

    for layer in range(layers):
        q = queries[layer].to(device=device, dtype=torch.float32)
        k = keys[layer].to(device=device, dtype=torch.float32)
        v = values[layer].to(device=device, dtype=torch.float32)
        rope_cos = cos[:tokens].to(device=device, dtype=torch.float32)
        rope_sin = sin[:tokens].to(device=device, dtype=torch.float32)
        q_post = apply_row_rope(q, rope_cos, rope_sin).permute(1, 0, 2)
        k_post = apply_row_rope(k, rope_cos, rope_sin).permute(1, 0, 2)
        key_repeated = k_post.repeat_interleave(group_size, dim=0)
        value_repeated = v.permute(1, 0, 2).repeat_interleave(group_size, dim=0)
        reference_logits = torch.matmul(
            q_post, key_repeated.transpose(-1, -2)
        ) * scale
        masked_reference = reference_logits.masked_fill(
            ~causal.unsqueeze(0), float("-inf")
        )
        reference_probabilities = torch.softmax(masked_reference, dim=-1)
        reference_output = torch.matmul(reference_probabilities, value_repeated)

        for method_index, condition in enumerate(CONDITIONS):
            row_rotation = rotations[condition.method][layer].to(
                device=device, dtype=torch.float32
            )
            codec_rotation = to_codec_rotations(row_rotation)
            head_first_keys = k.permute(1, 0, 2)
            reconstructed = apply_codec(
                head_first_keys,
                codec_rotation,
                centroids,
                norm_correction=False,
            )
            norms = torch.linalg.vector_norm(
                head_first_keys, dim=-1, keepdim=True
            )
            normalized = head_first_keys / norms.clamp_min(
                torch.finfo(torch.float32).eps
            )
            reconstructed_normalized = reconstructed / norms.clamp_min(
                torch.finfo(torch.float32).eps
            )
            normalized_error = reconstructed_normalized - normalized
            original_error = reconstructed - head_first_keys
            accumulator["normalized_key_sse"][method_index, layer].add_(
                normalized_error.square().sum(dim=(1, 2)).double().cpu()
            )
            accumulator["norm_restored_key_sse"][method_index, layer].add_(
                original_error.square().sum(dim=(1, 2)).double().cpu()
            )
            accumulator["key_energy"][method_index, layer].add_(
                head_first_keys.square().sum(dim=(1, 2)).double().cpu()
            )
            rotated = torch.matmul(normalized, row_rotation)
            accumulator["rotated_channel_sumsq"][method_index, layer].add_(
                rotated.square().sum(dim=1).double().cpu()
            )
            maxima = rotated.abs().amax(dim=(1, 2)).double().cpu()
            accumulator["maximum_channel_magnitude"][method_index, layer] = torch.maximum(
                accumulator["maximum_channel_magnitude"][method_index, layer], maxima
            )

            reconstructed_post = apply_row_rope(
                reconstructed.permute(1, 0, 2), rope_cos, rope_sin
            ).permute(1, 0, 2)
            reconstructed_repeated = reconstructed_post.repeat_interleave(
                group_size, dim=0
            )
            reconstructed_logits = torch.matmul(
                q_post, reconstructed_repeated.transpose(-1, -2)
            ) * scale
            logit_error = reconstructed_logits - reference_logits
            logit_sse = (
                logit_error.square() * causal_float
            ).reshape(kv_heads, group_size, tokens, tokens).sum(dim=(1, 2, 3))
            accumulator["attention_logit_sse"][method_index, layer].add_(
                logit_sse.double().cpu()
            )

            masked_reconstructed = reconstructed_logits.masked_fill(
                ~causal.unsqueeze(0), float("-inf")
            )
            reconstructed_probabilities = torch.softmax(
                masked_reconstructed, dim=-1
            )
            kl = torch.sum(
                reference_probabilities
                * (
                    torch.log(reference_probabilities.clamp_min(1e-12))
                    - torch.log(reconstructed_probabilities.clamp_min(1e-12))
                ),
                dim=-1,
            )
            kl_by_head = kl.reshape(kv_heads, group_size, tokens).sum(dim=(1, 2))
            accumulator["attention_kl_sum"][method_index, layer].add_(
                kl_by_head.double().cpu()
            )
            reconstructed_output = torch.matmul(
                reconstructed_probabilities, value_repeated
            )
            output_error = reconstructed_output - reference_output
            output_sse = output_error.square().reshape(
                kv_heads, group_size, tokens, head_dim
            ).sum(dim=(1, 2, 3))
            accumulator["attention_output_sse"][method_index, layer].add_(
                output_sse.double().cpu()
            )

        for calibration_method_index, method in enumerate(
            ("activation_k_pca_h", "attention_q_pca_h")
        ):
            for half_index in range(2):
                row_rotation = half_rotations[method][half_index, layer].to(
                    device=device, dtype=torch.float32
                )
                reconstructed = apply_codec(
                    k.permute(1, 0, 2),
                    to_codec_rotations(row_rotation),
                    centroids,
                    norm_correction=False,
                )
                norms = torch.linalg.vector_norm(
                    k.permute(1, 0, 2), dim=-1, keepdim=True
                )
                normalized = k.permute(1, 0, 2) / norms.clamp_min(
                    torch.finfo(torch.float32).eps
                )
                reconstructed_normalized = reconstructed / norms.clamp_min(
                    torch.finfo(torch.float32).eps
                )
                accumulator["half_normalized_key_sse"][
                    calibration_method_index, half_index, layer
                ].add_(
                    (reconstructed_normalized - normalized)
                    .square()
                    .sum(dim=(1, 2))
                    .double()
                    .cpu()
                )

        print(
            f"[{lb.utc_now()}] held-out diagnostics layer={layer + 1}/{layers}",
            flush=True,
        )

    accumulator["token_count"] += tokens
    accumulator["logit_values_per_head"] += group_size * tokens * (tokens + 1) // 2
    accumulator["query_rows_per_head"] += group_size * tokens
    accumulator["output_values_per_head"] += group_size * tokens * head_dim


def finalize_diagnostic_rows(
    accumulator: dict[str, Any], assets: dict[str, Any]
) -> list[dict[str, Any]]:
    layers = int(assets["model"]["num_hidden_layers"])
    heads = int(assets["model"]["num_key_value_heads"])
    head_dim = int(assets["model"]["head_dim"])
    tokens = int(accumulator["token_count"])
    key_values_per_head = tokens * head_dim
    logit_values = int(accumulator["logit_values_per_head"])
    query_rows = int(accumulator["query_rows_per_head"])
    output_values = int(accumulator["output_values_per_head"])
    rows: list[dict[str, Any]] = []
    for method_index, condition in enumerate(CONDITIONS):
        channel_sumsq = accumulator["rotated_channel_sumsq"][method_index]
        channel_moments = channel_sumsq / max(tokens, 1)
        rms = torch.sqrt(channel_sumsq.sum(dim=-1) / max(key_values_per_head, 1))
        max_over_rms = accumulator["maximum_channel_magnitude"][method_index] / rms.clamp_min(
            1e-30
        )
        channel_ratio = channel_moments.amax(dim=-1) / channel_moments.mean(
            dim=-1
        ).clamp_min(1e-30)
        for layer in range(layers):
            for head in range(heads):
                rows.append(
                    {
                        "condition_id": condition.condition_id,
                        "method": condition.method,
                        "scope": "head",
                        "layer": layer,
                        "kv_head": head,
                        "key_vectors": tokens,
                        "normalized_key_mse": float(
                            accumulator["normalized_key_sse"][method_index, layer, head]
                            / key_values_per_head
                        ),
                        "norm_restored_key_mse": float(
                            accumulator["norm_restored_key_sse"][
                                method_index, layer, head
                            ]
                            / key_values_per_head
                        ),
                        "relative_key_mse": float(
                            accumulator["norm_restored_key_sse"][
                                method_index, layer, head
                            ]
                            / accumulator["key_energy"][method_index, layer, head].clamp_min(
                                1e-30
                            )
                        ),
                        "maximum_channel_magnitude": float(
                            accumulator["maximum_channel_magnitude"][
                                method_index, layer, head
                            ]
                        ),
                        "max_over_rms": float(max_over_rms[layer, head]),
                        "channel_second_moment_max_over_mean": float(
                            channel_ratio[layer, head]
                        ),
                        "attention_pre_softmax_logit_mse": float(
                            accumulator["attention_logit_sse"][
                                method_index, layer, head
                            ]
                            / logit_values
                        ),
                        "attention_probability_kl": float(
                            accumulator["attention_kl_sum"][
                                method_index, layer, head
                            ]
                            / query_rows
                        ),
                        "attention_output_mse": float(
                            accumulator["attention_output_sse"][
                                method_index, layer, head
                            ]
                            / output_values
                        ),
                    }
                )
        total_heads = layers * heads
        overall_channel_moments = channel_sumsq.reshape(-1, head_dim).sum(dim=0) / (
            total_heads * max(tokens, 1)
        )
        overall_rms = torch.sqrt(
            channel_sumsq.sum() / (total_heads * max(key_values_per_head, 1))
        )
        rows.insert(
            len(rows) - total_heads,
            {
                "condition_id": condition.condition_id,
                "method": condition.method,
                "scope": "overall",
                "layer": "",
                "kv_head": "",
                "key_vectors": total_heads * tokens,
                "normalized_key_mse": float(
                    accumulator["normalized_key_sse"][method_index].sum()
                    / (total_heads * key_values_per_head)
                ),
                "norm_restored_key_mse": float(
                    accumulator["norm_restored_key_sse"][method_index].sum()
                    / (total_heads * key_values_per_head)
                ),
                "relative_key_mse": float(
                    accumulator["norm_restored_key_sse"][method_index].sum()
                    / accumulator["key_energy"][method_index].sum().clamp_min(1e-30)
                ),
                "maximum_channel_magnitude": float(
                    accumulator["maximum_channel_magnitude"][method_index].max()
                ),
                "max_over_rms": float(
                    accumulator["maximum_channel_magnitude"][method_index].max()
                    / overall_rms.clamp_min(1e-30)
                ),
                "channel_second_moment_max_over_mean": float(
                    overall_channel_moments.max()
                    / overall_channel_moments.mean().clamp_min(1e-30)
                ),
                "attention_pre_softmax_logit_mse": float(
                    accumulator["attention_logit_sse"][method_index].sum()
                    / (total_heads * logit_values)
                ),
                "attention_probability_kl": float(
                    accumulator["attention_kl_sum"][method_index].sum()
                    / (total_heads * query_rows)
                ),
                "attention_output_mse": float(
                    accumulator["attention_output_sse"][method_index].sum()
                    / (total_heads * output_values)
                ),
            },
        )
    return rows


def update_stability_with_heldout(
    args: argparse.Namespace,
    accumulator: dict[str, Any],
    assets: dict[str, Any],
) -> None:
    path = args.output_dir / "calibration_stability.csv"
    rows = lb.read_csv(path)
    layers = int(assets["model"]["num_hidden_layers"])
    heads = int(assets["model"]["num_key_value_heads"])
    head_dim = int(assets["model"]["head_dim"])
    tokens = int(accumulator["token_count"])
    denominator = tokens * head_dim
    for row in rows:
        method_index = (
            0 if row["method"] == "activation_k_pca_h" else 1
        )
        if row["scope"] == "head":
            layer = int(row["layer"])
            head = int(row["kv_head"])
            values = [
                float(
                    accumulator["half_normalized_key_sse"][
                        method_index, half, layer, head
                    ]
                    / denominator
                )
                for half in range(2)
            ]
        else:
            values = [
                float(
                    accumulator["half_normalized_key_sse"][method_index, half].sum()
                    / (layers * heads * denominator)
                )
                for half in range(2)
            ]
        row["heldout_half1_normalized_key_mse"] = values[0]
        row["heldout_half2_normalized_key_mse"] = values[1]
        row["heldout_half2_minus_half1_normalized_key_mse"] = values[1] - values[0]
    lb.write_csv(path, rows)


def diagnostics_stage(
    args: argparse.Namespace,
    assets: dict[str, Any],
    inputs: dict[str, Any],
) -> None:
    diagnostic_path = args.output_dir / "heldout_diagnostics.csv"
    metadata_path = args.output_dir / "heldout_diagnostics.json"
    if diagnostic_path.is_file() and metadata_path.is_file():
        rows = lb.read_csv(diagnostic_path)
        if len(rows) == len(CONDITIONS) * (
            1
            + int(assets["model"]["num_hidden_layers"])
            * int(assets["model"]["num_key_value_heads"])
        ):
            print("held-out diagnostics already complete", flush=True)
            return

    rotation_hashes = validate_rotation_artifacts(args, assets, deep=True)
    protocol_hash = lb.sha256_json(
        {
            "heldout_manifest": inputs["hashes"]["heldout_manifest"],
            "rotation_hashes": rotation_hashes,
            "codebook_sha256": assets["codebook_sha256"],
            "implementation_hashes": assets["implementation_hashes"],
            "metrics": [
                "normalized Key MSE",
                "norm-restored Key MSE",
                "relative Key MSE",
                "maximum channel magnitude and max/RMS",
                "channel second-moment max/mean",
                "attention pre-softmax logit MSE",
                "KL(p_BF16 || p_quantized)",
                "attention output MSE with BF16 values",
            ],
        }
    )
    checkpoint_path = args.output_dir / "heldout_diagnostic_accumulators.pt"
    if checkpoint_path.exists():
        accumulator = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        if accumulator.get("protocol_sha256") != protocol_hash:
            raise RuntimeError("held-out diagnostic checkpoint uses a different protocol")
    else:
        accumulator = new_diagnostic_accumulator(assets, protocol_hash)

    sequences = torch.load(
        args.output_dir / "pca_sequences.pt", map_location="cpu", weights_only=True
    )["heldout"]
    rotations = load_row_rotations(args)
    half_payload = torch.load(
        args.output_dir / "calibration_half_rotations.pt",
        map_location="cpu",
        weights_only=True,
    )
    half_rotations = {
        method: half_payload[method].double()
        for method in ("activation_k_pca_h", "attention_q_pca_h")
    }
    device = torch.device(args.device)
    centroids = codebook_tensor(KEY_BITS, 128, device=device)
    model = lb.load_model(args.model, device)
    try:
        for sequence_index in range(
            int(accumulator["processed_sequences"]), SEQUENCE_COUNT
        ):
            started = time.perf_counter()
            capture = capture_activations(
                model,
                sequences[sequence_index],
                capture_queries_and_values=True,
                device=device,
            )
            accumulate_diagnostic_sequence(
                accumulator,
                capture,
                rotations,
                half_rotations,
                centroids=centroids,
                device=device,
            )
            accumulator["processed_sequences"] = sequence_index + 1
            atomic_torch_save(accumulator, checkpoint_path)
            del capture
            cleanup_cuda()
            print(
                f"[{lb.utc_now()}] held-out sequence={sequence_index + 1}/{SEQUENCE_COUNT} "
                f"seconds={time.perf_counter() - started:.3f}",
                flush=True,
            )
    finally:
        del model
        cleanup_cuda()

    if int(accumulator["token_count"]) != CALIBRATION_TOKENS:
        raise RuntimeError(
            f"held-out diagnostic token count is {accumulator['token_count']}"
        )
    rows = finalize_diagnostic_rows(accumulator, assets)
    if any(
        not math.isfinite(float(row[field]))
        for row in rows
        for field in (
            "normalized_key_mse",
            "norm_restored_key_mse",
            "relative_key_mse",
            "maximum_channel_magnitude",
            "max_over_rms",
            "channel_second_moment_max_over_mean",
            "attention_pre_softmax_logit_mse",
            "attention_probability_kl",
            "attention_output_mse",
        )
    ):
        raise RuntimeError("held-out diagnostics contain non-finite metrics")
    lb.write_csv(diagnostic_path, rows)
    update_stability_with_heldout(args, accumulator, assets)
    lb.write_json(
        metadata_path,
        {
            "generated_at": lb.utc_now(),
            "protocol_sha256": protocol_hash,
            "heldout_manifest_sha256": inputs["hashes"]["heldout_manifest"],
            "rotation_artifact_hashes": rotation_hashes,
            "sequences": SEQUENCE_COUNT,
            "tokens_per_sequence": SEQUENCE_LENGTH,
            "total_tokens": CALIBRATION_TOKENS,
            "attention_logits": "scaled QK.T/sqrt(128), causal positions only",
            "attention_kl": "mean KL(p_BF16 || p_quantized) over layers, GQA query heads, and query positions",
            "attention_output": "values remain BF16; only attention probabilities differ",
            "channel_metrics": "computed on normalized rotated Key z = k_hat @ R",
            "optional_tinystories_domain_shift": "not run; optional in specification",
        },
    )
    print(f"[{lb.utc_now()}] held-out diagnostics complete", flush=True)


def load_input_artifacts(args: argparse.Namespace, assets: dict[str, Any]) -> dict[str, Any]:
    calibration = lb.read_json(args.output_dir / "pca_calibration_manifest.json")
    heldout = lb.read_json(args.output_dir / "heldout_manifest.json")
    subset = lb.read_json(args.output_dir / "longbench_subset_manifest.json")
    if calibration is None or heldout is None or subset is None:
        raise FileNotFoundError("input manifests are incomplete; run --stage validate")
    expected_subset = lb.build_subset_manifest(
        args.data_dir,
        dataset_revision=args.dataset_revision,
        data_hashes=assets["longbench"]["data_hashes"],
    )
    if subset != expected_subset or lb.read_json(args.reference_subset_manifest) != subset:
        raise RuntimeError("analytic LongBench manifest no longer matches the reference")
    for manifest, split in ((calibration, "train"), (heldout, "validation")):
        if (
            manifest.get("dataset_revision") != args.wikitext_revision
            or manifest.get("split") != split
            or int(manifest.get("sampling_seed", -1)) != CALIBRATION_SEED
            or int(manifest.get("sequence_count", -1)) != SEQUENCE_COUNT
            or int(manifest.get("tokens_per_sequence", -1)) != SEQUENCE_LENGTH
            or int(manifest.get("total_tokens", -1)) != CALIBRATION_TOKENS
        ):
            raise RuntimeError(f"invalid saved WikiText {split} manifest")
    token_path = args.output_dir / "pca_sequences.pt"
    tokens = torch.load(token_path, map_location="cpu", weights_only=True)
    for name, manifest in (("calibration", calibration), ("heldout", heldout)):
        if tuple(tokens[name].shape) != (SEQUENCE_COUNT, SEQUENCE_LENGTH):
            raise RuntimeError(f"invalid {name} token tensor shape")
        if tensor_sha256(tokens[name]) != manifest["token_tensor_sha256"]:
            raise RuntimeError(f"{name} token tensor hash differs from its manifest")
        for row, sequence in zip(manifest["sequences"], tokens[name]):
            if canonical_token_hash(sequence.tolist()) != row["token_sha256"]:
                raise RuntimeError(f"{name} sequence hash mismatch")
    paths = {
        "calibration_manifest": str(
            (args.output_dir / "pca_calibration_manifest.json").resolve()
        ),
        "heldout_manifest": str((args.output_dir / "heldout_manifest.json").resolve()),
        "longbench_subset_manifest": str(
            (args.output_dir / "longbench_subset_manifest.json").resolve()
        ),
        "tokens": str(token_path.resolve()),
    }
    return {
        "calibration": calibration,
        "heldout": heldout,
        "subset": subset,
        "paths": paths,
        "hashes": {
            "calibration_manifest": lb.sha256_json(calibration),
            "heldout_manifest": lb.sha256_json(heldout),
            "longbench_subset_manifest": lb.sha256_json(subset),
            "tokens_file": lb.sha256_file(token_path),
        },
    }


def analytic_run_directory(
    output_dir: Path, mode: str, condition: AnalyticCondition
) -> Path:
    return output_dir / mode / condition.condition_id


def theoretical_kv_bytes_per_token(assets: dict[str, Any]) -> int:
    layers = int(assets["model"]["num_hidden_layers"])
    heads = int(assets["model"]["num_key_value_heads"])
    head_dim = int(assets["model"]["head_dim"])
    key_bytes = layers * heads * ((head_dim * KEY_BITS) // 8 + 4)
    value_bytes = layers * heads * head_dim * 2
    return key_bytes + value_bytes


def analytic_protocol_payload(
    args: argparse.Namespace,
    assets: dict[str, Any],
    inputs: dict[str, Any],
    rotation_hashes: dict[str, str],
    condition: AnalyticCondition,
    mode: str,
) -> dict[str, Any]:
    return {
        "specification": assets["specification"],
        "mode": mode,
        "condition": {
            "condition_id": condition.condition_id,
            "method": condition.method,
            "label": condition.label,
            "key_bit_width": KEY_BITS,
            "value_bit_width": VALUE_BITS,
            "seed": None,
        },
        "model": assets["model"],
        "implementation_hashes": assets["implementation_hashes"],
        "longbench": assets["longbench"],
        "subset": {
            "manifest": inputs["paths"]["longbench_subset_manifest"],
            "manifest_sha256": inputs["hashes"]["longbench_subset_manifest"],
            "reference_manifest": str(args.reference_subset_manifest),
            "sampling_seed": CALIBRATION_SEED,
            "examples_per_condition": int(inputs["subset"]["example_count"]),
            "samples_per_task": int(inputs["subset"]["samples_per_task"]),
            "samples_per_length_bucket": int(
                inputs["subset"]["samples_per_length_bucket"]
            ),
        },
        "rotation": {
            "artifact": str(
                (args.output_dir / condition.artifact_filename).resolve()
            ),
            "artifact_sha256": rotation_hashes[condition.method],
            "rotation_metadata_sha256": lb.sha256_file(
                args.output_dir / "rotation_metadata.json"
            ),
            "calibration_manifest_sha256": (
                None
                if condition.method == "wk_pca_h"
                else inputs["hashes"]["calibration_manifest"]
            ),
            "saved_convention": "row-vector z = k @ R",
            "codec_adapter": "transpose saved R once for legacy x @ R_codec.T implementation",
        },
        "codebook_sha256": assets["codebook_sha256"],
        "task_order": list(lb.TASKS),
        "categories": {key: list(value) for key, value in lb.CATEGORIES.items()},
        "decoding": {
            "strategy": "greedy_argmax",
            "temperature": 0,
            "num_logits_to_keep": 1,
            "maximum_batch_size": 1 if mode == "smoke" else args.batch_size,
            "task_batch_sizes": {
                task: 1 if mode == "smoke" else min(args.batch_size, size)
                for task, size in lb.TASK_BATCH_SIZES.items()
            },
            "batch_token_budget": args.batch_token_budget,
            "batching": "contiguous within task; fixed before resume filtering",
            "finished_sequence_handling": "compact legacy KV batches after each stop token",
            "maximum_generation_lengths": assets["maximum_generation_lengths"],
            "max_context_length": assets["max_context_length"],
            "middle_truncation": "preserve tokenized front and back halves",
            "chat_policy": "native instruct template with official LongBench exclusions",
            "no_chat_tasks": sorted(lb.NO_CHAT_TASKS),
        },
        "quantization": {
            "implementation": "turboquant_plus analytic head-wise PCA rotation quality emulation",
            "component": "pre-RoPE key only; values remain BF16",
            "prompt_and_generated_tokens": True,
            "norm_correction": False,
            "key_bits": KEY_BITS,
            "value_bits": VALUE_BITS,
            "codebook": "local fixed TurboQuant Lloyd-Max",
            "cache_storage": "reconstructed keys stored in BF16; no packed-memory or optimized-latency claim",
            "common_post_mixing": "fixed normalized Sylvester H_128",
        },
    }


def load_condition_rotation(
    args: argparse.Namespace,
    condition: AnalyticCondition,
    assets: dict[str, Any],
    device: torch.device,
) -> torch.Tensor:
    artifact = torch.load(
        args.output_dir / condition.artifact_filename,
        map_location="cpu",
        weights_only=True,
    )
    row_rotation = artifact["rotations"]
    expected = (
        int(assets["model"]["num_hidden_layers"]),
        int(assets["model"]["num_key_value_heads"]),
        int(assets["model"]["head_dim"]),
        int(assets["model"]["head_dim"]),
    )
    if tuple(row_rotation.shape) != expected:
        raise RuntimeError(f"rotation shape mismatch for {condition.condition_id}")
    return to_codec_rotations(row_rotation).to(device=device, dtype=torch.float32)


def run_inference(
    args: argparse.Namespace,
    mode: str,
    assets: dict[str, Any],
    inputs: dict[str, Any],
) -> None:
    if args.condition is None:
        raise ValueError(f"--condition is required for --stage {mode}")
    condition = condition_by_id(args.condition)
    rotation_hashes = validate_rotation_artifacts(args, assets, deep=False)
    device = torch.device(args.device)
    run_dir = analytic_run_directory(args.output_dir, mode, condition)
    run_dir.mkdir(parents=True, exist_ok=True)
    protocol = analytic_protocol_payload(
        args, assets, inputs, rotation_hashes, condition, mode
    )
    lb.initialize_run_config(
        run_dir, protocol, lb.environment_metadata(device)
    )
    expected = lb.canonical_examples(
        args.data_dir,
        inputs["subset"],
        smoke=mode == "smoke",
    )
    lb.update_run_status(run_dir, expected_predictions=len(expected), status="running")
    prediction_path = run_dir / "predictions.jsonl"
    existing = lb.read_predictions(prediction_path)
    existing_lookup = {
        (str(row["task"]), str(row["example_id"])): row for row in existing
    }
    completed = set(existing_lookup)
    expected_keys = {(task, str(example["_id"])) for task, _, example in expected}
    extra = completed - expected_keys
    if extra:
        raise RuntimeError(f"unexpected existing predictions: {sorted(extra)[:5]}")
    if len(completed) == len(expected):
        predictions = lb.canonicalize_predictions(prediction_path, expected)
        summary = lb.score_condition(run_dir, predictions, args.longbench_repo)
        lb.update_run_status(
            run_dir,
            status="complete",
            completed_predictions=len(predictions),
            completed_at=lb.utc_now(),
            summary=summary,
        )
        print(f"[{lb.utc_now()}] {mode} {condition.condition_id} already complete", flush=True)
        return
    if mode == "full":
        smoke_config = lb.read_json(
            analytic_run_directory(args.output_dir, "smoke", condition)
            / "run_config.json",
            {},
        )
        if smoke_config.get("status") != "complete" or int(
            smoke_config.get("completed_predictions", 0)
        ) != len(lb.TASKS):
            raise RuntimeError(
                f"complete 13-task smoke test required before {condition.condition_id}"
            )

    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    model = lb.load_model(args.model, device)
    rotations = load_condition_rotation(args, condition, assets, device)
    centroids = codebook_tensor(KEY_BITS, 128, device=device)
    codec_context = install_key_codec_hooks(
        model, rotations, centroids, norm_correction=False
    )
    effective_batch_size = 1 if mode == "smoke" else args.batch_size
    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = tokenizer.eos_token_id
    if pad_token_id is None:
        pad_token_id = 0
    kv_bytes = theoretical_kv_bytes_per_token(assets)

    try:
        with codec_context:
            batches = lb.prepared_batches(
                expected,
                tokenizer,
                assets["prompts"],
                assets["maximum_generation_lengths"],
                max_context_length=int(assets["max_context_length"]),
                maximum_batch_size=effective_batch_size,
                batch_token_budget=args.batch_token_budget,
            )
            for batch_ordinal, (batch_id, batch) in enumerate(batches, start=1):
                batch_keys = [
                    (value.task, str(value.example["_id"])) for value in batch
                ]
                if all(key in completed for key in batch_keys):
                    continue
                task = batch[0].task
                max_new_tokens = int(assets["maximum_generation_lengths"][task])
                generated_batch, measured = lb.greedy_generate_batch(
                    model,
                    [value.input_ids for value in batch],
                    max_new_tokens=max_new_tokens,
                    stop_ids=lb.eos_token_ids(model, tokenizer, task),
                    suppress_initial_stop=task == "samsum",
                    pad_token_id=int(pad_token_id),
                    device=device,
                )
                logical_decode_tokens = int(measured["batch_logical_decode_tokens"])
                decode_seconds_per_token = (
                    float(measured["batch_decode_seconds"]) / logical_decode_tokens
                    if logical_decode_tokens
                    else 0.0
                )
                prefill_share = float(measured["batch_prefill_seconds"]) / len(batch)
                overhead_share = max(
                    float(measured["batch_total_seconds"])
                    - float(measured["batch_prefill_seconds"])
                    - float(measured["batch_decode_seconds"]),
                    0.0,
                ) / len(batch)
                new_rows: list[dict[str, Any]] = []
                for prepared, generated_ids, key in zip(
                    batch, generated_batch, batch_keys
                ):
                    example = prepared.example
                    prediction = tokenizer.decode(
                        generated_ids, skip_special_tokens=True
                    )
                    example_decode_tokens = max(len(generated_ids) - 1, 0)
                    decode_share = decode_seconds_per_token * example_decode_tokens
                    row = {
                        "condition_id": condition.condition_id,
                        "method": condition.method,
                        "bit_width": KEY_BITS,
                        "seed": None,
                        "task": task,
                        "task_index": lb.TASKS.index(task),
                        "example_index": prepared.example_index,
                        "example_id": str(example["_id"]),
                        "dataset_length": int(example["length"]),
                        "length_bucket": lb.length_bucket(int(example["length"])),
                        **prepared.prompt_metadata,
                        "max_new_tokens": max_new_tokens,
                        "generated_tokens": len(generated_ids),
                        "prediction": prediction,
                        "answers": example["answers"],
                        "all_classes": example["all_classes"],
                        "batch_id": batch_id,
                        "batch_size": int(measured["batch_size"]),
                        "batch_padded_prompt_tokens": int(
                            measured["batch_padded_prompt_tokens"]
                        ),
                        "batch_prefill_seconds": float(
                            measured["batch_prefill_seconds"]
                        ),
                        "batch_decode_seconds": float(
                            measured["batch_decode_seconds"]
                        ),
                        "batch_decode_steps": int(measured["batch_decode_steps"]),
                        "batch_total_seconds": float(measured["batch_total_seconds"]),
                        "prefill_seconds": prefill_share,
                        "decode_seconds": decode_share,
                        "decode_seconds_per_token": decode_seconds_per_token,
                        "total_seconds": prefill_share + decode_share + overhead_share,
                        "peak_gpu_memory_bytes": int(measured["peak_gpu_memory_bytes"]),
                        "theoretical_kv_bytes_per_token": kv_bytes,
                        "actual_cache_dtype": "bfloat16 reconstructed-key emulation",
                        "created_at": lb.utc_now(),
                    }
                    missing_fields = set(lb.PREDICTION_FIELDS) - row.keys()
                    if missing_fields:
                        raise AssertionError(
                            f"prediction missing fields: {sorted(missing_fields)}"
                        )
                    if key in completed:
                        previous = existing_lookup[key]
                        if (
                            str(previous["prediction"]) != prediction
                            or int(previous["generated_tokens"]) != len(generated_ids)
                        ):
                            raise RuntimeError(
                                f"replayed batch changed {condition.condition_id} {key}"
                            )
                    else:
                        new_rows.append(row)
                lb.append_prediction_batch(prediction_path, new_rows)
                for row in new_rows:
                    key = (str(row["task"]), str(row["example_id"]))
                    completed.add(key)
                    existing_lookup[key] = row
                lb.update_run_status(run_dir, completed_predictions=len(completed))
                print(
                    f"[{lb.utc_now()}] {mode} {condition.condition_id} "
                    f"batch={batch_ordinal} {batch_id} size={len(batch)} "
                    f"completed={len(completed)}/{len(expected)} "
                    f"padded_prompt_tokens={measured['batch_padded_prompt_tokens']} "
                    f"seconds={float(measured['batch_total_seconds']):.3f}",
                    flush=True,
                )
                del generated_batch, new_rows
    except BaseException as error:
        lb.update_run_status(
            run_dir,
            status="failed",
            completed_predictions=len(completed),
            error=f"{type(error).__name__}: {error}",
        )
        raise
    finally:
        del model, rotations, centroids
        cleanup_cuda()

    predictions = lb.canonicalize_predictions(prediction_path, expected)
    summary = lb.score_condition(run_dir, predictions, args.longbench_repo)
    lb.update_run_status(
        run_dir,
        status="complete",
        completed_predictions=len(predictions),
        completed_at=lb.utc_now(),
        summary=summary,
    )
    print(f"[{lb.utc_now()}] completed {mode} {condition.condition_id}", flush=True)


def completed_full_run(
    args: argparse.Namespace,
    condition: AnalyticCondition,
    inputs: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    run_dir = analytic_run_directory(args.output_dir, "full", condition)
    config = lb.read_json(run_dir / "run_config.json", {})
    expected_count = int(inputs["subset"]["example_count"])
    if config.get("status") != "complete" or int(
        config.get("completed_predictions", 0)
    ) != expected_count:
        raise RuntimeError(f"full run is incomplete: {condition.condition_id}")
    predictions = lb.read_predictions(run_dir / "predictions.jsonl")
    scores = lb.read_csv(run_dir / "scores.csv")
    if len(predictions) != expected_count or len(scores) != expected_count:
        raise RuntimeError(
            f"{condition.condition_id} has {len(predictions)} predictions and "
            f"{len(scores)} scores, expected {expected_count}"
        )
    expected = [
        (
            str(row["task"]),
            int(row["dataset_index"]),
            str(row["example_id"]),
            int(row["dataset_length"]),
            str(row["length_bucket"]),
        )
        for row in inputs["subset"]["examples"]
    ]
    prediction_ids = [
        (
            str(row["task"]),
            int(row["example_index"]),
            str(row["example_id"]),
            int(row["dataset_length"]),
            str(row["length_bucket"]),
        )
        for row in predictions
    ]
    score_ids = [
        (
            str(row["task"]),
            int(row["example_index"]),
            str(row["example_id"]),
            int(row["dataset_length"]),
            str(row["length_bucket"]),
        )
        for row in scores
    ]
    if prediction_ids != expected or score_ids != expected:
        raise RuntimeError(f"manifest identity/order mismatch: {condition.condition_id}")
    if any(row["condition_id"] != condition.condition_id for row in predictions):
        raise RuntimeError(f"condition ID mismatch: {condition.condition_id}")
    return predictions, scores


def task_macro(scores: Sequence[dict[str, str]]) -> tuple[dict[str, float], float]:
    means = lb.task_means(scores)
    return means, lb.macro_average(means)


def paired_downstream_analysis(
    scores_by_method: dict[str, list[dict[str, str]]],
    *,
    bootstrap_samples: int,
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    paired_rows: list[dict[str, Any]] = []
    bootstrap_rows: list[dict[str, Any]] = []
    task_rows: list[dict[str, Any]] = []
    category_rows: list[dict[str, Any]] = []
    length_rows: list[dict[str, Any]] = []
    for pair_index, (first_method, second_method) in enumerate(PAIRWISE_COMPARISONS):
        first_condition = METHOD_TO_CONDITION[first_method]
        second_condition = METHOD_TO_CONDITION[second_method]
        pair_id = f"{first_method}_minus_{second_method}"
        first = {
            (row["task"], row["example_id"]): row
            for row in scores_by_method[first_method]
        }
        second = {
            (row["task"], row["example_id"]): row
            for row in scores_by_method[second_method]
        }
        if set(first) != set(second):
            raise RuntimeError(f"paired identities differ for {pair_id}")
        task_arrays: list[np.ndarray] = []
        task_differences: dict[str, float] = {}
        pair_example_differences: list[float] = []
        for task in lb.TASKS:
            task_keys = [
                (row["task"], row["example_id"])
                for row in scores_by_method[first_method]
                if row["task"] == task
            ]
            differences = np.asarray(
                [
                    float(first[key]["score"]) - float(second[key]["score"])
                    for key in task_keys
                ],
                dtype=np.float64,
            )
            if differences.size != 15:
                raise RuntimeError(f"{pair_id}:{task} has {differences.size} pairs")
            task_arrays.append(differences)
            task_differences[task] = float(differences.mean())
            pair_example_differences.extend(differences.tolist())
            task_rows.append(
                {
                    "pair_id": pair_id,
                    "first_condition_id": first_condition.condition_id,
                    "second_condition_id": second_condition.condition_id,
                    "task": task,
                    "category": lb.TASK_TO_CATEGORY[task],
                    "examples": differences.size,
                    "mean_difference": float(differences.mean()),
                    "example_wins": int(np.sum(differences > 0)),
                    "example_ties": int(np.sum(differences == 0)),
                    "example_losses": int(np.sum(differences < 0)),
                }
            )
            for key, difference in zip(task_keys, differences):
                source = first[key]
                paired_rows.append(
                    {
                        "pair_id": pair_id,
                        "first_condition_id": first_condition.condition_id,
                        "second_condition_id": second_condition.condition_id,
                        "task": task,
                        "category": source["category"],
                        "example_index": source["example_index"],
                        "example_id": source["example_id"],
                        "dataset_length": source["dataset_length"],
                        "length_bucket": source["length_bucket"],
                        "first_score": float(first[key]["score"]),
                        "second_score": float(second[key]["score"]),
                        "difference": float(difference),
                    }
                )

        low, high = lb.bootstrap_ci(
            task_arrays,
            samples=bootstrap_samples,
            seed=CALIBRATION_SEED + pair_index,
        )
        macro_difference = statistics.fmean(task_differences.values())
        values = np.asarray(pair_example_differences)
        bootstrap_rows.append(
            {
                "pair_id": pair_id,
                "first_condition_id": first_condition.condition_id,
                "second_condition_id": second_condition.condition_id,
                "first_minus_second_macro_average": macro_difference,
                "confidence_interval_low": low,
                "confidence_interval_high": high,
                "bootstrap_samples": bootstrap_samples,
                "bootstrap_seed": CALIBRATION_SEED + pair_index,
                "positive_tasks": sum(value > 0 for value in task_differences.values()),
                "example_wins": int(np.sum(values > 0)),
                "example_ties": int(np.sum(values == 0)),
                "example_losses": int(np.sum(values < 0)),
            }
        )
        for category, category_tasks in lb.CATEGORIES.items():
            category_rows.append(
                {
                    "pair_id": pair_id,
                    "category": category,
                    "tasks": len(category_tasks),
                    "macro_difference": statistics.fmean(
                        task_differences[task] for task in category_tasks
                    ),
                }
            )
        for bucket in lb.LENGTH_BUCKETS:
            per_task: list[float] = []
            example_count = 0
            for task in lb.TASKS:
                values = [
                    float(row["difference"])
                    for row in paired_rows
                    if row["pair_id"] == pair_id
                    and row["task"] == task
                    and row["length_bucket"] == bucket
                ]
                if values:
                    per_task.append(statistics.fmean(values))
                    example_count += len(values)
            length_rows.append(
                {
                    "pair_id": pair_id,
                    "length_bucket": bucket,
                    "tasks": len(per_task),
                    "examples": example_count,
                    "macro_difference": statistics.fmean(per_task),
                }
            )
    return paired_rows, bootstrap_rows, task_rows, category_rows, length_rows


def aggregate_system_metrics(
    predictions_by_method: dict[str, list[dict[str, Any]]]
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for condition in CONDITIONS:
        predictions = predictions_by_method[condition.method]
        generated = sum(int(row["generated_tokens"]) for row in predictions)
        decode = sum(float(row["decode_seconds"]) for row in predictions)
        rows.append(
            {
                "condition_id": condition.condition_id,
                "method": condition.method,
                "bit_width": KEY_BITS,
                "examples": len(predictions),
                "prompt_tokens": sum(int(row["prompt_tokens"]) for row in predictions),
                "generated_tokens": generated,
                "prefill_seconds": sum(float(row["prefill_seconds"]) for row in predictions),
                "decode_seconds": decode,
                "decode_seconds_per_token": decode / max(generated, 1),
                "total_seconds": sum(float(row["total_seconds"]) for row in predictions),
                "peak_gpu_memory_bytes": max(
                    int(row["peak_gpu_memory_bytes"]) for row in predictions
                ),
                "theoretical_kv_bytes_per_token": int(
                    predictions[0]["theoretical_kv_bytes_per_token"]
                ),
                "empty_outputs": sum(
                    not str(row["prediction"]).strip() for row in predictions
                ),
                "generation_failures": 0,
            }
        )
    return rows


def rankdata(values: Sequence[float]) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    order = np.argsort(array, kind="mergesort")
    ranks = np.empty_like(array)
    start = 0
    while start < len(array):
        stop = start + 1
        while stop < len(array) and array[order[stop]] == array[order[start]]:
            stop += 1
        ranks[order[start:stop]] = (start + stop - 1) / 2.0 + 1.0
        start = stop
    return ranks


def offline_downstream_relationships(
    diagnostic_rows: Sequence[dict[str, str]],
    downstream_scores: dict[str, float],
) -> list[dict[str, Any]]:
    overall = {
        row["method"]: row for row in diagnostic_rows if row["scope"] == "overall"
    }
    metrics = (
        "normalized_key_mse",
        "norm_restored_key_mse",
        "relative_key_mse",
        "maximum_channel_magnitude",
        "max_over_rms",
        "channel_second_moment_max_over_mean",
        "attention_pre_softmax_logit_mse",
        "attention_probability_kl",
        "attention_output_mse",
    )
    methods = [condition.method for condition in CONDITIONS]
    downstream = np.asarray([downstream_scores[method] for method in methods])
    result: list[dict[str, Any]] = []
    for metric in metrics:
        values = np.asarray([float(overall[method][metric]) for method in methods])
        pearson = float(np.corrcoef(values, downstream)[0, 1])
        spearman = float(
            np.corrcoef(rankdata(values), rankdata(downstream))[0, 1]
        )
        result.append(
            {
                "offline_metric": metric,
                "lower_is_better": True,
                "methods": len(methods),
                "pearson_r_raw_metric_vs_longbench": pearson,
                "spearman_r_raw_metric_vs_longbench": spearman,
                "best_offline_method": methods[int(np.argmin(values))],
                "best_downstream_method": methods[int(np.argmax(downstream))],
                "same_best_method": methods[int(np.argmin(values))]
                == methods[int(np.argmax(downstream))],
                "interpretation": "three-condition exploratory association only",
            }
        )
    return result


def reference_baseline_payload(
    args: argparse.Namespace,
    assets: dict[str, Any],
    inputs: dict[str, Any],
) -> dict[str, Any]:
    reference_dir = args.reference_study_dir or args.reference_subset_manifest.parent
    study = lb.read_json(reference_dir / "study_config.json", {})
    fp16_config = lb.read_json(
        reference_dir / "full" / "fp16_K16_V16" / "run_config.json", {}
    )
    checks = {
        "study_complete": study.get("status") == "complete",
        "model_revision": study.get("model", {}).get("revision")
        == assets["model"]["revision"],
        "longbench_commit": study.get("longbench", {}).get("commit")
        == assets["longbench"]["commit"],
        "dataset_revision": study.get("longbench", {}).get("dataset_revision")
        == assets["longbench"]["dataset_revision"],
        "manifest": study.get("subset_manifest", {}).get("sha256")
        == inputs["hashes"]["longbench_subset_manifest"],
        "codebook_k2": study.get("codebook_hashes", {}).get("2")
        == assets["codebook_sha256"],
        "shared_longbench_implementation": study.get("implementation_hashes", {}).get(
            "longbench_runner"
        )
        == assets["implementation_hashes"]["longbench_shared_runner"],
        "prompt_config": study.get("longbench", {}).get("prompt_config_sha256")
        == assets["longbench"]["prompt_config_sha256"],
        "max_generation_config": study.get("longbench", {}).get(
            "max_generation_config_sha256"
        )
        == assets["longbench"]["max_generation_config_sha256"],
        "batch_size": int(study.get("batch_size", -1)) == args.batch_size,
        "batch_token_budget": int(study.get("batch_token_budget", -1))
        == args.batch_token_budget,
        "greedy_decoding": fp16_config.get("protocol", {})
        .get("decoding", {})
        .get("strategy")
        == "greedy_argmax",
        "key_quantization_scope": fp16_config.get("protocol", {})
        .get("quantization", {})
        .get("component")
        == "pre-RoPE key only; values remain BF16",
        "norm_correction": fp16_config.get("protocol", {})
        .get("quantization", {})
        .get("norm_correction")
        is False,
    }
    compatible = all(checks.values())
    payload: dict[str, Any] = {
        "reference_study_dir": str(reference_dir.resolve()),
        "compatible": compatible,
        "checks": checks,
        "note": "reference only; no FP16/Identity/Random/Learned condition was rerun",
    }
    if compatible:
        overall = lb.read_csv(reference_dir / "overall_summary.csv")
        k2 = next(row for row in overall if int(row["bit_width"]) == 2)
        payload["scores"] = {
            "fp16_K16_V16": float(k2["fp16"]),
            "identity_K2_V16": float(k2["identity"]),
            "random_K2_V16_three_seed_mean": float(k2["random_mean"]),
            "learned_K2_V16_three_seed_mean": float(k2["learned_mean"]),
        }
        payload["source_overall_summary_sha256"] = lb.sha256_file(
            reference_dir / "overall_summary.csv"
        )
    return payload


def fmt(value: Any, digits: int = 6) -> str:
    return f"{float(value):.{digits}f}"


def analyze_full_study(
    args: argparse.Namespace,
    assets: dict[str, Any],
    inputs: dict[str, Any],
) -> dict[str, Any]:
    validate_rotation_artifacts(args, assets, deep=True)
    diagnostic_rows = lb.read_csv(args.output_dir / "heldout_diagnostics.csv")
    expected_diagnostics = len(CONDITIONS) * (
        1
        + int(assets["model"]["num_hidden_layers"])
        * int(assets["model"]["num_key_value_heads"])
    )
    if len(diagnostic_rows) != expected_diagnostics:
        raise RuntimeError("held-out diagnostics are incomplete")

    predictions_by_method: dict[str, list[dict[str, Any]]] = {}
    scores_by_method: dict[str, list[dict[str, str]]] = {}
    task_summaries: list[dict[str, str]] = []
    category_summaries: list[dict[str, str]] = []
    length_summaries: list[dict[str, str]] = []
    downstream_scores: dict[str, float] = {}
    for condition in CONDITIONS:
        predictions, scores = completed_full_run(args, condition, inputs)
        predictions_by_method[condition.method] = predictions
        scores_by_method[condition.method] = scores
        task_means, overall = task_macro(scores)
        downstream_scores[condition.method] = overall
        run_dir = analytic_run_directory(args.output_dir, "full", condition)
        task_summaries.extend(lb.read_csv(run_dir / "task_summary.csv"))
        category_summaries.extend(lb.read_csv(run_dir / "category_summary.csv"))
        length_summaries.extend(lb.read_csv(run_dir / "length_summary.csv"))

    paired, bootstrap, task_pairs, category_pairs, length_pairs = (
        paired_downstream_analysis(
            scores_by_method,
            bootstrap_samples=args.bootstrap_samples,
        )
    )
    combined_predictions = [
        row
        for condition in CONDITIONS
        for row in predictions_by_method[condition.method]
    ]
    combined_scores = [
        row for condition in CONDITIONS for row in scores_by_method[condition.method]
    ]
    write_jsonl(args.output_dir / "predictions.jsonl", combined_predictions)
    lb.write_csv(args.output_dir / "scores.csv", combined_scores)
    lb.write_csv(args.output_dir / "task_summary.csv", task_summaries)
    lb.write_csv(args.output_dir / "category_summary.csv", category_summaries)
    lb.write_csv(args.output_dir / "length_summary.csv", length_summaries)
    lb.write_csv(args.output_dir / "paired_comparison.csv", paired)
    lb.write_csv(args.output_dir / "bootstrap_summary.csv", bootstrap)
    lb.write_csv(args.output_dir / "pairwise_task_summary.csv", task_pairs)
    lb.write_csv(args.output_dir / "pairwise_category_summary.csv", category_pairs)
    lb.write_csv(args.output_dir / "pairwise_length_summary.csv", length_pairs)
    system_rows = aggregate_system_metrics(predictions_by_method)
    lb.write_csv(args.output_dir / "system_metrics.csv", system_rows)

    diagnostics_by_method = {
        row["method"]: row
        for row in diagnostic_rows
        if row["scope"] == "overall"
    }
    overall_rows: list[dict[str, Any]] = []
    for condition in CONDITIONS:
        diagnostic = diagnostics_by_method[condition.method]
        overall_rows.append(
            {
                "condition_id": condition.condition_id,
                "method": condition.method,
                "label": condition.label,
                "longbench_macro_average": downstream_scores[condition.method],
                "normalized_key_mse": float(diagnostic["normalized_key_mse"]),
                "norm_restored_key_mse": float(
                    diagnostic["norm_restored_key_mse"]
                ),
                "relative_key_mse": float(diagnostic["relative_key_mse"]),
                "attention_pre_softmax_logit_mse": float(
                    diagnostic["attention_pre_softmax_logit_mse"]
                ),
                "attention_probability_kl": float(
                    diagnostic["attention_probability_kl"]
                ),
                "attention_output_mse": float(diagnostic["attention_output_mse"]),
                "empty_outputs": next(
                    row["empty_outputs"]
                    for row in system_rows
                    if row["method"] == condition.method
                ),
            }
        )
    lb.write_csv(args.output_dir / "overall_summary.csv", overall_rows)
    relationships = offline_downstream_relationships(
        diagnostic_rows, downstream_scores
    )
    lb.write_csv(
        args.output_dir / "offline_downstream_relationship.csv", relationships
    )
    reference = reference_baseline_payload(args, assets, inputs)
    lb.write_json(args.output_dir / "reference_baselines.json", reference)

    bootstrap_by_pair = {row["pair_id"]: row for row in bootstrap}
    activation_vs_wk = bootstrap_by_pair["activation_k_pca_h_minus_wk_pca_h"]
    attention_vs_wk = bootstrap_by_pair["attention_q_pca_h_minus_wk_pca_h"]
    attention_vs_activation = bootstrap_by_pair[
        "attention_q_pca_h_minus_activation_k_pca_h"
    ]
    wk_diag = diagnostics_by_method["wk_pca_h"]
    activation_diag = diagnostics_by_method["activation_k_pca_h"]
    attention_diag = diagnostics_by_method["attention_q_pca_h"]
    h1 = float(activation_diag["normalized_key_mse"]) < float(
        wk_diag["normalized_key_mse"]
    )
    h2 = all(
        float(attention_diag[metric])
        < min(float(wk_diag[metric]), float(activation_diag[metric]))
        for metric in (
            "attention_pre_softmax_logit_mse",
            "attention_probability_kl",
        )
    )
    h3 = downstream_scores["attention_q_pca_h"] == max(downstream_scores.values())
    h4_evidence = (
        float(activation_vs_wk["confidence_interval_low"]) <= 0
        <= float(activation_vs_wk["confidence_interval_high"])
        and float(attention_vs_wk["confidence_interval_low"]) <= 0
        <= float(attention_vs_wk["confidence_interval_high"])
    )
    hypothesis_rows = [
        {
            "hypothesis": "H1",
            "supported": h1,
            "evidence": (
                f"Activation normalized Key MSE {float(activation_diag['normalized_key_mse']):.8g} "
                f"vs Wk {float(wk_diag['normalized_key_mse']):.8g}"
            ),
        },
        {
            "hypothesis": "H2",
            "supported": h2,
            "evidence": (
                "Attention-aware is lowest on both held-out logit MSE and KL"
                if h2
                else "Attention-aware is not lowest on both held-out logit MSE and KL"
            ),
        },
        {
            "hypothesis": "H3",
            "supported": h3,
            "evidence": (
                f"Attention-aware LongBench macro {downstream_scores['attention_q_pca_h']:.6f}; "
                f"best macro {max(downstream_scores.values()):.6f}"
            ),
        },
        {
            "hypothesis": "H4",
            "supported": h4_evidence,
            "evidence": (
                "Both calibration-method vs Wk pilot confidence intervals include zero"
                if h4_evidence
                else "At least one calibration-method vs Wk pilot confidence interval excludes zero"
            ),
        },
    ]
    lb.write_csv(args.output_dir / "hypothesis_summary.csv", hypothesis_rows)

    downstream_table = [
        [
            condition.label,
            fmt(downstream_scores[condition.method], 3),
            fmt(diagnostics_by_method[condition.method]["normalized_key_mse"], 8),
            fmt(
                diagnostics_by_method[condition.method][
                    "attention_pre_softmax_logit_mse"
                ],
                8,
            ),
            fmt(diagnostics_by_method[condition.method]["attention_probability_kl"], 8),
        ]
        for condition in CONDITIONS
    ]
    pair_table = [
        [
            row["pair_id"],
            fmt(row["first_minus_second_macro_average"], 3),
            f"[{fmt(row['confidence_interval_low'], 3)}, {fmt(row['confidence_interval_high'], 3)}]",
            str(row["positive_tasks"]),
            f"{row['example_wins']}/{row['example_ties']}/{row['example_losses']}",
        ]
        for row in bootstrap
    ]
    stability_overall = [
        row
        for row in lb.read_csv(args.output_dir / "calibration_stability.csv")
        if row["scope"] == "overall"
    ]
    stability_table = [
        [
            row["method"],
            fmt(row["covariance_relative_frobenius_difference"], 4),
            fmt(row["principal_subspace_similarity_top16"], 4),
            fmt(row["principal_subspace_similarity_top32"], 4),
            fmt(row["principal_subspace_similarity_top64"], 4),
            fmt(row["heldout_half2_minus_half1_normalized_key_mse"], 8),
        ]
        for row in stability_overall
    ]
    hypothesis_table = [
        [row["hypothesis"], "Supported" if row["supported"] else "Not supported", row["evidence"]]
        for row in hypothesis_rows
    ]
    reference_lines = [
        f"- Compatibility checks passed: `{reference['compatible']}`.",
        "- These are reference lines only; no FP16, Identity, Random, or Learned condition was rerun.",
    ]
    if reference.get("compatible"):
        reference_lines.extend(
            f"- {key}: {float(value):.3f}"
            for key, value in reference["scores"].items()
        )
    report = "\n".join(
        [
            "# Analytic head-wise PCA rotations: LongBench-E pilot",
            "",
            f"Generated: {lb.utc_now()}",
            "",
            "## Overall results",
            "",
            lb.markdown_table(
                ["Method", "LongBench macro", "Normalized Key MSE", "Logit MSE", "Attention KL"],
                downstream_table,
            ),
            "",
            "## Paired downstream comparisons",
            "",
            lb.markdown_table(
                ["First minus second", "Macro difference", "95% paired CI", "Positive tasks", "Win/tie/loss examples"],
                pair_table,
            ),
            "",
            f"Confidence intervals use {args.bootstrap_samples:,} paired, task-stratified resamples of the same 195 identities.",
            "",
            "## Hypotheses",
            "",
            lb.markdown_table(["Hypothesis", "Outcome", "Evidence"], hypothesis_table),
            "",
            "## Calibration stability",
            "",
            lb.markdown_table(
                ["Method", "Covariance relative Frobenius", "Top-16", "Top-32", "Top-64", "Held-out half2-half1 Key MSE"],
                stability_table,
            ),
            "",
            "No numeric instability threshold was specified, so the 4,096-token run is preserved and no unrequested 16,384-token robustness expansion was launched.",
            "",
            "## Matched prior references",
            "",
            *reference_lines,
            "",
            "## Protocol and guardrails",
            "",
            f"- Model revision: `{assets['model']['revision']}`.",
            f"- LongBench commit: `{assets['longbench']['commit']}`; dataset revision: `{assets['longbench']['dataset_revision']}`.",
            f"- Calibration: WikiText-2 train, 8 x 512 tokens, seed {CALIBRATION_SEED}; held-out: WikiText-2 validation, 8 x 512 tokens.",
            "- Every method uses descending, sign-fixed float64 eigendecomposition and the same normalized Sylvester H_128 without random signs or permutations.",
            "- Keys alone are quantized to K2 before RoPE; Values remain BF16 for every condition.",
            "- This is quality emulation: reconstructed Keys are stored in BF16. Theoretical packed bytes are reported, but measured memory/latency is not a compressed-cache claim.",
            "- The 195-example LongBench-E subset is a method-selection pilot, not a final benchmark result.",
            "- Offline distortion and downstream association uses only three methods and is exploratory.",
            "- TinyStories validation was optional and was not run.",
            "",
            "See `heldout_diagnostics.csv`, `rotation_metadata.json`, `paired_comparison.csv`, and `offline_downstream_relationship.csv` for raw evidence.",
            "",
        ]
    )
    (args.output_dir / "report.md").write_text(report)
    summary = {
        "generated_at": lb.utc_now(),
        "all_conditions_complete": True,
        "conditions": len(CONDITIONS),
        "examples_per_condition": int(inputs["subset"]["example_count"]),
        "total_predictions": len(combined_predictions),
        "heldout_tokens": CALIBRATION_TOKENS,
        "overall": overall_rows,
        "bootstrap": bootstrap,
        "hypotheses": hypothesis_rows,
        "reference_baselines_compatible": bool(reference["compatible"]),
    }
    lb.write_json(args.output_dir / "summary.json", summary)
    return summary


def child_command(
    args: argparse.Namespace,
    stage: str,
    condition: AnalyticCondition | None,
) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "experiments.spin_turboquant.longbench_analytic",
        "--stage",
        stage,
        "--model",
        str(args.model),
        "--longbench-repo",
        str(args.longbench_repo),
        "--data-dir",
        str(args.data_dir),
        "--reference-subset-manifest",
        str(args.reference_subset_manifest),
        "--output-dir",
        str(args.output_dir),
        "--device",
        str(args.device),
        "--batch-size",
        str(args.batch_size),
        "--batch-token-budget",
        str(args.batch_token_budget),
        "--bootstrap-samples",
        str(args.bootstrap_samples),
        "--dataset-revision",
        str(args.dataset_revision),
        "--longbench-commit",
        str(args.longbench_commit),
        "--wikitext-revision",
        str(args.wikitext_revision),
    ]
    if args.reference_study_dir is not None:
        command.extend(["--reference-study-dir", str(args.reference_study_dir)])
    if args.max_context_length is not None:
        command.extend(["--max-context-length", str(args.max_context_length)])
    if condition is not None:
        command.extend(["--condition", condition.condition_id])
    return command


def run_child(
    args: argparse.Namespace,
    stage: str,
    condition: AnalyticCondition | None,
) -> None:
    command = child_command(args, stage, condition)
    log_path = args.output_dir / "orchestrator.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as log:
        header = f"[{lb.utc_now()}] $ {' '.join(command)}\n"
        print(header, end="", flush=True)
        log.write(header)
        log.flush()
        process = subprocess.Popen(
            command,
            cwd=Path.cwd(),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            log.write(line)
            log.flush()
        return_code = process.wait()
    if return_code:
        raise subprocess.CalledProcessError(return_code, command)


def orchestrate(
    args: argparse.Namespace,
    assets: dict[str, Any],
    inputs: dict[str, Any],
) -> None:
    stages: dict[str, Any] = {
        "inputs": "complete",
        "rotations": "pending",
        "diagnostics": "pending",
        "smoke": {condition.condition_id: "pending" for condition in CONDITIONS},
        "full": {condition.condition_id: "pending" for condition in CONDITIONS},
        "report": "pending",
    }
    existing = lb.read_json(args.output_dir / "study_config.json", {})
    if isinstance(existing.get("stages"), dict):
        stages.update(existing["stages"])
    write_study_config(
        args,
        assets,
        inputs,
        "running",
        stages=stages,
        started_at=existing.get("started_at", lb.utc_now()),
    )
    try:
        run_child(args, "rotations", None)
        stages["rotations"] = "complete"
        write_study_config(args, assets, inputs, "running", stages=stages)

        run_child(args, "diagnostics", None)
        stages["diagnostics"] = "complete"
        write_study_config(args, assets, inputs, "running", stages=stages)

        for condition in CONDITIONS:
            run_child(args, "smoke", condition)
            stages["smoke"][condition.condition_id] = "complete"
            write_study_config(args, assets, inputs, "running", stages=stages)
        for condition in CONDITIONS:
            run_child(args, "full", condition)
            stages["full"][condition.condition_id] = "complete"
            write_study_config(args, assets, inputs, "running", stages=stages)

        run_child(args, "report", None)
        stages["report"] = "complete"
    except BaseException as error:
        write_study_config(
            args,
            assets,
            inputs,
            "failed",
            stages=stages,
            error=f"{type(error).__name__}: {error}",
        )
        raise
    write_study_config(
        args,
        assets,
        inputs,
        "complete",
        stages=stages,
        completed_at=lb.utc_now(),
        all_conditions_complete=True,
    )


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    args.model = args.model.resolve()
    args.longbench_repo = args.longbench_repo.resolve()
    args.data_dir = args.data_dir.resolve()
    args.reference_subset_manifest = args.reference_subset_manifest.resolve()
    args.output_dir = args.output_dir.resolve()
    if args.reference_study_dir is not None:
        args.reference_study_dir = args.reference_study_dir.resolve()
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.set_float32_matmul_precision("highest")

    assets = validate_static_assets(args)
    if args.stage == "validate":
        inputs = ensure_input_artifacts(args, assets)
        write_study_config(args, assets, inputs, "validated", stages={"inputs": "complete"})
        print(
            json.dumps(
                {
                    "model_revision": assets["model"]["revision"],
                    "longbench_commit": assets["longbench"]["commit"],
                    "longbench_dataset_revision": assets["longbench"][
                        "dataset_revision"
                    ],
                    "wikitext_revision": assets["wikitext"]["revision"],
                    "calibration_tokens": CALIBRATION_TOKENS,
                    "heldout_tokens": CALIBRATION_TOKENS,
                    "examples_per_condition": inputs["subset"]["example_count"],
                    "conditions": [condition.condition_id for condition in CONDITIONS],
                    "expected_predictions": inputs["subset"]["example_count"]
                    * len(CONDITIONS),
                },
                indent=2,
            ),
            flush=True,
        )
        return

    inputs = load_input_artifacts(args, assets)
    if args.stage == "rotations":
        rotation_stage(args, assets, inputs)
    elif args.stage == "diagnostics":
        diagnostics_stage(args, assets, inputs)
    elif args.stage in {"smoke", "full"}:
        run_inference(args, args.stage, assets, inputs)
    elif args.stage == "report":
        summary = analyze_full_study(args, assets, inputs)
        write_study_config(
            args,
            assets,
            inputs,
            "complete",
            all_conditions_complete=True,
            summary=summary,
        )
        print(f"[{lb.utc_now()}] analytic report complete", flush=True)
    elif args.stage == "orchestrate":
        orchestrate(args, assets, inputs)
    else:
        raise AssertionError(args.stage)


if __name__ == "__main__":
    main()
