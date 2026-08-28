"""Run the pinned LongBench-E channel-allocation x rotation study.

This is a dedicated runner for
``longbench_e_rotation_allocation_12_condition_plan.md``.  It deliberately
does not change :mod:`experiments.spin_turboquant.longbench`, whose public
contract belongs to an older K-only study.  The generic LongBench prompt,
generation, resume, and official-scoring machinery is reused through a small
set of experiment-local patches.

One-shot invocation from the ``turboquant_plus`` checkout::

    conda run -n stq python -m experiments.spin_turboquant.longbench_e_rotation_allocation \
      --stage orchestrate \
      --model /path/to/Meta-Llama-3.1-8B-Instruct/snapshot \
      --source-dir experiments/spin_turboquant/results/instruct \
      --longbench-repo ../LongBench_official \
      --data-dir ../LongBench_data/5e628be450b7e67fb7ae6e201bd6d8f7056f7672/data \
      --output-dir experiments/spin_turboquant/results/longbench_e_rotation_allocation_12

The codec is a quality emulation: quantized indices and FP16 group norms are
accounted exactly, while reconstructed BF16 K/V tensors are retained by the
Transformers cache.  Measured GPU memory and latency therefore are not claims
about a packed production kernel.
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
import platform
import statistics
import subprocess
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from experiments.spin_turboquant import longbench as base
from experiments.spin_turboquant import longbench_e_rotation_allocation_protocol as protocol


ROOT_SEED = 35
MODEL_REVISION = "d10aef7999a2b5ba950ab3974312feeedbfe0b77"
LONG_BENCH_COMMIT = "2e00731f8d0bff23dc4325161044d0ed8af94c1e"
LONG_BENCH_DATASET_REVISION = "5e628be450b7e67fb7ae6e201bd6d8f7056f7672"
WIKITEXT_REVISION = "b08601e04326c79dfdd32d625aee71d232d685c3"
LONG_BENCH_DATA_HASHES = {
    "2wikimqa": "525b5b182089a4012cc7429c33f4208358778615173c4a09349429fc80c89641",
    "gov_report": "0a3902fcf3d49f228549f02a2ef1ae84dbe8578b8be3d4611d54459487bdef84",
    "hotpotqa": "26a90a291cca5b2515bf466c6c3d1f57d8a4e67b0cf5aa39e1834913a15e6309",
    "lcc": "d6f21d2a1d6d52a350f134bfafc71e7d13ec6eacbe317556cfa6395808e9a8cc",
    "multi_news": "029d5e2d44d381ba817ad3dcd753d8da7773ad87bbd0f6beb512afedf48a70f5",
    "multifieldqa_en": "678a51335e3c90e0dd43bf1131045e4f2859cc693e7eded89cc6a1ee8d18faff",
    "passage_count": "160525cc4b6bb4e8c584eb012097c5b96574290d0cae52eeafdc8d16d29bb9ee",
    "passage_retrieval_en": "46af1c77465170fb8ffffe3d488bb9c52cc3684118ab0d8a1ed490b39ce97512",
    "qasper": "97d95c01221a17a2ce51f9180d65a671bf2998504a2ff0cccffe97e9b08444d8",
    "repobench-p": "83c6bde23707f190b4ba04cf5fefc9c3779b26be5aed333bc5ac98d5952b233b",
    "samsum": "2ea79e3cfba856aa0dfe588972acbef915f12aef90d949807a2deac4ea65b9fa",
    "trec": "bb1a5afbcd7f6e89411b16663d08f083e14019075e4ac40a6160944d1aecd66e",
    "triviaqa": "3c4cc5ca18f2578d1f6458d1219387c741c6fafad7e8518addea898c08ff0d73",
}
CALIBRATION_TOKENS = 4_096
VALIDATION_TOKENS = 4_096
CALIBRATION_TOKEN_SHA256 = (
    "a6d3e5d14489c852d2eb52cad9544e8dd0756fbcc641acf3b2c2722eca2ab57b"
)
VALIDATION_TOKEN_SHA256 = (
    "9c2ed4e2dac69d9e21370a9e95b6dc2e16da592754c0eab0911fe148d88951f7"
)
DIAGNOSTIC_TOKENS = 256
TRAINING_STEPS = 10_000
TRAINING_BATCH_TOKENS = 256
INITIAL_LEARNING_RATE = 0.005
FINAL_LEARNING_RATE = 0.00025
GRADIENT_CLIP = 1.0
BOOTSTRAP_SAMPLES = 10_000
ORTHOGONALITY_TOLERANCE = 5e-5
NORM_DTYPE = "float16"
NORM_BITS = 16
PACKING = "each group index payload is byte-aligned per layer/head/token vector"
SPECIFICATION_PATH = (
    Path(__file__).resolve().parents[3]
    / "longbench_e_rotation_allocation_12_condition_plan.md"
)
DEFAULT_OUTPUT_DIR = (
    Path("experiments/spin_turboquant/results")
    / "longbench_e_rotation_allocation_12"
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage",
        choices=(
            "validate",
            "capture",
            "artifacts",
            "diagnostics",
            "gates",
            "smoke",
            "full",
            "report",
            "orchestrate",
        ),
        default="orchestrate",
    )
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument(
        "--source-dir",
        type=Path,
        required=True,
        help="model-matched SpinTurboQuant directory supplying fixed calibration tokens",
    )
    parser.add_argument("--longbench-repo", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--condition", choices=[value.condition_id for value in protocol.all_conditions()]
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-context-length", type=int)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--batch-token-budget", type=int, default=32_768)
    parser.add_argument("--bootstrap-samples", type=int, default=BOOTSTRAP_SAMPLES)
    parser.add_argument("--dataset-revision", default=LONG_BENCH_DATASET_REVISION)
    parser.add_argument("--longbench-commit", default=LONG_BENCH_COMMIT)
    parser.add_argument("--wikitext-revision", default=WIKITEXT_REVISION)
    parser.add_argument(
        "--checkpoint-interval",
        type=int,
        default=100,
        help="atomic optimizer-state checkpoint cadence; does not select a model",
    )
    parser.add_argument(
        "--diagnostic-tokens", type=int, default=DIAGNOSTIC_TOKENS
    )
    return parser.parse_args(argv)


def _resolve_args(args: argparse.Namespace) -> argparse.Namespace:
    for name in (
        "model",
        "source_dir",
        "longbench_repo",
        "data_dir",
        "output_dir",
    ):
        value = getattr(args, name, None)
        if value is not None:
            setattr(args, name, Path(value).resolve())
    # ``base.run_inference`` carries this legacy name through its loader API.
    # Our loader treats it as the allocation-artifact directory.
    args.rotation_dir = (args.output_dir / "rotation_artifacts").resolve()
    return args


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def tensor_sha256(value: torch.Tensor) -> str:
    contiguous = value.detach().cpu().contiguous()
    header = json.dumps(
        {"dtype": str(contiguous.dtype), "shape": list(contiguous.shape)},
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return sha256_bytes(header + contiguous.view(torch.uint8).numpy().tobytes())


def atomic_torch_save(value: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def json_safe(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.item() if value.ndim == 0 else value.detach().cpu().tolist()
    if isinstance(value, Mapping):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def write_json(path: Path, value: Any) -> None:
    base.write_json(path, value)


def read_json(path: Path, default: Any = None) -> Any:
    return base.read_json(path, default)


def write_csv(
    path: Path,
    rows: Iterable[dict[str, Any]],
    fieldnames: Sequence[str] | None = None,
) -> None:
    base.write_csv(path, rows, fieldnames=fieldnames)


def named_seed(namespace: str) -> int:
    """Derive a loop-order-independent 63-bit stream seed from root seed 35."""

    digest = hashlib.sha256(f"spin-turboquant:{ROOT_SEED}:{namespace}".encode()).digest()
    return int.from_bytes(digest[:8], "little") & ((1 << 63) - 1)


def rng_manifest(namespaces: Sequence[str]) -> dict[str, Any]:
    core = allocation_core()
    learned_minibatch_seeds = {
        allocation: core.derive_stream_seed(
            ROOT_SEED, allocation, "learned-key-minibatches"
        )
        for allocation in protocol.ALLOCATION_ORDER
    }
    payload = {
        "root_seed": ROOT_SEED,
        "rotation_and_training_derivation": (
            "little-endian first 64 bits of SHA256(JSON compact encoding of "
            "['rotation-allocation-v1', root_seed, *coordinates])"
        ),
        "random_rotation_coordinates": (
            "allocation, component, layer, head, group; every matrix owns an "
            "independent NumPy PCG64 stream"
        ),
        "learned_minibatch_coordinates": (
            "allocation, 'learned-key-minibatches'; one torch.Generator emits "
            "independent indexes for every layer/head"
        ),
        "literal_seed_exceptions": {
            "subset_sampling": ROOT_SEED,
            "sklearn_kmeans_random_state": ROOT_SEED,
            "paired_bootstrap": ROOT_SEED,
        },
        "learned_minibatch_seeds": learned_minibatch_seeds,
        "auxiliary_named_streams": {name: named_seed(name) for name in namespaces},
    }
    payload["stream_table_sha256"] = base.sha256_json(payload)
    return payload


def _environment_metadata(device: torch.device) -> dict[str, Any]:
    result: dict[str, Any] = {
        "timestamp": utc_now(),
        "python": sys.version,
        "python_prefix": sys.prefix,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "cuda_runtime": torch.version.cuda,
        "command": sys.argv,
    }
    for package in ("transformers", "datasets", "sklearn", "scipy"):
        module = __import__(package)
        result[package] = getattr(module, "__version__", "unknown")
    if device.type == "cuda":
        properties = torch.cuda.get_device_properties(device)
        result["gpu"] = {
            "name": properties.name,
            "total_memory_bytes": properties.total_memory,
            "capability": [properties.major, properties.minor],
        }
    try:
        result["git_commit"] = base.git_output(Path.cwd(), "rev-parse", "HEAD")
        result["git_status"] = base.git_output(Path.cwd(), "status", "--short").splitlines()
    except (OSError, subprocess.CalledProcessError):
        pass
    return result


def _tokenizer_hashes(model: Path) -> dict[str, str]:
    names = (
        "config.json",
        "generation_config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
    )
    return {
        name: base.sha256_file(model / name)
        for name in names
        if (model / name).is_file()
    }


def validate_assets(
    args: argparse.Namespace, *, deep: bool = True
) -> dict[str, Any]:
    """Validate pinned immutable inputs; optionally inspect prepared artifacts."""

    if Path(sys.prefix).name != "stq":
        raise RuntimeError(
            f"this experiment must run in conda environment stq; got {sys.prefix}"
        )
    import datasets
    import sklearn
    import transformers

    expected_versions = {
        "torch": "2.5.1+cu121",
        "transformers": "4.45.2",
        "datasets": "3.1.0",
        "numpy": "1.26.4",
        "scikit-learn": "1.5.2",
    }
    actual_versions = {
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "datasets": datasets.__version__,
        "numpy": np.__version__,
        "scikit-learn": sklearn.__version__,
    }
    if actual_versions != expected_versions:
        raise RuntimeError(
            f"stq package versions differ: actual={actual_versions}, "
            f"expected={expected_versions}"
        )
    if args.dataset_revision != LONG_BENCH_DATASET_REVISION:
        raise ValueError("--dataset-revision differs from the pinned plan")
    if args.longbench_commit != LONG_BENCH_COMMIT:
        raise ValueError("--longbench-commit differs from the pinned plan")
    if args.wikitext_revision != WIKITEXT_REVISION:
        raise ValueError("--wikitext-revision differs from the implemented token protocol")
    if args.bootstrap_samples != BOOTSTRAP_SAMPLES:
        raise ValueError(f"the plan requires exactly {BOOTSTRAP_SAMPLES} bootstrap samples")
    if args.checkpoint_interval < 1 or TRAINING_STEPS % args.checkpoint_interval:
        raise ValueError("--checkpoint-interval must be a positive divisor of 10000")
    if not 1 <= args.diagnostic_tokens <= VALIDATION_TOKENS:
        raise ValueError("--diagnostic-tokens must be in [1, 4096]")
    for label, path in {
        "model": args.model,
        "source directory": args.source_dir,
        "LongBench repository": args.longbench_repo,
        "LongBench data directory": args.data_dir,
        "experiment specification": SPECIFICATION_PATH,
    }.items():
        if not path.exists():
            raise FileNotFoundError(f"{label} does not exist: {path}")
    if args.model.name != MODEL_REVISION:
        raise RuntimeError(
            f"model revision is {args.model.name}, expected {MODEL_REVISION}"
        )
    if str(args.device).startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")

    actual_commit = base.git_output(args.longbench_repo, "rev-parse", "HEAD")
    if actual_commit != LONG_BENCH_COMMIT:
        raise RuntimeError(
            f"LongBench checkout is {actual_commit}, expected {LONG_BENCH_COMMIT}"
        )
    tracked_changes = base.git_output(
        args.longbench_repo, "status", "--porcelain", "--untracked-files=no"
    )
    if tracked_changes:
        raise RuntimeError(
            "LongBench checkout has tracked modifications despite the pinned "
            f"commit:\n{tracked_changes}"
        )
    dimensions = base.model_dimensions(args.model)
    expected_dimensions = {
        "num_hidden_layers": 32,
        "num_key_value_heads": 8,
        "head_dim": 128,
    }
    for key, expected in expected_dimensions.items():
        if int(dimensions[key]) != expected:
            raise RuntimeError(
                f"model {key}={dimensions[key]}, expected {expected}"
            )
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    if not tokenizer.chat_template:
        raise RuntimeError("the pinned Instruct tokenizer has no chat template")
    prompts, maximum_generation_lengths = base.load_official_configs(
        args.longbench_repo
    )
    max_context_length = args.max_context_length or dimensions["max_position_embeddings"]
    if max_context_length > dimensions["max_position_embeddings"]:
        raise ValueError("--max-context-length exceeds the model configuration")
    if max_context_length <= max(maximum_generation_lengths.values()):
        raise ValueError("max context length leaves no room for task generation")
    if args.batch_size < 1 or args.batch_token_budget < 1:
        raise ValueError("batch size and token budget must be positive")

    source_config = read_json(args.source_dir / "run_config.json", {})
    source_model = source_config.get("arguments", {}).get("model")
    if source_model is None or Path(source_model).resolve() != args.model:
        raise RuntimeError(
            "--source-dir is not a model-matched SpinTurboQuant run: "
            f"recorded model={source_model!r}"
        )
    source_tokens = args.source_dir / "tokens.pt"
    if not source_tokens.is_file():
        raise FileNotFoundError(
            f"fixed WikiText calibration tokens are missing: {source_tokens}"
        )
    source_calibration = _load_source_calibration_tokens(args)

    data_counts: dict[str, int] = {}
    data_hashes: dict[str, str] = {}
    for task in base.TASKS:
        path = args.data_dir / f"{task}_e.jsonl"
        if not path.is_file():
            raise FileNotFoundError(path)
        data_counts[task] = len(base.load_examples(args.data_dir, task))
        data_hashes[task] = base.sha256_file(path)
    if sum(data_counts.values()) != 3_668:
        raise RuntimeError(f"LongBench-E row count is {sum(data_counts.values())}, expected 3668")
    if data_hashes != LONG_BENCH_DATA_HASHES:
        changed = {
            task: {"actual": data_hashes.get(task), "expected": expected}
            for task, expected in LONG_BENCH_DATA_HASHES.items()
            if data_hashes.get(task) != expected
        }
        raise RuntimeError(
            f"LongBench-E files differ from dataset revision "
            f"{LONG_BENCH_DATASET_REVISION}: {changed}"
        )

    v1_dir = base.official_v1_dir(args.longbench_repo)
    config_dir = v1_dir / "config"
    implementation_files = {
        "runner": Path(__file__).resolve(),
        "protocol": Path(protocol.__file__).resolve(),
        "allocation_core": Path(__file__).with_name("rotation_allocation.py"),
        "base_longbench": Path(base.__file__).resolve(),
        "turboquant_codebook": Path(__file__).resolve().parents[2]
        / "turboquant/codebook.py",
        "turboquant_rotation": Path(__file__).resolve().parents[2]
        / "turboquant/rotation.py",
    }
    missing_implementation = [
        str(path) for path in implementation_files.values() if not path.is_file()
    ]
    if missing_implementation:
        raise FileNotFoundError(
            f"allocation implementation files are missing: {missing_implementation}"
        )
    assets: dict[str, Any] = {
        "model": {
            "path": str(args.model),
            "revision": MODEL_REVISION,
            "dtype": "bfloat16",
            "frozen": True,
            **dimensions,
            "tokenizer_and_config_hashes": _tokenizer_hashes(args.model),
        },
        "source": {
            "path": str(args.source_dir),
            "run_config_sha256": base.sha256_file(args.source_dir / "run_config.json"),
            "tokens_sha256": base.sha256_file(source_tokens),
            "calibration_tensor_sha256": tensor_sha256(source_calibration),
            "calibration_provenance": (
                "first 4096 token IDs from pinned WikiText-2-raw-v1 train using "
                "the pinned Instruct tokenizer; BOS once and EOS between nonempty rows"
            ),
        },
        "longbench": {
            "repository": str(args.longbench_repo),
            "commit": actual_commit,
            "dataset_revision": args.dataset_revision,
            "data_counts": data_counts,
            "data_hashes": data_hashes,
            "prompt_config_sha256": base.sha256_file(
                config_dir / "dataset2prompt.json"
            ),
            "max_generation_config_sha256": base.sha256_file(
                config_dir / "dataset2maxlen.json"
            ),
        },
        "wikitext": {
            "repository": "Salesforce/wikitext",
            "configuration": "wikitext-2-raw-v1",
            "revision": args.wikitext_revision,
            "calibration_split": "train",
            "validation_split": "validation",
            "tokens_per_split": CALIBRATION_TOKENS,
        },
        "specification": {
            "path": str(SPECIFICATION_PATH),
            "sha256": base.sha256_file(SPECIFICATION_PATH),
        },
        "implementation_hashes": {
            name: base.sha256_file(path) for name, path in implementation_files.items()
        },
        "prompts": prompts,
        "maximum_generation_lengths": maximum_generation_lengths,
        "max_context_length": int(max_context_length),
        # Compatibility keys consumed by the generic inference function.
        "rotation_artifact_hashes": {},
        "codebook_hashes": {},
    }
    artifact_manifest_path = args.output_dir / "learned_rotation_manifest.json"
    if artifact_manifest_path.is_file():
        learned_manifest = read_json(artifact_manifest_path, {})
        assets["learned_rotation_manifest_sha256"] = base.sha256_file(
            artifact_manifest_path
        )
        assets["rotation_artifact_hashes"] = learned_manifest.get(
            "artifact_file_hashes", {}
        )
    codebook_manifest_path = args.output_dir / "codebook_manifest.json"
    if codebook_manifest_path.is_file():
        assets["codebook_hashes"] = read_json(codebook_manifest_path, {}).get(
            "codebook_hashes", {}
        )
    if deep and args.stage in {"smoke", "full", "report", "gates"}:
        gate = read_json(args.output_dir / "correctness_gates.json", {})
        if gate.get("static_status") != "passed":
            raise RuntimeError("static correctness gates have not passed")
    return assets


def _load_source_calibration_tokens(args: argparse.Namespace) -> torch.Tensor:
    payload = torch.load(
        args.source_dir / "tokens.pt", map_location="cpu", weights_only=True
    )
    tokens = payload.get("calibration")
    if not isinstance(tokens, torch.Tensor) or tuple(tokens.shape) != (
        CALIBRATION_TOKENS,
    ):
        raise RuntimeError(
            "source calibration tokens must have exact shape (4096,)"
        )
    tokens = tokens.to(dtype=torch.long).contiguous()
    actual_hash = tensor_sha256(tokens)
    if actual_hash != CALIBRATION_TOKEN_SHA256:
        raise RuntimeError(
            "source calibration tokens are not the pinned WikiText-2 train "
            f"stream: {actual_hash} != {CALIBRATION_TOKEN_SHA256}"
        )
    return tokens


def _wikitext_texts(args: argparse.Namespace, split: str) -> Iterator[str]:
    dataset = load_dataset(
        "Salesforce/wikitext",
        "wikitext-2-raw-v1",
        split=split,
        revision=args.wikitext_revision,
    )
    for row in dataset:
        yield str(row["text"])


def _validation_tokens(args: argparse.Namespace, tokenizer: Any) -> tuple[torch.Tensor, str]:
    # Prefer the already-pinned 4096-token held-out stream beside the source
    # run.  The deterministic dataset fallback makes the runner portable.
    candidate = args.source_dir.parent / "training_length_sweep/heldout_tokens.pt"
    candidate_config = (
        candidate.parent / "training_length_sweep_config.json"
    )
    if candidate.is_file() and candidate_config.is_file():
        config = read_json(candidate_config, {})
        recorded_model = config.get("model", {})
        recorded_file_hash = config.get("activations", {}).get(
            "heldout_tokens_file_sha256"
        )
        compatible = (
            recorded_model.get("revision") == MODEL_REVISION
            and Path(str(recorded_model.get("path", ""))).resolve() == args.model
            and recorded_model.get("tokenizer_and_config_hashes")
            == _tokenizer_hashes(args.model)
            and recorded_file_hash == base.sha256_file(candidate)
        )
    else:
        compatible = False
    if compatible:
        payload = torch.load(candidate, map_location="cpu", weights_only=True)
        value = payload.get("wikitext")
        if isinstance(value, torch.Tensor) and tuple(value.shape) == (VALIDATION_TOKENS,):
            value = value.to(dtype=torch.long).contiguous()
            if tensor_sha256(value) == VALIDATION_TOKEN_SHA256:
                return value, str(candidate)
    value = base_run_token_stream(
        tokenizer, _wikitext_texts(args, "validation"), VALIDATION_TOKENS
    )
    if tensor_sha256(value) != VALIDATION_TOKEN_SHA256:
        raise RuntimeError(
            "pinned WikiText-2 validation token stream differs from the expected hash"
        )
    return value, "Salesforce/wikitext:wikitext-2-raw-v1:validation"


def base_run_token_stream(
    tokenizer: Any, texts: Iterable[str], count: int
) -> torch.Tensor:
    # Local copy avoids importing the large end-to-end runner solely for a
    # small deterministic token concatenation primitive.
    token_ids: list[int] = []
    if tokenizer.bos_token_id is not None:
        token_ids.append(int(tokenizer.bos_token_id))
    for text in texts:
        if not text.strip():
            continue
        token_ids.extend(tokenizer.encode(text, add_special_tokens=False))
        if tokenizer.eos_token_id is not None:
            token_ids.append(int(tokenizer.eos_token_id))
        if len(token_ids) >= count:
            return torch.tensor(token_ids[:count], dtype=torch.long)
    raise RuntimeError(f"WikiText yielded only {len(token_ids)} of {count} tokens")


def prepare_tokens(args: argparse.Namespace) -> dict[str, Any]:
    path = args.output_dir / "tokens.pt"
    manifest_path = args.output_dir / "token_manifest.json"
    calibration = _load_source_calibration_tokens(args)
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    validation, validation_source = _validation_tokens(args, tokenizer)
    expected_manifest = {
        "schema_version": 1,
        "created_from": {
            "calibration": str(args.source_dir / "tokens.pt"),
            "validation": validation_source,
        },
        "dataset": {
            "repository": "Salesforce/wikitext",
            "configuration": "wikitext-2-raw-v1",
            "revision": args.wikitext_revision,
            "joining": "BOS once; nonempty raw rows; EOS between rows; first 4096 IDs",
        },
        "tokenizer_revision": MODEL_REVISION,
        "calibration_tokens": CALIBRATION_TOKENS,
        "validation_tokens": VALIDATION_TOKENS,
        "calibration_tensor_sha256": tensor_sha256(calibration),
        "validation_tensor_sha256": tensor_sha256(validation),
        "disjoint_tensor_streams": not torch.equal(calibration, validation),
    }
    if not expected_manifest["disjoint_tensor_streams"]:
        raise RuntimeError("calibration and validation token streams are identical")
    existing = read_json(manifest_path)
    if existing is not None and existing != expected_manifest:
        raise RuntimeError(f"existing token manifest differs: {manifest_path}")
    if path.is_file():
        stored = torch.load(path, map_location="cpu", weights_only=True)
        if not torch.equal(stored.get("calibration"), calibration) or not torch.equal(
            stored.get("validation"), validation
        ):
            raise RuntimeError(f"existing token tensors differ: {path}")
    else:
        atomic_torch_save(
            {"calibration": calibration, "validation": validation}, path
        )
    write_json(manifest_path, expected_manifest)
    return expected_manifest


def load_frozen_model(model_path: Path, device: torch.device) -> Any:
    print(f"[{utc_now()}] loading frozen BF16 model {model_path}", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        local_files_only=True,
        torch_dtype=torch.bfloat16 if device.type == "cuda" else torch.float32,
        low_cpu_mem_usage=True,
        attn_implementation="sdpa",
    ).to(device)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


@torch.inference_mode()
def capture_kv_activations(
    model: Any,
    input_ids: torch.Tensor,
    *,
    device: torch.device,
    diagnostic_tokens: int = 0,
) -> dict[str, torch.Tensor]:
    """Capture pre-RoPE K and post-projection V; optionally bounded Q/RoPE."""

    layers = model.model.layers
    layer_count = len(layers)
    kv_heads = int(model.config.num_key_value_heads)
    query_heads = int(model.config.num_attention_heads)
    captured: dict[str, list[torch.Tensor | None]] = {
        "k": [None] * layer_count,
        "v": [None] * layer_count,
    }
    if diagnostic_tokens:
        captured["q"] = [None] * layer_count
    handles: list[Any] = []

    def register(
        projection: torch.nn.Module,
        kind: str,
        layer_index: int,
        heads: int,
    ) -> None:
        def hook(
            _module: torch.nn.Module,
            _inputs: tuple[torch.Tensor, ...],
            output: torch.Tensor,
            *,
            selected_kind: str = kind,
            selected_layer: int = layer_index,
            selected_heads: int = heads,
        ) -> None:
            shaped = output.detach().reshape(
                output.shape[0], output.shape[1], selected_heads, -1
            )[0]
            if selected_kind == "q":
                shaped = shaped[:diagnostic_tokens]
                captured[selected_kind][selected_layer] = shaped.to(
                    device="cpu", dtype=torch.bfloat16
                )
            else:
                captured[selected_kind][selected_layer] = shaped.permute(
                    1, 0, 2
                ).contiguous().to(device="cpu", dtype=torch.bfloat16)

        handles.append(projection.register_forward_hook(hook))

    for layer_index, layer in enumerate(layers):
        register(layer.self_attn.k_proj, "k", layer_index, kv_heads)
        register(layer.self_attn.v_proj, "v", layer_index, kv_heads)
        if diagnostic_tokens:
            register(layer.self_attn.q_proj, "q", layer_index, query_heads)

    rope: dict[str, torch.Tensor] = {}
    if diagnostic_tokens:
        def rope_hook(
            _module: torch.nn.Module,
            _inputs: tuple[torch.Tensor, ...],
            output: tuple[torch.Tensor, torch.Tensor],
        ) -> None:
            rope["cos"] = output[0].detach()[0, :diagnostic_tokens].float().cpu()
            rope["sin"] = output[1].detach()[0, :diagnostic_tokens].float().cpu()

        handles.append(model.model.rotary_emb.register_forward_hook(rope_hook))

    try:
        model.model(input_ids=input_ids.unsqueeze(0).to(device), use_cache=False)
    finally:
        for handle in handles:
            handle.remove()

    result: dict[str, torch.Tensor] = {"input_ids": input_ids.cpu()}
    for kind, values in captured.items():
        if any(value is None for value in values):
            raise RuntimeError(f"failed to capture all {kind.upper()} projections")
        result[kind] = torch.stack([value for value in values if value is not None])
    if diagnostic_tokens:
        if set(rope) != {"cos", "sin"}:
            raise RuntimeError("failed to capture rotary cos/sin")
        result.update(rope)
    return result


def capture_stage(args: argparse.Namespace) -> None:
    validate_assets(args, deep=False)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    token_manifest = prepare_tokens(args)
    tokens = torch.load(
        args.output_dir / "tokens.pt", map_location="cpu", weights_only=True
    )
    activation_dir = args.output_dir / "activations"
    activation_dir.mkdir(parents=True, exist_ok=True)
    expected = {
        "calibration": activation_dir / "calibration.pt",
        "validation": activation_dir / "validation.pt",
    }
    token_manifest_file_hash = base.sha256_file(
        args.output_dir / "token_manifest.json"
    )
    capture_protocols = {
        name: {
            "schema_version": 1,
            "model_revision": MODEL_REVISION,
            "split": name,
            "input_ids_sha256": tensor_sha256(tokens[name]),
            "token_manifest_file_sha256": token_manifest_file_hash,
            "capture_implementation_sha256": base.sha256_file(Path(__file__)),
            "key_location": "k_proj output before RoPE",
            "value_location": "v_proj output before cache write",
            "diagnostic_tokens": args.diagnostic_tokens if name == "validation" else 0,
        }
        for name in expected
    }
    capture_protocol_hashes = {
        name: base.sha256_json(value) for name, value in capture_protocols.items()
    }
    device = torch.device(args.device)
    if not all(path.is_file() for path in expected.values()):
        model = load_frozen_model(args.model, device)
        try:
            if not expected["calibration"].is_file():
                print(f"[{utc_now()}] capturing 4096-token train K/V", flush=True)
                captured = capture_kv_activations(
                    model, tokens["calibration"], device=device
                )
                captured["capture_protocol_sha256"] = capture_protocol_hashes[
                    "calibration"
                ]
                atomic_torch_save(
                    captured,
                    expected["calibration"],
                )
            if not expected["validation"].is_file():
                print(
                    f"[{utc_now()}] capturing 4096-token validation K/V and "
                    f"{args.diagnostic_tokens}-token Q/RoPE",
                    flush=True,
                )
                captured = capture_kv_activations(
                    model,
                    tokens["validation"],
                    device=device,
                    diagnostic_tokens=args.diagnostic_tokens,
                )
                captured["capture_protocol_sha256"] = capture_protocol_hashes[
                    "validation"
                ]
                atomic_torch_save(
                    captured,
                    expected["validation"],
                )
        finally:
            del model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    required_shapes = {
        "calibration": {
            "k": (32, 8, CALIBRATION_TOKENS, 128),
            "v": (32, 8, CALIBRATION_TOKENS, 128),
        },
        "validation": {
            "k": (32, 8, VALIDATION_TOKENS, 128),
            "v": (32, 8, VALIDATION_TOKENS, 128),
            "q": (32, args.diagnostic_tokens, 32, 128),
            "cos": (args.diagnostic_tokens, 128),
            "sin": (args.diagnostic_tokens, 128),
        },
    }
    records: dict[str, Any] = {}
    for name, path in expected.items():
        payload = torch.load(path, map_location="cpu", weights_only=True)
        if payload.get("capture_protocol_sha256") != capture_protocol_hashes[name]:
            raise RuntimeError(
                f"{path} belongs to a different capture protocol; use a new "
                "output directory or remove the stale activation artifact"
            )
        stored_input_ids = payload.get("input_ids")
        if not isinstance(stored_input_ids, torch.Tensor) or not torch.equal(
            stored_input_ids, tokens[name]
        ):
            raise RuntimeError(f"{path}:input_ids differ from the fixed token stream")
        for key, shape in required_shapes[name].items():
            value = payload.get(key)
            if not isinstance(value, torch.Tensor) or tuple(value.shape) != shape:
                raise RuntimeError(
                    f"{path}:{key} shape is {getattr(value, 'shape', None)}, expected {shape}"
                )
            if not torch.isfinite(value.float()).all():
                raise RuntimeError(f"{path}:{key} contains NaN/Inf")
        records[name] = {
            "path": str(path),
            "file_sha256": base.sha256_file(path),
            "capture_protocol": capture_protocols[name],
            "capture_protocol_sha256": capture_protocol_hashes[name],
            "tensor_hashes": {
                key: tensor_sha256(value)
                for key, value in payload.items()
                if isinstance(value, torch.Tensor)
            },
            "shapes": {
                key: list(value.shape)
                for key, value in payload.items()
                if isinstance(value, torch.Tensor)
            },
        }
    write_json(
        args.output_dir / "activation_manifest.json",
        {
            "schema_version": 1,
            "model_revision": MODEL_REVISION,
            "token_manifest_sha256": base.sha256_file(
                args.output_dir / "token_manifest.json"
            ),
            "accumulation_contract": "channel statistics use float64 mean(abs(BF16 activation))",
            "diagnostic_protocol": (
                f"first {args.diagnostic_tokens} validation tokens; causal attention; "
                "bounded because full 4096x4096 diagnostics are quadratic"
            ),
            "activations": records,
            "token_manifest": token_manifest,
        },
    )
    print(f"[{utc_now()}] activation capture complete", flush=True)


# The artifact/training/codec implementation lives in a small dedicated core.
# Imports are intentionally delayed so ``--help`` and manifest-only tests do
# not initialize CUDA or require loading large tensor artifacts.
def allocation_core() -> Any:
    from experiments.spin_turboquant import rotation_allocation

    return rotation_allocation


def condition_artifact_path(output_dir: Path, allocation: str) -> Path:
    return output_dir / "rotation_artifacts" / f"{allocation}.pt"


def _channel_statistics(calibration: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    result: dict[str, torch.Tensor] = {}
    for component in ("k", "v"):
        values = calibration[component]
        statistics_tensor = torch.empty(
            values.shape[0], values.shape[1], values.shape[3], dtype=torch.float64
        )
        for layer in range(values.shape[0]):
            statistics_tensor[layer] = values[layer].double().abs().mean(dim=1)
        result[component] = statistics_tensor
    return result


def _statistics_json(statistics_by_component: Mapping[str, torch.Tensor]) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "definition": "mean absolute pre-RoPE K / post-projection V over 4096 WikiText-2 train tokens",
        "accumulation_dtype": "float64",
        "components": {
            component: {
                "shape": list(value.shape),
                "tensor_sha256": tensor_sha256(value),
                "values": value.tolist(),
            }
            for component, value in statistics_by_component.items()
        },
    }


def _audit_condition_artifact(
    artifact: Mapping[str, Any], *, orthogonality_tolerance: float
) -> dict[str, Any]:
    """Audit the composite partition + Identity/Random/Learned artifact."""

    core = allocation_core()
    failures: list[str] = []
    allocation = str(artifact.get("allocation"))
    partition = artifact.get("partition")
    identity = artifact.get("identity_rotation")
    random_rotation = artifact.get("random_rotation")
    learned_rotation = artifact.get("learned_rotation")
    try:
        partition_diagnostics = core.validate_partition_artifact(
            partition, allocation
        )
    except Exception as error:
        partition_diagnostics = {}
        failures.append(f"partition validation: {type(error).__name__}: {error}")
    rotation_diagnostics: dict[str, Any] = {}
    for name, value, expected_kind in (
        ("identity", identity, "identity"),
        ("random", random_rotation, "random"),
        ("learned", learned_rotation, "learned"),
    ):
        try:
            diagnostics = core.validate_rotation_artifact(
                value, allocation, partition
            )
            rotation_diagnostics[name] = diagnostics
            if value.get("kind") != expected_kind:
                failures.append(
                    f"{name} kind={value.get('kind')!r}, expected {expected_kind!r}"
                )
            if float(diagnostics["orthogonality_max_abs"]) > orthogonality_tolerance:
                failures.append(
                    f"{name} orthogonality {diagnostics['orthogonality_max_abs']} "
                    f"> {orthogonality_tolerance}"
                )
        except Exception as error:
            failures.append(f"{name} validation: {type(error).__name__}: {error}")
    random_key_hash = ""
    random_value_hash = ""
    learned_key_hash = ""
    learned_value_hash = ""
    if random_rotation and learned_rotation:
        random_key_hash = core.component_rotation_sha256(random_rotation, "key")
        random_value_hash = core.component_rotation_sha256(random_rotation, "value")
        learned_key_hash = core.component_rotation_sha256(learned_rotation, "key")
        learned_value_hash = core.component_rotation_sha256(learned_rotation, "value")
        recorded_step0 = learned_rotation.get("training", {}).get(
            "step0_key_sha256"
        )
        if recorded_step0 != random_key_hash:
            failures.append("Learned step-0 Key is not bitwise-identical to Random Key")
        if learned_value_hash != random_value_hash:
            failures.append("Learned Value rotation differs from Random Value")
    if int(artifact.get("step", -1)) != TRAINING_STEPS:
        failures.append("composite artifact is not fixed step 10000")
    return {
        "status": "passed" if not failures else "failed",
        "allocation": allocation,
        "partition_hash": core.artifact_sha256(partition) if partition else "",
        "partition_diagnostics": json_safe(partition_diagnostics),
        "rotation_diagnostics": rotation_diagnostics,
        "random_key_hash": random_key_hash,
        "random_value_hash": random_value_hash,
        "learned_key_hash": learned_key_hash,
        "learned_value_hash": learned_value_hash,
        "step0_key_matches_random_bitwise": (
            learned_rotation.get("training", {}).get("step0_key_sha256")
            == random_key_hash
            if learned_rotation
            else False
        ),
        "learned_value_matches_random_bitwise": learned_value_hash
        == random_value_hash,
        "orthogonality_tolerance": orthogonality_tolerance,
        "failures": failures,
    }


def artifacts_stage(args: argparse.Namespace) -> None:
    """Build statistics, partitions, Random tensors, and four learned artifacts."""

    validate_assets(args, deep=False)
    activation_path = args.output_dir / "activations/calibration.pt"
    validation_path = args.output_dir / "activations/validation.pt"
    if not activation_path.is_file() or not validation_path.is_file():
        raise FileNotFoundError("capture stage must complete before artifacts")
    core = allocation_core()
    calibration = torch.load(activation_path, map_location="cpu", weights_only=True)
    validation = torch.load(validation_path, map_location="cpu", weights_only=True)
    statistics_by_component = _channel_statistics(calibration)
    statistics_payload = _statistics_json(statistics_by_component)
    statistics_path = args.output_dir / "channel_statistics.json"
    existing = read_json(statistics_path)
    if existing is not None and existing != statistics_payload:
        raise RuntimeError(f"existing channel statistics differ: {statistics_path}")
    write_json(statistics_path, statistics_payload)

    partitions = {
        allocation: core.build_partition_artifact(
            allocation,
            statistics_by_component["k"].float(),
            statistics_by_component["v"].float(),
        )
        for allocation in protocol.ALLOCATION_ORDER
    }
    fixed_path = args.output_dir / "fixed32_partitions.pt"
    kmeans_path = args.output_dir / "kmeans2_partitions.pt"
    for allocation, path in (("fixed32", fixed_path), ("kmeans2", kmeans_path)):
        if path.is_file():
            existing_partition = torch.load(
                path, map_location="cpu", weights_only=True
            )
            if core.artifact_sha256(existing_partition) != core.artifact_sha256(
                partitions[allocation]
            ):
                raise RuntimeError(f"existing partition differs: {path}")
        else:
            atomic_torch_save(partitions[allocation], path)
        reloaded = torch.load(path, map_location="cpu", weights_only=True)
        if core.artifact_sha256(reloaded) != core.artifact_sha256(partitions[allocation]):
            raise RuntimeError(f"partition reload is not bitwise reproducible: {path}")

    partition_allocations: dict[str, Any] = {}
    for allocation, partition in partitions.items():
        diagnostics = core.validate_partition_artifact(partition, allocation)
        component_rows: dict[str, Any] = {}
        for component in ("key", "value"):
            counts = diagnostics[component]["outlier_counts"].flatten().tolist()
            component_rows[component] = {
                "outlier_min": min(counts),
                "outlier_mean": statistics.fmean(counts),
                "outlier_max": max(counts),
                "outlier_count_histogram": {
                    str(key): value for key, value in sorted(Counter(counts).items())
                },
                "outlier_indexes": [
                    [
                        torch.nonzero(mask, as_tuple=False).flatten().tolist()
                        for mask in layer
                    ]
                    for layer in partition["outlier_masks"][component]
                ],
            }
        partition_allocations[allocation] = {
            "sha256": core.artifact_sha256(partition),
            "parameters": partition["partition_parameters"],
            "components": component_rows,
        }
    partition_manifest = {
        "schema_version": 1,
        "channel_statistics_sha256": base.sha256_file(statistics_path),
        "fixed32_path": str(fixed_path),
        "fixed32_file_sha256": base.sha256_file(fixed_path),
        "kmeans2_path": str(kmeans_path),
        "kmeans2_file_sha256": base.sha256_file(kmeans_path),
        "kmeans_implementation": {
            "class": "sklearn.cluster.KMeans",
            "version": __import__("sklearn").__version__,
            "n_clusters": 2,
            "init": "k-means++",
            "n_init": 10,
            "random_state": ROOT_SEED,
            "size_constraint": None,
        },
        "fixed_tie_break": "stable descending magnitude; lower channel index wins",
        "allocations": partition_allocations,
    }
    write_json(args.output_dir / "partition_manifest.json", partition_manifest)

    args.rotation_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    training_rows = base.read_csv(args.output_dir / "training_curves.csv")
    random_manifest: dict[str, Any] = {
        "schema_version": 1,
        "root_seed": ROOT_SEED,
        "stream_derivation": rng_manifest(["rotation", "minibatch"]),
        "allocations": {},
    }
    learned_manifest: dict[str, Any] = {
        "schema_version": 1,
        "training": {
            "steps": TRAINING_STEPS,
            "optimizer": "Adam",
            "scheduler": "CosineAnnealingLR",
            "initial_lr": INITIAL_LEARNING_RATE,
            "eta_min": FINAL_LEARNING_RATE,
            "T_max": TRAINING_STEPS,
            "tokens_per_head_per_step": TRAINING_BATCH_TOKENS,
            "minibatch_sampling": "independent indexes for every layer/KV-head",
            "gradient_clip": GRADIENT_CLIP,
            "parameterization": "dense Cayley",
            "objective": "original-scale reconstructed full-128 Key MSE",
            "quantizer_gradient": "nearest centroid chosen from detached rotated coordinates",
            "checkpoint_selection": "none; fixed step 10000",
        },
        "allocations": {},
        "artifact_file_hashes": {},
    }

    device = torch.device(args.device)
    artifact_implementation_hashes = {
        "allocation_core": base.sha256_file(Path(core.__file__).resolve()),
        "cayley_core": base.sha256_file(Path(__file__).with_name("core.py")),
        "turboquant_codebook": base.sha256_file(
            Path(__file__).resolve().parents[2] / "turboquant/codebook.py"
        ),
        "turboquant_rotation": base.sha256_file(
            Path(__file__).resolve().parents[2] / "turboquant/rotation.py"
        ),
    }
    for allocation in protocol.ALLOCATION_ORDER:
        artifact_path = condition_artifact_path(args.output_dir, allocation)
        selected_partition = partitions[allocation]
        allocation_codebook_hash = codebook_sha256(
            allocation, selected_partition
        )
        expected_protocol = {
            "allocation": allocation,
            "model_revision": MODEL_REVISION,
            "calibration_sha256": base.sha256_file(activation_path),
            "validation_sha256": base.sha256_file(validation_path),
            "partition_hash": core.artifact_sha256(selected_partition),
            "bit_allocation_hash": bit_allocation_sha256(
                allocation, selected_partition
            ),
            "codebook_hash": allocation_codebook_hash,
            "implementation_hashes": artifact_implementation_hashes,
            "root_seed": ROOT_SEED,
            "steps": TRAINING_STEPS,
            "batch_tokens": TRAINING_BATCH_TOKENS,
            "initial_lr": INITIAL_LEARNING_RATE,
            "eta_min": FINAL_LEARNING_RATE,
            "gradient_clip": GRADIENT_CLIP,
            "norm_dtype": NORM_DTYPE,
            "packing": PACKING,
        }
        checkpoint_path = (
            args.output_dir / "checkpoints" / f"{allocation}_latest.pt"
        )
        fixed_checkpoint_path = (
            args.output_dir / "checkpoints" / f"{allocation}_step10000.pt"
        )
        artifact: dict[str, Any] | None = None
        if artifact_path.is_file():
            candidate = torch.load(
                artifact_path, map_location="cpu", weights_only=True
            )
            if candidate.get("protocol") != expected_protocol:
                raise RuntimeError(
                    f"existing artifact uses a different protocol: {artifact_path}"
                )
            if int(candidate.get("step", -1)) == TRAINING_STEPS:
                if not checkpoint_path.is_file() or not fixed_checkpoint_path.is_file():
                    raise RuntimeError(
                        f"complete artifact is missing its durable checkpoints: {allocation}"
                    )
                latest_checkpoint = torch.load(
                    checkpoint_path, map_location="cpu", weights_only=True
                )
                fixed_checkpoint = torch.load(
                    fixed_checkpoint_path, map_location="cpu", weights_only=True
                )
                if (
                    latest_checkpoint.get("protocol") != expected_protocol
                    or int(latest_checkpoint.get("step", -1)) != TRAINING_STEPS
                ):
                    raise RuntimeError(
                        f"latest checkpoint differs from the fixed protocol: {checkpoint_path}"
                    )
                if (
                    fixed_checkpoint.get("protocol") != expected_protocol
                    or int(fixed_checkpoint.get("step", -1)) != TRAINING_STEPS
                    or core.artifact_sha256(fixed_checkpoint)
                    != core.artifact_sha256(candidate)
                ):
                    raise RuntimeError(
                        f"step-10000 checkpoint differs from the composite artifact: "
                        f"{fixed_checkpoint_path}"
                    )
                artifact = candidate
                print(f"[{utc_now()}] reuse {artifact_path.name}", flush=True)
        if artifact is None:
            identity_rotation = core.build_identity_rotation_artifact(
                allocation, selected_partition
            )
            random_rotation = core.build_random_rotation_artifact(
                allocation, selected_partition, root_seed=ROOT_SEED
            )
            def checkpoint_callback(state: dict[str, Any]) -> None:
                state = {**state, "protocol": expected_protocol}
                atomic_torch_save(state, checkpoint_path)
                nonlocal training_rows
                training_rows = [
                    current
                    for current in training_rows
                    if current.get("allocation") != allocation
                ]
                training_rows.extend(
                    {"allocation": allocation, **row}
                    for row in state.get("curve_rows", [])
                )
                training_rows.sort(
                    key=lambda current: (
                        protocol.ALLOCATION_ORDER.index(current["allocation"]),
                        int(current["step"]),
                    )
                )
                write_csv(args.output_dir / "training_curves.csv", training_rows)

            resume = None
            if checkpoint_path.is_file():
                resume = torch.load(
                    checkpoint_path, map_location="cpu", weights_only=True
                )
                if resume.get("protocol") != expected_protocol:
                    raise RuntimeError(
                        f"training checkpoint protocol differs: {checkpoint_path}"
                    )
            training_result = core.train_learned_key_rotations(
                calibration["k"].to(device=device, dtype=torch.float32),
                allocation,
                selected_partition,
                random_rotation,
                validation_keys=validation["k"].to(
                    device=device, dtype=torch.float32
                ),
                steps=TRAINING_STEPS,
                batch_tokens=TRAINING_BATCH_TOKENS,
                learning_rate=INITIAL_LEARNING_RATE,
                minimum_learning_rate=FINAL_LEARNING_RATE,
                gradient_clip=GRADIENT_CLIP,
                root_seed=ROOT_SEED,
                curve_interval=args.checkpoint_interval,
                checkpoint_interval=args.checkpoint_interval,
                checkpoint_callback=checkpoint_callback,
                resume_state=resume,
            )
            if not training_result["completed"]:
                raise RuntimeError(f"training did not reach step 10000: {allocation}")
            # Ensure the final curve is durable even if no callback implementation
            # detail changes in the core.
            checkpoint_callback(training_result["checkpoint_state"])
            artifact = {
                "schema_version": 1,
                "protocol": expected_protocol,
                "allocation": allocation,
                "step": TRAINING_STEPS,
                "partition": selected_partition,
                "identity_rotation": identity_rotation,
                "random_rotation": random_rotation,
                "learned_rotation": training_result["artifact"],
                "training": {
                    key: value
                    for key, value in training_result.items()
                    if key not in {"artifact", "checkpoint_state", "curve_rows"}
                },
                "training_curve": training_result["curve_rows"],
            }
            atomic_torch_save(
                artifact,
                fixed_checkpoint_path,
            )
            # Publish the composite artifact only after its fixed-step copy is
            # durable, so a crash can never expose a "complete" artifact with
            # a missing step-10000 checkpoint.
            atomic_torch_save(artifact, artifact_path)

        if artifact.get("training_curve"):
            training_rows = [
                row for row in training_rows if row.get("allocation") != allocation
            ]
            training_rows.extend(
                {"allocation": allocation, **row}
                for row in artifact["training_curve"]
            )
            training_rows.sort(
                key=lambda row: (
                    protocol.ALLOCATION_ORDER.index(row["allocation"]),
                    int(row["step"]),
                )
            )
            write_csv(args.output_dir / "training_curves.csv", training_rows)
        gate = _audit_condition_artifact(
            artifact,
            orthogonality_tolerance=ORTHOGONALITY_TOLERANCE,
        )
        if gate["status"] != "passed":
            raise RuntimeError(
                f"rotation artifact gate failed for {allocation}: {gate['failures']}"
            )
        random_manifest["allocations"][allocation] = {
            "key_hash": gate["random_key_hash"],
            "value_hash": gate["random_value_hash"],
            "partition_hash": expected_protocol["partition_hash"],
            "stream_seeds": artifact["random_rotation"].get("stream_seeds", {}),
        }
        learned_manifest["allocations"][allocation] = {
            **gate,
            "path": str(artifact_path),
            "step": TRAINING_STEPS,
            "minibatch_stream_seed": core.derive_stream_seed(
                ROOT_SEED, allocation, "learned-key-minibatches"
            ),
            "training": artifact.get("training", {}),
        }
        learned_manifest["artifact_file_hashes"][allocation] = base.sha256_file(
            artifact_path
        )
        del artifact
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    write_json(args.output_dir / "random_rotation_manifest.json", random_manifest)
    write_json(args.output_dir / "learned_rotation_manifest.json", learned_manifest)
    codebook_entries: dict[str, Any] = {}
    for allocation, partition in partitions.items():
        spec = core.allocation_spec(allocation)
        for component in ("key", "value"):
            for bucket in core.build_group_buckets(spec, partition, component):
                key = f"bit{bucket.bit_width}_dim{bucket.dimension}"
                values = core.lloyd_max_codebook(
                    bucket.bit_width, bucket.dimension, device="cpu"
                )
                codebook_entries[key] = {
                    "bit_width": bucket.bit_width,
                    "group_dimension": bucket.dimension,
                    "tensor_sha256": core.tensor_sha256(values),
                    "values": values.tolist(),
                }
    codebook_manifest = {
        "implementation": "local turboquant.codebook.optimal_centroids",
        "codebooks": dict(sorted(codebook_entries.items())),
        "codebook_hashes": {
            key: row["tensor_sha256"] for key, row in sorted(codebook_entries.items())
        },
    }
    write_json(args.output_dir / "codebook_manifest.json", codebook_manifest)
    print(f"[{utc_now()}] all four allocation artifacts complete", flush=True)


def _condition_artifact(condition: Any, output_dir: Path) -> dict[str, Any]:
    path = condition_artifact_path(output_dir, condition.allocation)
    if not path.is_file():
        raise FileNotFoundError(f"allocation artifact is missing: {path}")
    return torch.load(path, map_location="cpu", weights_only=True)


def _selected_rotation_artifact(
    condition: Any, artifact: Mapping[str, Any]
) -> Mapping[str, Any]:
    if condition.method == "identity":
        return artifact["identity_rotation"]
    if condition.method == "random":
        return artifact["random_rotation"]
    if condition.method == "learned":
        return artifact["learned_rotation"]
    raise ValueError(f"unknown rotation method: {condition.method}")


def load_condition_rotations(
    condition: Any,
    _rotation_dir: Path,
    _dimensions: Mapping[str, Any],
    device: torch.device,
) -> tuple[dict[str, Any], dict[str, Any]]:
    artifact = _condition_artifact(condition, ACTIVE_ARGS.output_dir)
    selected = _selected_rotation_artifact(condition, artifact)
    runtime_rotation = {
        **selected,
        "rotations": {
            key: value.to(device=device, dtype=torch.float32)
            for key, value in selected["rotations"].items()
        },
    }
    bundle = {
        "allocation": condition.allocation,
        "partition": artifact["partition"],
        "rotation": runtime_rotation,
        "device": str(device),
    }
    # A tuple selects the base runner's K/V hook and counter path.  The custom
    # hook consumes the shared bundle once because the selected artifact owns
    # both component-specific rotation maps (Learned already fixes Random V).
    return bundle, bundle


@contextlib.contextmanager
def install_condition_kv_hooks(
    model: torch.nn.Module,
    key_codec: dict[str, Any],
    value_codec: dict[str, Any],
    _centroids: torch.Tensor,
    *,
    norm_correction: bool,
    counters: dict[str, int] | None = None,
) -> Iterator[None]:
    if norm_correction is not True:
        # The base tuple path passes True; the experiment core intentionally
        # ignores reconstructed-norm correction and records that fact.
        raise ValueError("unexpected base hook contract")
    if key_codec["allocation"] != value_codec["allocation"]:
        raise ValueError("K/V allocation bundles differ")
    if allocation_core().artifact_sha256(key_codec["rotation"]) != allocation_core().artifact_sha256(
        value_codec["rotation"]
    ):
        raise ValueError("K/V runtime rotation bundles differ")
    with allocation_core().install_projection_hooks(
        model,
        key_codec["allocation"],
        key_codec["partition"],
        key_codec["rotation"],
        counters=counters,
    ):
        yield


def theoretical_kv_bytes_per_token(
    condition: Any, dimensions: Mapping[str, Any]
) -> int:
    artifact = _condition_artifact(condition, ACTIVE_ARGS.output_dir)
    storage = allocation_core().storage_accounting(
        condition.allocation,
        artifact["partition"],
        norm_bits=NORM_BITS,
        rotation_artifact=_selected_rotation_artifact(condition, artifact),
    )
    if storage["num_layers"] != int(dimensions["num_hidden_layers"]) or storage[
        "num_heads"
    ] != int(dimensions["num_key_value_heads"]):
        raise RuntimeError("storage accounting model dimensions differ")
    return int(storage["theoretical_kv_bytes_per_token"])


def bit_allocation_sha256(
    allocation: str, partition: Mapping[str, Any]
) -> str:
    core = allocation_core()
    return base.sha256_json(
        {
            "allocation": core.allocation_spec(allocation).to_dict(),
            "partition_sha256": core.artifact_sha256(partition),
        }
    )


def codebook_sha256(allocation: str, partition: Mapping[str, Any]) -> str:
    core = allocation_core()
    codebook_keys: set[tuple[int, int]] = set()
    for component in ("key", "value"):
        for bucket in core.build_group_buckets(allocation, partition, component):
            codebook_keys.add((bucket.bit_width, bucket.dimension))
    return base.sha256_json(
        {
            f"bit{bits}_dim{dimension}": core.tensor_sha256(
                core.lloyd_max_codebook(bits, dimension, device="cpu")
            )
            for bits, dimension in sorted(codebook_keys)
        }
    )


def norm_and_packing_sha256() -> str:
    return base.sha256_json(
        {
            "norm_dtype": NORM_DTYPE,
            "norm_bits": NORM_BITS,
            "norm_correction": False,
            "packing": PACKING,
        }
    )


def protocol_payload(
    args: argparse.Namespace,
    assets: Mapping[str, Any],
    manifest: Mapping[str, Any],
    condition: Any,
    mode: str,
) -> dict[str, Any]:
    core = allocation_core()
    artifact_path = condition_artifact_path(args.output_dir, condition.allocation)
    artifact = torch.load(artifact_path, map_location="cpu", weights_only=True)
    partition = artifact["partition"]
    rotation = _selected_rotation_artifact(condition, artifact)
    partition_hash = core.artifact_sha256(partition)
    allocation_hash = bit_allocation_sha256(condition.allocation, partition)
    codebook_hash = codebook_sha256(condition.allocation, partition)
    storage = core.storage_accounting(
        condition.allocation,
        partition,
        norm_bits=NORM_BITS,
        rotation_artifact=rotation,
    )
    key_rotation_hash = core.component_rotation_sha256(rotation, "key")
    value_rotation_hash = core.component_rotation_sha256(rotation, "value")
    return {
        "specification": assets["specification"],
        "mode": mode,
        "condition": condition.to_dict(),
        "model": assets["model"],
        "implementation_hashes": assets["implementation_hashes"],
        "longbench": assets["longbench"],
        "subset": {
            "manifest": str(args.output_dir / "subset_manifest.json"),
            "manifest_sha256": base.sha256_json(manifest),
            "ordered_identity_sha256": manifest["ordered_identity_sha256"],
            "sampling_seed": ROOT_SEED,
            "examples_per_condition": int(manifest["example_count"]),
            "task_bucket_counts": manifest["actual_sample_counts"],
        },
        "resume_key": {
            "condition_id": condition.condition_id,
            "model_revision": MODEL_REVISION,
            "dataset_revision": args.dataset_revision,
            "subset_manifest_sha256": base.sha256_json(manifest),
            "partition_sha256": partition_hash,
            "bit_allocation_sha256": allocation_hash,
            "key_rotation_sha256": key_rotation_hash,
            "value_rotation_sha256": value_rotation_hash,
            "codebook_sha256": codebook_hash,
            "norm_and_packing_sha256": norm_and_packing_sha256(),
        },
        "partition_sha256": partition_hash,
        "bit_allocation_sha256": allocation_hash,
        "rotation_artifact_path": str(artifact_path),
        "rotation_artifact_sha256": base.sha256_file(artifact_path),
        "key_rotation_tensor_sha256": key_rotation_hash,
        "value_rotation_tensor_sha256": value_rotation_hash,
        "codebook_sha256": codebook_hash,
        "storage": storage,
        "task_order": list(base.TASKS),
        "categories": {key: list(value) for key, value in base.CATEGORIES.items()},
        "decoding": {
            "strategy": "greedy_argmax",
            "temperature": 0,
            "maximum_batch_size": 1 if mode == "smoke" else args.batch_size,
            "batch_token_budget": args.batch_token_budget,
            "maximum_generation_lengths": assets["maximum_generation_lengths"],
            "max_context_length": assets["max_context_length"],
            "middle_truncation": "preserve tokenized front and back halves",
            "chat_policy": "native Instruct template with official LongBench exclusions",
            "no_chat_tasks": sorted(base.NO_CHAT_TASKS),
        },
        "quantization": {
            "implementation": "turboquant_plus group-wise PolarQuant quality emulation",
            "component": "pre-RoPE keys and post-projection values",
            "prompt_and_generated_tokens": True,
            "group_normalization": "independent L2 norm per group",
            "norm_dtype_for_storage": NORM_DTYPE,
            "norm_computation_dtype": "float32",
            "norm_decode_roundtrip": "float32 norm -> float16 storage -> float32 rescaling",
            "norm_correction": False,
            "packing": PACKING,
            "cache_storage": "reconstructed BF16 K/V; packed storage is theoretical",
        },
    }


ACTIVE_ARGS: argparse.Namespace


def install_base_patches(args: argparse.Namespace) -> None:
    global ACTIVE_ARGS
    ACTIVE_ARGS = args
    base.Condition = protocol.Condition
    base.all_conditions = protocol.all_conditions
    base.condition_by_id = protocol.condition_by_id
    base.validate_assets = validate_assets
    base.ensure_subset_manifest = protocol.ensure_subset_manifest
    base.load_condition_rotations = load_condition_rotations
    base.theoretical_kv_bytes_per_token = theoretical_kv_bytes_per_token
    base.protocol_payload = protocol_payload
    base.install_kv_codec_hooks = install_condition_kv_hooks
    base.append_prediction_batch = append_prediction_batch_durably


def append_prediction_batch_durably(
    path: Path, records: Sequence[Mapping[str, Any]]
) -> None:
    """Persist each prediction as its own write+flush+fsync transaction."""

    for record in records:
        base.append_prediction(path, dict(record))


def recover_prediction_tail(run_dir: Path) -> dict[str, Any] | None:
    """Recover only an incomplete final JSONL transaction, preserving bytes."""

    path = run_dir / "predictions.jsonl"
    if not path.is_file():
        return None
    payload = path.read_bytes()
    if not payload or payload.endswith(b"\n"):
        return None
    final_newline = payload.rfind(b"\n")
    prefix = payload[: final_newline + 1]
    tail = payload[final_newline + 1 :]
    try:
        decoded = tail.decode("utf-8")
        json.loads(decoded)
    except (UnicodeDecodeError, json.JSONDecodeError):
        recovery_dir = run_dir / "recovery"
        recovery_dir.mkdir(parents=True, exist_ok=True)
        recovery_path = recovery_dir / (
            f"truncated_prediction_tail_{time.time_ns()}.bin"
        )
        recovery_path.write_bytes(tail)
        repaired = prefix
        action = "removed incomplete final JSONL bytes after preserving them"
    else:
        recovery_path = None
        repaired = payload + b"\n"
        action = "restored missing final newline"
    temporary = path.with_suffix(path.suffix + ".recovery.tmp")
    temporary.write_bytes(repaired)
    temporary.replace(path)
    record = {
        "recovered_at": utc_now(),
        "action": action,
        "tail_bytes": len(tail),
        "tail_sha256": sha256_bytes(tail),
        "preserved_tail_path": str(recovery_path) if recovery_path else None,
    }
    if (run_dir / "run_config.json").is_file():
        config = read_json(run_dir / "run_config.json", {})
        recoveries = list(config.get("prediction_tail_recoveries", []))
        recoveries.append(record)
        base.update_run_status(run_dir, prediction_tail_recoveries=recoveries)
    return record


def _smoke_counter_failures(
    config: Mapping[str, Any], condition: Any | None = None
) -> list[str]:
    counters = config.get("kv_codec_counters", {})
    required = (
        "key_vectors",
        "value_vectors",
        "key_prefill_vectors",
        "value_prefill_vectors",
        "key_decode_vectors",
        "value_decode_vectors",
        "quantized_coordinates",
    )
    failures = [name for name in required if int(counters.get(name, 0)) <= 0]
    if condition is not None:
        expected_bits = {int(condition.regular_bits)}
        if condition.outlier_bits is not None:
            expected_bits.add(int(condition.outlier_bits))
        for component in ("key", "value"):
            for bit_width in (2, 3, 4):
                name = f"{component}_bit{bit_width}_coordinates"
                count = int(counters.get(name, 0))
                if bit_width in expected_bits and count <= 0:
                    failures.append(name)
                if bit_width not in expected_bits and count != 0:
                    failures.append(f"unexpected_nonzero:{name}")
    return failures


def repair_smoke_codec_counters(args: argparse.Namespace, condition: Any) -> None:
    """Repair the narrow crash window after all smoke rows were appended."""

    run_dir = args.output_dir / "smoke" / condition.condition_id
    config = read_json(run_dir / "run_config.json", {})
    if not _smoke_counter_failures(config, condition):
        return
    predictions = base.read_predictions(run_dir / "predictions.jsonl")
    if config.get("status") != "complete" or len(predictions) != len(base.TASKS):
        return
    device = torch.device(args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    first = tokenizer.bos_token_id
    if first is None:
        first = tokenizer.eos_token_id
    if first is None:
        first = 0
    second = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else first
    probes = (
        torch.tensor([[first, second]], dtype=torch.long),
        torch.tensor([[second]], dtype=torch.long),
    )
    model = base.load_model(args.model, device)
    key_bundle, value_bundle = load_condition_rotations(
        condition, args.rotation_dir, {}, device
    )
    counters: dict[str, int] = {}
    try:
        with install_condition_kv_hooks(
            model,
            key_bundle,
            value_bundle,
            torch.empty(0, device=device),
            norm_correction=True,
            counters=counters,
        ):
            base.greedy_generate_batch(
                model,
                list(probes),
                max_new_tokens=2,
                stop_ids=set(),
                suppress_initial_stop=False,
                pad_token_id=int(second),
                device=device,
            )
    finally:
        del model, key_bundle, value_bundle
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    repaired_config = {"kv_codec_counters": counters}
    remaining = _smoke_counter_failures(repaired_config, condition)
    if remaining:
        raise RuntimeError(
            f"smoke counter recovery failed for {condition.condition_id}: {remaining}"
        )
    base.update_run_status(
        run_dir,
        kv_codec_counters=counters,
        kv_codec_counter_recovered_at=utc_now(),
    )


def _synthetic_codec_gate(artifact: Mapping[str, Any]) -> dict[str, Any]:
    core = allocation_core()
    failures: list[str] = []
    partition = artifact["partition"]
    allocation = artifact["allocation"]
    generator = torch.Generator().manual_seed(ROOT_SEED)
    source = torch.randn(
        2,
        3,
        int(partition["head_dim"]),
        generator=generator,
        dtype=torch.float32,
    )
    source[0, 0].zero_()
    checks: dict[str, Any] = {}
    for method, rotations in (
        ("identity", artifact["identity_rotation"]),
        ("random", artifact["random_rotation"]),
        ("learned", artifact["learned_rotation"]),
    ):
        for component in ("key", "value"):
            first = core.reconstruct_head(
                source,
                allocation,
                partition,
                rotations,
                component=component,
                layer=0,
                head=0,
            )
            second = core.reconstruct_head(
                source,
                allocation,
                partition,
                rotations,
                component=component,
                layer=0,
                head=0,
            )
            key = f"{method}_{component}"
            checks[key] = {
                "shape_preserved": tuple(first.shape) == tuple(source.shape),
                "dtype_preserved": first.dtype == source.dtype,
                "device_preserved": first.device == source.device,
                "finite": bool(torch.isfinite(first).all()),
                "zero_preserved": bool(torch.equal(first[0, 0], source[0, 0])),
                "quantization_changed_nonzero": not torch.equal(first[1], source[1]),
                "repeat_bitwise_equal": bool(torch.equal(first, second)),
            }
            failures.extend(
                f"{key}: {name}"
                for name, passed in checks[key].items()
                if not passed
            )
    width = int(partition["num_heads"]) * int(partition["head_dim"])
    batched_projection = torch.randn(
        3, 4, width, generator=generator, dtype=torch.float32
    )
    # A fully masked/padded projection is represented by zeros at the codec
    # seam.  The compacted tensors model the batch-size changes performed by
    # greedy generation as sequences finish.
    batched_projection[0, -1].zero_()
    runtime_cases: dict[str, Any] = {}
    for component in ("key", "value"):
        prefill = core.reconstruct_projection(
            batched_projection,
            allocation,
            partition,
            artifact["random_rotation"],
            component=component,
            layer=0,
        )
        repeat = core.reconstruct_projection(
            batched_projection,
            allocation,
            partition,
            artifact["random_rotation"],
            component=component,
            layer=0,
        )
        compact_source = batched_projection[[0, 2], :1]
        compact = core.reconstruct_projection(
            compact_source,
            allocation,
            partition,
            artifact["random_rotation"],
            component=component,
            layer=0,
        )
        single = core.reconstruct_projection(
            compact_source[:1],
            allocation,
            partition,
            artifact["random_rotation"],
            component=component,
            layer=0,
        )
        runtime_cases[component] = {
            "padded_batch_finite": bool(torch.isfinite(prefill).all()),
            "zero_padding_preserved": bool(
                torch.count_nonzero(prefill[0, -1]) == 0
            ),
            "repeat_bitwise_equal": bool(torch.equal(prefill, repeat)),
            "compacted_batch_finite": bool(torch.isfinite(compact).all()),
            "compaction_consistent": bool(
                torch.allclose(compact, prefill[[0, 2], :1], atol=1e-6, rtol=1e-6)
            ),
            "single_token_decode_finite": bool(torch.isfinite(single).all()),
            "single_token_decode_consistent": bool(
                torch.allclose(single, compact[:1], atol=1e-6, rtol=1e-6)
            ),
        }
        failures.extend(
            f"runtime_{component}: {name}"
            for name, passed in runtime_cases[component].items()
            if not passed
        )
    return {
        "status": "passed" if not failures else "failed",
        "cases": checks,
        "padding_compaction_decode_cases": runtime_cases,
        "failures": failures,
    }


def smoke_gate(args: argparse.Namespace) -> dict[str, Any]:
    failures: list[str] = []
    rows: dict[str, Any] = {}
    expected_tasks = set(base.TASKS)
    for condition in protocol.all_conditions():
        run_dir = args.output_dir / "smoke" / condition.condition_id
        config = read_json(run_dir / "run_config.json", {})
        predictions = base.read_predictions(run_dir / "predictions.jsonl")
        keys = [(str(row.get("task")), str(row.get("example_id"))) for row in predictions]
        condition_failures: list[str] = []
        if config.get("status") != "complete":
            condition_failures.append(f"status={config.get('status')!r}")
        if len(predictions) != len(base.TASKS) or len(set(keys)) != len(base.TASKS):
            condition_failures.append("not exactly one unique prediction per task")
        if {task for task, _ in keys} != expected_tasks:
            condition_failures.append("task coverage differs")
        if any(not str(row.get("prediction", "")).strip() for row in predictions):
            condition_failures.append("empty output")
        missing_counters = _smoke_counter_failures(config, condition)
        if missing_counters:
            condition_failures.append(f"codec counter failures: {missing_counters}")
        rows[condition.condition_id] = {
            "predictions": len(predictions),
            "counters": config.get("kv_codec_counters", {}),
            "failures": condition_failures,
        }
        failures.extend(
            f"{condition.condition_id}: {value}" for value in condition_failures
        )
    result = {
        "status": "passed" if not failures else "failed",
        "condition_count": len(rows),
        "expected_tasks_per_condition": len(base.TASKS),
        "conditions": rows,
        "failures": failures,
    }
    write_json(args.output_dir / "smoke_gate.json", result)
    return result


def static_gates(args: argparse.Namespace) -> dict[str, Any]:
    """Audit immutable sampling/partition/rotation/codec requirements."""

    assets = validate_assets(args, deep=False)
    manifest = protocol.ensure_subset_manifest(
        args.output_dir,
        args.data_dir,
        dataset_revision=args.dataset_revision,
        data_hashes=assets["longbench"]["data_hashes"],
    )
    core = allocation_core()
    failures: list[str] = []
    try:
        protocol.validate_subset_manifest(manifest)
        subset_exact = True
    except (TypeError, ValueError) as error:
        subset_exact = False
        failures.append(f"subset validation: {error}")
    checks: dict[str, Any] = {
        "subset_count": int(manifest["example_count"]) == 734,
        "subset_unique": len(
            {(row["task"], row["example_id"]) for row in manifest["examples"]}
        )
        == 734,
        "subset_exact_bucket_counts": subset_exact,
        "longbench_excluded_from_training": (
            assets["source"].get("calibration_tensor_sha256")
            == CALIBRATION_TOKEN_SHA256
            and assets["wikitext"].get("repository") == "Salesforce/wikitext"
            and set(assets["longbench"].get("data_hashes", {})) == set(base.TASKS)
        ),
    }
    for name, passed in checks.items():
        if not passed:
            failures.append(name)
    statistics_payload = read_json(args.output_dir / "channel_statistics.json", {})
    try:
        key_statistics = torch.tensor(
            statistics_payload["components"]["k"]["values"], dtype=torch.float32
        )
        value_statistics = torch.tensor(
            statistics_payload["components"]["v"]["values"], dtype=torch.float32
        )
        if tuple(key_statistics.shape) != (32, 8, 128) or tuple(
            value_statistics.shape
        ) != (32, 8, 128):
            raise ValueError("channel-statistics shape differs from 32x8x128")
    except (KeyError, TypeError, ValueError) as error:
        key_statistics = value_statistics = torch.empty(0)
        failures.append(f"channel statistics cannot rebuild partitions: {error}")
    artifact_audits: dict[str, Any] = {}
    value_hashes: dict[str, str] = {}
    for allocation in protocol.ALLOCATION_ORDER:
        path = condition_artifact_path(args.output_dir, allocation)
        if not path.is_file():
            failures.append(f"missing artifact {allocation}")
            continue
        artifact = torch.load(path, map_location="cpu", weights_only=True)
        if key_statistics.numel() and value_statistics.numel():
            rebuilt_partition = core.build_partition_artifact(
                allocation, key_statistics, value_statistics
            )
            if core.artifact_sha256(artifact["partition"]) != core.artifact_sha256(
                rebuilt_partition
            ):
                failures.append(
                    f"{allocation}: partition differs from the mean-magnitude rule"
                )
        if allocation in {"fixed32", "kmeans2"}:
            saved_partition_path = args.output_dir / f"{allocation}_partitions.pt"
            if not saved_partition_path.is_file():
                failures.append(f"{allocation}: saved partition file is missing")
            else:
                saved_partition = torch.load(
                    saved_partition_path, map_location="cpu", weights_only=True
                )
                if core.artifact_sha256(saved_partition) != core.artifact_sha256(
                    artifact["partition"]
                ):
                    failures.append(
                        f"{allocation}: saved and embedded partitions differ"
                    )
        audit = _audit_condition_artifact(
            artifact, orthogonality_tolerance=ORTHOGONALITY_TOLERANCE
        )
        artifact_audits[allocation] = audit
        if audit["status"] != "passed":
            failures.extend(
                f"{allocation}: {value}" for value in audit["failures"]
            )
        if int(artifact.get("step", -1)) != TRAINING_STEPS:
            failures.append(f"{allocation}: not step 10000")
        value_hashes[allocation] = audit.get("random_value_hash", "")
        storage = core.storage_accounting(
            allocation,
            artifact["partition"],
            norm_bits=NORM_BITS,
            rotation_artifact=artifact["learned_rotation"],
        )
        expected_index_bpe = {
            "uniform2": 2.0,
            "fixed32": 2.5,
            "uniform3": 3.0,
        }.get(allocation)
        if expected_index_bpe is not None and not math.isclose(
            float(storage["kv_average_index_bpe"]),
            expected_index_bpe,
            rel_tol=0,
            abs_tol=1e-12,
        ):
            failures.append(
                f"{allocation}: index BPE={storage['kv_average_index_bpe']}, "
                f"expected {expected_index_bpe}"
            )
        audit["storage"] = {
            "key": storage["components"]["key"],
            "value": storage["components"]["value"],
            "kv_average_index_bpe": storage["kv_average_index_bpe"],
            "kv_average_effective_bpe": storage["kv_average_effective_bpe"],
            "theoretical_kv_bytes_per_token": storage[
                "theoretical_kv_bytes_per_token"
            ],
        }
        codec_audit = _synthetic_codec_gate(artifact)
        artifact_audits[allocation]["synthetic_codec"] = codec_audit
        if codec_audit["status"] != "passed":
            failures.extend(
                f"{allocation}: {value}" for value in codec_audit["failures"]
            )
    result = {
        "static_status": "passed" if not failures else "failed",
        "root_seed": ROOT_SEED,
        "sampling": checks,
        "artifacts": artifact_audits,
        "random_learned_value_hashes": value_hashes,
        "failures": failures,
    }
    write_json(args.output_dir / "correctness_gates.json", result)
    if failures:
        raise RuntimeError("static correctness gates failed: " + "; ".join(failures))
    return result


def assert_all_smoke_passed(args: argparse.Namespace) -> None:
    gate = smoke_gate(args)
    if gate["status"] != "passed":
        raise RuntimeError(
            "all 12 smoke conditions must pass before any full run: "
            + "; ".join(gate["failures"])
        )


def _apply_rope(values: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    first, second = values.chunk(2, dim=-1)
    rotated_half = torch.cat((-second, first), dim=-1)
    return values * cos.unsqueeze(0) + rotated_half * sin.unsqueeze(0)


def _offline_condition_metrics(
    validation: Mapping[str, torch.Tensor],
    artifact: Mapping[str, Any],
    rotation: Mapping[str, Any],
    *,
    device: torch.device,
    attention_tokens: int,
) -> dict[str, Any]:
    core = allocation_core()
    allocation = str(artifact["allocation"])
    partition = artifact["partition"]
    original_sse = {"key": 0.0, "value": 0.0}
    normalized_key_sse = 0.0
    counts = {"key": 0, "value": 0}
    chunk_tokens = 512
    started = time.perf_counter()
    with torch.inference_mode():
        for component, tensor_name in (("key", "k"), ("value", "v")):
            source = validation[tensor_name]
            for layer in range(source.shape[0]):
                for head in range(source.shape[1]):
                    for start in range(0, source.shape[2], chunk_tokens):
                        batch = source[layer, head, start : start + chunk_tokens].to(
                            device=device, dtype=torch.float32
                        )
                        reconstructed = core.reconstruct_head(
                            batch,
                            allocation,
                            partition,
                            rotation,
                            component=component,
                            layer=layer,
                            head=head,
                        )
                        original_sse[component] += float(
                            (reconstructed - batch).square().sum()
                        )
                        counts[component] += batch.numel()
                        if component == "key":
                            source_norm = torch.linalg.vector_norm(
                                batch, dim=-1, keepdim=True
                            )
                            reconstructed_norm = torch.linalg.vector_norm(
                                reconstructed, dim=-1, keepdim=True
                            )
                            source_unit = batch / source_norm.clamp_min(1e-12)
                            reconstructed_unit = reconstructed / reconstructed_norm.clamp_min(
                                1e-12
                            )
                            valid = source_norm > 0
                            normalized_key_sse += float(
                                torch.where(
                                    valid,
                                    source_unit - reconstructed_unit,
                                    torch.zeros_like(source_unit),
                                )
                                .square()
                                .sum()
                            )

        length = attention_tokens
        cos = validation["cos"][:length].to(device=device, dtype=torch.float32)
        sin = validation["sin"][:length].to(device=device, dtype=torch.float32)
        causal = torch.ones((length, length), dtype=torch.bool, device=device).tril()
        logit_sse = probability_kl = output_sse = 0.0
        logit_values = probability_rows = output_values = 0
        query_heads = int(validation["q"].shape[2])
        kv_heads = int(partition["num_heads"])
        repeats = query_heads // kv_heads
        head_dim = int(partition["head_dim"])
        for layer in range(int(partition["num_layers"])):
            q = validation["q"][layer, :length].permute(1, 0, 2).to(
                device=device, dtype=torch.float32
            )
            k = validation["k"][layer, :, :length].to(
                device=device, dtype=torch.float32
            )
            v = validation["v"][layer, :, :length].to(
                device=device, dtype=torch.float32
            )
            reconstructed_k = torch.stack(
                [
                    core.reconstruct_head(
                        k[head],
                        allocation,
                        partition,
                        rotation,
                        component="key",
                        layer=layer,
                        head=head,
                    )
                    for head in range(kv_heads)
                ]
            )
            reconstructed_v = torch.stack(
                [
                    core.reconstruct_head(
                        v[head],
                        allocation,
                        partition,
                        rotation,
                        component="value",
                        layer=layer,
                        head=head,
                    )
                    for head in range(kv_heads)
                ]
            )
            q_rope = _apply_rope(q, cos, sin)
            k_rope = _apply_rope(k, cos, sin).repeat_interleave(repeats, dim=0)
            reconstructed_k_rope = _apply_rope(
                reconstructed_k, cos, sin
            ).repeat_interleave(repeats, dim=0)
            repeated_v = v.repeat_interleave(repeats, dim=0)
            repeated_reconstructed_v = reconstructed_v.repeat_interleave(
                repeats, dim=0
            )
            scale = 1.0 / math.sqrt(head_dim)
            logits = torch.matmul(q_rope, k_rope.transpose(-1, -2)) * scale
            reconstructed_logits = (
                torch.matmul(q_rope, reconstructed_k_rope.transpose(-1, -2))
                * scale
            )
            valid_error = (reconstructed_logits - logits).masked_select(
                causal.unsqueeze(0)
            )
            logit_sse += float(valid_error.square().sum())
            logit_values += valid_error.numel()
            logits = logits.masked_fill(~causal.unsqueeze(0), float("-inf"))
            reconstructed_logits = reconstructed_logits.masked_fill(
                ~causal.unsqueeze(0), float("-inf")
            )
            probabilities = torch.softmax(logits, dim=-1)
            reconstructed_probabilities = torch.softmax(
                reconstructed_logits, dim=-1
            )
            probability_kl += float(
                (
                    probabilities
                    * (
                        torch.log(probabilities.clamp_min(1e-12))
                        - torch.log(reconstructed_probabilities.clamp_min(1e-12))
                    )
                ).sum()
            )
            probability_rows += probabilities.shape[0] * probabilities.shape[1]
            original_output = torch.matmul(probabilities, repeated_v)
            reconstructed_output = torch.matmul(
                reconstructed_probabilities, repeated_reconstructed_v
            )
            output_sse += float(
                (reconstructed_output - original_output).square().sum()
            )
            output_values += reconstructed_output.numel()
    orthogonality = core.validate_rotation_artifact(
        rotation, allocation, partition
    )["orthogonality_max_abs"]
    return {
        "validation_tokens": int(validation["k"].shape[2]),
        "attention_tokens": attention_tokens,
        "original_scale_key_mse": original_sse["key"] / counts["key"],
        "normalized_key_mse": normalized_key_sse / counts["key"],
        "value_reconstruction_mse": original_sse["value"] / counts["value"],
        "attention_logit_mse": logit_sse / max(logit_values, 1),
        "attention_probability_kl": probability_kl / max(probability_rows, 1),
        "attention_output_mse": output_sse / max(output_values, 1),
        "orthogonality_max_abs": orthogonality,
        "elapsed_seconds": time.perf_counter() - started,
    }


def diagnostics_stage(args: argparse.Namespace) -> None:
    """Measure held-out reconstruction/attention distortion for all conditions."""

    static_gates(args)
    validation_path = args.output_dir / "activations/validation.pt"
    validation = torch.load(validation_path, map_location="cpu", weights_only=True)
    core = allocation_core()
    device = torch.device(args.device)
    rows: list[dict[str, Any]] = []
    for condition in protocol.all_conditions():
        artifact = _condition_artifact(condition, args.output_dir)
        selected_rotation = _selected_rotation_artifact(condition, artifact)
        print(f"[{utc_now()}] diagnostics {condition.condition_id}", flush=True)
        metrics = _offline_condition_metrics(
            validation,
            artifact,
            selected_rotation,
            device=device,
            attention_tokens=args.diagnostic_tokens,
        )
        rows.append({**condition.to_dict(), **metrics})
    write_csv(args.output_dir / "offline_diagnostics.csv", rows)
    print(f"[{utc_now()}] offline diagnostics complete", flush=True)


def child_command(
    args: argparse.Namespace, stage: str, condition: Any | None = None
) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "experiments.spin_turboquant.longbench_e_rotation_allocation",
        "--stage",
        stage,
        "--model",
        str(args.model),
        "--source-dir",
        str(args.source_dir),
        "--longbench-repo",
        str(args.longbench_repo),
        "--data-dir",
        str(args.data_dir),
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
        "--checkpoint-interval",
        str(args.checkpoint_interval),
        "--diagnostic-tokens",
        str(args.diagnostic_tokens),
    ]
    if args.max_context_length is not None:
        command.extend(["--max-context-length", str(args.max_context_length)])
    if condition is not None:
        command.extend(["--condition", condition.condition_id])
    return command


def run_child(args: argparse.Namespace, stage: str, condition: Any | None = None) -> None:
    command = child_command(args, stage, condition)
    log_path = args.output_dir / "orchestrator.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as log:
        header = f"[{utc_now()}] $ {' '.join(command)}\n"
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


def write_study_config(
    args: argparse.Namespace,
    assets: Mapping[str, Any],
    status: str,
    **updates: Any,
) -> None:
    manifest = protocol.ensure_subset_manifest(
        args.output_dir,
        args.data_dir,
        dataset_revision=args.dataset_revision,
        data_hashes=assets["longbench"]["data_hashes"],
    )
    path = args.output_dir / "study_config.json"
    existing = read_json(path, {})
    payload = {
        "schema_version": 1,
        "created_at": existing.get("created_at", utc_now()),
        "updated_at": utc_now(),
        "status": status,
        "specification": assets["specification"],
        "model": assets["model"],
        "source": assets["source"],
        "longbench": assets["longbench"],
        "wikitext": assets["wikitext"],
        "environment": existing.get("environment")
        or _environment_metadata(torch.device(args.device)),
        "subset_manifest": {
            "path": str(args.output_dir / "subset_manifest.json"),
            "sha256": base.sha256_json(manifest),
            "ordered_identity_sha256": manifest["ordered_identity_sha256"],
            "example_count": manifest["example_count"],
        },
        "root_seed": ROOT_SEED,
        "rng_streams": rng_manifest(
            [
                "rotation",
                *[f"minibatch:{name}" for name in protocol.ALLOCATION_ORDER],
                "bootstrap",
            ]
        ),
        "conditions": [value.to_dict() for value in protocol.all_conditions()],
        "condition_count": 12,
        "predictions_expected": 8_808,
        "execution_order": "Identity -> Random -> Learned within each allocation",
        "training": {
            "steps": TRAINING_STEPS,
            "batch_tokens_per_head": TRAINING_BATCH_TOKENS,
            "initial_lr": INITIAL_LEARNING_RATE,
            "eta_min": FINAL_LEARNING_RATE,
        },
        "bootstrap_samples": BOOTSTRAP_SAMPLES,
        "batch_size": args.batch_size,
        "batch_token_budget": args.batch_token_budget,
        "quality_emulation": True,
        **updates,
    }
    write_json(path, payload)


def orchestrate(args: argparse.Namespace) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    assets = validate_assets(args, deep=False)
    write_study_config(args, assets, "running", started_at=utc_now())
    try:
        run_child(args, "capture")
        run_child(args, "artifacts")
        run_child(args, "gates")
        run_child(args, "diagnostics")
        for condition in protocol.all_conditions():
            run_child(args, "smoke", condition)
        assert_all_smoke_passed(args)
        for condition in protocol.all_conditions():
            run_child(args, "full", condition)
        run_child(args, "report")
    except BaseException as error:
        write_study_config(
            args,
            assets,
            "failed",
            error=f"{type(error).__name__}: {error}",
        )
        raise
    audit = read_json(args.output_dir / "completion_audit.json", {})
    if audit.get("status") != "complete":
        raise RuntimeError("completion audit did not mark the study complete")
    write_study_config(args, assets, "complete", completed_at=utc_now(), audit=audit)


def _safe_float(value: Any) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return math.nan
    return result


def _combined_rows(args: argparse.Namespace) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    predictions: list[dict[str, Any]] = []
    scores: list[dict[str, Any]] = []
    for condition in protocol.all_conditions():
        run_dir = args.output_dir / "full" / condition.condition_id
        config = read_json(run_dir / "run_config.json", {})
        if config.get("status") != "complete":
            raise RuntimeError(f"full condition is incomplete: {condition.condition_id}")
        condition_predictions = base.read_predictions(run_dir / "predictions.jsonl")
        condition_scores = base.read_csv(run_dir / "scores.csv")
        if len(condition_predictions) != 734 or len(condition_scores) != 734:
            raise RuntimeError(
                f"{condition.condition_id} has {len(condition_predictions)} predictions "
                f"and {len(condition_scores)} scores; expected 734 each"
            )
        for row in condition_predictions:
            predictions.append({**row, "allocation": condition.allocation})
        for row in condition_scores:
            scores.append({**row, "allocation": condition.allocation})
    return predictions, scores


def _storage_and_cluster_rows(args: argparse.Namespace) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    core = allocation_core()
    storage_rows: list[dict[str, Any]] = []
    cluster_rows: list[dict[str, Any]] = []
    for allocation in protocol.ALLOCATION_ORDER:
        artifact = torch.load(
            condition_artifact_path(args.output_dir, allocation),
            map_location="cpu",
            weights_only=True,
        )
        partition = artifact["partition"]
        storage = core.storage_accounting(
            allocation,
            partition,
            norm_bits=NORM_BITS,
            rotation_artifact=artifact["learned_rotation"],
        )
        key_storage = storage["components"]["key"]
        value_storage = storage["components"]["value"]
        static_partition_bytes = sum(
            int(mask.numel() * mask.element_size())
            for mask in partition["outlier_masks"].values()
        )
        storage_rows.append(
            {
                "allocation": allocation,
                "key_index_bpe": key_storage["index_bpe"],
                "key_norm_bpe": key_storage["norm_bpe"],
                "key_alignment_bpe": key_storage["alignment_bpe"],
                "key_effective_bpe": key_storage["effective_bpe"],
                "value_index_bpe": value_storage["index_bpe"],
                "value_norm_bpe": value_storage["norm_bpe"],
                "value_alignment_bpe": value_storage["alignment_bpe"],
                "value_effective_bpe": value_storage["effective_bpe"],
                "kv_average_index_bpe": storage["kv_average_index_bpe"],
                "kv_average_effective_bpe": storage[
                    "kv_average_effective_bpe"
                ],
                "key_packed_bytes_per_head_vector": key_storage[
                    "packed_bytes_per_head_vector_mean"
                ],
                "value_packed_bytes_per_head_vector": value_storage[
                    "packed_bytes_per_head_vector_mean"
                ],
                "key_theoretical_bytes_per_token": key_storage[
                    "theoretical_bytes_per_token"
                ],
                "value_theoretical_bytes_per_token": value_storage[
                    "theoretical_bytes_per_token"
                ],
                "theoretical_kv_bytes_per_token": storage[
                    "theoretical_kv_bytes_per_token"
                ],
                "static_partition_bytes_bool_masks": static_partition_bytes,
                "static_rotation_bytes_fp32": storage["static_rotation_bytes"],
                "norm_bits": NORM_BITS,
                "packing": PACKING,
                "quality_emulation": True,
            }
        )
        spec = core.allocation_spec(allocation)
        for component in ("key", "value"):
            rows = storage["head_rows"][component]
            for row in rows:
                outlier_count = int(row["outlier_channels"])
                cluster_rows.append(
                    {
                        "allocation": allocation,
                        "component": component,
                        "layer": row["layer"],
                        "head": row["head"],
                        "regular_channels": int(partition["head_dim"])
                        - outlier_count,
                        "outlier_channels": outlier_count,
                        "regular_bits": spec.regular_bits,
                        "outlier_bits": spec.outlier_bits,
                        "index_bpe": row["index_bits"] / int(partition["head_dim"]),
                        "packed_index_bytes": row["packed_index_bytes"],
                        "norm_bytes": row["norm_bytes"],
                        "alignment_bytes": row["alignment_bytes"],
                        "effective_bpe": row["packed_bytes"]
                        * 8
                        / int(partition["head_dim"]),
                    }
                )
    return storage_rows, cluster_rows


def _write_plots(
    output_dir: Path,
    condition_summary: Sequence[Mapping[str, Any]],
    comparisons: Sequence[Mapping[str, Any]],
    storage_rows: Sequence[Mapping[str, Any]],
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plot_dir = output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    labels = [str(row["condition_id"]) for row in condition_summary]
    scores = [float(row["task_macro_average"]) for row in condition_summary]
    fig, axis = plt.subplots(figsize=(13, 5))
    axis.bar(labels, scores)
    axis.set_ylabel("LongBench-E task macro score")
    axis.tick_params(axis="x", rotation=65)
    fig.tight_layout()
    fig.savefig(plot_dir / "condition_scores.png", dpi=180)
    plt.close(fig)

    learned = [
        row for row in comparisons if row["contrast"] == "learned_minus_random"
    ]
    if len(learned) != len(protocol.ALLOCATION_ORDER):
        raise RuntimeError("plot requires one Learned-minus-Random row per allocation")
    fig, axis = plt.subplots(figsize=(8, 4))
    axis.bar(
        [str(row["allocation"]) for row in learned],
        [float(row["difference"]) for row in learned],
    )
    axis.axhline(0, color="black", linewidth=1)
    axis.set_ylabel("Learned - Random")
    fig.tight_layout()
    fig.savefig(plot_dir / "learned_minus_random.png", dpi=180)
    plt.close(fig)

    score_by_allocation = {
        row["allocation"]: float(row["task_macro_average"])
        for row in condition_summary
        if row["method"] == "learned"
    }
    fig, axis = plt.subplots(figsize=(7, 5))
    for row in storage_rows:
        allocation = str(row["allocation"])
        axis.scatter(
            [float(row["kv_average_effective_bpe"])],
            [score_by_allocation[allocation]],
            label=allocation,
        )
    axis.set_xlabel("K/V average effective BPE")
    axis.set_ylabel("Learned task macro score")
    axis.legend()
    fig.tight_layout()
    fig.savefig(plot_dir / "score_bpe_pareto.png", dpi=180)
    plt.close(fig)


def completion_audit(
    args: argparse.Namespace,
    predictions: Sequence[Mapping[str, Any]],
    scores: Sequence[Mapping[str, Any]],
    comparisons: Sequence[Mapping[str, Any]],
    storage_rows: Sequence[Mapping[str, Any]],
    cluster_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Perform the final, artifact-backed audit required by the study plan."""

    failures: list[str] = []
    current_assets = validate_assets(args, deep=False)
    prediction_keys = [
        (str(row["condition_id"]), str(row["task"]), str(row["example_id"]))
        for row in predictions
    ]
    score_keys = [
        (str(row["condition_id"]), str(row["task"]), str(row["example_id"]))
        for row in scores
    ]
    if len(predictions) != 8_808 or len(set(prediction_keys)) != 8_808:
        failures.append(
            f"prediction rows/unique={len(predictions)}/{len(set(prediction_keys))}, expected 8808"
        )
    if len(scores) != 8_808 or set(score_keys) != set(prediction_keys):
        failures.append("scores are not a one-to-one match for predictions")
    empty_count = sum(not str(row.get("prediction", "")).strip() for row in predictions)
    error_count = sum(
        bool(row.get(key))
        for row in predictions
        for key in ("error", "generation_error", "scoring_error")
    )
    nonfinite_scores = sum(
        not math.isfinite(_safe_float(row.get("score"))) for row in scores
    )
    truncation_count = sum(bool(row.get("prompt_truncated")) for row in predictions)
    if empty_count:
        failures.append(f"empty output count={empty_count}")
    if error_count:
        failures.append(f"error field count={error_count}")
    if nonfinite_scores:
        failures.append(f"nonfinite score count={nonfinite_scores}")

    manifest = read_json(args.output_dir / "subset_manifest.json", {})
    try:
        protocol.validate_subset_manifest(manifest)
    except (TypeError, ValueError) as error:
        failures.append(f"subset manifest validation failed: {error}")
    expected_subset_hash = base.sha256_json(manifest)
    expected_order = [
        (str(row["task"]), str(row["example_id"])) for row in manifest.get("examples", [])
    ]
    core = allocation_core()
    learned_manifest = read_json(
        args.output_dir / "learned_rotation_manifest.json", {}
    )
    artifact_audits: dict[str, Any] = {}
    expected_partition_hashes: dict[str, str] = {}
    expected_allocation_hashes: dict[str, str] = {}
    expected_codebook_hashes: dict[str, str] = {}
    expected_rotation_hashes: dict[str, dict[str, dict[str, str]]] = {}
    for allocation in protocol.ALLOCATION_ORDER:
        path = condition_artifact_path(args.output_dir, allocation)
        if not path.is_file():
            failures.append(f"{allocation}: allocation artifact is missing")
            continue
        artifact = torch.load(path, map_location="cpu", weights_only=True)
        partition_hash = core.artifact_sha256(artifact["partition"])
        expected_partition_hashes[allocation] = partition_hash
        expected_allocation_hashes[allocation] = bit_allocation_sha256(
            allocation, artifact["partition"]
        )
        expected_codebook_hashes[allocation] = codebook_sha256(
            allocation, artifact["partition"]
        )
        method_hashes: dict[str, dict[str, str]] = {}
        for method, artifact_key in (
            ("identity", "identity_rotation"),
            ("random", "random_rotation"),
            ("learned", "learned_rotation"),
        ):
            rotation = artifact[artifact_key]
            method_hashes[method] = {
                "key": core.component_rotation_sha256(rotation, "key"),
                "value": core.component_rotation_sha256(rotation, "value"),
            }
        expected_rotation_hashes[allocation] = method_hashes
        artifact_file_hash = base.sha256_file(path)
        completed_step = int(
            artifact.get("learned_rotation", {})
            .get("training", {})
            .get("completed_step", -1)
        )
        recorded_file_hash = (
            learned_manifest.get("artifact_file_hashes", {}).get(allocation)
        )
        artifact_failures: list[str] = []
        if int(artifact.get("step", -1)) != TRAINING_STEPS:
            artifact_failures.append("composite artifact step is not 10000")
        if completed_step != TRAINING_STEPS:
            artifact_failures.append("learned rotation completed_step is not 10000")
        if recorded_file_hash != artifact_file_hash:
            artifact_failures.append("learned manifest file hash differs")
        if method_hashes["random"]["value"] != method_hashes["learned"]["value"]:
            artifact_failures.append("Random/Learned Value hashes differ")
        latest_checkpoint_path = (
            args.output_dir / "checkpoints" / f"{allocation}_latest.pt"
        )
        fixed_checkpoint_path = (
            args.output_dir / "checkpoints" / f"{allocation}_step10000.pt"
        )
        latest_checkpoint_hash: str | None = None
        fixed_checkpoint_hash: str | None = None
        if latest_checkpoint_path.is_file():
            latest_checkpoint = torch.load(
                latest_checkpoint_path, map_location="cpu", weights_only=True
            )
            latest_checkpoint_hash = base.sha256_file(latest_checkpoint_path)
            if (
                latest_checkpoint.get("protocol") != artifact.get("protocol")
                or int(latest_checkpoint.get("step", -1)) != TRAINING_STEPS
            ):
                artifact_failures.append("latest optimizer checkpoint differs")
        else:
            artifact_failures.append("latest optimizer checkpoint is missing")
        if fixed_checkpoint_path.is_file():
            fixed_checkpoint = torch.load(
                fixed_checkpoint_path, map_location="cpu", weights_only=True
            )
            fixed_checkpoint_hash = base.sha256_file(fixed_checkpoint_path)
            if (
                fixed_checkpoint.get("protocol") != artifact.get("protocol")
                or int(fixed_checkpoint.get("step", -1)) != TRAINING_STEPS
                or core.artifact_sha256(fixed_checkpoint)
                != core.artifact_sha256(artifact)
            ):
                artifact_failures.append(
                    "step-10000 checkpoint differs from composite artifact"
                )
        else:
            artifact_failures.append("step-10000 checkpoint is missing")
        failures.extend(
            f"{allocation}: {message}" for message in artifact_failures
        )
        artifact_audits[allocation] = {
            "path": str(path),
            "file_sha256": artifact_file_hash,
            "manifest_file_sha256": recorded_file_hash,
            "partition_sha256": partition_hash,
            "step": int(artifact.get("step", -1)),
            "learned_completed_step": completed_step,
            "latest_checkpoint_file_sha256": latest_checkpoint_hash,
            "step10000_checkpoint_file_sha256": fixed_checkpoint_hash,
            "rotation_hashes": method_hashes,
            "failures": artifact_failures,
        }

    coverage: dict[str, Any] = {}
    value_hashes: dict[str, dict[str, str]] = defaultdict(dict)
    subset_hashes: dict[str, str] = {}
    partition_hashes: dict[str, dict[str, str]] = defaultdict(dict)
    for condition in protocol.all_conditions():
        rows = [row for row in predictions if row["condition_id"] == condition.condition_id]
        condition_scores = [
            row for row in scores if row["condition_id"] == condition.condition_id
        ]
        identities = [(str(row["task"]), str(row["example_id"])) for row in rows]
        score_identities = [
            (str(row["task"]), str(row["example_id"]))
            for row in condition_scores
        ]
        tasks = {str(row["task"]) for row in rows}
        categories = {
            base.TASK_TO_CATEGORY[task]
            for task in tasks
            if task in base.TASK_TO_CATEGORY
        }
        unique_identities = set(identities)
        duplicate_count = len(identities) - len(unique_identities)
        condition_empty_count = sum(
            not str(row.get("prediction", "")).strip() for row in rows
        )
        condition_error_count = sum(
            any(
                bool(row.get(key))
                for key in ("error", "generation_error", "scoring_error")
            )
            for row in rows
        )
        condition_nonfinite_scores = sum(
            not math.isfinite(_safe_float(row.get("score")))
            for row in condition_scores
        )
        condition_truncation_count = sum(
            bool(row.get("prompt_truncated")) for row in rows
        )
        condition_failures: list[str] = []
        if identities != expected_order:
            condition_failures.append("ordered identities differ from subset manifest")
        if score_identities != expected_order:
            condition_failures.append("ordered score identities differ from subset manifest")
        if tasks != set(base.TASKS):
            condition_failures.append("task coverage differs")
        if categories != set(base.CATEGORIES):
            condition_failures.append("category coverage differs")
        if duplicate_count:
            condition_failures.append(f"duplicate predictions={duplicate_count}")
        if condition_empty_count:
            condition_failures.append(f"empty outputs={condition_empty_count}")
        if condition_error_count:
            condition_failures.append(f"error outputs={condition_error_count}")
        if condition_nonfinite_scores:
            condition_failures.append(
                f"nonfinite scores={condition_nonfinite_scores}"
            )
        run_config = read_json(
            args.output_dir / "full" / condition.condition_id / "run_config.json", {}
        )
        run_protocol = run_config.get("protocol", {})
        resume = run_protocol.get("resume_key", {})
        subset_hash = str(resume.get("subset_manifest_sha256", ""))
        subset_hashes[condition.condition_id] = subset_hash
        partition_hash = str(resume.get("partition_sha256", ""))
        partition_hashes[condition.allocation][condition.method] = partition_hash
        value_hashes[condition.allocation][condition.method] = str(
            resume.get("value_rotation_sha256", "")
        )
        if run_config.get("status") != "complete":
            condition_failures.append("run_config status is not complete")
        if run_protocol.get("mode") != "full":
            condition_failures.append("run protocol mode is not full")
        if resume.get("condition_id") != condition.condition_id:
            condition_failures.append("resume condition ID differs")
        if resume.get("model_revision") != MODEL_REVISION:
            condition_failures.append("resume model revision differs")
        if resume.get("dataset_revision") != args.dataset_revision:
            condition_failures.append("resume dataset revision differs")
        if run_protocol.get("model") != current_assets.get("model"):
            condition_failures.append("model protocol differs from current assets")
        if run_protocol.get("longbench") != current_assets.get("longbench"):
            condition_failures.append("LongBench protocol differs from current assets")
        if run_protocol.get("specification") != current_assets.get("specification"):
            condition_failures.append("specification hash differs")
        if subset_hash != expected_subset_hash:
            condition_failures.append("resume subset manifest hash differs")
        if (
            run_protocol.get("subset", {}).get("manifest_sha256")
            != expected_subset_hash
        ):
            condition_failures.append("protocol subset manifest hash differs")
        if (
            run_protocol.get("subset", {}).get("ordered_identity_sha256")
            != manifest.get("ordered_identity_sha256")
        ):
            condition_failures.append("ordered identity hash differs")
        if partition_hash != expected_partition_hashes.get(condition.allocation):
            condition_failures.append("partition hash differs from allocation artifact")
        if (
            resume.get("bit_allocation_sha256")
            != expected_allocation_hashes.get(condition.allocation)
        ):
            condition_failures.append("bit-allocation hash differs")
        if (
            resume.get("codebook_sha256")
            != expected_codebook_hashes.get(condition.allocation)
        ):
            condition_failures.append("codebook hash differs")
        if resume.get("norm_and_packing_sha256") != norm_and_packing_sha256():
            condition_failures.append("norm-and-packing hash differs")
        if (
            run_protocol.get("implementation_hashes")
            != current_assets.get("implementation_hashes")
        ):
            condition_failures.append("implementation hashes differ")
        artifact_path = condition_artifact_path(
            args.output_dir, condition.allocation
        )
        if (
            not artifact_path.is_file()
            or run_protocol.get("rotation_artifact_sha256")
            != base.sha256_file(artifact_path)
        ):
            condition_failures.append("composite rotation artifact hash differs")
        expected_method_hashes = expected_rotation_hashes.get(
            condition.allocation, {}
        ).get(condition.method, {})
        if resume.get("key_rotation_sha256") != expected_method_hashes.get("key"):
            condition_failures.append("Key rotation hash differs")
        if resume.get("value_rotation_sha256") != expected_method_hashes.get("value"):
            condition_failures.append("Value rotation hash differs")
        coverage[condition.condition_id] = {
            "predictions": len(rows),
            "unique_predictions": len(unique_identities),
            "scores": len(condition_scores),
            "tasks": len(tasks),
            "categories": len(categories),
            "empty_outputs": condition_empty_count,
            "error_outputs": condition_error_count,
            "duplicate_predictions": duplicate_count,
            "nonfinite_scores": condition_nonfinite_scores,
            "prompt_truncations": condition_truncation_count,
            "subset_manifest_sha256": subset_hash,
            "partition_sha256": partition_hash,
            "failures": condition_failures,
        }
        failures.extend(
            f"{condition.condition_id}: {value}" for value in condition_failures
        )
    for allocation, hashes in value_hashes.items():
        if not hashes.get("random") or hashes.get("random") != hashes.get("learned"):
            failures.append(f"{allocation}: Random/Learned Value hashes differ")
        if len(set(partition_hashes[allocation].values())) != 1:
            failures.append(f"{allocation}: condition partition hashes differ")
    if set(subset_hashes.values()) != {expected_subset_hash}:
        failures.append("conditions do not all share the exact subset manifest hash")

    static = read_json(args.output_dir / "correctness_gates.json", {})
    smoke = read_json(args.output_dir / "smoke_gate.json", {})
    if static.get("static_status") != "passed":
        failures.append("static gate is not passed")
    if smoke.get("status") != "passed":
        failures.append("smoke gate is not passed")
    expected_comparison_ids = {
        contrast.comparison_id for contrast in protocol.PAIRED_CONTRASTS
    }
    comparison_ids = {str(row.get("comparison_id")) for row in comparisons}
    if len(comparisons) != len(expected_comparison_ids) or comparison_ids != expected_comparison_ids:
        failures.append(
            f"paired comparison IDs differ: rows={len(comparisons)}"
        )
    for row in comparisons:
        if (
            int(row.get("examples", -1)) != 734
            or int(row.get("tasks", -1)) != 13
            or int(row.get("bootstrap_samples", -1)) != BOOTSTRAP_SAMPLES
            or int(row.get("bootstrap_seed", -1)) != ROOT_SEED
        ):
            failures.append(
                f"{row.get('comparison_id')}: paired-analysis protocol differs"
            )

    storage_by_allocation = {
        str(row.get("allocation")): dict(row) for row in storage_rows
    }
    if set(storage_by_allocation) != set(protocol.ALLOCATION_ORDER):
        failures.append("storage summary does not contain four allocations")
    required_storage_fields = (
        "key_index_bpe",
        "key_effective_bpe",
        "value_index_bpe",
        "value_effective_bpe",
        "kv_average_index_bpe",
        "kv_average_effective_bpe",
        "theoretical_kv_bytes_per_token",
        "static_partition_bytes_bool_masks",
        "static_rotation_bytes_fp32",
    )
    for allocation, row in storage_by_allocation.items():
        for field in required_storage_fields:
            value = _safe_float(row.get(field))
            if not math.isfinite(value) or value < 0:
                failures.append(f"{allocation}: invalid storage field {field}")

    cluster_distribution: dict[str, dict[str, Any]] = {}
    for allocation in protocol.ALLOCATION_ORDER:
        cluster_distribution[allocation] = {}
        for component in ("key", "value"):
            values = [
                int(row["outlier_channels"])
                for row in cluster_rows
                if row.get("allocation") == allocation
                and row.get("component") == component
            ]
            if len(values) != 32 * 8:
                failures.append(
                    f"{allocation}/{component}: cluster rows={len(values)}, expected 256"
                )
                continue
            histogram = {
                str(key): value for key, value in sorted(Counter(values).items())
            }
            cluster_distribution[allocation][component] = {
                "rows": len(values),
                "min": min(values),
                "mean": statistics.fmean(values),
                "max": max(values),
                "histogram": histogram,
            }
            if allocation == "fixed32" and set(values) != {32}:
                failures.append(f"fixed32/{component}: outlier counts are not all 32")
            if allocation.startswith("uniform") and set(values) != {0}:
                failures.append(f"{allocation}/{component}: uniform outlier counts are nonzero")
            if allocation == "kmeans2" and (
                min(values) <= 0 or max(values) >= 128
            ):
                failures.append(f"kmeans2/{component}: an empty cluster is present")

    system_rows = base.read_csv(args.output_dir / "system_metrics.csv")
    if len(system_rows) != 8_808:
        failures.append(f"system metrics rows={len(system_rows)}, expected 8808")

    required = (
        "study_config.json",
        "subset_manifest.json",
        "token_manifest.json",
        "tokens.pt",
        "activation_manifest.json",
        "activations/calibration.pt",
        "activations/validation.pt",
        "channel_statistics.json",
        "fixed32_partitions.pt",
        "kmeans2_partitions.pt",
        "partition_manifest.json",
        "random_rotation_manifest.json",
        "learned_rotation_manifest.json",
        "codebook_manifest.json",
        "training_curves.csv",
        "offline_diagnostics.csv",
        "predictions.jsonl",
        "example_scores.csv",
        "task_summary.csv",
        "category_summary.csv",
        "length_summary.csv",
        "condition_summary.csv",
        "storage_summary.csv",
        "cluster_size_summary.csv",
        "paired_comparisons.csv",
        "bootstrap_summary.csv",
        "paired_example_scores.csv",
        "system_metrics.csv",
        "completion_audit.json",
        "report.md",
        "plots/condition_scores.png",
        "plots/learned_minus_random.png",
        "plots/score_bpe_pareto.png",
    )
    # completion_audit.json is atomically written at the end of this function.
    missing = [
        name
        for name in required
        if name != "completion_audit.json" and not (args.output_dir / name).is_file()
    ]
    for directory in ("checkpoints", "plots"):
        if not (args.output_dir / directory).is_dir():
            missing.append(directory + "/")
    for allocation in protocol.ALLOCATION_ORDER:
        for suffix in ("latest.pt", "step10000.pt"):
            checkpoint = args.output_dir / "checkpoints" / f"{allocation}_{suffix}"
            if not checkpoint.is_file():
                missing.append(str(checkpoint.relative_to(args.output_dir)))
    failures.extend(f"missing required artifact: {name}" for name in missing)
    audit = {
        "status": "complete" if not failures else "incomplete",
        "expected_predictions": 8_808,
        "prediction_count": len(predictions),
        "unique_prediction_count": len(set(prediction_keys)),
        "score_count": len(scores),
        "empty_outputs": empty_count,
        "error_fields": error_count,
        "nonfinite_scores": nonfinite_scores,
        "prompt_truncations": truncation_count,
        "condition_coverage": coverage,
        "shared_subset_manifest_sha256": expected_subset_hash,
        "condition_subset_hashes": subset_hashes,
        "condition_partition_hashes": partition_hashes,
        "random_learned_value_hashes": value_hashes,
        "step10000_artifacts": artifact_audits,
        "storage": storage_by_allocation,
        "cluster_distribution": cluster_distribution,
        "paired_comparisons": len(comparisons),
        "system_metrics": len(system_rows),
        "required_artifacts": list(required),
        "missing_artifacts": missing,
        "failures": failures,
    }
    write_json(args.output_dir / "completion_audit.json", audit)
    return audit


def report_stage(args: argparse.Namespace) -> None:
    static_gates(args)
    assert_all_smoke_passed(args)
    predictions, scores = _combined_rows(args)
    aggregations = protocol.aggregate_scores(scores)
    paired_analysis = protocol.analyze_paired_comparisons(
        scores,
        bootstrap_samples=args.bootstrap_samples,
        bootstrap_seed=ROOT_SEED,
    )
    comparisons = list(paired_analysis.summary_rows)
    paired_rows = list(paired_analysis.example_rows)
    bootstrap_rows = [
        {
            "comparison_id": row["comparison_id"],
            "allocation": row["allocation"],
            "contrast": row["contrast"],
            "difference": row["difference"],
            "confidence_interval_low": row["confidence_interval_low"],
            "confidence_interval_high": row["confidence_interval_high"],
            "bootstrap_samples": row["bootstrap_samples"],
            "bootstrap_seed": row["bootstrap_seed"],
        }
        for row in comparisons
    ]
    storage_rows, cluster_rows = _storage_and_cluster_rows(args)

    base.write_jsonl(args.output_dir / "predictions.jsonl", predictions)
    write_csv(args.output_dir / "example_scores.csv", scores)
    write_csv(args.output_dir / "task_summary.csv", aggregations.task_rows)
    write_csv(args.output_dir / "category_summary.csv", aggregations.category_rows)
    write_csv(args.output_dir / "length_summary.csv", aggregations.length_rows)
    write_csv(args.output_dir / "condition_summary.csv", aggregations.overall_rows)
    write_csv(args.output_dir / "paired_comparisons.csv", comparisons)
    write_csv(args.output_dir / "bootstrap_summary.csv", bootstrap_rows)
    write_csv(args.output_dir / "paired_example_scores.csv", paired_rows)
    write_csv(args.output_dir / "storage_summary.csv", storage_rows)
    write_csv(args.output_dir / "cluster_size_summary.csv", cluster_rows)

    system_rows: list[dict[str, Any]] = []
    for condition in protocol.all_conditions():
        path = args.output_dir / "full" / condition.condition_id / "system_metrics.csv"
        system_rows.extend(
            {**row, "allocation": condition.allocation}
            for row in base.read_csv(path)
        )
    write_csv(args.output_dir / "system_metrics.csv", system_rows)
    _write_plots(
        args.output_dir, aggregations.overall_rows, comparisons, storage_rows
    )

    learned_random = {
        row["allocation"]: row
        for row in comparisons
        if row["contrast"] == "learned_minus_random"
    }
    random_identity = {
        row["allocation"]: row
        for row in comparisons
        if row["contrast"] == "random_minus_identity"
    }
    prompt_truncations = sum(
        bool(row.get("prompt_truncated")) for row in predictions
    )
    report_lines = [
        "# LongBench-E channel allocation x rotation",
        "",
        "- Scope: 20% task x length-bucket-stratified pilot (734 examples/condition).",
        "- Conditions: 12 quantized K/V conditions; BF16 is not a primary condition.",
        "- Learned means Learned Key / fixed Random Value.",
        "- Cache path: reconstructed-BF16 quality emulation; storage is theoretical packed accounting.",
        "- Bootstrap: 10,000 task-stratified paired example resamples, root seed 35.",
        f"- Prompt truncations: {prompt_truncations} of {len(predictions)} predictions.",
        "",
        "## Condition scores",
        "",
        "| Condition | Task macro | Equal-category macro | K/V effective BPE |",
        "|---|---:|---:|---:|",
    ]
    storage_by_allocation = {row["allocation"]: row for row in storage_rows}
    for row in aggregations.overall_rows:
        storage = storage_by_allocation[row["allocation"]]
        report_lines.append(
            f"| {row['condition_id']} | {float(row['task_macro_average']):.6f} | "
            f"{float(row['equal_category_macro_average']):.6f} | "
            f"{float(storage['kv_average_effective_bpe']):.6f} |"
        )
    report_lines.extend(
        [
            "",
            "## Storage accounting",
            "",
            "| Allocation | K index BPE | K effective BPE | V index BPE | V effective BPE | K/V effective BPE | Theoretical K/V bytes/token |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for allocation in protocol.ALLOCATION_ORDER:
        row = storage_by_allocation[allocation]
        report_lines.append(
            f"| {allocation} | {float(row['key_index_bpe']):.6f} | "
            f"{float(row['key_effective_bpe']):.6f} | "
            f"{float(row['value_index_bpe']):.6f} | "
            f"{float(row['value_effective_bpe']):.6f} | "
            f"{float(row['kv_average_effective_bpe']):.6f} | "
            f"{int(row['theoretical_kv_bytes_per_token'])} |"
        )
    report_lines.extend(
        [
            "",
            "Index payloads, FP16 group norms, per-group byte packing, and alignment are separated in `storage_summary.csv`; static partition and rotation bytes are reported there as well.",
            "",
            "## All paired contrasts",
            "",
            "| Allocation | Contrast | Difference | 95% CI | Win / tie / loss |",
            "|---|---|---:|---:|---:|",
        ]
    )
    for row in comparisons:
        report_lines.append(
            f"| {row['allocation']} | {row['contrast']} | "
            f"{float(row['difference']):.6f} | "
            f"[{float(row['confidence_interval_low']):.6f}, "
            f"{float(row['confidence_interval_high']):.6f}] | "
            f"{int(row['wins'])} / {int(row['ties'])} / {int(row['losses'])} |"
        )
    report_lines.extend(
        [
            "",
            "## Random rotation versus Identity",
            "",
            "| Allocation | Difference | 95% CI | Verdict |",
            "|---|---:|---:|---|",
        ]
    )
    for allocation in protocol.ALLOCATION_ORDER:
        row = random_identity[allocation]
        difference = float(row["difference"])
        report_lines.append(
            f"| {allocation} | {difference:.6f} | "
            f"[{float(row['confidence_interval_low']):.6f}, "
            f"{float(row['confidence_interval_high']):.6f}] | "
            f"{'Supported' if difference > 0 else 'Not supported'} |"
        )
    report_lines.extend(
        [
            "",
            "## Learned Key versus Random Key",
            "",
            "| Allocation | Difference | 95% CI | Verdict |",
            "|---|---:|---:|---|",
        ]
    )
    for allocation in protocol.ALLOCATION_ORDER:
        row = learned_random[allocation]
        lower = float(row["confidence_interval_low"])
        difference = float(row["difference"])
        verdict = (
            "Supported" if difference > 0 and lower > 0 else
            "Promising but inconclusive" if difference > 0 else
            "Not supported"
        )
        report_lines.append(
            f"| {allocation} | {difference:.6f} | "
            f"[{float(row['confidence_interval_low']):.6f}, "
            f"{float(row['confidence_interval_high']):.6f}] | {verdict} |"
        )
    condition_scores = {
        (str(row["allocation"]), str(row["method"])): float(
            row["task_macro_average"]
        )
        for row in aggregations.overall_rows
    }
    kmeans_index_bpe = float(
        storage_by_allocation["kmeans2"]["kv_average_index_bpe"]
    )
    report_lines.extend(
        [
            "",
            "## Descriptive allocation comparisons",
            "",
            "These cross-allocation differences use the shared examples but are descriptive (no additional bootstrap family is introduced).",
            "",
            "| Method | Fixed32 - Uniform2 | KMeans2 - Fixed32 | KMeans2 interpretation |",
            "|---|---:|---:|---|",
        ]
    )
    for method in ("identity", "random", "learned"):
        fixed_difference = (
            condition_scores[("fixed32", method)]
            - condition_scores[("uniform2", method)]
        )
        adaptive_difference = (
            condition_scores[("kmeans2", method)]
            - condition_scores[("fixed32", method)]
        )
        if adaptive_difference > 0 and kmeans_index_bpe <= 2.5:
            adaptive_verdict = "Positive evidence at <=2.5 index BPE"
        elif adaptive_difference > 0:
            adaptive_verdict = "Pareto result only; index BPE >2.5"
        else:
            adaptive_verdict = "Not supported"
        report_lines.append(
            f"| {method} | {fixed_difference:.6f} | "
            f"{adaptive_difference:.6f} | {adaptive_verdict} |"
        )
    kmeans_distribution: dict[str, list[int]] = {
        component: [
            int(row["outlier_channels"])
            for row in cluster_rows
            if row["allocation"] == "kmeans2" and row["component"] == component
        ]
        for component in ("key", "value")
    }
    report_lines.extend(
        [
            "",
            "## Adaptive 2-means cluster sizes",
            "",
            "| Component | Heads | Min | Mean | Max | Histogram |",
            "|---|---:|---:|---:|---:|---|",
        ]
    )
    for component in ("key", "value"):
        values = kmeans_distribution[component]
        histogram = ", ".join(
            f"{count}:{frequency}"
            for count, frequency in sorted(Counter(values).items())
        )
        report_lines.append(
            f"| {component} | {len(values)} | {min(values)} | "
            f"{statistics.fmean(values):.6f} | {max(values)} | {histogram} |"
        )
    report_lines.extend(
        [
            "",
            "## Interpretation guardrails",
            "",
            "Fixed32 is a 2.5-bit index payload plus two FP16 norms and packing overhead.",
            "Adaptive 2-means is reported at its measured K/V BPE and is not labeled 2.5-bit.",
            "This one-seed 20% pilot does not establish seed generalization or a full LongBench-E score.",
            "Offline Key/Value reconstruction, attention-logit, probability-KL, and attention-output diagnostics are in `offline_diagnostics.csv`; downstream LongBench-E score remains the primary criterion.",
            "",
            "![Condition scores](plots/condition_scores.png)",
            "",
            "![Learned minus Random](plots/learned_minus_random.png)",
            "",
            "![Score-BPE Pareto](plots/score_bpe_pareto.png)",
        ]
    )
    (args.output_dir / "report.md").write_text("\n".join(report_lines) + "\n")
    audit = completion_audit(
        args,
        predictions,
        scores,
        comparisons,
        storage_rows,
        cluster_rows,
    )
    if audit["status"] != "complete":
        raise RuntimeError("completion audit failed: " + "; ".join(audit["failures"]))
    assets = validate_assets(args, deep=False)
    write_study_config(args, assets, "complete", completed_at=utc_now(), audit=audit)
    print(f"[{utc_now()}] report and completion audit complete", flush=True)


def _seed_process() -> None:
    # Artifact RNGs never use this global state.  These defaults only keep
    # model-side incidental behavior deterministic under greedy decoding.
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    torch.manual_seed(ROOT_SEED)
    np.random.seed(ROOT_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(ROOT_SEED)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.set_float32_matmul_precision("highest")


def main(argv: Sequence[str] | None = None) -> None:
    args = _resolve_args(parse_args(argv))
    _seed_process()
    install_base_patches(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.stage == "validate":
        assets = validate_assets(args, deep=False)
        manifest = protocol.ensure_subset_manifest(
            args.output_dir,
            args.data_dir,
            dataset_revision=args.dataset_revision,
            data_hashes=assets["longbench"]["data_hashes"],
        )
        write_study_config(args, assets, "validated")
        print(
            json.dumps(
                {
                    "model_revision": MODEL_REVISION,
                    "longbench_commit": LONG_BENCH_COMMIT,
                    "dataset_revision": LONG_BENCH_DATASET_REVISION,
                    "examples_per_condition": manifest["example_count"],
                    "conditions": len(protocol.all_conditions()),
                    "predictions": manifest["example_count"]
                    * len(protocol.all_conditions()),
                },
                indent=2,
            ),
            flush=True,
        )
    elif args.stage == "capture":
        capture_stage(args)
    elif args.stage == "artifacts":
        artifacts_stage(args)
    elif args.stage == "diagnostics":
        diagnostics_stage(args)
    elif args.stage == "gates":
        print(json.dumps(static_gates(args), indent=2), flush=True)
    elif args.stage == "smoke":
        static_gates(args)
        condition = protocol.condition_by_id(args.condition)
        recover_prediction_tail(
            args.output_dir / "smoke" / condition.condition_id
        )
        base.run_inference(args, "smoke")
        repair_smoke_codec_counters(args, condition)
    elif args.stage == "full":
        static_gates(args)
        assert_all_smoke_passed(args)
        condition = protocol.condition_by_id(args.condition)
        recover_prediction_tail(
            args.output_dir / "full" / condition.condition_id
        )
        base.run_inference(args, "full")
    elif args.stage == "report":
        report_stage(args)
    elif args.stage == "orchestrate":
        orchestrate(args)
    else:
        raise AssertionError(args.stage)


if __name__ == "__main__":
    main()
