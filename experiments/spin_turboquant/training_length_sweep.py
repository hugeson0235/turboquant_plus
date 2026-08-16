"""Run the independent learned-rotation training-length sweep.

This module implements ``../learned_rotation_training_length_sweep_plan.md``.
The sweep is intentionally separate from :mod:`experiments.spin_turboquant.run`
because every configured horizon owns a fresh Adam optimizer and its own cosine
schedule.  Completed condition directories are immutable resume units; the
large aggregate CSVs are rebuilt from those units after the sweep completes.
"""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import math
import os
import platform
import shutil
import subprocess
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from transformers import AutoTokenizer

from .core import (
    attention_distortion_metrics,
    build_random_rotations,
    cayley_rotation,
    codebook_tensor,
    quantize_to_centroids,
)
from .run import (
    capture_activations,
    check_configuration,
    cleanup_cuda,
    load_model,
    tinystories_rows,
    token_stream,
    wikitext_rows,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
SPECIFICATION_PATH = REPOSITORY_ROOT.parent / "learned_rotation_training_length_sweep_plan.md"
DEFAULT_REFERENCE_DIR = Path(__file__).resolve().parent / "results" / "instruct"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "results" / "training_length_sweep"
DEFAULT_LONGBENCH_REPO = REPOSITORY_ROOT.parent / "LongBench_official"
DEFAULT_LONGBENCH_DATA = (
    REPOSITORY_ROOT.parent
    / "LongBench_data"
    / "5e628be450b7e67fb7ae6e201bd6d8f7056f7672"
    / "data"
)
DEFAULT_REFERENCE_LONGBENCH = Path(__file__).resolve().parent / "results" / "longbench_subset"

BITS = (2, 3, 4)
SEEDS = (17, 29, 43)
HORIZONS = tuple(range(100, 1001, 100))
METRIC_INTERVAL = 10
EMA_COEFFICIENT = 0.95
LEARNING_RATE = 0.005
MINIMUM_LEARNING_RATE = 0.00025
BATCH_TOKENS = 256
CALIBRATION_TOKENS = 4096
VALIDATION_TOKENS = 4096
DOMAIN_TOKENS = 4096
ATTENTION_SEQUENCE_LENGTH = 1024
REFERENCE_STEPS = 80
SELECTION_TOLERANCE = 0.001
ADOPTION_THRESHOLD = 0.005
DOMAIN_REGRESSION_THRESHOLD = 0.002

TRAINING_CURVE_FIELDS = (
    "bit_width",
    "seed",
    "horizon_steps",
    "step",
    "minibatch_mse",
    "last_10_step_mean",
    "ema_mse",
    "learning_rate",
    "gradient_norm",
    "optimizer_elapsed_seconds",
    "evaluation_elapsed_seconds",
)
CHECKPOINT_FIELDS = (
    "bit_width",
    "seed",
    "horizon_steps",
    "step",
    "calibration_mse",
    "wikitext_validation_mse",
    "original_scale_validation_mse",
    "relative_improvement",
    "generalization_gap",
    "orthogonality_max_abs",
)
HEAD_CHECKPOINT_FIELDS = (
    "bit_width",
    "seed",
    "horizon_steps",
    "step",
    "dataset",
    "layer",
    "head",
    "normalized_key_mse",
    "original_scale_key_mse",
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
            "smoke",
            "sweep",
            "select",
            "sanity",
            "longbench",
            "report",
            "orchestrate",
        ),
        default="orchestrate",
    )
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--reference-dir", type=Path, default=DEFAULT_REFERENCE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--bits", nargs="+", type=int, default=list(BITS))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    parser.add_argument("--horizons", nargs="+", type=int, default=list(HORIZONS))
    parser.add_argument("--metric-interval", type=int, default=METRIC_INTERVAL)
    parser.add_argument("--calibration-tokens", type=int, default=CALIBRATION_TOKENS)
    parser.add_argument("--validation-tokens", type=int, default=VALIDATION_TOKENS)
    parser.add_argument("--domain-tokens", type=int, default=DOMAIN_TOKENS)
    parser.add_argument(
        "--attention-sequence-length", type=int, default=ATTENTION_SEQUENCE_LENGTH
    )
    parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE)
    parser.add_argument("--minimum-learning-rate", type=float, default=MINIMUM_LEARNING_RATE)
    parser.add_argument("--batch-tokens", type=int, default=BATCH_TOKENS)
    parser.add_argument("--ema-coefficient", type=float, default=EMA_COEFFICIENT)
    parser.add_argument("--longbench-repo", type=Path, default=DEFAULT_LONGBENCH_REPO)
    parser.add_argument("--longbench-data", type=Path, default=DEFAULT_LONGBENCH_DATA)
    parser.add_argument(
        "--reference-longbench-dir", type=Path, default=DEFAULT_REFERENCE_LONGBENCH
    )
    parser.add_argument("--skip-longbench", action="store_true")
    return parser.parse_args(argv)


def atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    )
    temporary.replace(path)


def read_json(path: Path, default: Any = None) -> Any:
    return json.loads(path.read_text()) if path.exists() else default


def write_csv(
    path: Path,
    rows: Iterable[dict[str, Any]],
    *,
    fieldnames: Sequence[str] | None = None,
) -> None:
    rows = list(rows)
    if fieldnames is None:
        fields: list[str] = []
        for row in rows:
            for field in row:
                if field not in fields:
                    fields.append(field)
        fieldnames = fields
    if not fieldnames:
        raise ValueError(f"cannot write schema-less CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def sha256_file(path: Path, *, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_json(payload: Any) -> str:
    encoded = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def tensor_sha256(tensor: torch.Tensor) -> str:
    value = tensor.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode())
    digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
    digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def atomic_torch_save(payload: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def git_output(*args: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(REPOSITORY_ROOT), *args], text=True
    ).strip()


def environment_metadata(device: torch.device) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "timestamp": utc_now(),
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "numpy": np.__version__,
        "command": sys.argv,
        "git_commit": git_output("rev-parse", "HEAD"),
        "git_status": git_output("status", "--short").splitlines(),
    }
    if device.type == "cuda":
        properties = torch.cuda.get_device_properties(device)
        metadata["gpu"] = {
            "name": properties.name,
            "total_memory_bytes": properties.total_memory,
            "capability": list(properties.major_minor) if hasattr(properties, "major_minor") else [properties.major, properties.minor],
        }
    return metadata


def optimizer_seed(bit_width: int, seed: int) -> int:
    """Return the existing experiment's bit/seed-specific minibatch seed."""

    return seed + bit_width * 100_000


def run_id(bit_width: int, seed: int, horizon: int) -> str:
    return f"bit{bit_width}_seed{seed}_h{horizon}"


def run_dir(args: argparse.Namespace, bit_width: int, seed: int, horizon: int) -> Path:
    return args.output_dir / "runs" / run_id(bit_width, seed, horizon)


def reference_rotation_path(args: argparse.Namespace, bit_width: int, seed: int) -> Path:
    return args.reference_dir / "rotations" / f"bit{bit_width}_seed{seed}.pt"


def selected_rotation_path(args: argparse.Namespace, bit_width: int, seed: int) -> Path:
    return args.output_dir / "final_rotation_artifacts" / f"bit{bit_width}_seed{seed}.pt"


def fixed_protocol(args: argparse.Namespace, dimensions: dict[str, Any]) -> dict[str, Any]:
    return {
        "bits": list(args.bits),
        "seeds": list(args.seeds),
        "horizons": list(args.horizons),
        "metric_interval": args.metric_interval,
        "calibration_tokens": args.calibration_tokens,
        "validation_tokens": args.validation_tokens,
        "domain_tokens": args.domain_tokens,
        "attention_sequence_length": args.attention_sequence_length,
        "optimizer": "Adam",
        "learning_rate": args.learning_rate,
        "minimum_learning_rate": args.minimum_learning_rate,
        "scheduler": "CosineAnnealingLR",
        "scheduler_t_max": "configured horizon_steps",
        "gradient_clip": 1.0,
        "batch_tokens_per_head": args.batch_tokens,
        "ema_coefficient": args.ema_coefficient,
        "initial_cayley_parameter": "zero",
        "independent_horizon_restarts": True,
        "minibatch_seed_formula": "seed + bit_width * 100000",
        "norm_correction": False,
        "quantization_position": "pre-RoPE key",
        "value_cache": "BF16",
        "model": str(args.model.resolve()),
        "model_revision": args.model.name,
        "reference_dir": str(args.reference_dir.resolve()),
        "dimensions": dimensions,
        "selection": {
            "seed_aggregation": "mean",
            "near_minimum_relative_tolerance": SELECTION_TOLERANCE,
            "longer_training_adoption_threshold": ADOPTION_THRESHOLD,
            "domain_regression_threshold": DOMAIN_REGRESSION_THRESHOLD,
            "reference_steps": REFERENCE_STEPS,
        },
    }


def validate_exact_design(args: argparse.Namespace) -> None:
    expected = {
        "bits": list(BITS),
        "seeds": list(SEEDS),
        "horizons": list(HORIZONS),
        "metric_interval": METRIC_INTERVAL,
        "calibration_tokens": CALIBRATION_TOKENS,
        "validation_tokens": VALIDATION_TOKENS,
        "domain_tokens": DOMAIN_TOKENS,
        "learning_rate": LEARNING_RATE,
        "minimum_learning_rate": MINIMUM_LEARNING_RATE,
        "batch_tokens": BATCH_TOKENS,
        "ema_coefficient": EMA_COEFFICIENT,
    }
    actual = {name: getattr(args, name) for name in expected}
    if actual != expected:
        raise ValueError(
            "the checked-in plan fixes the full sweep design; mismatched arguments: "
            + json.dumps(
                {key: {"expected": expected[key], "actual": actual[key]} for key in expected if actual[key] != expected[key]},
                sort_keys=True,
            )
        )
    if not 1 <= args.attention_sequence_length <= args.domain_tokens:
        raise ValueError("attention sequence length must fit in the TinyStories capture")


def validate_reference(args: argparse.Namespace, dimensions: dict[str, Any]) -> dict[str, Any]:
    if not args.reference_dir.exists():
        raise FileNotFoundError(args.reference_dir)
    reference_config = read_json(args.reference_dir / "run_config.json", {})
    reference_arguments = reference_config.get("arguments", {})
    checks = {
        "model": str(Path(reference_arguments.get("model", "")).resolve())
        == str(args.model.resolve()),
        "calibration_tokens": reference_arguments.get("calibration_tokens")
        == args.calibration_tokens,
        "train_steps": reference_arguments.get("train_steps") == REFERENCE_STEPS,
        "train_batch_tokens": reference_arguments.get("train_batch_tokens")
        == args.batch_tokens,
        "learning_rate": reference_arguments.get("learning_rate") == args.learning_rate,
        "norm_correction": reference_arguments.get("norm_correction") is False,
        "bits": reference_arguments.get("bits") == list(args.bits),
        "seeds": reference_arguments.get("seeds") == list(args.seeds),
    }
    if not all(checks.values()):
        raise RuntimeError(f"step-80 reference protocol mismatch: {checks}")

    calibration_path = args.reference_dir / "activations" / "calibration.pt"
    if not calibration_path.exists():
        raise FileNotFoundError(calibration_path)
    calibration = torch.load(calibration_path, map_location="cpu", weights_only=True)["k"]
    expected_shape = (
        dimensions["num_hidden_layers"],
        dimensions["num_key_value_heads"],
        args.calibration_tokens,
        dimensions["head_dim"],
    )
    if tuple(calibration.shape) != expected_shape:
        raise RuntimeError(
            f"reference calibration has shape {tuple(calibration.shape)}, expected {expected_shape}"
        )
    del calibration

    rotation_hashes: dict[str, dict[str, str]] = {}
    for bit_width in args.bits:
        for seed in args.seeds:
            path = reference_rotation_path(args, bit_width, seed)
            if not path.exists():
                raise FileNotFoundError(path)
            artifact = torch.load(path, map_location="cpu", weights_only=True)
            expected_rotation_shape = expected_shape[:2] + (dimensions["head_dim"],) * 2
            for key in ("random", "learned"):
                value = artifact.get(key)
                if not isinstance(value, torch.Tensor) or tuple(value.shape) != expected_rotation_shape:
                    raise RuntimeError(f"{path}:{key} has an invalid shape")
            regenerated = build_random_rotations(
                dimensions["num_hidden_layers"],
                dimensions["num_key_value_heads"],
                dimensions["head_dim"],
                seed,
            )
            if not torch.equal(artifact["random"], regenerated):
                raise RuntimeError(f"{path}: random rotation does not match seed {seed}")
            stats = artifact.get("stats", {})
            if stats.get("steps") != REFERENCE_STEPS:
                raise RuntimeError(f"{path}: expected a step-80 artifact")
            rotation_hashes[run_id(bit_width, seed, REFERENCE_STEPS)] = {
                "artifact_file_sha256": sha256_file(path),
                "random_tensor_sha256": tensor_sha256(artifact["random"]),
                "learned_tensor_sha256": tensor_sha256(artifact["learned"]),
            }
            del artifact, regenerated

    return {
        "run_config_sha256": sha256_file(args.reference_dir / "run_config.json"),
        "calibration_file_sha256": sha256_file(calibration_path),
        "rotation_hashes": rotation_hashes,
    }


def codebook_hashes(args: argparse.Namespace, head_dim: int) -> dict[str, str]:
    return {
        str(bit_width): hashlib.sha256(
            codebook_tensor(bit_width, head_dim, device="cpu").numpy().tobytes()
        ).hexdigest()
        for bit_width in args.bits
    }


def validate_stage(args: argparse.Namespace) -> dict[str, Any]:
    validate_exact_design(args)
    if not SPECIFICATION_PATH.exists():
        raise FileNotFoundError(SPECIFICATION_PATH)
    if not args.model.exists():
        raise FileNotFoundError(args.model)
    if str(args.device).startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required by the requested sweep")
    dimensions = check_configuration(
        argparse.Namespace(
            model=args.model,
            device=args.device,
            bits=list(args.bits),
            seeds=list(args.seeds),
            attention_lengths=[args.attention_sequence_length],
            evaluation_tokens=args.validation_tokens,
            ppl_tokens=2,
            ppl_sequence_length=2,
            retrieval_cases_per_length=1,
        )
    )
    reference = validate_reference(args, dimensions)
    protocol = fixed_protocol(args, dimensions)
    protocol_hash = sha256_json(protocol)
    config_path = args.output_dir / "training_length_sweep_config.json"
    existing = read_json(config_path, {})
    if existing and existing.get("protocol_sha256") != protocol_hash:
        raise RuntimeError(
            "output directory belongs to a different sweep protocol: "
            f"{config_path}"
        )

    tokenizer_files = [
        args.model / name
        for name in ("config.json", "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json")
        if (args.model / name).exists()
    ]
    implementation_files = {
        "sweep_runner": Path(__file__).resolve(),
        "spin_turboquant_core": Path(__file__).resolve().with_name("core.py"),
        "base_runner": Path(__file__).resolve().with_name("run.py"),
        "turboquant_codebook": REPOSITORY_ROOT / "turboquant" / "codebook.py",
        "turboquant_rotation": REPOSITORY_ROOT / "turboquant" / "rotation.py",
    }
    device = torch.device(args.device)
    payload = {
        "created_at": existing.get("created_at", utc_now()),
        "updated_at": utc_now(),
        "status": existing.get("status", "validated"),
        "protocol": protocol,
        "protocol_sha256": protocol_hash,
        "specification": {
            "path": str(SPECIFICATION_PATH),
            "sha256": sha256_file(SPECIFICATION_PATH),
        },
        "model": {
            "path": str(args.model.resolve()),
            "revision": args.model.name,
            "tokenizer_and_config_hashes": {
                path.name: sha256_file(path) for path in tokenizer_files
            },
        },
        "reference_step_80": reference,
        "codebook_hashes": codebook_hashes(args, dimensions["head_dim"]),
        "implementation_hashes": {
            name: sha256_file(path) for name, path in implementation_files.items()
        },
        "environment": environment_metadata(device),
        "progress": existing.get("progress", {}),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_json(config_path, payload)
    print(f"[{utc_now()}] validated fixed sweep protocol {protocol_hash}", flush=True)
    return payload


def update_config(args: argparse.Namespace, **updates: Any) -> None:
    path = args.output_dir / "training_length_sweep_config.json"
    payload = read_json(path, {})
    payload.update(updates)
    payload["updated_at"] = utc_now()
    atomic_write_json(path, payload)


def capture_stage(args: argparse.Namespace) -> None:
    validate_stage(args)
    activation_dir = args.output_dir / "activations"
    activation_dir.mkdir(parents=True, exist_ok=True)
    token_path = args.output_dir / "heldout_tokens.pt"
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    if token_path.exists():
        tokens = torch.load(token_path, map_location="cpu", weights_only=True)
    else:
        tokens = {
            "wikitext": token_stream(
                tokenizer, wikitext_rows("validation"), args.validation_tokens
            ),
            "tinystories": token_stream(
                tokenizer, tinystories_rows(), args.domain_tokens
            ),
        }
        atomic_torch_save(tokens, token_path)

    reference_tokens = torch.load(
        args.reference_dir / "tokens.pt", map_location="cpu", weights_only=True
    )
    for name in ("wikitext", "tinystories"):
        prefix = reference_tokens[name]
        if not torch.equal(tokens[name][: prefix.numel()], prefix):
            raise RuntimeError(f"new {name} token stream does not preserve the step-80 prefix")
    del reference_tokens

    expected = {
        "wikitext": activation_dir / "wikitext.pt",
        "tinystories": activation_dir / "tinystories.pt",
    }
    if not all(path.exists() for path in expected.values()):
        device = torch.device(args.device)
        model = load_model(args.model, device)
        if not expected["wikitext"].exists():
            print(f"[{utc_now()}] capturing 4096-token WikiText validation keys", flush=True)
            payload = capture_activations(
                model,
                tokens["wikitext"],
                capture_queries_and_values=False,
                device=device,
            )
            atomic_torch_save(payload, expected["wikitext"])
            del payload
            cleanup_cuda()
        if not expected["tinystories"].exists():
            print(f"[{utc_now()}] capturing 4096-token TinyStories keys and attention prefix", flush=True)
            payload = capture_activations(
                model,
                tokens["tinystories"],
                capture_queries_and_values=True,
                device=device,
            )
            keep = args.attention_sequence_length
            payload["q"] = payload["q"][:, :keep].contiguous()
            payload["v"] = payload["v"][:, :keep].contiguous()
            payload["cos"] = payload["cos"][:keep].contiguous()
            payload["sin"] = payload["sin"][:keep].contiguous()
            atomic_torch_save(payload, expected["tinystories"])
            del payload
            cleanup_cuda()
        del model
        cleanup_cuda()

    dimensions = read_json(
        args.output_dir / "training_length_sweep_config.json", {}
    )["protocol"]["dimensions"]
    manifests: dict[str, Any] = {}
    for name, path in expected.items():
        payload = torch.load(path, map_location="cpu", weights_only=True)
        shapes = {key: list(value.shape) for key, value in payload.items()}
        if name == "wikitext":
            expected_k = [
                dimensions["num_hidden_layers"],
                dimensions["num_key_value_heads"],
                args.validation_tokens,
                dimensions["head_dim"],
            ]
        else:
            expected_k = [
                dimensions["num_hidden_layers"],
                args.domain_tokens,
                dimensions["num_key_value_heads"],
                dimensions["head_dim"],
            ]
        if shapes.get("k") != expected_k:
            raise RuntimeError(f"{path} has unexpected key shape {shapes.get('k')}")
        manifests[name] = {
            "path": str(path),
            "file_sha256": sha256_file(path),
            "shapes": shapes,
            "token_tensor_sha256": tensor_sha256(tokens[name]),
        }
        atomic_write_json(path.with_suffix(".json"), manifests[name])
        del payload
    config = read_json(args.output_dir / "training_length_sweep_config.json", {})
    config["activations"] = {
        "calibration": {
            "path": str(args.reference_dir / "activations" / "calibration.pt"),
            "file_sha256": config["reference_step_80"]["calibration_file_sha256"],
            "shared_with_step_80": True,
        },
        **manifests,
        "heldout_tokens_file_sha256": sha256_file(token_path),
    }
    config["status"] = "captured"
    config["updated_at"] = utc_now()
    atomic_write_json(args.output_dir / "training_length_sweep_config.json", config)
    print(f"[{utc_now()}] held-out capture stage complete", flush=True)


def load_key_tensors(args: argparse.Namespace) -> tuple[torch.Tensor, torch.Tensor]:
    calibration = torch.load(
        args.reference_dir / "activations" / "calibration.pt",
        map_location="cpu",
        weights_only=True,
    )["k"]
    wikitext = torch.load(
        args.output_dir / "activations" / "wikitext.pt",
        map_location="cpu",
        weights_only=True,
    )["k"]
    return calibration, wikitext


def normalized_dataset(
    keys: torch.Tensor, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    layers, heads, tokens, head_dim = keys.shape
    flat = keys.reshape(layers * heads, tokens, head_dim).to(
        device=device, dtype=torch.float32
    )
    norms = torch.linalg.vector_norm(flat, dim=-1, keepdim=True)
    normalized = flat / norms.clamp_min(torch.finfo(torch.float32).eps)
    return normalized, norms.square()


@torch.inference_mode()
def evaluate_dataset(
    normalized: torch.Tensor,
    norm_squared: torch.Tensor,
    rotations: torch.Tensor,
    centroids: torch.Tensor,
    *,
    chunk_tokens: int = 256,
) -> dict[str, Any]:
    heads, tokens, head_dim = normalized.shape
    normalized_sse = torch.zeros(heads, dtype=torch.float64, device="cpu")
    original_sse = torch.zeros(heads, dtype=torch.float64, device="cpu")
    for start in range(0, tokens, chunk_tokens):
        stop = min(tokens, start + chunk_tokens)
        rotated = torch.matmul(
            normalized[:, start:stop], rotations.transpose(-1, -2)
        )
        targets = quantize_to_centroids(rotated, centroids)
        squared = (rotated - targets).square()
        normalized_sse += squared.sum(dim=(1, 2)).double().cpu()
        original_sse += (
            squared * norm_squared[:, start:stop]
        ).sum(dim=(1, 2)).double().cpu()
        del rotated, targets, squared
    denominator = max(tokens * head_dim, 1)
    per_head_normalized = normalized_sse / denominator
    per_head_original = original_sse / denominator
    return {
        "normalized_key_mse": float(per_head_normalized.mean()),
        "original_scale_key_mse": float(per_head_original.mean()),
        "per_head_normalized_key_mse": per_head_normalized.numpy(),
        "per_head_original_scale_key_mse": per_head_original.numpy(),
    }


def orthogonality_max_abs(rotations: torch.Tensor) -> float:
    value = rotations.double()
    identity = torch.eye(value.shape[-1], dtype=torch.float64, device=value.device)
    error = value.transpose(-1, -2) @ value - identity
    return float(error.abs().max())


def terminal_rotations(parameters: torch.Tensor, initial: torch.Tensor) -> torch.Tensor:
    with torch.inference_mode():
        return cayley_rotation(parameters.detach().double(), initial.double()).float()


def append_head_rows(
    rows: list[dict[str, Any]],
    *,
    bit_width: int,
    seed: int,
    horizon: int,
    step: int,
    dataset: str,
    metrics: dict[str, Any],
    layers: int,
    heads: int,
) -> None:
    normalized = metrics["per_head_normalized_key_mse"].reshape(layers, heads)
    original = metrics["per_head_original_scale_key_mse"].reshape(layers, heads)
    for layer in range(layers):
        for head in range(heads):
            rows.append(
                {
                    "bit_width": bit_width,
                    "seed": seed,
                    "horizon_steps": horizon,
                    "step": step,
                    "dataset": dataset,
                    "layer": layer,
                    "head": head,
                    "normalized_key_mse": float(normalized[layer, head]),
                    "original_scale_key_mse": float(original[layer, head]),
                }
            )


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def execute_training_run(
    args: argparse.Namespace,
    *,
    bit_width: int,
    seed: int,
    horizon: int,
    calibration_normalized: torch.Tensor,
    calibration_norm_squared: torch.Tensor,
    validation_normalized: torch.Tensor,
    validation_norm_squared: torch.Tensor,
    initial: torch.Tensor,
    centroids: torch.Tensor,
    record_curves: bool,
) -> dict[str, Any]:
    device = initial.device
    flat_heads, _, head_dim = calibration_normalized.shape
    layers = 32
    heads = flat_heads // layers
    parameters = torch.nn.Parameter(torch.zeros_like(initial))
    optimizer = torch.optim.Adam([parameters], lr=args.learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=horizon,
        eta_min=args.minimum_learning_rate,
    )
    generator = torch.Generator(device=device)
    generator.manual_seed(optimizer_seed(bit_width, seed))
    token_count = calibration_normalized.shape[1]

    training_rows: list[dict[str, Any]] = []
    checkpoint_rows: list[dict[str, Any]] = []
    head_rows: list[dict[str, Any]] = []
    losses: list[float] = []
    ema: float | None = None
    minibatch_prefix_digest = hashlib.sha256()
    optimizer_elapsed = 0.0
    evaluation_elapsed = 0.0
    initial_validation_mse: float | None = None

    def evaluate(step: int, rotations: torch.Tensor) -> tuple[dict[str, Any], dict[str, Any]]:
        nonlocal evaluation_elapsed, initial_validation_mse
        synchronize(device)
        started = time.perf_counter()
        calibration_metrics = evaluate_dataset(
            calibration_normalized,
            calibration_norm_squared,
            rotations,
            centroids,
        )
        validation_metrics = evaluate_dataset(
            validation_normalized,
            validation_norm_squared,
            rotations,
            centroids,
        )
        synchronize(device)
        evaluation_elapsed += time.perf_counter() - started
        if initial_validation_mse is None:
            initial_validation_mse = float(validation_metrics["normalized_key_mse"])
        checkpoint_rows.append(
            {
                "bit_width": bit_width,
                "seed": seed,
                "horizon_steps": horizon,
                "step": step,
                "calibration_mse": calibration_metrics["normalized_key_mse"],
                "wikitext_validation_mse": validation_metrics["normalized_key_mse"],
                "original_scale_validation_mse": validation_metrics[
                    "original_scale_key_mse"
                ],
                "relative_improvement": (
                    initial_validation_mse - validation_metrics["normalized_key_mse"]
                )
                / max(initial_validation_mse, 1e-30),
                "generalization_gap": validation_metrics["normalized_key_mse"]
                - calibration_metrics["normalized_key_mse"],
                "orthogonality_max_abs": orthogonality_max_abs(rotations),
            }
        )
        if record_curves:
            append_head_rows(
                head_rows,
                bit_width=bit_width,
                seed=seed,
                horizon=horizon,
                step=step,
                dataset="calibration",
                metrics=calibration_metrics,
                layers=layers,
                heads=heads,
            )
            append_head_rows(
                head_rows,
                bit_width=bit_width,
                seed=seed,
                horizon=horizon,
                step=step,
                dataset="wikitext_validation",
                metrics=validation_metrics,
                layers=layers,
                heads=heads,
            )
        return calibration_metrics, validation_metrics

    initial_metrics, initial_validation = evaluate(0, initial)
    del initial_metrics, initial_validation
    training_rows.append(
        {
            "bit_width": bit_width,
            "seed": seed,
            "horizon_steps": horizon,
            "step": 0,
            "minibatch_mse": "",
            "last_10_step_mean": "",
            "ema_mse": "",
            "learning_rate": args.learning_rate,
            "gradient_norm": "",
            "optimizer_elapsed_seconds": optimizer_elapsed,
            "evaluation_elapsed_seconds": evaluation_elapsed,
        }
    )

    latest_gradient_norm = float("nan")
    for step in range(1, horizon + 1):
        indices = torch.randint(
            token_count,
            (args.batch_tokens,),
            generator=generator,
            device=device,
        )
        if step <= 100:
            minibatch_prefix_digest.update(
                indices.detach().cpu().contiguous().numpy().tobytes()
            )
        batch = calibration_normalized[:, indices]
        synchronize(device)
        started = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        rotations = cayley_rotation(parameters, initial)
        rotated = torch.matmul(batch, rotations.transpose(-1, -2))
        targets = quantize_to_centroids(rotated.detach(), centroids)
        loss = torch.mean((rotated - targets) ** 2)
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_([parameters], max_norm=1.0)
        optimizer.step()
        scheduler.step()
        synchronize(device)
        optimizer_elapsed += time.perf_counter() - started
        loss_value = float(loss.detach())
        latest_gradient_norm = float(gradient_norm)
        losses.append(loss_value)
        ema = loss_value if ema is None else args.ema_coefficient * ema + (1.0 - args.ema_coefficient) * loss_value
        del batch, rotations, rotated, targets, loss, gradient_norm

        if step % args.metric_interval == 0 or step == horizon:
            measured_rotations = (
                terminal_rotations(parameters, initial)
                if step == horizon
                else cayley_rotation(parameters.detach(), initial)
            )
            evaluate(step, measured_rotations)
            training_rows.append(
                {
                    "bit_width": bit_width,
                    "seed": seed,
                    "horizon_steps": horizon,
                    "step": step,
                    "minibatch_mse": loss_value,
                    "last_10_step_mean": float(np.mean(losses[-args.metric_interval :])),
                    "ema_mse": ema,
                    "learning_rate": optimizer.param_groups[0]["lr"],
                    "gradient_norm": latest_gradient_norm,
                    "optimizer_elapsed_seconds": optimizer_elapsed,
                    "evaluation_elapsed_seconds": evaluation_elapsed,
                }
            )
            if record_curves:
                print(
                    f"  {run_id(bit_width, seed, horizon)} step {step:4d}/{horizon}: "
                    f"val={checkpoint_rows[-1]['wikitext_validation_mse']:.9g} "
                    f"cal={checkpoint_rows[-1]['calibration_mse']:.9g} "
                    f"lr={optimizer.param_groups[0]['lr']:.6g}",
                    flush=True,
                )
            if step == horizon:
                final_rotations = measured_rotations
            else:
                del measured_rotations

    terminal = checkpoint_rows[-1]
    result = {
        "training_rows": training_rows,
        "checkpoint_rows": checkpoint_rows,
        "head_rows": head_rows,
        "terminal": terminal,
        "rotations": final_rotations,
        "rotation_tensor_sha256": tensor_sha256(final_rotations),
        "first_100_minibatch_indices_sha256": minibatch_prefix_digest.hexdigest(),
        "optimizer_elapsed_seconds": optimizer_elapsed,
        "evaluation_elapsed_seconds": evaluation_elapsed,
    }
    del parameters, optimizer, scheduler
    return result


def completed_run_metadata(
    args: argparse.Namespace, bit_width: int, seed: int, horizon: int
) -> dict[str, Any] | None:
    directory = run_dir(args, bit_width, seed, horizon)
    metadata = read_json(directory / "run_config.json", {})
    required = (
        directory / "training_curve.csv",
        directory / "checkpoint_metrics.csv",
        directory / "head_checkpoint_metrics.csv",
    )
    protocol_hash = read_json(
        args.output_dir / "training_length_sweep_config.json", {}
    ).get("protocol_sha256")
    if (
        metadata.get("status") == "complete"
        and metadata.get("protocol_sha256") == protocol_hash
        and metadata.get("first_100_minibatch_indices_sha256")
        and all(path.exists() for path in required)
    ):
        return metadata
    return None


def persist_completed_run(
    args: argparse.Namespace,
    bit_width: int,
    seed: int,
    horizon: int,
    result: dict[str, Any],
) -> dict[str, Any]:
    directory = run_dir(args, bit_width, seed, horizon)
    directory.mkdir(parents=True, exist_ok=True)
    write_csv(
        directory / "training_curve.csv",
        result["training_rows"],
        fieldnames=TRAINING_CURVE_FIELDS,
    )
    write_csv(
        directory / "checkpoint_metrics.csv",
        result["checkpoint_rows"],
        fieldnames=CHECKPOINT_FIELDS,
    )
    write_csv(
        directory / "head_checkpoint_metrics.csv",
        result["head_rows"],
        fieldnames=HEAD_CHECKPOINT_FIELDS,
    )
    terminal = dict(result["terminal"])
    terminal["rotation_tensor_sha256"] = result["rotation_tensor_sha256"]
    terminal["optimizer_elapsed_seconds"] = result["optimizer_elapsed_seconds"]
    terminal["evaluation_elapsed_seconds"] = result["evaluation_elapsed_seconds"]
    atomic_write_json(directory / "terminal_metrics.json", terminal)
    metadata = {
        "status": "complete",
        "completed_at": utc_now(),
        "protocol_sha256": read_json(
            args.output_dir / "training_length_sweep_config.json", {}
        )["protocol_sha256"],
        "bit_width": bit_width,
        "seed": seed,
        "horizon_steps": horizon,
        "optimizer_seed": optimizer_seed(bit_width, seed),
        "first_100_minibatch_indices_sha256": result[
            "first_100_minibatch_indices_sha256"
        ],
        "scheduler": {
            "name": "CosineAnnealingLR",
            "T_max": horizon,
            "initial_learning_rate": args.learning_rate,
            "minimum_learning_rate": args.minimum_learning_rate,
        },
        "row_counts": {
            "training_curve": len(result["training_rows"]),
            "checkpoint_metrics": len(result["checkpoint_rows"]),
            "head_checkpoint_metrics": len(result["head_rows"]),
        },
        "terminal": terminal,
    }
    atomic_write_json(directory / "run_config.json", metadata)
    return metadata


def maybe_update_best_candidate(
    args: argparse.Namespace,
    bit_width: int,
    seed: int,
    horizon: int,
    result: dict[str, Any],
) -> None:
    path = args.output_dir / ".best_candidates" / f"bit{bit_width}_seed{seed}.pt"
    existing = torch.load(path, map_location="cpu", weights_only=True) if path.exists() else None
    validation_mse = float(result["terminal"]["wikitext_validation_mse"])
    if existing is None or validation_mse < float(existing["wikitext_validation_mse"]):
        atomic_torch_save(
            {
                "bit_width": bit_width,
                "seed": seed,
                "horizon_steps": horizon,
                "wikitext_validation_mse": validation_mse,
                "rotation_tensor_sha256": result["rotation_tensor_sha256"],
                "learned": result["rotations"].reshape(32, 8, 128, 128).cpu(),
            },
            path,
        )
    del existing


def load_normalized_inputs(
    args: argparse.Namespace,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    calibration, validation = load_key_tensors(args)
    device = torch.device(args.device)
    calibration_normalized, calibration_norm_squared = normalized_dataset(calibration, device)
    validation_normalized, validation_norm_squared = normalized_dataset(validation, device)
    del calibration, validation
    return (
        calibration_normalized,
        calibration_norm_squared,
        validation_normalized,
        validation_norm_squared,
    )


def execute_one_condition(
    args: argparse.Namespace,
    *,
    bit_width: int,
    seed: int,
    horizon: int,
    inputs: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
) -> dict[str, Any]:
    existing = completed_run_metadata(args, bit_width, seed, horizon)
    if existing is not None:
        print(f"[{utc_now()}] resume: {run_id(bit_width, seed, horizon)} is complete", flush=True)
        return existing
    calibration_normalized, calibration_norm_squared, validation_normalized, validation_norm_squared = inputs
    device = torch.device(args.device)
    initial = build_random_rotations(32, 8, 128, seed).reshape(256, 128, 128).to(device)
    centroids = codebook_tensor(bit_width, 128, device=device)
    print(f"[{utc_now()}] starting independent {run_id(bit_width, seed, horizon)}", flush=True)
    result = execute_training_run(
        args,
        bit_width=bit_width,
        seed=seed,
        horizon=horizon,
        calibration_normalized=calibration_normalized,
        calibration_norm_squared=calibration_norm_squared,
        validation_normalized=validation_normalized,
        validation_norm_squared=validation_norm_squared,
        initial=initial,
        centroids=centroids,
        record_curves=True,
    )
    metadata = persist_completed_run(args, bit_width, seed, horizon, result)
    maybe_update_best_candidate(args, bit_width, seed, horizon, result)
    del result, initial, centroids
    cleanup_cuda()
    return metadata


def aggregate_sweep_csvs(args: argparse.Namespace, *, require_complete: bool) -> dict[str, int]:
    training_rows: list[dict[str, str]] = []
    checkpoint_rows: list[dict[str, str]] = []
    head_paths: list[Path] = []
    missing: list[str] = []
    for bit_width in args.bits:
        for seed in args.seeds:
            for horizon in args.horizons:
                directory = run_dir(args, bit_width, seed, horizon)
                if completed_run_metadata(args, bit_width, seed, horizon) is None:
                    missing.append(run_id(bit_width, seed, horizon))
                    continue
                training_rows.extend(read_csv(directory / "training_curve.csv"))
                checkpoint_rows.extend(read_csv(directory / "checkpoint_metrics.csv"))
                head_paths.append(directory / "head_checkpoint_metrics.csv")
    if require_complete and missing:
        raise RuntimeError(f"sweep has {len(missing)} incomplete runs: {missing[:10]}")
    write_csv(args.output_dir / "training_curve.csv", training_rows, fieldnames=TRAINING_CURVE_FIELDS)
    write_csv(args.output_dir / "checkpoint_metrics.csv", checkpoint_rows, fieldnames=CHECKPOINT_FIELDS)
    terminal_rows = [
        row
        for row in checkpoint_rows
        if int(row["step"]) == int(row["horizon_steps"])
    ]
    terminal_fields = list(CHECKPOINT_FIELDS) + [
        "rotation_tensor_sha256",
        "terminal_optimizer_elapsed_seconds",
        "terminal_evaluation_elapsed_seconds",
    ]
    for row in terminal_rows:
        metadata = read_json(
            run_dir(
                args,
                int(row["bit_width"]),
                int(row["seed"]),
                int(row["horizon_steps"]),
            )
            / "run_config.json",
            {},
        )
        row["rotation_tensor_sha256"] = metadata["terminal"]["rotation_tensor_sha256"]
        row["terminal_optimizer_elapsed_seconds"] = metadata["terminal"][
            "optimizer_elapsed_seconds"
        ]
        row["terminal_evaluation_elapsed_seconds"] = metadata["terminal"][
            "evaluation_elapsed_seconds"
        ]
    write_csv(args.output_dir / "terminal_metrics.csv", terminal_rows, fieldnames=terminal_fields)

    head_output = args.output_dir / "head_checkpoint_metrics.csv"
    temporary = head_output.with_suffix(".csv.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=HEAD_CHECKPOINT_FIELDS)
        writer.writeheader()
        for path in head_paths:
            with path.open(encoding="utf-8", newline="") as source:
                for row in csv.DictReader(source):
                    writer.writerow({field: row[field] for field in HEAD_CHECKPOINT_FIELDS})
    temporary.replace(head_output)
    return {
        "complete_runs": len(head_paths),
        "missing_runs": len(missing),
        "training_rows": len(training_rows),
        "checkpoint_rows": len(checkpoint_rows),
        "terminal_rows": len(terminal_rows),
        "head_rows": sum(
            int(read_json(path.parent / "run_config.json")["row_counts"]["head_checkpoint_metrics"])
            for path in head_paths
        ),
    }


def smoke_stage(args: argparse.Namespace) -> None:
    capture_stage(args)
    inputs = load_normalized_inputs(args)
    started = time.perf_counter()
    metadata = execute_one_condition(
        args,
        bit_width=2,
        seed=17,
        horizon=100,
        inputs=inputs,
    )
    elapsed = time.perf_counter() - started
    rows = read_csv(run_dir(args, 2, 17, 100) / "training_curve.csv")
    checkpoints = read_csv(run_dir(args, 2, 17, 100) / "checkpoint_metrics.csv")
    expected_steps = list(range(0, 101, args.metric_interval))
    actual_steps = [int(row["step"]) for row in rows]
    if actual_steps != expected_steps or [int(row["step"]) for row in checkpoints] != expected_steps:
        raise RuntimeError("smoke run did not record the exact 10-step schedule")
    if not math.isclose(float(rows[0]["learning_rate"]), args.learning_rate, rel_tol=0, abs_tol=1e-12):
        raise RuntimeError("smoke scheduler did not start at 0.005")
    if not math.isclose(float(rows[-1]["learning_rate"]), args.minimum_learning_rate, rel_tol=0, abs_tol=1e-10):
        raise RuntimeError("smoke scheduler did not finish at 0.00025")
    average_checkpoint = float(metadata["terminal"]["evaluation_elapsed_seconds"]) / len(checkpoints)
    completed_optimizer = float(metadata["terminal"]["optimizer_elapsed_seconds"])
    estimated_optimizer = completed_optimizer * (sum(args.horizons) * 9 / 100)
    estimate = {
        "smoke_elapsed_seconds": elapsed,
        "smoke_optimizer_seconds": completed_optimizer,
        "smoke_checkpoint_count": len(checkpoints),
        "average_checkpoint_evaluation_seconds": average_checkpoint,
        "estimated_total_optimizer_seconds": estimated_optimizer,
        "estimated_total_checkpoint_seconds": average_checkpoint * 5040,
        "estimated_total_seconds": estimated_optimizer + average_checkpoint * 5040,
    }
    config = read_json(args.output_dir / "training_length_sweep_config.json", {})
    config["runtime_estimate"] = estimate
    config["status"] = "smoke_complete"
    config["updated_at"] = utc_now()
    atomic_write_json(args.output_dir / "training_length_sweep_config.json", config)
    del inputs
    cleanup_cuda()
    print(f"[{utc_now()}] smoke checks passed; ETA={estimate['estimated_total_seconds'] / 60:.1f} min", flush=True)


def sweep_stage(args: argparse.Namespace) -> None:
    smoke_stage(args)
    inputs = load_normalized_inputs(args)
    completed = 0
    total = len(args.bits) * len(args.seeds) * len(args.horizons)
    for bit_width in args.bits:
        for seed in args.seeds:
            for horizon in args.horizons:
                execute_one_condition(
                    args,
                    bit_width=bit_width,
                    seed=seed,
                    horizon=horizon,
                    inputs=inputs,
                )
                completed += 1
                config = read_json(args.output_dir / "training_length_sweep_config.json", {})
                config["status"] = "sweep_running"
                config["progress"] = {
                    "completed_runs": completed,
                    "total_runs": total,
                    "last_condition": run_id(bit_width, seed, horizon),
                    "updated_at": utc_now(),
                }
                atomic_write_json(args.output_dir / "training_length_sweep_config.json", config)
    counts = aggregate_sweep_csvs(args, require_complete=True)
    config = read_json(args.output_dir / "training_length_sweep_config.json", {})
    config["status"] = "sweep_complete"
    config["progress"] = {**counts, "updated_at": utc_now()}
    atomic_write_json(args.output_dir / "training_length_sweep_config.json", config)
    del inputs
    cleanup_cuda()
    print(f"[{utc_now()}] all {total} independent sweep runs are complete", flush=True)


def rerun_terminal_rotations(
    args: argparse.Namespace,
    *,
    bit_width: int,
    seed: int,
    horizon: int,
    calibration_normalized: torch.Tensor,
) -> torch.Tensor:
    """Deterministically replay one schedule without retaining checkpoints."""

    device = calibration_normalized.device
    initial = build_random_rotations(32, 8, 128, seed).reshape(256, 128, 128).to(device)
    centroids = codebook_tensor(bit_width, 128, device=device)
    parameters = torch.nn.Parameter(torch.zeros_like(initial))
    optimizer = torch.optim.Adam([parameters], lr=args.learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=horizon, eta_min=args.minimum_learning_rate
    )
    generator = torch.Generator(device=device)
    generator.manual_seed(optimizer_seed(bit_width, seed))
    token_count = calibration_normalized.shape[1]
    for _step in range(1, horizon + 1):
        indices = torch.randint(
            token_count,
            (args.batch_tokens,),
            generator=generator,
            device=device,
        )
        batch = calibration_normalized[:, indices]
        optimizer.zero_grad(set_to_none=True)
        rotations = cayley_rotation(parameters, initial)
        rotated = torch.matmul(batch, rotations.transpose(-1, -2))
        targets = quantize_to_centroids(rotated.detach(), centroids)
        loss = torch.mean((rotated - targets) ** 2)
        loss.backward()
        torch.nn.utils.clip_grad_norm_([parameters], max_norm=1.0)
        optimizer.step()
        scheduler.step()
        del indices, batch, rotations, rotated, targets, loss
    learned = terminal_rotations(parameters, initial)
    del parameters, optimizer, scheduler, initial, centroids
    return learned


def evaluate_reference_80(
    args: argparse.Namespace,
    inputs: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
) -> list[dict[str, Any]]:
    output_path = args.output_dir / "reference_80_metrics.csv"
    existing = read_csv(output_path)
    if len(existing) == len(args.bits) * len(args.seeds):
        return [dict(row) for row in existing]
    calibration_normalized, calibration_norm_squared, validation_normalized, validation_norm_squared = inputs
    device = calibration_normalized.device
    rows: list[dict[str, Any]] = []
    for bit_width in args.bits:
        centroids = codebook_tensor(bit_width, 128, device=device)
        for seed in args.seeds:
            artifact = torch.load(
                reference_rotation_path(args, bit_width, seed),
                map_location="cpu",
                weights_only=True,
            )
            rotations = artifact["learned"].reshape(256, 128, 128).to(device)
            calibration = evaluate_dataset(
                calibration_normalized,
                calibration_norm_squared,
                rotations,
                centroids,
            )
            validation = evaluate_dataset(
                validation_normalized,
                validation_norm_squared,
                rotations,
                centroids,
            )
            rows.append(
                {
                    "bit_width": bit_width,
                    "seed": seed,
                    "horizon_steps": REFERENCE_STEPS,
                    "calibration_mse": calibration["normalized_key_mse"],
                    "wikitext_validation_mse": validation["normalized_key_mse"],
                    "original_scale_validation_mse": validation[
                        "original_scale_key_mse"
                    ],
                    "generalization_gap": validation["normalized_key_mse"]
                    - calibration["normalized_key_mse"],
                    "rotation_tensor_sha256": tensor_sha256(artifact["learned"]),
                }
            )
            del artifact, rotations, calibration, validation
        del centroids
    write_csv(output_path, rows)
    return rows


def select_horizons(
    args: argparse.Namespace,
    terminal_rows: list[dict[str, str]],
    reference_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "selection_created_at": utc_now(),
        "criterion": "seed-mean terminal WikiText validation normalized-key MSE",
        "near_minimum_relative_tolerance": SELECTION_TOLERANCE,
        "adoption_threshold_relative_to_step_80": ADOPTION_THRESHOLD,
        "domain_regression_threshold_relative_to_step_80": DOMAIN_REGRESSION_THRESHOLD,
        "bits": {},
    }
    for bit_width in args.bits:
        means: dict[int, float] = {}
        seed_values: dict[int, dict[int, float]] = defaultdict(dict)
        for row in terminal_rows:
            if int(row["bit_width"]) != bit_width:
                continue
            seed_values[int(row["horizon_steps"])][int(row["seed"])] = float(
                row["wikitext_validation_mse"]
            )
        for horizon in args.horizons:
            values = seed_values[horizon]
            if set(values) != set(args.seeds):
                raise RuntimeError(f"bit {bit_width}, horizon {horizon} lacks all seeds")
            means[horizon] = float(np.mean([values[seed] for seed in args.seeds]))
        minimum_horizon = min(args.horizons, key=lambda horizon: means[horizon])
        minimum_mse = means[minimum_horizon]
        sweep_candidate = min(
            horizon
            for horizon in args.horizons
            if means[horizon] <= (1.0 + SELECTION_TOLERANCE) * minimum_mse
        )
        reference_mean = float(
            np.mean(
                [
                    float(row["wikitext_validation_mse"])
                    for row in reference_rows
                    if int(row["bit_width"]) == bit_width
                ]
            )
        )
        improvement = (reference_mean - means[sweep_candidate]) / max(reference_mean, 1e-30)
        adopted = sweep_candidate if improvement >= ADOPTION_THRESHOLD else REFERENCE_STEPS
        result["bits"][str(bit_width)] = {
            "seed_mean_terminal_validation_mse": {
                str(horizon): means[horizon] for horizon in args.horizons
            },
            "minimum_horizon_steps": minimum_horizon,
            "minimum_terminal_validation_mse": minimum_mse,
            "sweep_candidate_horizon_steps": sweep_candidate,
            "sweep_candidate_terminal_validation_mse": means[sweep_candidate],
            "reference_80_terminal_validation_mse": reference_mean,
            "candidate_relative_improvement_vs_80": improvement,
            "longer_training_adopted": adopted != REFERENCE_STEPS,
            "selected_horizon_steps": adopted,
            "search_boundary_reached": minimum_horizon == max(args.horizons),
            "decision": (
                f"adopt {adopted} steps"
                if adopted != REFERENCE_STEPS
                else "retain existing 80 steps because candidate improvement is below 0.5%"
            ),
        }
    return result


def build_final_rotation_artifacts(
    args: argparse.Namespace,
    selection: dict[str, Any],
    calibration_normalized: torch.Tensor,
) -> None:
    output_dir = args.output_dir / "final_rotation_artifacts"
    output_dir.mkdir(parents=True, exist_ok=True)
    for bit_width in args.bits:
        decision = selection["bits"][str(bit_width)]
        horizon = int(decision["selected_horizon_steps"])
        for seed in args.seeds:
            destination = selected_rotation_path(args, bit_width, seed)
            if destination.exists():
                artifact = torch.load(destination, map_location="cpu", weights_only=True)
                if (
                    int(artifact.get("horizon_steps", -1)) == horizon
                    and artifact.get("learned_tensor_sha256")
                    == tensor_sha256(artifact["learned"])
                ):
                    decision.setdefault("artifact_hashes", {})[str(seed)] = {
                        "file_sha256": sha256_file(destination),
                        "learned_tensor_sha256": artifact["learned_tensor_sha256"],
                        "deterministic_rerun_match": artifact.get(
                            "deterministic_rerun_match", True
                        ),
                    }
                    continue
            reference = torch.load(
                reference_rotation_path(args, bit_width, seed),
                map_location="cpu",
                weights_only=True,
            )
            if horizon == REFERENCE_STEPS:
                learned = reference["learned"].reshape(256, 128, 128).to(
                    calibration_normalized.device
                )
                expected_hash = tensor_sha256(reference["learned"])
                deterministic_match = True
                source = "existing_step_80_reference"
            else:
                print(
                    f"[{utc_now()}] deterministic selected-horizon rerun "
                    f"bit={bit_width}, seed={seed}, steps={horizon}",
                    flush=True,
                )
                learned = rerun_terminal_rotations(
                    args,
                    bit_width=bit_width,
                    seed=seed,
                    horizon=horizon,
                    calibration_normalized=calibration_normalized,
                )
                metadata = read_json(
                    run_dir(args, bit_width, seed, horizon) / "run_config.json", {}
                )
                expected_hash = metadata["terminal"]["rotation_tensor_sha256"]
                deterministic_match = tensor_sha256(learned) == expected_hash
                if not deterministic_match:
                    raise RuntimeError(
                        f"deterministic rerun hash mismatch for bit={bit_width}, "
                        f"seed={seed}, horizon={horizon}"
                    )
                source = run_id(bit_width, seed, horizon)
            learned_cpu = learned.reshape(32, 8, 128, 128).cpu()
            payload = {
                "bit_width": bit_width,
                "seed": seed,
                "horizon_steps": horizon,
                "source": source,
                "optimizer_seed": optimizer_seed(bit_width, seed),
                "random": reference["random"],
                "learned": learned_cpu,
                "learned_tensor_sha256": tensor_sha256(learned_cpu),
                "expected_sweep_tensor_sha256": expected_hash,
                "deterministic_rerun_match": deterministic_match,
                "protocol_sha256": read_json(
                    args.output_dir / "training_length_sweep_config.json", {}
                )["protocol_sha256"],
            }
            atomic_torch_save(payload, destination)
            decision.setdefault("artifact_hashes", {})[str(seed)] = {
                "file_sha256": sha256_file(destination),
                "learned_tensor_sha256": payload["learned_tensor_sha256"],
                "deterministic_rerun_match": deterministic_match,
            }
            del reference, learned, learned_cpu, payload


def select_stage(args: argparse.Namespace) -> None:
    sweep_stage(args)
    counts = aggregate_sweep_csvs(args, require_complete=True)
    terminal_rows = read_csv(args.output_dir / "terminal_metrics.csv")
    if len(terminal_rows) != 90:
        raise RuntimeError(f"terminal_metrics.csv has {len(terminal_rows)} rows, expected 90")
    inputs = load_normalized_inputs(args)
    reference_rows = evaluate_reference_80(args, inputs)
    selection = select_horizons(args, terminal_rows, reference_rows)
    build_final_rotation_artifacts(args, selection, inputs[0])
    selection["artifact_directory"] = str(args.output_dir / "final_rotation_artifacts")
    selection["sweep_counts"] = counts
    atomic_write_json(args.output_dir / "selected_horizons.json", selection)
    config = read_json(args.output_dir / "training_length_sweep_config.json", {})
    config["status"] = "selection_complete"
    config["selection_sha256"] = sha256_file(args.output_dir / "selected_horizons.json")
    config["updated_at"] = utc_now()
    atomic_write_json(args.output_dir / "training_length_sweep_config.json", config)
    del inputs
    cleanup_cuda()
    print(f"[{utc_now()}] horizon selection and deterministic reruns are complete", flush=True)


def load_tinystories_keys(payload: dict[str, torch.Tensor]) -> torch.Tensor:
    keys = payload["k"]
    if keys.ndim != 4:
        raise RuntimeError("TinyStories key capture must be four-dimensional")
    if keys.shape[1] == DOMAIN_TOKENS:
        return keys.permute(0, 2, 1, 3).contiguous()
    return keys


def attach_sanity_summary(
    args: argparse.Namespace,
    selection: dict[str, Any],
    sanity_rows: list[dict[str, Any]],
) -> None:
    """Attach derived TinyStories means after either a fresh or resumed sanity run."""
    for bit_width in args.bits:
        reference_mean = float(
            np.mean(
                [
                    float(row["normalized_key_mse"])
                    for row in sanity_rows
                    if int(row["bit_width"]) == bit_width
                    and row["condition"] == "existing_80_step"
                ]
            )
        )
        selected_mean = float(
            np.mean(
                [
                    float(row["normalized_key_mse"])
                    for row in sanity_rows
                    if int(row["bit_width"]) == bit_width
                    and row["condition"] == "selected_horizon"
                ]
            )
        )
        regression = (selected_mean - reference_mean) / max(reference_mean, 1e-30)
        selection["bits"][str(bit_width)]["tinystories"] = {
            "reference_80_normalized_key_mse": reference_mean,
            "selected_normalized_key_mse": selected_mean,
            "relative_change_selected_vs_80": regression,
            "domain_overfitting_warning": regression >= DOMAIN_REGRESSION_THRESHOLD,
        }


def sanity_stage(args: argparse.Namespace) -> None:
    select_stage(args)
    selection_path = args.output_dir / "selected_horizons.json"
    selection = read_json(selection_path, {})
    sanity_path = args.output_dir / "sanity_metrics.csv"
    diagnostic_path = args.output_dir / "headwise_diagnostics.csv"
    if sanity_path.exists() and diagnostic_path.exists():
        expected_sanity = len(args.bits) * len(args.seeds) * 3
        expected_diagnostics = len(args.bits) * len(args.seeds) * 4 * 32 * 8
        sanity_rows = read_csv(sanity_path)
        if (
            len(sanity_rows) == expected_sanity
            and len(read_csv(diagnostic_path)) == expected_diagnostics
        ):
            attach_sanity_summary(args, selection, sanity_rows)
            selection["sanity_metrics_sha256"] = sha256_file(sanity_path)
            selection["headwise_diagnostics_sha256"] = sha256_file(diagnostic_path)
            atomic_write_json(selection_path, selection)
            config = read_json(args.output_dir / "training_length_sweep_config.json", {})
            config["status"] = "sanity_complete"
            config["selection_sha256"] = sha256_file(selection_path)
            config["updated_at"] = utc_now()
            atomic_write_json(
                args.output_dir / "training_length_sweep_config.json", config
            )
            print(f"[{utc_now()}] TinyStories sanity artifacts already complete", flush=True)
            return

    device = torch.device(args.device)
    tiny = torch.load(
        args.output_dir / "activations" / "tinystories.pt",
        map_location="cpu",
        weights_only=True,
    )
    tiny_keys = load_tinystories_keys(tiny)
    tiny_normalized, tiny_norm_squared = normalized_dataset(tiny_keys, device)
    calibration, validation = load_key_tensors(args)
    calibration_normalized, _calibration_norms = normalized_dataset(calibration, device)
    validation_normalized, validation_norm_squared = normalized_dataset(validation, device)
    del calibration, validation, tiny_keys

    sanity_rows: list[dict[str, Any]] = []
    diagnostic_rows: list[dict[str, Any]] = []
    for bit_width in args.bits:
        centroids = codebook_tensor(bit_width, 128, device=device)
        selected_horizon = int(selection["bits"][str(bit_width)]["selected_horizon_steps"])
        for seed in args.seeds:
            reference = torch.load(
                reference_rotation_path(args, bit_width, seed),
                map_location="cpu",
                weights_only=True,
            )
            selected = torch.load(
                selected_rotation_path(args, bit_width, seed),
                map_location="cpu",
                weights_only=True,
            )
            conditions = {
                "random_initialization": reference["random"],
                "existing_80_step": reference["learned"],
                "selected_horizon": selected["learned"],
            }
            condition_metrics: dict[str, dict[str, Any]] = {}
            for condition, rotations_cpu in conditions.items():
                rotations = rotations_cpu.reshape(256, 128, 128).to(device)
                reconstruction = evaluate_dataset(
                    tiny_normalized,
                    tiny_norm_squared,
                    rotations,
                    centroids,
                )
                attention = attention_distortion_metrics(
                    tiny["q"],
                    tiny["k"][:, : args.attention_sequence_length],
                    tiny["v"],
                    tiny["cos"],
                    tiny["sin"],
                    rotations.reshape(32, 8, 128, 128),
                    centroids,
                    sequence_length=args.attention_sequence_length,
                    norm_correction=False,
                )
                condition_metrics[condition] = reconstruction
                sanity_rows.append(
                    {
                        "bit_width": bit_width,
                        "seed": seed,
                        "condition": condition,
                        "horizon_steps": (
                            0
                            if condition == "random_initialization"
                            else REFERENCE_STEPS
                            if condition == "existing_80_step"
                            else selected_horizon
                        ),
                        "tinystories_tokens": args.domain_tokens,
                        "attention_sequence_length": args.attention_sequence_length,
                        "normalized_key_mse": reconstruction["normalized_key_mse"],
                        "original_scale_key_mse": reconstruction[
                            "original_scale_key_mse"
                        ],
                        "attention_probability_kl": attention[
                            "attention_probability_kl"
                        ],
                        "attention_logit_mse": attention["attention_logit_mse"],
                        "attention_output_mse": attention["attention_output_mse"],
                        "rotation_tensor_sha256": tensor_sha256(rotations_cpu),
                    }
                )
                del rotations, reconstruction, attention

            thousand = rerun_terminal_rotations(
                args,
                bit_width=bit_width,
                seed=seed,
                horizon=1000,
                calibration_normalized=calibration_normalized,
            )
            expected_thousand_hash = read_json(
                run_dir(args, bit_width, seed, 1000) / "run_config.json", {}
            )["terminal"]["rotation_tensor_sha256"]
            if tensor_sha256(thousand) != expected_thousand_hash:
                raise RuntimeError(
                    f"1000-step diagnostic rerun mismatch for bit={bit_width}, seed={seed}"
                )
            diagnostic_conditions = {
                **conditions,
                "horizon_1000": thousand.reshape(32, 8, 128, 128).cpu(),
            }
            random_per_head: np.ndarray | None = None
            values_by_condition: dict[str, np.ndarray] = {}
            for condition, rotations_cpu in diagnostic_conditions.items():
                rotations = rotations_cpu.reshape(256, 128, 128).to(device)
                metrics = evaluate_dataset(
                    validation_normalized,
                    validation_norm_squared,
                    rotations,
                    centroids,
                )
                values = metrics["per_head_normalized_key_mse"]
                values_by_condition[condition] = values
                if condition == "random_initialization":
                    random_per_head = values
                del rotations, metrics
            assert random_per_head is not None
            for condition, values in values_by_condition.items():
                for flat_index, value in enumerate(values):
                    diagnostic_rows.append(
                        {
                            "bit_width": bit_width,
                            "seed": seed,
                            "condition": condition,
                            "layer": flat_index // 8,
                            "head": flat_index % 8,
                            "wikitext_validation_normalized_key_mse": float(value),
                            "relative_change_vs_random": float(
                                (value - random_per_head[flat_index])
                                / max(random_per_head[flat_index], 1e-30)
                            ),
                        }
                    )
            del reference, selected, conditions, condition_metrics, thousand, diagnostic_conditions
        del centroids

    write_csv(sanity_path, sanity_rows)
    write_csv(diagnostic_path, diagnostic_rows)
    attach_sanity_summary(args, selection, sanity_rows)
    selection["sanity_metrics_sha256"] = sha256_file(sanity_path)
    selection["headwise_diagnostics_sha256"] = sha256_file(diagnostic_path)
    atomic_write_json(selection_path, selection)
    config = read_json(args.output_dir / "training_length_sweep_config.json", {})
    config["status"] = "sanity_complete"
    config["selection_sha256"] = sha256_file(selection_path)
    config["updated_at"] = utc_now()
    atomic_write_json(args.output_dir / "training_length_sweep_config.json", config)
    del tiny, tiny_normalized, tiny_norm_squared, calibration_normalized, _calibration_norms, validation_normalized, validation_norm_squared
    cleanup_cuda()
    print(f"[{utc_now()}] TinyStories and headwise sanity evaluation complete", flush=True)


def meaningful_longbench_bits(selection: dict[str, Any]) -> list[int]:
    return [
        int(bit_width)
        for bit_width, decision in selection["bits"].items()
        if decision["longer_training_adopted"]
        and float(decision["candidate_relative_improvement_vs_80"])
        >= ADOPTION_THRESHOLD
    ]


def prepare_longbench_rotation_dir(args: argparse.Namespace) -> Path:
    destination = args.output_dir / "longbench_selected_rotations"
    rotation_dir = destination / "rotations"
    rotation_dir.mkdir(parents=True, exist_ok=True)
    reference_config = read_json(args.reference_dir / "run_config.json", {})
    reference_config["training_length_sweep"] = {
        "selected_horizons": str(args.output_dir / "selected_horizons.json"),
        "selected_horizons_sha256": sha256_file(
            args.output_dir / "selected_horizons.json"
        ),
    }
    atomic_write_json(destination / "run_config.json", reference_config)
    for bit_width in args.bits:
        for seed in args.seeds:
            source = selected_rotation_path(args, bit_width, seed)
            target = rotation_dir / source.name
            if target.exists() and sha256_file(target) == sha256_file(source):
                continue
            temporary = target.with_suffix(".pt.tmp")
            shutil.copy2(source, temporary)
            temporary.replace(target)
    return destination


def run_longbench_condition(
    args: argparse.Namespace,
    *,
    rotation_dir: Path,
    output_dir: Path,
    bit_width: int,
    seed: int,
    stage: str,
) -> None:
    condition = f"learned_K{bit_width}_V16_s{seed}"
    command = [
        sys.executable,
        "-m",
        "experiments.spin_turboquant.longbench",
        "--stage",
        stage,
        "--model",
        str(args.model),
        "--rotation-dir",
        str(rotation_dir),
        "--longbench-repo",
        str(args.longbench_repo),
        "--data-dir",
        str(args.longbench_data),
        "--output-dir",
        str(output_dir),
        "--condition",
        condition,
        "--device",
        str(args.device),
    ]
    print(f"[{utc_now()}] $ {' '.join(command)}", flush=True)
    subprocess.run(command, cwd=REPOSITORY_ROOT, check=True)


def summarize_longbench_paired(
    args: argparse.Namespace, bits: Sequence[int], selected_output: Path
) -> dict[str, Any]:
    paired_rows: list[dict[str, Any]] = []
    condition_summaries: list[dict[str, Any]] = []
    for bit_width in bits:
        for seed in args.seeds:
            condition = f"learned_K{bit_width}_V16_s{seed}"
            reference_dir = args.reference_longbench_dir / "full" / condition
            selected_dir = selected_output / "full" / condition
            reference_scores = read_csv(reference_dir / "scores.csv")
            selected_scores = read_csv(selected_dir / "scores.csv")
            if len(reference_scores) != 195 or len(selected_scores) != 195:
                raise RuntimeError(
                    f"LongBench-E {condition} must have 195 matched scores; "
                    f"found {len(reference_scores)} and {len(selected_scores)}"
                )
            reference_lookup = {
                (row["task"], row["example_id"]): row for row in reference_scores
            }
            selected_lookup = {
                (row["task"], row["example_id"]): row for row in selected_scores
            }
            if set(reference_lookup) != set(selected_lookup):
                raise RuntimeError(f"LongBench-E identities differ for {condition}")
            task_differences: dict[str, list[float]] = defaultdict(list)
            for key in sorted(reference_lookup):
                reference = reference_lookup[key]
                selected = selected_lookup[key]
                difference = float(selected["score"]) - float(reference["score"])
                task_differences[reference["task"]].append(difference)
                paired_rows.append(
                    {
                        "bit_width": bit_width,
                        "seed": seed,
                        "task": reference["task"],
                        "category": reference["category"],
                        "example_id": reference["example_id"],
                        "dataset_length": reference["dataset_length"],
                        "length_bucket": reference["length_bucket"],
                        "step_80_score": reference["score"],
                        "selected_horizon_score": selected["score"],
                        "difference": difference,
                    }
                )
            macro_difference = float(
                np.mean(
                    [np.mean(values) for values in task_differences.values()]
                )
            )
            selected_predictions = read_csv(selected_dir / "scores.csv")
            condition_summaries.append(
                {
                    "bit_width": bit_width,
                    "seed": seed,
                    "examples": len(selected_predictions),
                    "tasks": len(task_differences),
                    "task_macro_difference_selected_minus_80": macro_difference,
                }
            )
    write_csv(args.output_dir / "longbench_e_paired.csv", paired_rows)
    write_csv(args.output_dir / "longbench_e_condition_summary.csv", condition_summaries)
    summary: dict[str, Any] = {
        "status": "complete",
        "bits_evaluated": list(bits),
        "paired_rows": len(paired_rows),
        "conditions": condition_summaries,
        "bit_seed_mean_task_macro_difference": {
            str(bit_width): float(
                np.mean(
                    [
                        row["task_macro_difference_selected_minus_80"]
                        for row in condition_summaries
                        if row["bit_width"] == bit_width
                    ]
                )
            )
            for bit_width in bits
        },
        "subset_manifest_sha256": sha256_file(
            selected_output / "subset_manifest.json"
        ),
        "reference_subset_manifest_sha256": sha256_file(
            args.reference_longbench_dir / "subset_manifest.json"
        ),
    }
    if summary["subset_manifest_sha256"] != summary["reference_subset_manifest_sha256"]:
        raise RuntimeError("selected LongBench-E subset manifest differs from step-80 reference")
    atomic_write_json(args.output_dir / "longbench_e_decision.json", summary)
    return summary


def longbench_stage(args: argparse.Namespace) -> None:
    sanity_stage(args)
    selection = read_json(args.output_dir / "selected_horizons.json", {})
    bits = meaningful_longbench_bits(selection)
    decision_path = args.output_dir / "longbench_e_decision.json"
    if not bits:
        atomic_write_json(
            decision_path,
            {
                "status": "not_required",
                "reason": "no bit width met the 0.5% adoption threshold",
                "bits_evaluated": [],
            },
        )
        print(f"[{utc_now()}] LongBench-E is not required by the decision rule", flush=True)
        return
    if args.skip_longbench:
        selected_output = args.output_dir / "longbench_e_selected"
        partial_progress: dict[str, dict[str, int]] = {}
        for stage in ("smoke", "full"):
            stage_progress: dict[str, int] = {}
            stage_dir = selected_output / stage
            if stage_dir.exists():
                for condition_dir in sorted(path for path in stage_dir.iterdir() if path.is_dir()):
                    predictions_path = condition_dir / "predictions.jsonl"
                    stage_progress[condition_dir.name] = (
                        sum(1 for line in predictions_path.open() if line.strip())
                        if predictions_path.exists()
                        else 0
                    )
            partial_progress[stage] = stage_progress
        atomic_write_json(
            decision_path,
            {
                "status": "skipped_by_explicit_flag",
                "reason": "LongBench-E was skipped at the user's explicit request.",
                "bits_requiring_evaluation": bits,
                "partial_progress_predictions": partial_progress,
            },
        )
        print(f"[{utc_now()}] LongBench-E skipped by --skip-longbench", flush=True)
        return
    for path in (
        args.longbench_repo,
        args.longbench_data,
        args.reference_longbench_dir / "subset_manifest.json",
    ):
        if not path.exists():
            raise FileNotFoundError(path)
    rotation_dir = prepare_longbench_rotation_dir(args)
    output_dir = args.output_dir / "longbench_e_selected"
    output_dir.mkdir(parents=True, exist_ok=True)
    subset_target = output_dir / "subset_manifest.json"
    if not subset_target.exists():
        shutil.copy2(args.reference_longbench_dir / "subset_manifest.json", subset_target)
    for bit_width in bits:
        for seed in args.seeds:
            run_longbench_condition(
                args,
                rotation_dir=rotation_dir,
                output_dir=output_dir,
                bit_width=bit_width,
                seed=seed,
                stage="smoke",
            )
            run_longbench_condition(
                args,
                rotation_dir=rotation_dir,
                output_dir=output_dir,
                bit_width=bit_width,
                seed=seed,
                stage="full",
            )
    summarize_longbench_paired(args, bits, output_dir)
    config = read_json(args.output_dir / "training_length_sweep_config.json", {})
    config["status"] = "longbench_complete"
    config["updated_at"] = utc_now()
    atomic_write_json(args.output_dir / "training_length_sweep_config.json", config)
    print(f"[{utc_now()}] paired LongBench-E evaluation complete for bits {bits}", flush=True)


def grouped_mean_std(
    rows: Iterable[dict[str, str]],
    *,
    value_field: str,
) -> dict[tuple[int, int], tuple[float, float]]:
    grouped: dict[tuple[int, int], list[float]] = defaultdict(list)
    for row in rows:
        grouped[(int(row["bit_width"]), int(row["horizon_steps"]))].append(
            float(row[value_field])
        )
    return {
        key: (float(np.mean(values)), float(np.std(values, ddof=0)))
        for key, values in grouped.items()
    }


def plot_terminal_metric(
    args: argparse.Namespace,
    terminal_rows: list[dict[str, str]],
    reference_rows: list[dict[str, str]],
    *,
    field: str,
    ylabel: str,
    filename: str,
) -> None:
    values = grouped_mean_std(terminal_rows, value_field=field)
    reference = grouped_mean_std(reference_rows, value_field=field)
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = {2: "#1f77b4", 3: "#ff7f0e", 4: "#2ca02c"}
    for bit_width in args.bits:
        means = np.asarray([values[(bit_width, horizon)][0] for horizon in args.horizons])
        stds = np.asarray([values[(bit_width, horizon)][1] for horizon in args.horizons])
        ax.plot(args.horizons, means, marker="o", label=f"{bit_width}-bit", color=colors[bit_width])
        ax.fill_between(args.horizons, means - stds, means + stds, color=colors[bit_width], alpha=0.18)
        reference_mean = reference[(bit_width, REFERENCE_STEPS)][0]
        ax.scatter([REFERENCE_STEPS], [reference_mean], marker="*", s=130, color=colors[bit_width], edgecolor="black", linewidth=0.5)
    ax.set_xlabel("Configured independent training horizon (steps)")
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(args.output_dir / "plots" / filename, dpi=180)
    plt.close(fig)


def plot_relative_improvement(
    args: argparse.Namespace, terminal_rows: list[dict[str, str]]
) -> None:
    values = grouped_mean_std(terminal_rows, value_field="relative_improvement")
    fig, ax = plt.subplots(figsize=(8, 5))
    for bit_width in args.bits:
        means = np.asarray([100 * values[(bit_width, horizon)][0] for horizon in args.horizons])
        stds = np.asarray([100 * values[(bit_width, horizon)][1] for horizon in args.horizons])
        ax.plot(args.horizons, means, marker="o", label=f"{bit_width}-bit")
        ax.fill_between(args.horizons, means - stds, means + stds, alpha=0.18)
    ax.set_xlabel("Configured independent training horizon (steps)")
    ax.set_ylabel("Terminal WikiText MSE reduction vs random (%)")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(args.output_dir / "plots" / "relative_improvement_by_horizon.png", dpi=180)
    plt.close(fig)


def plot_within_run_curves(
    args: argparse.Namespace, training_rows: list[dict[str, str]]
) -> None:
    plot_dir = args.output_dir / "plots"
    grouped: dict[tuple[int, int, int], dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for row in training_rows:
        step = int(row["step"])
        if step == 0:
            continue
        key = (int(row["bit_width"]), int(row["horizon_steps"]), step)
        for field in ("minibatch_mse", "ema_mse", "gradient_norm"):
            grouped[key][field].append(float(row[field]))

    fig, axes = plt.subplots(2, 5, figsize=(20, 8), sharey=False)
    for axis, horizon in zip(axes.flat, args.horizons):
        for bit_width in args.bits:
            steps = list(range(args.metric_interval, horizon + 1, args.metric_interval))
            minibatch = [np.mean(grouped[(bit_width, horizon, step)]["minibatch_mse"]) for step in steps]
            ema = [np.mean(grouped[(bit_width, horizon, step)]["ema_mse"]) for step in steps]
            axis.plot(steps, minibatch, alpha=0.35, linewidth=0.9, label=f"b{bit_width} batch")
            axis.plot(steps, ema, linewidth=1.6, label=f"b{bit_width} EMA")
        axis.set_title(f"Independent H={horizon}")
        axis.ticklabel_format(axis="y", style="sci", scilimits=(-3, 3))
        axis.grid(alpha=0.2)
    axes[0, 0].legend(fontsize=7, ncol=2)
    fig.supxlabel("Optimizer step")
    fig.supylabel("Normalized-key minibatch MSE")
    fig.tight_layout()
    fig.savefig(plot_dir / "within_run_loss_curves.png", dpi=170)
    plt.close(fig)

    fig, axes = plt.subplots(2, 5, figsize=(20, 8), sharey=True)
    for axis, horizon in zip(axes.flat, args.horizons):
        rows = [
            row
            for row in training_rows
            if int(row["bit_width"]) == args.bits[0]
            and int(row["seed"]) == args.seeds[0]
            and int(row["horizon_steps"]) == horizon
        ]
        axis.plot([int(row["step"]) for row in rows], [float(row["learning_rate"]) for row in rows])
        axis.set_title(f"Independent H={horizon}")
        axis.grid(alpha=0.2)
    fig.supxlabel("Optimizer step")
    fig.supylabel("Learning rate")
    fig.tight_layout()
    fig.savefig(plot_dir / "learning_rate_curves.png", dpi=170)
    plt.close(fig)

    fig, axes = plt.subplots(2, 5, figsize=(20, 8), sharey=False)
    for axis, horizon in zip(axes.flat, args.horizons):
        for bit_width in args.bits:
            steps = list(range(args.metric_interval, horizon + 1, args.metric_interval))
            values = [np.mean(grouped[(bit_width, horizon, step)]["gradient_norm"]) for step in steps]
            axis.plot(steps, values, label=f"{bit_width}-bit")
        axis.set_title(f"Independent H={horizon}")
        axis.grid(alpha=0.2)
    axes[0, 0].legend(fontsize=8)
    fig.supxlabel("Optimizer step")
    fig.supylabel("Gradient norm before clipping")
    fig.tight_layout()
    fig.savefig(plot_dir / "gradient_norm_curves.png", dpi=170)
    plt.close(fig)


def plot_headwise_distribution(args: argparse.Namespace) -> None:
    rows = read_csv(args.output_dir / "headwise_diagnostics.csv")
    conditions = (
        "random_initialization",
        "existing_80_step",
        "selected_horizon",
        "horizon_1000",
    )
    labels = ("Random", "80-step", "Selected", "1000-step")
    fig, axes = plt.subplots(3, 2, figsize=(12, 13))
    for row_index, bit_width in enumerate(args.bits):
        absolute = [
            [
                float(row["wikitext_validation_normalized_key_mse"])
                for row in rows
                if int(row["bit_width"]) == bit_width and row["condition"] == condition
            ]
            for condition in conditions
        ]
        relative = [
            [
                100 * float(row["relative_change_vs_random"])
                for row in rows
                if int(row["bit_width"]) == bit_width and row["condition"] == condition
            ]
            for condition in conditions
        ]
        axes[row_index, 0].boxplot(absolute, tick_labels=labels, showfliers=False)
        axes[row_index, 0].set_ylim(bottom=0)
        axes[row_index, 0].set_ylabel(f"{bit_width}-bit absolute MSE")
        axes[row_index, 0].grid(axis="y", alpha=0.2)
        axes[row_index, 1].boxplot(relative, tick_labels=labels, showfliers=False)
        axes[row_index, 1].axhline(0, color="black", linewidth=0.8)
        axes[row_index, 1].set_ylabel("Relative change vs random (%)")
        axes[row_index, 1].grid(axis="y", alpha=0.2)
    fig.suptitle("WikiText validation layer/KV-head distributions (3 seeds)")
    fig.tight_layout()
    fig.savefig(args.output_dir / "plots" / "headwise_distribution.png", dpi=180)
    plt.close(fig)


def plots_stage(args: argparse.Namespace) -> None:
    plot_dir = args.output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    terminal_rows = read_csv(args.output_dir / "terminal_metrics.csv")
    reference_rows = read_csv(args.output_dir / "reference_80_metrics.csv")
    training_rows = read_csv(args.output_dir / "training_curve.csv")
    plot_terminal_metric(
        args,
        terminal_rows,
        reference_rows,
        field="wikitext_validation_mse",
        ylabel="Terminal WikiText validation normalized-key MSE",
        filename="terminal_validation_by_horizon.png",
    )
    plot_terminal_metric(
        args,
        terminal_rows,
        reference_rows,
        field="calibration_mse",
        ylabel="Terminal full-calibration normalized-key MSE",
        filename="terminal_calibration_by_horizon.png",
    )
    plot_terminal_metric(
        args,
        terminal_rows,
        reference_rows,
        field="generalization_gap",
        ylabel="Validation MSE - calibration MSE",
        filename="generalization_gap_by_horizon.png",
    )
    wall_rows = [dict(row) for row in terminal_rows]
    for row in wall_rows:
        row["wall_clock"] = row["terminal_optimizer_elapsed_seconds"]
    reference_wall_rows = [dict(row) for row in reference_rows]
    reference_training = read_csv(args.reference_dir / "training.csv")
    reference_elapsed = {
        (int(row["bit_width"]), int(row["seed"])): float(row["elapsed_seconds"])
        for row in reference_training
    }
    for row in reference_wall_rows:
        row["wall_clock"] = reference_elapsed[(int(row["bit_width"]), int(row["seed"]))]
    plot_terminal_metric(
        args,
        wall_rows,
        reference_wall_rows,
        field="wall_clock",
        ylabel="Terminal optimizer wall-clock time (seconds)",
        filename="wall_clock_by_horizon.png",
    )
    plot_relative_improvement(args, terminal_rows)
    plot_within_run_curves(args, training_rows)
    plot_headwise_distribution(args)


def markdown_table(headers: Sequence[str], rows: Iterable[Sequence[Any]]) -> list[str]:
    lines = [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join("---" for _ in headers) + "|",
    ]
    lines.extend("| " + " | ".join(str(value) for value in row) + " |" for row in rows)
    return lines


def build_report(args: argparse.Namespace) -> None:
    selection = read_json(args.output_dir / "selected_horizons.json", {})
    terminal_rows = read_csv(args.output_dir / "terminal_metrics.csv")
    sanity_rows = read_csv(args.output_dir / "sanity_metrics.csv")
    longbench = read_json(args.output_dir / "longbench_e_decision.json", {})
    lines = [
        "# Learned rotation training-length sweep",
        "",
        f"Generated: {utc_now()}",
        "",
        "## Result",
        "",
        "The primary selection used only the seed-mean terminal WikiText-2 validation normalized-key MSE. "
        "TinyStories and attention metrics were evaluated only after selection, and LongBench-E was gated by the predeclared 0.5% improvement rule.",
        "",
        "## Selected horizons",
        "",
    ]
    selection_rows = []
    for bit_width in args.bits:
        item = selection["bits"][str(bit_width)]
        tiny = item.get("tinystories", {})
        selection_rows.append(
            (
                bit_width,
                item["minimum_horizon_steps"],
                item["sweep_candidate_horizon_steps"],
                item["selected_horizon_steps"],
                f"{100 * item['candidate_relative_improvement_vs_80']:.3f}%",
                "yes" if item["search_boundary_reached"] else "no",
                f"{100 * tiny.get('relative_change_selected_vs_80', 0):.3f}%",
                "yes" if tiny.get("domain_overfitting_warning", False) else "no",
            )
        )
    lines.extend(
        markdown_table(
            (
                "bits",
                "argmin H",
                "0.1%-tolerant H*",
                "adopted H",
                "WikiText gain vs 80",
                "boundary",
                "TinyStories change",
                "domain warning",
            ),
            selection_rows,
        )
    )
    lines.extend(["", "A negative TinyStories change means lower MSE than the 80-step reference.", ""])

    lines.extend(["## Terminal sweep metrics", ""])
    terminal_summary_rows = []
    for bit_width in args.bits:
        for horizon in args.horizons:
            rows = [
                row
                for row in terminal_rows
                if int(row["bit_width"]) == bit_width
                and int(row["horizon_steps"]) == horizon
            ]
            terminal_summary_rows.append(
                (
                    bit_width,
                    horizon,
                    f"{np.mean([float(row['calibration_mse']) for row in rows]):.9g}",
                    f"{np.mean([float(row['wikitext_validation_mse']) for row in rows]):.9g}",
                    f"{np.mean([float(row['generalization_gap']) for row in rows]):.9g}",
                    f"{np.mean([float(row['terminal_optimizer_elapsed_seconds']) for row in rows]):.3f}",
                )
            )
    lines.extend(
        markdown_table(
            ("bits", "H", "calibration MSE", "validation MSE", "gap", "optimizer sec"),
            terminal_summary_rows,
        )
    )

    lines.extend(["", "## TinyStories post-selection sanity", ""])
    sanity_summary_rows = []
    for bit_width in args.bits:
        for condition in ("random_initialization", "existing_80_step", "selected_horizon"):
            rows = [
                row
                for row in sanity_rows
                if int(row["bit_width"]) == bit_width and row["condition"] == condition
            ]
            sanity_summary_rows.append(
                (
                    bit_width,
                    condition,
                    f"{np.mean([float(row['normalized_key_mse']) for row in rows]):.9g}",
                    f"{np.mean([float(row['original_scale_key_mse']) for row in rows]):.9g}",
                    f"{np.mean([float(row['attention_probability_kl']) for row in rows]):.9g}",
                    f"{np.mean([float(row['attention_logit_mse']) for row in rows]):.9g}",
                )
            )
    lines.extend(
        markdown_table(
            ("bits", "condition", "normalized MSE", "original MSE", "attention KL", "logit MSE"),
            sanity_summary_rows,
        )
    )
    lines.extend(["", "Attention metrics were not used to revise the selected horizons.", ""])

    lines.extend(["## LongBench-E decision", ""])
    if longbench.get("status") == "complete":
        lines.append(
            f"Paired evaluation completed for bits {longbench['bits_evaluated']} on the exact step-80 subset manifest."
        )
        for bit_width, difference in longbench["bit_seed_mean_task_macro_difference"].items():
            lines.append(f"- {bit_width}-bit selected minus step-80 task-macro score: {difference:.6f}")
    else:
        lines.append(f"Status: {longbench.get('status')}. {longbench.get('reason', '')}")
        partial_progress = longbench.get("partial_progress_predictions", {})
        if any(partial_progress.get(stage) for stage in ("smoke", "full")):
            lines.append(
                "Partial artifacts were retained but were not used for a paired "
                "LongBench-E conclusion: "
                + ", ".join(
                    f"{stage} "
                    + "; ".join(
                        f"{condition}={count}/{'13' if stage == 'smoke' else '195'}"
                        for condition, count in partial_progress.get(stage, {}).items()
                    )
                    for stage in ("smoke", "full")
                    if partial_progress.get(stage)
                )
                + "."
            )
    lines.extend(
        [
            "",
            "## Interpretation guardrails",
            "",
            "- Every 100-1000 horizon is an independent zero-Cayley/Adam restart with the same bit/seed minibatch prefix and its own T_max=H cosine schedule.",
            "- Primary comparisons are terminal checkpoints; equal intermediate step numbers across horizons do not share a learning rate.",
            "- A minimum at 1000 steps is reported as a search-boundary result, not a confirmed global optimum.",
            "- These offline reconstruction and attention measurements do not by themselves establish end-to-end perplexity or generation quality.",
            "",
            "## Reproduce",
            "",
            "```bash",
            "conda activate stq",
            f"python -m experiments.spin_turboquant.training_length_sweep --stage orchestrate --model {args.model} --reference-dir {args.reference_dir} --output-dir {args.output_dir}"
            + (" --skip-longbench" if longbench.get("status") == "skipped_by_explicit_flag" else ""),
            "```",
            "",
            "Raw condition evidence is under `runs/`; aggregate CSVs, selected rotation artifacts, post-selection metrics, plots, and the completion audit are in this directory.",
        ]
    )
    (args.output_dir / "report.md").write_text("\n".join(lines) + "\n")


def completion_audit(args: argparse.Namespace) -> dict[str, Any]:
    required_plots = (
        "terminal_validation_by_horizon.png",
        "terminal_calibration_by_horizon.png",
        "generalization_gap_by_horizon.png",
        "within_run_loss_curves.png",
        "learning_rate_curves.png",
        "headwise_distribution.png",
        "wall_clock_by_horizon.png",
    )
    selection = read_json(args.output_dir / "selected_horizons.json", {})
    longbench = read_json(args.output_dir / "longbench_e_decision.json", {})
    meaningful_bits = meaningful_longbench_bits(selection)
    minibatch_prefixes = {
        (bit, seed): {
            read_json(
                run_dir(args, bit, seed, horizon) / "run_config.json", {}
            ).get("first_100_minibatch_indices_sha256")
            for horizon in args.horizons
        }
        for bit in args.bits
        for seed in args.seeds
    }
    checks: dict[str, Any] = {
        "all_90_independent_runs_complete": sum(
            completed_run_metadata(args, bit, seed, horizon) is not None
            for bit in args.bits
            for seed in args.seeds
            for horizon in args.horizons
        )
        == 90,
        "same_first_100_minibatch_prefix_per_bit_seed": all(
            len(hashes) == 1 and None not in hashes
            for hashes in minibatch_prefixes.values()
        ),
        "training_curve_rows_5040": len(read_csv(args.output_dir / "training_curve.csv")) == 5040,
        "checkpoint_rows_5040": len(read_csv(args.output_dir / "checkpoint_metrics.csv")) == 5040,
        "terminal_rows_90": len(read_csv(args.output_dir / "terminal_metrics.csv")) == 90,
        "head_checkpoint_rows_2580480": sum(1 for _ in (args.output_dir / "head_checkpoint_metrics.csv").open()) - 1 == 2_580_480,
        "selection_has_all_bits": set(selection.get("bits", {})) == {"2", "3", "4"},
        "nine_final_artifacts": sum(
            selected_rotation_path(args, bit, seed).exists()
            for bit in args.bits
            for seed in args.seeds
        )
        == 9,
        "all_rerun_hashes_match": all(
            value.get("deterministic_rerun_match")
            for decision in selection.get("bits", {}).values()
            for value in decision.get("artifact_hashes", {}).values()
        ),
        "sanity_rows_27": len(read_csv(args.output_dir / "sanity_metrics.csv")) == 27,
        "headwise_diagnostic_rows_9216": len(read_csv(args.output_dir / "headwise_diagnostics.csv")) == 9216,
        "required_plots_exist": all(
            (args.output_dir / "plots" / name).exists() for name in required_plots
        ),
        "report_exists": (args.output_dir / "report.md").exists(),
        "longbench_disposition_accepted": (
            longbench.get("status") == "not_required"
            if not meaningful_bits
            else longbench.get("status") == "complete"
            or (
                args.skip_longbench
                and longbench.get("status") == "skipped_by_explicit_flag"
            )
        ),
    }
    checks["all_checks_pass"] = all(checks.values())
    audit = {
        "audited_at": utc_now(),
        "checks": checks,
        "required_plots": list(required_plots),
        "meaningful_longbench_bits": meaningful_bits,
    }
    atomic_write_json(args.output_dir / "completion_audit.json", audit)
    return audit


def report_stage(args: argparse.Namespace) -> None:
    longbench_stage(args)
    plots_stage(args)
    build_report(args)
    audit = completion_audit(args)
    if not audit["checks"]["all_checks_pass"]:
        raise RuntimeError(f"completion audit failed: {audit['checks']}")
    config = read_json(args.output_dir / "training_length_sweep_config.json", {})
    longbench = read_json(args.output_dir / "longbench_e_decision.json", {})
    config["status"] = (
        "complete_with_longbench_skipped"
        if longbench.get("status") == "skipped_by_explicit_flag"
        else "complete"
    )
    config["completion_audit_sha256"] = sha256_file(
        args.output_dir / "completion_audit.json"
    )
    config["updated_at"] = utc_now()
    atomic_write_json(args.output_dir / "training_length_sweep_config.json", config)
    print(f"[{utc_now()}] report and completion audit passed", flush=True)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    args.model = args.model.resolve()
    args.reference_dir = args.reference_dir.resolve()
    args.output_dir = args.output_dir.resolve()
    torch.set_float32_matmul_precision("highest")
    if args.stage == "validate":
        validate_stage(args)
    elif args.stage == "capture":
        capture_stage(args)
    elif args.stage == "smoke":
        smoke_stage(args)
    elif args.stage == "sweep":
        sweep_stage(args)
    elif args.stage == "select":
        select_stage(args)
    elif args.stage == "sanity":
        sanity_stage(args)
    elif args.stage == "longbench":
        longbench_stage(args)
    elif args.stage in {"report", "orchestrate"}:
        report_stage(args)
    else:
        raise ValueError(args.stage)


if __name__ == "__main__":
    main()
