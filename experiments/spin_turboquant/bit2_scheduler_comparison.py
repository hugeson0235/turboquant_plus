"""Run the fixed seed-35, 10,000-step 2-bit scheduler comparison.

The frozen Llama model is used only through previously captured activations.
Each scheduler owns an independent Adam optimizer, while both conditions use
the same Haar rotation and minibatch-index stream.  Per-condition directories
are durable resume units and the requested top-level artifacts are rebuilt
from them.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from .core import (
    attention_distortion_metrics,
    build_random_rotations,
    cayley_rotation,
    codebook_tensor,
    quantize_to_centroids,
)
from .training_length_sweep import (
    atomic_torch_save,
    atomic_write_json,
    cleanup_cuda,
    evaluate_dataset,
    normalized_dataset,
    orthogonality_max_abs,
    read_csv,
    read_json,
    sha256_file,
    sha256_json,
    synchronize,
    tensor_sha256,
    terminal_rotations,
    write_csv,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
SPECIFICATION_PATH = REPOSITORY_ROOT.parent / "bit2_scheduler_10000_step_plan.md"
DEFAULT_SOURCE_DIR = (
    Path(__file__).resolve().parent / "results" / "training_length_sweep"
)
DEFAULT_REFERENCE_DIR = Path(__file__).resolve().parent / "results" / "instruct"
DEFAULT_OUTPUT_DIR = (
    Path(__file__).resolve().parent / "results" / "bit2_scheduler_10000"
)

SCHEDULERS = ("cosine", "exponential")
SEED = 35
BIT_WIDTH = 2
STEPS = 10_000
LOG_INTERVAL = 10
EVALUATION_INTERVAL = 100
RESUME_INTERVAL = 1_000
LEARNING_RATE = 0.005
FINAL_LEARNING_RATE = 0.00025
EXPONENTIAL_GAMMA = (FINAL_LEARNING_RATE / LEARNING_RATE) ** (1.0 / STEPS)
BATCH_TOKENS = 256
EMA_COEFFICIENT = 0.95
TOKENS = 4_096
ATTENTION_SEQUENCE_LENGTH = 1_024
SELECTION_TOLERANCE = 0.001

TRAINING_FIELDS = (
    "scheduler",
    "seed",
    "configured_steps",
    "step",
    "minibatch_mse",
    "last_10_step_mean",
    "ema_mse",
    "learning_rate",
    "gradient_norm",
    "optimizer_elapsed_seconds",
    "evaluation_elapsed_seconds",
    "wall_elapsed_seconds",
)
CHECKPOINT_FIELDS = (
    "scheduler",
    "seed",
    "configured_steps",
    "step",
    "calibration_mse",
    "wikitext_validation_mse",
    "original_scale_validation_mse",
    "generalization_gap",
    "orthogonality_max_abs",
    "rotation_tensor_sha256",
    "optimizer_elapsed_seconds",
    "evaluation_elapsed_seconds",
    "wall_elapsed_seconds",
)
SANITY_FIELDS = (
    "scheduler",
    "seed",
    "selected_step",
    "tinystories_tokens",
    "attention_sequence_length",
    "normalized_key_mse",
    "original_scale_key_mse",
    "attention_probability_kl",
    "attention_logit_mse",
    "attention_output_mse",
    "rotation_tensor_sha256",
    "evaluation_elapsed_seconds",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage",
        choices=("validate", "smoke", "train", "select", "sanity", "report", "orchestrate"),
        default="orchestrate",
    )
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE_DIR)
    parser.add_argument("--reference-dir", type=Path, default=DEFAULT_REFERENCE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args(argv)


def git_output(*arguments: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(REPOSITORY_ROOT), *arguments], text=True
    ).strip()


def environment_metadata(device: torch.device) -> dict[str, Any]:
    result: dict[str, Any] = {
        "timestamp": utc_now(),
        "python": sys.version,
        "python_prefix": sys.prefix,
        "conda_default_env": os.environ.get("CONDA_DEFAULT_ENV"),
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
        result["gpu"] = {
            "name": properties.name,
            "total_memory_bytes": properties.total_memory,
            "capability": [properties.major, properties.minor],
        }
    return result


def scheduler_object(optimizer: torch.optim.Optimizer, name: str, horizon: int):
    if name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=horizon, eta_min=FINAL_LEARNING_RATE
        )
    if name == "exponential":
        gamma = (FINAL_LEARNING_RATE / LEARNING_RATE) ** (1.0 / horizon)
        return torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)
    raise ValueError(f"unknown scheduler: {name}")


def optimizer_seed() -> int:
    return SEED + BIT_WIDTH * 100_000


def run_directory(args: argparse.Namespace, scheduler: str, *, smoke: bool = False) -> Path:
    parent = "smoke" if smoke else "runs"
    return args.output_dir / parent / f"{scheduler}_b2_seed{SEED}"


def rotation_artifact_path(args: argparse.Namespace, scheduler: str) -> Path:
    return args.output_dir / "final_rotation_artifacts" / f"{scheduler}_b2_seed{SEED}.pt"


def fixed_protocol(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "model": str(args.model.resolve()),
        "source_dir": str(args.source_dir.resolve()),
        "reference_dir": str(args.reference_dir.resolve()),
        "bit_width": BIT_WIDTH,
        "seed": SEED,
        "schedulers": list(SCHEDULERS),
        "total_runs": 2,
        "steps": STEPS,
        "optimizer": "Adam",
        "initial_learning_rate": LEARNING_RATE,
        "final_learning_rate": FINAL_LEARNING_RATE,
        "cosine": {"T_max": STEPS, "eta_min": FINAL_LEARNING_RATE},
        "exponential": {"gamma": EXPONENTIAL_GAMMA},
        "batch_tokens_per_head": BATCH_TOKENS,
        "gradient_clip": 1.0,
        "log_interval": LOG_INTERVAL,
        "evaluation_interval": EVALUATION_INTERVAL,
        "calibration_tokens": TOKENS,
        "validation_tokens": TOKENS,
        "domain_tokens": TOKENS,
        "attention_sequence_length": ATTENTION_SEQUENCE_LENGTH,
        "ema_coefficient": EMA_COEFFICIENT,
        "norm_correction": False,
        "value_cache": "BF16",
        "rotation": "layer/KV-head dense Cayley over shared Haar initialization",
        "minibatch_seed": optimizer_seed(),
        "selection_relative_tolerance": SELECTION_TOLERANCE,
    }


def _assert_shape(path: Path, key: str, expected: tuple[int, ...]) -> None:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    actual = tuple(payload[key].shape)
    del payload
    if actual != expected:
        raise RuntimeError(f"{path}:{key} shape {actual}, expected {expected}")


def validate_stage(args: argparse.Namespace) -> dict[str, Any]:
    if Path(sys.prefix).name != "stq":
        raise RuntimeError(f"this experiment must run in conda environment stq, got {sys.prefix}")
    if not SPECIFICATION_PATH.exists():
        raise FileNotFoundError(SPECIFICATION_PATH)
    if not args.model.exists():
        raise FileNotFoundError(args.model)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    source_config_path = args.source_dir / "training_length_sweep_config.json"
    source_config = read_json(source_config_path, {})
    if source_config.get("model", {}).get("path") != str(args.model.resolve()):
        raise RuntimeError("source activation model does not match requested model")
    reference_config = read_json(args.reference_dir / "run_config.json", {})
    reference_model = reference_config.get("arguments", {}).get("model")
    if str(Path(reference_model or "").resolve()) != str(args.model.resolve()):
        raise RuntimeError("calibration activation model does not match requested model")

    calibration_path = args.reference_dir / "activations" / "calibration.pt"
    validation_path = args.source_dir / "activations" / "wikitext.pt"
    domain_path = args.source_dir / "activations" / "tinystories.pt"
    _assert_shape(calibration_path, "k", (32, 8, TOKENS, 128))
    _assert_shape(validation_path, "k", (32, 8, TOKENS, 128))
    _assert_shape(domain_path, "k", (32, TOKENS, 8, 128))
    _assert_shape(domain_path, "q", (32, ATTENTION_SEQUENCE_LENGTH, 32, 128))
    _assert_shape(domain_path, "v", (32, ATTENTION_SEQUENCE_LENGTH, 8, 128))

    activation_records = source_config.get("activations", {})
    expected_hashes = {
        "calibration": activation_records.get("calibration", {}).get("file_sha256"),
        "wikitext": activation_records.get("wikitext", {}).get("file_sha256"),
        "tinystories": activation_records.get("tinystories", {}).get("file_sha256"),
    }
    actual_hashes = {
        "calibration": sha256_file(calibration_path),
        "wikitext": sha256_file(validation_path),
        "tinystories": sha256_file(domain_path),
    }
    if expected_hashes != actual_hashes:
        raise RuntimeError(
            f"activation hash validation failed: expected={expected_hashes}, actual={actual_hashes}"
        )
    codebook_hash = hashlib.sha256(
        codebook_tensor(BIT_WIDTH, 128, device="cpu").numpy().tobytes()
    ).hexdigest()
    if source_config.get("codebook_hashes", {}).get("2") != codebook_hash:
        raise RuntimeError("2-bit TurboQuant codebook hash mismatch")

    protocol = fixed_protocol(args)
    protocol_hash = sha256_json(protocol)
    config_path = args.output_dir / "scheduler_comparison_config.json"
    existing = read_json(config_path, {})
    if existing and existing.get("protocol_sha256") != protocol_hash:
        raise RuntimeError(f"output directory belongs to another protocol: {config_path}")
    implementation_files = {
        "runner": Path(__file__).resolve(),
        "core": Path(__file__).resolve().with_name("core.py"),
        "training_length_runner": Path(__file__).resolve().with_name("training_length_sweep.py"),
        "codebook": REPOSITORY_ROOT / "turboquant" / "codebook.py",
        "rotation": REPOSITORY_ROOT / "turboquant" / "rotation.py",
    }
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
        "model": source_config["model"],
        "activation_hashes": actual_hashes,
        "codebook_hash": codebook_hash,
        "source_config_sha256": sha256_file(source_config_path),
        "implementation_hashes": {
            name: sha256_file(path) for name, path in implementation_files.items()
        },
        "environment": environment_metadata(device),
        "progress": existing.get("progress", {}),
        "runtime_estimate": existing.get("runtime_estimate", {}),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_json(config_path, payload)
    print(f"[{utc_now()}] validated seed-35 two-run protocol {protocol_hash}", flush=True)
    return payload


def update_config(args: argparse.Namespace, **updates: Any) -> None:
    path = args.output_dir / "scheduler_comparison_config.json"
    payload = read_json(path, {})
    payload.update(updates)
    payload["updated_at"] = utc_now()
    atomic_write_json(path, payload)


def load_normalized_inputs(
    args: argparse.Namespace,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    calibration = torch.load(
        args.reference_dir / "activations" / "calibration.pt",
        map_location="cpu",
        weights_only=True,
    )["k"]
    validation = torch.load(
        args.source_dir / "activations" / "wikitext.pt",
        map_location="cpu",
        weights_only=True,
    )["k"]
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


def completed_run(directory: Path, protocol_hash: str, horizon: int) -> dict[str, Any] | None:
    metadata = read_json(directory / "run_config.json", {})
    training = read_csv(directory / "training_curve.csv")
    checkpoints = read_csv(directory / "checkpoint_metrics.csv")
    if (
        metadata.get("status") == "complete"
        and metadata.get("protocol_sha256") == protocol_hash
        and int(metadata.get("horizon", -1)) == horizon
        and len(training) == horizon // LOG_INTERVAL + 1
        and len(checkpoints) == horizon // EVALUATION_INTERVAL + 1
        and int(training[-1]["step"]) == horizon
        and int(checkpoints[-1]["step"]) == horizon
    ):
        return metadata
    return None


def _evaluate_checkpoint(
    scheduler_name: str,
    horizon: int,
    step: int,
    rotations: torch.Tensor,
    centroids: torch.Tensor,
    inputs: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    optimizer_elapsed: float,
    evaluation_elapsed_before: float,
    wall_elapsed: float,
) -> tuple[dict[str, Any], float]:
    device = rotations.device
    synchronize(device)
    started = time.perf_counter()
    calibration = evaluate_dataset(inputs[0], inputs[1], rotations, centroids)
    validation = evaluate_dataset(inputs[2], inputs[3], rotations, centroids)
    orthogonality = orthogonality_max_abs(rotations)
    synchronize(device)
    evaluation_elapsed = evaluation_elapsed_before + time.perf_counter() - started
    row = {
        "scheduler": scheduler_name,
        "seed": SEED,
        "configured_steps": horizon,
        "step": step,
        "calibration_mse": calibration["normalized_key_mse"],
        "wikitext_validation_mse": validation["normalized_key_mse"],
        "original_scale_validation_mse": validation["original_scale_key_mse"],
        "generalization_gap": validation["normalized_key_mse"] - calibration["normalized_key_mse"],
        "orthogonality_max_abs": orthogonality,
        "rotation_tensor_sha256": tensor_sha256(rotations),
        "optimizer_elapsed_seconds": optimizer_elapsed,
        "evaluation_elapsed_seconds": evaluation_elapsed,
        "wall_elapsed_seconds": wall_elapsed,
    }
    return row, evaluation_elapsed


def execute_run(
    args: argparse.Namespace,
    scheduler_name: str,
    horizon: int,
    inputs: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    *,
    smoke: bool,
    progress_callback=None,
) -> dict[str, Any]:
    config = read_json(args.output_dir / "scheduler_comparison_config.json", {})
    protocol_hash = config["protocol_sha256"]
    directory = run_directory(args, scheduler_name, smoke=smoke)
    existing = completed_run(directory, protocol_hash, horizon)
    if existing is not None:
        print(f"[{utc_now()}] reuse complete {directory.name} horizon={horizon}", flush=True)
        return existing
    directory.mkdir(parents=True, exist_ok=True)

    device = inputs[0].device
    initial = build_random_rotations(32, 8, 128, SEED).reshape(256, 128, 128).to(device)
    centroids = codebook_tensor(BIT_WIDTH, 128, device=device)
    parameters = torch.nn.Parameter(torch.zeros_like(initial))
    optimizer = torch.optim.Adam([parameters], lr=LEARNING_RATE)
    lr_scheduler = scheduler_object(optimizer, scheduler_name, horizon)
    generator = torch.Generator(device=device)
    generator.manual_seed(optimizer_seed())
    token_count = inputs[0].shape[1]

    training_rows: list[dict[str, Any]] = []
    checkpoint_rows: list[dict[str, Any]] = []
    losses: list[float] = []
    ema: float | None = None
    optimizer_elapsed = 0.0
    evaluation_elapsed = 0.0
    start_step = 0
    prefix_digest = hashlib.sha256()
    prefix_hash: str | None = None
    wall_offset = 0.0
    resume_path = directory / "resume_checkpoint.pt"
    if resume_path.exists():
        state = torch.load(resume_path, map_location=device, weights_only=False)
        if state.get("protocol_sha256") == protocol_hash and state.get("horizon") == horizon:
            parameters.data.copy_(state["parameters"])
            optimizer.load_state_dict(state["optimizer"])
            lr_scheduler.load_state_dict(state["lr_scheduler"])
            generator.set_state(state["generator_state"])
            training_rows = state["training_rows"]
            checkpoint_rows = state["checkpoint_rows"]
            losses = state["recent_losses"]
            ema = state["ema"]
            optimizer_elapsed = state["optimizer_elapsed"]
            evaluation_elapsed = state["evaluation_elapsed"]
            start_step = state["step"]
            prefix_hash = state["first_100_minibatch_indices_sha256"]
            wall_offset = state["wall_elapsed_seconds"]
            print(f"[{utc_now()}] resume {directory.name} at step {start_step}", flush=True)
        del state

    started_wall = time.perf_counter()
    if start_step == 0:
        checkpoint, evaluation_elapsed = _evaluate_checkpoint(
            scheduler_name,
            horizon,
            0,
            initial,
            centroids,
            inputs,
            optimizer_elapsed,
            evaluation_elapsed,
            0.0,
        )
        checkpoint_rows.append(checkpoint)
        training_rows.append(
            {
                "scheduler": scheduler_name,
                "seed": SEED,
                "configured_steps": horizon,
                "step": 0,
                "minibatch_mse": "",
                "last_10_step_mean": "",
                "ema_mse": "",
                "learning_rate": LEARNING_RATE,
                "gradient_norm": "",
                "optimizer_elapsed_seconds": optimizer_elapsed,
                "evaluation_elapsed_seconds": evaluation_elapsed,
                "wall_elapsed_seconds": 0.0,
            }
        )

    for step in range(start_step + 1, horizon + 1):
        indices = torch.randint(
            token_count,
            (BATCH_TOKENS,),
            generator=generator,
            device=device,
        )
        if step <= 100:
            prefix_digest.update(indices.detach().cpu().contiguous().numpy().tobytes())
        batch = inputs[0][:, indices]
        synchronize(device)
        optimizer_started = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        rotations = cayley_rotation(parameters, initial)
        rotated = torch.matmul(batch, rotations.transpose(-1, -2))
        targets = quantize_to_centroids(rotated.detach(), centroids)
        loss = torch.mean((rotated - targets) ** 2)
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_([parameters], max_norm=1.0)
        optimizer.step()
        lr_scheduler.step()
        synchronize(device)
        optimizer_elapsed += time.perf_counter() - optimizer_started
        loss_value = float(loss.detach())
        gradient_value = float(gradient_norm)
        losses.append(loss_value)
        losses = losses[-LOG_INTERVAL:]
        ema = loss_value if ema is None else EMA_COEFFICIENT * ema + (1 - EMA_COEFFICIENT) * loss_value
        if step == 100:
            prefix_hash = prefix_digest.hexdigest()
        del indices, batch, rotations, rotated, targets, loss, gradient_norm

        wall_elapsed = wall_offset + time.perf_counter() - started_wall
        if step % LOG_INTERVAL == 0:
            training_rows.append(
                {
                    "scheduler": scheduler_name,
                    "seed": SEED,
                    "configured_steps": horizon,
                    "step": step,
                    "minibatch_mse": loss_value,
                    "last_10_step_mean": float(np.mean(losses)),
                    "ema_mse": ema,
                    "learning_rate": optimizer.param_groups[0]["lr"],
                    "gradient_norm": gradient_value,
                    "optimizer_elapsed_seconds": optimizer_elapsed,
                    "evaluation_elapsed_seconds": evaluation_elapsed,
                    "wall_elapsed_seconds": wall_elapsed,
                }
            )
        if step % EVALUATION_INTERVAL == 0:
            measured = terminal_rotations(parameters, initial)
            checkpoint, evaluation_elapsed = _evaluate_checkpoint(
                scheduler_name,
                horizon,
                step,
                measured,
                centroids,
                inputs,
                optimizer_elapsed,
                evaluation_elapsed,
                wall_elapsed,
            )
            checkpoint_rows.append(checkpoint)
            training_rows[-1]["evaluation_elapsed_seconds"] = evaluation_elapsed
            write_csv(directory / "training_curve.csv", training_rows, fieldnames=TRAINING_FIELDS)
            write_csv(directory / "checkpoint_metrics.csv", checkpoint_rows, fieldnames=CHECKPOINT_FIELDS)
            running = {
                "status": "running",
                "scheduler": scheduler_name,
                "seed": SEED,
                "horizon": horizon,
                "protocol_sha256": protocol_hash,
                "completed_step": step,
                "first_100_minibatch_indices_sha256": prefix_hash,
                "initial_rotation_tensor_sha256": tensor_sha256(initial),
                "latest": checkpoint,
                "updated_at": utc_now(),
            }
            atomic_write_json(directory / "run_config.json", running)
            if step % RESUME_INTERVAL == 0 and step < horizon:
                atomic_torch_save(
                    {
                        "protocol_sha256": protocol_hash,
                        "horizon": horizon,
                        "step": step,
                        "parameters": parameters.detach(),
                        "optimizer": optimizer.state_dict(),
                        "lr_scheduler": lr_scheduler.state_dict(),
                        "generator_state": generator.get_state(),
                        "training_rows": training_rows,
                        "checkpoint_rows": checkpoint_rows,
                        "recent_losses": losses,
                        "ema": ema,
                        "optimizer_elapsed": optimizer_elapsed,
                        "evaluation_elapsed": evaluation_elapsed,
                        "wall_elapsed_seconds": wall_elapsed,
                        "first_100_minibatch_indices_sha256": prefix_hash,
                    },
                    resume_path,
                )
            if progress_callback is not None:
                progress_callback(step, horizon, checkpoint, wall_elapsed)
            elif step == horizon or step % 1_000 == 0:
                remaining = wall_elapsed * (horizon - step) / max(step, 1)
                print(
                    f"[{utc_now()}] {directory.name} {step}/{horizon} "
                    f"val={checkpoint['wikitext_validation_mse']:.9g} "
                    f"lr={optimizer.param_groups[0]['lr']:.7g} ETA={remaining / 60:.1f} min",
                    flush=True,
                )
            del measured

    final = checkpoint_rows[-1]
    metadata = {
        "status": "complete",
        "completed_at": utc_now(),
        "scheduler": scheduler_name,
        "seed": SEED,
        "bit_width": BIT_WIDTH,
        "horizon": horizon,
        "protocol_sha256": protocol_hash,
        "optimizer_seed": optimizer_seed(),
        "first_100_minibatch_indices_sha256": prefix_hash,
        "initial_rotation_tensor_sha256": tensor_sha256(initial),
        "scheduler_config": (
            {"name": "CosineAnnealingLR", "T_max": horizon, "eta_min": FINAL_LEARNING_RATE}
            if scheduler_name == "cosine"
            else {
                "name": "ExponentialLR",
                "gamma": (FINAL_LEARNING_RATE / LEARNING_RATE) ** (1 / horizon),
            }
        ),
        "row_counts": {
            "training_curve": len(training_rows),
            "checkpoint_metrics": len(checkpoint_rows),
        },
        "terminal": final,
    }
    atomic_write_json(directory / "run_config.json", metadata)
    resume_path.unlink(missing_ok=True)
    del parameters, optimizer, lr_scheduler, generator, centroids, initial
    cleanup_cuda()
    return metadata


def smoke_stage(args: argparse.Namespace) -> dict[str, Any]:
    validate_stage(args)
    inputs = load_normalized_inputs(args)
    records: dict[str, Any] = {}
    for scheduler_name in SCHEDULERS:
        records[scheduler_name] = execute_run(
            args, scheduler_name, 100, inputs, smoke=True
        )
    cosine = records["cosine"]
    exponential = records["exponential"]
    if cosine["first_100_minibatch_indices_sha256"] != exponential["first_100_minibatch_indices_sha256"]:
        raise RuntimeError("smoke minibatch streams differ between schedulers")
    if cosine["initial_rotation_tensor_sha256"] != exponential["initial_rotation_tensor_sha256"]:
        raise RuntimeError("smoke Haar initializations differ between schedulers")
    for scheduler_name in SCHEDULERS:
        rows = read_csv(run_directory(args, scheduler_name, smoke=True) / "training_curve.csv")
        checkpoints = read_csv(run_directory(args, scheduler_name, smoke=True) / "checkpoint_metrics.csv")
        if [int(row["step"]) for row in rows] != list(range(0, 101, 10)):
            raise RuntimeError(f"{scheduler_name} smoke logging schedule is incomplete")
        if [int(row["step"]) for row in checkpoints] != [0, 100]:
            raise RuntimeError(f"{scheduler_name} smoke evaluation schedule is incomplete")
        if not math.isclose(float(rows[0]["learning_rate"]), LEARNING_RATE, abs_tol=1e-12):
            raise RuntimeError(f"{scheduler_name} smoke initial LR mismatch")
        if not math.isclose(float(rows[-1]["learning_rate"]), FINAL_LEARNING_RATE, abs_tol=1e-10):
            raise RuntimeError(f"{scheduler_name} smoke final LR mismatch")

    training_eta = 0.0
    estimates: dict[str, Any] = {}
    for scheduler_name, metadata in records.items():
        terminal = metadata["terminal"]
        optimizer_estimate = float(terminal["optimizer_elapsed_seconds"]) * (STEPS / 100)
        evaluation_per_checkpoint = float(terminal["evaluation_elapsed_seconds"]) / 2
        evaluation_estimate = evaluation_per_checkpoint * (STEPS // EVALUATION_INTERVAL + 1)
        estimates[scheduler_name] = {
            "optimizer_seconds": optimizer_estimate,
            "checkpoint_evaluation_seconds": evaluation_estimate,
            "total_seconds": optimizer_estimate + evaluation_estimate,
        }
        training_eta += optimizer_estimate + evaluation_estimate
    estimate = {
        "measured_at": utc_now(),
        "basis": "paired 100-step smoke runs",
        "conditions": estimates,
        "two_run_training_seconds": training_eta,
        "two_run_training_minutes": training_eta / 60,
        "excludes_checkpoint_replay_tinystories_and_plotting": True,
    }
    update_config(args, status="smoke_complete", runtime_estimate=estimate)
    del inputs
    cleanup_cuda()
    print(
        f"[{utc_now()}] paired smoke passed; measured two-run training ETA={training_eta / 60:.1f} min",
        flush=True,
    )
    return estimate


def aggregate_training(args: argparse.Namespace, *, require_complete: bool = True) -> None:
    protocol_hash = read_json(args.output_dir / "scheduler_comparison_config.json", {})["protocol_sha256"]
    training_rows: list[dict[str, Any]] = []
    checkpoint_rows: list[dict[str, Any]] = []
    for scheduler_name in SCHEDULERS:
        directory = run_directory(args, scheduler_name)
        if require_complete and completed_run(directory, protocol_hash, STEPS) is None:
            raise RuntimeError(f"incomplete condition: {directory}")
        training_rows.extend(read_csv(directory / "training_curve.csv"))
        checkpoint_rows.extend(read_csv(directory / "checkpoint_metrics.csv"))
    training_rows.sort(key=lambda row: (SCHEDULERS.index(row["scheduler"]), int(row["step"])))
    checkpoint_rows.sort(key=lambda row: (SCHEDULERS.index(row["scheduler"]), int(row["step"])))
    write_csv(args.output_dir / "training_curve.csv", training_rows, fieldnames=TRAINING_FIELDS)
    write_csv(args.output_dir / "checkpoint_metrics.csv", checkpoint_rows, fieldnames=CHECKPOINT_FIELDS)


def train_stage(args: argparse.Namespace) -> None:
    estimate = smoke_stage(args)
    inputs = load_normalized_inputs(args)
    protocol_hash = read_json(args.output_dir / "scheduler_comparison_config.json", {})["protocol_sha256"]
    train_started = time.perf_counter()

    for condition_index, scheduler_name in enumerate(SCHEDULERS):
        if completed_run(run_directory(args, scheduler_name), protocol_hash, STEPS) is not None:
            print(f"[{utc_now()}] reuse completed full condition {scheduler_name}", flush=True)
            continue

        def progress(step: int, horizon: int, checkpoint: dict[str, Any], run_elapsed: float) -> None:
            completed_units = condition_index * horizon + step
            elapsed = time.perf_counter() - train_started
            remaining = elapsed * (len(SCHEDULERS) * horizon - completed_units) / max(completed_units, 1)
            progress_payload = {
                "phase": "training",
                "condition": scheduler_name,
                "condition_index": condition_index + 1,
                "total_conditions": len(SCHEDULERS),
                "step": step,
                "steps": horizon,
                "validation_mse": checkpoint["wikitext_validation_mse"],
                "training_eta_seconds": remaining,
                "updated_at": utc_now(),
            }
            if step % 1_000 == 0 or step == horizon:
                update_config(args, status="training", progress=progress_payload)
                print(
                    f"[{utc_now()}] {scheduler_name}/seed{SEED} {step}/{horizon} "
                    f"val={checkpoint['wikitext_validation_mse']:.9g} "
                    f"training ETA={remaining / 60:.1f} min",
                    flush=True,
                )

        execute_run(
            args,
            scheduler_name,
            STEPS,
            inputs,
            smoke=False,
            progress_callback=progress,
        )
    aggregate_training(args)
    actual_training_seconds = sum(
        float(
            read_json(run_directory(args, scheduler_name) / "run_config.json", {})[
                "terminal"
            ]["wall_elapsed_seconds"]
        )
        for scheduler_name in SCHEDULERS
    )
    update_config(
        args,
        status="training_complete",
        training_runtime={
            "two_run_wall_seconds": actual_training_seconds,
            "two_run_wall_minutes": actual_training_seconds / 60,
            "measurement": "sum of the two per-condition wall clocks",
        },
        progress={
            "phase": "training_complete",
            "completed_conditions": 2,
            "total_conditions": 2,
            "elapsed_seconds": time.perf_counter() - train_started,
            "smoke_estimated_seconds": estimate["two_run_training_seconds"],
            "updated_at": utc_now(),
        },
    )
    del inputs
    cleanup_cuda()
    print(f"[{utc_now()}] both 10,000-step conditions complete", flush=True)


def select_step(rows: list[dict[str, str]]) -> dict[str, Any]:
    ordered = sorted(rows, key=lambda row: int(row["step"]))
    values = [float(row["wikitext_validation_mse"]) for row in ordered]
    minimum = min(values)
    minimum_row = ordered[int(np.argmin(values))]
    eligible = [
        row
        for row in ordered
        if float(row["wikitext_validation_mse"]) <= minimum * (1 + SELECTION_TOLERANCE)
    ]
    selected = eligible[0]
    return {
        "minimum_step": int(minimum_row["step"]),
        "minimum_validation_mse": minimum,
        "selected_step": int(selected["step"]),
        "selected_validation_mse": float(selected["wikitext_validation_mse"]),
        "near_minimum_relative_tolerance": SELECTION_TOLERANCE,
        "eligible_steps": [int(row["step"]) for row in eligible],
        "checkpoint_rotation_tensor_sha256": selected["rotation_tensor_sha256"],
    }


def rerun_selected(
    args: argparse.Namespace,
    scheduler_name: str,
    selected_step: int,
    calibration_normalized: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, float]:
    device = calibration_normalized.device
    initial = build_random_rotations(32, 8, 128, SEED).reshape(256, 128, 128).to(device)
    centroids = codebook_tensor(BIT_WIDTH, 128, device=device)
    parameters = torch.nn.Parameter(torch.zeros_like(initial))
    optimizer = torch.optim.Adam([parameters], lr=LEARNING_RATE)
    lr_scheduler = scheduler_object(optimizer, scheduler_name, STEPS)
    generator = torch.Generator(device=device)
    generator.manual_seed(optimizer_seed())
    started = time.perf_counter()
    for step in range(1, selected_step + 1):
        indices = torch.randint(
            calibration_normalized.shape[1],
            (BATCH_TOKENS,),
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
        lr_scheduler.step()
        del indices, batch, rotations, rotated, targets, loss
        if step % 1_000 == 0 or step == selected_step:
            elapsed = time.perf_counter() - started
            eta = elapsed * (selected_step - step) / max(step, 1)
            print(
                f"[{utc_now()}] replay {scheduler_name}/seed{SEED} {step}/{selected_step} ETA={eta / 60:.1f} min",
                flush=True,
            )
    learned = terminal_rotations(parameters, initial)
    elapsed = time.perf_counter() - started
    random_cpu = initial.reshape(32, 8, 128, 128).cpu()
    learned_cpu = learned.reshape(32, 8, 128, 128).cpu()
    del parameters, optimizer, lr_scheduler, generator, centroids, initial, learned
    cleanup_cuda()
    return random_cpu, learned_cpu, elapsed


def select_stage(args: argparse.Namespace) -> dict[str, Any]:
    train_stage(args)
    checkpoint_rows = read_csv(args.output_dir / "checkpoint_metrics.csv")
    decisions = {
        scheduler_name: select_step(
            [row for row in checkpoint_rows if row["scheduler"] == scheduler_name]
        )
        for scheduler_name in SCHEDULERS
    }
    inputs = load_normalized_inputs(args)
    centroids = codebook_tensor(BIT_WIDTH, 128, device=inputs[0].device)
    args.output_dir.joinpath("final_rotation_artifacts").mkdir(parents=True, exist_ok=True)
    for scheduler_name, decision in decisions.items():
        path = rotation_artifact_path(args, scheduler_name)
        artifact_complete = False
        if path.exists():
            existing = torch.load(path, map_location="cpu", weights_only=True)
            artifact_complete = (
                existing.get("scheduler") == scheduler_name
                and existing.get("seed") == SEED
                and existing.get("selected_step") == decision["selected_step"]
            )
            del existing
        if not artifact_complete:
            random_rotation, learned_rotation, replay_elapsed = rerun_selected(
                args, scheduler_name, decision["selected_step"], inputs[0]
            )
            learned_hash = tensor_sha256(learned_rotation.reshape(256, 128, 128))
            if learned_hash != decision["checkpoint_rotation_tensor_sha256"]:
                raise RuntimeError(
                    f"deterministic replay hash mismatch for {scheduler_name}: "
                    f"{learned_hash} != {decision['checkpoint_rotation_tensor_sha256']}"
                )
            rotations = learned_rotation.reshape(256, 128, 128).to(inputs[0].device)
            calibration = evaluate_dataset(inputs[0], inputs[1], rotations, centroids)
            validation = evaluate_dataset(inputs[2], inputs[3], rotations, centroids)
            if not math.isclose(
                validation["normalized_key_mse"],
                decision["selected_validation_mse"],
                rel_tol=0,
                abs_tol=1e-12,
            ):
                raise RuntimeError(f"selected checkpoint metric mismatch for {scheduler_name}")
            artifact = {
                "random": random_rotation,
                "learned": learned_rotation,
                "scheduler": scheduler_name,
                "seed": SEED,
                "bit_width": BIT_WIDTH,
                "configured_steps": STEPS,
                "selected_step": decision["selected_step"],
                "model": str(args.model),
                "norm_correction": False,
                "optimizer_seed": optimizer_seed(),
                "codebook_hash": read_json(
                    args.output_dir / "scheduler_comparison_config.json", {}
                )["codebook_hash"],
                "replay_elapsed_seconds": replay_elapsed,
                "calibration_mse": calibration["normalized_key_mse"],
                "wikitext_validation_mse": validation["normalized_key_mse"],
                "rotation_tensor_sha256": learned_hash,
            }
            atomic_torch_save(artifact, path)
            del random_rotation, learned_rotation, rotations, calibration, validation, artifact
        stored = torch.load(path, map_location="cpu", weights_only=True)
        decision["artifact_path"] = str(path)
        decision["artifact_file_sha256"] = sha256_file(path)
        decision["deterministic_rerun_match"] = (
            stored["rotation_tensor_sha256"] == decision["checkpoint_rotation_tensor_sha256"]
        )
        decision["replay_elapsed_seconds"] = stored["replay_elapsed_seconds"]
        del stored

    winner = min(SCHEDULERS, key=lambda name: decisions[name]["selected_validation_mse"])
    selection = {
        "selected_at": utc_now(),
        "criterion": "lowest seed-35 WikiText validation normalized key MSE",
        "tie_rule": "earliest checkpoint within 0.1% of each scheduler minimum",
        "scheduler_decisions": decisions,
        "winner": winner,
        "winner_validation_mse": decisions[winner]["selected_validation_mse"],
    }
    atomic_write_json(args.output_dir / "selected_checkpoints.json", selection)
    update_config(
        args,
        status="selection_complete",
        selection_sha256=sha256_file(args.output_dir / "selected_checkpoints.json"),
        progress={"phase": "selection_complete", "winner": winner, "updated_at": utc_now()},
    )
    del inputs, centroids
    cleanup_cuda()
    print(f"[{utc_now()}] selected checkpoints; WikiText winner={winner}", flush=True)
    return selection


def sanity_stage(args: argparse.Namespace) -> list[dict[str, Any]]:
    selection = select_stage(args)
    sanity_path = args.output_dir / "sanity_metrics.csv"
    existing = read_csv(sanity_path)
    rows = [row for row in existing if row.get("scheduler") in SCHEDULERS]
    completed_names = {row["scheduler"] for row in rows}
    device = torch.device(args.device)
    tiny = torch.load(
        args.source_dir / "activations" / "tinystories.pt",
        map_location="cpu",
        weights_only=True,
    )
    tiny_keys = tiny["k"].permute(0, 2, 1, 3).contiguous()
    tiny_normalized, tiny_norm_squared = normalized_dataset(tiny_keys, device)
    centroids = codebook_tensor(BIT_WIDTH, 128, device=device)
    del tiny_keys
    for scheduler_name in SCHEDULERS:
        if scheduler_name in completed_names:
            continue
        artifact = torch.load(
            rotation_artifact_path(args, scheduler_name),
            map_location="cpu",
            weights_only=True,
        )
        rotations_cpu = artifact["learned"]
        rotations = rotations_cpu.reshape(256, 128, 128).to(device)
        started = time.perf_counter()
        reconstruction = evaluate_dataset(
            tiny_normalized, tiny_norm_squared, rotations, centroids
        )
        attention = attention_distortion_metrics(
            tiny["q"],
            tiny["k"][:, :ATTENTION_SEQUENCE_LENGTH],
            tiny["v"],
            tiny["cos"],
            tiny["sin"],
            rotations.reshape(32, 8, 128, 128),
            centroids,
            sequence_length=ATTENTION_SEQUENCE_LENGTH,
            norm_correction=False,
        )
        synchronize(device)
        elapsed = time.perf_counter() - started
        row = {
            "scheduler": scheduler_name,
            "seed": SEED,
            "selected_step": selection["scheduler_decisions"][scheduler_name]["selected_step"],
            "tinystories_tokens": TOKENS,
            "attention_sequence_length": ATTENTION_SEQUENCE_LENGTH,
            "normalized_key_mse": reconstruction["normalized_key_mse"],
            "original_scale_key_mse": reconstruction["original_scale_key_mse"],
            "attention_probability_kl": attention["attention_probability_kl"],
            "attention_logit_mse": attention["attention_logit_mse"],
            "attention_output_mse": attention["attention_output_mse"],
            "rotation_tensor_sha256": tensor_sha256(rotations_cpu.reshape(256, 128, 128)),
            "evaluation_elapsed_seconds": elapsed,
        }
        rows.append(row)
        rows.sort(key=lambda value: SCHEDULERS.index(value["scheduler"]))
        write_csv(sanity_path, rows, fieldnames=SANITY_FIELDS)
        print(
            f"[{utc_now()}] TinyStories {scheduler_name} complete "
            f"key_mse={row['normalized_key_mse']:.9g}",
            flush=True,
        )
        del artifact, rotations_cpu, rotations, reconstruction, attention
        cleanup_cuda()
    update_config(
        args,
        status="sanity_complete",
        sanity_metrics_sha256=sha256_file(sanity_path),
        progress={"phase": "sanity_complete", "updated_at": utc_now()},
    )
    del tiny, tiny_normalized, tiny_norm_squared, centroids
    cleanup_cuda()
    return rows


def plots_stage(args: argparse.Namespace) -> None:
    training = read_csv(args.output_dir / "training_curve.csv")
    checkpoints = read_csv(args.output_dir / "checkpoint_metrics.csv")
    plot_dir = args.output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    colors = {"cosine": "tab:blue", "exponential": "tab:orange"}

    fig, axis = plt.subplots(figsize=(8, 5))
    for name in SCHEDULERS:
        rows = [row for row in training if row["scheduler"] == name]
        axis.plot([int(row["step"]) for row in rows], [float(row["learning_rate"]) for row in rows], label=name, color=colors[name])
    axis.set(xlabel="Step", ylabel="Learning rate", title="Learning-rate schedules")
    axis.grid(alpha=0.25)
    axis.legend()
    fig.tight_layout()
    fig.savefig(plot_dir / "learning_rate.png", dpi=180)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(9, 5.5))
    for name in SCHEDULERS:
        rows = [row for row in checkpoints if row["scheduler"] == name]
        steps = [int(row["step"]) for row in rows]
        axis.plot(steps, [float(row["wikitext_validation_mse"]) for row in rows], label=f"{name} validation", color=colors[name])
        axis.plot(steps, [float(row["calibration_mse"]) for row in rows], label=f"{name} calibration", color=colors[name], linestyle="--", alpha=0.75)
    axis.set(xlabel="Step", ylabel="Normalized key MSE", title=f"Seed {SEED} calibration and WikiText validation")
    axis.grid(alpha=0.25)
    axis.legend()
    fig.tight_layout()
    fig.savefig(plot_dir / "validation_mse.png", dpi=180)
    plt.close(fig)

    by_name = {
        name: {int(row["step"]): float(row["wikitext_validation_mse"]) for row in checkpoints if row["scheduler"] == name}
        for name in SCHEDULERS
    }
    steps = sorted(set(by_name["cosine"]) & set(by_name["exponential"]))
    differences = [by_name["cosine"][step] - by_name["exponential"][step] for step in steps]
    fig, axis = plt.subplots(figsize=(8, 5))
    axis.axhline(0, color="black", linewidth=1)
    axis.plot(steps, differences, color="tab:purple")
    axis.set(xlabel="Step", ylabel="Cosine MSE - exponential MSE", title="Paired WikiText validation difference")
    axis.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(plot_dir / "scheduler_difference.png", dpi=180)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(8, 5))
    for name in SCHEDULERS:
        rows = [row for row in checkpoints if row["scheduler"] == name]
        axis.plot([int(row["step"]) for row in rows], [float(row["generalization_gap"]) for row in rows], label=name, color=colors[name])
    axis.axhline(0, color="black", linewidth=1)
    axis.set(xlabel="Step", ylabel="Validation MSE - calibration MSE", title="Calibration-validation generalization gap")
    axis.grid(alpha=0.25)
    axis.legend()
    fig.tight_layout()
    fig.savefig(plot_dir / "generalization_gap.png", dpi=180)
    plt.close(fig)


def build_report(args: argparse.Namespace, selection: dict[str, Any], sanity: list[dict[str, Any]]) -> None:
    checkpoints = read_csv(args.output_dir / "checkpoint_metrics.csv")
    config = read_json(args.output_dir / "scheduler_comparison_config.json", {})
    sanity_by_name = {row["scheduler"]: row for row in sanity}
    lines = [
        "# 2-bit 10,000-step scheduler comparison",
        "",
        f"Completed both requested conditions with frozen `{args.model.name}` activations: cosine/seed {SEED} and exponential/seed {SEED}. No other seeds were run.",
        "",
        "## Protocol",
        "",
        f"- Initial/final LR: {LEARNING_RATE} / {FINAL_LEARNING_RATE}",
        f"- Exponential gamma: {EXPONENTIAL_GAMMA:.12g}",
        f"- Steps per condition: {STEPS}",
        f"- Full calibration/validation interval: {EVALUATION_INTERVAL}",
        f"- Minibatch/log interval: {BATCH_TOKENS} tokens/head, every {LOG_INTERVAL} steps",
        "- Both schedulers used the same seed-35 Haar rotation and minibatch-index stream.",
        "",
        "## Selection",
        "",
        "| Scheduler | Absolute minimum step | Selected step | Selected WikiText validation MSE |",
        "|---|---:|---:|---:|",
    ]
    for name in SCHEDULERS:
        decision = selection["scheduler_decisions"][name]
        lines.append(
            f"| {name} | {decision['minimum_step']} | {decision['selected_step']} | {decision['selected_validation_mse']:.9g} |"
        )
    lines.extend(
        [
            "",
            f"Primary WikiText winner: **{selection['winner']}**.",
            "",
            "## TinyStories domain sanity check",
            "",
            "| Scheduler | Normalized key MSE | Original-scale key MSE | Attention KL | Logit MSE | Output MSE |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for name in SCHEDULERS:
        row = sanity_by_name[name]
        lines.append(
            f"| {name} | {float(row['normalized_key_mse']):.9g} | {float(row['original_scale_key_mse']):.9g} | "
            f"{float(row['attention_probability_kl']):.9g} | {float(row['attention_logit_mse']):.9g} | {float(row['attention_output_mse']):.9g} |"
        )
    lines.extend(
        [
            "",
            "TinyStories is a domain-generalization sanity check only; it does not override the WikiText selection criterion.",
            "",
            "## Runtime",
            "",
            f"- Smoke-estimated two-run training: {config.get('runtime_estimate', {}).get('two_run_training_minutes', float('nan')):.2f} minutes",
            f"- Actual two-run training: {config.get('training_runtime', {}).get('two_run_wall_minutes', float('nan')):.2f} minutes",
            "",
            "LongBench-E, perplexity, and retrieval were not run, as required by the specification.",
            "",
            "Raw per-condition evidence is under `runs/`; deterministic selected rotations are under `final_rotation_artifacts/`.",
        ]
    )
    (args.output_dir / "report.md").write_text("\n".join(lines) + "\n")


def completion_audit(args: argparse.Namespace) -> dict[str, Any]:
    config = read_json(args.output_dir / "scheduler_comparison_config.json", {})
    protocol_hash = config.get("protocol_sha256", "")
    training = read_csv(args.output_dir / "training_curve.csv")
    checkpoints = read_csv(args.output_dir / "checkpoint_metrics.csv")
    sanity = read_csv(args.output_dir / "sanity_metrics.csv")
    selection = read_json(args.output_dir / "selected_checkpoints.json", {})
    run_metadata = {
        name: read_json(run_directory(args, name) / "run_config.json", {})
        for name in SCHEDULERS
    }
    required = (
        "scheduler_comparison_config.json",
        "training_curve.csv",
        "checkpoint_metrics.csv",
        "selected_checkpoints.json",
        "sanity_metrics.csv",
        "plots/learning_rate.png",
        "plots/validation_mse.png",
        "plots/scheduler_difference.png",
        "plots/generalization_gap.png",
        "report.md",
    )
    checks = {
        "specification_seed_is_35": config.get("protocol", {}).get("seed") == 35,
        "exactly_two_schedulers": config.get("protocol", {}).get("schedulers") == list(SCHEDULERS),
        "both_full_runs_complete": all(completed_run(run_directory(args, name), protocol_hash, STEPS) is not None for name in SCHEDULERS),
        "no_other_full_run_directories": sorted(path.name for path in (args.output_dir / "runs").iterdir() if path.is_dir()) == [f"cosine_b2_seed{SEED}", f"exponential_b2_seed{SEED}"],
        "paired_initial_rotation": len({metadata.get("initial_rotation_tensor_sha256") for metadata in run_metadata.values()}) == 1,
        "paired_minibatch_stream": len({metadata.get("first_100_minibatch_indices_sha256") for metadata in run_metadata.values()}) == 1,
        "training_rows_2002": len(training) == 2 * (STEPS // LOG_INTERVAL + 1),
        "checkpoint_rows_202": len(checkpoints) == 2 * (STEPS // EVALUATION_INTERVAL + 1),
        "both_terminal_steps_10000": all(int(metadata.get("terminal", {}).get("step", -1)) == STEPS for metadata in run_metadata.values()),
        "both_lr_endpoints_exact": all(
            math.isclose(float([row for row in training if row["scheduler"] == name][0]["learning_rate"]), LEARNING_RATE, abs_tol=1e-12)
            and math.isclose(float([row for row in training if row["scheduler"] == name][-1]["learning_rate"]), FINAL_LEARNING_RATE, abs_tol=1e-10)
            for name in SCHEDULERS
        ),
        "finite_training_and_checkpoint_metrics": all(
            math.isfinite(float(row[field]))
            for row in training
            for field in ("learning_rate", "optimizer_elapsed_seconds", "evaluation_elapsed_seconds")
        ) and all(
            math.isfinite(float(row[field]))
            for row in checkpoints
            for field in ("calibration_mse", "wikitext_validation_mse", "orthogonality_max_abs")
        ),
        "selection_has_both_schedulers": set(selection.get("scheduler_decisions", {})) == set(SCHEDULERS),
        "two_deterministic_artifacts": all(rotation_artifact_path(args, name).exists() for name in SCHEDULERS),
        "all_replay_hashes_match": all(decision.get("deterministic_rerun_match") for decision in selection.get("scheduler_decisions", {}).values()),
        "two_sanity_rows": len(sanity) == 2 and {row["scheduler"] for row in sanity} == set(SCHEDULERS),
        "training_runtime_recorded": math.isfinite(
            float(config.get("training_runtime", {}).get("two_run_wall_seconds", float("nan")))
        ),
        "all_required_artifacts_exist": all((args.output_dir / path).exists() for path in required),
    }
    checks["all_checks_pass"] = all(checks.values())
    result = {"audited_at": utc_now(), "checks": checks, "required_artifacts": list(required)}
    atomic_write_json(args.output_dir / "completion_audit.json", result)
    return result


def report_stage(args: argparse.Namespace) -> None:
    sanity = sanity_stage(args)
    selection = read_json(args.output_dir / "selected_checkpoints.json", {})
    plots_stage(args)
    build_report(args, selection, sanity)
    audit = completion_audit(args)
    if not audit["checks"]["all_checks_pass"]:
        raise RuntimeError(f"completion audit failed: {audit['checks']}")
    update_config(
        args,
        status="complete",
        completion_audit_sha256=sha256_file(args.output_dir / "completion_audit.json"),
        progress={"phase": "complete", "updated_at": utc_now()},
    )
    print(f"[{utc_now()}] report and completion audit passed", flush=True)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    args.model = args.model.resolve()
    args.source_dir = args.source_dir.resolve()
    args.reference_dir = args.reference_dir.resolve()
    args.output_dir = args.output_dir.resolve()
    torch.set_float32_matmul_precision("highest")
    if args.stage == "validate":
        validate_stage(args)
    elif args.stage == "smoke":
        smoke_stage(args)
    elif args.stage == "train":
        train_stage(args)
    elif args.stage == "select":
        select_stage(args)
    elif args.stage == "sanity":
        sanity_stage(args)
    elif args.stage in {"report", "orchestrate"}:
        report_stage(args)
    else:
        raise ValueError(args.stage)


if __name__ == "__main__":
    main()
