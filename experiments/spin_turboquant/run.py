"""Run the experiment specified by the workspace's ``SpinTurboQuant.md`` end to end.

Example (from the turboquant_plus checkout):

    conda run -n stq python -m experiments.spin_turboquant.run \
        --model /path/to/Meta-Llama-3.1-8B \
        --output-dir experiments/spin_turboquant/results/main

The runner is stage-resumable.  Large activation and rotation tensors stay in
the selected output directory; tabular results and a Markdown report are also
written there so every conclusion can be traced to raw measurements.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import os
import platform
import random
import subprocess
import sys
import time
from collections import defaultdict
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from .core import (
    apply_codec,
    attention_distortion_metrics,
    build_random_rotations,
    codec_latency,
    codebook_tensor,
    install_key_codec_hooks,
    no_quantization_roundtrip_metrics,
    reconstruction_metrics,
    train_headwise_rotations,
)


@dataclass(frozen=True)
class Condition:
    method: str
    bit_width: int | None
    seed: int | None
    rotations_path: Path | None
    rotation_key: str | None

    @property
    def condition_id(self) -> str:
        if self.method == "fp16":
            return "fp16"
        seed = "none" if self.seed is None else str(self.seed)
        return f"{self.method}_b{self.bit_width}_s{seed}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--stage",
        choices=("all", "capture", "train", "offline", "end-to-end", "report"),
        default="all",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--calibration-tokens", type=int, default=4096)
    parser.add_argument("--evaluation-tokens", type=int, default=1024)
    parser.add_argument("--ppl-tokens", type=int, default=1024)
    parser.add_argument("--ppl-sequence-length", type=int, default=512)
    parser.add_argument("--bits", type=int, nargs="+", default=[2, 3, 4])
    parser.add_argument("--seeds", type=int, nargs="+", default=[17, 29, 43])
    parser.add_argument("--train-steps", type=int, default=80)
    parser.add_argument("--train-batch-tokens", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=0.005)
    parser.add_argument(
        "--attention-lengths", type=int, nargs="+", default=[128, 256, 512, 1024]
    )
    parser.add_argument(
        "--retrieval-lengths", type=int, nargs="+", default=[512, 1024, 2048]
    )
    parser.add_argument("--retrieval-cases-per-length", type=int, default=2)
    parser.add_argument("--latency-tokens", type=int, default=256)
    parser.add_argument("--norm-correction", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def read_json(path: Path, default: Any) -> Any:
    return json.loads(path.read_text()) if path.exists() else default


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fields: list[str] = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def read_csv(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def key_exists(rows: list[dict[str, Any]], **wanted: Any) -> bool:
    return any(all(str(row.get(key)) == str(value) for key, value in wanted.items()) for row in rows)


def cleanup_cuda() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def check_configuration(args: argparse.Namespace) -> dict[str, Any]:
    if not args.model.exists():
        raise FileNotFoundError(f"model path does not exist: {args.model}")
    if not torch.cuda.is_available() and str(args.device).startswith("cuda"):
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
    if len(set(args.bits)) != len(args.bits) or not set(args.bits).issubset({2, 3, 4}):
        raise ValueError("--bits must be unique values chosen from 2, 3, and 4")
    if len(set(args.seeds)) != len(args.seeds) or len(args.seeds) < 2:
        raise ValueError("at least two unique random seeds are required")
    if max(args.attention_lengths) > args.evaluation_tokens:
        raise ValueError("evaluation-tokens must cover the largest attention length")
    if args.ppl_tokens < 2 or args.ppl_sequence_length < 2:
        raise ValueError("perplexity settings must contain at least two tokens")
    if args.retrieval_cases_per_length < 1:
        raise ValueError("retrieval-cases-per-length must be positive")

    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    config = json.loads((args.model / "config.json").read_text())
    head_dim = config["hidden_size"] // config["num_attention_heads"]
    expected = {
        "num_hidden_layers": 32,
        "num_key_value_heads": 8,
        "head_dim": 128,
    }
    actual = {
        "num_hidden_layers": int(config["num_hidden_layers"]),
        "num_key_value_heads": int(config["num_key_value_heads"]),
        "head_dim": int(head_dim),
    }
    if actual != expected:
        raise ValueError(f"SpinTurboQuant.md requires {expected}, but model has {actual}")
    return {
        "model_type": config["model_type"],
        "model_architecture": config.get("architectures", [None])[0],
        "vocabulary_size": len(tokenizer),
        **actual,
    }


def token_stream(
    tokenizer: Any,
    texts: Iterable[str],
    count: int,
    *,
    add_bos: bool = True,
) -> torch.Tensor:
    ids: list[int] = []
    if add_bos and tokenizer.bos_token_id is not None:
        ids.append(int(tokenizer.bos_token_id))
    separator = tokenizer.eos_token_id
    for text in texts:
        if not text or not text.strip():
            continue
        encoded = tokenizer.encode(text, add_special_tokens=False)
        if not encoded:
            continue
        ids.extend(encoded)
        if separator is not None:
            ids.append(int(separator))
        if len(ids) >= count:
            return torch.tensor(ids[:count], dtype=torch.long)
    raise RuntimeError(f"dataset yielded only {len(ids)} of {count} requested tokens")


def wikitext_rows(split: str) -> Iterator[str]:
    dataset = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split=split)
    for row in dataset:
        yield row["text"]


def tinystories_rows() -> Iterator[str]:
    dataset = load_dataset("roneneldan/TinyStories", split="validation", streaming=True)
    for row in dataset:
        yield row["text"]


def prepare_tokens(args: argparse.Namespace, tokenizer: Any) -> dict[str, torch.Tensor]:
    token_path = args.output_dir / "tokens.pt"
    largest_retrieval = max(args.retrieval_lengths)
    filler_count = largest_retrieval * args.retrieval_cases_per_length + 8192
    if token_path.exists() and not args.force:
        return torch.load(token_path, map_location="cpu", weights_only=True)
    payload = {
        "calibration": token_stream(
            tokenizer, wikitext_rows("train"), args.calibration_tokens
        ),
        "wikitext": token_stream(
            tokenizer,
            wikitext_rows("validation"),
            max(args.evaluation_tokens, args.ppl_tokens),
        ),
        "tinystories": token_stream(
            tokenizer, tinystories_rows(), args.evaluation_tokens
        ),
        "retrieval_filler": token_stream(
            tokenizer, wikitext_rows("test"), filler_count, add_bos=False
        ),
    }
    torch.save(payload, token_path)
    return payload


def load_model(model_path: Path, device: torch.device) -> Any:
    print(f"[{utc_now()}] loading {model_path} on {device}", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        local_files_only=True,
        torch_dtype=torch.bfloat16 if device.type == "cuda" else torch.float32,
        low_cpu_mem_usage=True,
        attn_implementation="sdpa",
    )
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model.to(device)


@torch.inference_mode()
def capture_activations(
    model: Any,
    input_ids: torch.Tensor,
    *,
    capture_queries_and_values: bool,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    layers = model.model.layers
    captured: dict[str, list[torch.Tensor | None]] = {
        "k": [None] * len(layers),
    }
    kinds = ["k"]
    if capture_queries_and_values:
        captured["q"] = [None] * len(layers)
        captured["v"] = [None] * len(layers)
        kinds = ["q", "k", "v"]
    handles: list[Any] = []

    for layer_index, layer in enumerate(layers):
        for kind in kinds:
            projection = getattr(layer.self_attn, f"{kind}_proj")
            heads = (
                model.config.num_attention_heads
                if kind == "q"
                else model.config.num_key_value_heads
            )

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
                )
                captured[selected_kind][selected_layer] = shaped[0].to(
                    device="cpu", dtype=torch.bfloat16
                )

            handles.append(projection.register_forward_hook(hook))

    rope: dict[str, torch.Tensor] = {}

    def rope_hook(
        _module: torch.nn.Module,
        _inputs: tuple[torch.Tensor, ...],
        output: tuple[torch.Tensor, torch.Tensor],
    ) -> None:
        rope["cos"] = output[0].detach()[0].to(device="cpu", dtype=torch.float32)
        rope["sin"] = output[1].detach()[0].to(device="cpu", dtype=torch.float32)

    if capture_queries_and_values:
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
        stacked = torch.stack([value for value in values if value is not None])
        if kind == "k" and not capture_queries_and_values:
            # Training convention: (layers, KV heads, tokens, head dimension).
            stacked = stacked.permute(0, 2, 1, 3).contiguous()
        result[kind] = stacked
    result.update(rope)
    return result


def capture_stage(args: argparse.Namespace, device: torch.device) -> None:
    activation_dir = args.output_dir / "activations"
    expected = {
        "calibration": activation_dir / "calibration.pt",
        "wikitext": activation_dir / "wikitext.pt",
        "tinystories": activation_dir / "tinystories.pt",
    }
    if all(path.exists() for path in expected.values()) and not args.force:
        print("activation artifacts already exist; capture stage is complete", flush=True)
        return

    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    tokens = prepare_tokens(args, tokenizer)
    model = load_model(args.model, device)
    activation_dir.mkdir(parents=True, exist_ok=True)
    jobs = (
        ("calibration", False),
        ("wikitext", True),
        ("tinystories", True),
    )
    for name, full_capture in jobs:
        path = expected[name]
        if path.exists() and not args.force:
            continue
        count = args.calibration_tokens if name == "calibration" else args.evaluation_tokens
        print(f"[{utc_now()}] capturing {name}: {count} tokens", flush=True)
        payload = capture_activations(
            model,
            tokens[name][:count],
            capture_queries_and_values=full_capture,
            device=device,
        )
        torch.save(payload, path)
        shapes = {key: list(value.shape) for key, value in payload.items()}
        write_json(path.with_suffix(".json"), {"name": name, "shapes": shapes})
        del payload
        cleanup_cuda()
    del model
    cleanup_cuda()


def rotation_path(output_dir: Path, bit_width: int, seed: int) -> Path:
    return output_dir / "rotations" / f"bit{bit_width}_seed{seed}.pt"


def train_stage(args: argparse.Namespace, device: torch.device, dimensions: dict[str, Any]) -> None:
    calibration_path = args.output_dir / "activations" / "calibration.pt"
    if not calibration_path.exists():
        raise FileNotFoundError("capture stage must run before training")
    calibration = torch.load(calibration_path, map_location="cpu", weights_only=True)["k"]
    layers, heads, tokens, head_dim = calibration.shape
    if (layers, heads, head_dim) != (
        dimensions["num_hidden_layers"],
        dimensions["num_key_value_heads"],
        dimensions["head_dim"],
    ):
        raise RuntimeError("captured calibration tensor has unexpected model dimensions")
    flat_keys = calibration.reshape(layers * heads, tokens, head_dim).to(
        device=device, dtype=torch.float32
    )
    rotation_dir = args.output_dir / "rotations"
    rotation_dir.mkdir(parents=True, exist_ok=True)
    training_rows = read_csv(args.output_dir / "training.csv")

    random_by_seed: dict[int, torch.Tensor] = {}
    for seed in args.seeds:
        print(f"[{utc_now()}] constructing TurboQuant random rotations for seed {seed}", flush=True)
        random_by_seed[seed] = build_random_rotations(layers, heads, head_dim, seed)

    for bit_width in args.bits:
        centroids = codebook_tensor(bit_width, head_dim, device=device)
        for seed in args.seeds:
            path = rotation_path(args.output_dir, bit_width, seed)
            if path.exists() and not args.force:
                print(f"rotation artifact exists: {path.name}", flush=True)
                continue
            initial = random_by_seed[seed].reshape(layers * heads, head_dim, head_dim).to(device)
            print(
                f"[{utc_now()}] training bit={bit_width}, seed={seed}, "
                f"heads={layers * heads}, tokens/head={tokens}",
                flush=True,
            )

            def progress(step: int, total: int, loss: float) -> None:
                interval = max(total // 8, 1)
                if step == 1 or step == total or step % interval == 0:
                    print(
                        f"  step {step:4d}/{total}: minibatch normalized MSE={loss:.8g}",
                        flush=True,
                    )

            learned, stats = train_headwise_rotations(
                flat_keys,
                initial,
                centroids,
                steps=args.train_steps,
                batch_tokens=args.train_batch_tokens,
                learning_rate=args.learning_rate,
                optimizer_seed=seed + bit_width * 100_000,
                progress=progress,
            )
            artifact = {
                "bit_width": bit_width,
                "seed": seed,
                "random": random_by_seed[seed],
                "learned": learned.reshape(layers, heads, head_dim, head_dim).cpu(),
                "stats": stats.to_dict(),
            }
            torch.save(artifact, path)
            training_rows = [
                row
                for row in training_rows
                if not (
                    str(row.get("bit_width")) == str(bit_width)
                    and str(row.get("seed")) == str(seed)
                )
            ]
            training_rows.append(
                {"bit_width": bit_width, "seed": seed, **stats.to_dict()}
            )
            write_csv(args.output_dir / "training.csv", training_rows)
            del learned, initial, artifact
            cleanup_cuda()
    del flat_keys, calibration, random_by_seed
    cleanup_cuda()


def conditions(args: argparse.Namespace, *, include_fp16: bool) -> list[Condition]:
    result: list[Condition] = []
    if include_fp16:
        result.append(Condition("fp16", None, None, None, None))
    for bit_width in args.bits:
        result.append(Condition("identity", bit_width, None, None, None))
        for seed in args.seeds:
            path = rotation_path(args.output_dir, bit_width, seed)
            result.append(Condition("random", bit_width, seed, path, "random"))
            result.append(Condition("learned", bit_width, seed, path, "learned"))
    return result


def load_condition_rotations(
    condition: Condition,
    dimensions: dict[str, Any],
    device: torch.device,
) -> torch.Tensor:
    layers = dimensions["num_hidden_layers"]
    heads = dimensions["num_key_value_heads"]
    head_dim = dimensions["head_dim"]
    if condition.method == "identity":
        return torch.eye(head_dim, dtype=torch.float32, device=device).expand(
            layers, heads, head_dim, head_dim
        )
    if condition.rotations_path is None or condition.rotation_key is None:
        raise ValueError(f"condition {condition.condition_id} has no rotations")
    if not condition.rotations_path.exists():
        raise FileNotFoundError(condition.rotations_path)
    artifact = torch.load(condition.rotations_path, map_location="cpu", weights_only=True)
    return artifact[condition.rotation_key].to(device=device, dtype=torch.float32)


def offline_stage(args: argparse.Namespace, device: torch.device, dimensions: dict[str, Any]) -> None:
    metric_path = args.output_dir / "offline_metrics.csv"
    head_metric_path = args.output_dir / "head_metrics.csv"
    latency_path = args.output_dir / "latency.csv"
    rows = read_csv(metric_path)
    head_rows = read_csv(head_metric_path)
    latency_rows = read_csv(latency_path)
    datasets: dict[str, dict[str, torch.Tensor]] = {}
    for dataset_name in ("wikitext", "tinystories"):
        path = args.output_dir / "activations" / f"{dataset_name}.pt"
        if not path.exists():
            raise FileNotFoundError("capture stage must run before offline evaluation")
        datasets[dataset_name] = torch.load(path, map_location="cpu", weights_only=True)

    roundtrip_path = args.output_dir / "no_quantization_roundtrip.json"
    if not roundtrip_path.exists() or args.force:
        reference_condition = Condition(
            "random",
            args.bits[0],
            args.seeds[0],
            rotation_path(args.output_dir, args.bits[0], args.seeds[0]),
            "random",
        )
        reference_rotations = load_condition_rotations(
            reference_condition, dimensions, device
        )
        payload = datasets["wikitext"]
        roundtrip = no_quantization_roundtrip_metrics(
            payload["q"],
            payload["k"],
            payload["cos"],
            payload["sin"],
            reference_rotations,
            sequence_length=min(256, args.evaluation_tokens),
        )
        roundtrip.update(
            {"dataset": "wikitext", "rotation_seed": args.seeds[0]}
        )
        write_json(roundtrip_path, roundtrip)
        del reference_rotations
        cleanup_cuda()

    for condition in conditions(args, include_fp16=False):
        rotations = load_condition_rotations(condition, dimensions, device)
        centroids = codebook_tensor(
            int(condition.bit_width), dimensions["head_dim"], device=device
        )
        for dataset_name, payload in datasets.items():
            keys = payload["k"].permute(0, 2, 1, 3).contiguous()
            need_reconstruction = args.force or not key_exists(
                rows,
                condition_id=condition.condition_id,
                dataset=dataset_name,
                metric_scope="reconstruction",
                sequence_length=args.evaluation_tokens,
            )
            need_heads = args.force or not key_exists(
                head_rows,
                condition_id=condition.condition_id,
                dataset=dataset_name,
                layer=0,
                head=0,
            )
            if need_reconstruction or need_heads:
                print(
                    f"[{utc_now()}] reconstruction {condition.condition_id} on {dataset_name}",
                    flush=True,
                )
                reconstruction = reconstruction_metrics(
                    keys,
                    rotations,
                    centroids,
                    norm_correction=args.norm_correction,
                    return_per_head=True,
                )
                per_head = reconstruction.pop("_per_head")
                if need_reconstruction:
                    if args.force:
                        rows = [
                            row
                            for row in rows
                            if not (
                                row.get("condition_id") == condition.condition_id
                                and row.get("dataset") == dataset_name
                                and row.get("metric_scope") == "reconstruction"
                            )
                        ]
                    rows.append(
                        {
                            "condition_id": condition.condition_id,
                            "method": condition.method,
                            "bit_width": condition.bit_width,
                            "seed": condition.seed,
                            "dataset": dataset_name,
                            "metric_scope": "reconstruction",
                            "sequence_length": args.evaluation_tokens,
                            **reconstruction,
                        }
                    )
                    write_csv(metric_path, rows)
                if need_heads:
                    if args.force:
                        head_rows = [
                            row
                            for row in head_rows
                            if not (
                                row.get("condition_id") == condition.condition_id
                                and row.get("dataset") == dataset_name
                            )
                        ]
                    for layer in range(dimensions["num_hidden_layers"]):
                        for head in range(dimensions["num_key_value_heads"]):
                            head_rows.append(
                                {
                                    "condition_id": condition.condition_id,
                                    "method": condition.method,
                                    "bit_width": condition.bit_width,
                                    "seed": condition.seed,
                                    "dataset": dataset_name,
                                    "layer": layer,
                                    "head": head,
                                    "normalized_key_mse": per_head[
                                        "normalized_key_mse"
                                    ][layer][head],
                                    "original_key_mse": per_head[
                                        "original_key_mse"
                                    ][layer][head],
                                    "original_key_relative_mse": per_head[
                                        "original_key_relative_mse"
                                    ][layer][head],
                                }
                            )
                    write_csv(head_metric_path, head_rows)

            for length in args.attention_lengths:
                if not args.force and key_exists(
                    rows,
                    condition_id=condition.condition_id,
                    dataset=dataset_name,
                    metric_scope="attention",
                    sequence_length=length,
                ):
                    continue
                print(
                    f"[{utc_now()}] attention {condition.condition_id} on "
                    f"{dataset_name}, length={length}",
                    flush=True,
                )
                attention = attention_distortion_metrics(
                    payload["q"],
                    payload["k"],
                    payload["v"],
                    payload["cos"],
                    payload["sin"],
                    rotations,
                    centroids,
                    sequence_length=length,
                    norm_correction=args.norm_correction,
                )
                if args.force:
                    rows = [
                        row
                        for row in rows
                        if not (
                            row.get("condition_id") == condition.condition_id
                            and row.get("dataset") == dataset_name
                            and row.get("metric_scope") == "attention"
                            and str(row.get("sequence_length")) == str(length)
                        )
                    ]
                rows.append(
                    {
                        "condition_id": condition.condition_id,
                        "method": condition.method,
                        "bit_width": condition.bit_width,
                        "seed": condition.seed,
                        "dataset": dataset_name,
                        "metric_scope": "attention",
                        **attention,
                    }
                )
                write_csv(metric_path, rows)

        if args.force or not key_exists(latency_rows, condition_id=condition.condition_id):
            latency_keys = (
                datasets["wikitext"]["k"][:, : args.latency_tokens]
                .permute(0, 2, 1, 3)
                .contiguous()
            )
            measured = codec_latency(
                latency_keys,
                rotations,
                centroids,
                norm_correction=args.norm_correction,
            )
            if condition.method == "identity":
                measured["rotation_storage_bytes_fp32"] = 0
            if args.force:
                latency_rows = [
                    row
                    for row in latency_rows
                    if row.get("condition_id") != condition.condition_id
                ]
            latency_rows.append(
                {
                    "condition_id": condition.condition_id,
                    "method": condition.method,
                    "bit_width": condition.bit_width,
                    "seed": condition.seed,
                    **measured,
                }
            )
            write_csv(latency_path, latency_rows)
        del rotations, centroids
        cleanup_cuda()


@torch.inference_mode()
def perplexity(
    model: Any,
    token_ids: torch.Tensor,
    *,
    sequence_length: int,
    device: torch.device,
) -> dict[str, float | int]:
    total_nll = 0.0
    total_predictions = 0
    elapsed = 0.0
    for start in range(0, token_ids.numel() - 1, sequence_length):
        chunk = token_ids[start : min(start + sequence_length, token_ids.numel())]
        if chunk.numel() < 2:
            continue
        input_ids = chunk.unsqueeze(0).to(device)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        began = time.perf_counter()
        output = model(input_ids=input_ids, labels=input_ids, use_cache=False)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elapsed += time.perf_counter() - began
        predictions = chunk.numel() - 1
        total_nll += float(output.loss) * predictions
        total_predictions += predictions
        del output, input_ids
    mean_nll = total_nll / max(total_predictions, 1)
    return {
        "mean_nll": mean_nll,
        "perplexity": math.exp(min(mean_nll, 50.0)),
        "predicted_tokens": total_predictions,
        "elapsed_seconds": elapsed,
    }


def single_token_candidates(tokenizer: Any) -> tuple[list[str], list[int]]:
    labels = ["avocado", "telescope", "violin", "cinnamon"]
    token_ids: list[int] = []
    for label in labels:
        encoded = tokenizer.encode(" " + label, add_special_tokens=False)
        if len(encoded) != 1:
            raise RuntimeError(f"retrieval candidate {label!r} is not one token: {encoded}")
        token_ids.append(int(encoded[0]))
    return labels, token_ids


def build_retrieval_tasks(
    tokenizer: Any,
    filler: torch.Tensor,
    lengths: Sequence[int],
    cases_per_length: int,
) -> list[dict[str, Any]]:
    labels, candidate_ids = single_token_candidates(tokenizer)
    tasks: list[dict[str, Any]] = []
    filler_offset = 0
    positions = (0.1, 0.5, 0.9)
    bos = [int(tokenizer.bos_token_id)] if tokenizer.bos_token_id is not None else []
    for length_index, length in enumerate(lengths):
        for case_index in range(cases_per_length):
            target_index = (length_index * cases_per_length + case_index) % len(labels)
            position = positions[case_index % len(positions)]
            label = labels[target_index]
            vaults = ["ALPHA", "BETA", "GAMMA", "DELTA"]
            facts_text = "Important records:\n" + "".join(
                f"The secret passphrase for vault {vault} is {candidate}.\n"
                for vault, candidate in zip(vaults, labels)
            )
            fact = tokenizer.encode(facts_text, add_special_tokens=False)
            question = tokenizer.encode(
                f"\nQuestion: What is the secret passphrase for vault {vaults[target_index]}? "
                f"Answer: The secret passphrase for vault {vaults[target_index]} is",
                add_special_tokens=False,
            )
            filler_needed = length - len(bos) - len(fact) - len(question)
            if filler_needed <= 0:
                raise ValueError(f"retrieval length {length} is too short for the prompt")
            if filler_offset + filler_needed > filler.numel():
                filler_offset = 0
            selected_filler = filler[filler_offset : filler_offset + filler_needed].tolist()
            filler_offset += filler_needed
            before = int(filler_needed * position)
            prompt = bos + selected_filler[:before] + fact + selected_filler[before:] + question
            if len(prompt) != length:
                raise AssertionError("retrieval prompt length construction failed")
            tasks.append(
                {
                    "task_id": f"length{length}_case{case_index}",
                    "length": length,
                    "position_fraction": position,
                    "target": label,
                    "target_index": target_index,
                    "candidate_labels": labels,
                    "candidate_ids": candidate_ids,
                    "input_ids": torch.tensor(prompt, dtype=torch.long),
                }
            )
    return tasks


@torch.inference_mode()
def retrieval_metrics(
    model: Any,
    tasks: list[dict[str, Any]],
    *,
    device: torch.device,
) -> tuple[dict[str, float | int], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    elapsed = 0.0
    for task in tasks:
        input_ids = task["input_ids"].unsqueeze(0).to(device)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        began = time.perf_counter()
        output = model(input_ids=input_ids, use_cache=False)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elapsed += time.perf_counter() - began
        logits = output.logits[0, -1].float()
        candidate_logits = logits[torch.tensor(task["candidate_ids"], device=device)]
        prediction_index = int(torch.argmax(candidate_logits))
        target_index = int(task["target_index"])
        distractors = torch.cat(
            (candidate_logits[:target_index], candidate_logits[target_index + 1 :])
        )
        rows.append(
            {
                "task_id": task["task_id"],
                "length": task["length"],
                "position_fraction": task["position_fraction"],
                "target": task["target"],
                "prediction": task["candidate_labels"][prediction_index],
                "correct": int(prediction_index == target_index),
                "target_logit": float(candidate_logits[target_index]),
                "target_margin": float(candidate_logits[target_index] - distractors.max()),
            }
        )
        del output, logits, candidate_logits, input_ids
    return (
        {
            "retrieval_accuracy": sum(row["correct"] for row in rows) / max(len(rows), 1),
            "retrieval_mean_target_margin": float(
                np.mean([row["target_margin"] for row in rows])
            ),
            "retrieval_tasks": len(rows),
            "elapsed_seconds": elapsed,
        },
        rows,
    )


def end_to_end_stage(
    args: argparse.Namespace,
    device: torch.device,
    dimensions: dict[str, Any],
) -> None:
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    tokens = prepare_tokens(args, tokenizer)
    tasks = build_retrieval_tasks(
        tokenizer,
        tokens["retrieval_filler"],
        args.retrieval_lengths,
        args.retrieval_cases_per_length,
    )
    model = load_model(args.model, device)
    ppl_rows = read_csv(args.output_dir / "perplexity.csv")
    retrieval_rows = read_csv(args.output_dir / "retrieval.csv")
    retrieval_detail_rows = read_csv(args.output_dir / "retrieval_details.csv")

    for condition in conditions(args, include_fp16=True):
        if args.force:
            ppl_rows = [
                row
                for row in ppl_rows
                if row.get("condition_id") != condition.condition_id
            ]
            retrieval_rows = [
                row
                for row in retrieval_rows
                if row.get("condition_id") != condition.condition_id
            ]
            retrieval_detail_rows = [
                row
                for row in retrieval_detail_rows
                if row.get("condition_id") != condition.condition_id
            ]
        rotations: torch.Tensor | None = None
        centroids: torch.Tensor | None = None
        if condition.method != "fp16":
            rotations = load_condition_rotations(condition, dimensions, device)
            centroids = codebook_tensor(
                int(condition.bit_width), dimensions["head_dim"], device=device
            )
        context = (
            nullcontext()
            if condition.method == "fp16"
            else install_key_codec_hooks(
                model,
                rotations,
                centroids,
                norm_correction=args.norm_correction,
            )
        )
        with context:
            if args.force or not key_exists(ppl_rows, condition_id=condition.condition_id):
                print(f"[{utc_now()}] perplexity {condition.condition_id}", flush=True)
                measured = perplexity(
                    model,
                    tokens["wikitext"][: args.ppl_tokens],
                    sequence_length=args.ppl_sequence_length,
                    device=device,
                )
                ppl_rows.append(
                    {
                        "condition_id": condition.condition_id,
                        "method": condition.method,
                        "bit_width": condition.bit_width,
                        "seed": condition.seed,
                        **measured,
                    }
                )
                write_csv(args.output_dir / "perplexity.csv", ppl_rows)

            if args.force or not key_exists(
                retrieval_rows, condition_id=condition.condition_id
            ):
                print(f"[{utc_now()}] retrieval {condition.condition_id}", flush=True)
                summary, details = retrieval_metrics(model, tasks, device=device)
                retrieval_rows.append(
                    {
                        "condition_id": condition.condition_id,
                        "method": condition.method,
                        "bit_width": condition.bit_width,
                        "seed": condition.seed,
                        **summary,
                    }
                )
                for detail in details:
                    retrieval_detail_rows.append(
                        {
                            "condition_id": condition.condition_id,
                            "method": condition.method,
                            "bit_width": condition.bit_width,
                            "seed": condition.seed,
                            **detail,
                        }
                    )
                write_csv(args.output_dir / "retrieval.csv", retrieval_rows)
                write_csv(
                    args.output_dir / "retrieval_details.csv", retrieval_detail_rows
                )
        del rotations, centroids
        cleanup_cuda()
    del model
    cleanup_cuda()


def number(value: Any) -> float:
    return float(value)


def aggregate_paired(
    rows: list[dict[str, Any]],
    metric: str,
    *,
    scopes: dict[str, str] | None = None,
    pair_fields: Sequence[str] = ("bit_width", "seed"),
    lower_is_better: bool = True,
) -> dict[int, dict[str, float]]:
    scopes = scopes or {}
    selected = [
        row
        for row in rows
        if all(str(row.get(key)) == str(value) for key, value in scopes.items())
        and row.get("method") in {"random", "learned"}
    ]
    grouped: dict[tuple[str, ...], dict[str, float]] = defaultdict(dict)
    for row in selected:
        grouped[tuple(str(row.get(field, "")) for field in pair_fields)][
            row["method"]
        ] = number(row[metric])
    output: dict[int, dict[str, float]] = {}
    for bit_width in sorted({int(key[0]) for key in grouped}):
        pairs = [value for key, value in grouped.items() if int(key[0]) == bit_width]
        pairs = [pair for pair in pairs if {"random", "learned"} <= pair.keys()]
        if not pairs:
            continue
        if lower_is_better:
            improvements = [
                (pair["random"] - pair["learned"])
                / max(abs(pair["random"]), 1e-30)
                for pair in pairs
            ]
        else:
            improvements = [
                (pair["learned"] - pair["random"])
                / max(abs(pair["random"]), 1e-30)
                for pair in pairs
            ]
        output[bit_width] = {
            "random_mean": float(np.mean([pair["random"] for pair in pairs])),
            "learned_mean": float(np.mean([pair["learned"] for pair in pairs])),
            "relative_improvement_mean": float(np.mean(improvements)),
            "paired_win_rate": float(np.mean([value > 0 for value in improvements])),
            "pairs": len(pairs),
        }
    return output


def markdown_table(headers: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)


def report_stage(args: argparse.Namespace, dimensions: dict[str, Any]) -> None:
    training = read_csv(args.output_dir / "training.csv")
    offline = read_csv(args.output_dir / "offline_metrics.csv")
    head_metrics = read_csv(args.output_dir / "head_metrics.csv")
    ppl = read_csv(args.output_dir / "perplexity.csv")
    retrieval = read_csv(args.output_dir / "retrieval.csv")
    retrieval_details = read_csv(args.output_dir / "retrieval_details.csv")
    latency = read_csv(args.output_dir / "latency.csv")
    roundtrip = read_json(args.output_dir / "no_quantization_roundtrip.json", None)
    required = {
        "training.csv": training,
        "offline_metrics.csv": offline,
        "head_metrics.csv": head_metrics,
        "perplexity.csv": ppl,
        "retrieval.csv": retrieval,
        "latency.csv": latency,
    }
    missing = [name for name, rows in required.items() if not rows]
    if missing:
        raise RuntimeError(f"cannot generate a final report; missing results in {missing}")
    if roundtrip is None:
        raise RuntimeError("cannot generate a final report; no-quantization gate is missing")
    quantized_conditions = len(args.bits) * (1 + 2 * len(args.seeds))
    all_conditions = 1 + quantized_conditions
    expected_counts = {
        "training.csv": len(args.bits) * len(args.seeds),
        "offline_metrics.csv": quantized_conditions
        * 2
        * (1 + len(args.attention_lengths)),
        "head_metrics.csv": quantized_conditions
        * 2
        * dimensions["num_hidden_layers"]
        * dimensions["num_key_value_heads"],
        "perplexity.csv": all_conditions,
        "retrieval.csv": all_conditions,
        "latency.csv": quantized_conditions,
        "retrieval_details.csv": all_conditions
        * len(args.retrieval_lengths)
        * args.retrieval_cases_per_length,
    }
    actual_counts = {
        "training.csv": len(training),
        "offline_metrics.csv": len(offline),
        "head_metrics.csv": len(head_metrics),
        "perplexity.csv": len(ppl),
        "retrieval.csv": len(retrieval),
        "latency.csv": len(latency),
        "retrieval_details.csv": len(retrieval_details),
    }
    incomplete = {
        name: {"expected": expected, "actual": actual_counts[name]}
        for name, expected in expected_counts.items()
        if actual_counts[name] != expected
    }
    if incomplete:
        raise RuntimeError(f"result matrix is incomplete or duplicated: {incomplete}")

    recon_summaries: dict[str, dict[int, dict[str, float]]] = {}
    for dataset in ("wikitext", "tinystories"):
        recon_summaries[dataset] = aggregate_paired(
            offline,
            "normalized_key_mse",
            scopes={"dataset": dataset, "metric_scope": "reconstruction"},
            pair_fields=("bit_width", "seed", "dataset", "sequence_length"),
        )
    attention_summary = aggregate_paired(
        offline,
        "attention_probability_kl",
        scopes={"metric_scope": "attention"},
        pair_fields=(
            "bit_width",
            "seed",
            "dataset",
            "metric_scope",
            "sequence_length",
        ),
    )
    head_summary = aggregate_paired(
        head_metrics,
        "normalized_key_mse",
        pair_fields=("bit_width", "seed", "dataset", "layer", "head"),
    )
    ppl_summary = aggregate_paired(ppl, "perplexity")
    retrieval_summary = aggregate_paired(
        retrieval, "retrieval_accuracy", lower_is_better=False
    )
    retrieval_margin_summary = aggregate_paired(
        retrieval, "retrieval_mean_target_margin", lower_is_better=False
    )

    training_table: list[list[str]] = []
    calibration_improvements: dict[int, float] = {}
    for bit_width in args.bits:
        selected = [row for row in training if int(row["bit_width"]) == bit_width]
        calibration_improvements[bit_width] = float(
            np.mean([number(row["relative_improvement"]) for row in selected])
        )
        training_table.append(
            [
                str(bit_width),
                f"{np.mean([number(row['initial_loss']) for row in selected]):.6g}",
                f"{np.mean([number(row['final_loss']) for row in selected]):.6g}",
                f"{100*np.mean([number(row['relative_improvement']) for row in selected]):.2f}%",
                f"{max(number(row['orthogonality_max_abs']) for row in selected):.3g}",
            ]
        )

    reconstruction_table: list[list[str]] = []
    for bit_width in args.bits:
        for dataset in ("wikitext", "tinystories"):
            summary = recon_summaries[dataset].get(bit_width, {})
            identity_value = np.mean(
                [
                    number(row["normalized_key_mse"])
                    for row in offline
                    if row["method"] == "identity"
                    and int(row["bit_width"]) == bit_width
                    and row["dataset"] == dataset
                    and row["metric_scope"] == "reconstruction"
                ]
            )
            reconstruction_table.append(
                [
                    str(bit_width),
                    dataset,
                    f"{identity_value:.6g}",
                    f"{summary.get('random_mean', float('nan')):.6g}",
                    f"{summary.get('learned_mean', float('nan')):.6g}",
                    f"{100*summary.get('relative_improvement_mean', float('nan')):.2f}%",
                    f"{100*summary.get('paired_win_rate', float('nan')):.0f}%",
                ]
            )

    generalization_table: list[list[str]] = []
    for bit_width in args.bits:
        calibration_reduction = calibration_improvements[bit_width]
        wiki_reduction = recon_summaries["wikitext"][bit_width][
            "relative_improvement_mean"
        ]
        domain_reduction = recon_summaries["tinystories"][bit_width][
            "relative_improvement_mean"
        ]
        heads = head_summary[bit_width]
        generalization_table.append(
            [
                str(bit_width),
                f"{100*calibration_reduction:.2f}%",
                f"{100*wiki_reduction:.2f}%",
                f"{100*(wiki_reduction-calibration_reduction):+.2f} pp",
                f"{100*domain_reduction:.2f}%",
                f"{100*(domain_reduction-calibration_reduction):+.2f} pp",
                f"{100*heads['paired_win_rate']:.1f}%",
            ]
        )
    attention_table: list[list[str]] = []
    for bit_width in args.bits:
        summary = attention_summary.get(bit_width, {})
        identity_values = [
            number(row["attention_probability_kl"])
            for row in offline
            if row["method"] == "identity"
            and int(row["bit_width"]) == bit_width
            and row["metric_scope"] == "attention"
        ]
        attention_table.append(
            [
                str(bit_width),
                f"{np.mean(identity_values):.6g}",
                f"{summary.get('random_mean', float('nan')):.6g}",
                f"{summary.get('learned_mean', float('nan')):.6g}",
                f"{100*summary.get('relative_improvement_mean', float('nan')):.2f}%",
                f"{100*summary.get('paired_win_rate', float('nan')):.0f}%",
            ]
        )

    end_to_end_table: list[list[str]] = []
    fp16_ppl = next(number(row["perplexity"]) for row in ppl if row["method"] == "fp16")
    fp16_retrieval = next(
        number(row["retrieval_accuracy"]) for row in retrieval if row["method"] == "fp16"
    )
    for bit_width in args.bits:
        p = ppl_summary.get(bit_width, {})
        r = retrieval_summary.get(bit_width, {})
        margin = retrieval_margin_summary.get(bit_width, {})
        identity_ppl = next(
            number(row["perplexity"])
            for row in ppl
            if row["method"] == "identity" and int(row["bit_width"]) == bit_width
        )
        identity_retrieval = next(
            number(row["retrieval_accuracy"])
            for row in retrieval
            if row["method"] == "identity" and int(row["bit_width"]) == bit_width
        )
        end_to_end_table.append(
            [
                str(bit_width),
                f"{identity_ppl:.4f}",
                f"{p.get('random_mean', float('nan')):.4f}",
                f"{p.get('learned_mean', float('nan')):.4f}",
                f"{100*p.get('relative_improvement_mean', float('nan')):.2f}%",
                f"{100*p.get('paired_win_rate', float('nan')):.0f}%",
                f"{identity_retrieval:.3f}",
                f"{r.get('random_mean', float('nan')):.3f}",
                f"{r.get('learned_mean', float('nan')):.3f}",
                f"{margin.get('random_mean', float('nan')):.3f}",
                f"{margin.get('learned_mean', float('nan')):.3f}",
            ]
        )

    all_recon_wins = all(
        summary[bit]["paired_win_rate"] == 1.0
        for summary in recon_summaries.values()
        for bit in args.bits
    )
    attention_wins = all(
        attention_summary[bit]["paired_win_rate"] == 1.0 for bit in args.bits
    )
    criterion_supported = all_recon_wins and attention_wins
    perplexity_consistent = all(
        ppl_summary[bit]["paired_win_rate"] == 1.0 for bit in args.bits
    )
    if criterion_supported and perplexity_consistent:
        verdict = "Supported in this initial experiment"
    elif criterion_supported:
        verdict = "Distortion hypothesis supported; end-to-end perplexity evidence is mixed"
    else:
        verdict = "Not consistently supported in this initial experiment"

    command = " ".join(
        [
            "conda run -n stq python -m experiments.spin_turboquant.run",
            f"--model {args.model}",
            f"--output-dir {args.output_dir}",
        ]
    )
    latency_table: list[list[str]] = []
    for bit_width in args.bits:
        for method in ("identity", "random", "learned"):
            selected = [
                row
                for row in latency
                if int(row["bit_width"]) == bit_width and row["method"] == method
            ]
            latency_table.append(
                [
                    str(bit_width),
                    method,
                    f"{np.mean([number(row['rotation_forward_latency_ms']) for row in selected]):.3f}",
                    f"{np.mean([number(row['rotation_forward_microseconds_per_vector']) for row in selected]):.4f}",
                    f"{np.mean([number(row['codec_latency_ms']) for row in selected]):.3f}",
                ]
            )
    report = f"""# SpinTurboQuant experiment report

Generated: {utc_now()}

## Verdict

**{verdict}.** The strict automated criterion requires learned rotations to beat
their paired random initialization on held-out normalized-key MSE for every seed
in both domains and on every paired attention-KL comparison across seeds,
domains, and sequence lengths. This is an initial controlled experiment, not a
claim of broad model-family generality.

## Fixed setup

- Model: `{args.model}` ({dimensions['num_hidden_layers']} layers,
  {dimensions['num_key_value_heads']} KV heads, head dimension {dimensions['head_dim']})
- Calibration: WikiText-2 train, {args.calibration_tokens} tokens
- Held out in-domain: WikiText-2 validation, {args.evaluation_tokens} tokens
- Held out cross-domain: TinyStories validation, {args.evaluation_tokens} tokens
- Bit widths: {', '.join(map(str, args.bits))}
- Random seeds: {', '.join(map(str, args.seeds))}
- Optimization: {args.train_steps} Adam steps, batch {args.train_batch_tokens} tokens/head,
  learning rate {args.learning_rate}
- Attention lengths: {', '.join(map(str, args.attention_lengths))}
- Codec norm correction: {args.norm_correction}
- Existing LLM parameters and TurboQuant Lloyd-Max codebooks were frozen.

## No-quantization correctness gate

On real captured keys, forward rotation followed by inverse rotation before RoPE
had maximum key error {number(roundtrip['pre_rope_key_roundtrip_max_abs']):.3g} and
maximum attention-logit error
{number(roundtrip['attention_logit_roundtrip_max_abs']):.3g}
(MSE {number(roundtrip['attention_logit_roundtrip_mse']):.3g}). This verifies the
pre-RoPE restore-then-RoPE path independently of quantization.

## Calibration objective

{markdown_table(['bits', 'initial MSE', 'learned MSE', 'relative reduction', 'max abs(R^T R-I)'], training_table)}

The explicit calibration-to-held-out gap and the fraction of layer/head pairs
that improve are:

{markdown_table(['bits', 'calibration reduction', 'WikiText reduction', 'WikiText gap', 'TinyStories reduction', 'TinyStories gap', 'head-pair wins'], generalization_table)}

## Held-out paired results

Positive reductions mean learned is better than its exact paired random start.
Win rates are computed across seeds for reconstruction and across seeds, domains,
and sequence lengths for attention.

{markdown_table(['bits', 'domain', 'identity MSE', 'random MSE', 'learned MSE', 'paired reduction', 'paired wins'], reconstruction_table)}

Attention probability KL divergence, pooled across both held-out domains and all
configured sequence lengths:

{markdown_table(['bits', 'identity KL', 'random KL', 'learned KL', 'paired reduction', 'paired wins'], attention_table)}

## End-to-end results

The uncompressed model has perplexity {fp16_ppl:.4f} and retrieval accuracy
{fp16_retrieval:.3f}. Perplexity was measured on {args.ppl_tokens} held-out tokens;
retrieval used {len(args.retrieval_lengths) * args.retrieval_cases_per_length}
multi-key prompts at lengths {', '.join(map(str, args.retrieval_lengths))}.
The target margin is the target-token logit minus the strongest distractor logit.

{markdown_table(['bits', 'identity PPL', 'random PPL', 'learned PPL', 'paired PPL reduction', 'PPL wins', 'identity retrieval', 'random retrieval', 'learned retrieval', 'random margin', 'learned margin'], end_to_end_table)}

The main reconstruction/attention criterion is
{'satisfied' if criterion_supported else 'not satisfied'}. End-to-end perplexity
improves for all seeds only when the PPL win column is 100%; retrieval accuracy
may saturate, so the candidate margin is reported as a secondary diagnostic.

## Cost

Each learned dense rotation set stores
{dimensions['num_hidden_layers'] * dimensions['num_key_value_heads'] * dimensions['head_dim'] * dimensions['head_dim'] * 4:,}
bytes ({dimensions['num_hidden_layers'] * dimensions['num_key_value_heads'] * dimensions['head_dim'] * dimensions['head_dim'] * 4 / 2**20:.1f} MiB)
in FP32. Detailed measured dense-codec timings are in `latency.csv`; this prototype
does not claim an inference-speed improvement.

{markdown_table(['bits', 'method', 'forward rotation ms', 'us/vector', 'full codec ms'], latency_table)}

## Reproduce

```bash
{command}
```

Raw evidence: `training.csv`, `offline_metrics.csv`, `head_metrics.csv`, `perplexity.csv`,
`retrieval.csv`, `retrieval_details.csv`, `latency.csv`, saved rotations, captured
activations, and `run_config.json` in this directory.
"""
    (args.output_dir / "report.md").write_text(report)
    write_json(
        args.output_dir / "summary.json",
        {
            "generated_at": utc_now(),
            "verdict": verdict,
            "criterion_supported": criterion_supported,
            "perplexity_consistent": perplexity_consistent,
            "no_quantization_roundtrip": roundtrip,
            "reconstruction": recon_summaries,
            "attention": attention_summary,
            "headwise_reconstruction": head_summary,
            "perplexity": ppl_summary,
            "retrieval": retrieval_summary,
            "retrieval_margin": retrieval_margin_summary,
        },
    )
    print(f"wrote {args.output_dir / 'report.md'}", flush=True)


def environment_metadata(device: torch.device) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "timestamp": utc_now(),
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "numpy": np.__version__,
        "command": sys.argv,
    }
    if device.type == "cuda":
        properties = torch.cuda.get_device_properties(device)
        metadata["gpu"] = {
            "name": properties.name,
            "total_memory_bytes": properties.total_memory,
            "capability": list(properties.major_minor) if hasattr(properties, "major_minor") else [properties.major, properties.minor],
        }
    try:
        metadata["git_commit"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip()
        metadata["git_status"] = subprocess.check_output(
            ["git", "status", "--short"], text=True
        ).splitlines()
    except (OSError, subprocess.CalledProcessError):
        pass
    return metadata


def main() -> None:
    args = parse_args()
    args.model = args.model.resolve()
    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(0)
        # Learned-rotation gains are small enough that TF32 rounding can pollute
        # reconstruction and orthogonality measurements. Model weights remain
        # BF16, but every experiment-side FP32 matrix product uses full FP32.
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.set_float32_matmul_precision("highest")

    dimensions = check_configuration(args)
    config_path = args.output_dir / "run_config.json"
    existing_config = read_json(config_path, {})
    canonical_arguments = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
        if key not in {"stage", "force"}
    }
    existing_arguments = existing_config.get("arguments", {})
    if existing_arguments:
        existing_arguments = {
            key: value
            for key, value in existing_arguments.items()
            if key not in {"stage", "force"}
        }
        mismatches = {
            key: {"saved": existing_arguments.get(key), "requested": value}
            for key, value in canonical_arguments.items()
            if existing_arguments.get(key) != value
        }
        if mismatches and not args.force:
            raise RuntimeError(
                "output directory was created with different experiment settings; "
                f"use a new directory (mismatches: {mismatches})"
            )
    config_payload = {
        "created_at": existing_config.get("created_at", utc_now()),
        "updated_at": utc_now(),
        "arguments": canonical_arguments,
        "last_invocation": sys.argv,
        "last_stage": args.stage,
        "model": dimensions,
        "environment": environment_metadata(device),
    }
    write_json(config_path, config_payload)

    stages = (
        ["capture", "train", "offline", "end-to-end", "report"]
        if args.stage == "all"
        else [args.stage]
    )
    for stage in stages:
        print(f"\n[{utc_now()}] ===== {stage} stage =====", flush=True)
        if stage == "capture":
            capture_stage(args, device)
        elif stage == "train":
            train_stage(args, device, dimensions)
        elif stage == "offline":
            offline_stage(args, device, dimensions)
        elif stage == "end-to-end":
            end_to_end_stage(args, device, dimensions)
        elif stage == "report":
            report_stage(args, dimensions)
    print(f"[{utc_now()}] requested stages complete", flush=True)


if __name__ == "__main__":
    main()
