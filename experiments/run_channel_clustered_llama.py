#!/usr/bin/env python3
"""Run the ClusterTurboQuant.md experiment on Llama-3.1-8B.

The runner learns layer/head-specific channel partitions from WikiText-2 train
activations, then evaluates disjoint validation/test tokens.  Quantization is
inserted at each attention layer's ``k_proj`` output, before RoPE, for downstream
perplexity and retrieval runs.  Offline metrics use the same pre-RoPE tensors
and the exact PolarQuant rotations/codebooks from ``turboquant_plus``.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import os
import platform
import sys
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch
from datasets import load_dataset
from sklearn.metrics import adjusted_rand_score
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb, repeat_kv

from turboquant.channel_cluster import ChannelClusteredTurboQuant, fit_channel_groups


DEFAULT_MODEL = "NousResearch/Meta-Llama-3.1-8B"
DEFAULT_REVISION = "1f47e50cdbe801ad8a5174156ec3a0655108fb9f"
LOGGER = logging.getLogger("cluster_turboquant")


@dataclass(frozen=True)
class Condition:
    bit_width: int
    n_clusters: int
    rotation_seed: int

    @property
    def name(self) -> str:
        return f"b{self.bit_width}_k{self.n_clusters}_seed{self.rotation_seed}"


@dataclass
class ProjectionCapture:
    keys: list[torch.Tensor]
    queries: list[torch.Tensor] | None
    values: list[torch.Tensor] | None


class LayerHeadCodecs:
    """One channel-clustered codec for every layer and KV head."""

    def __init__(
        self,
        channel_magnitudes: np.ndarray,
        condition: Condition,
        *,
        norm_bits: int,
        cluster_seed: int,
    ) -> None:
        self.condition = condition
        self.norm_bits = norm_bits
        self.cluster_seed = cluster_seed
        num_layers, num_kv_heads, _ = channel_magnitudes.shape
        self.codecs: list[list[ChannelClusteredTurboQuant]] = []
        for layer_idx in range(num_layers):
            layer_codecs = []
            for head_idx in range(num_kv_heads):
                codec_seed = condition.rotation_seed + layer_idx * 100_003 + head_idx * 1_009
                layer_codecs.append(
                    ChannelClusteredTurboQuant(
                        channel_magnitudes[layer_idx, head_idx],
                        condition.n_clusters,
                        condition.bit_width,
                        seed=codec_seed,
                        cluster_seed=cluster_seed,
                        norm_bits=norm_bits,
                    )
                )
            self.codecs.append(layer_codecs)

    @property
    def effective_bits_per_channel(self) -> float:
        return self.codecs[0][0].effective_bits_per_channel

    @property
    def static_metadata_bytes(self) -> int:
        bits = sum(codec.static_metadata_bits() for layer in self.codecs for codec in layer)
        return math.ceil(bits / 8)

    def quantize_layer(self, layer_idx: int, key_states: torch.Tensor) -> list[Any]:
        """Encode ``[batch, seq, kv_heads, head_dim]`` pre-RoPE keys."""

        return [
            codec.quantize_torch(key_states[:, :, head_idx, :])
            for head_idx, codec in enumerate(self.codecs[layer_idx])
        ]

    def dequantize_layer(self, layer_idx: int, compressed: Sequence[Any]) -> torch.Tensor:
        reconstructed = [
            codec.dequantize_torch(encoded)
            for codec, encoded in zip(self.codecs[layer_idx], compressed, strict=True)
        ]
        return torch.stack(reconstructed, dim=2)

    def reconstruct_projection(self, layer_idx: int, projected: torch.Tensor) -> torch.Tensor:
        batch, seq_len, _ = projected.shape
        num_kv_heads = len(self.codecs[layer_idx])
        head_dim = self.codecs[layer_idx][0].d
        key_states = projected.view(batch, seq_len, num_kv_heads, head_dim)
        reconstructed = self.dequantize_layer(
            layer_idx, self.quantize_layer(layer_idx, key_states)
        )
        return reconstructed.reshape_as(projected)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--revision", default=DEFAULT_REVISION)
    parser.add_argument("--output-root", type=Path, default=REPO_ROOT / "experiments" / "results")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16")
    parser.add_argument("--bit-widths", type=int, nargs="+", default=[3])
    parser.add_argument("--clusters", type=int, nargs="+", default=[1, 2, 4])
    parser.add_argument("--rotation-seeds", type=int, nargs="+", default=[11, 29, 47])
    parser.add_argument("--cluster-seed", type=int, default=0)
    parser.add_argument("--norm-bits", type=int, choices=(16, 32), default=16)
    parser.add_argument("--calibration-tokens", type=int, default=2048)
    parser.add_argument("--calibration-seq-len", type=int, default=512)
    parser.add_argument("--eval-seq-lengths", type=int, nargs="+", default=[512, 2048])
    parser.add_argument("--attention-seq-len", type=int, default=512)
    parser.add_argument("--ppl-tokens", type=int, default=1024)
    parser.add_argument("--retrieval-tokens", type=int, default=4096)
    parser.add_argument("--retrieval-positions", type=float, nargs="+", default=[0.1, 0.5, 0.9])
    parser.add_argument("--skip-ppl", action="store_true")
    parser.add_argument("--skip-retrieval", action="store_true")
    parser.add_argument(
        "--downstream-bit-widths",
        type=int,
        nargs="+",
        default=None,
        help="Only run PPL/retrieval for these widths; default is every --bit-widths value.",
    )
    parser.add_argument(
        "--max-layers",
        type=int,
        default=None,
        help="Debug-only metric limit. Downstream hooks still quantize every fitted layer.",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if min(args.bit_widths) < 1:
        raise ValueError("all bit widths must be positive")
    if sorted(set(args.clusters)) != sorted(args.clusters):
        raise ValueError("--clusters must not contain duplicates")
    if any(k < 1 for k in args.clusters):
        raise ValueError("all cluster counts must be positive")
    if args.calibration_tokens < args.calibration_seq_len:
        raise ValueError("calibration-tokens must be at least calibration-seq-len")
    if args.attention_seq_len not in args.eval_seq_lengths:
        raise ValueError("attention-seq-len must appear in eval-seq-lengths")
    if any(not 0.0 < position < 1.0 for position in args.retrieval_positions):
        raise ValueError("retrieval positions must lie strictly between 0 and 1")


def setup_output(args: argparse.Namespace) -> Path:
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_dir = args.output_root / run_id
    output_dir.mkdir(parents=True, exist_ok=False)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(output_dir / "run.log", encoding="utf-8"),
        ],
    )
    return output_dir


def torch_dtype(name: str) -> torch.dtype:
    return {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[name]


def load_model_and_tokenizer(args: argparse.Namespace) -> tuple[Any, Any]:
    LOGGER.info("Loading %s at revision %s", args.model, args.revision)
    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=args.revision, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        revision=args.revision,
        torch_dtype=torch_dtype(args.dtype),
        attn_implementation="eager",
        low_cpu_mem_usage=True,
    )
    model.to(args.device)
    model.eval()
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    LOGGER.info(
        "Loaded %d layers, %d attention heads, %d KV heads, head_dim=%d",
        model.config.num_hidden_layers,
        model.config.num_attention_heads,
        model.config.num_key_value_heads,
        model.config.hidden_size // model.config.num_attention_heads,
    )
    return model, tokenizer


def token_stream(tokenizer: Any, split: str, needed_tokens: int) -> torch.Tensor:
    LOGGER.info("Loading WikiText-2 %s token stream (%d tokens)", split, needed_tokens)
    dataset = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split=split)
    pieces: list[int] = []
    separator = tokenizer.encode("\n\n", add_special_tokens=False)
    for row in dataset:
        text = row["text"].strip()
        if not text:
            continue
        pieces.extend(tokenizer.encode(text, add_special_tokens=False))
        pieces.extend(separator)
        if len(pieces) >= needed_tokens:
            break
    if len(pieces) < needed_tokens:
        raise RuntimeError(f"WikiText-2 {split} yielded only {len(pieces)} tokens")
    return torch.tensor(pieces[:needed_tokens], dtype=torch.long)


def split_windows(stream: torch.Tensor, lengths: Sequence[int]) -> list[torch.Tensor]:
    windows = []
    offset = 0
    for length in lengths:
        windows.append(stream[offset : offset + length])
        offset += length
    if any(window.numel() != length for window, length in zip(windows, lengths, strict=True)):
        raise ValueError("token stream is too short for requested windows")
    return windows


@contextmanager
def projection_hooks(
    model: Any,
    *,
    capture_qv: bool,
) -> Iterator[ProjectionCapture]:
    num_layers = len(model.model.layers)
    keys: list[torch.Tensor | None] = [None] * num_layers
    queries: list[torch.Tensor | None] | None = [None] * num_layers if capture_qv else None
    values: list[torch.Tensor | None] | None = [None] * num_layers if capture_qv else None
    handles = []

    def save(target: list[torch.Tensor | None], layer_idx: int):
        def hook(_module: Any, _inputs: Any, output: torch.Tensor) -> None:
            target[layer_idx] = output.detach().to(device="cpu", dtype=torch.float32)

        return hook

    for layer_idx, layer in enumerate(model.model.layers):
        handles.append(layer.self_attn.k_proj.register_forward_hook(save(keys, layer_idx)))
        if capture_qv:
            assert queries is not None and values is not None
            handles.append(layer.self_attn.q_proj.register_forward_hook(save(queries, layer_idx)))
            handles.append(layer.self_attn.v_proj.register_forward_hook(save(values, layer_idx)))
    capture = ProjectionCapture(keys, queries, values)  # type: ignore[arg-type]
    try:
        yield capture
    finally:
        for handle in handles:
            handle.remove()
        if any(value is None for value in keys):
            raise RuntimeError("not every k_proj hook ran")
        if capture_qv and (
            any(value is None for value in queries or []) or any(value is None for value in values or [])
        ):
            raise RuntimeError("not every q_proj/v_proj hook ran")


def capture_projections(
    model: Any,
    token_ids: torch.Tensor,
    device: str,
    *,
    capture_qv: bool,
) -> ProjectionCapture:
    with projection_hooks(model, capture_qv=capture_qv) as capture:
        with torch.inference_mode():
            model.model(token_ids[None, :].to(device), use_cache=False, return_dict=True)
    return capture


def calibrate_channel_magnitudes(
    model: Any,
    calibration_ids: torch.Tensor,
    *,
    seq_len: int,
    device: str,
) -> np.ndarray:
    config = model.config
    num_layers = config.num_hidden_layers
    num_kv_heads = config.num_key_value_heads
    head_dim = config.hidden_size // config.num_attention_heads
    sums = np.zeros((num_layers, num_kv_heads, head_dim), dtype=np.float64)
    counts = np.zeros(num_layers, dtype=np.int64)
    handles = []

    def accumulate(layer_idx: int):
        def hook(_module: Any, _inputs: Any, output: torch.Tensor) -> None:
            values = output.detach().to(torch.float32).reshape(-1, num_kv_heads, head_dim)
            sums[layer_idx] += values.square().sum(dim=0).to(torch.float64).cpu().numpy()
            counts[layer_idx] += values.shape[0]

        return hook

    for layer_idx, layer in enumerate(model.model.layers):
        handles.append(layer.self_attn.k_proj.register_forward_hook(accumulate(layer_idx)))
    try:
        for start in range(0, calibration_ids.numel(), seq_len):
            window = calibration_ids[start : start + seq_len]
            if window.numel() == 0:
                continue
            LOGGER.info("Calibration forward: tokens %d..%d", start, start + window.numel())
            with torch.inference_mode():
                model.model(window[None, :].to(device), use_cache=False, return_dict=True)
    finally:
        for handle in handles:
            handle.remove()
    if np.any(counts == 0):
        raise RuntimeError("calibration hooks did not observe every layer")
    magnitudes = np.sqrt(sums / counts[:, None, None])
    LOGGER.info(
        "Calibration RMS range: %.6g .. %.6g from %d tokens/layer",
        magnitudes.min(),
        magnitudes.max(),
        counts.min(),
    )
    return magnitudes


def reshape_projection(
    projected: torch.Tensor,
    num_heads: int,
    head_dim: int,
) -> torch.Tensor:
    batch, seq_len, _ = projected.shape
    return projected.view(batch, seq_len, num_heads, head_dim)


def safe_unit_norm(values: torch.Tensor) -> torch.Tensor:
    norms = torch.linalg.vector_norm(values, dim=-1, keepdim=True)
    return values / torch.where(norms > 0, norms, torch.ones_like(norms))


def timed_layer_round_trip(
    codecs: LayerHeadCodecs,
    layer_idx: int,
    key_states: torch.Tensor,
) -> tuple[torch.Tensor, float, float]:
    if key_states.is_cuda:
        torch.cuda.synchronize(key_states.device)
    started = time.perf_counter()
    compressed = codecs.quantize_layer(layer_idx, key_states)
    if key_states.is_cuda:
        torch.cuda.synchronize(key_states.device)
    quantize_ms = (time.perf_counter() - started) * 1000.0
    started = time.perf_counter()
    reconstructed = codecs.dequantize_layer(layer_idx, compressed)
    if key_states.is_cuda:
        torch.cuda.synchronize(key_states.device)
    dequantize_ms = (time.perf_counter() - started) * 1000.0
    return reconstructed, quantize_ms, dequantize_ms


def evaluate_attention_layer(
    attention: Any,
    query_projection: torch.Tensor,
    original_keys: torch.Tensor,
    reconstructed_keys: torch.Tensor,
    value_projection: torch.Tensor,
) -> dict[str, Any]:
    batch, seq_len, num_query_heads, head_dim = query_projection.shape
    num_kv_heads = original_keys.shape[2]
    queries = query_projection.transpose(1, 2)
    keys = original_keys.transpose(1, 2)
    reconstructed = reconstructed_keys.transpose(1, 2)
    values = value_projection.transpose(1, 2)
    position_ids = torch.arange(seq_len, device=queries.device)[None, :].expand(batch, -1)
    cos, sin = attention.rotary_emb(values, position_ids)
    queries, keys = apply_rotary_pos_emb(queries, keys, cos, sin)
    _, reconstructed = apply_rotary_pos_emb(queries, reconstructed, cos, sin)
    kv_groups = num_query_heads // num_kv_heads
    keys = repeat_kv(keys, kv_groups).to(torch.float32)
    reconstructed = repeat_kv(reconstructed, kv_groups).to(torch.float32)
    values = repeat_kv(values, kv_groups).to(torch.float32)
    queries = queries.to(torch.float32)

    logits = torch.matmul(queries, keys.transpose(2, 3)) / math.sqrt(head_dim)
    reconstructed_logits = torch.matmul(queries, reconstructed.transpose(2, 3)) / math.sqrt(head_dim)
    valid = torch.ones((seq_len, seq_len), dtype=torch.bool, device=queries.device).tril()
    valid_4d = valid[None, None, :, :]
    differences = reconstructed_logits - logits
    valid_differences = differences.masked_select(valid_4d.expand_as(differences))
    per_head_logit_mse = (
        differences.square().masked_fill(~valid_4d, 0.0).sum(dim=(0, 2, 3))
        / (batch * valid.sum())
    )

    logits = logits.masked_fill(~valid_4d, -torch.inf)
    reconstructed_logits = reconstructed_logits.masked_fill(~valid_4d, -torch.inf)
    probabilities = torch.softmax(logits, dim=-1)
    reconstructed_probabilities = torch.softmax(reconstructed_logits, dim=-1)
    epsilon = 1e-12
    kl = (
        probabilities
        * (
            torch.log(probabilities.clamp_min(epsilon))
            - torch.log(reconstructed_probabilities.clamp_min(epsilon))
        )
    ).sum(dim=-1)
    outputs = probabilities @ values
    reconstructed_outputs = reconstructed_probabilities @ values
    return {
        "logit_squared_error_sum": float(valid_differences.square().sum().item()),
        "logit_error_sum": float(valid_differences.sum().item()),
        "logit_count": int(valid_differences.numel()),
        "probability_kl_sum": float(kl.sum().item()),
        "probability_kl_count": int(kl.numel()),
        "attention_output_squared_error_sum": float(
            (reconstructed_outputs - outputs).square().sum().item()
        ),
        "attention_output_count": int(outputs.numel()),
        "per_head_logit_mse": per_head_logit_mse.detach().cpu().numpy(),
    }


def evaluate_condition_window(
    model: Any,
    capture: ProjectionCapture,
    calibration_magnitudes: np.ndarray,
    codecs: LayerHeadCodecs,
    *,
    seq_len: int,
    attention_enabled: bool,
    device: str,
    max_layers: int | None,
) -> dict[str, Any]:
    config = model.config
    num_kv_heads = config.num_key_value_heads
    num_query_heads = config.num_attention_heads
    head_dim = config.hidden_size // config.num_attention_heads
    num_layers = min(config.num_hidden_layers, max_layers or config.num_hidden_layers)
    key_error_sum = 0.0
    key_count = 0
    normalized_error_sum = 0.0
    normalized_count = 0
    quantize_ms = 0.0
    dequantize_ms = 0.0
    observed_vectors = 0
    worst_key = (-1.0, -1, -1)
    partition_aris = []
    magnitude_drifts = []
    attention_totals = {
        "logit_squared_error_sum": 0.0,
        "logit_error_sum": 0.0,
        "logit_count": 0,
        "probability_kl_sum": 0.0,
        "probability_kl_count": 0,
        "attention_output_squared_error_sum": 0.0,
        "attention_output_count": 0,
    }
    worst_logit = (-1.0, -1, -1)

    for layer_idx in range(num_layers):
        original = reshape_projection(
            capture.keys[layer_idx].to(device), num_kv_heads, head_dim
        ).to(torch.float32)
        reconstructed, layer_quantize_ms, layer_dequantize_ms = timed_layer_round_trip(
            codecs, layer_idx, original
        )
        reconstructed = reconstructed.to(torch.float32)
        quantize_ms += layer_quantize_ms
        dequantize_ms += layer_dequantize_ms
        observed_vectors += original.shape[0] * original.shape[1] * original.shape[2]

        errors = (reconstructed - original).square()
        key_error_sum += float(errors.sum().item())
        key_count += errors.numel()
        normalized_errors = (safe_unit_norm(reconstructed) - safe_unit_norm(original)).square()
        normalized_error_sum += float(normalized_errors.sum().item())
        normalized_count += normalized_errors.numel()
        per_head_mse = errors.mean(dim=(0, 1, 3)).detach().cpu().numpy()
        head_idx = int(np.argmax(per_head_mse))
        if per_head_mse[head_idx] > worst_key[0]:
            worst_key = (float(per_head_mse[head_idx]), layer_idx, head_idx)

        heldout_magnitudes = torch.sqrt(original.square().mean(dim=(0, 1))).cpu().numpy()
        for head_idx, codec in enumerate(codecs.codecs[layer_idx]):
            heldout_labels, _, _ = fit_channel_groups(
                heldout_magnitudes[head_idx],
                codecs.condition.n_clusters,
                seed=codecs.cluster_seed,
            )
            partition_aris.append(adjusted_rand_score(codec.labels, heldout_labels))
            reference = calibration_magnitudes[layer_idx, head_idx]
            magnitude_drifts.append(
                float(np.linalg.norm(heldout_magnitudes[head_idx] - reference) / np.linalg.norm(reference))
            )

        if attention_enabled:
            if capture.queries is None or capture.values is None:
                raise ValueError("attention metrics requested without Q/V capture")
            queries = reshape_projection(
                capture.queries[layer_idx].to(device), num_query_heads, head_dim
            ).to(torch.float32)
            values = reshape_projection(
                capture.values[layer_idx].to(device), num_kv_heads, head_dim
            ).to(torch.float32)
            layer_metrics = evaluate_attention_layer(
                model.model.layers[layer_idx].self_attn,
                queries,
                original,
                reconstructed,
                values,
            )
            for key in attention_totals:
                attention_totals[key] += layer_metrics[key]
            per_head = layer_metrics["per_head_logit_mse"]
            query_head_idx = int(np.argmax(per_head))
            if per_head[query_head_idx] > worst_logit[0]:
                worst_logit = (float(per_head[query_head_idx]), layer_idx, query_head_idx)

        del original, reconstructed

    row: dict[str, Any] = {
        "condition": codecs.condition.name,
        "bit_width": codecs.condition.bit_width,
        "n_clusters": codecs.condition.n_clusters,
        "rotation_seed": codecs.condition.rotation_seed,
        "norm_bits": codecs.norm_bits,
        "effective_bits_per_channel": codecs.effective_bits_per_channel,
        "static_metadata_bytes": codecs.static_metadata_bytes,
        "eval_seq_len": seq_len,
        "layers_evaluated": num_layers,
        "key_mse": key_error_sum / key_count,
        "normalized_key_mse": normalized_error_sum / normalized_count,
        "worst_key_mse": worst_key[0],
        "worst_key_layer": worst_key[1],
        "worst_key_kv_head": worst_key[2],
        "quantize_ms": quantize_ms,
        "dequantize_ms": dequantize_ms,
        "quantize_us_per_vector": quantize_ms * 1000.0 / observed_vectors,
        "dequantize_us_per_vector": dequantize_ms * 1000.0 / observed_vectors,
        "partition_ari_mean": float(np.mean(partition_aris)),
        "partition_ari_min": float(np.min(partition_aris)),
        "magnitude_relative_drift_mean": float(np.mean(magnitude_drifts)),
        "magnitude_relative_drift_max": float(np.max(magnitude_drifts)),
    }
    if attention_enabled:
        row.update(
            {
                "attention_logit_mse": attention_totals["logit_squared_error_sum"]
                / attention_totals["logit_count"],
                "attention_logit_bias": attention_totals["logit_error_sum"]
                / attention_totals["logit_count"],
                "attention_probability_kl": attention_totals["probability_kl_sum"]
                / attention_totals["probability_kl_count"],
                "attention_output_mse": attention_totals["attention_output_squared_error_sum"]
                / attention_totals["attention_output_count"],
                "worst_attention_logit_mse": worst_logit[0],
                "worst_attention_layer": worst_logit[1],
                "worst_attention_query_head": worst_logit[2],
            }
        )
    return row


@contextmanager
def quantized_key_projections(model: Any, codecs: LayerHeadCodecs | None) -> Iterator[None]:
    if codecs is None:
        yield
        return
    handles = []

    def quantize(layer_idx: int):
        def hook(_module: Any, _inputs: Any, output: torch.Tensor) -> torch.Tensor:
            return codecs.reconstruct_projection(layer_idx, output)

        return hook

    for layer_idx, layer in enumerate(model.model.layers):
        handles.append(layer.self_attn.k_proj.register_forward_hook(quantize(layer_idx)))
    try:
        yield
    finally:
        for handle in handles:
            handle.remove()


def evaluate_perplexity(
    model: Any,
    token_ids: torch.Tensor,
    *,
    device: str,
    codecs: LayerHeadCodecs | None,
) -> tuple[float, float]:
    with quantized_key_projections(model, codecs), torch.inference_mode():
        inputs = token_ids[None, :].to(device)
        started = time.perf_counter()
        output = model(input_ids=inputs, labels=inputs, use_cache=False, return_dict=True)
        if inputs.is_cuda:
            torch.cuda.synchronize(inputs.device)
        elapsed = time.perf_counter() - started
    loss = float(output.loss.item())
    return math.exp(loss), elapsed


def make_retrieval_prompt(
    tokenizer: Any,
    filler_ids: torch.Tensor,
    *,
    total_tokens: int,
    position: float,
    access_word: str,
) -> torch.Tensor:
    prefix = tokenizer.encode(
        "Document follows. Read it carefully and answer the final question.\n\n",
        add_special_tokens=False,
    )
    needle = tokenizer.encode(
        f"\nPASSKEY RECORD: The accessword is {access_word}. Remember this exact accessword.\n",
        add_special_tokens=False,
    )
    suffix = tokenizer.encode(
        "\n\nQuestion: According to the PASSKEY RECORD, the accessword is",
        add_special_tokens=False,
    )
    available = total_tokens - len(prefix) - len(needle) - len(suffix)
    if available <= 0:
        raise ValueError("retrieval-tokens is too short for prompt scaffolding")
    if filler_ids.numel() < available:
        repeats = math.ceil(available / filler_ids.numel())
        filler_ids = filler_ids.repeat(repeats)
    filler = filler_ids[:available].tolist()
    insert_at = round(len(filler) * position)
    ids = prefix + filler[:insert_at] + needle + filler[insert_at:] + suffix
    return torch.tensor(ids, dtype=torch.long)


def evaluate_retrieval(
    model: Any,
    tokenizer: Any,
    filler_ids: torch.Tensor,
    positions: Sequence[float],
    *,
    total_tokens: int,
    device: str,
    codecs: LayerHeadCodecs | None,
) -> tuple[int, list[dict[str, Any]], float]:
    candidate_texts = [" banana", " apple", " orange", " tiger", " cedar", " quartz"]
    encoded_candidates = [
        tokenizer.encode(candidate, add_special_tokens=False) for candidate in candidate_texts
    ]
    if any(len(tokens) != 1 for tokens in encoded_candidates):
        raise RuntimeError("retrieval candidate set must contain one-token answers")
    candidate_ids = torch.tensor(
        [tokens[0] for tokens in encoded_candidates], dtype=torch.long, device=device
    )
    details = []
    passed = 0
    elapsed_total = 0.0
    for index, position in enumerate(positions):
        correct_index = index % len(candidate_texts)
        access_word = candidate_texts[correct_index].strip()
        prompt = make_retrieval_prompt(
            tokenizer,
            filler_ids,
            total_tokens=total_tokens,
            position=position,
            access_word=access_word,
        )
        with quantized_key_projections(model, codecs), torch.inference_mode():
            inputs = prompt[None, :].to(device)
            started = time.perf_counter()
            hidden = model.model(
                input_ids=inputs,
                attention_mask=torch.ones_like(inputs),
                use_cache=False,
                return_dict=True,
            )
            logits = model.lm_head(hidden.last_hidden_state[:, -1, :]).to(torch.float32)
            candidate_scores = logits[0].index_select(0, candidate_ids)
            if inputs.is_cuda:
                torch.cuda.synchronize(inputs.device)
            elapsed = time.perf_counter() - started
        predicted_index = int(candidate_scores.argmax().item())
        success = predicted_index == correct_index
        passed += int(success)
        elapsed_total += elapsed
        details.append(
            {
                "position": position,
                "access_word": access_word,
                "predicted_word": candidate_texts[predicted_index].strip(),
                "success": success,
                "candidate_logit_scores": {
                    candidate.strip(): float(score)
                    for candidate, score in zip(
                        candidate_texts, candidate_scores.detach().cpu().tolist(), strict=True
                    )
                },
                "elapsed_seconds": elapsed,
            }
        )
    return passed, details, elapsed_total


def write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    if not rows:
        return
    fieldnames = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def numeric_summary(
    rows: Sequence[dict[str, Any]],
    group_keys: Sequence[str],
) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(tuple(row[key] for key in group_keys), []).append(row)
    summaries = []
    for group, group_rows in sorted(grouped.items()):
        summary = dict(zip(group_keys, group, strict=True))
        summary["n_rotation_seeds"] = len(group_rows)
        numeric_keys = [
            key
            for key, value in group_rows[0].items()
            if key not in group_keys and isinstance(value, (int, float)) and not isinstance(value, bool)
        ]
        for key in numeric_keys:
            values = [float(row[key]) for row in group_rows if key in row and row[key] is not None]
            if values:
                summary[f"{key}_mean"] = float(np.mean(values))
                summary[f"{key}_std"] = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
        summaries.append(summary)
    return summaries


def markdown_table(headers: Sequence[str], rows: Iterable[Sequence[Any]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    lines.extend("| " + " | ".join(str(value) for value in row) + " |" for row in rows)
    return "\n".join(lines)


def render_report(
    args: argparse.Namespace,
    metric_summary: Sequence[dict[str, Any]],
    downstream_summary: Sequence[dict[str, Any]],
    baseline: dict[str, Any],
) -> str:
    metric_rows = []
    for row in metric_summary:
        metric_rows.append(
            (
                row["bit_width"],
                row["n_clusters"],
                f"{row['effective_bits_per_channel_mean']:.3f}",
                row["eval_seq_len"],
                f"{row['key_mse_mean']:.4e} +/- {row['key_mse_std']:.1e}",
                f"{row['normalized_key_mse_mean']:.4e}",
                f"{row.get('attention_logit_mse_mean', float('nan')):.4e}",
                f"{row.get('attention_probability_kl_mean', float('nan')):.4e}",
                f"{row['partition_ari_mean_mean']:.3f}",
            )
        )
    sections = [
        "# Channel-Clustered TurboQuant experiment",
        "",
        f"Run completed: {datetime.now(timezone.utc).isoformat()}",
        "",
        "Calibration used WikiText-2 train tokens. Every reported evaluation uses disjoint "
        "WikiText-2 validation/test tokens. Quantization occurs at `k_proj`, before RoPE.",
        "",
        "## Reconstruction and attention",
        "",
        markdown_table(
            ["bits", "K", "effective b/ch", "seq", "key MSE", "normalized MSE", "logit MSE", "prob KL", "partition ARI"],
            metric_rows,
        ),
        "",
        "Effective rate includes all fp16/fp32 group norms. Rotation, codebook, and partition "
        "metadata are reported separately in the CSV because they are static across tokens.",
    ]
    if baseline:
        sections.extend(
            [
                "",
                "## Downstream baseline",
                "",
                f"Unquantized perplexity: {baseline.get('perplexity', 'not run')}",
                "",
                f"Unquantized retrieval: {baseline.get('retrieval_passed', 'not run')}/"
                f"{baseline.get('retrieval_total', 'not run')}",
            ]
        )
    if downstream_summary:
        downstream_rows = []
        for row in downstream_summary:
            downstream_rows.append(
                (
                    row["bit_width"],
                    row["n_clusters"],
                    f"{row['effective_bits_per_channel_mean']:.3f}",
                    f"{row.get('perplexity_mean', float('nan')):.4f}",
                    f"{row.get('retrieval_accuracy_mean', float('nan')):.3f}",
                    f"{row.get('downstream_seconds_mean', float('nan')):.2f}",
                )
            )
        sections.extend(
            [
                "",
                "## Quantized downstream results",
                "",
                markdown_table(
                    ["bits", "K", "effective b/ch", "PPL", "retrieval", "seconds"],
                    downstream_rows,
                ),
            ]
        )
    comparison_rows = []
    bits = sorted({int(row["bit_width"]) for row in metric_summary})
    for bit_width in bits:
        matching = [
            row
            for row in metric_summary
            if int(row["bit_width"]) == bit_width
            and int(row["eval_seq_len"]) == args.attention_seq_len
        ]
        baseline_metric = next(
            (row for row in matching if int(row["n_clusters"]) == 1), None
        )
        baseline_downstream = next(
            (
                row
                for row in downstream_summary
                if int(row["bit_width"]) == bit_width and int(row["n_clusters"]) == 1
            ),
            None,
        )
        if baseline_metric is None:
            continue
        for row in matching:
            if int(row["n_clusters"]) == 1:
                continue
            downstream = next(
                (
                    item
                    for item in downstream_summary
                    if int(item["bit_width"]) == bit_width
                    and int(item["n_clusters"]) == int(row["n_clusters"])
                ),
                None,
            )
            ppl_delta = float("nan")
            retrieval = float("nan")
            if downstream is not None and baseline_downstream is not None:
                if "perplexity_mean" in downstream and "perplexity_mean" in baseline_downstream:
                    ppl_delta = 100.0 * (
                        downstream["perplexity_mean"] / baseline_downstream["perplexity_mean"] - 1.0
                    )
                retrieval = downstream.get("retrieval_accuracy_mean", float("nan"))
            comparison_rows.append(
                (
                    bit_width,
                    int(row["n_clusters"]),
                    f"{100.0 * (row['effective_bits_per_channel_mean'] / baseline_metric['effective_bits_per_channel_mean'] - 1.0):+.1f}%",
                    f"{100.0 * (row['key_mse_mean'] / baseline_metric['key_mse_mean'] - 1.0):+.1f}%",
                    f"{100.0 * (row['attention_logit_mse_mean'] / baseline_metric['attention_logit_mse_mean'] - 1.0):+.1f}%",
                    f"{ppl_delta:+.1f}%",
                    f"{retrieval:.3f}",
                )
            )
    if comparison_rows:
        sections.extend(
            [
                "",
                "## Result against K=1",
                "",
                markdown_table(
                    ["bits", "K", "rate delta", "key MSE delta", "logit MSE delta", "PPL delta", "retrieval"],
                    comparison_rows,
                ),
                "",
                "In this run, channel grouping improves reconstruction but does not satisfy the "
                "main validation criterion. Every grouped condition uses more dynamic storage than "
                "K=1 and has worse attention-logit MSE; mean downstream perplexity does not improve. "
                "The hypothesis is therefore not supported for this 3-bit Llama-3.1-8B setting. "
                "This is a scoped experimental conclusion, not a claim about other bit widths, "
                "calibration sets, or grouping objectives.",
            ]
        )
    sections.extend(
        [
            "",
            "## Interpretation guardrail",
            "",
            "K=1, K=2, and K=4 use the same scalar bit width and norm precision, but their "
            "effective rates differ because K group norms are stored per token. A same-bit "
            "improvement is therefore not by itself proof of an equal-storage improvement. "
            "Use the effective-rate column and, when multiple bit widths are run, the Pareto "
            "frontier before accepting the proposal's main validation criterion.",
            "",
            "Raw per-seed values, worst layer/head errors, latency, and retrieval candidate scores are "
            "in `metrics.csv`, `downstream.csv`, and `results.json`.",
        ]
    )
    return "\n".join(sections) + "\n"


def environment_record(model: Any, args: argparse.Namespace) -> dict[str, Any]:
    import datasets
    import scipy
    import sklearn
    import transformers

    return {
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "transformers": transformers.__version__,
        "datasets": datasets.__version__,
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "scikit_learn": sklearn.__version__,
        "model_name": args.model,
        "requested_revision": args.revision,
        "loaded_commit_hash": getattr(model.config, "_commit_hash", None),
        "conda_prefix": os.environ.get("CONDA_PREFIX"),
    }


def main() -> None:
    args = parse_args()
    validate_args(args)
    output_dir = setup_output(args)
    LOGGER.info("Writing results to %s", output_dir)
    (output_dir / "config.json").write_text(
        json.dumps(vars(args), indent=2, default=str) + "\n", encoding="utf-8"
    )

    model, tokenizer = load_model_and_tokenizer(args)
    (output_dir / "environment.json").write_text(
        json.dumps(environment_record(model, args), indent=2) + "\n", encoding="utf-8"
    )
    config = model.config
    head_dim = config.hidden_size // config.num_attention_heads
    if max(args.clusters) > head_dim:
        raise ValueError(f"cluster count exceeds head dimension {head_dim}")

    calibration_ids = token_stream(tokenizer, "train", args.calibration_tokens)
    calibration_magnitudes = calibrate_channel_magnitudes(
        model,
        calibration_ids,
        seq_len=args.calibration_seq_len,
        device=args.device,
    )
    np.save(output_dir / "calibration_magnitudes.npy", calibration_magnitudes)

    partition_payload = {"calibration_magnitudes": calibration_magnitudes}
    for n_clusters in args.clusters:
        labels = np.empty_like(calibration_magnitudes, dtype=np.int16)
        for layer_idx in range(config.num_hidden_layers):
            for head_idx in range(config.num_key_value_heads):
                labels[layer_idx, head_idx], _, _ = fit_channel_groups(
                    calibration_magnitudes[layer_idx, head_idx],
                    n_clusters,
                    seed=args.cluster_seed,
                )
        partition_payload[f"labels_k{n_clusters}"] = labels
    np.savez_compressed(output_dir / "partitions.npz", **partition_payload)

    eval_stream = token_stream(tokenizer, "validation", sum(args.eval_seq_lengths))
    eval_windows = split_windows(eval_stream, args.eval_seq_lengths)
    captures: dict[int, ProjectionCapture] = {}
    for seq_len, token_ids in zip(args.eval_seq_lengths, eval_windows, strict=True):
        LOGGER.info("Capturing held-out pre-RoPE projections at sequence length %d", seq_len)
        captures[seq_len] = capture_projections(
            model,
            token_ids,
            args.device,
            capture_qv=seq_len == args.attention_seq_len,
        )

    conditions = [
        Condition(bit_width, n_clusters, rotation_seed)
        for bit_width in args.bit_widths
        for n_clusters in args.clusters
        for rotation_seed in args.rotation_seeds
    ]
    metric_rows = []
    codec_cache: dict[Condition, LayerHeadCodecs] = {}
    for condition in conditions:
        LOGGER.info("Offline evaluation: %s", condition.name)
        codecs = LayerHeadCodecs(
            calibration_magnitudes,
            condition,
            norm_bits=args.norm_bits,
            cluster_seed=args.cluster_seed,
        )
        codec_cache[condition] = codecs
        for seq_len in args.eval_seq_lengths:
            row = evaluate_condition_window(
                model,
                captures[seq_len],
                calibration_magnitudes,
                codecs,
                seq_len=seq_len,
                attention_enabled=seq_len == args.attention_seq_len,
                device=args.device,
                max_layers=args.max_layers,
            )
            metric_rows.append(row)
            LOGGER.info(
                "%s seq=%d key_mse=%.6g logit_mse=%s effective=%.3f b/ch",
                condition.name,
                seq_len,
                row["key_mse"],
                f"{row['attention_logit_mse']:.6g}" if "attention_logit_mse" in row else "n/a",
                row["effective_bits_per_channel"],
            )
    write_csv(output_dir / "metrics.csv", metric_rows)
    metric_summary = numeric_summary(
        metric_rows, ["bit_width", "n_clusters", "norm_bits", "eval_seq_len"]
    )
    write_csv(output_dir / "metrics_summary.csv", metric_summary)

    downstream_widths = set(args.downstream_bit_widths or args.bit_widths)
    test_needed = max(args.ppl_tokens, args.retrieval_tokens)
    test_ids = token_stream(tokenizer, "test", test_needed)
    baseline: dict[str, Any] = {}
    if not args.skip_ppl:
        LOGGER.info("Running unquantized perplexity baseline")
        baseline_ppl, baseline_seconds = evaluate_perplexity(
            model, test_ids[: args.ppl_tokens], device=args.device, codecs=None
        )
        baseline.update(perplexity=baseline_ppl, perplexity_seconds=baseline_seconds)
        LOGGER.info("Unquantized PPL %.6f", baseline_ppl)
    if not args.skip_retrieval:
        LOGGER.info("Running unquantized long-context retrieval baseline")
        passed, details, elapsed = evaluate_retrieval(
            model,
            tokenizer,
            test_ids,
            args.retrieval_positions,
            total_tokens=args.retrieval_tokens,
            device=args.device,
            codecs=None,
        )
        baseline.update(
            retrieval_passed=passed,
            retrieval_total=len(args.retrieval_positions),
            retrieval_details=details,
            retrieval_seconds=elapsed,
        )
        LOGGER.info("Unquantized retrieval %d/%d", passed, len(args.retrieval_positions))

    downstream_rows = []
    for condition in conditions:
        if condition.bit_width not in downstream_widths:
            continue
        codecs = codec_cache[condition]
        LOGGER.info("Downstream evaluation: %s", condition.name)
        row: dict[str, Any] = {
            "condition": condition.name,
            **asdict(condition),
            "norm_bits": args.norm_bits,
            "effective_bits_per_channel": codecs.effective_bits_per_channel,
        }
        elapsed = 0.0
        if not args.skip_ppl:
            ppl, ppl_seconds = evaluate_perplexity(
                model, test_ids[: args.ppl_tokens], device=args.device, codecs=codecs
            )
            row.update(
                perplexity=ppl,
                perplexity_delta=ppl - baseline["perplexity"],
                perplexity_relative_delta=ppl / baseline["perplexity"] - 1.0,
                perplexity_seconds=ppl_seconds,
            )
            elapsed += ppl_seconds
        if not args.skip_retrieval:
            passed, details, retrieval_seconds = evaluate_retrieval(
                model,
                tokenizer,
                test_ids,
                args.retrieval_positions,
                total_tokens=args.retrieval_tokens,
                device=args.device,
                codecs=codecs,
            )
            row.update(
                retrieval_passed=passed,
                retrieval_total=len(args.retrieval_positions),
                retrieval_accuracy=passed / len(args.retrieval_positions),
                retrieval_details=details,
                retrieval_seconds=retrieval_seconds,
            )
            elapsed += retrieval_seconds
        row["downstream_seconds"] = elapsed
        downstream_rows.append(row)
        LOGGER.info(
            "%s PPL=%s retrieval=%s",
            condition.name,
            f"{row['perplexity']:.6f}" if "perplexity" in row else "skipped",
            f"{row['retrieval_passed']}/{row['retrieval_total']}"
            if "retrieval_passed" in row
            else "skipped",
        )
    csv_downstream_rows = [
        {key: value for key, value in row.items() if key != "retrieval_details"}
        for row in downstream_rows
    ]
    write_csv(output_dir / "downstream.csv", csv_downstream_rows)
    downstream_summary = numeric_summary(
        csv_downstream_rows, ["bit_width", "n_clusters", "norm_bits"]
    )
    write_csv(output_dir / "downstream_summary.csv", downstream_summary)

    results = {
        "config": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "baseline": baseline,
        "metrics": metric_rows,
        "metric_summary": metric_summary,
        "downstream": downstream_rows,
        "downstream_summary": downstream_summary,
    }
    (output_dir / "results.json").write_text(
        json.dumps(results, indent=2) + "\n", encoding="utf-8"
    )
    report = render_report(args, metric_summary, downstream_summary, baseline)
    (output_dir / "report.md").write_text(report, encoding="utf-8")
    LOGGER.info("Experiment complete: %s", output_dir)


if __name__ == "__main__":
    main()
