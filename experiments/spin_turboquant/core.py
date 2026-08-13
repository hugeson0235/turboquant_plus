"""Core math and metrics for head-wise learned TurboQuant rotations.

The implementation deliberately imports the random Haar rotation and Lloyd-Max
codebook from this checkout's :mod:`turboquant` package.  The only new codec
parameter is a dense orthogonal matrix for each layer/KV head.

Tensor conventions used in this module:

* a rotation has shape ``(..., d, d)`` and acts on a column vector as ``R @ x``;
* batched vectors use row-vector storage, so the forward rotation is ``x @ R.T``;
* captured keys use ``(layers, kv_heads, tokens, head_dim)``;
* captured attention projections use ``(layers, sequence, heads, head_dim)``.
"""

from __future__ import annotations

import math
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from typing import Callable, Iterator

import numpy as np
import torch

from turboquant.codebook import optimal_centroids
from turboquant.rotation import random_rotation_dense


@dataclass
class TrainingStats:
    """Summary emitted by one bit-width/seed rotation optimization."""

    initial_loss: float
    final_loss: float
    relative_improvement: float
    orthogonality_max_abs: float
    elapsed_seconds: float
    steps: int
    learning_rate: float
    batch_tokens: int

    def to_dict(self) -> dict[str, float | int]:
        return asdict(self)


def codebook_tensor(
    bit_width: int,
    head_dim: int,
    *,
    device: torch.device | str,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Return the exact TurboQuant+ Lloyd-Max codebook as a Torch tensor."""

    values = optimal_centroids(bit_width, head_dim)
    return torch.as_tensor(values, device=device, dtype=dtype)


def build_random_rotations(
    num_layers: int,
    num_kv_heads: int,
    head_dim: int,
    seed: int,
) -> torch.Tensor:
    """Draw head-specific Haar rotations with TurboQuant+'s implementation."""

    rng = np.random.default_rng(seed)
    rotations = np.empty(
        (num_layers, num_kv_heads, head_dim, head_dim), dtype=np.float32
    )
    for layer in range(num_layers):
        for head in range(num_kv_heads):
            rotations[layer, head] = random_rotation_dense(head_dim, rng).astype(
                np.float32, copy=False
            )
    return torch.from_numpy(rotations)


def cayley_rotation(parameters: torch.Tensor, initial: torch.Tensor) -> torch.Tensor:
    """Construct ``(I-A)(I+A)^-1 R0`` with ``A = B-B.T``.

    A transposed linear solve is used so the multiplication order exactly
    matches the specification; no explicit matrix inverse is formed.
    """

    if parameters.shape != initial.shape or parameters.shape[-1] != parameters.shape[-2]:
        raise ValueError(
            "parameters and initial must have the same (..., d, d) shape; "
            f"got {tuple(parameters.shape)} and {tuple(initial.shape)}"
        )
    skew = parameters - parameters.transpose(-1, -2)
    identity = torch.eye(
        skew.shape[-1], dtype=skew.dtype, device=skew.device
    ).expand_as(skew)
    left = identity - skew
    right = identity + skew
    # C.T = solve((I+A).T, (I-A).T), hence C = (I-A)(I+A)^-1.
    cayley = torch.linalg.solve(
        right.transpose(-1, -2), left.transpose(-1, -2)
    ).transpose(-1, -2)
    return cayley @ initial


def quantize_to_centroids(values: torch.Tensor, centroids: torch.Tensor) -> torch.Tensor:
    """Nearest-centroid scalar quantization matching ``turboquant.codebook``."""

    if centroids.ndim != 1 or centroids.numel() < 2:
        raise ValueError("centroids must be a sorted one-dimensional tensor")
    boundaries = (centroids[:-1] + centroids[1:]) * 0.5
    indices = torch.bucketize(values.contiguous(), boundaries)
    return centroids[indices]


def apply_codec(
    vectors: torch.Tensor,
    rotations: torch.Tensor,
    centroids: torch.Tensor,
    *,
    norm_correction: bool = False,
) -> torch.Tensor:
    """Quantize and reconstruct head-batched vectors.

    Args:
        vectors: ``(heads, tokens, d)`` original-scale vectors.
        rotations: ``(heads, d, d)`` orthogonal forward rotations.
        centroids: fixed TurboQuant codebook.
        norm_correction: If true, reproduce TurboQuant+'s production
            reconstructed-norm correction.  The SpinTurboQuant specification's
            objective and default path leave this false.
    """

    if vectors.ndim != 3 or rotations.ndim != 3:
        raise ValueError("vectors and rotations must have shapes (h,t,d) and (h,d,d)")
    if vectors.shape[0] != rotations.shape[0] or vectors.shape[-1] != rotations.shape[-1]:
        raise ValueError(
            f"incompatible vectors {tuple(vectors.shape)} and rotations {tuple(rotations.shape)}"
        )

    norms = torch.linalg.vector_norm(vectors, dim=-1, keepdim=True)
    normalized = vectors / norms.clamp_min(torch.finfo(vectors.dtype).eps)
    rotated = torch.matmul(normalized, rotations.transpose(-1, -2))
    reconstructed_rotated = quantize_to_centroids(rotated, centroids)
    if norm_correction:
        reconstructed_rotated = reconstructed_rotated / torch.linalg.vector_norm(
            reconstructed_rotated, dim=-1, keepdim=True
        ).clamp_min(torch.finfo(reconstructed_rotated.dtype).eps)
    reconstructed_normalized = torch.matmul(reconstructed_rotated, rotations)
    reconstructed = reconstructed_normalized * norms
    # Preserve exact zeros rather than mapping them to non-zero centroids.
    return torch.where(norms > 0, reconstructed, torch.zeros_like(reconstructed))


def _codec_objective(
    normalized: torch.Tensor,
    rotations: torch.Tensor,
    centroids: torch.Tensor,
    *,
    chunk_tokens: int = 1024,
) -> float:
    """Evaluate the document's rotated-space MSE objective."""

    total_error = 0.0
    total_values = 0
    with torch.inference_mode():
        for start in range(0, normalized.shape[1], chunk_tokens):
            batch = normalized[:, start : start + chunk_tokens]
            rotated = torch.matmul(batch, rotations.transpose(-1, -2))
            reconstructed = quantize_to_centroids(rotated, centroids)
            total_error += torch.sum((rotated - reconstructed) ** 2).item()
            total_values += rotated.numel()
    return total_error / max(total_values, 1)


def train_headwise_rotations(
    keys: torch.Tensor,
    initial_rotations: torch.Tensor,
    centroids: torch.Tensor,
    *,
    steps: int,
    batch_tokens: int,
    learning_rate: float,
    optimizer_seed: int,
    progress: Callable[[int, int, float], None] | None = None,
) -> tuple[torch.Tensor, TrainingStats]:
    """Train all layer/head rotations independently in one batched optimizer.

    ``keys`` and ``initial_rotations`` are flattened over layer/head and have
    shapes ``(heads, tokens, d)`` and ``(heads, d, d)``.  The original LLM and
    codebook never participate in optimization.
    """

    if steps < 1:
        raise ValueError("steps must be positive")
    if keys.ndim != 3 or initial_rotations.ndim != 3:
        raise ValueError("keys and rotations must be three-dimensional")
    if keys.shape[0] != initial_rotations.shape[0] or keys.shape[-1] != initial_rotations.shape[-1]:
        raise ValueError("keys and rotations have incompatible head or dimension counts")

    device = initial_rotations.device
    keys = keys.to(device=device, dtype=torch.float32)
    initial_rotations = initial_rotations.to(device=device, dtype=torch.float32)
    norms = torch.linalg.vector_norm(keys, dim=-1, keepdim=True)
    normalized = keys / norms.clamp_min(torch.finfo(torch.float32).eps)
    initial_loss = _codec_objective(normalized, initial_rotations, centroids)

    parameters = torch.nn.Parameter(torch.zeros_like(initial_rotations))
    optimizer = torch.optim.Adam(params=[parameters], lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=steps, eta_min=learning_rate * 0.05
    )
    generator = torch.Generator(device=device)
    generator.manual_seed(optimizer_seed)
    token_count = normalized.shape[1]
    actual_batch = min(batch_tokens, token_count)

    started = time.perf_counter()
    last_loss = initial_loss
    for step in range(1, steps + 1):
        if actual_batch == token_count:
            batch = normalized
        else:
            indices = torch.randint(
                token_count, (actual_batch,), generator=generator, device=device
            )
            batch = normalized[:, indices]

        optimizer.zero_grad(set_to_none=True)
        rotations = cayley_rotation(parameters, initial_rotations)
        rotated = torch.matmul(batch, rotations.transpose(-1, -2))
        # The selected centroid is a fixed target for this optimization step.
        targets = quantize_to_centroids(rotated.detach(), centroids)
        loss = torch.mean((rotated - targets) ** 2)
        loss.backward()
        torch.nn.utils.clip_grad_norm_([parameters], max_norm=1.0)
        optimizer.step()
        scheduler.step()
        last_loss = float(loss.detach())
        if progress is not None:
            progress(step, steps, last_loss)

    with torch.inference_mode():
        # Batched CUDA solves may use reduced-precision internal paths.  Form
        # the persisted codec matrix once in FP64 so the advertised Cayley
        # orthogonality is not weakened by a TF32/FP32 solve artifact, then
        # store the result in FP32 as specified by the experiment.
        learned = cayley_rotation(
            parameters.detach().double(), initial_rotations.double()
        ).float()
    elapsed = time.perf_counter() - started
    final_loss = _codec_objective(normalized, learned, centroids)
    learned_double = learned.double()
    identity = torch.eye(
        learned.shape[-1], dtype=torch.float64, device=learned.device
    )
    gram_error = learned_double.transpose(-1, -2) @ learned_double - identity
    orthogonality_max_abs = float(gram_error.abs().max())
    stats = TrainingStats(
        initial_loss=initial_loss,
        final_loss=final_loss,
        relative_improvement=(initial_loss - final_loss) / max(initial_loss, 1e-30),
        orthogonality_max_abs=orthogonality_max_abs,
        elapsed_seconds=elapsed,
        steps=steps,
        learning_rate=learning_rate,
        batch_tokens=actual_batch,
    )
    del parameters, optimizer, scheduler, normalized, keys, learned_double
    return learned, stats


def reconstruction_metrics(
    keys: torch.Tensor,
    rotations: torch.Tensor,
    centroids: torch.Tensor,
    *,
    norm_correction: bool = False,
    chunk_tokens: int = 256,
    return_per_head: bool = False,
) -> dict[str, object]:
    """Measure normalized/original-scale error, including worst layer/head."""

    if keys.ndim != 4:
        raise ValueError("keys must have shape (layers, heads, tokens, d)")
    layers, heads, tokens, head_dim = keys.shape
    if rotations.shape != (layers, heads, head_dim, head_dim):
        raise ValueError(
            f"rotation shape {tuple(rotations.shape)} does not match keys {tuple(keys.shape)}"
        )
    device = rotations.device
    flat_rotations = rotations.reshape(layers * heads, head_dim, head_dim).float()
    normalized_sse = torch.zeros(layers * heads, dtype=torch.float64)
    original_sse = torch.zeros(layers * heads, dtype=torch.float64)
    original_energy = torch.zeros(layers * heads, dtype=torch.float64)
    cosine_sum = torch.zeros(layers * heads, dtype=torch.float64)
    vector_count = 0

    with torch.inference_mode():
        for start in range(0, tokens, chunk_tokens):
            stop = min(start + chunk_tokens, tokens)
            batch = (
                keys[:, :, start:stop]
                .reshape(layers * heads, stop - start, head_dim)
                .to(device=device, dtype=torch.float32)
            )
            norms = torch.linalg.vector_norm(batch, dim=-1, keepdim=True)
            normalized = batch / norms.clamp_min(torch.finfo(torch.float32).eps)
            reconstructed = apply_codec(
                batch,
                flat_rotations,
                centroids,
                norm_correction=norm_correction,
            )
            reconstructed_normalized = reconstructed / norms.clamp_min(
                torch.finfo(torch.float32).eps
            )
            valid = norms.squeeze(-1) > 0
            normalized_error = torch.where(
                valid.unsqueeze(-1),
                normalized - reconstructed_normalized,
                torch.zeros_like(normalized),
            )
            normalized_sse += (
                normalized_error.square().sum(dim=(1, 2)).double().cpu()
            )
            original_sse += ((batch - reconstructed).square().sum(dim=(1, 2))).double().cpu()
            original_energy += batch.square().sum(dim=(1, 2)).double().cpu()
            cosine = torch.nn.functional.cosine_similarity(
                batch, reconstructed, dim=-1, eps=1e-12
            )
            cosine_sum += torch.where(valid, cosine, torch.ones_like(cosine)).sum(dim=1).double().cpu()
            vector_count += stop - start

    per_head_mse = normalized_sse / max(vector_count * head_dim, 1)
    worst_flat = int(torch.argmax(per_head_mse))
    total_values = layers * heads * vector_count * head_dim
    result: dict[str, object] = {
        "normalized_key_mse": float(normalized_sse.sum() / max(total_values, 1)),
        "original_key_mse": float(original_sse.sum() / max(total_values, 1)),
        "original_key_relative_mse": float(
            original_sse.sum() / original_energy.sum().clamp_min(1e-30)
        ),
        "mean_key_cosine": float(cosine_sum.sum() / max(layers * heads * vector_count, 1)),
        "worst_head_normalized_mse": float(per_head_mse[worst_flat]),
        "worst_layer": worst_flat // heads,
        "worst_head": worst_flat % heads,
        "vectors_per_head": vector_count,
    }
    if return_per_head:
        original_head_mse = original_sse / max(vector_count * head_dim, 1)
        original_head_relative_mse = original_sse / original_energy.clamp_min(1e-30)
        result["_per_head"] = {
            "normalized_key_mse": per_head_mse.reshape(layers, heads).tolist(),
            "original_key_mse": original_head_mse.reshape(layers, heads).tolist(),
            "original_key_relative_mse": original_head_relative_mse.reshape(
                layers, heads
            ).tolist(),
        }
    return result


def rotate_half(values: torch.Tensor) -> torch.Tensor:
    """Llama rotary helper, kept local to make the metric path explicit."""

    first, second = values.chunk(2, dim=-1)
    return torch.cat((-second, first), dim=-1)


def _apply_rope(values: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    return values * cos.unsqueeze(0) + rotate_half(values) * sin.unsqueeze(0)


def attention_distortion_metrics(
    queries: torch.Tensor,
    keys: torch.Tensor,
    values: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    rotations: torch.Tensor,
    centroids: torch.Tensor,
    *,
    sequence_length: int,
    norm_correction: bool = False,
) -> dict[str, float | int]:
    """Compute causal GQA attention distortion from captured pre-RoPE Q/K/V."""

    if queries.ndim != 4 or keys.ndim != 4 or values.ndim != 4:
        raise ValueError("Q/K/V must have shapes (layers, sequence, heads, d)")
    layers, available, query_heads, head_dim = queries.shape
    key_layers, key_available, kv_heads, key_dim = keys.shape
    if (key_layers, key_available, key_dim) != (layers, available, head_dim):
        raise ValueError("Q and K capture shapes are incompatible")
    if values.shape != keys.shape:
        raise ValueError("K and V capture shapes must match")
    if query_heads % kv_heads != 0:
        raise ValueError("query head count must be divisible by KV head count")
    if not 1 <= sequence_length <= available:
        raise ValueError("sequence_length exceeds the captured sequence")

    device = rotations.device
    length = sequence_length
    rope_cos = cos[:length].to(device=device, dtype=torch.float32)
    rope_sin = sin[:length].to(device=device, dtype=torch.float32)
    causal = torch.ones((length, length), dtype=torch.bool, device=device).tril()
    valid_logits = query_heads * length * (length + 1) // 2
    logit_error_sum = 0.0
    logit_error_sq_sum = 0.0
    kl_sum = 0.0
    output_error_sum = 0.0
    layer_kls: list[float] = []
    repeats = query_heads // kv_heads

    with torch.inference_mode():
        for layer in range(layers):
            q = queries[layer, :length].permute(1, 0, 2).to(device=device, dtype=torch.float32)
            k = keys[layer, :length].permute(1, 0, 2).to(device=device, dtype=torch.float32)
            v = values[layer, :length].permute(1, 0, 2).to(device=device, dtype=torch.float32)
            reconstructed_k = apply_codec(
                k,
                rotations[layer].float(),
                centroids,
                norm_correction=norm_correction,
            )

            q_rope = _apply_rope(q, rope_cos, rope_sin)
            k_rope = _apply_rope(k, rope_cos, rope_sin).repeat_interleave(repeats, dim=0)
            reconstructed_k_rope = _apply_rope(
                reconstructed_k, rope_cos, rope_sin
            ).repeat_interleave(repeats, dim=0)
            repeated_v = v.repeat_interleave(repeats, dim=0)

            scale = 1.0 / math.sqrt(head_dim)
            logits = torch.matmul(q_rope, k_rope.transpose(-1, -2)) * scale
            reconstructed_logits = (
                torch.matmul(q_rope, reconstructed_k_rope.transpose(-1, -2)) * scale
            )
            error = reconstructed_logits - logits
            valid_error = error.masked_select(causal.unsqueeze(0))
            logit_error_sum += float(valid_error.sum())
            logit_error_sq_sum += float(valid_error.square().sum())

            logits = logits.masked_fill(~causal.unsqueeze(0), float("-inf"))
            reconstructed_logits = reconstructed_logits.masked_fill(
                ~causal.unsqueeze(0), float("-inf")
            )
            probabilities = torch.softmax(logits, dim=-1)
            reconstructed_probabilities = torch.softmax(reconstructed_logits, dim=-1)
            kl = torch.sum(
                probabilities
                * (
                    torch.log(probabilities.clamp_min(1e-12))
                    - torch.log(reconstructed_probabilities.clamp_min(1e-12))
                ),
                dim=-1,
            )
            layer_kl = float(kl.mean())
            layer_kls.append(layer_kl)
            kl_sum += float(kl.sum())

            output = torch.matmul(probabilities, repeated_v)
            reconstructed_output = torch.matmul(reconstructed_probabilities, repeated_v)
            output_error_sum += float((reconstructed_output - output).square().sum())

    total_valid_logits = layers * valid_logits
    total_queries = layers * query_heads * length
    total_output_values = total_queries * head_dim
    bias = logit_error_sum / max(total_valid_logits, 1)
    mse = logit_error_sq_sum / max(total_valid_logits, 1)
    worst_layer = int(np.argmax(layer_kls))
    return {
        "attention_logit_mse": mse,
        "attention_logit_bias": bias,
        "attention_logit_error_variance": max(mse - bias * bias, 0.0),
        "attention_probability_kl": kl_sum / max(total_queries, 1),
        "attention_output_mse": output_error_sum / max(total_output_values, 1),
        "worst_attention_kl": layer_kls[worst_layer],
        "worst_attention_layer": worst_layer,
        "sequence_length": length,
    }


def no_quantization_roundtrip_metrics(
    queries: torch.Tensor,
    keys: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    rotations: torch.Tensor,
    *,
    sequence_length: int,
) -> dict[str, float | int]:
    """Verify inverse rotation followed by RoPE preserves original attention.

    This is the explicit no-quantization gate required by ``SpinTurboQuant.md``.
    It exercises real captured Llama projections instead of relying only on an
    abstract orthogonality check.
    """

    layers, available, query_heads, head_dim = queries.shape
    key_layers, key_available, kv_heads, key_dim = keys.shape
    if (key_layers, key_available, key_dim) != (layers, available, head_dim):
        raise ValueError("Q and K capture shapes are incompatible")
    if rotations.shape != (layers, kv_heads, head_dim, head_dim):
        raise ValueError("rotation shape does not match captured keys")
    if query_heads % kv_heads != 0:
        raise ValueError("query head count must be divisible by KV head count")
    if not 1 <= sequence_length <= available:
        raise ValueError("sequence_length exceeds the captured sequence")

    device = rotations.device
    length = sequence_length
    rope_cos = cos[:length].to(device=device, dtype=torch.float32)
    rope_sin = sin[:length].to(device=device, dtype=torch.float32)
    repeats = query_heads // kv_heads
    pre_rope_sse = 0.0
    pre_rope_values = 0
    pre_rope_max = 0.0
    logit_sse = 0.0
    logit_values = 0
    logit_max = 0.0

    with torch.inference_mode():
        for layer in range(layers):
            q = queries[layer, :length].permute(1, 0, 2).to(device=device, dtype=torch.float32)
            k = keys[layer, :length].permute(1, 0, 2).to(device=device, dtype=torch.float32)
            rotation = rotations[layer].float()
            norms = torch.linalg.vector_norm(k, dim=-1, keepdim=True)
            normalized = k / norms.clamp_min(torch.finfo(torch.float32).eps)
            rotated = torch.matmul(normalized, rotation.transpose(-1, -2))
            restored = torch.matmul(rotated, rotation) * norms
            error = restored - k
            pre_rope_sse += float(error.square().sum())
            pre_rope_values += error.numel()
            pre_rope_max = max(pre_rope_max, float(error.abs().max()))

            q_rope = _apply_rope(q, rope_cos, rope_sin)
            original_k_rope = _apply_rope(k, rope_cos, rope_sin).repeat_interleave(
                repeats, dim=0
            )
            restored_k_rope = _apply_rope(
                restored, rope_cos, rope_sin
            ).repeat_interleave(repeats, dim=0)
            scale = 1.0 / math.sqrt(head_dim)
            original_logits = (
                torch.matmul(q_rope, original_k_rope.transpose(-1, -2)) * scale
            )
            restored_logits = (
                torch.matmul(q_rope, restored_k_rope.transpose(-1, -2)) * scale
            )
            logit_error = restored_logits - original_logits
            logit_sse += float(logit_error.square().sum())
            logit_values += logit_error.numel()
            logit_max = max(logit_max, float(logit_error.abs().max()))

    return {
        "sequence_length": length,
        "pre_rope_key_roundtrip_mse": pre_rope_sse / max(pre_rope_values, 1),
        "pre_rope_key_roundtrip_max_abs": pre_rope_max,
        "attention_logit_roundtrip_mse": logit_sse / max(logit_values, 1),
        "attention_logit_roundtrip_max_abs": logit_max,
    }


def _projection_codec(
    output: torch.Tensor,
    rotations: torch.Tensor,
    centroids: torch.Tensor,
    *,
    norm_correction: bool,
) -> torch.Tensor:
    """Apply the codec to one Llama ``k_proj`` output tensor."""

    batch, sequence, width = output.shape
    heads, head_dim, other_dim = rotations.shape
    if head_dim != other_dim or width != heads * head_dim:
        raise ValueError(
            f"projection width {width} is incompatible with rotations {tuple(rotations.shape)}"
        )
    vectors = output.reshape(batch, sequence, heads, head_dim)
    head_first = vectors.permute(2, 0, 1, 3).reshape(heads, batch * sequence, head_dim)
    reconstructed = apply_codec(
        head_first.float(),
        rotations.float(),
        centroids,
        norm_correction=norm_correction,
    )
    return (
        reconstructed.reshape(heads, batch, sequence, head_dim)
        .permute(1, 2, 0, 3)
        .reshape(batch, sequence, width)
        .to(dtype=output.dtype)
    )


@contextmanager
def install_key_codec_hooks(
    model: torch.nn.Module,
    rotations: torch.Tensor,
    centroids: torch.Tensor,
    *,
    norm_correction: bool = False,
) -> Iterator[None]:
    """Temporarily quantize reconstructed pre-RoPE K in every selected layer."""

    layers = model.model.layers
    if rotations.ndim != 4 or rotations.shape[0] > len(layers):
        raise ValueError("rotations must have shape (selected_layers, heads, d, d)")
    handles: list[torch.utils.hooks.RemovableHandle] = []
    for layer_index in range(rotations.shape[0]):
        layer_rotation = rotations[layer_index]

        def hook(
            _module: torch.nn.Module,
            _inputs: tuple[torch.Tensor, ...],
            output: torch.Tensor,
            *,
            selected_rotation: torch.Tensor = layer_rotation,
        ) -> torch.Tensor:
            return _projection_codec(
                output,
                selected_rotation,
                centroids,
                norm_correction=norm_correction,
            )

        handles.append(layers[layer_index].self_attn.k_proj.register_forward_hook(hook))
    try:
        yield
    finally:
        for handle in handles:
            handle.remove()


def codec_latency(
    keys: torch.Tensor,
    rotations: torch.Tensor,
    centroids: torch.Tensor,
    *,
    norm_correction: bool = False,
    warmup: int = 3,
    repeats: int = 10,
) -> dict[str, float | int]:
    """Benchmark dense rotate/quantize/inverse-rotate latency on captured keys."""

    if keys.ndim != 4:
        raise ValueError("keys must have shape (layers, heads, tokens, d)")
    layers, heads, tokens, head_dim = keys.shape
    device = rotations.device
    vectors = keys.to(device=device, dtype=torch.float32).reshape(
        layers * heads, tokens, head_dim
    )
    flat_rotations = rotations.reshape(layers * heads, head_dim, head_dim).float()
    norms = torch.linalg.vector_norm(vectors, dim=-1, keepdim=True)
    normalized = vectors / norms.clamp_min(torch.finfo(torch.float32).eps)
    with torch.inference_mode():
        for _ in range(warmup):
            torch.matmul(normalized, flat_rotations.transpose(-1, -2))
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        rotation_started = time.perf_counter()
        for _ in range(repeats):
            torch.matmul(normalized, flat_rotations.transpose(-1, -2))
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        rotation_elapsed = time.perf_counter() - rotation_started

        for _ in range(warmup):
            apply_codec(
                vectors,
                flat_rotations,
                centroids,
                norm_correction=norm_correction,
            )
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        started = time.perf_counter()
        for _ in range(repeats):
            apply_codec(
                vectors,
                flat_rotations,
                centroids,
                norm_correction=norm_correction,
            )
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - started
    vectors_per_repeat = layers * heads * tokens
    return {
        "rotation_forward_latency_ms": rotation_elapsed * 1000.0 / repeats,
        "rotation_forward_microseconds_per_vector": rotation_elapsed
        * 1e6
        / (repeats * vectors_per_repeat),
        "codec_latency_ms": elapsed * 1000.0 / repeats,
        "codec_microseconds_per_vector": elapsed * 1e6 / (repeats * vectors_per_repeat),
        "latency_vectors": vectors_per_repeat,
        "rotation_storage_bytes_fp32": int(rotations.numel() * 4),
    }
