"""Pure math for the analytic head-wise PCA rotation study.

The analytic specification uses row vectors throughout.  A saved rotation
``R`` therefore acts as ``z = x @ R`` and reconstructs as ``x_hat = z @ R.T``.
The older SpinTurboQuant codec uses the transposed convention internally;
``to_codec_rotations`` is the single explicit adapter between the two.
"""

from __future__ import annotations

import hashlib
import json
import math
from typing import Any

import torch


def tensor_sha256(tensor: torch.Tensor) -> str:
    """Hash tensor dtype, shape, and canonical contiguous CPU bytes."""

    value = tensor.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode())
    digest.update(json.dumps(list(value.shape), separators=(",", ":")).encode())
    digest.update(value.numpy().tobytes(order="C"))
    return digest.hexdigest()


def normalized_hadamard(order: int, *, dtype: torch.dtype = torch.float64) -> torch.Tensor:
    """Return the deterministic normalized Sylvester Hadamard matrix."""

    if order < 1 or order & (order - 1):
        raise ValueError("Hadamard order must be a positive power of two")
    matrix = torch.ones((1, 1), dtype=dtype)
    while matrix.shape[0] < order:
        matrix = torch.cat(
            (
                torch.cat((matrix, matrix), dim=1),
                torch.cat((matrix, -matrix), dim=1),
            ),
            dim=0,
        )
    return matrix / math.sqrt(order)


def fix_eigenvector_signs(eigenvectors: torch.Tensor) -> torch.Tensor:
    """Make the largest-magnitude entry in every eigenvector positive."""

    if eigenvectors.ndim < 2 or eigenvectors.shape[-1] != eigenvectors.shape[-2]:
        raise ValueError("eigenvectors must have shape (..., d, d)")
    indexes = eigenvectors.abs().argmax(dim=-2, keepdim=True)
    pivots = eigenvectors.gather(-2, indexes).squeeze(-2)
    signs = torch.where(pivots < 0, -torch.ones_like(pivots), torch.ones_like(pivots))
    return eigenvectors * signs.unsqueeze(-2)


def spectral_rotation(
    covariance: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build ``R = U H`` from float64 PSD second moments.

    Returns descending eigenvalues, sign-fixed eigenvectors, and row-vector
    rotations.  All calculations and returned tensors remain float64.
    """

    if covariance.dtype != torch.float64:
        raise ValueError("spectral construction requires torch.float64 covariance")
    if covariance.ndim < 2 or covariance.shape[-1] != covariance.shape[-2]:
        raise ValueError("covariance must have shape (..., d, d)")
    symmetric = (covariance + covariance.transpose(-1, -2)) * 0.5
    eigenvalues, eigenvectors = torch.linalg.eigh(symmetric)
    eigenvalues = torch.flip(eigenvalues, dims=(-1,))
    eigenvectors = torch.flip(eigenvectors, dims=(-1,))
    eigenvectors = fix_eigenvector_signs(eigenvectors)
    hadamard = normalized_hadamard(
        covariance.shape[-1], dtype=torch.float64
    ).to(covariance.device)
    rotations = eigenvectors @ hadamard
    return eigenvalues, eigenvectors, rotations


def weight_second_moment(
    projection_weight: torch.Tensor,
    *,
    num_kv_heads: int,
) -> torch.Tensor:
    """Compute per-head ``W_K.T @ W_K`` from a PyTorch linear weight.

    ``projection_weight`` has PyTorch shape ``(heads * d, d_model)``.  Each
    contiguous output block is transposed to the row-vector convention
    ``W_K in R^(d_model x d)`` without mean centering.
    """

    if projection_weight.ndim != 2:
        raise ValueError("projection weight must be a matrix")
    output_width, _ = projection_weight.shape
    if output_width % num_kv_heads:
        raise ValueError("projection width is not divisible by KV heads")
    head_dim = output_width // num_kv_heads
    blocks = projection_weight.detach().to(dtype=torch.float64).reshape(
        num_kv_heads, head_dim, -1
    )
    return blocks @ blocks.transpose(-1, -2)


def normalized_key_second_moment(keys: torch.Tensor) -> tuple[torch.Tensor, int]:
    """Return the uncentered normalized-Key second-moment sum.

    ``keys`` has shape ``(tokens, heads, d)``.  The return value is the sum,
    not the average, so independent sequences can be accumulated exactly.
    """

    if keys.ndim != 3:
        raise ValueError("keys must have shape (tokens, heads, d)")
    values = keys.to(dtype=torch.float64)
    norms = torch.linalg.vector_norm(values, dim=-1, keepdim=True)
    normalized = values / norms.clamp_min(torch.finfo(torch.float64).eps)
    moment = torch.einsum("thd,the->hde", normalized, normalized)
    return moment, int(keys.shape[0])


def rotate_half(values: torch.Tensor) -> torch.Tensor:
    first, second = values.chunk(2, dim=-1)
    return torch.cat((-second, first), dim=-1)


def apply_row_rope(
    values: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    """Apply Hugging Face Llama RoPE to ``(tokens, heads, d)`` row vectors."""

    if values.ndim != 3 or cos.ndim != 2 or sin.shape != cos.shape:
        raise ValueError("expected values (tokens, heads, d) and cos/sin (tokens, d)")
    if values.shape[0] != cos.shape[0] or values.shape[-1] != cos.shape[-1]:
        raise ValueError("RoPE tensors have incompatible shapes")
    return values * cos.unsqueeze(1) + rotate_half(values) * sin.unsqueeze(1)


def row_rope_matrix(cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Materialize the row-vector RoPE matrix for tests and diagnostics."""

    if cos.ndim != 1 or sin.shape != cos.shape or cos.numel() % 2:
        raise ValueError("cos and sin must be equal even-width vectors")
    half = cos.numel() // 2
    if not torch.allclose(cos[:half], cos[half:]) or not torch.allclose(
        sin[:half], sin[half:]
    ):
        raise ValueError("Llama RoPE frequencies must repeat across both halves")
    result = torch.zeros((cos.numel(), cos.numel()), dtype=cos.dtype, device=cos.device)
    index = torch.arange(half, device=cos.device)
    c = cos[:half]
    s = sin[:half]
    result[index, index] = c
    result[index, index + half] = s
    result[index + half, index] = -s
    result[index + half, index + half] = c
    return result


def rope_transform_covariances(
    covariances: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    """Compute ``P_s C_s P_s.T`` using Llama's block-diagonal RoPE.

    ``covariances`` has shape ``(tokens, ..., d, d)`` and cos/sin have shape
    ``(tokens, d)``.  The implementation is O(d^2) per position and does not
    materialize dense causal-pair or RoPE matrices.
    """

    if covariances.ndim < 3 or covariances.shape[-1] != covariances.shape[-2]:
        raise ValueError("covariances must have shape (tokens, ..., d, d)")
    tokens, head_dim = covariances.shape[0], covariances.shape[-1]
    if cos.shape != (tokens, head_dim) or sin.shape != cos.shape or head_dim % 2:
        raise ValueError("RoPE shapes are incompatible with covariance")
    half = head_dim // 2
    if not torch.allclose(cos[:, :half], cos[:, half:], atol=1e-12, rtol=1e-12):
        raise ValueError("cos frequencies do not repeat across Llama RoPE halves")
    if not torch.allclose(sin[:, :half], sin[:, half:], atol=1e-12, rtol=1e-12):
        raise ValueError("sin frequencies do not repeat across Llama RoPE halves")
    middle_dims = covariances.ndim - 3
    c_row = cos[:, :half].reshape(tokens, *([1] * middle_dims), half, 1)
    s_row = sin[:, :half].reshape(tokens, *([1] * middle_dims), half, 1)
    top = covariances[..., :half, :]
    bottom = covariances[..., half:, :]
    left = torch.cat(
        (c_row * top + s_row * bottom, -s_row * top + c_row * bottom), dim=-2
    )
    c_col = cos[:, :half].reshape(tokens, *([1] * middle_dims), 1, half)
    s_col = sin[:, :half].reshape(tokens, *([1] * middle_dims), 1, half)
    left_columns = left[..., :, :half]
    right_columns = left[..., :, half:]
    return torch.cat(
        (
            c_col * left_columns + s_col * right_columns,
            -s_col * left_columns + c_col * right_columns,
        ),
        dim=-1,
    )


def attention_query_second_moment(
    pre_rope_queries: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    *,
    num_kv_heads: int,
) -> tuple[torch.Tensor, int]:
    """Compute the future-query-normalized attention target moment sum.

    The input has shape ``(tokens, query_heads, d)``.  Four (or generally
    ``query_heads / kv_heads``) GQA query heads are pooled for each KV head.
    """

    if pre_rope_queries.ndim != 3:
        raise ValueError("queries must have shape (tokens, query_heads, d)")
    tokens, query_heads, head_dim = pre_rope_queries.shape
    if query_heads % num_kv_heads:
        raise ValueError("query heads must be divisible by KV heads")
    group_size = query_heads // num_kv_heads
    query = pre_rope_queries.to(dtype=torch.float64)
    rope_cos = cos.to(device=query.device, dtype=torch.float64)
    rope_sin = sin.to(device=query.device, dtype=torch.float64)
    post_rope = apply_row_rope(query, rope_cos, rope_sin).reshape(
        tokens, num_kv_heads, group_size, head_dim
    )
    token_moments = torch.einsum("thgd,thge->thde", post_rope, post_rope)
    suffix = torch.flip(
        torch.cumsum(torch.flip(token_moments, dims=(0,)), dim=0), dims=(0,)
    )
    future_counts = torch.arange(
        tokens, 0, -1, dtype=torch.float64, device=query.device
    )
    suffix = suffix / (future_counts[:, None, None, None] * group_size)
    transformed = rope_transform_covariances(suffix, rope_cos, rope_sin)
    return transformed.sum(dim=0), tokens


def to_codec_rotations(row_rotations: torch.Tensor) -> torch.Tensor:
    """Adapt saved row-forward rotations to the legacy codec convention."""

    if row_rotations.ndim < 2 or row_rotations.shape[-1] != row_rotations.shape[-2]:
        raise ValueError("rotations must have shape (..., d, d)")
    return row_rotations.transpose(-1, -2).contiguous()


def relative_frobenius_difference(first: torch.Tensor, second: torch.Tensor) -> float:
    numerator = torch.linalg.matrix_norm(first - second, ord="fro")
    denominator = torch.linalg.matrix_norm((first + second) * 0.5, ord="fro")
    return float(numerator / denominator.clamp_min(torch.finfo(first.dtype).eps))


def principal_subspace_similarity(
    first_eigenvectors: torch.Tensor,
    second_eigenvectors: torch.Tensor,
    rank: int,
) -> float:
    """Return mean squared canonical correlation of two leading subspaces."""

    if not 1 <= rank <= first_eigenvectors.shape[-1]:
        raise ValueError("invalid subspace rank")
    first = first_eigenvectors[..., :rank]
    second = second_eigenvectors[..., :rank]
    overlap = first.transpose(-1, -2) @ second
    return float(overlap.square().sum() / rank)


def rotation_checks(rotation: torch.Tensor) -> dict[str, Any]:
    """Return determinant and orthogonality checks for one matrix."""

    value = rotation.to(dtype=torch.float64)
    identity = torch.eye(value.shape[-1], dtype=torch.float64, device=value.device)
    error = value.transpose(-1, -2) @ value - identity
    sign, logabsdet = torch.linalg.slogdet(value)
    return {
        "orthogonality_frobenius": float(torch.linalg.matrix_norm(error, ord="fro")),
        "orthogonality_max_abs": float(error.abs().max()),
        "determinant_sign": float(sign),
        "determinant_log_abs": float(logabsdet),
    }
