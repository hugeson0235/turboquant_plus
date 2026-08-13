"""Focused tests for the SpinTurboQuant experiment implementation."""

import numpy as np
import torch

from experiments.spin_turboquant.core import (
    _projection_codec,
    apply_codec,
    build_random_rotations,
    cayley_rotation,
    codebook_tensor,
    no_quantization_roundtrip_metrics,
    quantize_to_centroids,
    train_headwise_rotations,
)
from turboquant.codebook import nearest_centroid_indices, optimal_centroids
from turboquant.polar_quant import PolarQuant


def test_cayley_zero_starts_at_initial_and_stays_orthogonal():
    initial = build_random_rotations(1, 2, 8, seed=17).reshape(2, 8, 8)
    zero = torch.zeros_like(initial)
    torch.testing.assert_close(cayley_rotation(zero, initial), initial)

    generator = torch.Generator().manual_seed(4)
    parameters = torch.randn((2, 8, 8), generator=generator) * 0.05
    rotated = cayley_rotation(parameters, initial)
    identity = torch.eye(8).expand(2, 8, 8)
    torch.testing.assert_close(
        rotated.transpose(-1, -2) @ rotated,
        identity,
        atol=2e-5,
        rtol=2e-5,
    )


def test_torch_quantizer_matches_turboquant_plus_codebook_lookup():
    rng = np.random.default_rng(9)
    values = rng.normal(size=(7, 11)).astype(np.float32)
    numpy_centroids = optimal_centroids(3, 128).astype(np.float32)
    expected = numpy_centroids[nearest_centroid_indices(values, numpy_centroids)]
    actual = quantize_to_centroids(
        torch.from_numpy(values), torch.from_numpy(numpy_centroids)
    )
    np.testing.assert_array_equal(actual.numpy(), expected)


def test_apply_codec_matches_polar_quant_without_norm_correction():
    rng = np.random.default_rng(12)
    vectors = rng.normal(size=(6, 16)).astype(np.float32)
    polar = PolarQuant(d=16, bit_width=3, seed=91, norm_correction=False)
    indices, norms = polar.quantize(vectors)
    expected = polar.dequantize(indices, norms)

    actual = apply_codec(
        torch.from_numpy(vectors).unsqueeze(0),
        torch.from_numpy(polar.rotation.astype(np.float32)).unsqueeze(0),
        codebook_tensor(3, 16, device="cpu"),
    ).squeeze(0)
    np.testing.assert_allclose(actual.numpy(), expected, atol=2e-6, rtol=2e-6)


def test_projection_codec_preserves_projection_contract():
    output = torch.randn(2, 5, 3 * 8, dtype=torch.bfloat16)
    rotations = build_random_rotations(1, 3, 8, seed=3)[0]
    reconstructed = _projection_codec(
        output,
        rotations,
        codebook_tensor(4, 8, device="cpu"),
        norm_correction=False,
    )
    assert reconstructed.shape == output.shape
    assert reconstructed.dtype == output.dtype
    assert torch.isfinite(reconstructed.float()).all()


def test_no_quantization_roundtrip_preserves_attention_logits():
    generator = torch.Generator().manual_seed(22)
    queries = torch.randn((2, 12, 4, 8), generator=generator)
    keys = torch.randn((2, 12, 2, 8), generator=generator)
    rotations = build_random_rotations(2, 2, 8, seed=30)
    metrics = no_quantization_roundtrip_metrics(
        queries,
        keys,
        torch.ones(12, 8),
        torch.zeros(12, 8),
        rotations,
        sequence_length=12,
    )
    assert metrics["pre_rope_key_roundtrip_max_abs"] < 2e-6
    assert metrics["attention_logit_roundtrip_max_abs"] < 2e-6


def test_training_reduces_turboquant_objective_on_structured_keys():
    generator = torch.Generator().manual_seed(123)
    # A deliberately anisotropic distribution gives the learned basis a clear
    # signal while remaining small enough for a fast CPU regression test.
    keys = torch.randn((2, 384, 8), generator=generator)
    keys[..., 0] *= 9.0
    keys[..., 1] *= 4.0
    initial = build_random_rotations(1, 2, 8, seed=8).reshape(2, 8, 8)
    learned, stats = train_headwise_rotations(
        keys,
        initial,
        codebook_tensor(2, 8, device="cpu"),
        steps=60,
        batch_tokens=128,
        learning_rate=0.02,
        optimizer_seed=44,
    )
    assert stats.final_loss < stats.initial_loss
    assert stats.relative_improvement > 0.01
    assert stats.orthogonality_max_abs < 2e-5
    assert learned.shape == initial.shape
