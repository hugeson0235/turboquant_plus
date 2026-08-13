"""Tests for magnitude-clustered group-wise TurboQuant."""

import numpy as np
import pytest

from turboquant.channel_cluster import ChannelClusteredTurboQuant, fit_channel_groups
from turboquant.polar_quant import PolarQuant


def test_fit_groups_orders_centroids_and_covers_channels():
    magnitudes = np.array([9.0, 1.0, 1.2, 8.0, 4.0, 4.2])
    labels, groups, centroids = fit_channel_groups(magnitudes, 3, seed=7)

    assert labels.shape == (6,)
    assert np.array_equal(np.sort(np.concatenate(groups)), np.arange(6))
    assert all(group.size > 0 for group in groups)
    assert np.all(np.diff(centroids) > 0)


def test_degenerate_magnitudes_still_form_nonempty_partition():
    labels, groups, centroids = fit_channel_groups(np.ones(8), 4, seed=1)

    assert set(labels.tolist()) == {0, 1, 2, 3}
    assert [group.size for group in groups] == [2, 2, 2, 2]
    np.testing.assert_allclose(centroids, 1.0)


def test_k1_matches_reference_polarquant_at_fp32_norm_precision():
    rng = np.random.default_rng(11)
    values = rng.standard_normal((7, 16)).astype(np.float32)
    clustered = ChannelClusteredTurboQuant(
        np.ones(16), 1, 3, seed=19, norm_bits=32
    )
    reference = PolarQuant(16, bit_width=3, seed=19)

    indices, norms = reference.quantize(values)
    expected = reference.dequantize(indices, norms.astype(np.float32))
    actual = clustered.reconstruct(values)

    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("shape", [(16,), (5, 16), (2, 3, 16)])
def test_round_trip_preserves_shape_and_zero_vectors(shape):
    codec = ChannelClusteredTurboQuant(
        np.linspace(1, 4, 16), 4, 3, seed=2, norm_bits=16
    )
    values = np.zeros(shape, dtype=np.float32)
    compressed = codec.quantize(values)
    reconstructed = codec.dequantize(compressed)

    assert reconstructed.shape == shape
    np.testing.assert_allclose(reconstructed, 0.0, atol=0.0)
    assert all(norm.dtype == np.float16 for norm in compressed.norms)


def test_storage_accounting_includes_every_group_norm():
    codec = ChannelClusteredTurboQuant(
        np.arange(128, dtype=np.float64), 4, 3, seed=3, norm_bits=16
    )

    assert codec.per_vector_bits == 128 * 3 + 4 * 16
    assert codec.effective_bits_per_channel == pytest.approx(3.5)
    assert codec.static_metadata_bits() > 0


def test_group_normalization_protects_regular_channels_from_scale_outliers():
    rng = np.random.default_rng(4)
    values = rng.standard_normal((512, 32)).astype(np.float32)
    values[:, -4:] *= 50.0
    magnitudes = np.sqrt(np.mean(values.astype(np.float64) ** 2, axis=0))

    global_codec = ChannelClusteredTurboQuant(magnitudes, 1, 2, seed=9, norm_bits=32)
    grouped_codec = ChannelClusteredTurboQuant(magnitudes, 2, 2, seed=9, norm_bits=32)
    global_reconstructed = global_codec.reconstruct(values)
    grouped_reconstructed = grouped_codec.reconstruct(values)

    regular = np.arange(28)
    global_regular_mse = np.mean((values[:, regular] - global_reconstructed[:, regular]) ** 2)
    grouped_regular_mse = np.mean((values[:, regular] - grouped_reconstructed[:, regular]) ** 2)
    assert grouped_regular_mse < 0.25 * global_regular_mse


def test_torch_and_numpy_paths_use_same_reference_parameters():
    torch = pytest.importorskip("torch")
    rng = np.random.default_rng(5)
    values = rng.standard_normal((13, 16)).astype(np.float32)
    codec = ChannelClusteredTurboQuant(
        np.linspace(0.5, 5.0, 16), 3, 3, seed=12, norm_bits=16
    )

    numpy_reconstructed = codec.reconstruct(values)
    torch_reconstructed = codec.reconstruct_torch(torch.from_numpy(values)).numpy()

    np.testing.assert_allclose(torch_reconstructed, numpy_reconstructed, rtol=2e-5, atol=2e-5)


@pytest.mark.parametrize(
    ("magnitudes", "clusters", "message"),
    [
        (np.ones((2, 2)), 2, "one-dimensional"),
        (np.ones(2), 3, "n_clusters"),
        (np.array([1.0, np.nan]), 1, "finite"),
        (np.array([1.0, -1.0]), 1, "non-negative"),
    ],
)
def test_invalid_partitions_are_rejected(magnitudes, clusters, message):
    with pytest.raises(ValueError, match=message):
        fit_channel_groups(magnitudes, clusters)
