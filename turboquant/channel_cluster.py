# Copyright 2026 Tom Turney
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Magnitude-clustered, group-wise PolarQuant for KV key vectors.

This module implements the method proposed in ``ClusterTurboQuant.md``:

* estimate one RMS magnitude per channel on calibration activations;
* partition channels with ordinary one-dimensional k-means;
* normalize, rotate, and scalar-quantize each group independently; and
* scatter reconstructed groups back into the original channel order.

The NumPy path delegates every group codec to :class:`PolarQuant`.  The Torch
path uses the exact rotations and codebooks created by those same PolarQuant
objects so it can be inserted into a model's ``k_proj`` forward path without a
CPU round trip.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil, log2
from typing import Any

import numpy as np
from sklearn.cluster import KMeans

from turboquant.polar_quant import PolarQuant


@dataclass
class ChannelClusteredCompressed:
    """Serialized-value components for a batch of clustered vectors.

    ``indices[g]`` has shape ``(n_vectors, group_size[g])`` and ``norms[g]``
    has shape ``(n_vectors,)``.  ``leading_shape`` records all input dimensions
    before the final channel dimension so dequantization restores the input
    shape exactly.
    """

    indices: list[np.ndarray]
    norms: list[np.ndarray]
    leading_shape: tuple[int, ...]


@dataclass
class TorchChannelClusteredCompressed:
    """Torch equivalent of :class:`ChannelClusteredCompressed`."""

    indices: list[Any]
    norms: list[Any]
    leading_shape: tuple[int, ...]
    output_dtype: Any


def fit_channel_groups(
    channel_magnitudes: np.ndarray,
    n_clusters: int,
    *,
    seed: int = 0,
    n_init: int = 20,
) -> tuple[np.ndarray, tuple[np.ndarray, ...], np.ndarray]:
    """Cluster scalar channel magnitudes and return a stable partition.

    Cluster IDs are reordered by increasing centroid, making group 0 the
    lowest-magnitude group and the final group the highest-magnitude group.
    K-means is undefined when fewer than ``n_clusters`` distinct values exist;
    in that degenerate case a stable magnitude-sorted split is used solely to
    guarantee the method's required non-empty partition.
    """

    magnitudes = np.asarray(channel_magnitudes, dtype=np.float64)
    if magnitudes.ndim != 1:
        raise ValueError("channel_magnitudes must be one-dimensional")
    if not np.all(np.isfinite(magnitudes)):
        raise ValueError("channel_magnitudes must contain only finite values")
    if np.any(magnitudes < 0):
        raise ValueError("channel_magnitudes must be non-negative")
    d = magnitudes.size
    if d == 0:
        raise ValueError("channel_magnitudes must not be empty")
    if not 1 <= n_clusters <= d:
        raise ValueError(f"n_clusters must be in [1, {d}], got {n_clusters}")

    if n_clusters == 1:
        labels = np.zeros(d, dtype=np.int64)
    elif np.unique(magnitudes).size < n_clusters:
        # KMeans cannot produce K non-empty clusters from fewer than K unique
        # points. A stable sorted split is deterministic and preserves the
        # intended magnitude ordering in this otherwise undefined edge case.
        labels = np.empty(d, dtype=np.int64)
        order = np.argsort(magnitudes, kind="stable")
        for group_id, channel_ids in enumerate(np.array_split(order, n_clusters)):
            labels[channel_ids] = group_id
    else:
        estimator = KMeans(
            n_clusters=n_clusters,
            init="k-means++",
            n_init=n_init,
            max_iter=300,
            algorithm="lloyd",
            random_state=seed,
        )
        raw_labels = estimator.fit_predict(magnitudes[:, None])
        raw_centroids = estimator.cluster_centers_[:, 0]
        centroid_order = np.argsort(raw_centroids, kind="stable")
        remap = np.empty(n_clusters, dtype=np.int64)
        remap[centroid_order] = np.arange(n_clusters)
        labels = remap[raw_labels]

    groups = tuple(np.flatnonzero(labels == group_id) for group_id in range(n_clusters))
    if any(group.size == 0 for group in groups):  # defensive; sklearn should not do this
        raise RuntimeError("channel clustering produced an empty group")
    centroids = np.asarray([magnitudes[group].mean() for group in groups])
    return labels, groups, centroids


class ChannelClusteredTurboQuant:
    """Group-wise TurboQuant codec learned from channel RMS magnitudes.

    Args:
        channel_magnitudes: Calibration RMS for each channel.
        n_clusters: Number of channel groups (``K`` in the proposal).
        bit_width: Scalar code-index width used identically in every group.
        seed: Random-rotation seed.
        cluster_seed: Optional independent k-means seed. If omitted, ``seed``
            is used for backward compatibility. Experiments that average over
            rotation seeds should hold this value fixed.
        norm_bits: Stored precision for each group norm. Supported values are
            16 (IEEE fp16) and 32 (IEEE fp32).
        n_init: Number of ordinary k-means initializations.
    """

    def __init__(
        self,
        channel_magnitudes: np.ndarray,
        n_clusters: int,
        bit_width: int,
        *,
        seed: int = 0,
        cluster_seed: int | None = None,
        norm_bits: int = 16,
        n_init: int = 20,
    ) -> None:
        if bit_width < 1:
            raise ValueError("bit_width must be at least 1")
        if norm_bits not in (16, 32):
            raise ValueError("norm_bits must be either 16 or 32")

        self.channel_magnitudes = np.asarray(channel_magnitudes, dtype=np.float64).copy()
        self.d = self.channel_magnitudes.size
        self.n_clusters = n_clusters
        self.bit_width = bit_width
        self.seed = seed
        self.cluster_seed = seed if cluster_seed is None else cluster_seed
        self.norm_bits = norm_bits
        self.labels, self.groups, self.group_centroids = fit_channel_groups(
            self.channel_magnitudes,
            n_clusters,
            seed=self.cluster_seed,
            n_init=n_init,
        )
        self.codecs = tuple(
            PolarQuant(group.size, bit_width=bit_width, seed=seed + 1009 * group_id)
            for group_id, group in enumerate(self.groups)
        )
        self._torch_cache: dict[tuple[int, str], tuple[tuple[Any, Any, Any], ...]] = {}

    @property
    def group_sizes(self) -> tuple[int, ...]:
        """Number of channels in each magnitude-ordered group."""

        return tuple(group.size for group in self.groups)

    @property
    def per_vector_bits(self) -> int:
        """Dynamic serialized bits per vector, including all group norms."""

        return self.d * self.bit_width + self.n_clusters * self.norm_bits

    @property
    def effective_bits_per_channel(self) -> float:
        """Dynamic bits/channel, including group-norm overhead."""

        return self.per_vector_bits / self.d

    def static_metadata_bits(self, *, rotation_bits: int = 32, centroid_bits: int = 32) -> int:
        """Bits in the fixed partition, rotations, and scalar codebooks.

        Static metadata is shared by all tokens, so it is intentionally not
        folded into :attr:`effective_bits_per_channel`.
        """

        label_bits = 0 if self.n_clusters == 1 else self.d * ceil(log2(self.n_clusters))
        rotation_values = sum(size * size for size in self.group_sizes)
        codebook_values = self.n_clusters * (1 << self.bit_width)
        return label_bits + rotation_values * rotation_bits + codebook_values * centroid_bits

    def _store_numpy_norms(self, norms: np.ndarray) -> np.ndarray:
        dtype = np.float16 if self.norm_bits == 16 else np.float32
        return np.asarray(norms, dtype=dtype)

    def quantize(self, x: np.ndarray) -> ChannelClusteredCompressed:
        """Quantize one vector or an arbitrary batch whose final size is ``d``."""

        values = np.asarray(x)
        if values.ndim < 1 or values.shape[-1] != self.d:
            raise ValueError(f"expected final dimension {self.d}, got shape {values.shape}")
        leading_shape = values.shape[:-1]
        flat = values.reshape(-1, self.d)
        all_indices: list[np.ndarray] = []
        all_norms: list[np.ndarray] = []
        for group, codec in zip(self.groups, self.codecs, strict=True):
            indices, norms = codec.quantize(flat[:, group])
            all_indices.append(indices)
            all_norms.append(self._store_numpy_norms(norms))
        return ChannelClusteredCompressed(all_indices, all_norms, leading_shape)

    def dequantize(self, compressed: ChannelClusteredCompressed) -> np.ndarray:
        """Reconstruct values and scatter each group into original channel order."""

        if len(compressed.indices) != self.n_clusters or len(compressed.norms) != self.n_clusters:
            raise ValueError("compressed group count does not match codec")
        n_vectors = int(np.prod(compressed.leading_shape, dtype=np.int64)) if compressed.leading_shape else 1
        output = np.empty((n_vectors, self.d), dtype=np.float64)
        for group, codec, indices, norms in zip(
            self.groups,
            self.codecs,
            compressed.indices,
            compressed.norms,
            strict=True,
        ):
            output[:, group] = codec.dequantize(indices, norms)
        return output.reshape(*compressed.leading_shape, self.d)

    def reconstruct(self, x: np.ndarray) -> np.ndarray:
        """Convenience wrapper for a complete NumPy quantize/dequantize pass."""

        return self.dequantize(self.quantize(x))

    def _torch_parameters(self, device: Any) -> tuple[tuple[Any, Any, Any], ...]:
        import torch

        device_index = -1 if device.index is None else device.index
        cache_key = (device_index, device.type)
        cached = self._torch_cache.get(cache_key)
        if cached is not None:
            return cached
        params = tuple(
            (
                torch.as_tensor(group, dtype=torch.long, device=device),
                torch.as_tensor(codec.rotation, dtype=torch.float32, device=device),
                torch.as_tensor(codec.centroids, dtype=torch.float32, device=device),
            )
            for group, codec in zip(self.groups, self.codecs, strict=True)
        )
        self._torch_cache[cache_key] = params
        return params

    def quantize_torch(self, x: Any) -> TorchChannelClusteredCompressed:
        """Torch quantization using the reference PolarQuant parameters."""

        import torch

        if x.ndim < 1 or x.shape[-1] != self.d:
            raise ValueError(f"expected final dimension {self.d}, got shape {tuple(x.shape)}")
        leading_shape = tuple(x.shape[:-1])
        flat = x.reshape(-1, self.d).to(torch.float32)
        all_indices: list[Any] = []
        all_norms: list[Any] = []
        norm_dtype = torch.float16 if self.norm_bits == 16 else torch.float32
        for group, rotation, centroids in self._torch_parameters(x.device):
            group_values = flat.index_select(1, group)
            norms = torch.linalg.vector_norm(group_values, dim=1)
            safe_norms = torch.where(norms > 0, norms, torch.ones_like(norms))
            normalized = group_values / safe_norms[:, None]
            rotated = normalized @ rotation.T
            boundaries = (centroids[:-1] + centroids[1:]) / 2
            indices = torch.bucketize(rotated.contiguous(), boundaries)
            all_indices.append(indices)
            all_norms.append(norms.to(norm_dtype))
        return TorchChannelClusteredCompressed(all_indices, all_norms, leading_shape, x.dtype)

    def dequantize_torch(self, compressed: TorchChannelClusteredCompressed, *, device: Any = None) -> Any:
        """Torch reconstruction from group indices and stored norms."""

        import torch

        if len(compressed.indices) != self.n_clusters or len(compressed.norms) != self.n_clusters:
            raise ValueError("compressed group count does not match codec")
        if device is None:
            device = compressed.indices[0].device
        n_vectors = int(np.prod(compressed.leading_shape, dtype=np.int64)) if compressed.leading_shape else 1
        output = torch.empty((n_vectors, self.d), dtype=torch.float32, device=device)
        for (group, rotation, centroids), indices, norms in zip(
            self._torch_parameters(device),
            compressed.indices,
            compressed.norms,
            strict=True,
        ):
            reconstructed_rotated = centroids[indices]
            reconstructed_norms = torch.linalg.vector_norm(
                reconstructed_rotated, dim=1, keepdim=True
            )
            reconstructed_norms = torch.where(
                reconstructed_norms > 1e-10,
                reconstructed_norms,
                torch.ones_like(reconstructed_norms),
            )
            reconstructed_rotated = reconstructed_rotated / reconstructed_norms
            reconstructed = (reconstructed_rotated @ rotation) * norms.to(torch.float32)[:, None]
            output[:, group] = reconstructed
        return output.reshape(*compressed.leading_shape, self.d).to(compressed.output_dtype)

    def reconstruct_torch(self, x: Any) -> Any:
        """Convenience wrapper used by model projection hooks."""

        return self.dequantize_torch(self.quantize_torch(x), device=x.device)
