"""Channel-allocation and group-wise rotation primitives for KV-cache studies.

The saved partition and rotation artifacts produced here are deliberately plain
``dict``/``list``/scalar/Tensor trees.  They can therefore be written with
``torch.save`` and loaded by an experiment runner without importing a custom
checkpoint class.  Dataclasses in this module are transient descriptions used
to validate those trees and to bucket variable-size groups efficiently.

Tensor conventions follow :mod:`experiments.spin_turboquant.core`: vectors are
stored as rows, a forward rotation is ``x @ R.T``, and its inverse is ``z @ R``.
"""

from __future__ import annotations

import hashlib
import json
import math
from contextlib import contextmanager
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping, MutableMapping, Sequence

import numpy as np
import torch

from experiments.spin_turboquant.core import cayley_rotation
from turboquant.codebook import optimal_centroids
from turboquant.rotation import random_rotation_dense


ROOT_SEED = 35
COMPONENTS = ("key", "value")
GROUP_ALL = "all"
GROUP_REGULAR = "regular"
GROUP_OUTLIER = "outlier"
ALLOCATION_NAMES = ("uniform2", "fixed32", "kmeans2", "uniform3")


_ALLOCATION_CONFIG = {
    "uniform2": {
        "partition_method": "uniform",
        "regular_bits": 2,
        "outlier_bits": None,
        "fixed_outlier_channels": None,
    },
    "fixed32": {
        "partition_method": "fixed32",
        "regular_bits": 2,
        "outlier_bits": 4,
        "fixed_outlier_channels": 32,
    },
    "kmeans2": {
        "partition_method": "kmeans2",
        "regular_bits": 2,
        "outlier_bits": 4,
        "fixed_outlier_channels": None,
    },
    "uniform3": {
        "partition_method": "uniform",
        "regular_bits": 3,
        "outlier_bits": None,
        "fixed_outlier_channels": None,
    },
}


@dataclass(frozen=True)
class AllocationSpec:
    """Validated description of one of the four pinned allocation methods."""

    name: str

    def __post_init__(self) -> None:
        if self.name not in _ALLOCATION_CONFIG:
            raise ValueError(
                f"unknown allocation {self.name!r}; expected one of {ALLOCATION_NAMES}"
            )

    @property
    def partition_method(self) -> str:
        return str(_ALLOCATION_CONFIG[self.name]["partition_method"])

    @property
    def regular_bits(self) -> int:
        return int(_ALLOCATION_CONFIG[self.name]["regular_bits"])

    @property
    def outlier_bits(self) -> int | None:
        value = _ALLOCATION_CONFIG[self.name]["outlier_bits"]
        return None if value is None else int(value)

    @property
    def fixed_outlier_channels(self) -> int | None:
        value = _ALLOCATION_CONFIG[self.name]["fixed_outlier_channels"]
        return None if value is None else int(value)

    @property
    def is_split(self) -> bool:
        return self.partition_method != "uniform"

    @property
    def group_count(self) -> int:
        return 2 if self.is_split else 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "partition_method": self.partition_method,
            "regular_bits": self.regular_bits,
            "outlier_bits": self.outlier_bits,
            "fixed_outlier_channels": self.fixed_outlier_channels,
        }


UNIFORM2 = AllocationSpec("uniform2")
FIXED32 = AllocationSpec("fixed32")
KMEANS2 = AllocationSpec("kmeans2")
UNIFORM3 = AllocationSpec("uniform3")
ALLOCATION_SPECS = {
    spec.name: spec for spec in (UNIFORM2, FIXED32, KMEANS2, UNIFORM3)
}


def allocation_spec(value: str | AllocationSpec) -> AllocationSpec:
    """Return a validated :class:`AllocationSpec`."""

    return value if isinstance(value, AllocationSpec) else AllocationSpec(str(value))


@dataclass(frozen=True)
class GroupDefinition:
    """One channel group for a single layer/head/component."""

    name: str
    bit_width: int
    indices: torch.Tensor

    @property
    def dimension(self) -> int:
        return int(self.indices.numel())


@dataclass(frozen=True)
class GroupBucket:
    """Groups sharing component, bit width, and dimension.

    ``indices`` has shape ``(entries, dimension)``.  The three tuples share the
    same entry order and provide the mapping back to flat artifact keys.
    """

    bucket_id: str
    component: str
    bit_width: int
    dimension: int
    layer_indices: torch.Tensor
    head_indices: torch.Tensor
    indices: torch.Tensor
    rotation_keys: tuple[str, ...]


def _validate_component(component: str) -> str:
    if component not in COMPONENTS:
        raise ValueError(f"component must be one of {COMPONENTS}, got {component!r}")
    return component


def channel_mean_magnitude(activations: torch.Tensor) -> torch.Tensor:
    """Compute per-channel mean absolute magnitude.

    Args:
        activations: Tensor with shape ``(layers, heads, tokens, channels)``.

    Returns:
        A CPU float32 tensor with shape ``(layers, heads, channels)``.
    """

    if activations.ndim != 4:
        raise ValueError(
            "activations must have shape (layers, heads, tokens, channels)"
        )
    if activations.shape[2] < 1 or activations.shape[3] < 1:
        raise ValueError("activations must contain at least one token and channel")
    if not torch.isfinite(activations.float()).all():
        raise ValueError("activations contain NaN or Inf")
    return activations.detach().float().abs().mean(dim=2).cpu().contiguous()


def fixed32_partition(statistics: torch.Tensor) -> torch.Tensor:
    """Select the 32 largest channels per layer/head with stable tie-breaking."""

    if statistics.ndim != 3:
        raise ValueError("statistics must have shape (layers, heads, channels)")
    if statistics.shape[-1] <= 32:
        raise ValueError("fixed32 requires more than 32 channels")
    values = statistics.detach().float().cpu()
    if not torch.isfinite(values).all():
        raise ValueError("statistics contain NaN or Inf")
    # Stable sorting means equal magnitudes select the lower channel index.
    selected = torch.argsort(values, dim=-1, descending=True, stable=True)[..., :32]
    mask = torch.zeros_like(values, dtype=torch.bool)
    mask.scatter_(-1, selected, True)
    return mask.contiguous()


def kmeans2_partition(statistics: torch.Tensor) -> torch.Tensor:
    """Run the pinned one-dimensional sklearn KMeans partition per head.

    The larger-centroid cluster is the outlier group.  scikit-learn is a hard
    dependency of this construction path so the experiment cannot silently
    produce partitions with a different implementation.
    """

    if statistics.ndim != 3:
        raise ValueError("statistics must have shape (layers, heads, channels)")
    values = statistics.detach().float().cpu()
    if values.shape[-1] < 2:
        raise ValueError("kmeans2 requires at least two channels")
    if not torch.isfinite(values).all():
        raise ValueError("statistics contain NaN or Inf")
    try:
        from sklearn.cluster import KMeans
    except ImportError as error:  # pragma: no cover - exercised without experiment env
        raise RuntimeError(
            "kmeans2 partition construction requires scikit-learn==1.5.2"
        ) from error

    layers, heads, channels = values.shape
    result = torch.zeros((layers, heads, channels), dtype=torch.bool)
    for layer in range(layers):
        for head in range(heads):
            samples = values[layer, head].numpy().astype(np.float64).reshape(-1, 1)
            estimator = KMeans(
                n_clusters=2,
                init="k-means++",
                n_init=10,
                random_state=ROOT_SEED,
            )
            labels = estimator.fit_predict(samples)
            counts = np.bincount(labels, minlength=2)
            if np.any(counts == 0):
                raise RuntimeError(
                    f"kmeans2 produced an empty cluster at layer={layer}, head={head}"
                )
            centroids = estimator.cluster_centers_.reshape(-1)
            if not np.isfinite(centroids).all() or centroids[0] == centroids[1]:
                raise RuntimeError(
                    f"kmeans2 centroids are invalid at layer={layer}, head={head}: "
                    f"{centroids.tolist()}"
                )
            outlier_label = int(np.argmax(centroids))
            result[layer, head] = torch.from_numpy(labels == outlier_label)
    return result.contiguous()


def build_component_partition(
    spec: str | AllocationSpec, statistics: torch.Tensor
) -> torch.Tensor:
    """Build one component's outlier mask for an allocation."""

    selected = allocation_spec(spec)
    if selected.partition_method == "uniform":
        if statistics.ndim != 3:
            raise ValueError("statistics must have shape (layers, heads, channels)")
        if not torch.isfinite(statistics.detach().float()).all():
            raise ValueError("statistics contain NaN or Inf")
        return torch.zeros(tuple(statistics.shape), dtype=torch.bool)
    if selected.partition_method == "fixed32":
        return fixed32_partition(statistics)
    if selected.partition_method == "kmeans2":
        return kmeans2_partition(statistics)
    raise AssertionError(selected.partition_method)


def build_partition_artifact(
    spec: str | AllocationSpec,
    key_statistics: torch.Tensor,
    value_statistics: torch.Tensor,
) -> dict[str, Any]:
    """Build the serializable K/V partition artifact from channel statistics."""

    selected = allocation_spec(spec)
    if key_statistics.shape != value_statistics.shape or key_statistics.ndim != 3:
        raise ValueError(
            "key/value statistics must share shape (layers, heads, channels)"
        )
    layers, heads, head_dim = (int(value) for value in key_statistics.shape)
    artifact: dict[str, Any] = {
        "schema_version": 1,
        "allocation": selected.name,
        "num_layers": layers,
        "num_heads": heads,
        "head_dim": head_dim,
        "outlier_masks": {
            "key": build_component_partition(selected, key_statistics),
            "value": build_component_partition(selected, value_statistics),
        },
        "statistics_sha256": {
            "key": tensor_sha256(key_statistics),
            "value": tensor_sha256(value_statistics),
        },
        "partition_parameters": {
            "fixed_outlier_channels": selected.fixed_outlier_channels,
            "kmeans": (
                {
                    "implementation": "sklearn.cluster.KMeans",
                    "n_clusters": 2,
                    "init": "k-means++",
                    "n_init": 10,
                    "random_state": ROOT_SEED,
                    "size_constraint": None,
                }
                if selected.partition_method == "kmeans2"
                else None
            ),
        },
    }
    validate_partition_artifact(artifact, selected)
    return artifact


def validate_partition_artifact(
    artifact: Mapping[str, Any], spec: str | AllocationSpec | None = None
) -> dict[str, Any]:
    """Validate a loaded partition artifact and return count diagnostics."""

    selected = allocation_spec(spec or str(artifact.get("allocation")))
    if artifact.get("schema_version") != 1:
        raise ValueError("unsupported partition artifact schema")
    if artifact.get("allocation") != selected.name:
        raise ValueError("partition artifact allocation mismatch")
    layers = int(artifact.get("num_layers", 0))
    heads = int(artifact.get("num_heads", 0))
    head_dim = int(artifact.get("head_dim", 0))
    if min(layers, heads, head_dim) < 1:
        raise ValueError("partition dimensions must be positive")
    masks = artifact.get("outlier_masks")
    if not isinstance(masks, Mapping):
        raise ValueError("partition artifact has no outlier_masks mapping")
    diagnostics: dict[str, Any] = {}
    for component in COMPONENTS:
        mask = masks.get(component)
        if not isinstance(mask, torch.Tensor) or mask.dtype != torch.bool:
            raise ValueError(f"{component} outlier mask must be a bool Tensor")
        if tuple(mask.shape) != (layers, heads, head_dim):
            raise ValueError(
                f"{component} mask has shape {tuple(mask.shape)}, expected "
                f"{(layers, heads, head_dim)}"
            )
        counts = mask.sum(dim=-1)
        if selected.partition_method == "uniform" and bool(mask.any()):
            raise ValueError("uniform allocation cannot contain outlier channels")
        if selected.partition_method == "fixed32" and not torch.all(counts == 32):
            raise ValueError("fixed32 masks must select exactly 32 channels per head")
        if selected.is_split and (
            bool(torch.any(counts == 0)) or bool(torch.any(counts == head_dim))
        ):
            raise ValueError("split allocation groups must both be non-empty")
        diagnostics[component] = {
            "outlier_min": int(counts.min()),
            "outlier_mean": float(counts.float().mean()),
            "outlier_max": int(counts.max()),
            "outlier_counts": counts.cpu().contiguous(),
        }
    return diagnostics


def group_definitions(
    spec: str | AllocationSpec, outlier_mask: torch.Tensor
) -> tuple[GroupDefinition, ...]:
    """Return the ordered groups for one head's boolean outlier mask."""

    selected = allocation_spec(spec)
    if outlier_mask.ndim != 1 or outlier_mask.dtype != torch.bool:
        raise ValueError("outlier_mask must be a one-dimensional bool Tensor")
    head_dim = int(outlier_mask.numel())
    if not selected.is_split:
        if bool(outlier_mask.any()):
            raise ValueError("uniform allocation received a non-empty outlier mask")
        return (
            GroupDefinition(
                GROUP_ALL,
                selected.regular_bits,
                torch.arange(
                    head_dim, dtype=torch.long, device=outlier_mask.device
                ),
            ),
        )
    regular = torch.nonzero(~outlier_mask, as_tuple=False).flatten().long()
    outlier = torch.nonzero(outlier_mask, as_tuple=False).flatten().long()
    if regular.numel() == 0 or outlier.numel() == 0:
        raise ValueError("split allocation groups must both be non-empty")
    assert selected.outlier_bits is not None
    return (
        GroupDefinition(GROUP_REGULAR, selected.regular_bits, regular),
        GroupDefinition(GROUP_OUTLIER, selected.outlier_bits, outlier),
    )


def rotation_key(component: str, layer: int, head: int, group: str) -> str:
    """Return the stable flat key used in serialized rotation artifacts."""

    _validate_component(component)
    if layer < 0 or head < 0:
        raise ValueError("layer and head indexes must be non-negative")
    if group not in (GROUP_ALL, GROUP_REGULAR, GROUP_OUTLIER):
        raise ValueError(f"unknown rotation group {group!r}")
    return f"{component}__layer{layer:04d}__head{head:04d}__{group}"


def build_group_buckets(
    spec: str | AllocationSpec,
    partition_artifact: Mapping[str, Any],
    component: str,
) -> tuple[GroupBucket, ...]:
    """Bucket variable-size groups for batched training math."""

    selected = allocation_spec(spec)
    component = _validate_component(component)
    validate_partition_artifact(partition_artifact, selected)
    masks = partition_artifact["outlier_masks"][component]
    layers = int(partition_artifact["num_layers"])
    heads = int(partition_artifact["num_heads"])
    pending: dict[tuple[int, int], list[tuple[int, int, GroupDefinition]]] = {}
    for layer in range(layers):
        for head in range(heads):
            for group in group_definitions(selected, masks[layer, head]):
                pending.setdefault((group.bit_width, group.dimension), []).append(
                    (layer, head, group)
                )
    result: list[GroupBucket] = []
    for bit_width, dimension in sorted(pending):
        entries = pending[(bit_width, dimension)]
        bucket_id = f"{component}__bit{bit_width}__dim{dimension:04d}"
        result.append(
            GroupBucket(
                bucket_id=bucket_id,
                component=component,
                bit_width=bit_width,
                dimension=dimension,
                layer_indices=torch.tensor(
                    [entry[0] for entry in entries], dtype=torch.long
                ),
                head_indices=torch.tensor(
                    [entry[1] for entry in entries], dtype=torch.long
                ),
                indices=torch.stack([entry[2].indices for entry in entries]),
                rotation_keys=tuple(
                    rotation_key(component, layer, head, group.name)
                    for layer, head, group in entries
                ),
            )
        )
    return tuple(result)


def derive_stream_seed(root_seed: int, *coordinates: Any) -> int:
    """Derive a stable independent NumPy RNG stream from a root seed."""

    payload = json.dumps(
        ["rotation-allocation-v1", int(root_seed), *coordinates],
        ensure_ascii=True,
        separators=(",", ":"),
    ).encode()
    # PCG64 accepts arbitrary non-negative Python integers; 64 bits are ample
    # while remaining convenient to record in JSON manifests.
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little")


def _rotation_layout(
    spec: AllocationSpec, partition_artifact: Mapping[str, Any]
) -> Iterator[tuple[str, int, int, GroupDefinition]]:
    validate_partition_artifact(partition_artifact, spec)
    masks = partition_artifact["outlier_masks"]
    for component in COMPONENTS:
        for layer in range(int(partition_artifact["num_layers"])):
            for head in range(int(partition_artifact["num_heads"])):
                for group in group_definitions(spec, masks[component][layer, head]):
                    yield component, layer, head, group


def build_identity_rotation_artifact(
    spec: str | AllocationSpec, partition_artifact: Mapping[str, Any]
) -> dict[str, Any]:
    """Build group-wise identity rotations as a plain Tensor artifact."""

    selected = allocation_spec(spec)
    rotations: dict[str, torch.Tensor] = {}
    for component, layer, head, group in _rotation_layout(
        selected, partition_artifact
    ):
        rotations[rotation_key(component, layer, head, group.name)] = torch.eye(
            group.dimension, dtype=torch.float32
        )
    artifact = {
        "schema_version": 1,
        "kind": "identity",
        "allocation": selected.name,
        "root_seed": None,
        "partition_sha256": artifact_sha256(partition_artifact),
        "num_layers": int(partition_artifact["num_layers"]),
        "num_heads": int(partition_artifact["num_heads"]),
        "head_dim": int(partition_artifact["head_dim"]),
        "rotations": rotations,
        "stream_seeds": {},
    }
    validate_rotation_artifact(artifact, selected, partition_artifact)
    return artifact


def build_random_rotation_artifact(
    spec: str | AllocationSpec,
    partition_artifact: Mapping[str, Any],
    *,
    root_seed: int = ROOT_SEED,
) -> dict[str, Any]:
    """Build independent Haar rotations for every component/layer/head/group."""

    selected = allocation_spec(spec)
    rotations: dict[str, torch.Tensor] = {}
    stream_seeds: dict[str, int] = {}
    for component, layer, head, group in _rotation_layout(
        selected, partition_artifact
    ):
        key = rotation_key(component, layer, head, group.name)
        stream_seed = derive_stream_seed(
            root_seed, selected.name, component, layer, head, group.name
        )
        generator = np.random.default_rng(stream_seed)
        matrix = random_rotation_dense(group.dimension, generator).astype(
            np.float32, copy=False
        )
        rotations[key] = torch.from_numpy(matrix).contiguous()
        stream_seeds[key] = stream_seed
    artifact = {
        "schema_version": 1,
        "kind": "random",
        "allocation": selected.name,
        "root_seed": int(root_seed),
        "partition_sha256": artifact_sha256(partition_artifact),
        "num_layers": int(partition_artifact["num_layers"]),
        "num_heads": int(partition_artifact["num_heads"]),
        "head_dim": int(partition_artifact["head_dim"]),
        "rotations": rotations,
        "stream_seeds": stream_seeds,
    }
    validate_rotation_artifact(artifact, selected, partition_artifact)
    return artifact


def _artifact_rotations(artifact: Mapping[str, Any]) -> Mapping[str, torch.Tensor]:
    rotations = artifact.get("rotations")
    if not isinstance(rotations, Mapping):
        raise ValueError("rotation artifact has no rotations mapping")
    return rotations


def component_rotation_sha256(
    artifact: Mapping[str, Any], component: str
) -> str:
    """Hash all matrices for one component in stable key order."""

    component = _validate_component(component)
    prefix = component + "__"
    rotations = _artifact_rotations(artifact)
    selected = {key: rotations[key] for key in sorted(rotations) if key.startswith(prefix)}
    if not selected:
        raise ValueError(f"rotation artifact contains no {component} matrices")
    return artifact_sha256(selected)


def orthogonality_error(matrix: torch.Tensor) -> float:
    """Return maximum absolute ``R.T @ R - I`` error."""

    if matrix.ndim < 2 or matrix.shape[-1] != matrix.shape[-2]:
        raise ValueError("rotation must have square trailing dimensions")
    value = matrix.detach().double()
    identity = torch.eye(value.shape[-1], dtype=torch.float64, device=value.device)
    return float((value.transpose(-1, -2) @ value - identity).abs().max())


def rotation_orthogonality_errors(
    artifact: Mapping[str, Any]
) -> dict[str, float]:
    """Return per-matrix orthogonality errors for a loaded artifact."""

    return {
        key: orthogonality_error(matrix)
        for key, matrix in sorted(_artifact_rotations(artifact).items())
    }


def validate_rotation_artifact(
    artifact: Mapping[str, Any],
    spec: str | AllocationSpec,
    partition_artifact: Mapping[str, Any],
    *,
    orthogonality_atol: float = 2e-5,
) -> dict[str, Any]:
    """Validate layout, shapes, finiteness, hashes, and orthogonality."""

    selected = allocation_spec(spec)
    validate_partition_artifact(partition_artifact, selected)
    if artifact.get("schema_version") != 1:
        raise ValueError("unsupported rotation artifact schema")
    if artifact.get("allocation") != selected.name:
        raise ValueError("rotation artifact allocation mismatch")
    for field in ("num_layers", "num_heads", "head_dim"):
        if int(artifact.get(field, -1)) != int(partition_artifact[field]):
            raise ValueError(f"rotation artifact {field} mismatch")
    expected_partition_hash = artifact_sha256(partition_artifact)
    if artifact.get("partition_sha256") != expected_partition_hash:
        raise ValueError("rotation artifact partition hash mismatch")
    rotations = _artifact_rotations(artifact)
    expected: dict[str, int] = {}
    for component, layer, head, group in _rotation_layout(
        selected, partition_artifact
    ):
        expected[rotation_key(component, layer, head, group.name)] = group.dimension
    if set(rotations) != set(expected):
        missing = sorted(set(expected) - set(rotations))
        extra = sorted(set(rotations) - set(expected))
        raise ValueError(f"rotation keys differ: missing={missing[:3]}, extra={extra[:3]}")
    errors: dict[str, float] = {}
    for key, dimension in expected.items():
        matrix = rotations[key]
        if not isinstance(matrix, torch.Tensor):
            raise ValueError(f"rotation {key} is not a Tensor")
        if tuple(matrix.shape) != (dimension, dimension):
            raise ValueError(
                f"rotation {key} has shape {tuple(matrix.shape)}, expected "
                f"{(dimension, dimension)}"
            )
        if not torch.is_floating_point(matrix) or not torch.isfinite(matrix).all():
            raise ValueError(f"rotation {key} must be a finite floating Tensor")
        error = orthogonality_error(matrix)
        errors[key] = error
        if error > orthogonality_atol:
            raise ValueError(
                f"rotation {key} orthogonality error {error:.6g} exceeds "
                f"{orthogonality_atol:.6g}"
            )
    return {
        "matrix_count": len(errors),
        "orthogonality_max_abs": max(errors.values(), default=0.0),
        "key_sha256": component_rotation_sha256(artifact, "key"),
        "value_sha256": component_rotation_sha256(artifact, "value"),
    }


def audit_rotation_artifact(
    artifact: Mapping[str, Any],
    spec: str | AllocationSpec,
    partition_artifact: Mapping[str, Any],
    *,
    expected_kind: str | None = None,
    expected_root_seed: int | None = None,
    expected_value_sha256: str | None = None,
    orthogonality_atol: float = 2e-5,
) -> dict[str, Any]:
    """Return a non-throwing, serializable audit of a rotation artifact."""

    failures: list[str] = []
    diagnostics: dict[str, Any] = {}
    try:
        diagnostics = validate_rotation_artifact(
            artifact,
            spec,
            partition_artifact,
            orthogonality_atol=orthogonality_atol,
        )
    except (KeyError, TypeError, ValueError) as error:
        failures.append(f"{type(error).__name__}: {error}")
    if expected_kind is not None and artifact.get("kind") != expected_kind:
        failures.append(
            f"kind={artifact.get('kind')!r}, expected {expected_kind!r}"
        )
    if expected_root_seed is not None and artifact.get("root_seed") != expected_root_seed:
        failures.append(
            f"root_seed={artifact.get('root_seed')!r}, expected {expected_root_seed!r}"
        )
    actual_value_hash = diagnostics.get("value_sha256")
    if (
        expected_value_sha256 is not None
        and actual_value_hash != expected_value_sha256
    ):
        failures.append(
            f"value_sha256={actual_value_hash!r}, expected {expected_value_sha256!r}"
        )
    return {
        "status": "complete" if not failures else "failed",
        "failures": failures,
        "artifact_sha256": artifact_sha256(artifact),
        "partition_sha256": artifact_sha256(partition_artifact),
        **diagnostics,
    }


@lru_cache(maxsize=None)
def _codebook_values(bit_width: int, dimension: int) -> tuple[float, ...]:
    if bit_width < 1 or dimension < 1:
        raise ValueError("bit width and group dimension must be positive")
    return tuple(
        float(value) for value in optimal_centroids(bit_width, dimension).tolist()
    )


def lloyd_max_codebook(
    bit_width: int,
    dimension: int,
    *,
    device: torch.device | str,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Return the exact local TurboQuant Lloyd-Max codebook for a group."""

    return torch.tensor(
        _codebook_values(int(bit_width), int(dimension)),
        device=device,
        dtype=dtype,
    )


def quantize_lloyd_max(
    values: torch.Tensor, bit_width: int, dimension: int
) -> torch.Tensor:
    """Nearest-centroid scalar quantization for one group dimension."""

    centroids = lloyd_max_codebook(
        bit_width, dimension, device=values.device, dtype=values.dtype
    )
    return _quantize_with_centroids(values, centroids)


def _quantize_with_centroids(
    values: torch.Tensor, centroids: torch.Tensor
) -> torch.Tensor:
    boundaries = (centroids[:-1] + centroids[1:]) * 0.5
    indexes = torch.bucketize(values.contiguous(), boundaries)
    return centroids[indexes]


def grouped_reconstruct(
    vectors: torch.Tensor,
    spec: str | AllocationSpec,
    outlier_mask: torch.Tensor,
    rotations: Mapping[str, torch.Tensor],
    *,
    codebook_cache: MutableMapping[tuple[int, int, str], torch.Tensor] | None = None,
) -> torch.Tensor:
    """Apply exact group-wise normalize/rotate/quantize/invert/scatter.

    Args:
        vectors: Arbitrary leading dimensions followed by ``head_dim``.
        outlier_mask: Boolean channel mask for this head.
        rotations: Mapping from group name (``all`` or ``regular``/``outlier``)
            to its square forward rotation.

    The computation runs in float32, rounds each rescaling norm through FP16
    exactly as the accounted cache payload does, and returns the source
    dtype/device/shape. Exact zero groups remain zero.
    """

    selected = allocation_spec(spec)
    if vectors.ndim < 1 or vectors.shape[-1] != outlier_mask.numel():
        raise ValueError("vectors and outlier mask have incompatible head dimensions")
    if not torch.is_floating_point(vectors):
        raise ValueError("vectors must be floating point")
    source_dtype = vectors.dtype
    source_device = vectors.device
    work = vectors.float()
    reconstructed = torch.zeros_like(work)
    groups = group_definitions(selected, outlier_mask)
    if set(rotations) != {group.name for group in groups}:
        raise ValueError(
            f"group rotation names differ: got {sorted(rotations)}, "
            f"expected {sorted(group.name for group in groups)}"
        )
    for group in groups:
        indexes = group.indices.to(device=source_device)
        matrix = rotations[group.name].to(device=source_device, dtype=torch.float32)
        if tuple(matrix.shape) != (group.dimension, group.dimension):
            raise ValueError(
                f"{group.name} rotation shape {tuple(matrix.shape)} does not match "
                f"group dimension {group.dimension}"
            )
        gathered = work.index_select(-1, indexes)
        norms = torch.linalg.vector_norm(gathered, dim=-1, keepdim=True)
        stored_norms = norms.to(torch.float16).to(torch.float32)
        normalized = gathered / norms.clamp_min(torch.finfo(torch.float32).eps)
        rotated = torch.matmul(normalized, matrix.transpose(-1, -2))
        cache_key = (group.bit_width, group.dimension, str(source_device))
        centroids = (
            codebook_cache.get(cache_key) if codebook_cache is not None else None
        )
        if centroids is None:
            centroids = lloyd_max_codebook(
                group.bit_width,
                group.dimension,
                device=source_device,
                dtype=torch.float32,
            )
            if codebook_cache is not None:
                codebook_cache[cache_key] = centroids
        quantized = _quantize_with_centroids(rotated, centroids)
        restored = torch.matmul(quantized, matrix) * stored_norms
        restored = torch.where(norms > 0, restored, torch.zeros_like(restored))
        reconstructed.index_copy_(-1, indexes, restored)
    return reconstructed.to(device=source_device, dtype=source_dtype)


def reconstruct_head(
    vectors: torch.Tensor,
    spec: str | AllocationSpec,
    partition_artifact: Mapping[str, Any],
    rotation_artifact: Mapping[str, Any],
    *,
    component: str,
    layer: int,
    head: int,
    codebook_cache: MutableMapping[tuple[int, int, str], torch.Tensor] | None = None,
) -> torch.Tensor:
    """High-level reconstruction for one saved layer/head artifact entry."""

    selected = allocation_spec(spec)
    component = _validate_component(component)
    masks = partition_artifact["outlier_masks"]
    mask = masks[component][layer, head]
    flat = _artifact_rotations(rotation_artifact)
    per_group = {
        group.name: flat[rotation_key(component, layer, head, group.name)]
        for group in group_definitions(selected, mask)
    }
    return grouped_reconstruct(
        vectors,
        selected,
        mask,
        per_group,
        codebook_cache=codebook_cache,
    )


def reconstruct_projection(
    output: torch.Tensor,
    spec: str | AllocationSpec,
    partition_artifact: Mapping[str, Any],
    rotation_artifact: Mapping[str, Any],
    *,
    component: str,
    layer: int,
    codebook_cache: MutableMapping[tuple[int, int, str], torch.Tensor] | None = None,
) -> torch.Tensor:
    """Reconstruct a Llama K/V projection of shape ``(batch, sequence, H*d)``."""

    selected = allocation_spec(spec)
    component = _validate_component(component)
    if output.ndim != 3:
        raise ValueError("projection output must have shape (batch, sequence, width)")
    heads = int(partition_artifact["num_heads"])
    head_dim = int(partition_artifact["head_dim"])
    if output.shape[-1] != heads * head_dim:
        raise ValueError(
            f"projection width {output.shape[-1]} does not equal {heads}*{head_dim}"
        )
    vectors = output.reshape(output.shape[0], output.shape[1], heads, head_dim)
    reconstructed = torch.empty_like(vectors)
    for head in range(heads):
        reconstructed[:, :, head] = reconstruct_head(
            vectors[:, :, head],
            selected,
            partition_artifact,
            rotation_artifact,
            component=component,
            layer=layer,
            head=head,
            codebook_cache=codebook_cache,
        )
    return reconstructed.reshape_as(output)


def _initialize_counters(counters: MutableMapping[str, int]) -> None:
    counters.setdefault("quantized_coordinates", 0)
    counters.setdefault("group_calls", 0)
    counters.setdefault("group_vectors", 0)
    for bit_width in (2, 3, 4):
        counters.setdefault(f"bit{bit_width}_coordinates", 0)
    for component in COMPONENTS:
        counters.setdefault(f"{component}_vectors", 0)
        counters.setdefault(f"{component}_prefill_vectors", 0)
        counters.setdefault(f"{component}_decode_vectors", 0)
        counters.setdefault(f"{component}_group_calls", 0)
        counters.setdefault(f"{component}_group_vectors", 0)
        for bit_width in (2, 3, 4):
            counters.setdefault(f"{component}_bit{bit_width}_coordinates", 0)


@contextmanager
def install_projection_hooks(
    model: torch.nn.Module,
    spec: str | AllocationSpec,
    partition_artifact: Mapping[str, Any],
    rotation_artifact: Mapping[str, Any],
    *,
    counters: MutableMapping[str, int] | None = None,
) -> Iterator[MutableMapping[str, int]]:
    """Temporarily install group-wise pre-RoPE K and post-projection V hooks.

    A projection with sequence length greater than one is counted as prefill;
    sequence length one is counted as decode.  Counts are head vectors, i.e.
    ``batch * sequence * num_kv_heads``.
    """

    selected = allocation_spec(spec)
    validate_partition_artifact(partition_artifact, selected)
    validate_rotation_artifact(rotation_artifact, selected, partition_artifact)
    layers = model.model.layers
    if len(layers) != int(partition_artifact["num_layers"]):
        raise ValueError("model layer count differs from partition artifact")
    counts: MutableMapping[str, int] = counters if counters is not None else {}
    _initialize_counters(counts)
    heads = int(partition_artifact["num_heads"])
    handles: list[torch.utils.hooks.RemovableHandle] = []
    # Materialize indexes, matrices, and codebooks once per device.  Decode
    # invokes every projection hook once per generated token, so rebuilding
    # group definitions (and especially copying tiny index tensors CPU->GPU)
    # inside the hook would dominate the quality-emulation runtime.
    # bit width, head indexes, per-head channel indexes, per-head rotations,
    # shared Lloyd-Max centroids.  Equal-dimension groups are batched so the
    # common uniform/fixed paths issue one/two GPU matmuls per projection layer
    # instead of one matmul per KV head and group.
    RuntimeBucket = tuple[
        int, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
    ]
    runtime_groups: dict[
        str, dict[tuple[str, int], tuple[RuntimeBucket, ...]]
    ] = {}

    def groups_on(
        device: torch.device,
    ) -> dict[tuple[str, int], tuple[RuntimeBucket, ...]]:
        cache_key = str(device)
        cached = runtime_groups.get(cache_key)
        if cached is not None:
            return cached
        flat = _artifact_rotations(rotation_artifact)
        cached = {}
        for component in COMPONENTS:
            masks = partition_artifact["outlier_masks"][component]
            for layer in range(int(partition_artifact["num_layers"])):
                pending: dict[
                    tuple[int, int],
                    list[tuple[int, GroupDefinition, torch.Tensor]],
                ] = {}
                for head in range(heads):
                    for group in group_definitions(selected, masks[layer, head]):
                        pending.setdefault(
                            (group.bit_width, group.dimension), []
                        ).append(
                            (
                                head,
                                group,
                                flat[rotation_key(component, layer, head, group.name)],
                            )
                        )
                buckets: list[RuntimeBucket] = []
                for (bit_width, dimension), entries in sorted(pending.items()):
                    buckets.append(
                        (
                            bit_width,
                            torch.tensor(
                                [entry[0] for entry in entries],
                                dtype=torch.long,
                                device=device,
                            ),
                            torch.stack(
                                [entry[1].indices for entry in entries]
                            ).to(device=device),
                            torch.stack([entry[2] for entry in entries]).to(
                                device=device, dtype=torch.float32
                            ),
                            lloyd_max_codebook(
                                bit_width,
                                dimension,
                                device=device,
                                dtype=torch.float32,
                            ),
                        )
                    )
                cached[(component, layer)] = tuple(buckets)
        runtime_groups[cache_key] = cached
        return cached

    def reconstruct_prepared(
        output: torch.Tensor,
        prepared_buckets: tuple[RuntimeBucket, ...],
    ) -> torch.Tensor:
        if output.ndim != 3:
            raise ValueError(
                "projection output must have shape (batch, sequence, width)"
            )
        head_dim = int(partition_artifact["head_dim"])
        if output.shape[-1] != heads * head_dim:
            raise ValueError("projection width differs from the partition artifact")
        batch, sequence = int(output.shape[0]), int(output.shape[1])
        vectors = output.reshape(batch, sequence, heads, head_dim).float()
        result = torch.zeros_like(vectors)
        for _bits, head_indexes, channel_indexes, matrices, centroids in prepared_buckets:
            selected_vectors = vectors.index_select(2, head_indexes)
            group_count, dimension = channel_indexes.shape
            gather_indexes = channel_indexes.view(1, 1, group_count, dimension).expand(
                batch, sequence, -1, -1
            )
            gathered = selected_vectors.gather(-1, gather_indexes)
            norms = torch.linalg.vector_norm(gathered, dim=-1, keepdim=True)
            stored_norms = norms.to(torch.float16).to(torch.float32)
            normalized = gathered / norms.clamp_min(torch.finfo(torch.float32).eps)
            rotated = torch.matmul(
                normalized.unsqueeze(-2), matrices.transpose(-1, -2)
            ).squeeze(-2)
            quantized = _quantize_with_centroids(rotated, centroids)
            restored = torch.matmul(
                quantized.unsqueeze(-2), matrices
            ).squeeze(-2) * stored_norms
            restored = torch.where(norms > 0, restored, torch.zeros_like(restored))
            bucket_result = torch.zeros_like(selected_vectors)
            bucket_result.scatter_(-1, gather_indexes, restored)
            result.index_add_(2, head_indexes, bucket_result)
        return result.reshape_as(output).to(dtype=output.dtype)

    def register(
        projection: torch.nn.Module, layer: int, component: str
    ) -> None:
        def hook(
            _module: torch.nn.Module,
            _inputs: tuple[torch.Tensor, ...],
            output: torch.Tensor,
            *,
            selected_layer: int = layer,
            selected_component: str = component,
        ) -> torch.Tensor:
            prepared_buckets = groups_on(output.device)[
                (selected_component, selected_layer)
            ]
            result = reconstruct_prepared(
                output, prepared_buckets
            )
            vector_count = int(output.shape[0] * output.shape[1] * heads)
            phase = "prefill" if int(output.shape[1]) > 1 else "decode"
            counts[f"{selected_component}_vectors"] += vector_count
            counts[f"{selected_component}_{phase}_vectors"] += vector_count
            batch_sequence = int(output.shape[0] * output.shape[1])
            group_calls = sum(
                int(bucket[1].numel()) for bucket in prepared_buckets
            )
            group_vectors = batch_sequence * group_calls
            coordinate_count = batch_sequence * sum(
                int(bucket[2].numel()) for bucket in prepared_buckets
            )
            counts["quantized_coordinates"] += coordinate_count
            counts["group_calls"] += group_calls
            counts["group_vectors"] += group_vectors
            counts[f"{selected_component}_group_calls"] += group_calls
            counts[f"{selected_component}_group_vectors"] += group_vectors
            for bit_width, _heads, indexes, _matrices, _centroids in prepared_buckets:
                coordinates = batch_sequence * int(indexes.numel())
                counts[f"bit{bit_width}_coordinates"] += coordinates
                counts[
                    f"{selected_component}_bit{bit_width}_coordinates"
                ] += coordinates
            return result

        handles.append(projection.register_forward_hook(hook))

    for layer_index, layer in enumerate(layers):
        register(layer.self_attn.k_proj, layer_index, "key")
        register(layer.self_attn.v_proj, layer_index, "value")
    try:
        yield counts
    finally:
        for handle in handles:
            handle.remove()
        runtime_groups.clear()


def _align_up(value: int, alignment: int) -> int:
    if alignment < 1:
        raise ValueError("byte alignment must be positive")
    return ((value + alignment - 1) // alignment) * alignment


def storage_accounting(
    spec: str | AllocationSpec,
    partition_artifact: Mapping[str, Any],
    *,
    norm_bits: int = 16,
    byte_alignment: int = 1,
    rotation_artifact: Mapping[str, Any] | None = None,
    rotation_dtype_bytes: int = 4,
) -> dict[str, Any]:
    """Compute exact packed and theoretical storage for one cached token.

    Every group's index stream is byte-packed independently, then its FP norm
    is stored.  The combined per-head vector payload is rounded up to
    ``byte_alignment``.  Theoretical bytes/token sum this aligned payload over
    every layer and KV head.  BPE values are per scalar channel.
    """

    selected = allocation_spec(spec)
    validate_partition_artifact(partition_artifact, selected)
    if norm_bits < 1 or norm_bits % 8:
        raise ValueError("norm_bits must be a positive whole number of bytes")
    if rotation_dtype_bytes < 1:
        raise ValueError("rotation_dtype_bytes must be positive")
    layers = int(partition_artifact["num_layers"])
    heads = int(partition_artifact["num_heads"])
    head_dim = int(partition_artifact["head_dim"])
    component_rows: dict[str, list[dict[str, int]]] = {}
    component_summary: dict[str, dict[str, Any]] = {}
    for component in COMPONENTS:
        rows: list[dict[str, int]] = []
        masks = partition_artifact["outlier_masks"][component]
        for layer in range(layers):
            for head in range(heads):
                groups = group_definitions(selected, masks[layer, head])
                index_bits = sum(group.dimension * group.bit_width for group in groups)
                packed_index_bytes = sum(
                    math.ceil(group.dimension * group.bit_width / 8)
                    for group in groups
                )
                norm_bytes = len(groups) * (norm_bits // 8)
                unaligned_bytes = packed_index_bytes + norm_bytes
                packed_bytes = _align_up(unaligned_bytes, byte_alignment)
                rows.append(
                    {
                        "layer": layer,
                        "head": head,
                        "groups": len(groups),
                        "outlier_channels": int(masks[layer, head].sum()),
                        "index_bits": index_bits,
                        "packed_index_bytes": packed_index_bytes,
                        "norm_bytes": norm_bytes,
                        "unaligned_bytes": unaligned_bytes,
                        "alignment_bytes": packed_bytes - unaligned_bytes,
                        "packed_bytes": packed_bytes,
                    }
                )
        vector_count = layers * heads
        index_bits_total = sum(row["index_bits"] for row in rows)
        norm_bits_total = sum(row["norm_bytes"] * 8 for row in rows)
        packed_bytes_total = sum(row["packed_bytes"] for row in rows)
        unaligned_bits_total = index_bits_total + norm_bits_total
        outlier_counts = [row["outlier_channels"] for row in rows]
        component_rows[component] = rows
        component_summary[component] = {
            "index_bpe": index_bits_total / (vector_count * head_dim),
            "norm_bpe": norm_bits_total / (vector_count * head_dim),
            "unaligned_effective_bpe": unaligned_bits_total
            / (vector_count * head_dim),
            "alignment_bpe": (
                packed_bytes_total * 8 - unaligned_bits_total
            )
            / (vector_count * head_dim),
            "effective_bpe": packed_bytes_total * 8
            / (vector_count * head_dim),
            "packed_bytes_per_head_vector_mean": packed_bytes_total / vector_count,
            "theoretical_bytes_per_token": packed_bytes_total,
            "outlier_channels_min": min(outlier_counts),
            "outlier_channels_mean": sum(outlier_counts) / len(outlier_counts),
            "outlier_channels_max": max(outlier_counts),
        }

    if rotation_artifact is None:
        static_rotation_bytes = 0
        for _component, _layer, _head, group in _rotation_layout(
            selected, partition_artifact
        ):
            static_rotation_bytes += (
                group.dimension * group.dimension * rotation_dtype_bytes
            )
    else:
        static_rotation_bytes = sum(
            int(matrix.numel() * matrix.element_size())
            for matrix in _artifact_rotations(rotation_artifact).values()
        )
    total_bytes = sum(
        int(component_summary[component]["theoretical_bytes_per_token"])
        for component in COMPONENTS
    )
    return {
        "allocation": selected.name,
        "norm_bits": norm_bits,
        "byte_alignment": byte_alignment,
        "pack_groups_separately": True,
        "num_layers": layers,
        "num_heads": heads,
        "head_dim": head_dim,
        "components": component_summary,
        "head_rows": component_rows,
        "kv_average_index_bpe": sum(
            float(component_summary[c]["index_bpe"]) for c in COMPONENTS
        )
        / 2.0,
        "kv_average_effective_bpe": sum(
            float(component_summary[c]["effective_bpe"]) for c in COMPONENTS
        )
        / 2.0,
        "theoretical_kv_bytes_per_token": total_bytes,
        "static_rotation_bytes": static_rotation_bytes,
    }


def tensor_sha256(tensor: torch.Tensor) -> str:
    """Hash tensor dtype, shape, and canonical contiguous CPU bytes."""

    value = tensor.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode())
    digest.update(json.dumps(list(value.shape), separators=(",", ":")).encode())
    digest.update(value.view(torch.uint8).numpy().tobytes(order="C"))
    return digest.hexdigest()


def _update_artifact_digest(digest: Any, value: Any) -> None:
    if isinstance(value, torch.Tensor):
        digest.update(b"tensor:")
        digest.update(tensor_sha256(value).encode())
    elif isinstance(value, Mapping):
        digest.update(b"mapping{")
        for key in sorted(value, key=lambda item: str(item)):
            _update_artifact_digest(digest, str(key))
            _update_artifact_digest(digest, value[key])
        digest.update(b"}")
    elif isinstance(value, (list, tuple)):
        digest.update(b"sequence[")
        for item in value:
            _update_artifact_digest(digest, item)
        digest.update(b"]")
    elif isinstance(value, np.ndarray):
        _update_artifact_digest(digest, torch.from_numpy(np.ascontiguousarray(value)))
    elif isinstance(value, Path):
        _update_artifact_digest(digest, str(value))
    elif value is None or isinstance(value, (str, int, float, bool)):
        digest.update(type(value).__name__.encode() + b":")
        digest.update(
            json.dumps(value, ensure_ascii=True, allow_nan=False, separators=(",", ":")).encode()
        )
    else:
        raise TypeError(f"unsupported artifact value type: {type(value).__name__}")


def artifact_sha256(artifact: Any) -> str:
    """Hash a plain nested artifact independent of dict order or Tensor device."""

    digest = hashlib.sha256()
    _update_artifact_digest(digest, artifact)
    return digest.hexdigest()


def _flat_rotations_from_artifact_or_mapping(
    rotations: Mapping[str, Any],
) -> Mapping[str, torch.Tensor]:
    nested = rotations.get("rotations")
    if isinstance(nested, Mapping):
        return nested
    if not all(isinstance(value, torch.Tensor) for value in rotations.values()):
        raise ValueError("rotations must be a flat Tensor mapping or rotation artifact")
    return rotations  # type: ignore[return-value]


def bucket_rotation_tensors(
    buckets: Sequence[GroupBucket],
    rotations: Mapping[str, Any],
    *,
    device: torch.device | str,
    dtype: torch.dtype = torch.float32,
) -> dict[str, torch.Tensor]:
    """Stack a flat saved artifact into differentiable runtime buckets."""

    flat = _flat_rotations_from_artifact_or_mapping(rotations)
    return {
        bucket.bucket_id: torch.stack(
            [flat[key].to(device=device, dtype=dtype) for key in bucket.rotation_keys]
        )
        for bucket in buckets
    }


def flatten_bucket_rotations(
    buckets: Sequence[GroupBucket], rotation_buckets: Mapping[str, torch.Tensor]
) -> dict[str, torch.Tensor]:
    """Convert runtime buckets back to the flat serializable matrix mapping."""

    result: dict[str, torch.Tensor] = {}
    for bucket in buckets:
        matrices = rotation_buckets[bucket.bucket_id]
        if tuple(matrices.shape) != (
            len(bucket.rotation_keys),
            bucket.dimension,
            bucket.dimension,
        ):
            raise ValueError(f"rotation bucket {bucket.bucket_id} has invalid shape")
        for index, key in enumerate(bucket.rotation_keys):
            result[key] = matrices[index].detach().cpu().contiguous()
    return result


def bucketed_grouped_objective(
    keys: torch.Tensor,
    buckets: Sequence[GroupBucket],
    rotation_buckets: Mapping[str, torch.Tensor],
    *,
    token_indices: torch.Tensor | None = None,
    codebooks: Mapping[str, torch.Tensor] | None = None,
) -> torch.Tensor:
    """Differentiable full original-scale grouped reconstruction MSE.

    ``keys`` has shape ``(layers, heads, tokens, head_dim)``.  ``token_indices``
    may have shape ``(layers, heads, sampled_tokens)``.  The selected centroids
    are detached targets for each optimization step.  For orthogonal rotations,
    the norm-weighted rotated-space squared error used here is exactly equal to
    the full gather/inverse/scatter reconstruction squared error, while retaining
    the useful piecewise-constant quantizer surrogate gradient.
    """

    if keys.ndim != 4:
        raise ValueError("keys must have shape (layers, heads, tokens, head_dim)")
    device = keys.device
    work = keys.float()
    if token_indices is not None:
        expected_prefix = tuple(keys.shape[:2])
        if token_indices.ndim != 3 or tuple(token_indices.shape[:2]) != expected_prefix:
            raise ValueError(
                "token_indices must have shape (layers, heads, sampled_tokens)"
            )
        token_indices = token_indices.to(device=device, dtype=torch.long)
    total_squared = torch.zeros((), device=device, dtype=torch.float32)
    total_values = 0
    for bucket in buckets:
        layers = bucket.layer_indices.to(device=device)
        heads = bucket.head_indices.to(device=device)
        selected = work[layers, heads]
        if token_indices is not None:
            selected_tokens = token_indices[layers, heads]
            selected = selected.gather(
                1,
                selected_tokens.unsqueeze(-1).expand(
                    -1, -1, selected.shape[-1]
                ),
            )
        channels = bucket.indices.to(device=device)
        gathered = selected.gather(
            -1, channels.unsqueeze(1).expand(-1, selected.shape[1], -1)
        )
        norms = torch.linalg.vector_norm(gathered, dim=-1, keepdim=True)
        stored_norms = norms.to(torch.float16).to(torch.float32)
        normalized = gathered / norms.clamp_min(torch.finfo(torch.float32).eps)
        matrices = rotation_buckets[bucket.bucket_id]
        rotated = torch.matmul(normalized, matrices.transpose(-1, -2))
        centroids = (
            codebooks[bucket.bucket_id]
            if codebooks is not None
            else lloyd_max_codebook(
                bucket.bit_width,
                bucket.dimension,
                device=device,
                dtype=torch.float32,
            )
        )
        targets = _quantize_with_centroids(rotated.detach(), centroids)
        # Decoder rescaling uses the actually stored FP16 norm.  Orthogonal
        # invariance lets us compute the exact original-scale reconstruction
        # error in rotated coordinates without materializing inverse/scatter:
        # ||n*u - n16*q*R|| == ||n*(u*R.T) - n16*q||.
        total_squared = total_squared + torch.sum(
            (rotated * norms - targets * stored_norms).square()
        )
        total_values += gathered.numel()
    if total_values == 0:
        raise ValueError("group buckets are empty")
    return total_squared / total_values


def grouped_original_scale_objective(
    keys: torch.Tensor,
    spec: str | AllocationSpec,
    partition_artifact: Mapping[str, Any],
    rotations: Mapping[str, Any],
    *,
    token_indices: torch.Tensor | None = None,
) -> torch.Tensor:
    """Convenience wrapper around :func:`bucketed_grouped_objective` for Keys."""

    selected = allocation_spec(spec)
    buckets = build_group_buckets(selected, partition_artifact, "key")
    runtime = bucket_rotation_tensors(
        buckets, rotations, device=keys.device, dtype=torch.float32
    )
    codebooks = {
        bucket.bucket_id: lloyd_max_codebook(
            bucket.bit_width,
            bucket.dimension,
            device=keys.device,
            dtype=torch.float32,
        )
        for bucket in buckets
    }
    return bucketed_grouped_objective(
        keys,
        buckets,
        runtime,
        token_indices=token_indices,
        codebooks=codebooks,
    )


def _tree_to_cpu(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, Mapping):
        return {key: _tree_to_cpu(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_tree_to_cpu(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_tree_to_cpu(item) for item in value)
    return value


def _move_optimizer_state(optimizer: torch.optim.Optimizer, device: torch.device) -> None:
    for state in optimizer.state.values():
        for key, value in list(state.items()):
            if isinstance(value, torch.Tensor):
                state[key] = value.to(device=device)


def _training_state(
    *,
    selected: AllocationSpec,
    partition_hash: str,
    random_hash: str,
    step: int,
    steps: int,
    batch_tokens: int,
    learning_rate: float,
    minimum_learning_rate: float,
    root_seed: int,
    parameters: torch.nn.ParameterDict,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    generator: torch.Generator,
    curve_rows: Sequence[Mapping[str, Any]],
    initial_objective_mse: float,
    validation_keys_sha256: str | None,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "allocation": selected.name,
        "partition_sha256": partition_hash,
        "random_artifact_sha256": random_hash,
        "step": int(step),
        "target_steps": int(steps),
        "batch_tokens": int(batch_tokens),
        "learning_rate": float(learning_rate),
        "minimum_learning_rate": float(minimum_learning_rate),
        "root_seed": int(root_seed),
        "parameter_state": {
            key: value.detach().cpu().clone() for key, value in parameters.items()
        },
        "optimizer_state": _tree_to_cpu(optimizer.state_dict()),
        "scheduler_state": _tree_to_cpu(scheduler.state_dict()),
        "generator_state": generator.get_state().cpu().clone(),
        "curve_rows": [dict(row) for row in curve_rows],
        "initial_objective_mse": float(initial_objective_mse),
        "validation_keys_sha256": validation_keys_sha256,
    }


def train_learned_key_rotations(
    calibration_keys: torch.Tensor,
    spec: str | AllocationSpec,
    partition_artifact: Mapping[str, Any],
    random_artifact: Mapping[str, Any],
    *,
    validation_keys: torch.Tensor | None = None,
    steps: int = 10_000,
    batch_tokens: int = 256,
    learning_rate: float = 0.005,
    minimum_learning_rate: float = 0.00025,
    root_seed: int = ROOT_SEED,
    gradient_clip: float = 1.0,
    curve_interval: int = 100,
    checkpoint_interval: int = 100,
    checkpoint_callback: Callable[[dict[str, Any]], None] | None = None,
    resume_state: Mapping[str, Any] | None = None,
    stop_after_step: int | None = None,
) -> dict[str, Any]:
    """Train allocation-specific Key rotations from their exact Random start.

    The optimizer is Adam over dense Cayley parameters.  Every step samples
    ``batch_tokens`` independently for each layer/head, computes full
    original-scale reconstruction MSE across all groups (including the FP16
    stored-norm roundtrip), clips the aggregate
    gradient norm, and advances ``CosineAnnealingLR(T_max=steps,
    eta_min=minimum_learning_rate)``.  Value rotations are copied byte-for-byte
    from ``random_artifact`` into the returned learned artifact.

    When ``validation_keys`` is supplied, its fixed full held-out objective is
    evaluated at step zero and every curve interval (including the terminal
    step).  It is diagnostic only and never selects a checkpoint.  Resume
    state records its hash so a different held-out tensor cannot be substituted.

    ``resume_state`` and states passed to ``checkpoint_callback`` are plain
    serializable trees.  ``stop_after_step`` supports controlled interruption
    while retaining the original scheduler horizon in ``steps``.
    """

    selected = allocation_spec(spec)
    validate_partition_artifact(partition_artifact, selected)
    validate_rotation_artifact(random_artifact, selected, partition_artifact)
    if random_artifact.get("kind") != "random":
        raise ValueError("learned training must start from a random rotation artifact")
    if int(random_artifact.get("root_seed", -1)) != int(root_seed):
        raise ValueError("random artifact root seed differs from training root seed")
    if calibration_keys.ndim != 4:
        raise ValueError(
            "calibration_keys must have shape (layers, heads, tokens, head_dim)"
        )
    expected_shape = (
        int(partition_artifact["num_layers"]),
        int(partition_artifact["num_heads"]),
        int(calibration_keys.shape[2]),
        int(partition_artifact["head_dim"]),
    )
    if tuple(calibration_keys.shape) != expected_shape:
        raise ValueError(
            f"calibration key shape {tuple(calibration_keys.shape)} does not match "
            f"{expected_shape}"
        )
    if not torch.isfinite(calibration_keys.float()).all():
        raise ValueError("calibration keys contain NaN or Inf")
    if min(steps, batch_tokens, curve_interval, checkpoint_interval) < 1:
        raise ValueError("steps, batch size, and intervals must be positive")
    if learning_rate <= 0 or not 0 <= minimum_learning_rate <= learning_rate:
        raise ValueError("learning rates must satisfy 0 <= minimum <= initial")
    if gradient_clip <= 0:
        raise ValueError("gradient_clip must be positive")
    end_step = steps if stop_after_step is None else int(stop_after_step)
    if not 0 <= end_step <= steps:
        raise ValueError("stop_after_step must lie between zero and steps")

    device = calibration_keys.device
    keys = calibration_keys.detach().to(device=device, dtype=torch.float32)
    validation: torch.Tensor | None = None
    validation_hash: str | None = None
    if validation_keys is not None:
        expected_validation = (
            int(partition_artifact["num_layers"]),
            int(partition_artifact["num_heads"]),
            int(validation_keys.shape[2]) if validation_keys.ndim == 4 else -1,
            int(partition_artifact["head_dim"]),
        )
        if validation_keys.ndim != 4 or tuple(validation_keys.shape) != expected_validation:
            raise ValueError(
                "validation_keys must have shape (partition layers, heads, "
                "tokens, head_dim)"
            )
        if not torch.isfinite(validation_keys.float()).all():
            raise ValueError("validation keys contain NaN or Inf")
        validation_hash = tensor_sha256(validation_keys)
        validation = validation_keys.detach().to(device=device, dtype=torch.float32)
    buckets = build_group_buckets(selected, partition_artifact, "key")
    initial_buckets = bucket_rotation_tensors(
        buckets, random_artifact, device=device, dtype=torch.float32
    )
    training_codebooks = {
        bucket.bucket_id: lloyd_max_codebook(
            bucket.bit_width,
            bucket.dimension,
            device=device,
            dtype=torch.float32,
        )
        for bucket in buckets
    }
    parameters = torch.nn.ParameterDict(
        {
            bucket.bucket_id: torch.nn.Parameter(
                torch.zeros_like(initial_buckets[bucket.bucket_id])
            )
            for bucket in buckets
        }
    )
    optimizer = torch.optim.Adam(parameters.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=steps, eta_min=minimum_learning_rate
    )
    generator = torch.Generator(device=device)
    generator.manual_seed(
        derive_stream_seed(root_seed, selected.name, "learned-key-minibatches")
    )
    partition_hash = artifact_sha256(partition_artifact)
    random_hash = artifact_sha256(random_artifact)
    start_step = 0
    curve_rows: list[dict[str, Any]] = []

    with torch.no_grad():
        initial_objective = float(
            bucketed_grouped_objective(
                keys,
                buckets,
                initial_buckets,
                codebooks=training_codebooks,
            ).cpu()
        )
        initial_validation = (
            float(
                bucketed_grouped_objective(
                    validation,
                    buckets,
                    initial_buckets,
                    codebooks=training_codebooks,
                ).cpu()
            )
            if validation is not None
            else None
        )
    if resume_state is None:
        curve_rows.append(
            {
                "step": 0,
                "minibatch_original_scale_mse": None,
                "full_original_scale_mse": initial_objective,
                "validation_original_scale_mse": initial_validation,
                "learning_rate": learning_rate,
                "gradient_norm": None,
            }
        )
    else:
        required = {
            "schema_version": 1,
            "allocation": selected.name,
            "partition_sha256": partition_hash,
            "random_artifact_sha256": random_hash,
            "target_steps": steps,
            "batch_tokens": batch_tokens,
            "learning_rate": learning_rate,
            "minimum_learning_rate": minimum_learning_rate,
            "root_seed": root_seed,
            "validation_keys_sha256": validation_hash,
        }
        for field, expected in required.items():
            if resume_state.get(field) != expected:
                raise ValueError(
                    f"resume state {field}={resume_state.get(field)!r}, "
                    f"expected {expected!r}"
                )
        parameter_state = resume_state.get("parameter_state")
        if not isinstance(parameter_state, Mapping) or set(parameter_state) != set(parameters):
            raise ValueError("resume parameter bucket layout differs")
        with torch.no_grad():
            for key, parameter in parameters.items():
                parameter.copy_(parameter_state[key].to(device=device, dtype=torch.float32))
        optimizer.load_state_dict(resume_state["optimizer_state"])
        _move_optimizer_state(optimizer, device)
        scheduler.load_state_dict(resume_state["scheduler_state"])
        generator.set_state(resume_state["generator_state"].cpu())
        start_step = int(resume_state["step"])
        curve_rows = [dict(row) for row in resume_state.get("curve_rows", [])]
        recorded_initial = float(resume_state.get("initial_objective_mse", math.nan))
        if not math.isclose(recorded_initial, initial_objective, rel_tol=0, abs_tol=1e-12):
            raise ValueError("resume initial objective differs from random step zero")
    if end_step < start_step:
        raise ValueError("stop_after_step precedes the resumed step")

    latest_loss: float | None = None
    latest_gradient_norm: float | None = None
    for step in range(start_step + 1, end_step + 1):
        token_indices = torch.randint(
            int(keys.shape[2]),
            (int(keys.shape[0]), int(keys.shape[1]), batch_tokens),
            generator=generator,
            device=device,
        )
        optimizer.zero_grad(set_to_none=True)
        current_buckets = {
            bucket.bucket_id: cayley_rotation(
                parameters[bucket.bucket_id], initial_buckets[bucket.bucket_id]
            )
            for bucket in buckets
        }
        loss = bucketed_grouped_objective(
            keys,
            buckets,
            current_buckets,
            token_indices=token_indices,
            codebooks=training_codebooks,
        )
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            parameters.parameters(), max_norm=gradient_clip
        )
        optimizer.step()
        scheduler.step()
        latest_loss = float(loss.detach())
        latest_gradient_norm = float(gradient_norm.detach())
        if step % curve_interval == 0 or step == steps:
            with torch.no_grad():
                measured_buckets = {
                    bucket.bucket_id: cayley_rotation(
                        parameters[bucket.bucket_id],
                        initial_buckets[bucket.bucket_id],
                    )
                    for bucket in buckets
                }
                validation_mse = (
                    float(
                        bucketed_grouped_objective(
                            validation,
                            buckets,
                            measured_buckets,
                            codebooks=training_codebooks,
                        ).cpu()
                    )
                    if validation is not None
                    else None
                )
            curve_rows.append(
                {
                    "step": step,
                    "minibatch_original_scale_mse": latest_loss,
                    "full_original_scale_mse": None,
                    "validation_original_scale_mse": validation_mse,
                    "learning_rate": float(optimizer.param_groups[0]["lr"]),
                    "gradient_norm": latest_gradient_norm,
                }
            )
        if checkpoint_callback is not None and (
            step % checkpoint_interval == 0 or step == end_step or step == steps
        ):
            checkpoint_callback(
                _training_state(
                    selected=selected,
                    partition_hash=partition_hash,
                    random_hash=random_hash,
                    step=step,
                    steps=steps,
                    batch_tokens=batch_tokens,
                    learning_rate=learning_rate,
                    minimum_learning_rate=minimum_learning_rate,
                    root_seed=root_seed,
                    parameters=parameters,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    generator=generator,
                    curve_rows=curve_rows,
                    initial_objective_mse=initial_objective,
                    validation_keys_sha256=validation_hash,
                )
            )

    # Persist Cayley matrices through an FP64 solve, then store specified FP32.
    learned_buckets: dict[str, torch.Tensor] = {}
    with torch.no_grad():
        for bucket in buckets:
            learned_buckets[bucket.bucket_id] = cayley_rotation(
                parameters[bucket.bucket_id].detach().double(),
                initial_buckets[bucket.bucket_id].double(),
            ).float()
        final_objective = float(
            bucketed_grouped_objective(
                keys,
                buckets,
                learned_buckets,
                codebooks=training_codebooks,
            ).cpu()
        )
    learned_key = flatten_bucket_rotations(buckets, learned_buckets)
    random_rotations = _artifact_rotations(random_artifact)
    output_rotations = {
        key: value.detach().cpu().contiguous().clone()
        for key, value in random_rotations.items()
    }
    output_rotations.update(learned_key)
    completed = end_step == steps
    learned_artifact: dict[str, Any] = {
        "schema_version": 1,
        "kind": "learned" if completed else "learned_partial",
        "allocation": selected.name,
        "root_seed": int(root_seed),
        "partition_sha256": partition_hash,
        "num_layers": int(partition_artifact["num_layers"]),
        "num_heads": int(partition_artifact["num_heads"]),
        "head_dim": int(partition_artifact["head_dim"]),
        "rotations": output_rotations,
        "stream_seeds": dict(random_artifact.get("stream_seeds", {})),
        "training": {
            "optimizer": "Adam",
            "parameterization": "dense Cayley",
            "objective": "full original-scale grouped reconstructed-Key MSE",
            "steps": int(steps),
            "completed_step": int(end_step),
            "batch_tokens_per_head": int(batch_tokens),
            "learning_rate": float(learning_rate),
            "minimum_learning_rate": float(minimum_learning_rate),
            "scheduler": "CosineAnnealingLR",
            "scheduler_t_max": int(steps),
            "gradient_clip": float(gradient_clip),
            "initial_objective_mse": initial_objective,
            "final_objective_mse": final_objective,
            "validation_keys_sha256": validation_hash,
            "validation_checkpoint_selection": False,
            "random_artifact_sha256": random_hash,
            "step0_key_sha256": component_rotation_sha256(random_artifact, "key"),
            "fixed_value_sha256": component_rotation_sha256(random_artifact, "value"),
        },
    }
    diagnostics = validate_rotation_artifact(
        learned_artifact, selected, partition_artifact
    )
    if component_rotation_sha256(learned_artifact, "value") != component_rotation_sha256(
        random_artifact, "value"
    ):
        raise AssertionError("learned artifact changed the fixed Random Value rotations")
    final_state = _training_state(
        selected=selected,
        partition_hash=partition_hash,
        random_hash=random_hash,
        step=end_step,
        steps=steps,
        batch_tokens=batch_tokens,
        learning_rate=learning_rate,
        minimum_learning_rate=minimum_learning_rate,
        root_seed=root_seed,
        parameters=parameters,
        optimizer=optimizer,
        scheduler=scheduler,
        generator=generator,
        curve_rows=curve_rows,
        initial_objective_mse=initial_objective,
        validation_keys_sha256=validation_hash,
    )
    return {
        "artifact": learned_artifact,
        "curve_rows": curve_rows,
        "checkpoint_state": final_state,
        "completed": completed,
        "completed_step": end_step,
        "initial_objective_mse": initial_objective,
        "final_objective_mse": final_objective,
        "final_validation_original_scale_mse": (
            next(
                (
                    row["validation_original_scale_mse"]
                    for row in reversed(curve_rows)
                    if row.get("validation_original_scale_mse") is not None
                ),
                None,
            )
        ),
        "step0_key_sha256": component_rotation_sha256(random_artifact, "key"),
        "learned_key_sha256": component_rotation_sha256(learned_artifact, "key"),
        "value_sha256": component_rotation_sha256(learned_artifact, "value"),
        "orthogonality_max_abs": diagnostics["orthogonality_max_abs"],
        "last_minibatch_mse": latest_loss,
        "last_gradient_norm": latest_gradient_norm,
    }


__all__ = [
    "ALLOCATION_NAMES",
    "ALLOCATION_SPECS",
    "COMPONENTS",
    "FIXED32",
    "GroupBucket",
    "GroupDefinition",
    "KMEANS2",
    "ROOT_SEED",
    "UNIFORM2",
    "UNIFORM3",
    "AllocationSpec",
    "allocation_spec",
    "artifact_sha256",
    "audit_rotation_artifact",
    "bucket_rotation_tensors",
    "bucketed_grouped_objective",
    "build_component_partition",
    "build_group_buckets",
    "build_identity_rotation_artifact",
    "build_partition_artifact",
    "build_random_rotation_artifact",
    "channel_mean_magnitude",
    "component_rotation_sha256",
    "derive_stream_seed",
    "fixed32_partition",
    "flatten_bucket_rotations",
    "group_definitions",
    "grouped_original_scale_objective",
    "grouped_reconstruct",
    "install_projection_hooks",
    "kmeans2_partition",
    "lloyd_max_codebook",
    "orthogonality_error",
    "quantize_lloyd_max",
    "reconstruct_head",
    "reconstruct_projection",
    "rotation_key",
    "rotation_orthogonality_errors",
    "storage_accounting",
    "tensor_sha256",
    "train_learned_key_rotations",
    "validate_partition_artifact",
    "validate_rotation_artifact",
]
