"""CPU tests for the standalone channel-allocation/rotation core."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from experiments.spin_turboquant.rotation_allocation import (
    ALLOCATION_NAMES,
    AllocationSpec,
    artifact_sha256,
    audit_rotation_artifact,
    build_identity_rotation_artifact,
    build_partition_artifact,
    build_random_rotation_artifact,
    component_rotation_sha256,
    fixed32_partition,
    group_definitions,
    grouped_original_scale_objective,
    grouped_reconstruct,
    install_projection_hooks,
    kmeans2_partition,
    orthogonality_error,
    quantize_lloyd_max,
    reconstruct_head,
    rotation_key,
    storage_accounting,
    tensor_sha256,
    train_learned_key_rotations,
    validate_partition_artifact,
    validate_rotation_artifact,
)


def _statistics(
    head_dim: int, *, layers: int = 1, heads: int = 1
) -> tuple[torch.Tensor, torch.Tensor]:
    base = torch.linspace(0.05, 2.0, head_dim)
    key = torch.stack(
        [base + 0.01 * index for index in range(layers * heads)]
    ).reshape(layers, heads, head_dim)
    value = torch.flip(key, dims=(-1,))
    return key, value


def _partition(name: str, head_dim: int, *, layers: int = 1, heads: int = 1):
    key, value = _statistics(head_dim, layers=layers, heads=heads)
    if name == "kmeans2":
        # Force an unambiguous low/high two-cluster problem in every head.
        split = head_dim // 2
        low = torch.linspace(0.01, 0.1, split)
        high = torch.linspace(3.0, 4.0, head_dim - split)
        row = torch.cat((low, high))
        key = row.expand(layers, heads, -1).clone()
        value = torch.flip(key, dims=(-1,))
    return build_partition_artifact(name, key, value)


def test_allocation_models_and_deterministic_fixed_and_kmeans_partitions():
    assert tuple(AllocationSpec(name).name for name in ALLOCATION_NAMES) == ALLOCATION_NAMES
    assert AllocationSpec("uniform2").regular_bits == 2
    assert AllocationSpec("uniform3").regular_bits == 3
    assert AllocationSpec("fixed32").fixed_outlier_channels == 32
    assert AllocationSpec("kmeans2").outlier_bits == 4
    with pytest.raises(ValueError, match="unknown allocation"):
        AllocationSpec("other")

    values = torch.arange(40, dtype=torch.float32).reshape(1, 1, 40)
    fixed = fixed32_partition(values)
    assert fixed.dtype == torch.bool
    assert fixed.sum().item() == 32
    assert torch.equal(torch.nonzero(fixed[0, 0]).flatten(), torch.arange(8, 40))
    ties = fixed32_partition(torch.ones(1, 1, 40))
    assert torch.equal(torch.nonzero(ties[0, 0]).flatten(), torch.arange(32))

    clustered = torch.tensor(
        [[[0.01, 0.02, 0.05, 0.1, 8.0, 9.0, 10.0, 11.0]]]
    )
    first = kmeans2_partition(clustered)
    second = kmeans2_partition(clustered)
    assert torch.equal(first, second)
    assert torch.equal(torch.nonzero(first[0, 0]).flatten(), torch.arange(4, 8))

    artifact = _partition("fixed32", 40, layers=2, heads=2)
    diagnostics = validate_partition_artifact(artifact, "fixed32")
    assert diagnostics["key"]["outlier_min"] == 32
    assert diagnostics["key"]["outlier_max"] == 32
    assert not torch.equal(
        artifact["outlier_masks"]["key"], artifact["outlier_masks"]["value"]
    )


def test_grouped_codec_matches_manual_group_math_and_is_zero_safe():
    partition = _partition("fixed32", 40)
    artifact = build_identity_rotation_artifact("fixed32", partition)
    mask = partition["outlier_masks"]["key"][0, 0]
    groups = group_definitions("fixed32", mask)
    rotations = {
        group.name: artifact["rotations"][rotation_key("key", 0, 0, group.name)]
        for group in groups
    }
    generator = torch.Generator().manual_seed(7)
    vectors = torch.randn((3, 4, 40), generator=generator, dtype=torch.float32)
    vectors[0] = 0
    regular = next(group for group in groups if group.name == "regular")
    vectors[1, :, regular.indices] = 0

    source = vectors.bfloat16()
    actual = grouped_reconstruct(source, "fixed32", mask, rotations)
    manual_work = source.float()
    expected = torch.zeros_like(manual_work)
    for group in groups:
        selected = manual_work.index_select(-1, group.indices)
        norms = torch.linalg.vector_norm(selected, dim=-1, keepdim=True)
        stored_norms = norms.to(torch.float16).to(torch.float32)
        normalized = selected / norms.clamp_min(torch.finfo(torch.float32).eps)
        quantized = quantize_lloyd_max(
            normalized, group.bit_width, group.dimension
        )
        restored = quantized * stored_norms
        restored = torch.where(norms > 0, restored, torch.zeros_like(restored))
        expected.index_copy_(-1, group.indices, restored)

    assert actual.shape == vectors.shape
    assert actual.dtype == torch.bfloat16
    assert actual.device == vectors.device
    torch.testing.assert_close(actual.float(), expected.bfloat16().float())
    assert torch.count_nonzero(actual[0]) == 0
    assert torch.count_nonzero(actual[1, :, regular.indices]) == 0
    assert torch.isfinite(actual.float()).all()


def test_random_artifacts_are_independent_reproducible_orthogonal_and_serializable(
    tmp_path,
):
    partition = _partition("uniform2", 8, layers=2, heads=2)
    first = build_random_rotation_artifact("uniform2", partition)
    second = build_random_rotation_artifact("uniform2", partition)
    assert artifact_sha256(first) == artifact_sha256(second)
    assert len(set(first["stream_seeds"].values())) == 8
    assert not torch.equal(
        first["rotations"][rotation_key("key", 0, 0, "all")],
        first["rotations"][rotation_key("value", 0, 0, "all")],
    )
    validation = validate_rotation_artifact(first, "uniform2", partition)
    assert validation["matrix_count"] == 8
    assert validation["orthogonality_max_abs"] < 2e-5
    assert all(
        orthogonality_error(matrix) < 2e-5
        for matrix in first["rotations"].values()
    )

    path = tmp_path / "rotations.pt"
    torch.save(first, path)
    loaded = torch.load(path, map_location="cpu", weights_only=True)
    assert artifact_sha256(loaded) == artifact_sha256(first)
    value = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    assert tensor_sha256(value) == tensor_sha256(value.clone())
    assert tensor_sha256(value) != tensor_sha256(value.reshape(4, 3))
    assert tensor_sha256(value) != tensor_sha256(value.double())


class _Attention(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.k_proj = torch.nn.Identity()
        self.v_proj = torch.nn.Identity()


class _Layer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn = _Attention()


class _Model(torch.nn.Module):
    def __init__(self, layers: int):
        super().__init__()
        self.model = SimpleNamespace(
            layers=torch.nn.ModuleList([_Layer() for _ in range(layers)])
        )


def test_runtime_hooks_preserve_contract_and_split_prefill_decode_counters():
    partition = _partition("uniform2", 4, layers=1, heads=2)
    artifact = build_random_rotation_artifact("uniform2", partition)
    model = _Model(1)
    generator = torch.Generator().manual_seed(11)
    prefill = torch.randn((2, 3, 8), generator=generator).bfloat16()
    decode = torch.randn((2, 1, 8), generator=generator).bfloat16()
    counters = {}
    with install_projection_hooks(
        model, "uniform2", partition, artifact, counters=counters
    ) as returned:
        key_prefill = model.model.layers[0].self_attn.k_proj(prefill)
        value_prefill = model.model.layers[0].self_attn.v_proj(prefill)
        key_decode = model.model.layers[0].self_attn.k_proj(decode)
        value_decode = model.model.layers[0].self_attn.v_proj(decode)
        assert returned is counters

    for source, result in (
        (prefill, key_prefill),
        (prefill, value_prefill),
        (decode, key_decode),
        (decode, value_decode),
    ):
        assert result.shape == source.shape
        assert result.dtype == source.dtype
        assert result.device == source.device
        assert torch.isfinite(result.float()).all()
        assert not torch.equal(result, source)
    expected = {
        "key_vectors": 16,
        "key_prefill_vectors": 12,
        "key_decode_vectors": 4,
        "value_vectors": 16,
        "value_prefill_vectors": 12,
        "value_decode_vectors": 4,
        "quantized_coordinates": 128,
        "bit2_coordinates": 128,
        "bit3_coordinates": 0,
        "bit4_coordinates": 0,
        "group_calls": 8,
        "group_vectors": 32,
        "key_group_calls": 4,
        "key_group_vectors": 16,
        "value_group_calls": 4,
        "value_group_vectors": 16,
        "key_bit2_coordinates": 64,
        "value_bit2_coordinates": 64,
    }
    for key, value in expected.items():
        assert counters[key] == value
    assert torch.equal(model.model.layers[0].self_attn.k_proj(prefill), prefill)


def test_mixed_precision_hooks_count_actual_group_coordinates_by_bit():
    partition = _partition("fixed32", 40)
    artifact = build_identity_rotation_artifact("fixed32", partition)
    model = _Model(1)
    source = torch.randn((1, 2, 40), generator=torch.Generator().manual_seed(13))
    with install_projection_hooks(model, "fixed32", partition, artifact) as counters:
        model.model.layers[0].self_attn.k_proj(source)
        model.model.layers[0].self_attn.v_proj(source)
    assert counters["quantized_coordinates"] == 160
    assert counters["bit2_coordinates"] == 32
    assert counters["bit3_coordinates"] == 0
    assert counters["bit4_coordinates"] == 128
    assert counters["group_calls"] == 4
    assert counters["group_vectors"] == 8


def test_storage_accounting_is_exact_for_uniform_fixed_alignment_and_rotations():
    uniform = _partition("uniform2", 128)
    uniform_rotations = build_identity_rotation_artifact("uniform2", uniform)
    uniform_rows = storage_accounting(
        "uniform2", uniform, rotation_artifact=uniform_rotations
    )
    assert uniform_rows["components"]["key"]["index_bpe"] == 2.0
    assert uniform_rows["components"]["key"]["norm_bpe"] == 0.125
    assert uniform_rows["components"]["key"]["effective_bpe"] == 2.125
    assert uniform_rows["theoretical_kv_bytes_per_token"] == 68
    assert uniform_rows["static_rotation_bytes"] == 2 * 128 * 128 * 4

    fixed = _partition("fixed32", 128, layers=2, heads=3)
    fixed_rotations = build_identity_rotation_artifact("fixed32", fixed)
    accounting = storage_accounting(
        "fixed32", fixed, rotation_artifact=fixed_rotations
    )
    for component in ("key", "value"):
        summary = accounting["components"][component]
        assert summary["index_bpe"] == 2.5
        assert summary["norm_bpe"] == 0.25
        assert summary["effective_bpe"] == 2.75
        assert summary["packed_bytes_per_head_vector_mean"] == 44
        assert summary["theoretical_bytes_per_token"] == 2 * 3 * 44
    assert accounting["theoretical_kv_bytes_per_token"] == 528
    expected_static = 2 * 2 * 3 * (96 * 96 + 32 * 32) * 4
    assert accounting["static_rotation_bytes"] == expected_static

    aligned = storage_accounting("fixed32", fixed, byte_alignment=16)
    assert aligned["components"]["key"]["effective_bpe"] == 3.0
    assert aligned["components"]["key"]["alignment_bpe"] == 0.25
    assert aligned["components"]["key"]["packed_bytes_per_head_vector_mean"] == 48


def test_bucket_objective_equals_explicit_full_reconstruction_and_has_gradient():
    partition = _partition("kmeans2", 8)
    artifact = build_random_rotation_artifact("kmeans2", partition)
    generator = torch.Generator().manual_seed(21)
    keys = torch.randn((1, 1, 7, 8), generator=generator)
    objective = grouped_original_scale_objective(
        keys, "kmeans2", partition, artifact
    )
    reconstructed = reconstruct_head(
        keys[0, 0],
        "kmeans2",
        partition,
        artifact,
        component="key",
        layer=0,
        head=0,
    )
    explicit = torch.mean((reconstructed - keys[0, 0]).square())
    torch.testing.assert_close(objective, explicit, atol=2e-6, rtol=2e-6)

    differentiable = {
        key: value.detach().clone().requires_grad_(key.startswith("key__"))
        for key, value in artifact["rotations"].items()
    }
    loss = grouped_original_scale_objective(
        keys, "kmeans2", partition, differentiable
    )
    loss.backward()
    key_gradients = [
        matrix.grad
        for key, matrix in differentiable.items()
        if key.startswith("key__")
    ]
    assert all(gradient is not None for gradient in key_gradients)
    assert sum(float(gradient.abs().sum()) for gradient in key_gradients) > 0


@pytest.mark.parametrize(
    ("name", "head_dim"),
    (("uniform2", 8), ("fixed32", 40), ("kmeans2", 8), ("uniform3", 8)),
)
def test_learned_training_supports_all_four_allocation_objectives(name, head_dim):
    partition = _partition(name, head_dim)
    random_artifact = build_random_rotation_artifact(name, partition)
    generator = torch.Generator().manual_seed(31)
    keys = torch.randn((1, 1, 10, head_dim), generator=generator)
    keys[..., 0] *= 5
    validation_keys = torch.randn((1, 1, 6, head_dim), generator=generator)
    result = train_learned_key_rotations(
        keys,
        name,
        partition,
        random_artifact,
        validation_keys=validation_keys,
        steps=2,
        batch_tokens=4,
        curve_interval=1,
        checkpoint_interval=1,
    )
    learned = result["artifact"]
    assert result["completed"] is True
    assert result["completed_step"] == 2
    assert result["step0_key_sha256"] == component_rotation_sha256(
        random_artifact, "key"
    )
    assert result["value_sha256"] == component_rotation_sha256(
        random_artifact, "value"
    )
    assert result["orthogonality_max_abs"] < 2e-5
    assert learned["training"]["optimizer"] == "Adam"
    assert learned["training"]["scheduler"] == "CosineAnnealingLR"
    assert learned["training"]["scheduler_t_max"] == 2
    assert learned["training"]["batch_tokens_per_head"] == 4
    assert learned["training"]["gradient_clip"] == 1.0
    assert result["curve_rows"][0]["step"] == 0
    assert result["curve_rows"][-1]["step"] == 2
    assert all(
        row["validation_original_scale_mse"] is not None
        for row in result["curve_rows"]
    )
    assert result["final_validation_original_scale_mse"] == result["curve_rows"][-1][
        "validation_original_scale_mse"
    ]
    assert learned["training"]["validation_checkpoint_selection"] is False
    assert learned["training"]["validation_keys_sha256"] == tensor_sha256(
        validation_keys
    )
    assert result["curve_rows"][-1]["learning_rate"] == pytest.approx(0.00025)
    validate_rotation_artifact(learned, name, partition)


def test_training_checkpoint_resume_matches_uninterrupted_result_bitwise():
    partition = _partition("uniform2", 6)
    random_artifact = build_random_rotation_artifact("uniform2", partition)
    keys = torch.randn((1, 1, 12, 6), generator=torch.Generator().manual_seed(44))
    checkpoints = []
    uninterrupted = train_learned_key_rotations(
        keys,
        "uniform2",
        partition,
        random_artifact,
        steps=4,
        batch_tokens=5,
        curve_interval=1,
        checkpoint_interval=2,
        checkpoint_callback=checkpoints.append,
    )
    partial = train_learned_key_rotations(
        keys,
        "uniform2",
        partition,
        random_artifact,
        steps=4,
        batch_tokens=5,
        curve_interval=1,
        checkpoint_interval=2,
        stop_after_step=2,
    )
    assert partial["completed"] is False
    resumed = train_learned_key_rotations(
        keys,
        "uniform2",
        partition,
        random_artifact,
        steps=4,
        batch_tokens=5,
        curve_interval=1,
        checkpoint_interval=2,
        resume_state=partial["checkpoint_state"],
    )
    assert resumed["completed"] is True
    assert component_rotation_sha256(
        uninterrupted["artifact"], "key"
    ) == component_rotation_sha256(resumed["artifact"], "key")
    assert artifact_sha256(uninterrupted["artifact"]["rotations"]) == artifact_sha256(
        resumed["artifact"]["rotations"]
    )
    assert uninterrupted["curve_rows"] == resumed["curve_rows"]
    assert [state["step"] for state in checkpoints] == [2, 4]


def test_loaded_artifact_validation_rejects_partition_or_nonorthogonal_changes():
    partition = _partition("uniform3", 8)
    artifact = build_random_rotation_artifact("uniform3", partition)
    audit = audit_rotation_artifact(
        artifact,
        "uniform3",
        partition,
        expected_kind="random",
        expected_root_seed=35,
        expected_value_sha256=component_rotation_sha256(artifact, "value"),
    )
    assert audit["status"] == "complete"
    assert audit["failures"] == []
    changed_partition = _partition("uniform3", 8)
    changed_partition["statistics_sha256"]["key"] = "different"
    with pytest.raises(ValueError, match="partition hash mismatch"):
        validate_rotation_artifact(artifact, "uniform3", changed_partition)

    broken = dict(artifact)
    broken["rotations"] = dict(artifact["rotations"])
    key = rotation_key("key", 0, 0, "all")
    broken["rotations"][key] = artifact["rotations"][key].clone()
    broken["rotations"][key][0, 0] += 0.1
    with pytest.raises(ValueError, match="orthogonality error"):
        validate_rotation_artifact(broken, "uniform3", partition)
    failed_audit = audit_rotation_artifact(broken, "uniform3", partition)
    assert failed_audit["status"] == "failed"
    assert failed_audit["failures"]
