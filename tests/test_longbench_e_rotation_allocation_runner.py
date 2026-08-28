"""Integration-focused CPU tests for the 12-condition LongBench-E runner."""

from __future__ import annotations

import argparse
import json
from types import SimpleNamespace

import torch
from transformers import LlamaConfig, LlamaForCausalLM

from experiments.spin_turboquant import longbench_e_rotation_allocation as runner
from experiments.spin_turboquant import longbench_e_rotation_allocation_protocol as protocol
from experiments.spin_turboquant import rotation_allocation as core


class _Attention(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.k_proj = torch.nn.Identity()
        self.v_proj = torch.nn.Identity()


class _Layer(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.self_attn = _Attention()


class _Model(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = SimpleNamespace(layers=torch.nn.ModuleList([_Layer()]))


def _tiny_artifact(allocation: str = "uniform2") -> dict:
    statistics = torch.tensor([[[1.0, 2.0, 3.0, 4.0]]])
    partition = core.build_partition_artifact(allocation, statistics, statistics)
    identity = core.build_identity_rotation_artifact(allocation, partition)
    random_rotation = core.build_random_rotation_artifact(
        allocation, partition, root_seed=runner.ROOT_SEED
    )
    learned = {
        **random_rotation,
        "kind": "learned",
        "rotations": {
            key: value.clone() for key, value in random_rotation["rotations"].items()
        },
        "training": {
            "steps": runner.TRAINING_STEPS,
            "completed_step": runner.TRAINING_STEPS,
            "step0_key_sha256": core.component_rotation_sha256(
                random_rotation, "key"
            ),
            "fixed_value_sha256": core.component_rotation_sha256(
                random_rotation, "value"
            ),
        },
    }
    core.validate_rotation_artifact(learned, allocation, partition)
    return {
        "schema_version": 1,
        "allocation": allocation,
        "step": runner.TRAINING_STEPS,
        "protocol": {"allocation": allocation},
        "partition": partition,
        "identity_rotation": identity,
        "random_rotation": random_rotation,
        "learned_rotation": learned,
        "training": {},
    }


def _write_tiny_artifact(tmp_path, allocation: str = "uniform2") -> dict:
    artifact = _tiny_artifact(allocation)
    path = runner.condition_artifact_path(tmp_path, allocation)
    runner.atomic_torch_save(artifact, path)
    return artifact


def test_named_streams_are_stable_distinct_and_rooted_at_35():
    assert runner.named_seed("rotation") == runner.named_seed("rotation")
    assert runner.named_seed("rotation") != runner.named_seed("bootstrap")
    manifest = runner.rng_manifest(["rotation", "bootstrap"])
    assert manifest["root_seed"] == 35
    assert manifest["literal_seed_exceptions"] == {
        "subset_sampling": 35,
        "sklearn_kmeans_random_state": 35,
        "paired_bootstrap": 35,
    }


def test_child_command_pins_self_module_all_protocol_arguments(tmp_path):
    args = argparse.Namespace(
        model=tmp_path / "model",
        source_dir=tmp_path / "source",
        longbench_repo=tmp_path / "longbench",
        data_dir=tmp_path / "data",
        output_dir=tmp_path / "output",
        device="cuda:0",
        batch_size=8,
        batch_token_budget=32768,
        bootstrap_samples=10000,
        dataset_revision=runner.LONG_BENCH_DATASET_REVISION,
        longbench_commit=runner.LONG_BENCH_COMMIT,
        wikitext_revision=runner.WIKITEXT_REVISION,
        checkpoint_interval=100,
        diagnostic_tokens=256,
        max_context_length=None,
    )
    condition = protocol.condition_by_id("fixed32_random_2p5_s35")
    command = runner.child_command(args, "full", condition)
    assert command[command.index("-m") + 1] == (
        "experiments.spin_turboquant.longbench_e_rotation_allocation"
    )
    assert command[command.index("--condition") + 1] == condition.condition_id
    assert command[command.index("--bootstrap-samples") + 1] == "10000"
    assert command[command.index("--wikitext-revision") + 1] == runner.WIKITEXT_REVISION


def test_prediction_tail_recovery_preserves_only_incomplete_final_bytes(tmp_path):
    run_dir = tmp_path / "full" / "condition"
    run_dir.mkdir(parents=True)
    path = run_dir / "predictions.jsonl"
    complete = {"task": "qasper", "example_id": "one"}
    tail = b'{"task":"qasper"'
    path.write_bytes((json.dumps(complete) + "\n").encode() + tail)
    recovery = runner.recover_prediction_tail(run_dir)
    assert recovery is not None
    assert path.read_text() == json.dumps(complete) + "\n"
    preserved = recovery["preserved_tail_path"]
    assert preserved is not None
    assert runner.Path(preserved).read_bytes() == tail


def test_durable_batch_append_writes_individually_readable_rows(tmp_path):
    path = tmp_path / "predictions.jsonl"
    rows = [
        {"task": "qasper", "example_id": "one"},
        {"task": "qasper", "example_id": "two"},
    ]
    runner.append_prediction_batch_durably(path, rows)
    assert runner.base.read_predictions(path) == rows
    assert path.read_bytes().endswith(b"\n")


def test_smoke_counter_gate_requires_only_the_allocated_bit_widths():
    condition = protocol.condition_by_id("fixed32_random_2p5_s35")
    counters = {
        name: 1
        for name in (
            "key_vectors",
            "value_vectors",
            "key_prefill_vectors",
            "value_prefill_vectors",
            "key_decode_vectors",
            "value_decode_vectors",
            "quantized_coordinates",
            "key_bit2_coordinates",
            "key_bit4_coordinates",
            "value_bit2_coordinates",
            "value_bit4_coordinates",
        )
    }
    assert not runner._smoke_counter_failures(
        {"kv_codec_counters": counters}, condition
    )
    counters["key_bit3_coordinates"] = 1
    assert "unexpected_nonzero:key_bit3_coordinates" in runner._smoke_counter_failures(
        {"kv_codec_counters": counters}, condition
    )


def test_channel_statistics_are_unsigned_float64_means():
    calibration = {
        "k": torch.tensor([[[[-2.0, 1.0], [4.0, -3.0]]]], dtype=torch.bfloat16),
        "v": torch.tensor([[[[1.0, -5.0], [-3.0, 1.0]]]], dtype=torch.bfloat16),
    }
    statistics = runner._channel_statistics(calibration)
    assert statistics["k"].dtype == torch.float64
    assert torch.equal(statistics["k"], torch.tensor([[[3.0, 2.0]]], dtype=torch.float64))
    assert torch.equal(statistics["v"], torch.tensor([[[2.0, 3.0]]], dtype=torch.float64))


def test_capture_hooks_match_transformers_llama_projection_and_rope_layout():
    config = LlamaConfig(
        vocab_size=64,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=32,
    )
    model = LlamaForCausalLM(config).eval()
    input_ids = torch.arange(8, dtype=torch.long) % config.vocab_size
    captured = runner.capture_kv_activations(
        model,
        input_ids,
        device=torch.device("cpu"),
        diagnostic_tokens=4,
    )
    assert captured["k"].shape == (2, 2, 8, 4)
    assert captured["v"].shape == (2, 2, 8, 4)
    assert captured["q"].shape == (2, 4, 4, 4)
    assert captured["cos"].shape == (4, 4)
    assert captured["sin"].shape == (4, 4)
    assert torch.equal(captured["input_ids"], input_ids)


def test_runner_wrapper_selects_groupwise_kv_path_and_phase_counters(tmp_path):
    _write_tiny_artifact(tmp_path)
    condition = protocol.condition_by_id("uniform2_random_s35")
    runner.ACTIVE_ARGS = SimpleNamespace(output_dir=tmp_path)
    key_bundle, value_bundle = runner.load_condition_rotations(
        condition, tmp_path / "unused", {}, torch.device("cpu")
    )
    model = _Model()
    counters: dict[str, int] = {}
    prefill = torch.tensor(
        [[[0.1, -0.2, 0.3, -0.4], [0.5, -0.6, 0.7, -0.8]]],
        dtype=torch.bfloat16,
    )
    decode = prefill[:, :1]
    with runner.install_condition_kv_hooks(
        model,
        key_bundle,
        value_bundle,
        torch.empty(0),
        norm_correction=True,
        counters=counters,
    ):
        reconstructed_prefill_k = model.model.layers[0].self_attn.k_proj(prefill)
        reconstructed_prefill_v = model.model.layers[0].self_attn.v_proj(prefill)
        model.model.layers[0].self_attn.k_proj(decode)
        model.model.layers[0].self_attn.v_proj(decode)
    assert reconstructed_prefill_k.shape == prefill.shape
    assert reconstructed_prefill_k.dtype == prefill.dtype
    assert not torch.equal(reconstructed_prefill_k, prefill)
    assert not torch.equal(reconstructed_prefill_v, prefill)
    assert counters["key_prefill_vectors"] == 2
    assert counters["value_prefill_vectors"] == 2
    assert counters["key_decode_vectors"] == 1
    assert counters["value_decode_vectors"] == 1
    assert counters["quantized_coordinates"] > 0
    assert counters["bit2_coordinates"] > 0


def test_composite_artifact_gate_and_storage_wrapper(tmp_path):
    artifact = _write_tiny_artifact(tmp_path)
    audit = runner._audit_condition_artifact(
        artifact, orthogonality_tolerance=runner.ORTHOGONALITY_TOLERANCE
    )
    assert audit["status"] == "passed"
    assert audit["step0_key_matches_random_bitwise"] is True
    assert audit["learned_value_matches_random_bitwise"] is True
    condition = protocol.condition_by_id("uniform2_learned_s35_step10000")
    runner.ACTIVE_ARGS = SimpleNamespace(output_dir=tmp_path)
    # d=4, 2-bit indexes => one byte plus one FP16 norm for each of K and V.
    assert runner.theoretical_kv_bytes_per_token(
        condition,
        {"num_hidden_layers": 1, "num_key_value_heads": 1},
    ) == 6


def test_synthetic_codec_gate_covers_zero_batch_and_repeatability():
    gate = runner._synthetic_codec_gate(_tiny_artifact())
    assert gate["status"] == "passed"
    assert all(
        case["zero_preserved"] and case["repeat_bitwise_equal"]
        for case in gate["cases"].values()
    )


def test_plots_select_the_exact_learned_minus_random_contrast(tmp_path):
    condition_summary = [
        {
            **condition.to_dict(),
            "task_macro_average": float(index),
        }
        for index, condition in enumerate(protocol.all_conditions())
    ]
    comparisons = [
        {
            "comparison_id": contrast.comparison_id,
            "allocation": contrast.allocation,
            "contrast": contrast.contrast,
            "difference": 0.25,
        }
        for contrast in protocol.PAIRED_CONTRASTS
    ]
    storage_rows = [
        {"allocation": allocation, "kv_average_effective_bpe": 2.5}
        for allocation in protocol.ALLOCATION_ORDER
    ]
    runner._write_plots(tmp_path, condition_summary, comparisons, storage_rows)
    assert (tmp_path / "plots/condition_scores.png").is_file()
    assert (tmp_path / "plots/learned_minus_random.png").is_file()
    assert (tmp_path / "plots/score_bpe_pareto.png").is_file()


def test_tiny_artifact_stage_writes_reloadable_manifests_and_checkpoints(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(runner, "TRAINING_STEPS", 2)
    monkeypatch.setattr(runner, "TRAINING_BATCH_TOKENS", 4)
    monkeypatch.setattr(runner, "validate_assets", lambda *_args, **_kwargs: {})
    generator = torch.Generator().manual_seed(123)
    calibration = {
        "k": torch.randn(1, 1, 8, 40, generator=generator).to(torch.bfloat16),
        "v": torch.randn(1, 1, 8, 40, generator=generator).to(torch.bfloat16),
    }
    validation = {
        "k": torch.randn(1, 1, 7, 40, generator=generator).to(torch.bfloat16),
        "v": torch.randn(1, 1, 7, 40, generator=generator).to(torch.bfloat16),
    }
    runner.atomic_torch_save(calibration, tmp_path / "activations/calibration.pt")
    runner.atomic_torch_save(validation, tmp_path / "activations/validation.pt")
    args = SimpleNamespace(
        output_dir=tmp_path,
        rotation_dir=tmp_path / "rotation_artifacts",
        device="cpu",
        checkpoint_interval=1,
    )
    runner.artifacts_stage(args)
    assert (tmp_path / "fixed32_partitions.pt").is_file()
    assert (tmp_path / "kmeans2_partitions.pt").is_file()
    assert (tmp_path / "partition_manifest.json").is_file()
    assert (tmp_path / "random_rotation_manifest.json").is_file()
    assert (tmp_path / "learned_rotation_manifest.json").is_file()
    assert (tmp_path / "training_curves.csv").is_file()
    first_hashes = {
        allocation: runner.base.sha256_file(
            runner.condition_artifact_path(tmp_path, allocation)
        )
        for allocation in protocol.ALLOCATION_ORDER
    }
    # A second invocation must validate and reuse every fixed step artifact.
    runner.artifacts_stage(args)
    assert first_hashes == {
        allocation: runner.base.sha256_file(
            runner.condition_artifact_path(tmp_path, allocation)
        )
        for allocation in protocol.ALLOCATION_ORDER
    }
    for allocation in protocol.ALLOCATION_ORDER:
        artifact = torch.load(
            runner.condition_artifact_path(tmp_path, allocation),
            map_location="cpu",
            weights_only=True,
        )
        assert artifact["step"] == 2
        assert runner._audit_condition_artifact(
            artifact,
            orthogonality_tolerance=runner.ORTHOGONALITY_TOLERANCE,
        )["status"] == "passed"
        assert (tmp_path / "checkpoints" / f"{allocation}_latest.pt").is_file()
