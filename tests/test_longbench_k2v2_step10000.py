import argparse
from types import SimpleNamespace

import pytest
import torch

from experiments.spin_turboquant.core import build_random_rotation_pair, install_kv_codec_hooks
from experiments.spin_turboquant.longbench_k2v2_step10000 import (
    CONDITION_ID,
    Condition,
    EXPECTED_KEY_HASH,
    EXPECTED_VALUE_HASH,
    FORBIDDEN_CONDITIONS,
    all_conditions,
    audit_completed_run,
    child_command,
    condition_by_id,
    theoretical_kv_bytes_per_token,
)
from experiments.spin_turboquant import longbench as base


class Attention(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.k_proj = torch.nn.Identity()
        self.v_proj = torch.nn.Identity()


class Layer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn = Attention()


class FakeModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = SimpleNamespace(layers=torch.nn.ModuleList([Layer()]))


def test_focused_runner_exposes_only_exact_learned_condition():
    assert [condition.condition_id for condition in all_conditions()] == [CONDITION_ID]
    assert condition_by_id(CONDITION_ID) == Condition()
    for excluded in ("fp16_K16_V16", "identity_K2_V2", "random_K2_V2_s35"):
        with pytest.raises(ValueError):
            condition_by_id(excluded)


def test_seed35_key_and_value_rotation_streams_are_independent_and_reproducible():
    first_key, first_value = build_random_rotation_pair(1, 2, 4, 35)
    second_key, second_value = build_random_rotation_pair(1, 2, 4, 35)
    assert torch.equal(first_key, second_key)
    assert torch.equal(first_value, second_value)
    assert not torch.equal(first_key, first_value)


def test_k2v2_hook_changes_both_projections_and_preserves_contract():
    model = FakeModel()
    rotations = torch.eye(4).reshape(1, 1, 4, 4)
    centroids = torch.tensor([-0.75, -0.25, 0.25, 0.75])
    source = torch.tensor([[[0.11, -0.27, 0.63, -0.91]]], dtype=torch.bfloat16)
    counters = {}
    with install_kv_codec_hooks(
        model, rotations, rotations.clone(), centroids,
        norm_correction=True, counters=counters,
    ):
        key = model.model.layers[0].self_attn.k_proj(source)
        value = model.model.layers[0].self_attn.v_proj(source)
    assert key.shape == value.shape == source.shape
    assert key.dtype == value.dtype == source.dtype
    assert key.device == value.device == source.device
    assert not torch.equal(key, source)
    assert not torch.equal(value, source)
    assert torch.equal(key, value)
    assert counters == {"key_vectors": 1, "value_vectors": 1}


def test_k2v2_theoretical_cache_bytes_and_child_condition():
    dimensions = {"num_hidden_layers": 32, "num_key_value_heads": 8, "head_dim": 128}
    assert theoretical_kv_bytes_per_token(Condition(), dimensions) == 18_432
    args = argparse.Namespace(
        model="model", rotation_dir="rotations", longbench_repo="longbench",
        data_dir="data", output_dir="output", device="cuda", dataset_mode="longbench_e",
        batch_size=8, batch_token_budget=32768, max_context_length=None,
    )
    command = child_command(args, "smoke")
    assert command[command.index("-m") + 1] != "__main__"
    assert command[command.index("--condition") + 1] == CONDITION_ID
    assert command[command.index("--dataset-mode") + 1] == "longbench_e"
    assert not any(excluded in command for excluded in (
        "fp16_K16_V16", "identity_K2_V2", "random_K2_V2_s35"
    ))


def _write_auditable_fixture(root):
    condition = {"condition_id": CONDITION_ID, "method": "learned", "key_bit_width": 2, "value_bit_width": 2, "seed": 35}
    protocol = {"condition": condition, "key_rotation_tensor_sha256": EXPECTED_KEY_HASH, "value_rotation_tensor_sha256": EXPECTED_VALUE_HASH}
    full = root / "full" / CONDITION_ID
    smoke = root / "smoke" / CONDITION_ID
    base.write_json(full / "run_config.json", {"status": "complete", "protocol": protocol})
    base.write_json(smoke / "run_config.json", {"status": "complete", "protocol": protocol, "kv_codec_counters": {"key_vectors": 1, "value_vectors": 1}})
    predictions = []
    scores = []
    for index in range(len(base.TASKS) * 3):
        task = base.TASKS[index % len(base.TASKS)]
        row = {"condition_id": CONDITION_ID, "task": task, "example_id": str(index), "prediction": "ok"}
        predictions.append(row)
        scores.append({"condition_id": CONDITION_ID, "task": task, "example_id": str(index), "score": "1.0"})
    base.write_jsonl(full / "predictions.jsonl", predictions)
    base.write_csv(full / "scores.csv", scores)
    base.write_csv(full / "task_summary.csv", [{"task": task, "mean_score": 1.0} for task in base.TASKS])
    base.write_csv(full / "category_summary.csv", [{"category": category, "macro_average": 1.0} for category in base.CATEGORIES])
    base.write_jsonl(smoke / "predictions.jsonl", [{"task": task, "example_id": task, "prediction": "ok"} for task in base.TASKS])


def test_completion_audit_accepts_exact_learned_singleton(tmp_path):
    _write_auditable_fixture(tmp_path)
    audit, _ = audit_completed_run(tmp_path)
    assert audit["status"] == "complete"
    assert audit["unique_prediction_count"] == len(base.TASKS) * 3
    assert audit["task_count"] == len(base.TASKS)
    assert audit["category_count"] == len(base.CATEGORIES)
    assert audit["failures"] == []


def test_completion_audit_rejects_forbidden_directory_without_touching_it(tmp_path):
    _write_auditable_fixture(tmp_path)
    forbidden = tmp_path / "full" / FORBIDDEN_CONDITIONS[-1]
    forbidden.mkdir(parents=True)
    audit, _ = audit_completed_run(tmp_path)
    assert audit["status"] == "incomplete"
    assert str(forbidden) in audit["forbidden_condition_directories"]
    assert forbidden.is_dir()


def test_report_stage_is_in_child_command_and_orchestrator_sequence_is_pinned():
    args = argparse.Namespace(model="m", rotation_dir="r", longbench_repo="l", data_dir="d", output_dir="o", dataset_mode="longbench", device="cuda", batch_size=8, batch_token_budget=32768, max_context_length=None)
    command = child_command(args, "report")
    assert command[command.index("-m") + 1] != "__main__"
    assert command[command.index("--stage") + 1] == "report"
    assert command[command.index("--condition") + 1] == CONDITION_ID
    assert command[command.index("--dataset-mode") + 1] == "longbench"
