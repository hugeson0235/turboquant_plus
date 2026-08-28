"""Focused CPU tests for the allocation-study protocol layer."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from dataclasses import FrozenInstanceError

import pytest

from experiments.spin_turboquant import longbench_e_rotation_allocation_protocol as protocol


EXPECTED_IDS = (
    "uniform2_identity",
    "uniform2_random_s35",
    "uniform2_learned_s35_step10000",
    "fixed32_identity_2p5",
    "fixed32_random_2p5_s35",
    "fixed32_learned_2p5_s35_step10000",
    "kmeans2_identity_mixed",
    "kmeans2_random_mixed_s35",
    "kmeans2_learned_mixed_s35_step10000",
    "uniform3_identity",
    "uniform3_random_s35",
    "uniform3_learned_s35_step10000",
)


def _write_pinned_data(root):
    lengths = {"0-4k": 1000, "4-8k": 5000, "8k+": 9000}
    for task in protocol.TASKS:
        rows = []
        for bucket in protocol.LENGTH_BUCKETS:
            count = protocol.EXPECTED_ORIGINAL_COUNTS[task][bucket]
            for index in range(count):
                rows.append(
                    {
                        "_id": f"{task}-{bucket}-{index}",
                        "input": "question",
                        "context": "context",
                        "answers": ["answer"],
                        "all_classes": None,
                        "length": lengths[bucket] + index,
                    }
                )
        (root / f"{task}_e.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
        )


def _score_rows():
    rows = []
    for condition in protocol.CONDITIONS:
        method_offset = {"identity": 0.0, "random": 1.0, "learned": 3.0}[
            condition.method
        ]
        manifest_index = 0
        for task_index, task in enumerate(protocol.TASKS):
            for bucket_index, bucket in enumerate(protocol.LENGTH_BUCKETS):
                rows.append(
                    {
                        "condition_id": condition.condition_id,
                        "task": task,
                        "category": protocol.TASK_TO_CATEGORY[task],
                        "example_id": f"{task}-{bucket}",
                        "manifest_index": manifest_index,
                        "dataset_length": (1000, 5000, 9000)[bucket_index],
                        "length_bucket": bucket,
                        "score": task_index + bucket_index + method_offset,
                    }
                )
                manifest_index += 1
    return rows


def test_condition_table_is_exact_ordered_and_immutable():
    assert protocol.CONDITION_IDS == EXPECTED_IDS
    assert tuple(value.condition_id for value in protocol.all_conditions()) == EXPECTED_IDS
    assert protocol.ALLOCATION_ORDER == (
        "uniform2",
        "fixed32",
        "kmeans2",
        "uniform3",
    )
    assert len(protocol.PAIRED_CONTRASTS) == 12
    assert [row.contrast for row in protocol.PAIRED_CONTRASTS[:3]] == [
        "random_minus_identity",
        "learned_minus_random",
        "learned_minus_identity",
    ]
    for allocation in protocol.ALLOCATION_ORDER:
        identity, random_condition, learned = protocol.conditions_for_allocation(
            allocation
        )
        assert [identity.method, random_condition.method, learned.method] == [
            "identity",
            "random",
            "learned",
        ]
        assert learned.value_rotation == random_condition.value_rotation == "random"
        assert learned.seed == random_condition.seed == 35
        assert learned.learned_steps == 10_000
        assert identity.bit_width == identity.regular_bits
        assert identity.value_bit_width == identity.regular_bits
        assert identity.rotation_key is None
        assert random_condition.rotation_key == "random"
        assert learned.rotation_key == "learned"
        assert json.loads(json.dumps(learned.to_dict())) == learned.to_dict()
    assert protocol.condition_by_id("fixed32_identity_2p5").nominal_index_bpe == 2.5
    assert protocol.condition_by_id("kmeans2_identity_mixed").nominal_index_bpe is None
    with pytest.raises(FrozenInstanceError):
        protocol.CONDITIONS[0].method = "random"
    with pytest.raises(TypeError):
        protocol.CONDITIONS[0] = protocol.CONDITIONS[1]
    with pytest.raises(ValueError):
        protocol.condition_by_id("not-a-condition")


def test_manifest_is_exact_734_deterministic_and_stably_sorted(tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    _write_pinned_data(data_dir)

    first = protocol.build_subset_manifest(data_dir)
    second = protocol.build_subset_manifest(data_dir)
    assert first == second
    assert first["root_seed"] == first["sampling_seed"] == 35
    assert first["example_count"] == 734
    assert first["condition_count"] == 12
    assert first["total_prediction_count"] == 8_808
    assert first["condition_ids"] == list(EXPECTED_IDS)
    assert first["manifest_sha256"] == protocol.subset_manifest_sha256(first)
    assert len(
        {(row["task"], row["example_id"]) for row in first["examples"]}
    ) == 734

    counts = Counter(
        (row["task"], row["length_bucket"]) for row in first["examples"]
    )
    by_cell = defaultdict(list)
    for row in first["examples"]:
        by_cell[(row["task"], row["length_bucket"])].append(row["dataset_index"])
    for task in protocol.TASKS:
        for bucket in protocol.LENGTH_BUCKETS:
            assert counts[(task, bucket)] == protocol.EXPECTED_SAMPLE_COUNTS[task][
                bucket
            ]
            assert by_cell[(task, bucket)] == sorted(by_cell[(task, bucket)])

    output_dir = tmp_path / "results"
    ensured = protocol.ensure_subset_manifest(output_dir, data_dir)
    assert ensured == first
    assert protocol.load_subset_manifest(output_dir / "subset_manifest.json") == first


def test_manifest_reload_rejects_corruption_and_valid_protocol_mismatch(tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    _write_pinned_data(data_dir)
    output_dir = tmp_path / "results"
    manifest = protocol.ensure_subset_manifest(output_dir, data_dir)
    path = output_dir / "subset_manifest.json"

    corrupted = json.loads(json.dumps(manifest))
    corrupted["examples"][0]["example_id"] = "changed"
    path.write_text(json.dumps(corrupted), encoding="utf-8")
    with pytest.raises(ValueError, match="hash mismatch"):
        protocol.load_subset_manifest(path)

    valid_but_different = json.loads(json.dumps(manifest))
    valid_but_different["sampling_method"] += "; incompatible change"
    valid_but_different["manifest_sha256"] = protocol.subset_manifest_sha256(
        valid_but_different
    )
    protocol.validate_subset_manifest(valid_but_different)
    path.write_text(json.dumps(valid_but_different), encoding="utf-8")
    with pytest.raises(RuntimeError, match="differs from the pinned protocol"):
        protocol.ensure_subset_manifest(output_dir, data_dir)


def test_manifest_requires_the_pinned_source_count_table(tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    _write_pinned_data(data_dir)
    path = data_dir / "qasper_e.jsonl"
    lines = path.read_text(encoding="utf-8").splitlines()
    path.write_text("\n".join(lines[:-1]) + "\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="pinned dataset requires"):
        protocol.build_subset_manifest(data_dir)


def test_paired_analysis_builds_all_contrasts_with_exact_default_bootstrap():
    rows = _score_rows()
    examples_per_condition = len(protocol.TASKS) * len(protocol.LENGTH_BUCKETS)
    analysis = protocol.analyze_paired_comparisons(
        rows, expected_examples=examples_per_condition
    )
    assert len(analysis.summary_rows) == 12
    assert len(analysis.example_rows) == 12 * examples_per_condition
    expected_differences = {
        "random_minus_identity": 1.0,
        "learned_minus_random": 2.0,
        "learned_minus_identity": 3.0,
    }
    for summary in analysis.summary_rows:
        expected = expected_differences[summary["contrast"]]
        assert summary["difference"] == pytest.approx(expected)
        assert summary["confidence_interval_low"] == pytest.approx(expected)
        assert summary["confidence_interval_high"] == pytest.approx(expected)
        assert summary["wins"] == examples_per_condition
        assert summary["ties"] == summary["losses"] == 0
        assert summary["bootstrap_samples"] == 10_000
        assert summary["bootstrap_seed"] == 35
    assert {
        row["comparison_id"] for row in analysis.example_rows
    } == {contrast.comparison_id for contrast in protocol.PAIRED_CONTRASTS}


def test_pairing_rejects_reordered_or_missing_condition_identities():
    rows = _score_rows()
    examples_per_condition = len(protocol.TASKS) * len(protocol.LENGTH_BUCKETS)
    second_start = examples_per_condition
    reordered = list(rows)
    reordered[second_start], reordered[second_start + 1] = (
        reordered[second_start + 1],
        reordered[second_start],
    )
    with pytest.raises(ValueError, match="ordered paired identities differ"):
        protocol.validate_exact_paired_identities(
            reordered, expected_examples=examples_per_condition
        )

    missing = list(rows)
    missing.pop(second_start)
    with pytest.raises(ValueError, match="ordered paired identities differ"):
        protocol.validate_exact_paired_identities(
            missing, expected_examples=examples_per_condition
        )


def test_task_stratified_bootstrap_is_deterministic_and_requires_all_tasks():
    differences = {task: [1.0, 1.0, 1.0] for task in protocol.TASKS}
    first = protocol.task_stratified_paired_bootstrap(differences)
    second = protocol.task_stratified_paired_bootstrap(differences)
    assert first == second == (1.0, 1.0)
    differences.pop(protocol.TASKS[-1])
    with pytest.raises(ValueError, match="exactly all 13 tasks"):
        protocol.task_stratified_paired_bootstrap(differences)


def test_aggregation_reports_task_category_equal_category_and_length_macros():
    all_rows = _score_rows()
    condition_id = "uniform2_identity"
    rows = [row for row in all_rows if row["condition_id"] == condition_id]
    result = protocol.aggregate_scores(rows)
    assert len(result.task_rows) == 13
    assert len(result.category_rows) == 6
    assert len(result.length_rows) == 3
    assert len(result.overall_rows) == 1

    task_lookup = {row["task"]: row for row in result.task_rows}
    assert task_lookup[protocol.TASKS[0]]["mean_score"] == pytest.approx(1.0)
    assert task_lookup[protocol.TASKS[-1]]["mean_score"] == pytest.approx(13.0)

    category_lookup = {row["category"]: row for row in result.category_rows}
    assert category_lookup["Single QA"]["macro_average"] == pytest.approx(1.5)
    assert category_lookup["Few-shot"]["macro_average"] == pytest.approx(8.0)

    overall = result.overall_rows[0]
    assert overall["task_macro_average"] == pytest.approx(7.0)
    expected_equal_category = (1.5 + 3.5 + 5.5 + 8.0 + 10.5 + 12.5) / 6
    assert overall["equal_category_macro_average"] == pytest.approx(
        expected_equal_category
    )
    assert overall["equal_category_macro_average"] != pytest.approx(
        overall["task_macro_average"]
    )

    length_lookup = {row["length_bucket"]: row for row in result.length_rows}
    assert length_lookup["0-4k"]["macro_average"] == pytest.approx(6.0)
    assert length_lookup["4-8k"]["macro_average"] == pytest.approx(7.0)
    assert length_lookup["8k+"]["macro_average"] == pytest.approx(8.0)


def test_aggregation_rejects_nonfinite_and_incomplete_length_coverage():
    rows = [
        row
        for row in _score_rows()
        if row["condition_id"] == "uniform2_identity"
    ]
    nonfinite = [dict(row) for row in rows]
    nonfinite[0]["score"] = float("nan")
    with pytest.raises(ValueError, match="non-finite"):
        protocol.aggregate_scores(nonfinite)

    incomplete = rows[:-1]
    with pytest.raises(ValueError, match="has no"):
        protocol.aggregate_length_scores(incomplete)
