"""Focused unit tests for the resumable LongBench-E runner."""

import json
from collections import Counter
from types import SimpleNamespace

import numpy as np

from experiments.spin_turboquant.longbench import (
    TASKS,
    Condition,
    analyze_full_study,
    all_conditions,
    bootstrap_ci,
    build_subset_manifest,
    middle_truncate,
    paired_task_bootstrap_ci,
    prepared_batches,
    prepare_prompt,
    score_prediction,
    theoretical_kv_bytes_per_token,
    verdict_for_bit,
    write_csv,
    write_json,
)


class FakeTokenizer:
    eos_token_id = 2

    def __call__(self, text, **_kwargs):
        return SimpleNamespace(input_ids=[ord(character) for character in text])

    def decode(self, token_ids, **_kwargs):
        return "".join(chr(token) for token in token_ids)

    def apply_chat_template(self, messages, **_kwargs):
        return [1] + [ord(character) for character in messages[0]["content"]] + [2, 3]


def test_longbench_condition_matrix_and_pair_order():
    conditions = all_conditions()
    assert len(conditions) == 22
    assert [condition.condition_id for condition in conditions[:4]] == [
        "fp16_K16_V16",
        "identity_K2_V16",
        "random_K2_V16_s17",
        "learned_K2_V16_s17",
    ]
    assert [condition.condition_id for condition in conditions[4:8]] == [
        "random_K2_V16_s29",
        "learned_K2_V16_s29",
        "random_K2_V16_s43",
        "learned_K2_V16_s43",
    ]
    assert conditions[8].condition_id == "identity_K3_V16"
    assert conditions[15].condition_id == "identity_K4_V16"


def test_subset_manifest_is_deterministic_balanced_and_identity_complete(tmp_path):
    for task in TASKS:
        rows = []
        for bucket_index, base_length in enumerate((1000, 5000, 9000)):
            for example_index in range(7):
                rows.append(
                    {
                        "_id": f"{task}-{bucket_index}-{example_index}",
                        "input": "question",
                        "context": "context",
                        "answers": ["answer"],
                        "all_classes": None,
                        "length": base_length + example_index,
                    }
                )
        (tmp_path / f"{task}_e.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in rows)
        )
    hashes = {task: f"hash-{task}" for task in TASKS}
    first = build_subset_manifest(
        tmp_path, dataset_revision="revision", data_hashes=hashes
    )
    second = build_subset_manifest(
        tmp_path, dataset_revision="revision", data_hashes=hashes
    )
    assert first == second
    assert first["sampling_seed"] == 20020305
    assert first["example_count"] == 195
    assert len({(row["task"], row["example_id"]) for row in first["examples"]}) == 195
    counts = Counter(
        (row["task"], row["length_bucket"]) for row in first["examples"]
    )
    assert set(counts.values()) == {5}


def test_middle_truncation_preserves_both_ends():
    assert middle_truncate(list(range(10)), 6) == [0, 1, 2, 7, 8, 9]
    assert middle_truncate(list(range(5)), 7) == list(range(5))


def test_prompt_uses_chat_template_and_reserves_generation_room():
    input_ids, metadata = prepare_prompt(
        FakeTokenizer(),
        "prefix:{context}:suffix:{input}",
        {"context": "abcdefghij", "input": "question", "_id": "x"},
        "qasper",
        max_context_length=16,
        max_new_tokens=4,
    )
    assert input_ids.shape == (1, 12)
    assert input_ids[0, 0].item() == 1
    assert input_ids[0, -1].item() == 3
    assert metadata["prompt_truncated"] is True
    assert metadata["chat_template_applied"] is True


def test_task_specific_batches_keep_variable_qa_single_and_group_summaries():
    tokenizer = FakeTokenizer()
    expected = []
    for task, count in (("qasper", 2), ("gov_report", 5)):
        for index in range(count):
            expected.append(
                (
                    task,
                    index,
                    {
                        "_id": f"{task}-{index}",
                        "context": "context",
                        "input": "question",
                    },
                )
            )
    batches = list(
        prepared_batches(
            expected,
            tokenizer,
            {"qasper": "{context}{input}", "gov_report": "{context}{input}"},
            {"qasper": 8, "gov_report": 8},
            max_context_length=128,
            maximum_batch_size=8,
            batch_token_budget=1024,
        )
    )
    assert [len(batch) for _, batch in batches] == [1, 1, 4, 1]


def test_theoretical_key_only_cache_accounting():
    dimensions = {"num_hidden_layers": 32, "num_key_value_heads": 8, "head_dim": 128}
    assert theoretical_kv_bytes_per_token(Condition("fp16"), dimensions) == 131_072
    assert theoretical_kv_bytes_per_token(Condition("identity", 2), dimensions) == 74_752
    assert theoretical_kv_bytes_per_token(Condition("random", 3, 17), dimensions) == 78_848
    assert theoretical_kv_bytes_per_token(Condition("learned", 4, 17), dimensions) == 82_944


def test_official_style_scoring_takes_best_reference_and_first_line():
    class Metrics:
        @staticmethod
        def qa_f1_score(prediction, answer, **_kwargs):
            return 1.0 if prediction == answer else 0.25

    row = {
        "task": "triviaqa",
        "prediction": "answer\nexplanation",
        "answers": ["wrong", "answer"],
        "all_classes": None,
    }
    assert score_prediction(Metrics, row) == 100.0


def test_bootstrap_is_paired_deterministic_and_verdict_rules_are_strict():
    arrays = [np.asarray([1.0, 2.0]), np.asarray([3.0, 4.0])]
    first = bootstrap_ci(arrays, samples=200, seed=9)
    second = bootstrap_ci(arrays, samples=200, seed=9)
    assert first == second
    assert first[0] > 0
    shared = paired_task_bootstrap_ci(
        [np.asarray([[1.0, 2.0], [3.0, 4.0]])], samples=200, seed=9
    )
    assert shared[0] > 0
    assert verdict_for_bit(0.2, [0.1, 0.2, 0.3], 0.01, False) == "Supported"
    assert (
        verdict_for_bit(0.2, [0.1, -0.1, 0.3], -0.01, False)
        == "Promising pilot"
    )
    assert verdict_for_bit(-0.1, [-0.1, 0.1, -0.3], -0.2, False) == "Not supported"


def test_complete_study_analysis_builds_required_tables_and_verdicts(
    tmp_path, monkeypatch
):
    import experiments.spin_turboquant.longbench as longbench

    output_dir = tmp_path / "study"
    for condition in all_conditions():
        run_dir = output_dir / "full" / condition.condition_id
        run_dir.mkdir(parents=True)
        write_json(run_dir / "run_config.json", {"status": "complete"})
        if condition.method == "fp16":
            score = 60.0
        elif condition.method == "identity":
            score = 30.0 + int(condition.bit_width)
        elif condition.method == "random":
            score = 40.0 + int(condition.bit_width)
        else:
            score = 41.0 + int(condition.bit_width)
        predictions = []
        scores = []
        for index, task in enumerate(TASKS):
            example_id = f"{task}-0"
            bucket = ("0-4k", "4-8k", "8k+")[index % 3]
            predictions.append(
                {
                    "condition_id": condition.condition_id,
                    "method": condition.method,
                    "bit_width": condition.bit_width,
                    "seed": condition.seed,
                    "task": task,
                    "example_index": 0,
                    "example_id": example_id,
                    "dataset_length": (1000, 5000, 9000)[index % 3],
                    "length_bucket": bucket,
                    "prediction": "answer",
                    "prompt_tokens": 10 + index,
                    "generated_tokens": 2,
                    "prefill_seconds": 0.1,
                    "decode_seconds": 0.01,
                    "total_seconds": 0.11,
                    "peak_gpu_memory_bytes": 100,
                    "theoretical_kv_bytes_per_token": 100,
                }
            )
            scores.append(
                {
                    "condition_id": condition.condition_id,
                    "task": task,
                    "example_id": example_id,
                    "score": score,
                }
            )
        (run_dir / "predictions.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in predictions)
        )
        write_csv(run_dir / "scores.csv", scores)
        write_csv(
            run_dir / "task_summary.csv",
            [{"condition_id": condition.condition_id, "task": task} for task in TASKS],
        )
        write_csv(
            run_dir / "category_summary.csv",
            [{"condition_id": condition.condition_id, "category": "test"}],
        )
        write_csv(
            run_dir / "length_summary.csv",
            [
                {
                    "condition_id": condition.condition_id,
                    "length_bucket": "0-4k",
                    "macro_average": score,
                }
            ],
        )

    assets = {
        "model": {"revision": "model-revision"},
        "longbench": {
            "commit": "longbench-commit",
            "dataset_revision": "dataset-revision",
            "data_counts": {task: 1 for task in TASKS},
            "data_hashes": {task: f"hash-{task}" for task in TASKS},
        },
    }
    monkeypatch.setattr(longbench, "validate_assets", lambda *_args, **_kwargs: assets)
    monkeypatch.setattr(longbench, "write_plots", lambda *_args, **_kwargs: None)
    manifest = {
        "sampling_seed": 20020305,
        "example_count": len(TASKS),
    }
    monkeypatch.setattr(
        longbench, "ensure_subset_manifest", lambda *_args, **_kwargs: manifest
    )
    monkeypatch.setattr(
        longbench,
        "canonical_examples",
        lambda *_args, **_kwargs: [
            (task, 0, {"_id": f"{task}-0"}) for task in TASKS
        ],
    )
    args = SimpleNamespace(
        output_dir=output_dir,
        data_dir=tmp_path / "data",
        dataset_revision="dataset-revision",
        bootstrap_samples=100,
    )
    summary = analyze_full_study(args)
    assert summary["all_conditions_complete"] is True
    assert [row["verdict"] for row in summary["overall"]] == [
        "Supported",
        "Supported",
        "Supported",
    ]
    for name in (
        "overall_summary.csv",
        "category_comparison.csv",
        "task_paired_summary.csv",
        "paired_comparison.csv",
        "length_summary.csv",
        "length_paired_summary.csv",
        "predictions.jsonl",
        "scores.csv",
        "task_summary.csv",
        "category_summary.csv",
        "system_metrics.csv",
        "system_summary.csv",
        "report.md",
        "summary.json",
    ):
        assert (output_dir / name).is_file()
