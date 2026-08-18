"""Focused learned-K2/V2 step-10000 runner for the LongBench dataset."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import torch
from transformers import AutoTokenizer

if __package__ in (None, ""):
    _ROOT = Path(__file__).resolve().parents[2]
    if str(_ROOT) not in sys.path:
        sys.path.insert(0, str(_ROOT))

from experiments.spin_turboquant import longbench as base
from experiments.spin_turboquant.training_length_sweep import tensor_sha256


CONDITION_ID = "learned_K2_V2_s35_step10000"
EXPECTED_KEY_HASH = "824ac9f487464ed7837281b833bb7ce777e761fbbc66cdfa37cc125221af928b"
EXPECTED_VALUE_HASH = "9a46e3b1758695a4a0cf5d5c43b369ac590dcef01ae623b90c0a9143f901d0ff"
FORBIDDEN_CONDITIONS = ("fp16_K16_V16", "identity_K2_V2", "random_K2_V2_s35")
DATASET_MODE_LONGBENCH_E = "longbench_e"
DATASET_MODE_LONGBENCH = "longbench"

SPECIFICATION_PATHS = {
    DATASET_MODE_LONGBENCH_E: Path(__file__).resolve().parents[3]
    / "longbench_e_k2v2_step10000_plan.md",
    DATASET_MODE_LONGBENCH: Path(__file__).resolve().parents[3]
    / "longbench_experiment_plan.md",
}

TASKS_BY_MODE = {
    DATASET_MODE_LONGBENCH_E: (
        "qasper",
        "multifieldqa_en",
        "hotpotqa",
        "2wikimqa",
        "gov_report",
        "multi_news",
        "trec",
        "triviaqa",
        "samsum",
        "passage_count",
        "passage_retrieval_en",
        "lcc",
        "repobench-p",
    ),
    DATASET_MODE_LONGBENCH: (
        "narrativeqa",
        "qasper",
        "multifieldqa_en",
        "multifieldqa_zh",
        "hotpotqa",
        "2wikimqa",
        "musique",
        "dureader",
        "gov_report",
        "qmsum",
        "multi_news",
        "vcsum",
        "trec",
        "triviaqa",
        "samsum",
        "lsht",
        "passage_count",
        "passage_retrieval_en",
        "passage_retrieval_zh",
        "lcc",
        "repobench-p",
    ),
}

METRIC_NAMES_BY_MODE = {
    DATASET_MODE_LONGBENCH_E: {
        "qasper": "qa_f1_score",
        "multifieldqa_en": "qa_f1_score",
        "hotpotqa": "qa_f1_score",
        "2wikimqa": "qa_f1_score",
        "gov_report": "rouge_score",
        "multi_news": "rouge_score",
        "trec": "classification_score",
        "triviaqa": "qa_f1_score",
        "samsum": "rouge_score",
        "passage_count": "count_score",
        "passage_retrieval_en": "retrieval_score",
        "lcc": "code_sim_score",
        "repobench-p": "code_sim_score",
    },
    DATASET_MODE_LONGBENCH: {
        "narrativeqa": "qa_f1_score",
        "qasper": "qa_f1_score",
        "multifieldqa_en": "qa_f1_score",
        "multifieldqa_zh": "qa_f1_zh_score",
        "hotpotqa": "qa_f1_score",
        "2wikimqa": "qa_f1_score",
        "musique": "qa_f1_score",
        "dureader": "rouge_zh_score",
        "gov_report": "rouge_score",
        "qmsum": "rouge_score",
        "multi_news": "rouge_score",
        "vcsum": "rouge_zh_score",
        "trec": "classification_score",
        "triviaqa": "qa_f1_score",
        "samsum": "rouge_score",
        "lsht": "classification_score",
        "passage_count": "count_score",
        "passage_retrieval_en": "retrieval_score",
        "passage_retrieval_zh": "retrieval_zh_score",
        "lcc": "code_sim_score",
        "repobench-p": "code_sim_score",
    },
}

CATEGORIES_BY_MODE = {
    DATASET_MODE_LONGBENCH_E: {
        "Single QA": ("qasper", "multifieldqa_en"),
        "Multi QA": ("hotpotqa", "2wikimqa"),
        "Summarization": ("gov_report", "multi_news"),
        "Few-shot": ("trec", "triviaqa", "samsum"),
        "Synthetic": ("passage_count", "passage_retrieval_en"),
        "Code": ("lcc", "repobench-p"),
    },
    DATASET_MODE_LONGBENCH: {
        "Single QA": (
            "narrativeqa",
            "qasper",
            "multifieldqa_en",
            "multifieldqa_zh",
            "hotpotqa",
            "2wikimqa",
            "musique",
        ),
        "Summarization": (
            "gov_report",
            "qmsum",
            "multi_news",
            "vcsum",
        ),
        "Few-shot": ("trec", "triviaqa", "samsum", "lsht"),
        "Synthetic": (
            "passage_count",
            "passage_retrieval_en",
            "passage_retrieval_zh",
        ),
        "Code": ("lcc", "repobench-p"),
        "Retrieval": ("dureader",),
    },
}

TASK_BATCH_SIZES_BY_MODE = {
    DATASET_MODE_LONGBENCH_E: {
        "qasper": 1,
        "multifieldqa_en": 1,
        "hotpotqa": 1,
        "2wikimqa": 1,
        "gov_report": 4,
        "multi_news": 4,
        "trec": 8,
        "triviaqa": 8,
        "samsum": 8,
        "passage_count": 4,
        "passage_retrieval_en": 8,
        "lcc": 8,
        "repobench-p": 8,
    },
    DATASET_MODE_LONGBENCH: {
        "narrativeqa": 1,
        "qasper": 1,
        "multifieldqa_en": 1,
        "multifieldqa_zh": 1,
        "hotpotqa": 1,
        "2wikimqa": 1,
        "musique": 1,
        "dureader": 4,
        "gov_report": 4,
        "qmsum": 4,
        "multi_news": 4,
        "vcsum": 4,
        "trec": 8,
        "triviaqa": 8,
        "samsum": 8,
        "lsht": 8,
        "passage_count": 4,
        "passage_retrieval_en": 8,
        "passage_retrieval_zh": 8,
        "lcc": 8,
        "repobench-p": 8,
    },
}

NO_CHAT_TASKS_BY_MODE = {
    DATASET_MODE_LONGBENCH_E: {"trec", "triviaqa", "samsum", "lcc", "repobench-p"},
    DATASET_MODE_LONGBENCH: {"trec", "triviaqa", "samsum", "lsht", "lcc", "repobench-p"},
}

FIRST_LINE_TASKS_BY_MODE = {
    DATASET_MODE_LONGBENCH_E: {"trec", "triviaqa", "samsum"},
    DATASET_MODE_LONGBENCH: {"trec", "triviaqa", "samsum", "lsht"},
}

TASK_BATCH_SIZE_DEFAULT = 1
ACTIVE_DATASET_MODE = DATASET_MODE_LONGBENCH_E
ORIGINAL_PROTOCOL_PAYLOAD = base.protocol_payload


def dataset_mode_name(mode: str) -> str:
    return "LongBench-E" if mode == DATASET_MODE_LONGBENCH_E else "LongBench"


def dataset_file_suffix(mode: str) -> str:
    return "_e" if mode == DATASET_MODE_LONGBENCH_E else ""


def validate_dataset_mode(mode: str) -> str:
    if mode not in (DATASET_MODE_LONGBENCH_E, DATASET_MODE_LONGBENCH):
        raise ValueError(
            f"unknown dataset mode: {mode}. expected {DATASET_MODE_LONGBENCH_E} or {DATASET_MODE_LONGBENCH}"
        )
    return mode


def apply_dataset_mode(mode: str) -> None:
    global ACTIVE_DATASET_MODE
    mode = validate_dataset_mode(mode)
    ACTIVE_DATASET_MODE = mode
    base.TASKS = TASKS_BY_MODE[mode]
    categories = CATEGORIES_BY_MODE[mode]
    base.CATEGORIES = categories
    base.TASK_TO_CATEGORY = {
        task: category for category, tasks in categories.items() for task in tasks
    }
    base.TASK_BATCH_SIZES = {
        task: TASK_BATCH_SIZES_BY_MODE[mode].get(task, TASK_BATCH_SIZE_DEFAULT)
        for task in base.TASKS
    }
    base.NO_CHAT_TASKS = NO_CHAT_TASKS_BY_MODE[mode]
    base.METRIC_NAMES = METRIC_NAMES_BY_MODE[mode]
    base.FIRST_LINE_TASKS = FIRST_LINE_TASKS_BY_MODE[mode]


def active_specification_path() -> Path:
    return SPECIFICATION_PATHS[ACTIVE_DATASET_MODE]


def load_examples_for_mode(data_dir: Path, task: str, dataset_mode: str) -> list[dict[str, Any]]:
    path = data_dir / f"{task}{dataset_file_suffix(dataset_mode)}.jsonl"
    examples: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            required = {
                "input",
                "context",
                "answers",
                "length",
                "all_classes",
                "_id",
            }
            missing = required - value.keys()
            if missing:
                raise ValueError(f"{path}:{line_number} is missing {sorted(missing)}")
            examples.append(value)
    if not examples:
        raise ValueError(f"LongBench task is empty: {path}")
    ids = [str(example["_id"]) for example in examples]
    if len(ids) != len(set(ids)):
        raise ValueError(f"duplicate example IDs in {path}")
    return examples


def load_examples(data_dir: Path, task: str) -> list[dict[str, Any]]:
    return load_examples_for_mode(data_dir, task, ACTIVE_DATASET_MODE)


@dataclass(frozen=True)
class Condition:
    method: str = "learned"
    bit_width: int = 2
    value_bit_width: int = 2
    seed: int = 35

    @property
    def condition_id(self) -> str:
        return CONDITION_ID

    @property
    def rotation_key(self) -> str:
        return "learned"


def all_conditions() -> list[Condition]:
    return [Condition()]


def condition_by_id(condition_id: str) -> Condition:
    if condition_id != CONDITION_ID:
        raise ValueError(
            f"this focused runner accepts only {CONDITION_ID}; got {condition_id}"
        )
    return Condition()


def artifact_path(rotation_dir: Path) -> Path:
    return rotation_dir / "final_rotation_artifacts" / "cosine_b2_seed35_step10000.pt"


def build_full_manifest(
    data_dir: Path, *, dataset_revision: str, data_hashes: dict[str, str]
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for task_index, task in enumerate(base.TASKS):
        for dataset_index, example in enumerate(base.load_examples(data_dir, task)):
            rows.append(
                {
                    "manifest_index": len(rows),
                    "task": task,
                    "task_index": task_index,
                    "task_example_order": dataset_index,
                    "length_bucket": base.length_bucket(int(example["length"])),
                    "dataset_index": dataset_index,
                    "example_id": str(example["_id"]),
                    "dataset_length": int(example["length"]),
                }
            )
    return {
        "schema_version": 2,
        "sampling_seed": 35,
        "sampling_method": (
            f"none; every {dataset_mode_name(ACTIVE_DATASET_MODE)} example in canonical task/source order"
        ),
        "samples_per_length_bucket": 0,
        "samples_per_task": len(rows),
        "example_count": len(rows),
        "task_order": list(base.TASKS),
        "length_bucket_order": list(base.LENGTH_BUCKETS),
        "dataset_revision": dataset_revision,
        "data_hashes": data_hashes,
        "examples": rows,
    }


def ensure_full_manifest(
    output_dir: Path,
    data_dir: Path,
    *,
    dataset_revision: str,
    data_hashes: dict[str, str],
) -> dict[str, Any]:
    expected = build_full_manifest(
        data_dir, dataset_revision=dataset_revision, data_hashes=data_hashes
    )
    path = output_dir / "full_manifest.json"
    existing = base.read_json(path)
    if existing is None:
        base.write_json(path, expected)
    elif existing != expected:
        raise RuntimeError(f"existing full manifest differs from pinned protocol: {path}")
    return expected


def validate_assets(args: argparse.Namespace, *, deep: bool = True) -> dict[str, Any]:
    if Path(sys.prefix).name != "stq":
        raise RuntimeError(f"this experiment must run in conda environment stq, got {sys.prefix}")
    for label, path in {
        "model": args.model,
        "rotation directory": args.rotation_dir,
        "LongBench repository": args.longbench_repo,
        "LongBench data directory": args.data_dir,
        "specification": active_specification_path(),
    }.items():
        if not path.exists():
            raise FileNotFoundError(f"{label} does not exist: {path}")
    actual_commit = base.git_output(args.longbench_repo, "rev-parse", "HEAD")
    if actual_commit != args.longbench_commit:
        raise RuntimeError(f"LongBench checkout is {actual_commit}, expected {args.longbench_commit}")
    dimensions = base.model_dimensions(args.model)
    if args.model.name != "d10aef7999a2b5ba950ab3974312feeedbfe0b77":
        raise RuntimeError(f"model revision mismatch: {args.model.name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    if not tokenizer.chat_template:
        raise RuntimeError("the pinned Instruct tokenizer has no chat template")
    prompts, max_lengths = base.load_official_configs(args.longbench_repo)
    max_context = args.max_context_length or dimensions["max_position_embeddings"]
    path = artifact_path(args.rotation_dir)
    if not path.exists():
        raise FileNotFoundError(
            f"exact step-10000 artifact is missing: {path}; run export-step10000 first"
        )
    artifact = torch.load(path, map_location="cpu", weights_only=True)
    required = {
        "scheduler": "cosine",
        "seed": 35,
        "bit_width": 2,
        "selected_step": 10000,
        "rotation_tensor_sha256": EXPECTED_KEY_HASH,
    }
    for key, expected in required.items():
        if artifact.get(key) != expected:
            raise RuntimeError(f"artifact {key}={artifact.get(key)!r}, expected {expected!r}")
    key_rotation = artifact.get("learned")
    value_rotation = artifact.get("value")
    shape = (32, 8, 128, 128)
    if tuple(key_rotation.shape) != shape or tuple(value_rotation.shape) != shape:
        raise RuntimeError("step-10000 K/V rotation tensor shape mismatch")
    if tensor_sha256(key_rotation.reshape(256, 128, 128)) != EXPECTED_KEY_HASH:
        raise RuntimeError("step-10000 learned tensor content hash mismatch")
    value_hash = tensor_sha256(value_rotation.reshape(256, 128, 128))
    if value_hash != EXPECTED_VALUE_HASH:
        raise RuntimeError(
            f"seed-35 Value rotation hash mismatch: {value_hash} != {EXPECTED_VALUE_HASH}"
        )
    del artifact, key_rotation, value_rotation
    data_hashes = {
        task: base.sha256_file(args.data_dir / f"{task}{dataset_file_suffix(ACTIVE_DATASET_MODE)}.jsonl") for task in base.TASKS
    }
    codebook = base.codebook_tensor(2, 128, device="cpu").numpy()
    return {
        "model": {"path": str(args.model), "revision": args.model.name, **dimensions},
        "rotation_artifact_hashes": {"bit2_seed35": base.sha256_file(path)},
        "key_rotation_tensor_sha256": EXPECTED_KEY_HASH,
        "value_rotation_tensor_sha256": value_hash,
        "codebook_hashes": {"2": hashlib.sha256(codebook.tobytes()).hexdigest()},
        "implementation_hashes": {
            "focused_runner": base.sha256_file(Path(__file__)),
            "longbench_runner": base.sha256_file(Path(base.__file__)),
            "spin_turboquant_core": base.sha256_file(Path(base.__file__).with_name("core.py")),
        },
        "specification": {
            "path": str(active_specification_path()),
            "sha256": base.sha256_file(active_specification_path()),
        },
        "longbench": {
            "repository": str(args.longbench_repo), "commit": actual_commit,
            "dataset_revision": args.dataset_revision, "data_hashes": data_hashes,
            "data_counts": {task: len(base.load_examples(args.data_dir, task)) for task in base.TASKS},
        },
        "prompts": prompts,
        "maximum_generation_lengths": max_lengths,
        "max_context_length": max_context,
    }


def load_condition_rotations(
    _condition: Condition, rotation_dir: Path, _dimensions: dict[str, Any], device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    artifact = torch.load(artifact_path(rotation_dir), map_location="cpu", weights_only=True)
    return (
        artifact["learned"].to(device=device, dtype=torch.float32),
        artifact["value"].to(device=device, dtype=torch.float32),
    )


def theoretical_kv_bytes_per_token(_condition: Condition, dimensions: dict[str, Any]) -> int:
    vectors = int(dimensions["num_hidden_layers"]) * int(dimensions["num_key_value_heads"])
    packed_vector_bytes = int(dimensions["head_dim"]) * 2 // 8 + 4
    return 2 * vectors * packed_vector_bytes


def protocol_payload(args, assets, manifest, condition, mode):
    payload = ORIGINAL_PROTOCOL_PAYLOAD(args, assets, manifest, condition, mode)
    payload["condition"]["value_bit_width"] = 2
    payload["rotation_artifact_sha256"] = assets["rotation_artifact_hashes"]["bit2_seed35"]
    payload["key_rotation_tensor_sha256"] = assets["key_rotation_tensor_sha256"]
    payload["value_rotation_tensor_sha256"] = assets["value_rotation_tensor_sha256"]
    payload["quantization"].update(
        component="pre-RoPE keys and post-projection values",
        norm_correction=True,
        cache_storage="quality emulation stores reconstructed K/V in model dtype",
    )
    return payload


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("validate", "smoke", "full", "report", "orchestrate"), default="orchestrate")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--rotation-dir", type=Path, required=True)
    parser.add_argument("--longbench-repo", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--condition", choices=(CONDITION_ID,), default=CONDITION_ID)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-context-length", type=int)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--batch-token-budget", type=int, default=32768)
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument(
        "--dataset-mode",
        default=DATASET_MODE_LONGBENCH_E,
        choices=(DATASET_MODE_LONGBENCH_E, DATASET_MODE_LONGBENCH),
        help="longbench_e for LongBench-E, longbench for the full LongBench benchmark",
    )
    parser.add_argument("--dataset-revision", default=base.LONG_BENCH_DATASET_REVISION)
    parser.add_argument("--longbench-commit", default=base.LONG_BENCH_COMMIT)
    return parser.parse_args(argv)


def install_patches() -> None:
    base.Condition = Condition
    base.all_conditions = all_conditions
    base.condition_by_id = condition_by_id
    base.validate_assets = validate_assets
    base.ensure_subset_manifest = ensure_full_manifest
    base.load_condition_rotations = load_condition_rotations
    base.theoretical_kv_bytes_per_token = theoretical_kv_bytes_per_token
    base.protocol_payload = protocol_payload
    base.load_examples = load_examples


def _self_command_invocation() -> list[str]:
    script_or_module = []
    if __package__:
        script_or_module = ["-m", f"{__package__}.{Path(__file__).stem}"]
    else:
        script_or_module = [str(Path(__file__).resolve())]
    return [sys.executable, *script_or_module]


def child_command(args: argparse.Namespace, stage: str) -> list[str]:
    command = [* _self_command_invocation(), "--stage", stage]
    for flag in ("model", "rotation_dir", "longbench_repo", "data_dir", "output_dir", "device"):
        command.extend(["--" + flag.replace("_", "-"), str(getattr(args, flag))])
    command.extend(["--dataset-mode", getattr(args, "dataset_mode", DATASET_MODE_LONGBENCH_E)])
    command.extend(["--condition", CONDITION_ID, "--batch-size", str(args.batch_size), "--batch-token-budget", str(args.batch_token_budget)])
    if args.max_context_length is not None:
        command.extend(["--max-context-length", str(args.max_context_length)])
    return command


def expected_examples_from_manifest(output_dir: Path) -> int:
    manifest = base.read_json(output_dir / "full_manifest.json", {})
    if manifest:
        return int(manifest.get("example_count", 0))
    return 0


def _identity_issues(row: dict[str, Any], *, label: str) -> list[str]:
    condition = row.get("protocol", {}).get("condition", {})
    expected = {
        "condition_id": CONDITION_ID,
        "method": "learned",
        "key_bit_width": 2,
        "value_bit_width": 2,
        "seed": 35,
    }
    issues = [
        f"{label} condition {key}={condition.get(key)!r}, expected {value!r}"
        for key, value in expected.items()
        if condition.get(key) != value
    ]
    protocol = row.get("protocol", {})
    for key, expected_hash in (
        ("key_rotation_tensor_sha256", EXPECTED_KEY_HASH),
        ("value_rotation_tensor_sha256", EXPECTED_VALUE_HASH),
    ):
        if protocol.get(key) != expected_hash:
            issues.append(f"{label} {key}={protocol.get(key)!r}, expected {expected_hash}")
    return issues


def audit_completed_run(output_dir: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    """Audit existing outputs without invoking inference or changing result rows."""
    expected_predictions = expected_examples_from_manifest(output_dir)
    if not expected_predictions:
        expected_predictions = None
    full_dir = output_dir / "full" / CONDITION_ID
    smoke_dir = output_dir / "smoke" / CONDITION_ID
    full_config = base.read_json(full_dir / "run_config.json", {})
    smoke_config = base.read_json(smoke_dir / "run_config.json", {})
    predictions = base.read_predictions(full_dir / "predictions.jsonl")
    scores = base.read_csv(full_dir / "scores.csv")
    task_rows = base.read_csv(full_dir / "task_summary.csv")
    category_rows = base.read_csv(full_dir / "category_summary.csv")
    smoke_predictions = base.read_predictions(smoke_dir / "predictions.jsonl")
    failures: list[str] = []

    if full_config.get("status") != "complete":
        failures.append(f"full run status is {full_config.get('status')!r}, expected 'complete'")
    if smoke_config.get("status") != "complete":
        failures.append(f"smoke status is {smoke_config.get('status')!r}, expected 'complete'")
    failures.extend(_identity_issues(full_config, label="full"))
    failures.extend(_identity_issues(smoke_config, label="smoke"))

    prediction_keys = [(str(row.get("task")), str(row.get("example_id"))) for row in predictions]
    score_keys = [(str(row.get("task")), str(row.get("example_id"))) for row in scores]
    if expected_predictions is not None and len(predictions) != expected_predictions:
        failures.append(
            f"prediction count is {len(predictions)}, expected {expected_predictions}"
        )
    if expected_predictions is not None and len(set(prediction_keys)) != expected_predictions:
        failures.append(
            f"unique prediction identity count is {len(set(prediction_keys))}, expected {expected_predictions}"
        )
    if expected_predictions is not None:
        expected_scores = expected_predictions
    else:
        expected_scores = len(predictions)
    if len(scores) != expected_scores or len(prediction_keys) != len(score_keys) or set(score_keys) != set(prediction_keys):
        failures.append("score rows are not a one-to-one match for predictions")
    tasks = {task for task, _ in prediction_keys}
    if tasks != set(base.TASKS):
        failures.append(f"prediction task set differs: {sorted(tasks)}")
    if len(task_rows) != len(base.TASKS) or {row.get("task") for row in task_rows} != set(base.TASKS):
        failures.append("task_summary.csv does not contain exactly all benchmark tasks")
    if len(category_rows) != len(base.CATEGORIES) or {row.get("category") for row in category_rows} != set(base.CATEGORIES):
        failures.append(
            f"category_summary.csv does not contain exactly all {len(base.CATEGORIES)} categories"
        )

    empty_count = sum(not str(row.get("prediction", "")).strip() for row in predictions)
    error_count = sum(bool(row.get(key)) for row in predictions for key in ("error", "generation_error", "scoring_error"))
    nonfinite_count = 0
    for row in scores:
        try:
            nonfinite_count += not math.isfinite(float(row.get("score", "nan")))
        except (TypeError, ValueError):
            nonfinite_count += 1
    summary_nonfinite_count = 0
    for row, field in ((*[(row, "mean_score") for row in task_rows], *[(row, "macro_average") for row in category_rows])):
        try:
            summary_nonfinite_count += not math.isfinite(float(row.get(field, "nan")))
        except (TypeError, ValueError):
            summary_nonfinite_count += 1
    wrong_prediction_conditions = sum(row.get("condition_id") != CONDITION_ID for row in predictions)
    wrong_score_conditions = sum(row.get("condition_id") != CONDITION_ID for row in scores)
    if empty_count:
        failures.append(f"empty prediction count is {empty_count}")
    if error_count:
        failures.append(f"prediction error field count is {error_count}")
    if nonfinite_count:
        failures.append(f"non-finite score count is {nonfinite_count}")
    if summary_nonfinite_count:
        failures.append(f"non-finite task/category summary count is {summary_nonfinite_count}")
    if wrong_prediction_conditions or wrong_score_conditions:
        failures.append(
            f"wrong condition identities: predictions={wrong_prediction_conditions}, scores={wrong_score_conditions}"
        )

    smoke_keys = {(str(row.get("task")), str(row.get("example_id"))) for row in smoke_predictions}
    counters = smoke_config.get("kv_codec_counters", {})
    if len(smoke_predictions) != len(base.TASKS) or len(smoke_keys) != len(base.TASKS) or {x[0] for x in smoke_keys} != set(base.TASKS):
        failures.append("smoke predictions are not exactly one unique example for each task")
    if min(int(counters.get("key_vectors", 0)), int(counters.get("value_vectors", 0))) <= 0:
        failures.append(f"smoke K/V counters are missing or zero: {counters}")

    forbidden_paths = [
        str(parent / condition)
        for parent in (output_dir / "smoke", output_dir / "full")
        for condition in FORBIDDEN_CONDITIONS
        if (parent / condition).exists()
    ]
    if forbidden_paths:
        failures.append(f"forbidden condition directories exist: {forbidden_paths}")

    audit = {
        "status": "complete" if not failures else "incomplete",
        "condition_id": CONDITION_ID,
        "expected_predictions": expected_predictions,
        "prediction_count": len(predictions),
        "unique_prediction_count": len(set(prediction_keys)),
        "score_count": len(scores),
        "task_count": len({row.get('task') for row in task_rows}),
        "category_count": len({row.get('category') for row in category_rows}),
        "empty_outputs": empty_count,
        "error_fields": error_count,
        "nonfinite_scores": nonfinite_count,
        "nonfinite_summaries": summary_nonfinite_count,
        "wrong_condition_identity_rows": wrong_prediction_conditions + wrong_score_conditions,
        "key_rotation_tensor_sha256": full_config.get("protocol", {}).get("key_rotation_tensor_sha256"),
        "value_rotation_tensor_sha256": full_config.get("protocol", {}).get("value_rotation_tensor_sha256"),
        "smoke_prediction_count": len(smoke_predictions),
        "smoke_kv_codec_counters": counters,
        "forbidden_condition_directories": forbidden_paths,
        "failures": failures,
    }
    return audit, {"full_config": full_config, "smoke_config": smoke_config, "predictions": predictions, "scores": scores, "task_rows": task_rows, "category_rows": category_rows}


def _write_singleton_plots(output_dir: Path, overall: float, category_rows: Sequence[dict[str, Any]]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plots = output_dir / "plots"
    plots.mkdir(parents=True, exist_ok=True)
    fig, axis = plt.subplots(figsize=(5, 4))
    axis.bar([CONDITION_ID], [overall])
    axis.set_ylabel(f"{dataset_mode_name(ACTIVE_DATASET_MODE)} task macro score")
    axis.tick_params(axis="x", rotation=15)
    fig.tight_layout()
    fig.savefig(plots / "overall_comparison.png", dpi=180)
    plt.close(fig)
    fig, axis = plt.subplots(figsize=(8, 4))
    axis.bar([str(row["category"]) for row in category_rows], [float(row["macro_average"]) for row in category_rows])
    axis.set_ylabel("Category macro score")
    axis.tick_params(axis="x", rotation=25)
    fig.tight_layout()
    fig.savefig(plots / "category_comparison.png", dpi=180)
    plt.close(fig)
    fig, axis = plt.subplots(figsize=(8, 3))
    axis.axis("off")
    axis.text(0.5, 0.5, "Paired comparison unavailable\nRandom condition was forbidden by explicit override.", ha="center", va="center")
    fig.tight_layout()
    fig.savefig(plots / "task_paired_difference.png", dpi=180)
    plt.close(fig)


def write_completion_report(args: argparse.Namespace) -> dict[str, Any]:
    audit, rows = audit_completed_run(args.output_dir)
    base.write_json(args.output_dir / "completion_audit.json", audit)
    if audit["status"] != "complete":
        raise RuntimeError("completion audit failed: " + "; ".join(audit["failures"]))

    config = rows["full_config"]
    protocol = config["protocol"]
    artifact = artifact_path(args.rotation_dir)
    artifact_payload = torch.load(artifact, map_location="cpu", weights_only=True)
    actual_key_hash = tensor_sha256(artifact_payload["learned"].reshape(256, 128, 128))
    actual_value_hash = tensor_sha256(artifact_payload["value"].reshape(256, 128, 128))
    artifact_file_hash = base.sha256_file(artifact)
    artifact_metadata_ok = all((
        artifact_payload.get("scheduler") == "cosine",
        artifact_payload.get("seed") == 35,
        artifact_payload.get("bit_width") == 2,
        artifact_payload.get("selected_step") == 10000,
        artifact_file_hash == protocol.get("rotation_artifact_sha256"),
    ))
    if (actual_key_hash, actual_value_hash) != (EXPECTED_KEY_HASH, EXPECTED_VALUE_HASH) or not artifact_metadata_ok:
        audit["status"] = "incomplete"
        audit["failures"].append("rotation artifact metadata, file hash, or tensor hashes differ from the pinned protocol")
        base.write_json(args.output_dir / "completion_audit.json", audit)
        raise RuntimeError(audit["failures"][-1])

    study_config = {
        "status": "complete", "scope": "learned-only explicit override",
        "condition_ids": [CONDITION_ID], "forbidden_condition_ids": list(FORBIDDEN_CONDITIONS),
        "expected_examples": len(rows["predictions"]), "model": protocol["model"],
        "longbench": protocol["longbench"], "decoding": protocol["decoding"],
        "quantization": protocol["quantization"], "specification": protocol["specification"],
    }
    rotation_manifest = {
        "condition_id": CONDITION_ID, "artifact_path": str(artifact),
        "artifact_file_sha256": artifact_file_hash, "scheduler": artifact_payload.get("scheduler"),
        "seed": artifact_payload.get("seed"), "selected_step": artifact_payload.get("selected_step"),
        "key_rotation_tensor_sha256": actual_key_hash, "value_rotation_tensor_sha256": actual_value_hash,
    }
    base.write_json(args.output_dir / "study_config.json", study_config)
    base.write_json(args.output_dir / "rotation_manifest.json", rotation_manifest)

    task_values = [float(row["mean_score"]) for row in rows["task_rows"]]
    overall = statistics.fmean(task_values)
    base.write_csv(args.output_dir / "overall_summary.csv", [{
        "condition_id": CONDITION_ID, "method": "learned", "key_bit_width": 2,
        "value_bit_width": 2, "seed": 35, "examples": len(rows["predictions"]),
        "tasks": len(base.TASKS), "categories": len(base.CATEGORIES), "task_macro_average": overall,
        "paired_comparison_status": "unavailable_random_forbidden",
    }])
    predictions = rows["predictions"]
    generated = sum(int(row["generated_tokens"]) for row in predictions)
    decode = sum(float(row["decode_seconds"]) for row in predictions)
    base.write_csv(args.output_dir / "system_metrics.csv", [{
        "condition_id": CONDITION_ID, "examples": len(predictions),
        "prompt_tokens": sum(int(row["prompt_tokens"]) for row in predictions),
        "generated_tokens": generated, "prefill_seconds": sum(float(row["prefill_seconds"]) for row in predictions),
        "decode_seconds": decode, "decode_seconds_per_token": decode / max(generated - len(predictions), 1),
        "total_seconds": sum(float(row["total_seconds"]) for row in predictions),
        "peak_gpu_memory_bytes": max(int(row["peak_gpu_memory_bytes"]) for row in predictions),
        "theoretical_kv_bytes_per_token": int(predictions[0]["theoretical_kv_bytes_per_token"]),
        "actual_cache_dtype": predictions[0]["actual_cache_dtype"], "empty_outputs": 0,
    }])
    base.write_csv(args.output_dir / "paired_comparison.csv", [{
        "status": "unavailable", "reason": "random_K2_V2_s35 was forbidden by explicit override",
        "learned_condition_id": CONDITION_ID, "reference_condition_id": "random_K2_V2_s35",
        "difference": "", "confidence_interval_low": "", "confidence_interval_high": "",
    }])
    _write_singleton_plots(args.output_dir, overall, rows["category_rows"])
    report = "\n".join([
        f"# {dataset_mode_name(ACTIVE_DATASET_MODE)} learned K2/V2 step-10000 report", "",
        f"- Completion status: complete",
        f"- Predictions: {len(rows['predictions'])} rows",
        f"- Coverage: {len(base.TASKS)} tasks and {len(base.CATEGORIES)} categories",
        f"- Learned task macro average: {overall:.6f}", "- Empty outputs/errors/non-finite scores: 0/0/0",
        f"- Key rotation tensor SHA-256: `{actual_key_hash}`", f"- Value rotation tensor SHA-256: `{actual_value_hash}`", "",
        "## Comparison availability", "",
        "Random was forbidden by the explicit learned-only override. Learned-minus-Random delta, confidence interval, and hypothesis verdict are unavailable and were not fabricated.", "",
        "![Overall singleton result](plots/overall_comparison.png)", "",
        "![Category singleton results](plots/category_comparison.png)", "",
        "![Paired comparison unavailable](plots/task_paired_difference.png)",
    ])
    (args.output_dir / "report.md").write_text(report + "\n")
    return audit


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    for field in ("model", "rotation_dir", "longbench_repo", "data_dir", "output_dir"):
        setattr(args, field, getattr(args, field).resolve())
    apply_dataset_mode(args.dataset_mode)
    install_patches()
    if args.stage == "validate":
        assets = validate_assets(args)
        manifest = ensure_full_manifest(args.output_dir, args.data_dir, dataset_revision=args.dataset_revision, data_hashes=assets["longbench"]["data_hashes"])
        print(json.dumps({"condition": CONDITION_ID, "examples": manifest["example_count"], "key_hash": EXPECTED_KEY_HASH, "value_hash": assets["value_rotation_tensor_sha256"]}, indent=2))
    elif args.stage in {"smoke", "full"}:
        base.run_inference(args, args.stage)
    elif args.stage == "report":
        print(json.dumps(write_completion_report(args), indent=2))
    else:
        validate_assets(args)
        for stage in ("smoke", "full", "report"):
            subprocess.run(child_command(args, stage), check=True, cwd=Path.cwd())


if __name__ == "__main__":
    main()
