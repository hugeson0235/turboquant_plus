"""Pinned protocol helpers for the 12-condition LongBench-E allocation study.

This module intentionally contains no model or codec implementation.  It owns
the pieces that must be identical across every process in the study:

* the immutable condition and paired-contrast tables;
* deterministic construction and validation of the 734-example subset;
* exact paired-identity checks and task-stratified bootstrap comparisons; and
* task, category, and length-bucket score aggregation.

The functions are strict because accepting a nearly-compatible manifest or a
partially paired score table would invalidate the experiment's conclusions.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from experiments.spin_turboquant import longbench as base


ROOT_SEED = 35
BOOTSTRAP_SEED = 35
BOOTSTRAP_SAMPLES = 10_000
LEARNED_STEPS = 10_000
SUBSET_FRACTION = 0.20
SUBSET_EXAMPLES_PER_CONDITION = 734
CONDITION_COUNT = 12
TOTAL_PREDICTIONS = CONDITION_COUNT * SUBSET_EXAMPLES_PER_CONDITION
MANIFEST_SCHEMA_VERSION = 1

TASKS = tuple(base.TASKS)
LENGTH_BUCKETS = tuple(base.LENGTH_BUCKETS)
CATEGORIES = MappingProxyType(
    {category: tuple(tasks) for category, tasks in base.CATEGORIES.items()}
)
TASK_TO_CATEGORY = MappingProxyType(
    {
        task: category
        for category, category_tasks in CATEGORIES.items()
        for task in category_tasks
    }
)

if len(TASKS) != 13 or len(CATEGORIES) != 6:
    raise RuntimeError("the allocation protocol requires 13 tasks and 6 categories")


EXPECTED_ORIGINAL_COUNTS = MappingProxyType(
    {
        "qasper": MappingProxyType({"0-4k": 100, "4-8k": 100, "8k+": 24}),
        "multifieldqa_en": MappingProxyType(
            {"0-4k": 67, "4-8k": 70, "8k+": 13}
        ),
        "hotpotqa": MappingProxyType({"0-4k": 100, "4-8k": 100, "8k+": 100}),
        "2wikimqa": MappingProxyType({"0-4k": 100, "4-8k": 100, "8k+": 100}),
        "gov_report": MappingProxyType({"0-4k": 100, "4-8k": 100, "8k+": 100}),
        "multi_news": MappingProxyType({"0-4k": 100, "4-8k": 100, "8k+": 94}),
        "trec": MappingProxyType({"0-4k": 100, "4-8k": 100, "8k+": 100}),
        "triviaqa": MappingProxyType({"0-4k": 100, "4-8k": 100, "8k+": 100}),
        "samsum": MappingProxyType({"0-4k": 100, "4-8k": 100, "8k+": 100}),
        "passage_count": MappingProxyType(
            {"0-4k": 100, "4-8k": 100, "8k+": 100}
        ),
        "passage_retrieval_en": MappingProxyType(
            {"0-4k": 100, "4-8k": 100, "8k+": 100}
        ),
        "lcc": MappingProxyType({"0-4k": 100, "4-8k": 100, "8k+": 100}),
        "repobench-p": MappingProxyType(
            {"0-4k": 100, "4-8k": 100, "8k+": 100}
        ),
    }
)

EXPECTED_SAMPLE_COUNTS = MappingProxyType(
    {
        task: MappingProxyType(
            {
                bucket: int(round(SUBSET_FRACTION * count))
                for bucket, count in bucket_counts.items()
            }
        )
        for task, bucket_counts in EXPECTED_ORIGINAL_COUNTS.items()
    }
)

if sum(
    count
    for bucket_counts in EXPECTED_SAMPLE_COUNTS.values()
    for count in bucket_counts.values()
) != SUBSET_EXAMPLES_PER_CONDITION:
    raise RuntimeError("the pinned LongBench-E sample-count table does not total 734")


@dataclass(frozen=True, slots=True)
class Condition:
    """One immutable allocation/rotation condition."""

    condition_id: str
    allocation: str
    method: str
    key_rotation: str
    value_rotation: str
    regular_bits: int
    outlier_bits: int | None
    fixed_outlier_channels: int | None
    nominal_index_bpe: float | None
    seed: int | None
    learned_steps: int | None

    @property
    def group_count(self) -> int:
        return 1 if self.outlier_bits is None else 2

    @property
    def has_variable_index_bpe(self) -> bool:
        return self.nominal_index_bpe is None

    @property
    def bit_width(self) -> int:
        """Compatibility name used by the shared LongBench inference loop."""

        return self.regular_bits

    @property
    def value_bit_width(self) -> int:
        """Compatibility width; mixed allocations carry details in the artifact."""

        return self.regular_bits

    @property
    def rotation_key(self) -> str | None:
        """Artifact selector expected by the base runner."""

        return self.method if self.method in {"random", "learned"} else None

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-safe condition metadata, including runner aliases."""

        return {
            "condition_id": self.condition_id,
            "allocation": self.allocation,
            "method": self.method,
            "key_rotation": self.key_rotation,
            "value_rotation": self.value_rotation,
            "regular_bits": self.regular_bits,
            "outlier_bits": self.outlier_bits,
            "fixed_outlier_channels": self.fixed_outlier_channels,
            "nominal_index_bpe": self.nominal_index_bpe,
            "seed": self.seed,
            "learned_steps": self.learned_steps,
            "bit_width": self.bit_width,
            "value_bit_width": self.value_bit_width,
            "rotation_key": self.rotation_key,
        }


CONDITIONS = (
    Condition(
        "uniform2_identity",
        "uniform2",
        "identity",
        "identity",
        "identity",
        2,
        None,
        None,
        2.0,
        None,
        None,
    ),
    Condition(
        "uniform2_random_s35",
        "uniform2",
        "random",
        "random",
        "random",
        2,
        None,
        None,
        2.0,
        ROOT_SEED,
        None,
    ),
    Condition(
        "uniform2_learned_s35_step10000",
        "uniform2",
        "learned",
        "learned",
        "random",
        2,
        None,
        None,
        2.0,
        ROOT_SEED,
        LEARNED_STEPS,
    ),
    Condition(
        "fixed32_identity_2p5",
        "fixed32",
        "identity",
        "identity",
        "identity",
        2,
        4,
        32,
        2.5,
        None,
        None,
    ),
    Condition(
        "fixed32_random_2p5_s35",
        "fixed32",
        "random",
        "random",
        "random",
        2,
        4,
        32,
        2.5,
        ROOT_SEED,
        None,
    ),
    Condition(
        "fixed32_learned_2p5_s35_step10000",
        "fixed32",
        "learned",
        "learned",
        "random",
        2,
        4,
        32,
        2.5,
        ROOT_SEED,
        LEARNED_STEPS,
    ),
    Condition(
        "kmeans2_identity_mixed",
        "kmeans2",
        "identity",
        "identity",
        "identity",
        2,
        4,
        None,
        None,
        None,
        None,
    ),
    Condition(
        "kmeans2_random_mixed_s35",
        "kmeans2",
        "random",
        "random",
        "random",
        2,
        4,
        None,
        None,
        ROOT_SEED,
        None,
    ),
    Condition(
        "kmeans2_learned_mixed_s35_step10000",
        "kmeans2",
        "learned",
        "learned",
        "random",
        2,
        4,
        None,
        None,
        ROOT_SEED,
        LEARNED_STEPS,
    ),
    Condition(
        "uniform3_identity",
        "uniform3",
        "identity",
        "identity",
        "identity",
        3,
        None,
        None,
        3.0,
        None,
        None,
    ),
    Condition(
        "uniform3_random_s35",
        "uniform3",
        "random",
        "random",
        "random",
        3,
        None,
        None,
        3.0,
        ROOT_SEED,
        None,
    ),
    Condition(
        "uniform3_learned_s35_step10000",
        "uniform3",
        "learned",
        "learned",
        "random",
        3,
        None,
        None,
        3.0,
        ROOT_SEED,
        LEARNED_STEPS,
    ),
)

CONDITION_IDS = tuple(condition.condition_id for condition in CONDITIONS)
ALLOCATION_ORDER = ("uniform2", "fixed32", "kmeans2", "uniform3")

if len(CONDITIONS) != CONDITION_COUNT or len(set(CONDITION_IDS)) != CONDITION_COUNT:
    raise RuntimeError("the condition table must contain 12 unique entries")


def condition_by_id(condition_id: str) -> Condition:
    """Return the one condition with ``condition_id`` or reject the ID."""

    for condition in CONDITIONS:
        if condition.condition_id == condition_id:
            return condition
    raise ValueError(f"unknown allocation-study condition: {condition_id}")


def all_conditions() -> list[Condition]:
    """Return a mutable copy of the immutable protocol-ordered condition table."""

    return list(CONDITIONS)


def conditions_for_allocation(allocation: str) -> tuple[Condition, Condition, Condition]:
    """Return Identity, Random, Learned for one allocation in protocol order."""

    if allocation not in ALLOCATION_ORDER:
        raise ValueError(f"unknown allocation: {allocation}")
    values = tuple(
        condition for condition in CONDITIONS if condition.allocation == allocation
    )
    if len(values) != 3:
        raise AssertionError(f"allocation {allocation} does not have three conditions")
    return values  # type: ignore[return-value]


@dataclass(frozen=True, slots=True)
class PairedContrast:
    """A candidate-minus-reference paired downstream comparison."""

    comparison_id: str
    allocation: str
    contrast: str
    candidate_condition_id: str
    reference_condition_id: str


def _build_paired_contrasts() -> tuple[PairedContrast, ...]:
    result: list[PairedContrast] = []
    for allocation in ALLOCATION_ORDER:
        identity, random_condition, learned = conditions_for_allocation(allocation)
        result.extend(
            (
                PairedContrast(
                    f"{allocation}_random_minus_identity",
                    allocation,
                    "random_minus_identity",
                    random_condition.condition_id,
                    identity.condition_id,
                ),
                PairedContrast(
                    f"{allocation}_learned_minus_random",
                    allocation,
                    "learned_minus_random",
                    learned.condition_id,
                    random_condition.condition_id,
                ),
                PairedContrast(
                    f"{allocation}_learned_minus_identity",
                    allocation,
                    "learned_minus_identity",
                    learned.condition_id,
                    identity.condition_id,
                ),
            )
        )
    return tuple(result)


PAIRED_CONTRASTS = _build_paired_contrasts()


def canonical_json_sha256(payload: Any) -> str:
    """Hash a JSON-compatible value with a stable canonical encoding."""

    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _plain_count_table(
    table: Mapping[str, Mapping[str, int]],
) -> dict[str, dict[str, int]]:
    return {
        task: {bucket: int(table[task][bucket]) for bucket in LENGTH_BUCKETS}
        for task in TASKS
    }


def _data_hashes(
    data_dir: Path, supplied: Mapping[str, str] | None
) -> dict[str, str]:
    if supplied is None:
        return {
            task: base.sha256_file(data_dir / f"{task}_e.jsonl") for task in TASKS
        }
    if set(supplied) != set(TASKS):
        missing = sorted(set(TASKS) - set(supplied))
        extra = sorted(set(supplied) - set(TASKS))
        raise ValueError(f"data hashes have wrong task keys: missing={missing}, extra={extra}")
    return {task: str(supplied[task]) for task in TASKS}


def _row_identity_hash(
    *,
    dataset_revision: str,
    data_sha256: str,
    task: str,
    dataset_index: int,
    example_id: str,
    dataset_length: int,
) -> str:
    return canonical_json_sha256(
        {
            "dataset_revision": dataset_revision,
            "data_sha256": data_sha256,
            "task": task,
            "dataset_index": dataset_index,
            "example_id": example_id,
            "dataset_length": dataset_length,
        }
    )


def subset_manifest_sha256(manifest: Mapping[str, Any]) -> str:
    """Recompute the manifest hash, excluding its self-referential field."""

    payload = dict(manifest)
    payload.pop("manifest_sha256", None)
    return canonical_json_sha256(payload)


def build_subset_manifest(
    data_dir: Path,
    *,
    dataset_revision: str = base.LONG_BENCH_DATASET_REVISION,
    data_hashes: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Build the exact root-seed-35, task/bucket-stratified 20% subset."""

    data_dir = Path(data_dir)
    hashes = _data_hashes(data_dir, data_hashes)
    generator = random.Random(ROOT_SEED)
    selected_rows: list[dict[str, Any]] = []
    actual_original_counts: dict[str, dict[str, int]] = {}
    actual_sample_counts: dict[str, dict[str, int]] = {}

    for task_index, task in enumerate(TASKS):
        examples = base.load_examples(data_dir, task)
        actual_original_counts[task] = {}
        actual_sample_counts[task] = {}
        task_example_order = 0
        for bucket_index, bucket in enumerate(LENGTH_BUCKETS):
            candidates = [
                (dataset_index, example)
                for dataset_index, example in enumerate(examples)
                if base.length_bucket(int(example["length"])) == bucket
            ]
            expected_original = int(EXPECTED_ORIGINAL_COUNTS[task][bucket])
            if len(candidates) != expected_original:
                raise RuntimeError(
                    f"{task}:{bucket} has {len(candidates)} examples; "
                    f"the pinned dataset requires {expected_original}"
                )
            sample_count = int(EXPECTED_SAMPLE_COUNTS[task][bucket])
            if sample_count != round(SUBSET_FRACTION * len(candidates)):
                raise AssertionError(f"sample-count table drift for {task}:{bucket}")
            chosen = sorted(
                generator.sample(candidates, sample_count), key=lambda value: value[0]
            )
            actual_original_counts[task][bucket] = len(candidates)
            actual_sample_counts[task][bucket] = len(chosen)
            for dataset_index, example in chosen:
                example_id = str(example["_id"])
                dataset_length = int(example["length"])
                selected_rows.append(
                    {
                        "manifest_index": len(selected_rows),
                        "task": task,
                        "task_index": task_index,
                        "category": TASK_TO_CATEGORY[task],
                        "task_example_order": task_example_order,
                        "length_bucket": bucket,
                        "length_bucket_index": bucket_index,
                        "dataset_index": dataset_index,
                        "example_id": example_id,
                        "dataset_length": dataset_length,
                        "source_identity_sha256": _row_identity_hash(
                            dataset_revision=dataset_revision,
                            data_sha256=hashes[task],
                            task=task,
                            dataset_index=dataset_index,
                            example_id=example_id,
                            dataset_length=dataset_length,
                        ),
                    }
                )
                task_example_order += 1

    ordered_identities = [
        [row["task"], row["example_id"]] for row in selected_rows
    ]
    manifest: dict[str, Any] = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "root_seed": ROOT_SEED,
        "sampling_seed": ROOT_SEED,
        "sampling_fraction": SUBSET_FRACTION,
        "sampling_rng": "python.random.Random(MT19937)",
        "sampling_method": (
            "one root-seed-35 stream; random.sample without replacement in "
            "canonical task/bucket order; selected source indexes sorted per cell"
        ),
        "rounding_rule": "Python round(0.20 * N); exact counts pinned below",
        "dataset_revision": str(dataset_revision),
        "data_hashes": hashes,
        "task_order": list(TASKS),
        "category_order": list(CATEGORIES),
        "categories": {key: list(value) for key, value in CATEGORIES.items()},
        "length_bucket_order": list(LENGTH_BUCKETS),
        "length_bucket_boundaries": {
            "0-4k": {"minimum_inclusive": 0, "maximum_exclusive": 4000},
            "4-8k": {"minimum_inclusive": 4000, "maximum_exclusive": 8000},
            "8k+": {"minimum_inclusive": 8000, "maximum_exclusive": None},
        },
        "expected_original_counts": _plain_count_table(EXPECTED_ORIGINAL_COUNTS),
        "expected_sample_counts": _plain_count_table(EXPECTED_SAMPLE_COUNTS),
        "actual_original_counts": actual_original_counts,
        "actual_sample_counts": actual_sample_counts,
        "example_count": len(selected_rows),
        "condition_count": CONDITION_COUNT,
        "condition_ids": list(CONDITION_IDS),
        "total_prediction_count": TOTAL_PREDICTIONS,
        "ordered_identity_sha256": canonical_json_sha256(ordered_identities),
        "examples": selected_rows,
    }
    manifest["manifest_sha256"] = subset_manifest_sha256(manifest)
    validate_subset_manifest(manifest)
    return manifest


def validate_subset_manifest(manifest: Mapping[str, Any]) -> None:
    """Reject any manifest that is not the exact pinned 734-example protocol."""

    if int(manifest.get("schema_version", -1)) != MANIFEST_SCHEMA_VERSION:
        raise ValueError("subset manifest schema version mismatch")
    if int(manifest.get("root_seed", -1)) != ROOT_SEED or int(
        manifest.get("sampling_seed", -1)
    ) != ROOT_SEED:
        raise ValueError("subset manifest seed mismatch")
    if list(manifest.get("task_order", ())) != list(TASKS):
        raise ValueError("subset manifest task order mismatch")
    if list(manifest.get("category_order", ())) != list(CATEGORIES):
        raise ValueError("subset manifest category order mismatch")
    if list(manifest.get("length_bucket_order", ())) != list(LENGTH_BUCKETS):
        raise ValueError("subset manifest length-bucket order mismatch")
    if list(manifest.get("condition_ids", ())) != list(CONDITION_IDS):
        raise ValueError("subset manifest condition order mismatch")
    if int(manifest.get("condition_count", -1)) != CONDITION_COUNT:
        raise ValueError("subset manifest condition count mismatch")
    if int(manifest.get("total_prediction_count", -1)) != TOTAL_PREDICTIONS:
        raise ValueError("subset manifest total prediction count mismatch")
    stored_hash = str(manifest.get("manifest_sha256", ""))
    actual_hash = subset_manifest_sha256(manifest)
    if stored_hash != actual_hash:
        raise ValueError(
            f"subset manifest hash mismatch: stored={stored_hash}, actual={actual_hash}"
        )

    rows = manifest.get("examples")
    if not isinstance(rows, list) or len(rows) != SUBSET_EXAMPLES_PER_CONDITION:
        raise ValueError("subset manifest must contain exactly 734 example rows")
    if int(manifest.get("example_count", -1)) != len(rows):
        raise ValueError("subset manifest example_count mismatch")
    hashes = manifest.get("data_hashes")
    if not isinstance(hashes, Mapping) or set(hashes) != set(TASKS):
        raise ValueError("subset manifest data hashes are incomplete")

    identities: list[tuple[str, str]] = []
    counts: Counter[tuple[str, str]] = Counter()
    previous_cell = (-1, -1)
    previous_dataset_index = -1
    for manifest_index, row in enumerate(rows):
        if int(row.get("manifest_index", -1)) != manifest_index:
            raise ValueError("subset manifest indexes are not contiguous")
        task = str(row.get("task"))
        bucket = str(row.get("length_bucket"))
        if task not in TASKS or bucket not in LENGTH_BUCKETS:
            raise ValueError(f"unknown task/bucket in subset manifest: {task}/{bucket}")
        task_index = int(row.get("task_index", -1))
        bucket_index = int(row.get("length_bucket_index", -1))
        if task_index != TASKS.index(task) or bucket_index != LENGTH_BUCKETS.index(bucket):
            raise ValueError("subset manifest task/bucket index mismatch")
        cell = (task_index, bucket_index)
        dataset_index = int(row.get("dataset_index", -1))
        if cell < previous_cell:
            raise ValueError("subset manifest cells are not in canonical order")
        if cell == previous_cell and dataset_index <= previous_dataset_index:
            raise ValueError("selected source indexes are not strictly sorted per cell")
        if cell != previous_cell:
            previous_dataset_index = -1
        previous_cell = cell
        previous_dataset_index = dataset_index

        example_id = str(row.get("example_id"))
        dataset_length = int(row.get("dataset_length", -1))
        if base.length_bucket(dataset_length) != bucket:
            raise ValueError("subset manifest dataset length/bucket mismatch")
        if str(row.get("category")) != TASK_TO_CATEGORY[task]:
            raise ValueError("subset manifest category mismatch")
        expected_row_hash = _row_identity_hash(
            dataset_revision=str(manifest.get("dataset_revision")),
            data_sha256=str(hashes[task]),
            task=task,
            dataset_index=dataset_index,
            example_id=example_id,
            dataset_length=dataset_length,
        )
        if str(row.get("source_identity_sha256")) != expected_row_hash:
            raise ValueError("subset manifest source identity hash mismatch")
        identities.append((task, example_id))
        counts[(task, bucket)] += 1

    if len(set(identities)) != len(identities):
        raise ValueError("subset manifest contains duplicate task/example identities")
    for task in TASKS:
        for bucket in LENGTH_BUCKETS:
            expected = int(EXPECTED_SAMPLE_COUNTS[task][bucket])
            if counts[(task, bucket)] != expected:
                raise ValueError(
                    f"subset manifest count mismatch for {task}:{bucket}: "
                    f"{counts[(task, bucket)]} != {expected}"
                )
    ordered_hash = canonical_json_sha256([[task, value] for task, value in identities])
    if str(manifest.get("ordered_identity_sha256")) != ordered_hash:
        raise ValueError("subset manifest ordered identity hash mismatch")
    for key, expected_table in (
        ("expected_original_counts", EXPECTED_ORIGINAL_COUNTS),
        ("actual_original_counts", EXPECTED_ORIGINAL_COUNTS),
        ("expected_sample_counts", EXPECTED_SAMPLE_COUNTS),
        ("actual_sample_counts", EXPECTED_SAMPLE_COUNTS),
    ):
        if manifest.get(key) != _plain_count_table(expected_table):
            raise ValueError(f"subset manifest {key} mismatch")


def load_subset_manifest(path: Path) -> dict[str, Any]:
    """Load and fully validate a persisted subset manifest."""

    path = Path(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"subset manifest is not a JSON object: {path}")
    validate_subset_manifest(payload)
    return payload


def ensure_subset_manifest(
    output_dir: Path,
    data_dir: Path,
    *,
    dataset_revision: str = base.LONG_BENCH_DATASET_REVISION,
    data_hashes: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Create or exact-match ``subset_manifest.json`` and verify its reload."""

    output_dir = Path(output_dir)
    expected = build_subset_manifest(
        Path(data_dir), dataset_revision=dataset_revision, data_hashes=data_hashes
    )
    path = output_dir / "subset_manifest.json"
    if path.exists():
        existing = load_subset_manifest(path)
        if existing != expected:
            raise RuntimeError(
                f"existing subset manifest differs from the pinned protocol: {path}"
            )
    else:
        base.write_json(path, expected)
    reloaded = load_subset_manifest(path)
    if reloaded != expected:
        raise RuntimeError(f"subset manifest changed during persistence: {path}")
    return reloaded


def _finite_score(row: Mapping[str, Any]) -> float:
    try:
        value = float(row["score"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"invalid score row: {row}") from error
    if not math.isfinite(value):
        raise ValueError(f"non-finite score row: {row}")
    return value


def _group_condition_scores(
    score_rows: Iterable[Mapping[str, Any]],
) -> dict[str, list[Mapping[str, Any]]]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    seen: set[tuple[str, str, str]] = set()
    for row in score_rows:
        condition_id = str(row.get("condition_id"))
        if condition_id not in CONDITION_IDS:
            raise ValueError(f"unknown condition in score rows: {condition_id}")
        task = str(row.get("task"))
        if task not in TASKS:
            raise ValueError(f"unknown task in score rows: {task}")
        example_id = str(row.get("example_id"))
        key = (condition_id, task, example_id)
        if key in seen:
            raise ValueError(f"duplicate score identity: {key}")
        seen.add(key)
        _finite_score(row)
        grouped[condition_id].append(row)
    return dict(grouped)


def validate_exact_paired_identities(
    score_rows: Iterable[Mapping[str, Any]],
    *,
    expected_examples: int | None = SUBSET_EXAMPLES_PER_CONDITION,
) -> tuple[tuple[str, str], ...]:
    """Require every condition to contain the same ordered task/example IDs."""

    grouped = _group_condition_scores(score_rows)
    if set(grouped) != set(CONDITION_IDS):
        missing = [value for value in CONDITION_IDS if value not in grouped]
        extra = sorted(set(grouped) - set(CONDITION_IDS))
        raise ValueError(f"score condition coverage mismatch: missing={missing}, extra={extra}")
    canonical_rows = grouped[CONDITION_IDS[0]]
    canonical = tuple(
        (str(row["task"]), str(row["example_id"])) for row in canonical_rows
    )
    if expected_examples is not None and len(canonical) != expected_examples:
        raise ValueError(
            f"condition has {len(canonical)} examples; expected {expected_examples}"
        )
    for condition_id in CONDITION_IDS[1:]:
        rows = grouped[condition_id]
        identities = tuple(
            (str(row["task"]), str(row["example_id"])) for row in rows
        )
        if identities != canonical:
            raise ValueError(
                f"ordered paired identities differ for condition {condition_id}"
            )
        for reference, candidate in zip(canonical_rows, rows):
            for field in (
                "manifest_index",
                "example_index",
                "dataset_length",
                "length_bucket",
                "category",
            ):
                if field in reference and field in candidate and reference[field] != candidate[field]:
                    raise ValueError(
                        f"paired metadata differs for {condition_id} "
                        f"{candidate['task']}:{candidate['example_id']} field={field}"
                    )
    task_set = {task for task, _ in canonical}
    if task_set != set(TASKS):
        raise ValueError("paired score rows do not cover exactly all 13 tasks")
    return canonical


def task_stratified_paired_bootstrap(
    differences_by_task: Mapping[str, Sequence[float] | np.ndarray],
    *,
    samples: int = BOOTSTRAP_SAMPLES,
    seed: int = BOOTSTRAP_SEED,
) -> tuple[float, float]:
    """Bootstrap examples within each task, then macro-average the 13 tasks."""

    if samples < 100:
        raise ValueError("paired bootstrap requires at least 100 resamples")
    if set(differences_by_task) != set(TASKS):
        raise ValueError("paired bootstrap requires exactly all 13 tasks")
    rng = np.random.default_rng(seed)
    distribution = np.zeros(samples, dtype=np.float64)
    for task in TASKS:
        values = np.asarray(differences_by_task[task], dtype=np.float64)
        if values.ndim != 1 or values.size == 0 or not np.all(np.isfinite(values)):
            raise ValueError(f"invalid paired differences for task {task}")
        indexes = rng.integers(0, values.size, size=(samples, values.size))
        distribution += values[indexes].mean(axis=1)
    distribution /= len(TASKS)
    low, high = np.percentile(distribution, [2.5, 97.5])
    return float(low), float(high)


@dataclass(frozen=True, slots=True)
class PairedAnalysis:
    example_rows: tuple[dict[str, Any], ...]
    summary_rows: tuple[dict[str, Any], ...]


def analyze_paired_comparisons(
    score_rows: Iterable[Mapping[str, Any]],
    *,
    expected_examples: int | None = SUBSET_EXAMPLES_PER_CONDITION,
    bootstrap_samples: int = BOOTSTRAP_SAMPLES,
    bootstrap_seed: int = BOOTSTRAP_SEED,
) -> PairedAnalysis:
    """Produce all 12 exact paired contrasts and their bootstrap summaries."""

    rows = list(score_rows)
    canonical = validate_exact_paired_identities(
        rows, expected_examples=expected_examples
    )
    grouped = _group_condition_scores(rows)
    lookups = {
        condition_id: {
            (str(row["task"]), str(row["example_id"])): row
            for row in condition_rows
        }
        for condition_id, condition_rows in grouped.items()
    }
    example_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []

    for contrast in PAIRED_CONTRASTS:
        candidate_lookup = lookups[contrast.candidate_condition_id]
        reference_lookup = lookups[contrast.reference_condition_id]
        differences_by_task: dict[str, list[float]] = {task: [] for task in TASKS}
        reference_by_task: dict[str, list[float]] = {task: [] for task in TASKS}
        candidate_by_task: dict[str, list[float]] = {task: [] for task in TASKS}
        wins = ties = losses = 0
        for task, example_id in canonical:
            key = (task, example_id)
            candidate_row = candidate_lookup[key]
            reference_row = reference_lookup[key]
            candidate_score = _finite_score(candidate_row)
            reference_score = _finite_score(reference_row)
            difference = candidate_score - reference_score
            if difference > 0:
                outcome = "win"
                wins += 1
            elif difference < 0:
                outcome = "loss"
                losses += 1
            else:
                outcome = "tie"
                ties += 1
            differences_by_task[task].append(difference)
            reference_by_task[task].append(reference_score)
            candidate_by_task[task].append(candidate_score)
            example_rows.append(
                {
                    "comparison_id": contrast.comparison_id,
                    "allocation": contrast.allocation,
                    "contrast": contrast.contrast,
                    "candidate_condition_id": contrast.candidate_condition_id,
                    "reference_condition_id": contrast.reference_condition_id,
                    "task": task,
                    "category": TASK_TO_CATEGORY[task],
                    "example_id": example_id,
                    "manifest_index": reference_row.get("manifest_index", ""),
                    "dataset_length": reference_row.get("dataset_length", ""),
                    "length_bucket": reference_row.get("length_bucket", ""),
                    "candidate_score": candidate_score,
                    "reference_score": reference_score,
                    "difference": difference,
                    "outcome": outcome,
                }
            )
        reference_task_macro = statistics.fmean(
            statistics.fmean(reference_by_task[task]) for task in TASKS
        )
        candidate_task_macro = statistics.fmean(
            statistics.fmean(candidate_by_task[task]) for task in TASKS
        )
        observed_difference = statistics.fmean(
            statistics.fmean(differences_by_task[task]) for task in TASKS
        )
        low, high = task_stratified_paired_bootstrap(
            differences_by_task,
            samples=bootstrap_samples,
            seed=bootstrap_seed,
        )
        summary_rows.append(
            {
                "comparison_id": contrast.comparison_id,
                "allocation": contrast.allocation,
                "contrast": contrast.contrast,
                "candidate_condition_id": contrast.candidate_condition_id,
                "reference_condition_id": contrast.reference_condition_id,
                "examples": len(canonical),
                "tasks": len(TASKS),
                "reference_task_macro_average": reference_task_macro,
                "candidate_task_macro_average": candidate_task_macro,
                "difference": observed_difference,
                "confidence_interval_low": low,
                "confidence_interval_high": high,
                "wins": wins,
                "ties": ties,
                "losses": losses,
                "bootstrap_samples": bootstrap_samples,
                "bootstrap_seed": bootstrap_seed,
            }
        )
    return PairedAnalysis(tuple(example_rows), tuple(summary_rows))


def paired_comparison_rows(
    score_rows: Iterable[Mapping[str, Any]],
    **kwargs: Any,
) -> tuple[dict[str, Any], ...]:
    """Convenience wrapper returning per-example paired rows."""

    return analyze_paired_comparisons(score_rows, **kwargs).example_rows


def paired_comparison_summaries(
    score_rows: Iterable[Mapping[str, Any]],
    **kwargs: Any,
) -> tuple[dict[str, Any], ...]:
    """Convenience wrapper returning the 12 paired summary rows."""

    return analyze_paired_comparisons(score_rows, **kwargs).summary_rows


def _present_condition_ids(grouped: Mapping[str, Any]) -> tuple[str, ...]:
    return tuple(condition_id for condition_id in CONDITION_IDS if condition_id in grouped)


def aggregate_task_scores(
    score_rows: Iterable[Mapping[str, Any]],
) -> tuple[dict[str, Any], ...]:
    """Compute official-style example means for every condition and task."""

    grouped = _group_condition_scores(score_rows)
    result: list[dict[str, Any]] = []
    for condition_id in _present_condition_ids(grouped):
        condition = condition_by_id(condition_id)
        rows = grouped[condition_id]
        for task in TASKS:
            values = [_finite_score(row) for row in rows if str(row["task"]) == task]
            if not values:
                raise ValueError(f"condition {condition_id} has no scores for {task}")
            result.append(
                {
                    "condition_id": condition_id,
                    "allocation": condition.allocation,
                    "method": condition.method,
                    "task": task,
                    "category": TASK_TO_CATEGORY[task],
                    "examples": len(values),
                    "mean_score": statistics.fmean(values),
                    "official_score_rounded": round(statistics.fmean(values), 2),
                    "population_stddev": statistics.pstdev(values),
                }
            )
    return tuple(result)


def aggregate_category_scores(
    task_rows: Iterable[Mapping[str, Any]],
) -> tuple[dict[str, Any], ...]:
    """Macro-average task scores within each of the six categories."""

    rows = list(task_rows)
    by_condition: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        condition_id = str(row.get("condition_id"))
        condition_by_id(condition_id)
        by_condition[condition_id].append(row)
    result: list[dict[str, Any]] = []
    for condition_id in _present_condition_ids(by_condition):
        condition = condition_by_id(condition_id)
        task_lookup = {
            str(row["task"]): float(row["mean_score"])
            for row in by_condition[condition_id]
        }
        if set(task_lookup) != set(TASKS):
            raise ValueError(f"task summary coverage mismatch for {condition_id}")
        for category, category_tasks in CATEGORIES.items():
            result.append(
                {
                    "condition_id": condition_id,
                    "allocation": condition.allocation,
                    "method": condition.method,
                    "category": category,
                    "tasks": len(category_tasks),
                    "macro_average": statistics.fmean(
                        task_lookup[task] for task in category_tasks
                    ),
                }
            )
    return tuple(result)


def aggregate_length_scores(
    score_rows: Iterable[Mapping[str, Any]],
) -> tuple[dict[str, Any], ...]:
    """Task-macro score within each pinned LongBench-E length bucket."""

    grouped = _group_condition_scores(score_rows)
    result: list[dict[str, Any]] = []
    for condition_id in _present_condition_ids(grouped):
        condition = condition_by_id(condition_id)
        normalized: list[tuple[Mapping[str, Any], str]] = []
        for row in grouped[condition_id]:
            if row.get("length_bucket") not in (None, ""):
                bucket = str(row["length_bucket"])
            elif row.get("dataset_length") not in (None, ""):
                bucket = base.length_bucket(int(row["dataset_length"]))
            else:
                raise ValueError("length aggregation requires length_bucket or dataset_length")
            if bucket not in LENGTH_BUCKETS:
                raise ValueError(f"unknown length bucket: {bucket}")
            if row.get("dataset_length") not in (None, "") and base.length_bucket(
                int(row["dataset_length"])
            ) != bucket:
                raise ValueError("score row length_bucket disagrees with dataset_length")
            normalized.append((row, bucket))
        for bucket in LENGTH_BUCKETS:
            per_task: list[float] = []
            example_count = 0
            for task in TASKS:
                values = [
                    _finite_score(row)
                    for row, row_bucket in normalized
                    if str(row["task"]) == task and row_bucket == bucket
                ]
                if not values:
                    raise ValueError(
                        f"condition {condition_id} has no {task} scores in {bucket}"
                    )
                per_task.append(statistics.fmean(values))
                example_count += len(values)
            result.append(
                {
                    "condition_id": condition_id,
                    "allocation": condition.allocation,
                    "method": condition.method,
                    "length_bucket": bucket,
                    "tasks": len(per_task),
                    "examples": example_count,
                    "macro_average": statistics.fmean(per_task),
                }
            )
    return tuple(result)


@dataclass(frozen=True, slots=True)
class AggregationResult:
    task_rows: tuple[dict[str, Any], ...]
    category_rows: tuple[dict[str, Any], ...]
    length_rows: tuple[dict[str, Any], ...]
    overall_rows: tuple[dict[str, Any], ...]


def aggregate_scores(score_rows: Iterable[Mapping[str, Any]]) -> AggregationResult:
    """Build task, category, length, and both overall macro summaries."""

    rows = list(score_rows)
    task_rows = aggregate_task_scores(rows)
    category_rows = aggregate_category_scores(task_rows)
    length_rows = aggregate_length_scores(rows)
    grouped = _group_condition_scores(rows)
    overall_rows: list[dict[str, Any]] = []
    for condition_id in _present_condition_ids(grouped):
        condition = condition_by_id(condition_id)
        condition_tasks = [
            row for row in task_rows if row["condition_id"] == condition_id
        ]
        condition_categories = [
            row for row in category_rows if row["condition_id"] == condition_id
        ]
        if len(condition_tasks) != len(TASKS) or len(condition_categories) != len(CATEGORIES):
            raise AssertionError("aggregate coverage changed unexpectedly")
        overall_rows.append(
            {
                "condition_id": condition_id,
                "allocation": condition.allocation,
                "method": condition.method,
                "examples": len(grouped[condition_id]),
                "tasks": len(TASKS),
                "categories": len(CATEGORIES),
                "task_macro_average": statistics.fmean(
                    float(row["mean_score"]) for row in condition_tasks
                ),
                "equal_category_macro_average": statistics.fmean(
                    float(row["macro_average"]) for row in condition_categories
                ),
            }
        )
    return AggregationResult(
        task_rows,
        category_rows,
        length_rows,
        tuple(overall_rows),
    )


__all__ = [
    "AggregationResult",
    "ALLOCATION_ORDER",
    "BOOTSTRAP_SAMPLES",
    "BOOTSTRAP_SEED",
    "CATEGORIES",
    "CONDITIONS",
    "CONDITION_COUNT",
    "CONDITION_IDS",
    "Condition",
    "EXPECTED_ORIGINAL_COUNTS",
    "EXPECTED_SAMPLE_COUNTS",
    "LEARNED_STEPS",
    "LENGTH_BUCKETS",
    "MANIFEST_SCHEMA_VERSION",
    "PAIRED_CONTRASTS",
    "PairedAnalysis",
    "PairedContrast",
    "ROOT_SEED",
    "SUBSET_EXAMPLES_PER_CONDITION",
    "SUBSET_FRACTION",
    "TASKS",
    "TASK_TO_CATEGORY",
    "TOTAL_PREDICTIONS",
    "aggregate_category_scores",
    "aggregate_length_scores",
    "aggregate_scores",
    "aggregate_task_scores",
    "all_conditions",
    "analyze_paired_comparisons",
    "build_subset_manifest",
    "canonical_json_sha256",
    "condition_by_id",
    "conditions_for_allocation",
    "ensure_subset_manifest",
    "load_subset_manifest",
    "paired_comparison_rows",
    "paired_comparison_summaries",
    "subset_manifest_sha256",
    "task_stratified_paired_bootstrap",
    "validate_exact_paired_identities",
    "validate_subset_manifest",
]
