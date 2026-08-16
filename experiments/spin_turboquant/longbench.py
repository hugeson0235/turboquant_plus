"""Run the pinned LongBench-E subset study from its checked-in plan.

The inference implementation deliberately reuses the codec hooks and learned
rotation artifacts from :mod:`experiments.spin_turboquant`.  Every benchmark
condition is executed in a fresh child process, predictions are appended one
example at a time, and completed examples are reused on restart.

Typical invocation (from the TurboQuant+ repository root)::

    python -m experiments.spin_turboquant.longbench \
      --stage orchestrate \
      --model /path/to/Meta-Llama-3.1-8B-Instruct/snapshot \
      --rotation-dir experiments/spin_turboquant/results/instruct \
      --longbench-repo ../LongBench_official \
      --data-dir ../LongBench_data/<revision>/data \
      --output-dir experiments/spin_turboquant/results/longbench_subset
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import gc
import hashlib
import importlib.util
import json
import math
import os
import platform
import random
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

import numpy as np
import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from experiments.spin_turboquant.core import (
    codebook_tensor,
    install_key_codec_hooks,
)


LONG_BENCH_COMMIT = "2e00731f8d0bff23dc4325161044d0ed8af94c1e"
LONG_BENCH_DATASET_REVISION = "5e628be450b7e67fb7ae6e201bd6d8f7056f7672"
SUBSET_SEED = 20_020_305
SAMPLES_PER_LENGTH_BUCKET = 5
BITS = (2, 3, 4)
SEEDS = (17, 29, 43)
LENGTH_BUCKETS = ("0-4k", "4-8k", "8k+")
SPECIFICATION_PATH = (
    Path(__file__).resolve().parents[3] / "longbench_subset_experiment_plan.md"
)
TASKS = (
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
)
CATEGORIES = {
    "Single QA": ("qasper", "multifieldqa_en"),
    "Multi QA": ("hotpotqa", "2wikimqa"),
    "Summarization": ("gov_report", "multi_news"),
    "Few-shot": ("trec", "triviaqa", "samsum"),
    "Synthetic": ("passage_count", "passage_retrieval_en"),
    "Code": ("lcc", "repobench-p"),
}
TASK_TO_CATEGORY = {
    task: category for category, tasks in CATEGORIES.items() for task in tasks
}
TASK_BATCH_SIZES = {
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
}
# This is the task exclusion list in the official LongBench ``pred.py``.  The
# other tasks receive the model's native instruct chat template.
NO_CHAT_TASKS = {"trec", "triviaqa", "samsum", "lcc", "repobench-p"}
METRIC_NAMES = {
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
}
FIRST_LINE_TASKS = {"trec", "triviaqa", "samsum"}
PREDICTION_FIELDS = (
    "condition_id",
    "method",
    "bit_width",
    "seed",
    "task",
    "task_index",
    "example_index",
    "example_id",
    "dataset_length",
    "length_bucket",
    "prompt_tokens_before_truncation",
    "prompt_tokens",
    "prompt_truncated",
    "chat_template_applied",
    "max_new_tokens",
    "generated_tokens",
    "prediction",
    "answers",
    "all_classes",
    "batch_id",
    "batch_size",
    "batch_padded_prompt_tokens",
    "batch_prefill_seconds",
    "batch_decode_seconds",
    "batch_decode_steps",
    "batch_total_seconds",
    "prefill_seconds",
    "decode_seconds",
    "decode_seconds_per_token",
    "total_seconds",
    "peak_gpu_memory_bytes",
    "theoretical_kv_bytes_per_token",
    "actual_cache_dtype",
    "created_at",
)


@dataclass(frozen=True)
class Condition:
    method: str
    bit_width: int | None = None
    seed: int | None = None

    @property
    def condition_id(self) -> str:
        if self.method == "fp16":
            return "fp16_K16_V16"
        suffix = "" if self.seed is None else f"_s{self.seed}"
        return f"{self.method}_K{self.bit_width}_V16{suffix}"

    @property
    def rotation_key(self) -> str | None:
        return self.method if self.method in {"random", "learned"} else None


@dataclass(frozen=True)
class PreparedExample:
    task: str
    example_index: int
    example: dict[str, Any]
    input_ids: torch.Tensor
    prompt_metadata: dict[str, Any]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage",
        choices=("validate", "smoke", "full", "report", "orchestrate"),
        default="orchestrate",
    )
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--rotation-dir", type=Path, required=True)
    parser.add_argument("--longbench-repo", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--condition", choices=[c.condition_id for c in all_conditions()])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-context-length", type=int)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--batch-token-budget", type=int, default=32_768)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument(
        "--dataset-revision", default=LONG_BENCH_DATASET_REVISION
    )
    parser.add_argument("--longbench-commit", default=LONG_BENCH_COMMIT)
    return parser.parse_args(argv)


def all_conditions() -> list[Condition]:
    result = [Condition("fp16")]
    for bit in BITS:
        result.append(Condition("identity", bit))
        for seed in SEEDS:
            result.append(Condition("random", bit, seed))
            result.append(Condition("learned", bit, seed))
    return result


def condition_by_id(condition_id: str) -> Condition:
    matches = [c for c in all_conditions() if c.condition_id == condition_id]
    if len(matches) != 1:
        raise ValueError(f"unknown condition: {condition_id}")
    return matches[0]


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    )
    temporary.replace(path)


def read_json(path: Path, default: Any = None) -> Any:
    return json.loads(path.read_text()) if path.exists() else default


def write_csv(
    path: Path,
    rows: Iterable[dict[str, Any]],
    fieldnames: Sequence[str] | None = None,
) -> None:
    rows = list(rows)
    if fieldnames is None:
        ordered_fields: list[str] = []
        for row in rows:
            for field in row:
                if field not in ordered_fields:
                    ordered_fields.append(field)
        fieldnames = ordered_fields
    if not fieldnames:
        raise ValueError(f"fieldnames are required for empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def sha256_file(path: Path, *, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_json(payload: Any) -> str:
    encoded = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def git_output(repository: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(repository), *args], text=True
    ).strip()


def official_v1_dir(repository: Path) -> Path:
    nested = repository / "LongBench"
    candidate = nested if (nested / "metrics.py").exists() else repository
    if not (candidate / "metrics.py").exists():
        raise FileNotFoundError(f"LongBench v1 files are missing under {repository}")
    return candidate


def rotation_path(rotation_dir: Path, bit_width: int, seed: int) -> Path:
    return rotation_dir / "rotations" / f"bit{bit_width}_seed{seed}.pt"


def load_examples(data_dir: Path, task: str) -> list[dict[str, Any]]:
    path = data_dir / f"{task}_e.jsonl"
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
        raise ValueError(f"LongBench-E task is empty: {path}")
    ids = [str(example["_id"]) for example in examples]
    if len(ids) != len(set(ids)):
        raise ValueError(f"duplicate example IDs in {path}")
    return examples


def load_official_configs(longbench_repo: Path) -> tuple[dict[str, str], dict[str, int]]:
    config_dir = official_v1_dir(longbench_repo) / "config"
    prompts = read_json(config_dir / "dataset2prompt.json")
    max_lengths = read_json(config_dir / "dataset2maxlen.json")
    missing_prompts = set(TASKS) - prompts.keys()
    missing_lengths = set(TASKS) - max_lengths.keys()
    if missing_prompts or missing_lengths:
        raise ValueError(
            f"official configs are incomplete: prompts={sorted(missing_prompts)}, "
            f"max_lengths={sorted(missing_lengths)}"
        )
    return (
        {task: str(prompts[task]) for task in TASKS},
        {task: int(max_lengths[task]) for task in TASKS},
    )


def model_dimensions(model_path: Path) -> dict[str, Any]:
    config = AutoConfig.from_pretrained(model_path, local_files_only=True)
    head_dim = int(config.hidden_size // config.num_attention_heads)
    actual = {
        "num_hidden_layers": int(config.num_hidden_layers),
        "num_key_value_heads": int(config.num_key_value_heads),
        "head_dim": head_dim,
    }
    expected = {
        "num_hidden_layers": 32,
        "num_key_value_heads": 8,
        "head_dim": 128,
    }
    if actual != expected:
        raise ValueError(f"LongBenchSubset.md requires {expected}, but model has {actual}")
    return {
        **actual,
        "model_type": str(config.model_type),
        "architectures": list(config.architectures or []),
        "max_position_embeddings": int(config.max_position_embeddings),
        "torch_dtype": str(config.torch_dtype),
    }


def validate_assets(args: argparse.Namespace, *, deep: bool = True) -> dict[str, Any]:
    for label, path in {
        "model": args.model,
        "rotation directory": args.rotation_dir,
        "LongBench repository": args.longbench_repo,
        "LongBench data directory": args.data_dir,
    }.items():
        if not path.exists():
            raise FileNotFoundError(f"{label} does not exist: {path}")
    if not SPECIFICATION_PATH.is_file():
        raise FileNotFoundError(f"experiment specification is missing: {SPECIFICATION_PATH}")
    if not torch.cuda.is_available() and str(args.device).startswith("cuda"):
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")

    actual_commit = git_output(args.longbench_repo, "rev-parse", "HEAD")
    if actual_commit != args.longbench_commit:
        raise RuntimeError(
            f"LongBench checkout is {actual_commit}, expected {args.longbench_commit}"
        )
    v1_dir = official_v1_dir(args.longbench_repo)
    prompts, maximum_generation_lengths = load_official_configs(args.longbench_repo)
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    if not tokenizer.chat_template:
        raise RuntimeError(
            "LongBenchSubset.md requires the model's instruct chat template, but this "
            f"tokenizer has none: {args.model}"
        )
    dimensions = model_dimensions(args.model)
    max_context_length = args.max_context_length or dimensions["max_position_embeddings"]
    if max_context_length > dimensions["max_position_embeddings"]:
        raise ValueError("--max-context-length exceeds the model configuration")
    if max_context_length <= max(maximum_generation_lengths.values()):
        raise ValueError("max context length leaves no room for task generation")
    if args.batch_size < 1:
        raise ValueError("--batch-size must be positive")
    if args.batch_token_budget < 1:
        raise ValueError("--batch-token-budget must be positive")

    rotation_config = read_json(args.rotation_dir / "run_config.json", {})
    trained_model = rotation_config.get("arguments", {}).get("model")
    if trained_model is None:
        raise RuntimeError("rotation run_config.json does not record its model")
    if Path(trained_model).resolve() != args.model.resolve():
        raise RuntimeError(
            "rotation/model mismatch: rotations were trained on "
            f"{trained_model}, requested model is {args.model}"
        )
    if rotation_config.get("arguments", {}).get("norm_correction") is not False:
        raise RuntimeError("rotation artifacts must use the fixed no-correction protocol")

    artifact_hashes: dict[str, str] = {}
    for bit in BITS:
        for seed in SEEDS:
            path = rotation_path(args.rotation_dir, bit, seed)
            if not path.exists():
                raise FileNotFoundError(path)
            artifact_hashes[f"bit{bit}_seed{seed}"] = sha256_file(path)
            if deep:
                artifact = torch.load(path, map_location="cpu", weights_only=True)
                for key in ("random", "learned"):
                    tensor = artifact.get(key)
                    expected_shape = (32, 8, 128, 128)
                    if not isinstance(tensor, torch.Tensor) or tuple(tensor.shape) != expected_shape:
                        raise ValueError(f"{path}:{key} does not have shape {expected_shape}")
                del artifact

    codebook_hashes: dict[str, str] = {}
    for bit in BITS:
        centroids = codebook_tensor(bit, 128, device="cpu").numpy()
        codebook_hashes[str(bit)] = hashlib.sha256(centroids.tobytes()).hexdigest()

    data_counts: dict[str, int] = {}
    data_hashes: dict[str, str] = {}
    for task in TASKS:
        path = args.data_dir / f"{task}_e.jsonl"
        if not path.exists():
            raise FileNotFoundError(path)
        data_counts[task] = len(load_examples(args.data_dir, task))
        data_hashes[task] = sha256_file(path)

    config_dir = v1_dir / "config"
    tokenizer_files = [
        args.model / name
        for name in ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json")
        if (args.model / name).exists()
    ]
    repository_root = Path(__file__).resolve().parents[2]
    implementation_files = {
        "longbench_runner": Path(__file__).resolve(),
        "spin_turboquant_core": Path(__file__).resolve().with_name("core.py"),
        "turboquant_codebook": repository_root / "turboquant" / "codebook.py",
        "turboquant_rotation": repository_root / "turboquant" / "rotation.py",
    }
    return {
        "model": {
            "path": str(args.model),
            "revision": args.model.name,
            **dimensions,
            "tokenizer_files": {
                path.name: sha256_file(path) for path in tokenizer_files
            },
        },
        "rotation_run_config": str(args.rotation_dir / "run_config.json"),
        "rotation_artifact_hashes": artifact_hashes,
        "codebook_hashes": codebook_hashes,
        "implementation_hashes": {
            name: sha256_file(path) for name, path in implementation_files.items()
        },
        "specification": {
            "path": str(SPECIFICATION_PATH),
            "sha256": sha256_file(SPECIFICATION_PATH),
        },
        "longbench": {
            "repository": str(args.longbench_repo),
            "commit": actual_commit,
            "dataset_revision": args.dataset_revision,
            "data_counts": data_counts,
            "data_hashes": data_hashes,
            "prompt_config_sha256": sha256_file(config_dir / "dataset2prompt.json"),
            "max_generation_config_sha256": sha256_file(
                config_dir / "dataset2maxlen.json"
            ),
        },
        "max_context_length": max_context_length,
        "prompts": prompts,
        "maximum_generation_lengths": maximum_generation_lengths,
    }


def environment_metadata(device: torch.device) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "timestamp": utc_now(),
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "numpy": np.__version__,
        "command": sys.argv,
    }
    for package in ("transformers", "rouge", "fuzzywuzzy", "Levenshtein", "jieba"):
        try:
            module = __import__(package)
            metadata[package] = getattr(module, "__version__", "unknown")
        except ImportError:
            metadata[package] = None
    if device.type == "cuda":
        properties = torch.cuda.get_device_properties(device)
        metadata["gpu"] = {
            "name": properties.name,
            "total_memory_bytes": properties.total_memory,
            "capability": [properties.major, properties.minor],
        }
    try:
        metadata["turboquant_plus_commit"] = git_output(Path.cwd(), "rev-parse", "HEAD")
        metadata["turboquant_plus_status"] = git_output(
            Path.cwd(), "status", "--short"
        ).splitlines()
    except (OSError, subprocess.CalledProcessError):
        pass
    return metadata


def length_bucket(length: int) -> str:
    if length < 4000:
        return "0-4k"
    if length < 8000:
        return "4-8k"
    return "8k+"


def build_subset_manifest(
    data_dir: Path,
    *,
    dataset_revision: str,
    data_hashes: dict[str, str],
) -> dict[str, Any]:
    """Build the deterministic, task/bucket-stratified subset specification."""

    generator = random.Random(SUBSET_SEED)
    selected_rows: list[dict[str, Any]] = []
    for task_index, task in enumerate(TASKS):
        examples = load_examples(data_dir, task)
        task_example_order = 0
        for bucket in LENGTH_BUCKETS:
            candidates = [
                (dataset_index, example)
                for dataset_index, example in enumerate(examples)
                if length_bucket(int(example["length"])) == bucket
            ]
            if len(candidates) < SAMPLES_PER_LENGTH_BUCKET:
                raise RuntimeError(
                    f"{task}:{bucket} has {len(candidates)} examples; "
                    f"{SAMPLES_PER_LENGTH_BUCKET} are required"
                )
            # Sampling is random without replacement. Sorting only fixes execution
            # order after selection; it does not change the sampled identities.
            chosen = sorted(
                generator.sample(candidates, SAMPLES_PER_LENGTH_BUCKET),
                key=lambda value: value[0],
            )
            for dataset_index, example in chosen:
                selected_rows.append(
                    {
                        "manifest_index": len(selected_rows),
                        "task": task,
                        "task_index": task_index,
                        "task_example_order": task_example_order,
                        "length_bucket": bucket,
                        "dataset_index": dataset_index,
                        "example_id": str(example["_id"]),
                        "dataset_length": int(example["length"]),
                    }
                )
                task_example_order += 1

    expected_count = (
        len(TASKS) * len(LENGTH_BUCKETS) * SAMPLES_PER_LENGTH_BUCKET
    )
    if len(selected_rows) != expected_count:
        raise AssertionError(
            f"subset construction produced {len(selected_rows)}, expected {expected_count}"
        )
    return {
        "schema_version": 1,
        "sampling_seed": SUBSET_SEED,
        "sampling_method": (
            "one Python random.Random stream; random.sample without replacement "
            "in canonical task and length-bucket order; selected source indexes "
            "sorted within each task/bucket"
        ),
        "samples_per_length_bucket": SAMPLES_PER_LENGTH_BUCKET,
        "samples_per_task": len(LENGTH_BUCKETS) * SAMPLES_PER_LENGTH_BUCKET,
        "example_count": expected_count,
        "task_order": list(TASKS),
        "length_bucket_order": list(LENGTH_BUCKETS),
        "length_bucket_boundaries": {
            "0-4k": {"minimum_inclusive": 0, "maximum_exclusive": 4000},
            "4-8k": {"minimum_inclusive": 4000, "maximum_exclusive": 8000},
            "8k+": {"minimum_inclusive": 8000, "maximum_exclusive": None},
        },
        "dataset_revision": dataset_revision,
        "data_hashes": data_hashes,
        "examples": selected_rows,
    }


def ensure_subset_manifest(
    output_dir: Path,
    data_dir: Path,
    *,
    dataset_revision: str,
    data_hashes: dict[str, str],
) -> dict[str, Any]:
    expected = build_subset_manifest(
        data_dir,
        dataset_revision=dataset_revision,
        data_hashes=data_hashes,
    )
    path = output_dir / "subset_manifest.json"
    existing = read_json(path)
    if existing is None:
        write_json(path, expected)
    elif existing != expected:
        raise RuntimeError(
            f"existing subset manifest does not match the pinned sampling protocol: {path}"
        )
    return expected


def middle_truncate(token_ids: Sequence[int], maximum: int) -> list[int]:
    if len(token_ids) <= maximum:
        return list(token_ids)
    front = maximum // 2
    back = maximum - front
    return list(token_ids[:front]) + list(token_ids[-back:])


def prepare_prompt(
    tokenizer: Any,
    prompt_format: str,
    example: dict[str, Any],
    task: str,
    *,
    max_context_length: int,
    max_new_tokens: int,
) -> tuple[torch.Tensor, dict[str, Any]]:
    raw_prompt = prompt_format.format(**example)
    max_prompt_tokens = max_context_length - max_new_tokens
    raw_ids = tokenizer(raw_prompt, truncation=False).input_ids
    before_truncation = len(raw_ids)
    raw_was_truncated = before_truncation > max_prompt_tokens
    if raw_was_truncated:
        truncated = middle_truncate(raw_ids, max_prompt_tokens)
        raw_prompt = tokenizer.decode(truncated, skip_special_tokens=True)

    chat_applied = task not in NO_CHAT_TASKS
    if chat_applied:
        final_ids = tokenizer.apply_chat_template(
            [{"role": "user", "content": raw_prompt}],
            tokenize=True,
            add_generation_prompt=True,
        )
    else:
        final_ids = tokenizer(raw_prompt, truncation=False).input_ids
    final_was_truncated = len(final_ids) > max_prompt_tokens
    if final_was_truncated:
        final_ids = middle_truncate(final_ids, max_prompt_tokens)
    if not final_ids:
        raise ValueError(f"empty tokenized prompt for {task}:{example['_id']}")
    return torch.tensor(final_ids, dtype=torch.long).unsqueeze(0), {
        "prompt_tokens_before_truncation": before_truncation,
        "prompt_tokens": len(final_ids),
        "prompt_truncated": raw_was_truncated or final_was_truncated,
        "chat_template_applied": chat_applied,
    }


def eos_token_ids(model: Any, tokenizer: Any, task: str) -> set[int]:
    configured = model.generation_config.eos_token_id
    if configured is None:
        values: list[int] = []
    elif isinstance(configured, int):
        values = [configured]
    else:
        values = [int(value) for value in configured]
    if tokenizer.eos_token_id is not None:
        values.append(int(tokenizer.eos_token_id))
    if task == "samsum":
        newline = tokenizer.encode("\n", add_special_tokens=False)
        if newline:
            values.append(int(newline[-1]))
    return set(values)


@torch.inference_mode()
def greedy_generate_batch(
    model: Any,
    input_ids: Sequence[torch.Tensor],
    *,
    max_new_tokens: int,
    stop_ids: set[int],
    suppress_initial_stop: bool,
    pad_token_id: int,
    device: torch.device,
) -> tuple[list[list[int]], dict[str, float | int]]:
    if not input_ids:
        raise ValueError("at least one prompt is required")
    batch_size = len(input_ids)
    lengths = [int(value.shape[-1]) for value in input_ids]
    maximum_prompt_length = max(lengths)
    padded_ids = torch.full(
        (batch_size, maximum_prompt_length),
        int(pad_token_id),
        dtype=torch.long,
        device=device,
    )
    attention_mask = torch.zeros_like(padded_ids, device=device)
    for row_index, value in enumerate(input_ids):
        length = lengths[row_index]
        padded_ids[row_index, -length:] = value[0].to(device)
        attention_mask[row_index, -length:] = 1
    position_ids = attention_mask.long().cumsum(-1) - 1
    position_ids.masked_fill_(attention_mask == 0, 1)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
    total_start = time.perf_counter()
    prefill_start = time.perf_counter()
    output = model(
        input_ids=padded_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        use_cache=True,
        return_dict=True,
        num_logits_to_keep=1,
    )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    prefill_seconds = time.perf_counter() - prefill_start
    next_logits = output.logits[:, -1, :]
    if suppress_initial_stop and stop_ids:
        next_logits = next_logits.clone()
        next_logits[:, list(stop_ids)] = -torch.inf
    next_token = torch.argmax(next_logits, dim=-1, keepdim=True)
    generated = [[int(next_token[row_index].item())] for row_index in range(batch_size)]
    past_key_values = output.past_key_values
    del output

    decode_seconds = 0.0
    decode_steps = 0
    generated_steps = 1
    active_indices = [
        row_index
        for row_index, tokens in enumerate(generated)
        if tokens[-1] not in stop_ids
    ]
    if len(active_indices) != batch_size:
        keep = torch.tensor(active_indices, dtype=torch.long, device=device)
        attention_mask = attention_mask.index_select(0, keep)
        next_token = next_token.index_select(0, keep)
        past_key_values = tuple(
            tuple(value.index_select(0, keep) for value in layer)
            for layer in past_key_values
        )
    while generated_steps < max_new_tokens and active_indices:
        attention_mask = torch.cat(
            [
                attention_mask,
                torch.ones(
                    (len(active_indices), 1),
                    dtype=attention_mask.dtype,
                    device=device,
                ),
            ],
            dim=-1,
        )
        next_position_ids = (attention_mask.long().sum(-1) - 1).clamp_min(0).unsqueeze(-1)
        decode_start = time.perf_counter()
        output = model(
            input_ids=next_token,
            attention_mask=attention_mask,
            position_ids=next_position_ids,
            past_key_values=past_key_values,
            use_cache=True,
            return_dict=True,
            num_logits_to_keep=1,
        )
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        decode_seconds += time.perf_counter() - decode_start
        next_token = torch.argmax(output.logits[:, -1, :], dim=-1, keepdim=True)
        past_key_values = output.past_key_values
        surviving_rows: list[int] = []
        surviving_positions: list[int] = []
        for active_position, row_index in enumerate(active_indices):
            token = int(next_token[active_position].item())
            generated[row_index].append(token)
            if token not in stop_ids:
                surviving_rows.append(row_index)
                surviving_positions.append(active_position)
        del output
        decode_steps += 1
        generated_steps += 1
        if generated_steps >= max_new_tokens:
            active_indices = []
        elif len(surviving_rows) != len(active_indices):
            active_indices = surviving_rows
            if active_indices:
                keep = torch.tensor(
                    surviving_positions, dtype=torch.long, device=device
                )
                attention_mask = attention_mask.index_select(0, keep)
                next_token = next_token.index_select(0, keep)
                past_key_values = tuple(
                    tuple(value.index_select(0, keep) for value in layer)
                    for layer in past_key_values
                )
        else:
            active_indices = surviving_rows

    total_seconds = time.perf_counter() - total_start
    peak_memory = (
        int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
    )
    measured_decode_tokens = sum(max(len(tokens) - 1, 0) for tokens in generated)
    metrics: dict[str, float | int] = {
        "batch_size": batch_size,
        "batch_padded_prompt_tokens": batch_size * maximum_prompt_length,
        "batch_prefill_seconds": prefill_seconds,
        "batch_decode_seconds": decode_seconds,
        "batch_decode_steps": decode_steps,
        "batch_total_seconds": total_seconds,
        "batch_logical_decode_tokens": measured_decode_tokens,
        "peak_gpu_memory_bytes": peak_memory,
    }
    del (
        past_key_values,
        next_token,
        attention_mask,
        position_ids,
        padded_ids,
    )
    return generated, metrics


def theoretical_kv_bytes_per_token(condition: Condition, dimensions: dict[str, Any]) -> int:
    layers = int(dimensions["num_hidden_layers"])
    heads = int(dimensions["num_key_value_heads"])
    head_dim = int(dimensions["head_dim"])
    value_bytes = layers * heads * head_dim * 2
    if condition.method == "fp16":
        key_bytes = value_bytes
    else:
        # The reference codec uses packed indices plus one FP32 norm for every
        # layer/head/token key vector. Values remain unquantized in this study.
        key_bytes = layers * heads * ((head_dim * int(condition.bit_width)) // 8 + 4)
    return key_bytes + value_bytes


def load_condition_rotations(
    condition: Condition,
    rotation_dir: Path,
    dimensions: dict[str, Any],
    device: torch.device,
) -> torch.Tensor:
    layers = int(dimensions["num_hidden_layers"])
    heads = int(dimensions["num_key_value_heads"])
    head_dim = int(dimensions["head_dim"])
    if condition.method == "identity":
        return torch.eye(head_dim, dtype=torch.float32, device=device).expand(
            layers, heads, head_dim, head_dim
        )
    if condition.seed is None or condition.bit_width is None or condition.rotation_key is None:
        raise ValueError(f"condition has no rotation artifact: {condition}")
    artifact = torch.load(
        rotation_path(rotation_dir, condition.bit_width, condition.seed),
        map_location="cpu",
        weights_only=True,
    )
    return artifact[condition.rotation_key].to(device=device, dtype=torch.float32)


def load_model(model_path: Path, device: torch.device) -> Any:
    print(f"[{utc_now()}] loading {model_path} on {device}", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        local_files_only=True,
        torch_dtype=torch.bfloat16 if device.type == "cuda" else torch.float32,
        low_cpu_mem_usage=True,
        attn_implementation="sdpa",
    )
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model.to(device)


def append_prediction(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def append_prediction_batch(path: Path, records: Sequence[dict[str, Any]]) -> None:
    if not records:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = "".join(
        json.dumps(record, ensure_ascii=False) + "\n" for record in records
    )
    with path.open("a", encoding="utf-8") as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())


def write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    temporary.replace(path)


def read_predictions(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    result: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"invalid JSON at {path}:{line_number}: {error}") from error
            result.append(value)
    keys = [(str(row["task"]), str(row["example_id"])) for row in result]
    if len(keys) != len(set(keys)):
        raise ValueError(f"duplicate predictions in {path}")
    return result


def canonical_examples(
    data_dir: Path,
    manifest: dict[str, Any],
    *,
    smoke: bool,
) -> list[tuple[str, int, dict[str, Any]]]:
    ordered: list[tuple[str, int, dict[str, Any]]] = []
    by_task = {task: load_examples(data_dir, task) for task in TASKS}
    seen_smoke_tasks: set[str] = set()
    for row in manifest["examples"]:
        task = str(row["task"])
        if smoke and task in seen_smoke_tasks:
            continue
        dataset_index = int(row["dataset_index"])
        examples = by_task.get(task)
        if examples is None or not 0 <= dataset_index < len(examples):
            raise RuntimeError(
                f"manifest points outside pinned data: {task}:{dataset_index}"
            )
        example = examples[dataset_index]
        if str(example["_id"]) != str(row["example_id"]):
            raise RuntimeError(
                "manifest identity mismatch at "
                f"{task}:{dataset_index}: {example['_id']} != {row['example_id']}"
            )
        if (
            int(example["length"]) != int(row["dataset_length"])
            or length_bucket(int(example["length"])) != str(row["length_bucket"])
        ):
            raise RuntimeError(f"manifest length mismatch at {task}:{dataset_index}")
        ordered.append((task, dataset_index, example))
        seen_smoke_tasks.add(task)

    expected = len(TASKS) if smoke else int(manifest["example_count"])
    if len(ordered) != expected:
        raise RuntimeError(
            f"manifest selected {len(ordered)} {'smoke' if smoke else 'full'} "
            f"examples, expected {expected}"
        )
    return ordered


def prepared_batches(
    expected: Sequence[tuple[str, int, dict[str, Any]]],
    tokenizer: Any,
    prompts: dict[str, str],
    maximum_generation_lengths: dict[str, int],
    *,
    max_context_length: int,
    maximum_batch_size: int,
    batch_token_budget: int,
) -> Iterator[tuple[str, list[PreparedExample]]]:
    current: list[PreparedExample] = []
    task_batch_index: dict[str, int] = {task: 0 for task in TASKS}

    def emit() -> tuple[str, list[PreparedExample]]:
        task = current[0].task
        batch_index = task_batch_index[task]
        task_batch_index[task] += 1
        first = current[0].example_index
        last = current[-1].example_index
        return f"{task}_batch{batch_index:04d}_examples{first}-{last}", list(current)

    for task, example_index, example in expected:
        max_new_tokens = int(maximum_generation_lengths[task])
        input_ids, prompt_metadata = prepare_prompt(
            tokenizer,
            prompts[task],
            example,
            task,
            max_context_length=max_context_length,
            max_new_tokens=max_new_tokens,
        )
        prepared = PreparedExample(
            task=task,
            example_index=example_index,
            example=example,
            input_ids=input_ids,
            prompt_metadata=prompt_metadata,
        )
        candidate = [*current, prepared]
        candidate_max_prompt = max(
            int(value.prompt_metadata["prompt_tokens"]) for value in candidate
        )
        candidate_padded_tokens = len(candidate) * (
            candidate_max_prompt + max_new_tokens
        )
        task_changed = bool(current and current[0].task != task)
        task_batch_size = min(maximum_batch_size, TASK_BATCH_SIZES[task])
        too_many = len(candidate) > task_batch_size
        over_budget = bool(current and candidate_padded_tokens > batch_token_budget)
        if task_changed or too_many or over_budget:
            yield emit()
            current = [prepared]
        else:
            current = candidate
    if current:
        yield emit()


def canonicalize_predictions(
    path: Path, expected: Sequence[tuple[str, int, dict[str, Any]]]
) -> list[dict[str, Any]]:
    rows = read_predictions(path)
    lookup = {(str(row["task"]), str(row["example_id"])): row for row in rows}
    expected_keys = [(task, str(example["_id"])) for task, _, example in expected]
    missing = [key for key in expected_keys if key not in lookup]
    extra = sorted(set(lookup) - set(expected_keys))
    if missing or extra:
        raise RuntimeError(
            f"prediction coverage mismatch for {path}: missing={missing[:5]}, extra={extra[:5]}"
        )
    canonical = [lookup[key] for key in expected_keys]
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in canonical:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary.replace(path)
    return canonical


def run_directory(output_dir: Path, mode: str, condition: Condition) -> Path:
    return output_dir / mode / condition.condition_id


def protocol_payload(
    args: argparse.Namespace,
    assets: dict[str, Any],
    manifest: dict[str, Any],
    condition: Condition,
    mode: str,
) -> dict[str, Any]:
    artifact_hash = None
    if condition.seed is not None and condition.bit_width is not None:
        artifact_hash = assets["rotation_artifact_hashes"][
            f"bit{condition.bit_width}_seed{condition.seed}"
        ]
    return {
        "specification": assets["specification"],
        "mode": mode,
        "condition": {
            "condition_id": condition.condition_id,
            "method": condition.method,
            "key_bit_width": 16 if condition.method == "fp16" else condition.bit_width,
            "value_bit_width": 16,
            "seed": condition.seed,
        },
        "model": assets["model"],
        "implementation_hashes": assets["implementation_hashes"],
        "longbench": assets["longbench"],
        "subset": {
            "manifest": str((args.output_dir / "subset_manifest.json").resolve()),
            "manifest_sha256": sha256_json(manifest),
            "sampling_seed": int(manifest["sampling_seed"]),
            "examples_per_condition": int(manifest["example_count"]),
            "samples_per_task": int(manifest["samples_per_task"]),
            "samples_per_length_bucket": int(
                manifest["samples_per_length_bucket"]
            ),
        },
        "rotation_artifact_sha256": artifact_hash,
        "rotation_artifact_key": condition.rotation_key,
        "codebook_sha256": (
            assets["codebook_hashes"].get(str(condition.bit_width))
            if condition.bit_width is not None
            else None
        ),
        "task_order": list(TASKS),
        "categories": {key: list(value) for key, value in CATEGORIES.items()},
        "decoding": {
            "strategy": "greedy_argmax",
            "temperature": 0,
            "num_logits_to_keep": 1,
            "maximum_batch_size": 1 if mode == "smoke" else args.batch_size,
            "task_batch_sizes": {
                task: 1 if mode == "smoke" else min(args.batch_size, size)
                for task, size in TASK_BATCH_SIZES.items()
            },
            "batch_token_budget": args.batch_token_budget,
            "batching": "contiguous within task; fixed from all examples before resume filtering",
            "finished_sequence_handling": "compact legacy KV batches after each stop token",
            "maximum_generation_lengths": assets["maximum_generation_lengths"],
            "max_context_length": assets["max_context_length"],
            "middle_truncation": "preserve tokenized front and back halves",
            "chat_policy": "native instruct template with official LongBench exclusions",
            "no_chat_tasks": sorted(NO_CHAT_TASKS),
        },
        "quantization": {
            "implementation": "turboquant_plus head-wise PolarQuant emulation",
            "component": "pre-RoPE key only; values remain BF16",
            "prompt_and_generated_tokens": True,
            "norm_correction": False,
            "codebook": "local fixed TurboQuant Lloyd-Max",
            "cache_storage": "quality emulation stores reconstructed keys in BF16",
        },
    }


def initialize_run_config(
    run_dir: Path,
    protocol: dict[str, Any],
    environment: dict[str, Any],
) -> dict[str, Any]:
    path = run_dir / "run_config.json"
    protocol_hash = sha256_json(protocol)
    existing = read_json(path, {})
    if existing and existing.get("protocol_sha256") != protocol_hash:
        raise RuntimeError(
            f"existing run uses a different protocol: {path}; use a new output directory"
        )
    payload = {
        "created_at": existing.get("created_at", utc_now()),
        "updated_at": utc_now(),
        "status": existing.get("status", "running"),
        "completed_predictions": existing.get("completed_predictions", 0),
        "expected_predictions": existing.get("expected_predictions"),
        "protocol_sha256": protocol_hash,
        "protocol": protocol,
        "environment": environment,
    }
    write_json(path, payload)
    return payload


def update_run_status(run_dir: Path, **updates: Any) -> None:
    path = run_dir / "run_config.json"
    payload = read_json(path, {})
    payload.update(updates)
    payload["updated_at"] = utc_now()
    write_json(path, payload)


def import_official_metrics(longbench_repo: Path) -> Any:
    path = official_v1_dir(longbench_repo) / "metrics.py"
    spec = importlib.util.spec_from_file_location("pinned_longbench_metrics", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot import official LongBench metrics from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def score_prediction(metrics: Any, row: dict[str, Any]) -> float:
    prediction = str(row["prediction"])
    if row["task"] in FIRST_LINE_TASKS:
        prediction = prediction.lstrip("\n").split("\n")[0]
    metric = getattr(metrics, METRIC_NAMES[str(row["task"])])
    score = 0.0
    for answer in row["answers"]:
        score = max(
            score,
            float(metric(prediction, answer, all_classes=row.get("all_classes"))),
        )
    return score * 100.0


def score_condition(
    run_dir: Path,
    predictions: Sequence[dict[str, Any]],
    longbench_repo: Path,
) -> dict[str, Any]:
    metrics = import_official_metrics(longbench_repo)
    score_rows: list[dict[str, Any]] = []
    for row in predictions:
        score = score_prediction(metrics, row)
        if not math.isfinite(score):
            raise RuntimeError(
                f"non-finite score for {row['task']}:{row['example_id']}"
            )
        score_rows.append(
            {
                "condition_id": row["condition_id"],
                "method": row["method"],
                "bit_width": row["bit_width"],
                "seed": row["seed"],
                "task": row["task"],
                "category": TASK_TO_CATEGORY[row["task"]],
                "example_index": row["example_index"],
                "example_id": row["example_id"],
                "dataset_length": row["dataset_length"],
                "length_bucket": row["length_bucket"],
                "score": score,
            }
        )
    write_csv(run_dir / "scores.csv", score_rows)

    task_rows: list[dict[str, Any]] = []
    task_means: dict[str, float] = {}
    for task in TASKS:
        values = [float(row["score"]) for row in score_rows if row["task"] == task]
        if not values:
            raise RuntimeError(f"missing scores for {task} in {run_dir}")
        mean = statistics.fmean(values)
        task_means[task] = mean
        task_rows.append(
            {
                "condition_id": predictions[0]["condition_id"],
                "method": predictions[0]["method"],
                "bit_width": predictions[0]["bit_width"],
                "seed": predictions[0]["seed"],
                "task": task,
                "category": TASK_TO_CATEGORY[task],
                "examples": len(values),
                "mean_score": mean,
                "official_score_rounded": round(mean, 2),
                "population_stddev": statistics.pstdev(values),
            }
        )
    write_csv(run_dir / "task_summary.csv", task_rows)

    category_rows: list[dict[str, Any]] = []
    category_means: dict[str, float] = {}
    for category, tasks in CATEGORIES.items():
        mean = statistics.fmean(task_means[task] for task in tasks)
        category_means[category] = mean
        category_rows.append(
            {
                "condition_id": predictions[0]["condition_id"],
                "method": predictions[0]["method"],
                "bit_width": predictions[0]["bit_width"],
                "seed": predictions[0]["seed"],
                "category": category,
                "tasks": len(tasks),
                "macro_average": mean,
            }
        )
    write_csv(run_dir / "category_summary.csv", category_rows)

    length_rows: list[dict[str, Any]] = []
    length_means: dict[str, float] = {}
    for bucket in LENGTH_BUCKETS:
        per_task: list[float] = []
        example_count = 0
        for task in TASKS:
            values = [
                float(row["score"])
                for row in score_rows
                if row["task"] == task and row["length_bucket"] == bucket
            ]
            if values:
                per_task.append(statistics.fmean(values))
                example_count += len(values)
        if not per_task:
            continue
        mean = statistics.fmean(per_task)
        length_means[bucket] = mean
        length_rows.append(
            {
                "condition_id": predictions[0]["condition_id"],
                "method": predictions[0]["method"],
                "bit_width": predictions[0]["bit_width"],
                "seed": predictions[0]["seed"],
                "length_bucket": bucket,
                "tasks": len(per_task),
                "examples": example_count,
                "macro_average": mean,
            }
        )
    write_csv(run_dir / "length_summary.csv", length_rows)

    system_rows = [
        {
            key: row[key]
            for key in (
                "condition_id",
                "method",
                "bit_width",
                "seed",
                "task",
                "example_index",
                "example_id",
                "batch_id",
                "batch_size",
                "batch_padded_prompt_tokens",
                "prompt_tokens",
                "generated_tokens",
                "batch_prefill_seconds",
                "batch_decode_seconds",
                "batch_decode_steps",
                "batch_total_seconds",
                "prefill_seconds",
                "decode_seconds",
                "decode_seconds_per_token",
                "total_seconds",
                "peak_gpu_memory_bytes",
                "theoretical_kv_bytes_per_token",
                "actual_cache_dtype",
            )
        }
        for row in predictions
    ]
    write_csv(run_dir / "system_metrics.csv", system_rows)
    write_csv(
        run_dir / "paired_comparison.csv",
        [],
        fieldnames=(
            "bit_width",
            "seed",
            "task",
            "example_id",
            "random_score",
            "learned_score",
            "difference",
        ),
    )

    overall = statistics.fmean(task_means.values())
    empty_outputs = sum(not str(row["prediction"]).strip() for row in predictions)
    report_lines = [
        f"# LongBench-E: {predictions[0]['condition_id']}",
        "",
        f"- Examples: {len(predictions)}",
        f"- Tasks: {len(task_rows)}",
        f"- Overall task macro average: {overall:.6f}",
        f"- Empty outputs: {empty_outputs}",
        "",
        "| Task | Category | Examples | Mean | Stddev |",
        "|---|---|---:|---:|---:|",
    ]
    report_lines.extend(
        f"| {row['task']} | {row['category']} | {row['examples']} | "
        f"{float(row['mean_score']):.4f} | {float(row['population_stddev']):.4f} |"
        for row in task_rows
    )
    (run_dir / "report.md").write_text("\n".join(report_lines) + "\n")
    return {
        "overall_macro_average": overall,
        "empty_outputs": empty_outputs,
        "task_scores": task_means,
        "category_scores": category_means,
        "length_scores": length_means,
    }


def run_inference(args: argparse.Namespace, mode: str) -> None:
    if args.condition is None:
        raise ValueError(f"--condition is required for --stage {mode}")
    condition = condition_by_id(args.condition)
    assets = validate_assets(args, deep=False)
    device = torch.device(args.device)
    run_dir = run_directory(args.output_dir, mode, condition)
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest = ensure_subset_manifest(
        args.output_dir,
        args.data_dir,
        dataset_revision=args.dataset_revision,
        data_hashes=assets["longbench"]["data_hashes"],
    )
    protocol = protocol_payload(args, assets, manifest, condition, mode)
    config = initialize_run_config(run_dir, protocol, environment_metadata(device))
    expected = canonical_examples(
        args.data_dir,
        manifest,
        smoke=mode == "smoke",
    )
    update_run_status(run_dir, expected_predictions=len(expected), status="running")
    prediction_path = run_dir / "predictions.jsonl"
    existing = read_predictions(prediction_path)
    existing_lookup = {
        (str(row["task"]), str(row["example_id"])): row for row in existing
    }
    completed = {(str(row["task"]), str(row["example_id"])) for row in existing}
    expected_keys = {(task, str(example["_id"])) for task, _, example in expected}
    extra = completed - expected_keys
    if extra:
        raise RuntimeError(f"unexpected existing predictions: {sorted(extra)[:5]}")

    if len(completed) == len(expected):
        predictions = canonicalize_predictions(prediction_path, expected)
        summary = score_condition(run_dir, predictions, args.longbench_repo)
        update_run_status(
            run_dir,
            status="complete",
            completed_predictions=len(predictions),
            completed_at=utc_now(),
            summary=summary,
        )
        print(f"[{utc_now()}] {mode} {condition.condition_id} already complete", flush=True)
        return

    if mode == "full":
        smoke_dir = run_directory(args.output_dir, "smoke", condition)
        smoke_config = read_json(smoke_dir / "run_config.json", {})
        if smoke_config.get("status") != "complete" or int(
            smoke_config.get("completed_predictions", 0)
        ) != len(TASKS):
            raise RuntimeError(
                f"complete 13-task smoke test is required before {condition.condition_id} full run"
            )

    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    model = load_model(args.model, device)
    rotations: torch.Tensor | None = None
    centroids: torch.Tensor | None = None
    if condition.method != "fp16":
        rotations = load_condition_rotations(
            condition, args.rotation_dir, assets["model"], device
        )
        centroids = codebook_tensor(
            int(condition.bit_width), int(assets["model"]["head_dim"]), device=device
        )
    codec_context = (
        contextlib.nullcontext()
        if condition.method == "fp16"
        else install_key_codec_hooks(
            model,
            rotations,
            centroids,
            norm_correction=False,
        )
    )
    prompts = assets["prompts"]
    maximum_generation_lengths = assets["maximum_generation_lengths"]
    kv_bytes = theoretical_kv_bytes_per_token(condition, assets["model"])
    effective_batch_size = 1 if mode == "smoke" else args.batch_size
    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = tokenizer.eos_token_id
    if pad_token_id is None:
        pad_token_id = 0

    try:
        with codec_context:
            batches = prepared_batches(
                expected,
                tokenizer,
                prompts,
                maximum_generation_lengths,
                max_context_length=int(assets["max_context_length"]),
                maximum_batch_size=effective_batch_size,
                batch_token_budget=args.batch_token_budget,
            )
            for batch_ordinal, (batch_id, batch) in enumerate(batches, start=1):
                batch_keys = [
                    (value.task, str(value.example["_id"])) for value in batch
                ]
                if all(key in completed for key in batch_keys):
                    continue
                task = batch[0].task
                max_new_tokens = int(maximum_generation_lengths[task])
                generated_batch, measured = greedy_generate_batch(
                    model,
                    [value.input_ids for value in batch],
                    max_new_tokens=max_new_tokens,
                    stop_ids=eos_token_ids(model, tokenizer, task),
                    suppress_initial_stop=task == "samsum",
                    pad_token_id=int(pad_token_id),
                    device=device,
                )
                logical_decode_tokens = int(measured["batch_logical_decode_tokens"])
                decode_seconds_per_token = (
                    float(measured["batch_decode_seconds"]) / logical_decode_tokens
                    if logical_decode_tokens
                    else 0.0
                )
                prefill_share = float(measured["batch_prefill_seconds"]) / len(batch)
                overhead_share = max(
                    float(measured["batch_total_seconds"])
                    - float(measured["batch_prefill_seconds"])
                    - float(measured["batch_decode_seconds"]),
                    0.0,
                ) / len(batch)
                new_rows: list[dict[str, Any]] = []
                for prepared, generated_ids, key in zip(
                    batch, generated_batch, batch_keys
                ):
                    example = prepared.example
                    prediction = tokenizer.decode(
                        generated_ids, skip_special_tokens=True
                    )
                    example_decode_tokens = max(len(generated_ids) - 1, 0)
                    decode_share = decode_seconds_per_token * example_decode_tokens
                    row = {
                        "condition_id": condition.condition_id,
                        "method": condition.method,
                        "bit_width": condition.bit_width,
                        "seed": condition.seed,
                        "task": task,
                        "task_index": TASKS.index(task),
                        "example_index": prepared.example_index,
                        "example_id": str(example["_id"]),
                        "dataset_length": int(example["length"]),
                        "length_bucket": length_bucket(int(example["length"])),
                        **prepared.prompt_metadata,
                        "max_new_tokens": max_new_tokens,
                        "generated_tokens": len(generated_ids),
                        "prediction": prediction,
                        "answers": example["answers"],
                        "all_classes": example["all_classes"],
                        "batch_id": batch_id,
                        "batch_size": int(measured["batch_size"]),
                        "batch_padded_prompt_tokens": int(
                            measured["batch_padded_prompt_tokens"]
                        ),
                        "batch_prefill_seconds": float(
                            measured["batch_prefill_seconds"]
                        ),
                        "batch_decode_seconds": float(
                            measured["batch_decode_seconds"]
                        ),
                        "batch_decode_steps": int(measured["batch_decode_steps"]),
                        "batch_total_seconds": float(
                            measured["batch_total_seconds"]
                        ),
                        "prefill_seconds": prefill_share,
                        "decode_seconds": decode_share,
                        "decode_seconds_per_token": decode_seconds_per_token,
                        "total_seconds": prefill_share + decode_share + overhead_share,
                        "peak_gpu_memory_bytes": int(
                            measured["peak_gpu_memory_bytes"]
                        ),
                        "theoretical_kv_bytes_per_token": kv_bytes,
                        "actual_cache_dtype": "bfloat16 reconstructed-key emulation",
                        "created_at": utc_now(),
                    }
                    missing_fields = set(PREDICTION_FIELDS) - row.keys()
                    if missing_fields:
                        raise AssertionError(
                            f"prediction is missing {sorted(missing_fields)}"
                        )
                    if key in completed:
                        previous = existing_lookup[key]
                        if (
                            str(previous["prediction"]) != prediction
                            or int(previous["generated_tokens"]) != len(generated_ids)
                        ):
                            raise RuntimeError(
                                "replayed incomplete batch changed an existing "
                                f"prediction: {condition.condition_id} {key}"
                            )
                    else:
                        new_rows.append(row)
                append_prediction_batch(prediction_path, new_rows)
                for row in new_rows:
                    key = (str(row["task"]), str(row["example_id"]))
                    completed.add(key)
                    existing_lookup[key] = row
                update_run_status(run_dir, completed_predictions=len(completed))
                print(
                    f"[{utc_now()}] {mode} {condition.condition_id} "
                    f"batch={batch_ordinal} {batch_id} size={len(batch)} "
                    f"completed={len(completed)}/{len(expected)} "
                    f"padded_prompt_tokens={measured['batch_padded_prompt_tokens']} "
                    f"seconds={float(measured['batch_total_seconds']):.3f}",
                    flush=True,
                )
                del generated_batch, new_rows
    except BaseException as error:
        update_run_status(
            run_dir,
            status="failed",
            completed_predictions=len(completed),
            error=f"{type(error).__name__}: {error}",
        )
        raise
    finally:
        del model, rotations, centroids
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    predictions = canonicalize_predictions(prediction_path, expected)
    summary = score_condition(run_dir, predictions, args.longbench_repo)
    update_run_status(
        run_dir,
        status="complete",
        completed_predictions=len(predictions),
        completed_at=utc_now(),
        summary=summary,
    )
    print(f"[{utc_now()}] completed {mode} {condition.condition_id}", flush=True)


def numeric(value: Any) -> float:
    if value in (None, "", "None"):
        return math.nan
    return float(value)


def completed_full_run(
    output_dir: Path,
    condition: Condition,
    expected_count: int,
) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    run_dir = run_directory(output_dir, "full", condition)
    config = read_json(run_dir / "run_config.json", {})
    if config.get("status") != "complete":
        raise RuntimeError(f"full run is not complete: {condition.condition_id}")
    predictions = read_predictions(run_dir / "predictions.jsonl")
    scores = read_csv(run_dir / "scores.csv")
    if len(predictions) != expected_count or len(scores) != expected_count:
        raise RuntimeError(
            f"{condition.condition_id} expected {expected_count} rows, got "
            f"{len(predictions)} predictions and {len(scores)} scores"
        )
    prediction_keys = {
        (str(row["task"]), str(row["example_id"])) for row in predictions
    }
    score_keys = {(row["task"], row["example_id"]) for row in scores}
    if prediction_keys != score_keys:
        raise RuntimeError(f"prediction/score identities differ: {condition.condition_id}")
    return predictions, scores


def task_means(scores: Sequence[dict[str, str]]) -> dict[str, float]:
    result: dict[str, float] = {}
    for task in TASKS:
        values = [numeric(row["score"]) for row in scores if row["task"] == task]
        if not values:
            raise RuntimeError(f"no scores for task {task}")
        result[task] = statistics.fmean(values)
    return result


def macro_average(task_scores: dict[str, float], tasks: Sequence[str] = TASKS) -> float:
    return statistics.fmean(task_scores[task] for task in tasks)


def bootstrap_ci(
    arrays: Sequence[np.ndarray],
    *,
    samples: int,
    seed: int,
) -> tuple[float, float]:
    if samples < 100:
        raise ValueError("at least 100 bootstrap samples are required")
    if not arrays or any(array.size == 0 for array in arrays):
        raise ValueError("bootstrap groups must all be non-empty")
    rng = np.random.default_rng(seed)
    distribution = np.empty(samples, dtype=np.float64)
    for sample_index in range(samples):
        means = [
            float(np.mean(array[rng.integers(0, array.size, size=array.size)]))
            for array in arrays
        ]
        distribution[sample_index] = statistics.fmean(means)
    low, high = np.percentile(distribution, [2.5, 97.5])
    return float(low), float(high)


def paired_task_bootstrap_ci(
    task_seed_differences: Sequence[np.ndarray],
    *,
    samples: int,
    seed: int,
) -> tuple[float, float]:
    """Bootstrap tasks independently while sharing example draws across seeds.

    Each array is shaped ``(rotation seeds, examples in one task)``. A task's
    example indexes are sampled once per replicate and reused for every paired
    Random/Learned seed, as required by the subset protocol.
    """

    if samples < 100:
        raise ValueError("at least 100 bootstrap samples are required")
    if not task_seed_differences:
        raise ValueError("at least one task is required")
    seed_count: int | None = None
    for array in task_seed_differences:
        if array.ndim != 2 or array.shape[0] == 0 or array.shape[1] == 0:
            raise ValueError("task/seed difference arrays must be non-empty 2D arrays")
        if seed_count is None:
            seed_count = int(array.shape[0])
        elif int(array.shape[0]) != seed_count:
            raise ValueError("all tasks must contain the same rotation seeds")

    rng = np.random.default_rng(seed)
    distribution = np.empty(samples, dtype=np.float64)
    for sample_index in range(samples):
        task_means: list[float] = []
        for array in task_seed_differences:
            indexes = rng.integers(0, array.shape[1], size=array.shape[1])
            task_means.append(float(np.mean(array[:, indexes])))
        distribution[sample_index] = statistics.fmean(task_means)
    low, high = np.percentile(distribution, [2.5, 97.5])
    return float(low), float(high)


def markdown_table(headers: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join("---" for _ in headers) + "|",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)


def verdict_for_bit(
    learned_minus_random: float,
    seed_differences: Sequence[float],
    confidence_low: float,
    repeated_large_category_regression: bool,
    long_context_difference: float | None = None,
) -> str:
    seed_wins = sum(value > 0 for value in seed_differences)
    direction_is_consistent = (
        not repeated_large_category_regression
        and (long_context_difference is None or long_context_difference >= 0)
    )
    if (
        learned_minus_random > 0
        and seed_wins >= 2
        and confidence_low > 0
        and direction_is_consistent
    ):
        return "Supported"
    if learned_minus_random > 0 and seed_wins >= 2 and direction_is_consistent:
        return "Promising pilot"
    if learned_minus_random <= 0:
        return "Not supported"
    return "Mixed"


def write_plots(
    output_dir: Path,
    overall_rows: Sequence[dict[str, Any]],
    category_rows: Sequence[dict[str, Any]],
    seed_rows: Sequence[dict[str, Any]],
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    bits = [int(row["bit_width"]) for row in overall_rows]
    width = 0.23
    x = np.arange(len(bits))
    fig, axis = plt.subplots(figsize=(8, 4.8))
    axis.bar(x - width, [row["identity"] for row in overall_rows], width, label="Identity")
    axis.bar(x, [row["random_mean"] for row in overall_rows], width, label="Random")
    axis.bar(x + width, [row["learned_mean"] for row in overall_rows], width, label="Learned")
    axis.axhline(float(overall_rows[0]["fp16"]), color="black", linestyle="--", label="FP16")
    axis.set_xticks(x, [str(bit) for bit in bits])
    axis.set_xlabel("Key bit width (Values BF16)")
    axis.set_ylabel("LongBench-E task macro score")
    axis.legend(ncol=4, fontsize=8)
    axis.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_dir / "overall_comparison.png", dpi=180)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(9, 5))
    for category in CATEGORIES:
        rows = [row for row in category_rows if row["category"] == category]
        rows.sort(key=lambda row: int(row["bit_width"]))
        axis.plot(
            [int(row["bit_width"]) for row in rows],
            [float(row["learned_minus_random"]) for row in rows],
            marker="o",
            label=category,
        )
    axis.axhline(0.0, color="black", linewidth=0.8)
    axis.set_xticks(list(BITS))
    axis.set_xlabel("Key bit width (Values BF16)")
    axis.set_ylabel("Learned minus Random category score")
    axis.legend(ncol=2, fontsize=8)
    axis.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_dir / "category_paired_differences.png", dpi=180)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(8, 4.8))
    for seed in SEEDS:
        rows = [row for row in seed_rows if int(row["seed"]) == seed]
        rows.sort(key=lambda row: int(row["bit_width"]))
        axis.plot(
            [int(row["bit_width"]) for row in rows],
            [float(row["difference"]) for row in rows],
            marker="o",
            label=f"seed {seed}",
        )
    axis.axhline(0.0, color="black", linewidth=0.8)
    axis.set_xticks(list(BITS))
    axis.set_xlabel("Key bit width (Values BF16)")
    axis.set_ylabel("Learned minus Random task macro score")
    axis.legend()
    axis.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_dir / "seed_paired_differences.png", dpi=180)
    plt.close(fig)


def analyze_full_study(args: argparse.Namespace) -> dict[str, Any]:
    assets = validate_assets(args, deep=False)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = ensure_subset_manifest(
        args.output_dir,
        args.data_dir,
        dataset_revision=args.dataset_revision,
        data_hashes=assets["longbench"]["data_hashes"],
    )
    expected_count = int(manifest["example_count"])
    expected_examples = canonical_examples(args.data_dir, manifest, smoke=False)
    expected_keys = [
        (task, str(example["_id"])) for task, _, example in expected_examples
    ]
    predictions_by_condition: dict[str, list[dict[str, Any]]] = {}
    scores_by_condition: dict[str, list[dict[str, str]]] = {}
    task_means_by_condition: dict[str, dict[str, float]] = {}
    for condition in all_conditions():
        predictions, scores = completed_full_run(
            args.output_dir, condition, expected_count
        )
        actual_keys = [
            (str(row["task"]), str(row["example_id"])) for row in predictions
        ]
        if actual_keys != expected_keys:
            raise RuntimeError(
                f"{condition.condition_id} does not use the exact manifest order"
            )
        predictions_by_condition[condition.condition_id] = predictions
        scores_by_condition[condition.condition_id] = scores
        task_means_by_condition[condition.condition_id] = task_means(scores)

    fp16_tasks = task_means_by_condition[Condition("fp16").condition_id]
    fp16_overall = macro_average(fp16_tasks)
    paired_rows: list[dict[str, Any]] = []
    seed_rows: list[dict[str, Any]] = []
    category_rows: list[dict[str, Any]] = []
    task_paired_rows: list[dict[str, Any]] = []
    overall_rows: list[dict[str, Any]] = []

    for bit in BITS:
        identity_id = Condition("identity", bit).condition_id
        identity_tasks = task_means_by_condition[identity_id]
        seed_differences: list[float] = []
        random_overalls: list[float] = []
        learned_overalls: list[float] = []
        per_task_arrays: dict[str, list[np.ndarray]] = {task: [] for task in TASKS}
        per_category_seed_differences: dict[str, list[float]] = {
            category: [] for category in CATEGORIES
        }

        for seed in SEEDS:
            random_id = Condition("random", bit, seed).condition_id
            learned_id = Condition("learned", bit, seed).condition_id
            random_task_means = task_means_by_condition[random_id]
            learned_task_means = task_means_by_condition[learned_id]
            random_overall = macro_average(random_task_means)
            learned_overall = macro_average(learned_task_means)
            difference = learned_overall - random_overall
            random_overalls.append(random_overall)
            learned_overalls.append(learned_overall)
            seed_differences.append(difference)
            seed_rows.append(
                {
                    "bit_width": bit,
                    "seed": seed,
                    "random_macro_average": random_overall,
                    "learned_macro_average": learned_overall,
                    "difference": difference,
                }
            )

            random_lookup = {
                (row["task"], row["example_id"]): numeric(row["score"])
                for row in scores_by_condition[random_id]
            }
            learned_lookup = {
                (row["task"], row["example_id"]): numeric(row["score"])
                for row in scores_by_condition[learned_id]
            }
            prediction_lookup = {
                (str(row["task"]), str(row["example_id"])): row
                for row in predictions_by_condition[random_id]
            }
            if random_lookup.keys() != learned_lookup.keys():
                raise RuntimeError(f"paired identities differ for bit={bit}, seed={seed}")
            for task in TASKS:
                differences: list[float] = []
                task_keys = [key for key in random_lookup if key[0] == task]
                task_keys.sort(key=lambda key: int(prediction_lookup[key]["example_index"]))
                for key in task_keys:
                    row = prediction_lookup[key]
                    random_score = random_lookup[key]
                    learned_score = learned_lookup[key]
                    paired_rows.append(
                        {
                            "bit_width": bit,
                            "seed": seed,
                            "task": task,
                            "category": TASK_TO_CATEGORY[task],
                            "example_index": row["example_index"],
                            "example_id": key[1],
                            "dataset_length": row["dataset_length"],
                            "length_bucket": row["length_bucket"],
                            "random_score": random_score,
                            "learned_score": learned_score,
                            "difference": learned_score - random_score,
                        }
                    )
                    differences.append(learned_score - random_score)
                array = np.asarray(differences, dtype=np.float64)
                per_task_arrays[task].append(array)
            for category, category_tasks in CATEGORIES.items():
                per_category_seed_differences[category].append(
                    macro_average(learned_task_means, category_tasks)
                    - macro_average(random_task_means, category_tasks)
                )

        random_mean = statistics.fmean(random_overalls)
        learned_mean = statistics.fmean(learned_overalls)
        overall_difference = learned_mean - random_mean
        stacked_task_arrays = {
            task: np.stack(per_task_arrays[task], axis=0) for task in TASKS
        }
        overall_low, overall_high = paired_task_bootstrap_ci(
            [stacked_task_arrays[task] for task in TASKS],
            samples=args.bootstrap_samples,
            seed=20260814 + bit,
        )

        for task in TASKS:
            random_values = [
                task_means_by_condition[Condition("random", bit, seed).condition_id][task]
                for seed in SEEDS
            ]
            learned_values = [
                task_means_by_condition[Condition("learned", bit, seed).condition_id][task]
                for seed in SEEDS
            ]
            low, high = paired_task_bootstrap_ci(
                [stacked_task_arrays[task]],
                samples=args.bootstrap_samples,
                seed=20260814 + bit * 100 + TASKS.index(task),
            )
            task_paired_rows.append(
                {
                    "bit_width": bit,
                    "task": task,
                    "category": TASK_TO_CATEGORY[task],
                    "random_mean": statistics.fmean(random_values),
                    "learned_mean": statistics.fmean(learned_values),
                    "difference": statistics.fmean(learned_values)
                    - statistics.fmean(random_values),
                    "confidence_interval_low": low,
                    "confidence_interval_high": high,
                    "seed_wins": sum(
                        learned > random
                        for learned, random in zip(learned_values, random_values)
                    ),
                }
            )

        repeated_large_category_regression = False
        for category, category_tasks in CATEGORIES.items():
            random_values = [
                macro_average(
                    task_means_by_condition[
                        Condition("random", bit, seed).condition_id
                    ],
                    category_tasks,
                )
                for seed in SEEDS
            ]
            learned_values = [
                macro_average(
                    task_means_by_condition[
                        Condition("learned", bit, seed).condition_id
                    ],
                    category_tasks,
                )
                for seed in SEEDS
            ]
            category_difference = statistics.fmean(learned_values) - statistics.fmean(
                random_values
            )
            if (
                all(value < 0 for value in per_category_seed_differences[category])
                and abs(category_difference) > max(overall_difference, 0.0)
            ):
                repeated_large_category_regression = True
            category_rows.append(
                {
                    "bit_width": bit,
                    "category": category,
                    "fp16": macro_average(fp16_tasks, category_tasks),
                    "identity": macro_average(identity_tasks, category_tasks),
                    "random_mean": statistics.fmean(random_values),
                    "learned_mean": statistics.fmean(learned_values),
                    "learned_minus_random": category_difference,
                }
            )

        bit_pairs = [row for row in paired_rows if int(row["bit_width"]) == bit]
        long_context_differences = [
            float(row["difference"])
            for row in bit_pairs
            if row["length_bucket"] == "8k+"
        ]
        long_context_difference = statistics.fmean(long_context_differences)
        positive_task_count = sum(
            float(row["difference"]) > 0
            for row in task_paired_rows
            if int(row["bit_width"]) == bit
        )
        positive_category_count = sum(
            float(row["learned_minus_random"]) > 0
            for row in category_rows
            if int(row["bit_width"]) == bit
        )
        verdict = verdict_for_bit(
            overall_difference,
            seed_differences,
            overall_low,
            repeated_large_category_regression,
            long_context_difference,
        )
        wins = sum(float(row["difference"]) > 0 for row in bit_pairs)
        ties = sum(float(row["difference"]) == 0 for row in bit_pairs)
        losses = len(bit_pairs) - wins - ties
        overall_rows.append(
            {
                "bit_width": bit,
                "fp16": fp16_overall,
                "identity": macro_average(identity_tasks),
                "random_mean": random_mean,
                "learned_mean": learned_mean,
                "learned_minus_random": overall_difference,
                "confidence_interval_low": overall_low,
                "confidence_interval_high": overall_high,
                "seed_wins": sum(value > 0 for value in seed_differences),
                "positive_tasks": positive_task_count,
                "positive_categories": positive_category_count,
                "long_8k_learned_minus_random": long_context_difference,
                "example_wins": wins,
                "example_ties": ties,
                "example_losses": losses,
                "verdict": verdict,
            }
        )

    combined_predictions: list[dict[str, Any]] = []
    combined_scores: list[dict[str, Any]] = []
    combined_task_rows: list[dict[str, Any]] = []
    combined_category_rows: list[dict[str, Any]] = []
    combined_length_rows: list[dict[str, Any]] = []
    for condition in all_conditions():
        condition_id = condition.condition_id
        run_dir = run_directory(args.output_dir, "full", condition)
        combined_predictions.extend(predictions_by_condition[condition_id])
        combined_scores.extend(scores_by_condition[condition_id])
        combined_task_rows.extend(read_csv(run_dir / "task_summary.csv"))
        combined_category_rows.extend(read_csv(run_dir / "category_summary.csv"))
        combined_length_rows.extend(read_csv(run_dir / "length_summary.csv"))
    write_jsonl(args.output_dir / "predictions.jsonl", combined_predictions)
    write_csv(args.output_dir / "scores.csv", combined_scores)
    write_csv(args.output_dir / "task_summary.csv", combined_task_rows)
    write_csv(args.output_dir / "category_summary.csv", combined_category_rows)
    write_csv(args.output_dir / "length_summary.csv", combined_length_rows)

    write_csv(args.output_dir / "paired_comparison.csv", paired_rows)
    write_csv(args.output_dir / "seed_summary.csv", seed_rows)
    write_csv(args.output_dir / "overall_summary.csv", overall_rows)
    write_csv(args.output_dir / "category_comparison.csv", category_rows)
    write_csv(args.output_dir / "task_paired_summary.csv", task_paired_rows)

    for bit in BITS:
        for seed in SEEDS:
            rows = [
                row
                for row in paired_rows
                if int(row["bit_width"]) == bit and int(row["seed"]) == seed
            ]
            for method in ("random", "learned"):
                run_dir = run_directory(
                    args.output_dir, "full", Condition(method, bit, seed)
                )
                write_csv(run_dir / "paired_comparison.csv", rows)

    length_paired_rows: list[dict[str, Any]] = []
    for bit in BITS:
        for bucket in ("0-4k", "4-8k", "8k+"):
            for method in ("random", "learned"):
                values = [
                    numeric(row[f"{method}_score"])
                    for row in paired_rows
                    if int(row["bit_width"]) == bit
                    and row["length_bucket"] == bucket
                ]
                length_paired_rows.append(
                    {
                        "bit_width": bit,
                        "method": method,
                        "length_bucket": bucket,
                        "paired_examples_across_seeds": len(values),
                        "mean_score": statistics.fmean(values) if values else math.nan,
                    }
                )
    write_csv(args.output_dir / "length_paired_summary.csv", length_paired_rows)

    system_summary_rows: list[dict[str, Any]] = []
    for condition in all_conditions():
        rows = predictions_by_condition[condition.condition_id]
        generated_tokens = sum(int(row["generated_tokens"]) for row in rows)
        decode_seconds = sum(float(row["decode_seconds"]) for row in rows)
        system_summary_rows.append(
            {
                "condition_id": condition.condition_id,
                "method": condition.method,
                "bit_width": condition.bit_width,
                "seed": condition.seed,
                "examples": len(rows),
                "prompt_tokens": sum(int(row["prompt_tokens"]) for row in rows),
                "generated_tokens": generated_tokens,
                "prefill_seconds": sum(float(row["prefill_seconds"]) for row in rows),
                "decode_seconds": decode_seconds,
                "decode_seconds_per_token": decode_seconds
                / max(generated_tokens - len(rows), 1),
                "total_seconds": sum(float(row["total_seconds"]) for row in rows),
                "peak_gpu_memory_bytes": max(
                    int(row["peak_gpu_memory_bytes"]) for row in rows
                ),
                "theoretical_kv_bytes_per_token": int(
                    rows[0]["theoretical_kv_bytes_per_token"]
                ),
                "empty_outputs": sum(not str(row["prediction"]).strip() for row in rows),
            }
        )
    write_csv(args.output_dir / "system_metrics.csv", system_summary_rows)
    write_csv(args.output_dir / "system_summary.csv", system_summary_rows)
    write_plots(args.output_dir, overall_rows, category_rows, seed_rows)

    overall_table = markdown_table(
        (
            "Key bits",
            "FP16",
            "Identity",
            "Random mean",
            "Learned mean",
            "Learned vs Random",
            "95% CI",
            "Seed wins",
            "8k+ difference",
            "Verdict",
        ),
        [
            (
                str(row["bit_width"]),
                f"{row['fp16']:.3f}",
                f"{row['identity']:.3f}",
                f"{row['random_mean']:.3f}",
                f"{row['learned_mean']:.3f}",
                f"{row['learned_minus_random']:+.3f}",
                f"[{row['confidence_interval_low']:+.3f}, {row['confidence_interval_high']:+.3f}]",
                f"{row['seed_wins']}/3",
                f"{row['long_8k_learned_minus_random']:+.3f}",
                str(row["verdict"]),
            )
            for row in overall_rows
        ],
    )
    category_table = markdown_table(
        ("Key bits", "Method", *CATEGORIES.keys(), "Average"),
        [
            (
                str(bit),
                method,
                *(
                    f"{next(float(row[field]) for row in category_rows if int(row['bit_width']) == bit and row['category'] == category):.3f}"
                    for category in CATEGORIES
                ),
                f"{next(float(row[method]) for row in overall_rows if int(row['bit_width']) == bit):.3f}",
            )
            for bit in BITS
            for method, field in (
                ("fp16", "fp16"),
                ("identity", "identity"),
                ("random_mean", "random_mean"),
                ("learned_mean", "learned_mean"),
            )
        ],
    )
    task_table = markdown_table(
        ("Key bits", "Task", "Random mean", "Learned mean", "Difference", "95% CI", "Wins"),
        [
            (
                str(row["bit_width"]),
                str(row["task"]),
                f"{row['random_mean']:.3f}",
                f"{row['learned_mean']:.3f}",
                f"{row['difference']:+.3f}",
                f"[{row['confidence_interval_low']:+.3f}, {row['confidence_interval_high']:+.3f}]",
                f"{row['seed_wins']}/3",
            )
            for row in task_paired_rows
        ],
    )
    report = "\n".join(
        [
            "# Head-wise learned rotation: LongBench-E 195-example subset",
            "",
            f"Generated: {utc_now()}",
            "",
            "## Overall comparison",
            "",
            overall_table,
            "",
            "The 95% confidence intervals use paired, task-stratified example "
            f"bootstrap with {args.bootstrap_samples:,} resamples. Each bit width "
            "is judged independently.",
            "",
            "## Category comparison",
            "",
            category_table,
            "",
            "![Category paired differences](category_paired_differences.png)",
            "",
            "## Per-task paired comparison",
            "",
            task_table,
            "",
            "## Protocol notes",
            "",
            f"- Model revision: `{assets['model']['revision']}`.",
            f"- LongBench commit: `{assets['longbench']['commit']}`.",
            f"- LongBench dataset revision: `{assets['longbench']['dataset_revision']}`.",
            f"- Subset seed: `{manifest['sampling_seed']}`; exactly 5 examples per "
            "task and length interval.",
            f"- Evaluated examples per condition: {expected_count:,} across 13 tasks.",
            "- Precision labels are K16/V16, K2/V16, K3/V16, and K4/V16. Values "
            "remain BF16 in every condition.",
            "- Quantized conditions emulate packed key-cache quality by "
            "quantizing and reconstructing every pre-RoPE key before BF16 cache "
            "storage. Values remain BF16. Reported KV bytes/token are theoretical; "
            "measured GPU peaks reflect the unfused emulation.",
            "- LongBench-E was not used for rotation training, model selection, or "
            "hyperparameter selection.",
            "",
            "![Overall comparison](overall_comparison.png)",
            "",
            "![Seed paired differences](seed_paired_differences.png)",
        ]
    )
    (args.output_dir / "report.md").write_text(report + "\n")
    summary = {
        "generated_at": utc_now(),
        "expected_examples_per_condition": expected_count,
        "conditions": len(all_conditions()),
        "overall": overall_rows,
        "all_conditions_complete": True,
    }
    write_json(args.output_dir / "summary.json", summary)
    return summary


def child_command(args: argparse.Namespace, stage: str, condition: Condition | None) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "experiments.spin_turboquant.longbench",
        "--stage",
        stage,
        "--model",
        str(args.model),
        "--rotation-dir",
        str(args.rotation_dir),
        "--longbench-repo",
        str(args.longbench_repo),
        "--data-dir",
        str(args.data_dir),
        "--output-dir",
        str(args.output_dir),
        "--device",
        str(args.device),
        "--bootstrap-samples",
        str(args.bootstrap_samples),
        "--batch-size",
        str(args.batch_size),
        "--batch-token-budget",
        str(args.batch_token_budget),
        "--dataset-revision",
        str(args.dataset_revision),
        "--longbench-commit",
        str(args.longbench_commit),
    ]
    if args.max_context_length is not None:
        command.extend(["--max-context-length", str(args.max_context_length)])
    if condition is not None:
        command.extend(["--condition", condition.condition_id])
    return command


def run_child(args: argparse.Namespace, stage: str, condition: Condition | None) -> None:
    command = child_command(args, stage, condition)
    log_path = args.output_dir / "orchestrator.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as log:
        header = f"[{utc_now()}] $ {' '.join(command)}\n"
        log.write(header)
        log.flush()
        print(header, end="", flush=True)
        process = subprocess.Popen(
            command,
            cwd=Path.cwd(),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            log.write(line)
            log.flush()
        return_code = process.wait()
    if return_code != 0:
        raise subprocess.CalledProcessError(return_code, command)


def write_study_config(
    args: argparse.Namespace, assets: dict[str, Any], status: str, **updates: Any
) -> None:
    manifest = ensure_subset_manifest(
        args.output_dir,
        args.data_dir,
        dataset_revision=args.dataset_revision,
        data_hashes=assets["longbench"]["data_hashes"],
    )
    path = args.output_dir / "study_config.json"
    existing = read_json(path, {})
    payload = {
        "created_at": existing.get("created_at", utc_now()),
        "updated_at": utc_now(),
        "status": status,
        "specification": assets["specification"],
        "subset_manifest": {
            "path": str((args.output_dir / "subset_manifest.json").resolve()),
            "sha256": sha256_json(manifest),
            "sampling_seed": int(manifest["sampling_seed"]),
            "example_count": int(manifest["example_count"]),
            "samples_per_task": int(manifest["samples_per_task"]),
            "samples_per_length_bucket": int(
                manifest["samples_per_length_bucket"]
            ),
        },
        "model": assets["model"],
        "rotation_artifact_hashes": assets["rotation_artifact_hashes"],
        "codebook_hashes": assets["codebook_hashes"],
        "implementation_hashes": assets["implementation_hashes"],
        "longbench": assets["longbench"],
        "conditions": [condition.condition_id for condition in all_conditions()],
        "condition_matrix": [
            {
                "condition_id": condition.condition_id,
                "method": condition.method,
                "key_bit_width": (
                    16 if condition.method == "fp16" else condition.bit_width
                ),
                "value_bit_width": 16,
                "seed": condition.seed,
            }
            for condition in all_conditions()
        ],
        "condition_count": len(all_conditions()),
        "execution_order": (
            "all 22 condition smoke tests first; then FP16 and, for each key "
            "width, Identity followed by adjacent Random/Learned seed pairs"
        ),
        "batch_size": args.batch_size,
        "batch_token_budget": args.batch_token_budget,
        "bootstrap_samples": args.bootstrap_samples,
        **updates,
    }
    write_json(path, payload)


def orchestrate(args: argparse.Namespace) -> None:
    assets = validate_assets(args, deep=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_study_config(args, assets, "running", started_at=utc_now())
    try:
        for condition in all_conditions():
            run_child(args, "smoke", condition)
        for condition in all_conditions():
            run_child(args, "full", condition)
        run_child(args, "report", None)
    except BaseException as error:
        write_study_config(
            args,
            assets,
            "failed",
            error=f"{type(error).__name__}: {error}",
        )
        raise
    write_study_config(args, assets, "complete", completed_at=utc_now())


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    args.model = args.model.resolve()
    args.rotation_dir = args.rotation_dir.resolve()
    args.longbench_repo = args.longbench_repo.resolve()
    args.data_dir = args.data_dir.resolve()
    args.output_dir = args.output_dir.resolve()
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.set_float32_matmul_precision("highest")

    if args.stage == "validate":
        assets = validate_assets(args, deep=True)
        args.output_dir.mkdir(parents=True, exist_ok=True)
        manifest = ensure_subset_manifest(
            args.output_dir,
            args.data_dir,
            dataset_revision=args.dataset_revision,
            data_hashes=assets["longbench"]["data_hashes"],
        )
        write_study_config(args, assets, "validated")
        print(
            json.dumps(
                {
                    "model_revision": assets["model"]["revision"],
                    "longbench_commit": assets["longbench"]["commit"],
                    "dataset_revision": assets["longbench"]["dataset_revision"],
                    "examples_per_condition": int(manifest["example_count"]),
                    "conditions": len(all_conditions()),
                },
                indent=2,
            ),
            flush=True,
        )
    elif args.stage in {"smoke", "full"}:
        run_inference(args, args.stage)
    elif args.stage == "report":
        summary = analyze_full_study(args)
        assets = validate_assets(args, deep=False)
        write_study_config(args, assets, "complete", summary=summary)
        print(f"[{utc_now()}] final LongBench-E report complete", flush=True)
    elif args.stage == "orchestrate":
        orchestrate(args)
    else:
        raise AssertionError(args.stage)


if __name__ == "__main__":
    main()
