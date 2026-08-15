"""Focused tests for the analytic PCA rotation experiment."""

import math

import torch

from experiments.spin_turboquant.analytic_core import (
    apply_row_rope,
    attention_query_second_moment,
    normalized_hadamard,
    normalized_key_second_moment,
    principal_subspace_similarity,
    rope_transform_covariances,
    row_rope_matrix,
    spectral_rotation,
    to_codec_rotations,
    weight_second_moment,
)
from experiments.spin_turboquant.longbench import CATEGORIES, TASKS, TASK_TO_CATEGORY
from experiments.spin_turboquant.longbench_analytic import (
    CALIBRATION_SEED,
    CONDITIONS,
    SEQUENCE_COUNT,
    SEQUENCE_LENGTH,
    finalize_diagnostic_rows,
    new_diagnostic_accumulator,
    paired_downstream_analysis,
    select_wikitext_sequences,
)


def test_spectral_rotation_is_deterministic_sign_fixed_and_orthogonal():
    covariance = torch.diag(torch.tensor([1.0, 9.0, 4.0, 2.0], dtype=torch.float64))
    values, vectors, rotation = spectral_rotation(covariance)
    torch.testing.assert_close(values, torch.tensor([9.0, 4.0, 2.0, 1.0], dtype=torch.float64))
    pivots = vectors.gather(-2, vectors.abs().argmax(dim=-2, keepdim=True)).squeeze(-2)
    assert torch.all(pivots > 0)
    torch.testing.assert_close(
        rotation.T @ rotation,
        torch.eye(4, dtype=torch.float64),
        atol=1e-12,
        rtol=1e-12,
    )
    torch.testing.assert_close(
        normalized_hadamard(4).T @ normalized_hadamard(4),
        torch.eye(4, dtype=torch.float64),
        atol=1e-12,
        rtol=1e-12,
    )


def test_weight_and_normalized_key_second_moments_follow_row_convention():
    weight = torch.arange(24, dtype=torch.float64).reshape(6, 4)
    moments = weight_second_moment(weight, num_kv_heads=2)
    assert moments.shape == (2, 3, 3)
    torch.testing.assert_close(moments[0], weight[:3] @ weight[:3].T)

    keys = torch.tensor(
        [[[3.0, 4.0], [0.0, 2.0]], [[0.0, 5.0], [2.0, 0.0]]],
        dtype=torch.float64,
    )
    actual, count = normalized_key_second_moment(keys)
    normalized = keys / torch.linalg.vector_norm(keys, dim=-1, keepdim=True)
    expected = torch.einsum("thd,the->hde", normalized, normalized)
    assert count == 2
    torch.testing.assert_close(actual, expected)


def test_fast_rope_covariance_transform_matches_dense_row_matrix():
    generator = torch.Generator().manual_seed(14)
    tokens, heads, dimension = 3, 2, 8
    angles = torch.randn((tokens, dimension // 2), generator=generator, dtype=torch.float64)
    cos = torch.cat((angles.cos(), angles.cos()), dim=-1)
    sin = torch.cat((angles.sin(), angles.sin()), dim=-1)
    base = torch.randn((tokens, heads, dimension, dimension), generator=generator, dtype=torch.float64)
    covariances = base @ base.transpose(-1, -2)
    actual = rope_transform_covariances(covariances, cos, sin)
    expected = torch.empty_like(actual)
    for token in range(tokens):
        matrix = row_rope_matrix(cos[token], sin[token])
        expected[token] = matrix @ covariances[token] @ matrix.T
    torch.testing.assert_close(actual, expected, atol=1e-11, rtol=1e-11)


def test_attention_query_moment_matches_explicit_causal_pairs():
    generator = torch.Generator().manual_seed(91)
    tokens, kv_heads, groups, dimension = 4, 2, 2, 8
    queries = torch.randn(
        (tokens, kv_heads * groups, dimension), generator=generator, dtype=torch.float64
    )
    angles = torch.randn((tokens, dimension // 2), generator=generator, dtype=torch.float64)
    cos = torch.cat((angles.cos(), angles.cos()), dim=-1)
    sin = torch.cat((angles.sin(), angles.sin()), dim=-1)
    actual, count = attention_query_second_moment(
        queries, cos, sin, num_kv_heads=kv_heads
    )
    post = apply_row_rope(queries, cos, sin).reshape(tokens, kv_heads, groups, dimension)
    expected = torch.zeros_like(actual)
    for position in range(tokens):
        matrix = row_rope_matrix(cos[position], sin[position])
        denominator = groups * (tokens - position)
        for future in range(position, tokens):
            for head in range(kv_heads):
                for group in range(groups):
                    effective = post[future, head, group] @ matrix.T
                    expected[head] += torch.outer(effective, effective) / denominator
    assert count == tokens
    torch.testing.assert_close(actual, expected, atol=1e-11, rtol=1e-11)


def test_codec_adapter_and_subspace_similarity():
    rotation = spectral_rotation(torch.eye(8, dtype=torch.float64))[2]
    vectors = torch.randn((5, 8), generator=torch.Generator().manual_seed(8))
    codec_rotation = to_codec_rotations(rotation)
    torch.testing.assert_close(vectors.double() @ codec_rotation.T, vectors.double() @ rotation)
    assert abs(principal_subspace_similarity(rotation, rotation, 4) - 1.0) < 1e-12


def test_analytic_condition_ids_and_execution_order_are_exact():
    assert [condition.condition_id for condition in CONDITIONS] == [
        "wk_pca_h_K2_V16",
        "activation_k_pca_h_K2_V16",
        "attention_q_pca_h_K2_V16",
    ]
    assert [condition.bit_width for condition in CONDITIONS] == [2, 2, 2]


def test_wikitext_document_sampling_is_deterministic_unique_and_manifested():
    class CharacterTokenizer:
        def encode(self, text, **_kwargs):
            return [ord(character) for character in text]

    rows = [""]
    for document in range(12):
        rows.append(f"= Document {document} =\n")
        rows.append(chr(65 + document) * 700 + "\n")
    first_tokens, first_manifest = select_wikitext_sequences(
        CharacterTokenizer(),
        rows,
        split="train",
        dataset_fingerprint="fingerprint",
        revision="revision",
    )
    second_tokens, second_manifest = select_wikitext_sequences(
        CharacterTokenizer(),
        rows,
        split="train",
        dataset_fingerprint="fingerprint",
        revision="revision",
    )
    torch.testing.assert_close(first_tokens, second_tokens)
    assert first_manifest == second_manifest
    assert tuple(first_tokens.shape) == (SEQUENCE_COUNT, SEQUENCE_LENGTH)
    assert first_manifest["sampling_seed"] == CALIBRATION_SEED
    assert len(
        {row["document_id"] for row in first_manifest["sequences"]}
    ) == SEQUENCE_COUNT
    assert {row["token_count"] for row in first_manifest["sequences"]} == {
        SEQUENCE_LENGTH
    }


def test_three_pairwise_downstream_comparisons_are_paired_and_task_stratified():
    method_values = {
        "wk_pca_h": 10.0,
        "activation_k_pca_h": 11.0,
        "attention_q_pca_h": 13.0,
    }
    scores = {}
    for method, value in method_values.items():
        rows = []
        for task in TASKS:
            for index in range(15):
                rows.append(
                    {
                        "condition_id": method,
                        "method": method,
                        "task": task,
                        "category": TASK_TO_CATEGORY[task],
                        "example_index": str(index),
                        "example_id": f"{task}-{index}",
                        "dataset_length": str(1000 + index),
                        "length_bucket": ("0-4k", "4-8k", "8k+")[index // 5],
                        "score": str(value),
                    }
                )
        scores[method] = rows
    paired, bootstrap, tasks, categories, lengths = paired_downstream_analysis(
        scores, bootstrap_samples=100
    )
    assert len(paired) == 3 * 195
    assert len(bootstrap) == 3
    assert len(tasks) == 3 * len(TASKS)
    assert len(categories) == 3 * len(CATEGORIES)
    assert len(lengths) == 3 * 3
    differences = {
        row["pair_id"]: row["first_minus_second_macro_average"]
        for row in bootstrap
    }
    assert differences == {
        "activation_k_pca_h_minus_wk_pca_h": 1.0,
        "attention_q_pca_h_minus_wk_pca_h": 3.0,
        "attention_q_pca_h_minus_activation_k_pca_h": 2.0,
    }
    assert all(row["confidence_interval_low"] == row["confidence_interval_high"] for row in bootstrap)


def test_diagnostic_finalization_emits_overall_and_every_head():
    assets = {
        "model": {
            "num_hidden_layers": 2,
            "num_key_value_heads": 2,
            "head_dim": 4,
        }
    }
    accumulator = new_diagnostic_accumulator(assets, "protocol")
    accumulator["token_count"] = 8
    accumulator["logit_values_per_head"] = 12
    accumulator["query_rows_per_head"] = 8
    accumulator["output_values_per_head"] = 32
    for key in (
        "normalized_key_sse",
        "norm_restored_key_sse",
        "key_energy",
        "rotated_channel_sumsq",
        "maximum_channel_magnitude",
        "attention_logit_sse",
        "attention_kl_sum",
        "attention_output_sse",
    ):
        accumulator[key].fill_(1.0)
    rows = finalize_diagnostic_rows(accumulator, assets)
    assert len(rows) == len(CONDITIONS) * (1 + 2 * 2)
    assert sum(row["scope"] == "overall" for row in rows) == len(CONDITIONS)
    numeric_fields = (
        "normalized_key_mse",
        "norm_restored_key_mse",
        "relative_key_mse",
        "attention_pre_softmax_logit_mse",
        "attention_probability_kl",
        "attention_output_mse",
    )
    assert all(math.isfinite(float(row[field])) for row in rows for field in numeric_fields)
