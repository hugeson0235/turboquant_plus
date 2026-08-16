"""Protocol tests for the independent learned-rotation length sweep."""

from argparse import Namespace

import torch

from experiments.spin_turboquant.training_length_sweep import (
    CHECKPOINT_FIELDS,
    TRAINING_CURVE_FIELDS,
    attach_sanity_summary,
    execute_training_run,
    optimizer_seed,
    select_horizons,
    tensor_sha256,
)


def synthetic_args() -> Namespace:
    return Namespace(
        learning_rate=0.005,
        minimum_learning_rate=0.00025,
        batch_tokens=8,
        metric_interval=10,
        ema_coefficient=0.95,
        bits=[2, 3, 4],
        seeds=[17, 29, 43],
        horizons=list(range(100, 1001, 100)),
    )


def test_optimizer_seed_is_fixed_by_bit_and_seed_not_horizon():
    assert optimizer_seed(2, 17) == 200_017
    assert optimizer_seed(2, 17) == optimizer_seed(2, 17)
    assert optimizer_seed(2, 17) != optimizer_seed(3, 17)
    assert optimizer_seed(2, 17) != optimizer_seed(2, 29)


def test_tensor_hash_includes_shape_dtype_and_values():
    value = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    assert tensor_sha256(value) == tensor_sha256(value.clone())
    assert tensor_sha256(value) != tensor_sha256(value.reshape(4, 3))
    assert tensor_sha256(value) != tensor_sha256(value.double())
    changed = value.clone()
    changed[0, 0] = -1
    assert tensor_sha256(value) != tensor_sha256(changed)


def test_training_run_records_step_zero_and_exact_terminal_schedule():
    args = synthetic_args()
    generator = torch.Generator().manual_seed(9)
    # The real run has 32 layers. One synthetic KV head per layer keeps the
    # runner's layer/head row contract while making this a small CPU test.
    calibration = torch.randn((32, 24, 4), generator=generator)
    validation = torch.randn((32, 24, 4), generator=generator)
    calibration_norms = torch.linalg.vector_norm(calibration, dim=-1, keepdim=True)
    validation_norms = torch.linalg.vector_norm(validation, dim=-1, keepdim=True)
    calibration = calibration / calibration_norms
    validation = validation / validation_norms
    initial = torch.eye(4).expand(32, 4, 4).clone()
    centroids = torch.tensor([-0.5, 0.5])

    result = execute_training_run(
        args,
        bit_width=2,
        seed=17,
        horizon=20,
        calibration_normalized=calibration,
        calibration_norm_squared=calibration_norms.square(),
        validation_normalized=validation,
        validation_norm_squared=validation_norms.square(),
        initial=initial,
        centroids=centroids,
        record_curves=True,
    )

    assert [row["step"] for row in result["training_rows"]] == [0, 10, 20]
    assert [row["step"] for row in result["checkpoint_rows"]] == [0, 10, 20]
    assert set(TRAINING_CURVE_FIELDS).issubset(result["training_rows"][-1])
    assert set(CHECKPOINT_FIELDS).issubset(result["checkpoint_rows"][-1])
    assert result["training_rows"][0]["learning_rate"] == 0.005
    assert abs(result["training_rows"][-1]["learning_rate"] - 0.00025) < 1e-12
    assert len(result["head_rows"]) == 3 * 2 * 32
    assert result["terminal"]["step"] == 20
    assert result["terminal"]["horizon_steps"] == 20
    assert result["terminal"]["orthogonality_max_abs"] < 1e-5
    assert result["rotation_tensor_sha256"] == tensor_sha256(result["rotations"])
    assert len(result["first_100_minibatch_indices_sha256"]) == 64


def test_selection_uses_seed_mean_tolerance_then_adoption_threshold():
    args = synthetic_args()
    terminal_rows = []
    reference_rows = []
    for bit_width in args.bits:
        reference_rows.extend(
            {
                "bit_width": bit_width,
                "seed": seed,
                "wikitext_validation_mse": 1.0,
            }
            for seed in args.seeds
        )
        for horizon in args.horizons:
            # The exact minimum is H=300 at 0.99. H=200 is within 0.1% of the
            # minimum and therefore wins the shorter-horizon tie rule.
            mean = 0.9905 if horizon == 200 else 0.99 if horizon == 300 else 1.02
            for seed, offset in zip(args.seeds, (-0.001, 0.0, 0.001)):
                terminal_rows.append(
                    {
                        "bit_width": str(bit_width),
                        "seed": str(seed),
                        "horizon_steps": str(horizon),
                        "wikitext_validation_mse": str(mean + offset),
                    }
                )

    selected = select_horizons(args, terminal_rows, reference_rows)
    for bit_width in args.bits:
        decision = selected["bits"][str(bit_width)]
        assert decision["minimum_horizon_steps"] == 300
        assert decision["sweep_candidate_horizon_steps"] == 200
        assert decision["selected_horizon_steps"] == 200
        assert decision["longer_training_adopted"] is True
        assert decision["search_boundary_reached"] is False


def test_selection_retains_80_when_gain_is_below_half_percent():
    args = synthetic_args()
    terminal_rows = []
    reference_rows = []
    for bit_width in args.bits:
        reference_rows.extend(
            {
                "bit_width": bit_width,
                "seed": seed,
                "wikitext_validation_mse": 1.0,
            }
            for seed in args.seeds
        )
        for horizon in args.horizons:
            mean = 0.996 if horizon == 400 else 1.01
            for seed in args.seeds:
                terminal_rows.append(
                    {
                        "bit_width": bit_width,
                        "seed": seed,
                        "horizon_steps": horizon,
                        "wikitext_validation_mse": mean,
                    }
                )
    selected = select_horizons(args, terminal_rows, reference_rows)
    assert all(
        item["selected_horizon_steps"] == 80
        and item["longer_training_adopted"] is False
        for item in selected["bits"].values()
    )


def test_resumed_sanity_rows_reattach_tinystories_summary():
    args = synthetic_args()
    selection = {"bits": {str(bit): {} for bit in args.bits}}
    rows = []
    for bit in args.bits:
        for seed in args.seeds:
            rows.extend(
                (
                    {
                        "bit_width": str(bit),
                        "seed": str(seed),
                        "condition": "existing_80_step",
                        "normalized_key_mse": "1.0",
                    },
                    {
                        "bit_width": str(bit),
                        "seed": str(seed),
                        "condition": "selected_horizon",
                        "normalized_key_mse": "0.9",
                    },
                )
            )

    attach_sanity_summary(args, selection, rows)

    for bit in args.bits:
        summary = selection["bits"][str(bit)]["tinystories"]
        assert abs(summary["relative_change_selected_vs_80"] + 0.1) < 1e-12
        assert summary["domain_overfitting_warning"] is False
