import math

import torch

from experiments.spin_turboquant.bit2_scheduler_comparison import (
    EXPONENTIAL_GAMMA,
    FINAL_LEARNING_RATE,
    LEARNING_RATE,
    SEED,
    scheduler_object,
    select_step,
)


def test_fixed_seed_is_35() -> None:
    assert SEED == 35


def test_both_schedulers_reach_the_exact_fixed_endpoint() -> None:
    for name in ("cosine", "exponential"):
        parameter = torch.nn.Parameter(torch.zeros(()))
        optimizer = torch.optim.Adam([parameter], lr=LEARNING_RATE)
        scheduler = scheduler_object(optimizer, name, 10_000)
        for _ in range(10_000):
            optimizer.step()
            scheduler.step()
        assert math.isclose(
            optimizer.param_groups[0]["lr"],
            FINAL_LEARNING_RATE,
            rel_tol=0,
            abs_tol=1e-10,
        )


def test_exponential_gamma_matches_documented_formula() -> None:
    expected = (FINAL_LEARNING_RATE / LEARNING_RATE) ** (1 / 10_000)
    assert EXPONENTIAL_GAMMA == expected


def test_selection_uses_earliest_checkpoint_within_point_one_percent() -> None:
    rows = [
        {
            "step": "100",
            "wikitext_validation_mse": "1.0009",
            "rotation_tensor_sha256": "early",
        },
        {
            "step": "200",
            "wikitext_validation_mse": "1.0",
            "rotation_tensor_sha256": "minimum",
        },
    ]
    decision = select_step(rows)
    assert decision["minimum_step"] == 200
    assert decision["selected_step"] == 100
    assert decision["checkpoint_rotation_tensor_sha256"] == "early"
