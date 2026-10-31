"""Guard pairing, subgroup selection and coverage interventions."""

import numpy as np
import pytest
import torch

from run_trials import analysis_bins, behavior, replay_pairs, resampled_region
from trial_stats import curriculum_comparisons, holm_adjust, paired_summary


def test_sign_test_counts_seeds_and_excludes_numeric_ties():
    result = paired_summary(np.ones(10), np.zeros(10))
    assert result["n"] == 10
    assert result["wins"] == 0 and result["losses"] == 10
    assert result["sign_test_p"] == pytest.approx(2 / 2**10)
    tied = paired_summary(np.ones(10) + 1e-8, np.ones(10))
    assert tied["ties"] == 10 and tied["sign_test_p"] == 1


def test_screen_does_not_select_heldout_winners():
    rows = []
    for seed in range(10):
        rows += [
            dict(seed=seed, arm="ou", folded=False, screen_error=1.0, test_error=1.0),
            dict(
                seed=seed,
                arm="icm_to_ou",
                folded=seed == 0,
                screen_error=0.5 if seed < 5 else 2.0,
                test_error=3.0,
            ),
        ]
    result = curriculum_comparisons(rows)
    assert result["screen_selected_seeds"] == [1, 2, 3, 4]
    assert result["comparisons"][1]["wins"] == 0
    for row in rows:
        if row["arm"] == "icm_to_ou":
            row["test_error"] = 0.0
    changed = curriculum_comparisons(rows)
    assert changed["screen_selected_seeds"] == result["screen_selected_seeds"]


def test_holm_adjustment_is_monotone_and_never_smaller():
    raw = [0.04, 0.001, 0.02]
    adjusted = holm_adjust(raw)
    assert adjusted == pytest.approx([0.04, 0.003, 0.04])
    assert np.all(np.asarray(adjusted) >= raw)


def test_replay_preserves_pairs_and_balances_occupied_bins():
    heights = torch.tensor(
        [np.exp(-0.05)] * 1000 + [np.exp(-1.4)] * 100 + [np.exp(-2.7)] * 100
    )
    x = torch.zeros(len(heights), 6)
    x[:, 1], x[:, 0] = heights, torch.arange(len(x))
    xp = x.clone()
    xp[:, 0] += 10000
    result = replay_pairs(dict(x=x, xp=xp), 999, "log")
    assert torch.equal(
        result["xp"][:, 0] - result["x"][:, 0], torch.full((len(x),), 10000.0)
    )
    counts = torch.stack(
        [(result["x"][:, 1] == value).sum() for value in heights.unique()]
    )
    np.testing.assert_allclose(counts.numpy() / len(x), 1 / 3, atol=0.05)
    assert len(result["x"]) == len(x)


def test_region_removal_keeps_budget_and_action_labels():
    c = behavior.Config()
    distance = torch.linspace(1, 16, 1200)
    x = behavior.sensor(distance, c)
    data = dict(x=x, xp=x.clone(), action=torch.arange(len(x)) % 3)
    for region in range(8):
        changed = resampled_region(data, 999, region)
        ix, valid = analysis_bins(changed["x"])
        assert bool(valid.all())
        assert not bool((ix // 4 == region).any())
        assert len(changed["x"]) == len(x)
        assert bool(((changed["action"] >= 0) & (changed["action"] <= 2)).all())
