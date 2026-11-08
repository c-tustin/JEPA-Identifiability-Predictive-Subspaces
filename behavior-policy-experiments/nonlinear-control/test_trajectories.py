"""Checks for the new simulator and observation-only sampling protocol."""

import unittest
import numpy as np
import torch
from run_trajectories import (
    Settings,
    advance,
    collect,
    reflect,
    sampling_weights,
    size_bin,
    largest_reversal_fraction,
)


class TrajectoryChecks(unittest.TestCase):
    def test_reversal_size_uses_cumulative_drop_and_is_scale_invariant(self):
        y = np.array([0.0, 2.0, 1.5, 1.0, 3.0])
        self.assertAlmostEqual(largest_reversal_fraction(y), 1 / 3)
        self.assertAlmostEqual(largest_reversal_fraction(7 * y + 8), 1 / 3)
        self.assertEqual(largest_reversal_fraction(np.ones(5)), 0)
        self.assertEqual(largest_reversal_fraction(np.arange(5)), 0)

    def test_reflection_handles_multiple_crossings(self):
        d = torch.tensor([-30.0, 0.0, 1.0, 16.0, 17.0, 47.0])
        torch.testing.assert_close(
            reflect(d), torch.tensor([2.0, 2.0, 1.0, 16.0, 15.0, 15.0])
        )

    def test_drift_changes_only_the_specified_term_interior(self):
        d = torch.linspace(2, 15, 30, dtype=torch.float64)
        action = torch.ones_like(d)
        noise = torch.linspace(-1, 1, 30, dtype=torch.float64)
        difference = advance(d, action, noise, True, 0.02) - advance(
            d, action, noise, False, 0.02
        )
        torch.testing.assert_close(
            difference, 0.02 * 0.3 * torch.sin(d), atol=1e-14, rtol=0
        )

    def test_collection_is_reproducible_and_has_no_training_labels(self):
        c = Settings(episodes=8, horizon=4, burnin=2, substeps=2)
        for policy in ["random", "persistent", "action_balanced"]:
            first = collect(c, 23, policy, True)
            second = collect(c, 23, policy, True)
            self.assertEqual(set(first), {"x", "xp", "action"})
            for key in first:
                torch.testing.assert_close(first[key], second[key], atol=0, rtol=0)
            self.assertEqual(first["x"].shape, (32, 6))
            self.assertTrue(
                bool(
                    (
                        (1 / first["xp"][:, 1] >= 1)
                        & (1 / first["xp"][:, 1] <= 16.00001)
                    ).all()
                )
            )

    def test_balancing_gives_equal_total_mass_to_each_visited_source_bin(self):
        c = Settings(episodes=16, horizon=8, burnin=2, substeps=2)
        data = collect(c, 23, "random", True)
        bins = size_bin(data["x"])
        weights = sampling_weights(data)
        totals = torch.zeros(32).scatter_add_(0, bins, weights)
        torch.testing.assert_close(
            totals[totals > 0], torch.ones_like(totals[totals > 0])
        )


if __name__ == "__main__":
    unittest.main()
