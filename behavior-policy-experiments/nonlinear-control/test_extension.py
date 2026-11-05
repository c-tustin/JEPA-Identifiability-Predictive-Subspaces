"""Independent checks of the matched-query and nonlinear-control protocol."""

import unittest
import torch
from run_coverage import reset_queries, fingerprint
from run_trajectories import Settings
from evaluate_control import make_goals, nominal_step, simulation_noise, base
from check_drift_sampling import deterministic_effect, noisy_contrasts


class CoverageChecks(unittest.TestCase):
    def test_sources_and_actions_match_across_worlds_but_successors_change(self):
        settings = Settings(episodes=16, horizon=8, substeps=5)
        zero = reset_queries(settings, 23, False)
        nonlinear = reset_queries(settings, 23, True)
        torch.testing.assert_close(zero["x"], nonlinear["x"], atol=0, rtol=0)
        torch.testing.assert_close(zero["action"], nonlinear["action"], atol=0, rtol=0)
        self.assertFalse(torch.equal(zero["xp"], nonlinear["xp"]))
        self.assertEqual(set(zero), {"x", "xp", "action"})

    def test_noise_is_independent_per_query_and_reproducible(self):
        settings = Settings(episodes=16, horizon=8, substeps=5)
        first = reset_queries(settings, 23, True)
        self.assertEqual(
            fingerprint(first), fingerprint(reset_queries(settings, 23, True))
        )
        self.assertNotEqual(
            fingerprint(first), fingerprint(reset_queries(settings, 24, True))
        )


class GoalChecks(unittest.TestCase):
    def test_goals_require_opposing_drift_and_have_a_candidate_witness(self):
        settings = Settings()
        for seed in [23, 24, 25]:
            start, goal, action = make_goals(settings, seed, 32)
            self.assertTrue(bool(((0.3 * torch.sin(start)).abs() <= 0.1).all()))
            self.assertTrue(bool(((action - 1) * torch.sin(start) <= 0).all()))
            self.assertTrue(bool(((goal - start).abs() >= 0.2).all()))
            replay = nominal_step(
                nominal_step(start, action, settings), action, settings
            )
            torch.testing.assert_close(replay, goal, atol=0, rtol=0)
            bank = base.candidate_bank(seed)
            for a in [0, 2]:
                witness = torch.ones(8, dtype=torch.long)
                witness[:2] = a
                self.assertTrue(bool((bank == witness).all(1).any()))

    def test_execution_noise_is_common_across_controllers(self):
        settings = Settings()
        a = simulation_noise(settings, 23, 32, True)
        torch.testing.assert_close(
            a, simulation_noise(settings, 23, 32, True), atol=0, rtol=0
        )
        self.assertGreater(float(a.std()), 0.9)
        self.assertEqual(
            float(simulation_noise(settings, 23, 32, False).abs().max()), 0
        )


class DriftSamplingChecks(unittest.TestCase):
    def test_finite_time_action_bias_has_the_derived_leading_term(self):
        distance = torch.tensor([2.0, 5.0, 9.0, 13.0], dtype=torch.float64)
        interval = 1e-4
        slope = (deterministic_effect(distance, interval) - 0.25) / interval
        torch.testing.assert_close(
            slope, 0.25 * 0.3 * torch.cos(distance) / 2, atol=3e-6, rtol=0
        )

    def test_independent_euler_contrast_variance_matches_formula(self):
        distance = torch.tensor([5.0], dtype=torch.float64)
        interval = 0.002
        for samples in [16, 64]:
            estimates = noisy_contrasts(
                distance, interval, groups=512, samples=samples, substeps=1
            )
            variance = estimates.var(-1, unbiased=True)
            expected = (0.05 * (1 + 0.1 * torch.cos(distance))).square() / (
                2 * samples * interval
            )
            ratio = float((variance / expected).item())
            self.assertGreater(ratio, 0.75)
            self.assertLess(ratio, 1.25)


if __name__ == "__main__":
    unittest.main()
