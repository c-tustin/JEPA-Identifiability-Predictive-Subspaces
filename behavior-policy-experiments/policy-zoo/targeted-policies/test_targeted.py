"""Check physical feasibility, regeneration and the Metropolis proposal ratio."""

from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import targeted_policies as policies
import zoo
import torch


class TargetedChecks(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def test_metropolis_detailed_balance(self):
        distance = torch.linspace(1.05, 15.7, 200, dtype=torch.float64)
        proposed = distance + 0.25
        for deviation, scaled in ((0.5, False), (0.5, True), (0.65, True)):
            policy = policies.GaussianMetropolis(deviation, scaled)
            forward = 0.5 * policy.activity(distance) * policy.log_acceptance(distance, proposed).clamp(max=0).exp()
            reverse = 0.5 * policy.activity(proposed) * policy.log_acceptance(proposed, distance).clamp(max=0).exp()
            left = policy.log_density(distance).exp() * forward
            right = policy.log_density(proposed).exp() * reverse
            torch.testing.assert_close(left, right, atol=1e-13, rtol=1e-12)

    def test_targets_respect_declared_bounds(self):
        torch.manual_seed(0)
        for deviation in (0.5, 0.55, 0.65):
            q = policies.truncated_log_targets(10000, deviation)
            self.assertTrue(bool((q >= 0).all()))
            self.assertTrue(bool((q <= torch.tensor(16.).log()).all()))

    def test_all_actions_stay_feasible_and_regenerate(self):
        c = zoo.behavior.Config(burnin=128, horizon=24, train_episodes=16)
        for name, factory in policies.POLICIES.items():
            first = zoo.collect_physical(c, factory(), 16, 123)
            second = zoo.collect_physical(c, factory(), 16, 123)
            for key in first:
                torch.testing.assert_close(first[key], second[key], atol=0, rtol=0)
            self.assertEqual(first["x"].shape, (16 * 24, 6))
            distance = 4 * (c.sigma * first["z"]).exp()
            proposed = distance + (first["action"].float() - 1) * c.move
            self.assertTrue(bool((proposed >= 1 - 2e-5).all()), name)
            self.assertTrue(bool((proposed <= 16 + 2e-5).all()), name)
            self.assertTrue(bool(torch.isfinite(first["x"]).all()), name)

    def test_policy_can_be_called_with_views_only(self):
        view = zoo.behavior.sensor(torch.tensor([1.01, 4., 15.99]), zoo.behavior.Config())
        for factory in policies.POLICIES.values():
            policy = factory()
            policy.reset(3)
            for _ in range(70):
                action = policy.act(view)
                self.assertEqual(action.shape, (3,))
                self.assertTrue(bool(((action >= 0) & (action <= 2)).all()))


if __name__ == "__main__":
    unittest.main()
