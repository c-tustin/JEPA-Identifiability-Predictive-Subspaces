"""The follow-up changes controllers, not the physical environment."""

from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import broad_policies
import targeted_policies
import zoo
import torch


class BroadChecks(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def test_regeneration_and_feasible_actions(self):
        c = zoo.behavior.Config(burnin=128, horizon=24)
        for name, factory in broad_policies.POLICIES.items():
            a = zoo.collect_physical(c, factory(), 16, 123)
            b = zoo.collect_physical(c, factory(), 16, 123)
            for key in a:
                torch.testing.assert_close(a[key], b[key], atol=0, rtol=0)
            d = 4 * (c.sigma * a["z"]).exp()
            proposal = d + (a["action"].float() - 1) * .25
            self.assertTrue(bool((proposal >= 1 - 2e-5).all()), name)
            self.assertTrue(bool((proposal <= 16 + 2e-5).all()), name)

    def test_no_escort_changes_only_the_initial_action_phase(self):
        torch.manual_seed(0)
        previous = targeted_policies.GaussianMetropolis(.65, True)
        previous.reset(16)
        torch.manual_seed(0)
        new = broad_policies.NoEscortLogDiffusion()
        new.reset(16)
        torch.testing.assert_close(previous.target, new.target, atol=0, rtol=0)
        self.assertEqual(previous.steps, 0)
        self.assertEqual(new.steps, 64)
        view = zoo.behavior.sensor(torch.linspace(1, 16, 16), zoo.behavior.Config())
        torch.manual_seed(1)
        a = previous.after_escort(view)
        torch.manual_seed(1)
        b = new.after_escort(view)
        torch.testing.assert_close(a, b, atol=0, rtol=0)

    def test_sweep_direction_turns_at_the_edges(self):
        policy = broad_policies.GuardedSweep()
        policy.reset(2)
        view = zoo.behavior.sensor(torch.tensor([1.01, 15.99]), zoo.behavior.Config())
        for _ in range(20):
            policy.act(view)
            torch.testing.assert_close(policy.direction, torch.tensor([1, -1]))


if __name__ == "__main__":
    unittest.main()
