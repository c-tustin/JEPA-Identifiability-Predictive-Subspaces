"""Behavior-policy checks, including upstream loss semantics and reset handling."""
import unittest

import torch
from torch.nn import functional as F

from experiment import (Config, ICM, Policy, collect, discounted_returns, initial,
                        move, sensor, train_policy, verify_upstream)


class BehaviorChecks(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(12)
        torch.set_num_threads(1)

    def test_sensor_matches_inverse_distance(self):
        c = Config()
        x = sensor(torch.tensor([2., 4., 8.]), c)
        torch.testing.assert_close(x[0], 2 * x[1])
        torch.testing.assert_close(x[1], 2 * x[2])

    def test_motion_reflects_and_stays_positive(self):
        c = Config()
        d = torch.tensor([1.1, 15.9, 8.])
        result = move(d, torch.tensor([0, 2, 1]), c)
        torch.testing.assert_close(result, torch.tensor([1.15, 15.85, 8.]))
        self.assertTrue(((result >= c.distance_min) & (result <= c.distance_max)).all())

    def test_forward_error_matches_upstream_sum_and_gradients(self):
        icm = ICM()
        x, xp, a = torch.randn(8, 6), torch.randn(8, 6, requires_grad=True), torch.arange(8) % 3
        error, logits = icm(x, xp, a)
        phi, phip = icm.features(torch.tanh(4 * x)), icm.features(torch.tanh(4 * xp))
        prediction = icm.forward_model(torch.cat([phi, F.one_hot(a, 3).float()], -1))
        torch.testing.assert_close(error.mean(), .5 * (prediction - phip).square().mean() * 32)
        self.assertEqual(logits.shape, (8, 3))
        # Unlike common later variants, the official 2017 code trains phi2 too.
        target_gradient = torch.autograd.grad(error.mean(), xp, retain_graph=True)[0]
        self.assertGreater(float(target_gradient.abs().sum()), 0)
        (.8 * F.cross_entropy(logits, a) + .2 * error.mean()).backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in icm.parameters()))

    def test_discounted_returns_stop_at_reset(self):
        rewards = torch.tensor([[1.], [2.], [3.]])
        done = torch.tensor([[False], [True], [False]])
        result = discounted_returns(rewards, done, torch.tensor([4.]), .5)
        torch.testing.assert_close(result, torch.tensor([[2.], [2.], [5.]]))

    def test_frozen_policy_collection_contains_only_physical_steps(self):
        c = Config(horizon=12, burnin=8)
        policy = Policy().eval()
        data = collect(c, "icm", 5, 99, policy)
        self.assertEqual(len(data["x"]), 60)
        d = 4 * torch.exp(c.sigma * data["z"])
        dp = 4 * torch.exp(c.sigma * data["zp"])
        torch.testing.assert_close(dp, move(d, data["action"], c), rtol=1e-5, atol=1e-5)
        torch.testing.assert_close(data["x"], sensor(d, c))

    def test_policy_training_changes_weights_and_source_is_pinned(self):
        self.assertEqual(verify_upstream(), "3e220c2177fc253916f12d980957fc40579d577a")
        c = Config(policy_updates=2, rollout=4, nenv=4)
        _, _, history, change = train_policy(c, 1)
        self.assertGreater(change, 0.)
        self.assertTrue(all(h["intrinsic_reward"] >= 0 for h in history))


if __name__ == "__main__":
    unittest.main()
