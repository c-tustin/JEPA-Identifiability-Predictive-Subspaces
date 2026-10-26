import unittest
import torch
from run_learned_ou import OUActionPolicy, learn_generator


class LearnedOUChecks(unittest.TestCase):
    def test_physical_distribution_has_floor_and_parameter_gradients(self):
        policy = OUActionPolicy()
        x = torch.randn(80, 6)
        x[:, 1] = torch.linspace(1 / 16, 1, 80)
        distribution, _ = policy(x)
        self.assertTrue(bool((distribution.probs >= 0.1 / 3 - 1e-7).all()))
        loss = -distribution.log_prob(torch.arange(80) % 3).mean()
        loss.backward()
        self.assertTrue(bool(torch.isfinite(policy.ou_parameters.grad).all()))
        self.assertGreater(float(policy.ou_parameters.grad.abs().sum()), 0)

    def test_generator_learns_without_changing_fixed_parameter(self):
        torch.set_num_threads(1)
        rho = learn_generator(13, "learn_rho", 30)
        spread = learn_generator(13, "learn_spread", 30)
        self.assertEqual(rho["spread"], 0.7)
        self.assertEqual(spread["rho"], 0.9)
        self.assertLess(rho["rho"], 0.9)
        self.assertTrue(0.3 < spread["spread"] < 1.2)


if __name__ == "__main__":
    unittest.main()
