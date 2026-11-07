"""Checks that make the zoo comparable with the earlier reliability study."""

import unittest

import numpy as np
import torch

import zoo
from run_zoo import fit_pure
from zoo import behavior, controls, reliability


class ZooChecks(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.c = behavior.Config(jepa_steps=40, policy_updates=2, nenv=4, rollout=4,
                                 train_episodes=8, eval_episodes=4, horizon=8,
                                 burnin=4, batch=16)
        self.reference = (behavior.collect(self.c, "ou", 4, 20000),
                          behavior.collect(self.c, "ou", 4, 30000))

    def test_pure_ou_reproduces_fixed_ou_condition(self):
        pairs = zoo.collect_pairs(self.c, 0.9, 3)
        ou = controls.ou_data(self.c, 3)
        for key in ["x", "xp"]:
            torch.testing.assert_close(pairs[key], ou[key], atol=0, rtol=0)
        views = {k: pairs[k] for k in ["x", "xp"]}
        a, _, _ = fit_pure(views, self.c, 3, self.reference)
        b, _, _ = reliability.fit(views, "ou", self.c, 3, 3, self.reference)
        for key, value in a.state_dict().items():
            torch.testing.assert_close(value, b.state_dict()[key], atol=0, rtol=0)

    def test_filtered_ou_is_standardized_with_predicted_correlation(self):
        for name in ["ou_traj", "ou_lp_w0.3", "ou_hp_w1", "ou_hp2_w3"]:
            stages = zoo.LATENT_FILTERS[name]
            process = zoo.FilteredOU(4000, stages, generator=torch.Generator().manual_seed(0))
            values = torch.stack([process.step().clone() for _ in range(200)])
            self.assertAlmostEqual(float(values.var()), 1.0, delta=0.03)
            lag = float(torch.corrcoef(torch.stack([values[:-1].flatten(),
                                                    values[1:].flatten()]))[0, 1])
            self.assertAlmostEqual(lag, zoo.filter_stats(stages)[1], delta=0.02)
        self.assertLess(zoo.source_rho("ou_hp2_w3"), -0.4)
        self.assertGreater(zoo.source_rho("ou_lp_w0.1"), 0.99)

    def test_colored_actions_keep_uniform_action_marginal(self):
        for stages in [[], [("lp", 0.1)], [("hp", 3.0), ("hp", 3.0)]]:
            torch.manual_seed(0)
            policy = zoo.ColoredActions(stages)
            policy.reset(4000)
            actions = torch.cat([policy.act(torch.zeros(4000, 6)) for _ in range(50)])
            frequencies = torch.bincount(actions, minlength=3).float() / len(actions)
            torch.testing.assert_close(frequencies, torch.full((3,), 1 / 3), atol=0.01, rtol=0)

    def test_closed_loop_rules_read_only_views(self):
        data = zoo.collect_physical(self.c, zoo.Sweep(epsilon=0.0), 4, 0)
        corrupted_latents = {**data, "z": data["z"] * 0}
        self.assertTrue(torch.equal(corrupted_latents["x"], data["x"]))
        # Sweep must reverse at the near edge: height >= 0.9 -> retreat.
        policy = zoo.Sweep(epsilon=0.0)
        policy.reset(2)
        x = torch.zeros(2, 6)
        x[:, 1] = torch.tensor([0.95, 0.05])
        self.assertEqual(policy.act(x).tolist(), [2, 0])

    def test_task_reward_peaks_at_target(self):
        x = torch.zeros(3, 6)
        x[:, 1] = torch.tensor([0.25, 0.5, 0.0625])
        reward = zoo.task_reward(x)
        self.assertAlmostEqual(float(reward[0]), 1.0)
        self.assertTrue(bool((reward[1:] < 0.1).all()))

    def test_every_source_has_training_shape(self):
        n = self.c.train_episodes * self.c.horizon
        for name, rule in zoo.PHYSICAL_RULES.items():
            data = zoo.collect_physical(self.c, rule(), self.c.train_episodes, 1)
            self.assertEqual(data["x"].shape, (n, 6), name)
        for name, stages in zoo.LATENT_FILTERS.items():
            data = zoo.collect_latent(self.c, stages, 1)
            self.assertEqual(data["x"].shape, (n, 6), name)
            self.assertTrue(np.isfinite(data["z"].numpy()).all())


if __name__ == "__main__":
    unittest.main()
