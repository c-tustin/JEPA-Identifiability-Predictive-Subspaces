"""Check schedule semantics and preserve the original training controls."""
import unittest

import torch

from run_schedules import (PairSampler, Schedule, behavior, entropy_coefficient,
                           epoch_fraction, parameters, progress, train_encoder)


class ScheduleChecks(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(31)
        self.c = behavior.Config(jepa_steps=4, policy_updates=2, nenv=4, rollout=4,
                                 train_episodes=8, eval_episodes=4, horizon=8, burnin=4)
        self.schedule = Schedule()

    def test_handover_has_fixed_initial_and_final_stages(self):
        for condition in ["icm_to_ou", "icm_entropy_decay_to_ou"]:
            self.assertEqual(parameters(condition, .25, self.c, self.schedule)["ou_fraction"], 0.)
            self.assertAlmostEqual(parameters(condition, .5, self.c, self.schedule)["ou_fraction"], .5)
            self.assertEqual(parameters(condition, .75, self.c, self.schedule)["ou_fraction"], 1.)

    def test_ou_parameters_change_independently(self):
        rho = parameters("ou_rho_schedule", 0, self.c, self.schedule)
        sigma = parameters("ou_sigma_schedule", 0, self.c, self.schedule)
        self.assertEqual(rho, {"ou_fraction": 1., "rho": .5, "sigma": .7})
        self.assertEqual(sigma, {"ou_fraction": 1., "rho": .9, "sigma": 1.})
        final = parameters("ou_rho_sigma_schedule", 1, self.c, self.schedule)
        self.assertEqual(final, {"ou_fraction": 1., "rho": .9, "sigma": .7})

    def test_epoch_schedule_is_constant_within_epoch(self):
        self.assertEqual(epoch_fraction(0, 4000, 16384, 512), epoch_fraction(31, 4000, 16384, 512))
        self.assertGreater(epoch_fraction(32, 4000, 16384, 512), 0)
        self.assertEqual(epoch_fraction(3999, 4000, 16384, 512), 1)

    def test_entropy_bonus_endpoints(self):
        self.assertEqual(entropy_coefficient(0, 600, self.c, self.schedule), .0005)
        self.assertEqual(entropy_coefficient(599, 600, self.c, self.schedule), 0.)
        values = [entropy_coefficient(i, 600, self.c, self.schedule) for i in range(600)]
        self.assertTrue(all(a >= b for a, b in zip(values, values[1:])))

    def test_pair_mixture_chooses_whole_pairs(self):
        data = {"x": torch.full((12000, 6), 999.), "xp": torch.full((12000, 6), 1000.)}
        sampler = PairSampler(data, self.c, 0)
        settings = {"ou_fraction": .5, "rho": .9, "sigma": .7}
        x, xp = sampler.sample(torch.arange(12000), settings)
        kept_icm = (x == 999.).all(-1)
        torch.testing.assert_close(kept_icm, (xp == 1000.).all(-1))
        self.assertAlmostEqual(float(kept_icm.float().mean()), .5, delta=.02)
        x0, xp0 = sampler.sample(torch.arange(12000), {**settings, "ou_fraction": 0.})
        torch.testing.assert_close(x0, data["x"])
        torch.testing.assert_close(xp0, data["xp"])

    def test_rho_change_preserves_standard_gaussian_states(self):
        data = {"x": torch.zeros(150000, 6)}
        sampler = PairSampler(data, self.c, 4)
        ix = torch.arange(150000)
        for rho in [.5, .9]:
            x, xp = sampler.sample(ix, {"ou_fraction": 1., "rho": rho, "sigma": .7})
            z = -torch.log(4 * x[:, 1]) / .7
            zp = -torch.log(4 * xp[:, 1]) / .7
            self.assertAlmostEqual(float(zp.mean()), 0, delta=.012)
            self.assertAlmostEqual(float(zp.var()), 1, delta=.015)
            self.assertAlmostEqual(float(torch.corrcoef(torch.stack([z, zp]))[0, 1]), rho, delta=.012)

    def test_static_training_matches_original_and_ignores_distance_labels(self):
        data = behavior.collect(self.c, "random", self.c.train_episodes, 10000)
        ref = (behavior.collect(self.c, "ou", 4, 20000), behavior.collect(self.c, "ou", 4, 30000))
        original, _ = behavior.train_encoder(data, self.c, 40000)
        scheduled, _ = train_encoder(data, "icm", self.c, self.schedule, 0, ref)
        changed = {**data, "z": data["z"] + 1000, "zp": data["zp"] * -99}
        no_labels, _ = train_encoder(changed, "icm", self.c, self.schedule, 0, ref)
        for key, value in original.state_dict().items():
            torch.testing.assert_close(scheduled.state_dict()[key], value, rtol=0, atol=0)
            torch.testing.assert_close(no_labels.state_dict()[key], value, rtol=0, atol=0)

    def test_constant_entropy_callback_preserves_original_policy(self):
        original, icm, _, _ = behavior.train_policy(self.c, 0)
        callback, callback_icm, _, _ = behavior.train_policy(self.c, 0, lambda i, n: self.c.entropy_weight)
        for one, two in [(original, callback), (icm, callback_icm)]:
            for key, value in one.state_dict().items():
                torch.testing.assert_close(two.state_dict()[key], value, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
