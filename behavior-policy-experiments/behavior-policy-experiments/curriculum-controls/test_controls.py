"""Check causal controls, exact legacy replay, and matched OU exposure."""
import copy
import unittest

import torch

from run_controls import (batch_at, behavior, exposure_stream, learning_rate,
                          make_plan, ou_data, run_segment, schedules, train_exposure)


class CurriculumChecks(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.c = behavior.Config(jepa_steps=40, policy_updates=2, nenv=4, rollout=4,
                                 train_episodes=8, eval_episodes=4, horizon=8, burnin=4, batch=16)
        self.seed = 9
        self.data = behavior.collect(self.c, "random", self.c.train_episodes, 10000 + self.seed)
        self.reference = (behavior.collect(self.c, "ou", 4, 20000 + self.seed),
                          behavior.collect(self.c, "ou", 4, 30000 + self.seed))

    def assert_weights_equal(self, one, two):
        for key, value in one.state_dict().items():
            torch.testing.assert_close(value, two.state_dict()[key], rtol=0, atol=0)

    def test_explicit_plan_matches_legacy_training_and_ignores_labels(self):
        legacy, _ = schedules.train_encoder(self.data, "icm_to_ou", self.c, schedules.Schedule(), self.seed, self.reference)
        initial, plan = make_plan(self.c, self.seed)
        optimizer = torch.optim.Adam(initial.parameters(), lr=.001)
        only_views = {key: self.data[key] for key in ["x", "xp"]}
        run_segment(initial, optimizer, only_views, ou_data(self.c, self.seed), plan, self.c, self.seed,
                    self.reference, 0, self.c.jepa_steps)
        self.assert_weights_equal(initial, legacy)

    def test_ou_sham_is_exact_fixed_ou(self):
        ou = ou_data(self.c, self.seed)
        baseline, _ = behavior.train_encoder(ou, self.c, 40000 + self.seed)
        model, plan = make_plan(self.c, self.seed)
        optimizer = torch.optim.Adam(model.parameters(), lr=.001)
        run_segment(model, optimizer, ou, ou, plan, self.c, self.seed, self.reference, 0, self.c.jepa_steps)
        self.assert_weights_equal(model, baseline)

    def test_forking_without_intervention_preserves_training(self):
        model, plan = make_plan(self.c, self.seed)
        ou = ou_data(self.c, self.seed)
        full = copy.deepcopy(model)
        full_opt = torch.optim.Adam(full.parameters(), lr=.001)
        run_segment(full, full_opt, self.data, ou, plan, self.c, self.seed, self.reference, 0, self.c.jepa_steps)
        prefix_opt = torch.optim.Adam(model.parameters(), lr=.001)
        run_segment(model, prefix_opt, self.data, ou, plan, self.c, self.seed, self.reference, 0, plan["tail_start"])
        fork = copy.deepcopy(model)
        fork_opt = torch.optim.Adam(fork.parameters(), lr=.001)
        fork_opt.load_state_dict(copy.deepcopy(prefix_opt.state_dict()))
        run_segment(fork, fork_opt, self.data, ou, plan, self.c, self.seed, self.reference,
                    plan["tail_start"], self.c.jepa_steps)
        self.assert_weights_equal(fork, full)
        fresh_opt = torch.optim.Adam(model.parameters(), lr=.001)
        self.assertEqual(len(fresh_opt.state), 0)
        self.assertGreater(len(prefix_opt.state), 0)

    def test_exposure_control_keeps_exact_ou_occurrences_in_order(self):
        initial, plan = make_plan(self.c, self.seed)
        ix, steps = exposure_stream(plan)
        expected = torch.cat([plan["indices"][i][mask] for i, mask in enumerate(plan["masks"])])
        torch.testing.assert_close(ix, expected, rtol=0, atol=0)
        self.assertEqual(len(ix), int(plan["masks"].sum()))
        self.assertTrue(bool((steps[1:] >= steps[:-1]).all()))
        for clock in [False, True]:
            _, _, metadata = train_exposure(initial, ou_data(self.c, self.seed), plan, self.c, self.seed,
                                             self.reference, clock)
            self.assertEqual(metadata["ou_pairs"], len(expected))
            self.assertEqual(metadata["updates"], (len(expected) + self.c.batch - 1) // self.c.batch)

    def test_tail_pairs_and_reset_learning_rates(self):
        _, plan = make_plan(self.c, self.seed)
        ou = ou_data(self.c, self.seed)
        for step in range(plan["tail_start"], self.c.jepa_steps):
            x, xp = batch_at(self.data, ou, plan, step)
            torch.testing.assert_close(x, ou["x"][plan["indices"][step]], rtol=0, atol=0)
            torch.testing.assert_close(xp, ou["xp"][plan["indices"][step]], rtol=0, atol=0)
        self.assertEqual(learning_rate(0, 100), .001)
        self.assertEqual(learning_rate(99, 100), 0.)
        self.assertLess(learning_rate(plan["tail_start"], self.c.jepa_steps), .001)
        self.assertEqual(learning_rate(0, self.c.jepa_steps - plan["tail_start"]), .001)

    def test_plan_reproduces_independently_of_intervening_randomness(self):
        one, a = make_plan(self.c, self.seed)
        torch.randn(1234)
        two, b = make_plan(self.c, self.seed)
        self.assertEqual(a["sha256"], b["sha256"])
        self.assert_weights_equal(one, two)


if __name__ == "__main__":
    unittest.main()
