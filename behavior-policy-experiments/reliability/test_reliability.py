"""Meaningful controls for alternative rewards and repair comparisons."""
import unittest
import copy

from policies import ActionBalancedPolicy, ExploratoryPolicy, RND, behavior, train
from run_study import MonotoneEncoder, controls, fit, make_plan
import torch


class ReliabilityChecks(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.c = behavior.Config(jepa_steps=40, policy_updates=2, nenv=4, rollout=4,
                                 train_episodes=8, eval_episodes=4, horizon=8, burnin=4, batch=16)
        self.reference = (behavior.collect(self.c, 'ou', 4, 20000), behavior.collect(self.c, 'ou', 4, 30000))

    def test_new_baseline_replays_legacy_weights(self):
        data = behavior.collect(self.c, 'random', 8, 10000)
        legacy, _ = controls.schedules.train_encoder(data, 'icm_to_ou', self.c, controls.schedules.Schedule(), 0, self.reference)
        current, _, _ = fit(data, 'icm', self.c, 0, 0, self.reference)
        for key in legacy.state_dict():
            torch.testing.assert_close(current.state_dict()[key], legacy.state_dict()[key], atol=0, rtol=0)
        corrupted = {**data, 'z': data['z'] * 100, 'zp': data['zp'] - 99}
        check, _, _ = fit(corrupted, 'icm', self.c, 0, 0, self.reference)
        for key in current.state_dict():
            torch.testing.assert_close(check.state_dict()[key], current.state_dict()[key], atol=0, rtol=0)

    def test_crossed_init_leaves_pair_stream_unchanged(self):
        one, a = make_plan(self.c, 0, 100, 'icm')
        two, b = make_plan(self.c, 0, 101, 'icm')
        self.assertEqual(a['sha256'], b['sha256'])
        self.assertFalse(torch.equal(one.net[0].weight, two.net[0].weight))

    def test_floor_and_earlier_tail_are_distinct_interventions(self):
        _, base = make_plan(self.c, 0, 0, 'icm')
        _, floor = make_plan(self.c, 0, 0, 'ou_floor')
        _, early = make_plan(self.c, 0, 0, 'earlier_ou')
        self.assertEqual(base['alpha'][0], 0)
        self.assertEqual(floor['alpha'][0], .2)
        self.assertGreater(floor['alpha'][15], base['alpha'][15])
        self.assertEqual(early['alpha'][20], 1)
        self.assertLess(base['alpha'][20], 1)

    def test_monotone_architecture_cannot_fold_after_update(self):
        model = MonotoneEncoder()
        optimizer = torch.optim.Adam(model.parameters(), lr=.1)
        x = torch.zeros(1024, 6); x[:, 1] = torch.linspace(.001, 3., 1024)
        for _ in range(4):
            loss = (model(x) - torch.sin(7 * x[:, 1])).square().mean()
            optimizer.zero_grad(); loss.backward(); optimizer.step()
        self.assertTrue(bool((torch.diff(model(x)) > 0).all()))

    def test_random_action_floor_and_rnd_target(self):
        p = ExploratoryPolicy(.2)
        with torch.no_grad(): p.actor.bias.copy_(torch.tensor([20., -20., -20.]))
        probabilities = p(torch.randn(40, 6))[0].probs
        self.assertTrue(bool((probabilities >= .2 / 3 - 1e-7).all()))
        model = RND(); before = copy.deepcopy(model.target.state_dict())
        loss = model.errors(torch.randn(40, 6), torch.randn(40, 6), torch.zeros(40, dtype=torch.long)).mean()
        loss.backward()
        self.assertTrue(all(p.grad is None for p in model.target.parameters()))
        for key in before: torch.testing.assert_close(model.target.state_dict()[key], before[key])

    def test_plain_icm_actor_port_is_unchanged(self):
        base, _, _, _ = behavior.train_policy(self.c, 0)
        port, _ = train(self.c, 0, 'icm')
        for key in base.state_dict():
            torch.testing.assert_close(base.state_dict()[key], port.state_dict()[key], rtol=0, atol=0)

    def test_action_balancer_uses_all_actions(self):
        torch.manual_seed(5)
        policy = ActionBalancedPolicy()
        x = torch.ones(1, 6) * .2
        actions = torch.stack([policy.sample(x) for _ in range(90)]).flatten()
        counts = torch.bincount(actions, minlength=3)
        self.assertLess(int(counts.max() - counts.min()), 8)


if __name__ == '__main__': unittest.main()
