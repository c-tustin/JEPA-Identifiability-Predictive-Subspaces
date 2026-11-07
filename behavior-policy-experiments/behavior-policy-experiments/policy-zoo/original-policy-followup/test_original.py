"""Tests invariances that distinguish the proposed mechanisms."""
import unittest
import numpy as np
import torch
import original as o


class PairSymmetry(unittest.TestCase):
    def test_full_objective_is_invariant_to_reversing_pair_direction(self):
        torch.set_num_threads(1)
        torch.manual_seed(123)
        model = o.behavior.Encoder().double()
        d = torch.linspace(1, 16, 64, dtype=torch.float64)
        x = o.behavior.sensor(d, o.config())
        xp = o.behavior.sensor(torch.roll(d, 7), o.config())
        regularizer = o.behavior.SIGReg1D().double()
        def gradient(first, second, alignment):
            y = model(torch.cat([first, second]))
            loss = .1*regularizer(y)
            if alignment:
                loss = loss + .9*(y[:64]-y[64:]).square().mean()
            gradients = torch.autograd.grad(loss, model.parameters())
            return float(loss), torch.cat([g.flatten() for g in gradients])
        loss, grad = gradient(x, xp, True)
        reverse_loss, reverse_grad = gradient(xp, x, True)
        self.assertAlmostEqual(loss, reverse_loss, places=11)
        torch.testing.assert_close(grad, reverse_grad, atol=1e-11, rtol=1e-11)
        # Without alignment, the same endpoints in any order give the same loss.
        loss, grad = gradient(x, xp, False)
        shuffled_loss, shuffled_grad = gradient(x, torch.roll(xp, 3, 0), False)
        self.assertAlmostEqual(loss, shuffled_loss, places=11)
        torch.testing.assert_close(grad, shuffled_grad, atol=1e-11, rtol=1e-11)

    def test_early_collection_keeps_actor_and_real_physical_transitions(self):
        import run_early
        torch.set_num_threads(1)
        actor = run_early.actor_for('icm', 13)
        before = {key: value.clone() for key, value in actor.state_dict().items()}
        collection = o.config()
        collection.burnin = 0
        data = o.behavior.collect(collection, 'icm', collection.train_episodes, 10013, actor)
        for key, value in actor.state_dict().items():
            torch.testing.assert_close(value, before[key], atol=0, rtol=0)
        torch.manual_seed(10013)
        initial = o.behavior.initial(collection.train_episodes, collection)
        torch.testing.assert_close(data['x'][:len(initial)], o.behavior.sensor(initial, collection), atol=0, rtol=0)
        self.assertEqual(len(data['x']), 16384)
        # Geometry inversion below is analysis only; the encoder receives x/xp.
        distance = 1 / data['x'][:, 1]
        successor = o.behavior.move(distance, data['action'], collection)
        torch.testing.assert_close(o.behavior.sensor(successor, collection), data['xp'], atol=2e-6, rtol=2e-6)

    def test_alias_witness_does_not_flag_monotone_oracle(self):
        class Monotone(torch.nn.Module):
            def forward(self, x):
                return x[:, 1]
        metrics, _ = o.geometry(Monotone())
        self.assertFalse(metrics['folded'])
        self.assertEqual(metrics['alias_distance_gap'], 0)
        self.assertAlmostEqual(metrics['absolute_rank_correlation'], 1)

    def test_alias_witness_recovers_two_actual_equal_outputs(self):
        class Fold(torch.nn.Module):
            def forward(self, x):
                return (1/x[:, 1] - 4).square()
        metrics, _ = o.geometry(Fold())
        self.assertTrue(metrics['folded'])
        self.assertGreater(metrics['alias_distance_gap'], 5)
        self.assertLess(metrics['alias_residual'], 1e-10)
        a,b = metrics['alias_goals']
        self.assertAlmostEqual((a-4)**2, (b-4)**2, places=10)
        self.assertAlmostEqual(metrics['alias_goal_error_bound'],(b-a)/2)


if __name__=='__main__':
    unittest.main()
