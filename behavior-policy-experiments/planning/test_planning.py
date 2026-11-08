import unittest
import torch
from evaluate_planning import (
    behavior,
    candidate_bank,
    choose_actions,
    goals,
    rollout,
    train_predictor,
)


class OracleEncoder(torch.nn.Module):
    def forward(self, x):
        return 1 / x[:, 1]


class OraclePredictor(torch.nn.Module):
    def forward(self, y, a):
        return y + 0.25 * (a.float() - 1)


class PlanningChecks(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def test_goal_bank_is_reachable_and_oracle_planner_works(self):
        c = behavior.Config()
        for seed in range(23):
            metrics, _ = rollout(OracleEncoder(), OraclePredictor(), 0.0, 1.0, c, seed)
            self.assertLess(metrics["oracle_mean_error"], 1e-5)
            self.assertGreaterEqual(metrics["success_within_one_step"], 0.95)
            self.assertLessEqual(metrics["mean_terminal_distance_error"], 0.25)

    def test_predictor_training_ignores_true_distance_labels(self):
        c = behavior.Config(train_episodes=4, horizon=8, burnin=4)
        data = behavior.collect(c, "random", 4, 7)
        changed = {**data, "z": data["z"] + 1000, "zp": data["zp"] * -100}
        a, ma, sa = train_predictor(OracleEncoder(), data, data, 1, 8)
        b, mb, sb = train_predictor(OracleEncoder(), changed, changed, 1, 8)
        self.assertEqual((ma, sa), (mb, sb))
        for key in a.state_dict():
            torch.testing.assert_close(
                a.state_dict()[key], b.state_dict()[key], rtol=0, atol=0
            )

    def test_candidate_sequences_repeat_exactly(self):
        torch.testing.assert_close(candidate_bank(1), candidate_bank(1), rtol=0, atol=0)
        self.assertFalse(torch.equal(candidate_bank(1), candidate_bank(2)))


if __name__ == "__main__":
    unittest.main()
