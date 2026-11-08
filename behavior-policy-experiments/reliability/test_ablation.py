"""Check that loss-source changes have no effect when the source views agree."""

import unittest
import torch
from policies import behavior
from ablate_losses import CELLS, fit


class LossAblationChecks(unittest.TestCase):
    def test_identical_views_make_all_four_cells_identical(self):
        torch.set_num_threads(1)
        c = behavior.Config(
            jepa_steps=8,
            train_episodes=4,
            eval_episodes=2,
            horizon=4,
            burnin=2,
            batch=8,
        )
        data = {"x": torch.ones(16, 6) * 0.2, "xp": torch.ones(16, 6) * 0.21}
        reference = (behavior.collect(c, "ou", 2, 1), behavior.collect(c, "ou", 2, 2))
        models = [fit(data, c, 3, a, r, reference) for a, r in CELLS]
        for model in models[1:]:
            for key, value in models[0].state_dict().items():
                torch.testing.assert_close(
                    model.state_dict()[key], value, rtol=0, atol=0
                )


if __name__ == "__main__":
    unittest.main()
