"""Guard the rotation-only diagnostic and paired improvement directions."""
import unittest
import torch
import run_ou as run
import analyze_ou as analysis


class ScaleModel:
    def __init__(self, scale):
        self.scale = scale
    def __call__(self, x):
        return self.scale * torch.log((1/x[:,1])/4) / run.original.config().sigma


class OUChecks(unittest.TestCase):
    def test_affine_score_does_not_hide_wrong_output_scale(self):
        wrong_scale = run.gaussian_metrics(ScaleModel(2), 13)
        self.assertGreater(wrong_scale['gaussian_r2'], .999999)
        self.assertGreater(wrong_scale['orthogonal_mse'], .8)
        sign_only = run.gaussian_metrics(ScaleModel(-1), 13)
        self.assertLess(sign_only['orthogonal_mse'], 1e-10)

    def test_benefit_direction_is_correct_for_error_and_recovery(self):
        def row(recovery,error):
            return {'metrics':{'affine_log_r2':recovery}, 'planning':{'goal_error':error}}
        a,b=[row(.99,.1)]*60,[row(.9,.3)]*60
        for key in ('affine_log_r2','goal_error'):
            result=analysis.compare(a,b,key)
            self.assertEqual(result['wins'],60)
            self.assertEqual(result['losses'],0)
            self.assertAlmostEqual(result['p'],2*.5**60)
        self.assertEqual(analysis.compare(a,a,'goal_error')['p'],1.)


if __name__=='__main__':
    unittest.main()
