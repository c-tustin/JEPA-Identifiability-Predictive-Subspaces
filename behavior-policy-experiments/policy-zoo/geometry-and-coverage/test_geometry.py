"""Checks the scientific controls, including exact matched exposure."""
import unittest
import numpy as np
import torch
import common_gc as study
import gc_statistics as stats


class GeometryControls(unittest.TestCase):
    def test_reset_pool_is_shared_and_stratified(self):
        d,x=study.reset_pool(13)
        d2,x2=study.reset_pool(13)
        torch.testing.assert_close(x,x2,atol=0,rtol=0)
        torch.testing.assert_close(d,d2,atol=0,rtol=0)
        bins=(d.log()/np.log(16)*32).long().clamp(0,31)
        # exp/log in float32 can move boundary-near samples one stratum.
        # The scientific invariant is exact equality of the pool across policies.
        self.assertLessEqual(int((bins.bincount(minlength=32)-512).abs().max()),2)

    def test_original_actor_and_reset_physics(self):
        torch.set_num_threads(1)
        for policy in ('icm','random','count'):
            data=study.reset_data(policy,13)
            distance,x=study.reset_pool(13)
            torch.testing.assert_close(data['x'],x,atol=0,rtol=0)
            expected=study.behavior.sensor(study.behavior.move(distance,data['action'],study.original.config()),study.original.config())
            torch.testing.assert_close(expected,data['xp'],atol=0,rtol=0)
        native=study.native_data('ou_actions_lp_w0.1',13)
        reset=study.reset_data('ou_actions_lp_w0.1',13,native)
        torch.testing.assert_close(reset['action'],native['action'],atol=0,rtol=0)

    def test_independent_regularizer_inputs_ignore_pairing(self):
        torch.set_num_threads(1)
        x=torch.arange(24,dtype=torch.float32).reshape(4,6)
        data={'x':x,'xp':torch.flip(x,[0])}
        a=study.pooled(data)
        b=study.pooled({'x':data['xp'],'xp':data['x']})
        # Same endpoint empirical law, irrespective of the original pairing.
        np.testing.assert_array_equal(a.numpy()[np.argsort(a[:,0])],b.numpy()[np.argsort(b[:,0])])
        empty=stats.comparisons([])
        self.assertEqual(len(empty['tests']),54)
        self.assertFalse(empty['complete'])
        self.assertTrue(all(r['holm_p']==1 for r in empty['tests']))


if __name__=='__main__':unittest.main()
