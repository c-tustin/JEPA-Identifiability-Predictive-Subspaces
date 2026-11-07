"""Small real-CIFAR test: a fresh process must resume identically to uninterrupted training.

Reduced test architecture/budget is not a paper experiment. No files from the
paused full run are touched.
"""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parent


class RestartTest(unittest.TestCase):
    def test_separate_process_resume_matches_uninterrupted(self):
        import torch
        upstream = Path(os.environ["PC_UPSTREAM"]).resolve()
        data = upstream / "data/cifar-10-batches-py"
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)

            def command(name, stop=None):
                output = directory / name
                output.mkdir(exist_ok=True)
                config = json.loads((upstream / "configs/traffic_train.json").read_text())
                config.update(run_id="restart_test", seed=303, data_root=str(data),
                              output_checkpoint=str(output / "jepa.pt"), output_json=str(output / "jepa.json"),
                              device="cpu", bf16=False, epochs=1, sequences_per_epoch=12,
                              batch_size=2, encoder_base_channels=4, encoder_representation_width=16,
                              encoder_frame_chunk_size=16, predictor_hidden_dim=8,
                              sigreg_projections=8, sigreg_chunk_size=4, warmup_steps=1,
                              log_every=1, evaluation_every=2, evaluation_batch_size=2,
                              evaluation_batches=1, cpu_threads=1, cpu_interop_threads=1,
                              prediction_weight=.9, sigreg_weight=.1)
                path = output / "config.json"
                path.write_text(json.dumps(config))
                args = [sys.executable, str(ROOT / "restart_traffic.py"), "--upstream", str(upstream),
                        "--config", str(path), "--checkpoint-every", "2"]
                if stop is not None:
                    args += ["--stop-after", str(stop)]
                result = subprocess.run(args, capture_output=True, text=True, timeout=120)
                self.assertEqual(result.returncode, 75 if stop is not None else 0, result.stderr)
                return output

            continuous = command("continuous")
            interrupted = command("resumed", stop=3)
            self.assertFalse((interrupted / "jepa.pt").exists())
            self.assertEqual(json.loads((interrupted / "restart/latest.json").read_text())["step"], 3)
            command("resumed")
            left = torch.load(continuous / "jepa.pt", weights_only=False)
            right = torch.load(interrupted / "jepa.pt", weights_only=False)
            self.assertEqual(left["completed_steps"], 6)
            self.assertEqual(right["completed_steps"], 6)
            for name, value in left["model_state_dict"].items():
                self.assertTrue(torch.equal(value, right["model_state_dict"][name]), name)
            self.assertEqual(left["learning_curve"], right["learning_curve"])
            self.assertEqual(left["evaluation_curve"], right["evaluation_curve"])
            a = torch.load(continuous / "restart/step-0000006.pt", weights_only=False)
            b = torch.load(interrupted / "restart/step-0000006.pt", weights_only=False)
            for name in ("data_rng", "torch_rng"):
                self.assertTrue(torch.equal(a[name], b[name]))
            self.assertEqual(a["optimizer"]["param_groups"], b["optimizer"]["param_groups"])
            for parameter, values in a["optimizer"]["state"].items():
                for name, value in values.items():
                    self.assertTrue(torch.equal(value, b["optimizer"]["state"][parameter][name]))
            self.assertLessEqual(len(list((interrupted / "restart").glob("step-*.pt"))), 2)


if __name__ == "__main__":
    unittest.main()
