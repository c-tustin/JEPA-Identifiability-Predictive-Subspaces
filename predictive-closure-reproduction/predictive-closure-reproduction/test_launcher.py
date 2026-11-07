"""Standard-library checks of the pinned job protocol; no data, Torch or training."""
import os
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from launch import (ROOT, build_records, main, official_runner, verify_source,
                    validate_runtime, restartable_traffic_stage)


class OfficialPlanTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.upstream = Path(os.environ.get("PC_UPSTREAM", str(ROOT / "upstream"))).resolve()
        if not (cls.upstream / "run.py").exists():
            raise unittest.SkipTest("Set PC_UPSTREAM to the pinned official checkout")
        verify_source(cls.upstream)
        cls.official = official_runner(cls.upstream)

    def jobs(self, **kwargs):
        return build_records(self.official, Path("/data/cifar-10-batches-py"), Path("/runs"), **kwargs)

    def test_exact_main_counts(self):
        rows = self.jobs()
        self.assertEqual(len(rows), 77)
        self.assertEqual({s: sum(r["seed"] == s for r in rows) for s in [303,304,305]},
                         {303:39,304:19,305:19})
        self.assertEqual({s: sum(r["study"] == s for r in rows) for s in
                          ["traffic","predictability","subset","reconstruction","side_input"]},
                         dict(traffic=15,predictability=15,subset=12,reconstruction=20,side_input=15))

    def test_reconstruction_has_one_seed_and_no_zero_duplicate(self):
        rows = [r for r in self.jobs() if r["study"] == "reconstruction"]
        self.assertEqual({r["seed"] for r in rows}, {303})
        self.assertEqual({r["beta"] for r in rows}, {.0425,.105,.135,.36})

    def test_cpu_profile_changes_only_runtime_precision_fields(self):
        ordinary = self.jobs()
        cpu = self.jobs(device="cpu", cpu_fp32=True)
        for left, right in zip(ordinary, cpu):
            for (lm, lc), (rm, rc) in zip(left["stages"], right["stages"]):
                self.assertEqual(lm, rm)
                self.assertEqual({k:v for k,v in lc.items() if k not in {"device","bf16"}},
                                 {k:v for k,v in rc.items() if k not in {"device","bf16"}})
                self.assertEqual((rc["device"],rc["bf16"]), ("cpu",False))

    def test_full_budgets_unchanged(self):
        for row in self.jobs():
            config = row["stages"][0][1]
            self.assertEqual(config["batch_size"], 512)
            self.assertEqual(config["sigreg_projections"], 1024)
            self.assertEqual(config["encoder_base_channels"], 32)
            self.assertEqual(config["encoder_representation_width"], 1024)
            self.assertEqual(config["latent_dim"], 3)
            if row["study"] == "subset":
                self.assertEqual(config["epochs"], 100)
            elif row["study"] == "traffic":
                self.assertEqual(config["epochs"], 25)
            else:
                self.assertEqual(config["steps"], 4900)
            self.assertTrue(config["bf16"])

    def test_endpoint_plan_keeps_full_training_budget(self):
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            main(["--upstream", str(self.upstream), "--output", directory,
                  "--study", "traffic", "--seeds", "303", "--values", "0", "1",
                  "--device", "cpu", "--cpu-fp32", "--local-intel", "--skip-plot"])
            rows = json.loads((Path(directory) / "launcher-plan.json").read_text())
            self.assertEqual([r["x"] for r in rows], [0, 1])
            self.assertTrue(all(r["stages"][0]["config"]["epochs"] == 25 for r in rows))

    def test_legacy_runtime_cannot_be_labelled_official(self):
        audit = {"packages": {"torch": "2.2.2", "numpy": "1.26.4", "matplotlib": "3.9.0"}}
        with self.assertRaisesRegex(RuntimeError, "torch must be 2.11.0"):
            validate_runtime(audit, "cpu", True)
        with self.assertRaisesRegex(ValueError, "requires --device cpu --cpu-fp32"):
            validate_runtime(audit, "cuda:0", False, local_intel=True)

    def test_restartable_does_not_replace_uncheckpointed_run(self):
        row = self.jobs(studies=["traffic"], seeds=[303])[0]
        with tempfile.TemporaryDirectory() as temporary:
            seed_output = Path(temporary)
            directory = seed_output / row["tag"]
            directory.mkdir()
            (directory / "config_0.json").write_text(json.dumps(row["stages"][0][1]))
            with self.assertRaisesRegex(RuntimeError, "no restart checkpoint"):
                restartable_traffic_stage(self.official, row, self.upstream, seed_output, 10)


if __name__ == "__main__":
    unittest.main()
