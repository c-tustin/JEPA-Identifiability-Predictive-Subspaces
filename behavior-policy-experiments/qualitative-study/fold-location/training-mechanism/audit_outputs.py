"""Audit all 60 comparison outcomes and the stated first-birth event."""

import json
from pathlib import Path
import re

import numpy as np
import torch

from run_mechanism import HERE, OUT, ORIGINAL, ScaledEncoder, trials


def main():
    torch.set_num_threads(1)
    cohort = json.loads((OUT/"cohort.json").read_text())
    settings = json.loads((OUT/"settings.json").read_text())
    c = trials.behavior.Config(**settings["config"])
    for row in cohort["rows"]:
        seed,gain,condition = row["seed"],row["gain"],row["condition"]
        if condition=="balanced":
            state = trials.load(OUT/"runs"/f"balanced-g{gain}-{seed}.pt")["encoder"]
        else:
            folder = ORIGINAL if gain==4 else HERE.parent/"output"/f"gain-{gain}"
            state = trials.load(folder/"encoders"/f"icm-{seed}.pt")["encoder"]
        model = ScaledEncoder(gain)
        model.load_state_dict(state)
        outcome = trials.representation(model,c,seed)
        assert outcome["folded"] == row["folded"]
        assert abs(outcome["r2"]-row["r2"]) < 1e-7
    assert len(cohort["rows"]) == 60
    paired = {r["gain"]:r for r in cohort["paired_transitions"]}
    assert paired[8]["introduced_seeds"] == [135]
    assert paired[8]["repaired_seeds"] == [131,132,136,139]
    trace = json.loads((OUT/"birth-interval-trace.json").read_text())
    assert [r["step"] for r in trace] == list(range(250))
    first = next(r["step"]+1 for r in trace if r["after_has_root"])
    event = json.loads((OUT/"birth-event.json").read_text())
    assert first == event["update_after"] == 205
    assert not event["before"]["turning_distances"]
    assert not event["alignment_moment_only"]["turning_distances"]
    assert len(event["after"]["turning_distances"]) == len(event["sigreg_moment_only"]["turning_distances"]) == 1
    assert event["replayed_update250_parameter_error"] == 0
    missing = []
    for document in (HERE/"README.md",HERE/"MATH.md"):
        source = document.read_text()
        assert source.count("$$") % 2 == 0
        for target in re.findall(r"\]\(([^)]+)\)",source):
            if "://" in target or target.startswith("#"):
                continue
            file = document.parent/target.split("#")[0]
            if not file.exists():
                missing.append(str(file))
    assert not missing, missing
    regularizer = trials.behavior.SIGReg1D().double()
    y = torch.linspace(-2.5,1.8,1024,dtype=torch.float64,requires_grad=True)
    automatic = torch.autograd.grad(regularizer(y),y)[0]
    phase = y[:,None]*regularizer.frequency
    cosine,sine = phase.cos().mean(0),phase.sin().mean(0)
    formula = 2*((regularizer.weight*regularizer.frequency)*(
        -(cosine-regularizer.target)*phase.sin()+sine*phase.cos())).sum(1)
    torch.testing.assert_close(automatic,formula,atol=1e-12,rtol=1e-12)
    trials.atomic_json(OUT/"sigreg-force-formula-check.json",dict(
        passed=True,outputs=1024,frequency_knots=17,
        max_derivative_difference=float((automatic-formula).abs().max().detach())))
    trials.atomic_json(OUT/"completion-audit.json",dict(
        passed=True,comparison_models_recomputed=60,
        sequential_birth_updates_inspected=250,first_detected_in_range_root_update=first,
        one_step_intervention_outcomes_confirmed=True,paired_failures_retained=True,
        local_document_links_exist=True,sigreg_force_formula_checked=True,
        scope="Existing distance-only environment; matched gain/sampling interventions and verified local mechanism. No universal convergence or exact landmark-root claim."))
    print("All comparison outcomes, birth-event claims, paired failures and local links audited")


if __name__ == "__main__":
    main()
