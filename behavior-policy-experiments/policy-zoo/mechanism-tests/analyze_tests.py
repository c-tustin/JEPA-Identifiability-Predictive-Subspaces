"""Targeted, exploratory paired tests using the zoo's existing 20 seeds."""

import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import analyze as original
import zoo
from zoo import behavior, controls
import numpy as np
from scipy import stats
import torch

OUT = HERE/"output"
ZOO_OUT = HERE.parent/"output"
SEEDS = tuple(range(8, 28))


def write(name, obj):
    OUT.mkdir(exist_ok=True)
    (OUT/name).write_text(json.dumps(obj, indent=2, allow_nan=False)+"\n")


def interval(values):
    rng = np.random.default_rng(20261104)
    v = np.asarray(values)
    return np.quantile(rng.choice(v, (20000, len(v))).mean(1), [.025, .975]).tolist()


def comparison(name, differences):
    d = np.asarray(differences)
    positive, negative = int((d > 1e-10).sum()), int((d < -1e-10).sum())
    n = positive+negative
    return dict(name=name, mean_difference=float(d.mean()), paired_bootstrap95=interval(d),
                wins=positive, losses=negative, ties=len(d)-n,
                p=1. if n == 0 else float(stats.binomtest(positive, n, .5).pvalue),
                differences=d.tolist())


def correct(rows):
    for row, p in zip(rows, original.holm([r["p"] for r in rows])):
        row["p_holm"] = p
    return rows


def main():
    torch.set_num_threads(1)
    runs = original.load(ZOO_OUT)
    def values(label):
        assert set(runs[label]) == set(SEEDS)
        return original.values(runs, label, SEEDS)
    ranking = [("count", "random"), ("icm", "random"), ("rnd", "icm"), ("disagreement", "icm")]
    tests = {"ranking": correct([comparison(f"{a} minus {b}", values(f"pure-{a}")-values(f"pure-{b}")) for a,b in ranking])}
    lp, hp, unfiltered = "ou_actions_lp_w0.1", "ou_actions_hp_w0.3", "ou_actions"
    persistence = [comparison(f"pure {a} minus {b}", values(f"pure-{a}")-values(f"pure-{b}"))
                   for a,b in [(lp, unfiltered), (hp, unfiltered), (hp, lp)]]
    interaction = ((values(f"handover-{hp}")-values(f"handover-{lp}"))
                   -(values(f"pure-{hp}")-values(f"pure-{lp}")))
    persistence.append(comparison("handover changes high-pass minus low-pass advantage", interaction))
    tests["persistence"] = correct(persistence)

    c = behavior.Config(**json.loads((ZOO_OUT/"config.json").read_text())["config"])
    initial, initial_folds, raw_curves = [], [], []
    for seed in SEEDS:
        torch.manual_seed(40000+seed)
        model = behavior.Encoder().eval()
        reference = (behavior.collect(c, "ou", c.eval_episodes, 20000+seed),
                     behavior.collect(c, "ou", c.eval_episodes, 30000+seed))
        metric, _ = controls.evaluate(model, reference, c, seed)
        initial.append(metric["common_grid_r2"])
        with torch.no_grad():
            z = torch.linspace(np.log(1/4)/c.sigma, np.log(16/4)/c.sigma, 1024)
            y = model(behavior.observe(z, c.sigma))
            sign = 1 if torch.cov(torch.stack([z,y]))[0,1] >= 0 else -1
            display = sign*(y-y.mean())/y.std(unbiased=False).clamp_min(1e-8)
            initial_folds.append(bool((display.diff() < -1e-6).any()))
            raw_curves.append(display.numpy())
    initial = np.asarray(initial)
    tests["baseline_change"] = correct([comparison(f"pure {policy} minus true initialization", values(f"pure-{policy}")-initial) for policy in zoo.PHYSICAL])
    write("initialization.json", dict(seeds=SEEDS, r2=initial.tolist(), mean_r2=float(initial.mean()),
                                       initial_folds=initial_folds, mean_first_update_ou=float(np.mean([runs['pure-ou'][s]['history'][0]['common_grid_r2'] for s in SEEDS]))))
    np.savez_compressed(OUT/"initialization-curves.npz", curves=np.stack(raw_curves), distance=(4*torch.exp(c.sigma*z)).numpy())

    equivalence = []
    for margin in (.001, .005):
        group = []
        for name in zoo.LATENT_FILTERS:
            if name == "ou_traj":
                continue
            d = values(f"pure-{name}")-values(f"pure-iid_{name}")
            mean, se = float(d.mean()), float(stats.sem(d))
            if se == 0:
                p = 0. if abs(mean) < margin else 1.
                ci = [mean, mean]
            else:
                p = max(float(stats.t.sf((mean+margin)/se, len(d)-1)),
                        float(stats.t.cdf((mean-margin)/se, len(d)-1)))
                half = stats.t.ppf(.95, len(d)-1)*se
                ci = [float(mean-half), float(mean+half)]
            group.append(dict(name=name, margin=margin, mean_difference=mean,
                              paired_t90=ci, p=p))
        equivalence.extend(correct(group))
    tests["equivalence_TOST"] = equivalence

    associations = []
    for seed in SEEDS:
        score = [runs[f"pure-{p}"][seed]["metrics"]["common_grid_r2"] for p in zoo.PHYSICAL]
        gaussian = [runs[f"pure-{p}"][seed]["source_stats"]["gaussian_cdf_distance"] for p in zoo.PHYSICAL]
        coverage = [runs[f"pure-{p}"][seed]["source_stats"]["size_bin_entropy"] for p in zoo.PHYSICAL]
        associations.append(dict(seed=seed, gaussian_rho=float(stats.spearmanr(score, gaussian).statistic),
                                 entropy_rho=float(stats.spearmanr(score, coverage).statistic)))
    tests["associations"] = correct([
        comparison("negative Gaussian discrepancy association", [-r["gaussian_rho"] for r in associations]),
        comparison("Gaussian association stronger than entropy", [-r["gaussian_rho"]-r["entropy_rho"] for r in associations])])
    tests["per_seed_associations"] = associations
    write("targeted-tests.json", dict(seeds=SEEDS, families=tests,
                                     scope="Exploratory tests; sign tests target non-tie win probabilities. TOST requires paired-difference distribution assumptions and post hoc margins."))
    print(json.dumps({k:[(r['name'],r.get('mean_difference'),r['p_holm']) for r in v] for k,v in tests.items() if k != 'per_seed_associations'},indent=2))


if __name__ == "__main__":
    main()
