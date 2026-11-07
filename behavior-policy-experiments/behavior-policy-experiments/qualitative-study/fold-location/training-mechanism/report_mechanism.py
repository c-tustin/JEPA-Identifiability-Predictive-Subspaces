"""Sampling controls, training histories and exact scale/adaptation budgets."""

import csv
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/kempner-qualitative-mpl")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from run_mechanism import HERE, OUT, ORIGINAL, GAINS, DIAGNOSTIC_SEEDS, ScaledEncoder, trials

LOCATION = HERE.parent / "output"
COLORS = {"ordinary": "#D55E00", "balanced": "#0072B2"}


def roots(q, slope):
    indices = np.flatnonzero(slope[:-1] * slope[1:] < 0)
    return [float(q[k] - slope[k]*(q[k+1]-q[k])/(slope[k+1]-slope[k])) for k in indices]


def load_run(condition, gain, seed):
    return trials.load(OUT / "runs" / f"{condition}-g{gain}-{seed}.pt")


def ordinary_profile(gain, seed):
    return trials.load(LOCATION / "profiles" / f"icm-gain-{gain}-{seed}.pt")


def write_csv(path, rows):
    if not rows:
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def save(fig, name):
    folder = OUT / "figures"
    folder.mkdir(exist_ok=True)
    for extension in ("png", "pdf"):
        fig.savefig(folder / f"{name}.{extension}", dpi=170, bbox_inches="tight")
    plt.close(fig)


def distance_axis(ax):
    ax.set_xscale("log")
    ax.set_xticks([1,2,4,8,16], ["1","2","4","8","16"])
    ax.axvspan(2,4,color=".8",alpha=.15,zorder=-2)
    ax.set_xlabel("Physical distance")
    ax.grid(alpha=.15)


def normalize(run):
    c = trials.behavior.Config(**run["config"])
    model = ScaledEncoder(run["gain"])
    model.load_state_dict(run["encoder"])
    generator = torch.Generator().manual_seed(1800000 + run["seed"])
    d = torch.exp(torch.rand(4096,generator=generator)*np.log(16))
    with torch.no_grad():
        y = model(trials.behavior.sensor(d,c))
    sign = 1 if float(torch.cov(torch.stack([y,d.log()]))[0,1]) >= 0 else -1
    factor = sign / max(float(y.std(unbiased=False)),1e-8)
    return factor, -factor*float(y.mean())


def cohort(seeds):
    rows, summary = [], []
    for gain in GAINS:
        for seed in seeds:
            old = ordinary_profile(gain,seed)
            new = load_run("balanced",gain,seed)
            for condition,p in (("ordinary",old),("balanced",new["diagnostic"])):
                rows.append(dict(seed=seed,gain=gain,condition=condition,
                                 folded=p["folded"],r2=p["r2"],
                                 reversal_fraction=p["reversal_fraction"]))
        for condition in COLORS:
            chosen = [r for r in rows if r["gain"]==gain and r["condition"]==condition]
            summary.append(dict(gain=gain,condition=condition,n=len(chosen),
                folded=sum(r["folded"] for r in chosen),mean_r2=float(np.mean([r["r2"] for r in chosen]))))
    paired = []
    for gain in GAINS:
        lookup = {(r["seed"],r["condition"]):r for r in rows if r["gain"]==gain}
        repaired = [s for s in seeds if lookup[s,"ordinary"]["folded"] and not lookup[s,"balanced"]["folded"]]
        introduced = [s for s in seeds if not lookup[s,"ordinary"]["folded"] and lookup[s,"balanced"]["folded"]]
        persistent = [s for s in seeds if lookup[s,"ordinary"]["folded"] and lookup[s,"balanced"]["folded"]]
        paired.append(dict(gain=gain,repaired_seeds=repaired,introduced_seeds=introduced,persistent_seeds=persistent))
    trials.atomic_json(OUT/"cohort.json",dict(rows=rows,summary=summary,paired_transitions=paired))
    write_csv(OUT/"cohort.csv",rows)
    for start in (0,5):
        fig,axes = plt.subplots(3,5,figsize=(16,9),layout="constrained")
        for row,gain in zip(axes,GAINS):
            for ax,seed in zip(row,seeds[start:start+5]):
                for condition in COLORS:
                    p = ordinary_profile(gain,seed) if condition=="ordinary" else load_run(condition,gain,seed)["diagnostic"]
                    ax.plot(p["distance"],p["embedding"],color=COLORS[condition],
                            label="Ordinary SIGReg" if condition=="ordinary" else "Balanced SIGReg")
                ax.set_title(f"Seed {seed} · gain {gain}")
                ax.set_ylabel("Learned coordinate")
                distance_axis(ax)
        handles,labels = axes[0,0].get_legend_handles_labels()
        fig.legend(handles,labels,loc="upper center",bbox_to_anchor=(.5,1.06),ncol=2,frameon=False)
        fig.suptitle("Pure ICM throughout · same alignment data, change only SIGReg sampling\nNo OU handover; affine display normalization per model")
        save(fig,f"pure_icm_sampling_{seeds[start]}_{seeds[start+4]}")
    return summary


def scale_budget(seeds):
    rows = []
    for seed in seeds:
        base = ordinary_profile(4,seed)["turning_distances"]
        if len(base)!=1:
            continue
        for gain in (2,8):
            target = ordinary_profile(gain,seed)["turning_distances"]
            frozen = base[0]*gain/4
            actual = target[0] if len(target)==1 else None
            input_term = float(np.log(gain/4))
            adaptation = None if actual is None else float(np.log(actual/frozen))
            total = None if actual is None else float(np.log(actual/base[0]))
            residual = None if actual is None else total-input_term-adaptation
            rows.append(dict(seed=seed,gain=gain,base_turn=base[0],frozen_turn=frozen,
                retrained_turn=actual,log_input_effect=input_term,log_weight_adaptation=adaptation,
                log_total_movement=total,identity_residual=residual,
                match_note="single extremum" if actual is not None else "no unique extremum in [1,16]"))
    trials.atomic_json(OUT/"scale-movement.json",rows)
    write_csv(OUT/"scale-movement.csv",rows)
    fig,axes = plt.subplots(1,2,figsize=(11,4),layout="constrained")
    for ax,gain in zip(axes,(2,8)):
        chosen = [r for r in rows if r["gain"]==gain]
        for i,r in enumerate(chosen):
            ax.plot([r["base_turn"],r["frozen_turn"]],[i,i],color=".75",lw=2)
            ax.scatter(r["base_turn"],i,color=".5",marker="s",label="Original gain-4 turn" if i==0 else None)
            ax.scatter(r["frozen_turn"],i,color="#0072B2",label="Change gain, freeze weights" if i==0 else None)
            if r["retrained_turn"] is not None:
                ax.plot([r["frozen_turn"],r["retrained_turn"]],[i,i],color="#D55E00",lw=2)
                ax.scatter(r["retrained_turn"],i,color="#D55E00",marker="D",label="After retraining" if i==0 else None)
            else:
                ax.annotate("no retrained extremum",(r["frozen_turn"],i),xytext=(5,7),textcoords="offset points",fontsize=7)
        ax.set_yticks(range(len(chosen)),[str(r["seed"]) for r in chosen])
        ax.set(xlabel="Physical turning distance",ylabel="Seed",title=f"Gain 4 → {gain}")
        ax.axvspan(0,1,color=".8",alpha=.3)
        ax.grid(alpha=.15)
    handles,labels = axes[1].get_legend_handles_labels()
    fig.legend(handles,labels,loc="upper center",bbox_to_anchor=(.5,1.08),ncol=3,frameon=False)
    fig.suptitle("Separate the exact input-scale shift from learned weight adaptation\nGray below d=1 is outside the evaluation interval; missing roots are not forced matches")
    save(fig,"scale_and_weight_adaptation")
    return rows


def training_analysis():
    update_rows, motion_rows, history_rows, events = [], [], [], []
    for seed in DIAGNOSTIC_SEEDS:
        for gain in GAINS:
            for condition in COLORS:
                run = load_run(condition,gain,seed)
                factor,_ = normalize(run)
                q = run["log_distance"]
                observed_states = {}
                for frame in run["frames"]:
                    observed_states[frame["step"]] = roots(q,frame["slope_before"])
                    observed_states[frame["step"]+1] = roots(q,frame["slope_after"])
                observed_states[4000] = roots(q,run["final_slope"])
                first = next((step for step in sorted(observed_states) if observed_states[step]),None)
                last_none = None if first is None else max((step for step in observed_states if step<first and not observed_states[step]),default=None)
                events.append(dict(seed=seed,gain=gain,condition=condition,
                    inherited_root=bool(observed_states[0]),first_observed_root_step=first,
                    last_observed_no_root_step=last_none,
                    first_observed_turns=[] if first is None else [float(np.exp(v)) for v in observed_states[first]],
                    turns_at_selected_steps={str(step):[float(np.exp(v)) for v in observed_states[step]] for step in (0,100,250,500,1000,2000,3000,4000)}))
                for frame in run["frames"]:
                    s = frame["slope_before"]
                    before_roots = roots(q,s)
                    after_roots = roots(q,frame["slope_after"])
                    history_rows.append(dict(seed=seed,gain=gain,condition=condition,step=frame["step"],
                        extrema_before=len(before_roots),turns_before=[float(np.exp(x)) for x in before_roots]))
                    observed = (frame["slope_after"]-s)*factor
                    predicted = (frame["delta_slope_a"]+frame["delta_slope_r"])*factor
                    residual = observed-predicted
                    signal = float(np.linalg.norm(observed))
                    update_rows.append(dict(seed=seed,gain=gain,condition=condition,step=frame["step"],
                        actual_slope_change_norm=signal,taylor_residual_norm=float(np.linalg.norm(residual)),
                        relative_taylor_error=float(np.linalg.norm(residual))/max(signal,1e-12),
                        parameter_update_error=frame["update_reconstruction_error"],
                        moment_error=frame["moment_reconstruction_error"],
                        alignment_slope_effect_norm=float(np.linalg.norm(frame["delta_slope_a"]*factor)),
                        sigreg_slope_effect_norm=float(np.linalg.norm(frame["delta_slope_r"]*factor))))
                    curvature = np.gradient(s,q)
                    for root in before_roots:
                        nearby = [x for x in after_roots if abs(x-root)<.05]
                        denominator = float(np.interp(root,q,curvature))
                        simple = abs(denominator)>1e-3*max(float(np.std(frame["curve_before"])),1e-8)
                        if len(nearby)!=1 or not simple:
                            motion_rows.append(dict(seed=seed,gain=gain,condition=condition,step=frame["step"],
                                before_turn=float(np.exp(root)),matched=False,alignment_log_shift=None,
                                sigreg_log_shift=None,actual_log_shift=None,predicted_log_shift=None,
                                reason="unmatched/ambiguous root" if len(nearby)!=1 else "near-degenerate root"))
                            continue
                        a = -float(np.interp(root,q,frame["delta_slope_a"]))/denominator
                        r = -float(np.interp(root,q,frame["delta_slope_r"]))/denominator
                        motion_rows.append(dict(seed=seed,gain=gain,condition=condition,step=frame["step"],
                            before_turn=float(np.exp(root)),matched=True,alignment_log_shift=a,
                            sigreg_log_shift=r,actual_log_shift=nearby[0]-root,predicted_log_shift=a+r,
                            reason="locally matched simple root"))
    write_csv(OUT/"slope-updates.csv",update_rows)
    write_csv(OUT/"root-motion.csv",motion_rows)
    trials.atomic_json(OUT/"root-history.json",history_rows)
    trials.atomic_json(OUT/"root-events.json",events)
    matched = [r for r in motion_rows if r["matched"]]
    active = [r for r in update_rows if r["actual_slope_change_norm"]>1e-6]
    accuracy = dict(
        probed_updates=len(update_rows),matched_roots=len(matched),unmatched_or_degenerate_roots=len(motion_rows)-len(matched),
        max_parameter_update_error=max(r["parameter_update_error"] for r in update_rows),
        max_moment_error=max(r["moment_error"] for r in update_rows),
        median_relative_taylor_error_all=float(np.median([r["relative_taylor_error"] for r in update_rows])),
        active_update_count=len(active),active_signal_threshold=1e-6,
        active_relative_taylor_error_quantiles=np.quantile([r["relative_taylor_error"] for r in active],[.5,.9,.99,1]).tolist(),
        root_prediction_absolute_error_quantiles=np.quantile([abs(r["actual_log_shift"]-r["predicted_log_shift"]) for r in matched],[.5,.9,.99,1]).tolist(),
        note="All rows retained; active threshold avoids relative errors dominated by rounding when the update is nearly zero")
    trials.atomic_json(OUT/"local-attribution-accuracy.json",accuracy)
    fig,axes = plt.subplots(1,2,figsize=(11,4),layout="constrained")
    for condition in COLORS:
        selected = [r for r in matched if r["condition"]==condition]
        axes[0].scatter([r["actual_log_shift"] for r in selected],[r["predicted_log_shift"] for r in selected],color=COLORS[condition],s=8,alpha=.4,label=condition)
        selected = [r for r in update_rows if r["condition"]==condition]
        axes[1].scatter([r["step"] for r in selected],[max(r["relative_taylor_error"],1e-8) for r in selected],color=COLORS[condition],s=5,alpha=.3,label=condition)
    limits = axes[0].get_xlim()
    axes[0].plot(limits,limits,"k--",lw=.7)
    axes[0].set(xlabel="Observed one-step change in log turning distance",ylabel="Predicted change",title="Simple, locally matched roots only")
    axes[1].set_yscale("log")
    axes[1].set(xlabel="Training update",ylabel="Relative slope-change Taylor error",title="Approximation failures and tiny-update rounding retained")
    for ax in axes:
        ax.grid(alpha=.15)
        ax.legend(frameon=False)
    save(fig,"local_attribution_accuracy")
    return accuracy


def slope_heatmaps(seed=131):
    runs = [load_run(condition,gain,seed) for condition in COLORS for gain in GAINS]
    values = []
    for run in runs:
        factor,_ = normalize(run)
        values.append(np.stack([f["slope_before"]*factor for f in run["frames"]]))
    limit = float(np.quantile(np.abs(np.concatenate([x.ravel() for x in values])),.99))
    fig,axes = plt.subplots(2,3,figsize=(13,7),layout="constrained")
    for ax,run,matrix in zip(axes.flat,runs,values):
        q = run["log_distance"]
        steps = [f["step"] for f in run["frames"]]
        image = ax.pcolormesh(np.exp(q),steps,matrix,cmap="RdBu",vmin=-limit,vmax=limit,shading="nearest",rasterized=True)
        for frame in run["frames"]:
            for root in roots(q,frame["slope_before"]):
                ax.scatter(np.exp(root),frame["step"],s=1,color="black")
        distance_axis(ax)
        ax.set(title=f"{run['condition']} SIGReg · gain {run['gain']}",ylabel="Updates completed")
    fig.colorbar(image,ax=list(axes.flat),label="Signed dh / d(log d), fixed final affine scale; clipped at 99th percentile")
    fig.suptitle(f"Seed {seed}: when and where slope reversals appear\nBlack dots: derivative roots; raw function and fixed-final-scale derivatives saved every 25 updates")
    save(fig,f"slope_evolution_seed{seed}")


def component_plots(seed=131,gain=4,condition="ordinary"):
    run = load_run(condition,gain,seed)
    factor,_ = normalize(run)
    frames = {f["step"]:f for f in run["frames"]}
    fig,axes = plt.subplots(2,3,figsize=(13,7),layout="constrained")
    for ax,step in zip(axes.flat,(0,25,100,250,500,1000)):
        f = frames[step]
        d = np.exp(run["log_distance"])
        a,r = f["delta_slope_a"]*factor,f["delta_slope_r"]*factor
        observed = (f["slope_after"]-f["slope_before"])*factor
        ax.plot(d,a,color="#0072B2",label="Alignment moment contribution")
        ax.plot(d,r,color="#D55E00",label="SIGReg moment contribution")
        ax.plot(d,a+r,color=".5",label="Predicted sum")
        ax.plot(d,observed,"k--",lw=1,label="Actual one-step slope change")
        ax.axhline(0,color="black",lw=.5)
        distance_axis(ax)
        ax.set(title=f"Update {step} → {step+1}",ylabel="Change in signed slope")
    handles,labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles,labels,loc="upper center",bbox_to_anchor=(.5,1.08),ncol=2,frameon=False)
    fig.suptitle(f"Seed {seed} · gain {gain} · {condition} sampling\nAdam numerator split; shared actual second-moment denominator; local attribution, not separate optimizers")
    save(fig,f"loss_components_seed{seed}_g{gain}_{condition}")


def birth_plot():
    event = trials.load(OUT/"birth-event.pt")
    run = load_run("ordinary",4,131)
    c = trials.behavior.Config(**run["config"])
    fig,axes = plt.subplots(1,2,figsize=(11,4),layout="constrained")
    q = torch.linspace(np.log(12),np.log(16),513,requires_grad=True)
    for key,label,color,style in (
        ("before","Before update 205",".5","--"),
        ("alignment_only","Alignment-moment component only","#0072B2","-"),
        ("sigreg_only","SIGReg-moment component only","#D55E00","-"),
        ("after","Actual combined update","black",":"),
    ):
        model = ScaledEncoder(4)
        model.load_state_dict(event[key])
        y = model(trials.behavior.sensor(q.exp(),c))
        slope = torch.autograd.grad(y.sum(),q)[0].detach().numpy()
        axes[0].plot(q.detach().exp().numpy(),slope,label=label,color=color,ls=style)
    axes[0].axhline(0,color="black",lw=.7)
    axes[0].set(xlabel="Physical distance (far-end zoom)",ylabel="Raw dh / d(log d)",
                title="Which one-step component introduces the reversal?")
    axes[0].legend(fontsize=7,frameon=False)
    history = []
    for frame in run["frames"]:
        for root in roots(run["log_distance"],frame["slope_before"]):
            history.append((frame["step"],np.exp(root)))
    history.append((4000,np.exp(roots(run["log_distance"],run["final_slope"])[0])))
    axes[1].plot([x[0] for x in history],[x[1] for x in history],color="black",lw=1)
    first = event["event"]["after"]["turning_distances"][0]
    axes[1].scatter(205,first,color="#D55E00",label=f"First in-range turn: {first:.2f}",s=40)
    axes[1].axhspan(2,4,color=".8",alpha=.3,label="Input-scale region 2–4")
    axes[1].set(xlabel="Updates completed",ylabel="Physical turning distance",ylim=(1,16.5),
                title="The turn then migrates inward during training")
    axes[1].legend(frameon=False,fontsize=8)
    for ax in axes:
        ax.grid(alpha=.15)
    fig.suptitle("Seed 131 · gain 4 · same before-state and shared optimizer denominator\nOne-step parameter interventions; root location is independent of global output sign")
    save(fig,"birth_and_inward_migration")


def main():
    torch.set_num_threads(1)
    settings = json.loads((OUT/"settings.json").read_text())
    seeds = settings["seeds"]
    if not all((OUT/f"completed-{s}.json").exists() for s in seeds):
        raise RuntimeError("Mechanism cohort incomplete")
    plt.rcParams.update({"font.size":9,"axes.spines.top":False,"axes.spines.right":False})
    summary = cohort(seeds)
    scale_budget(seeds)
    accuracy = training_analysis()
    for seed in DIAGNOSTIC_SEEDS:
        slope_heatmaps(seed)
    component_plots()
    birth_plot()
    print(json.dumps(dict(cohort=summary,accuracy=accuracy),indent=2))


if __name__ == "__main__":
    main()
