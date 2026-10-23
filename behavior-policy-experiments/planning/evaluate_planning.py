"""Goal-reaching MPC with frozen encoders, common or policy-native action data.

No distance labels enter the predictor or planner. True distances are used by
simulation and scoring only. OU pairs never receive fabricated action labels.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "reliability"))
import run_study as study
from policies import behavior
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class Predictor(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(4, 64), nn.ELU(), nn.Linear(64, 64), nn.ELU(), nn.Linear(64, 1))

    def forward(self, y, action):
        inputs = torch.cat([y[:, None], F.one_hot(action, 3).float()], -1)
        return y + self.net(inputs).flatten()


@torch.no_grad()
def encode(encoder, x, mean, scale):
    return (encoder(x) - mean) / scale


def train_predictor(encoder, data, common, seed, updates):
    torch.manual_seed(1200000 + seed)
    model = Predictor()
    optimizer = torch.optim.Adam(model.parameters(), lr=.001)
    with torch.no_grad():
        calibration = encoder(common["x"])
        mean, scale = float(calibration.mean()), float(calibration.std(unbiased=False).clamp_min(1e-6))
        y, yp = encode(encoder, data["x"], mean, scale), encode(encoder, data["xp"], mean, scale)
    generator = torch.Generator().manual_seed(1300000 + seed)
    for _ in range(updates):
        ix = torch.randint(len(y), (256,), generator=generator)
        loss = (model(y[ix], data["action"][ix]) - yp[ix]).square().mean()
        optimizer.zero_grad(); loss.backward(); optimizer.step()
    return model.eval(), mean, scale


def goals(c, seed, cases):
    generator = torch.Generator().manual_seed(1400000 + seed)
    start = 3 + 11 * torch.rand(cases, generator=generator)
    offsets = torch.randint(-8, 9, (cases,), generator=generator)
    offsets[offsets == 0] = 1
    return start, start + c.move * offsets


def candidate_bank(seed, count=128, horizon=8):
    generator = torch.Generator().manual_seed(1500000 + seed)
    bank = torch.randint(0, 3, (count, horizon), generator=generator)
    if count < 2 * horizon + 1:
        raise ValueError('Candidate bank must include each reachable net displacement')
    # Include a direct path to every integer displacement, then hold still.
    # This prevents a random terminal-only bank from selecting detours.
    for index, displacement in enumerate(range(-horizon, horizon + 1)):
        bank[index].fill_(1)
        bank[index, :abs(displacement)] = 0 if displacement < 0 else 2
    return bank


@torch.no_grad()
def choose_actions(predictor, y, goal, bank):
    count = len(bank)
    states = y[:, None].expand(-1, count).reshape(-1)
    stage_cost = torch.zeros_like(states)
    repeated_goal = goal.repeat_interleave(count)
    for step in range(bank.shape[1]):
        action = bank[:, step].repeat(len(y))
        states = predictor(states, action)
        stage_cost += (states - repeated_goal).square()
    cost = (states.reshape(len(y), count) - goal[:, None]).square()
    cost += .01 * stage_cost.reshape(len(y), count)
    # Tiny fixed tie-break cost; no true distance or state enters selection.
    cost += 1e-8 * (bank.float() - 1).square().sum(1)[None]
    return bank[cost.argmin(1), 0]


@torch.no_grad()
def rollout(encoder, predictor, mean, scale, c, seed, cases=32):
    start, goal = goals(c, seed, cases)
    distance = start.clone()
    goal_y = encode(encoder, behavior.sensor(goal, c), mean, scale)
    bank = candidate_bank(seed)
    path = [distance.numpy().copy()]
    for _ in range(12):
        y = encode(encoder, behavior.sensor(distance, c), mean, scale)
        action = choose_actions(predictor, y, goal_y, bank)
        distance = behavior.move(distance, action, c)
        path.append(distance.numpy().copy())
    # A known-dynamics greedy reference verifies every test goal is reachable.
    oracle = start.clone()
    for _ in range(12):
        action = torch.where(goal - oracle > c.move / 2, 2, torch.where(oracle - goal > c.move / 2, 0, 1)).long()
        oracle = behavior.move(oracle, action, c)
    error = (distance - goal).abs()
    return {"mean_terminal_distance_error": float(error.mean()),
            "median_terminal_distance_error": float(error.median()),
            "success_within_one_step": float((error <= c.move + 1e-5).float().mean()),
            "oracle_mean_error": float((oracle - goal).abs().mean()),
            "stay_mean_error": float((start - goal).abs().mean())}, {
                "start": start.numpy(), "goal": goal.numpy(), "terminal": distance.numpy(), "path": np.stack(path)}


def discover(root, source):
    if source == "legacy":
        folder = root / "curriculum-controls" / "output"
        result = json.loads((folder / "metrics.json").read_text())
        return [{"seed": r["seed"], "condition": name, "checkpoint": folder / f"encoder_{name}_seed{r['seed']}.pt",
                 "native_kind": name.split('_')[0], "native_path": None} for r in result["runs"]
                for name in ["ou_carry", "random_carry", "icm_carry"]]
    folder = root / "reliability" / f"output-{source}"
    result = json.loads((folder / "metrics.json").read_text())
    chosen = {"ou", "random", "icm", "icm_epsilon", "icm_coverage", "monotone"} if source == "repairs" else set(study.POLICY_CONDITIONS)
    return [{"seed": r["seed"], "condition": r["condition"], "checkpoint": folder / r["checkpoint"],
             "native_kind": study.source_name(r["condition"]),
             "native_path": folder / "sources" / f"{study.source_name(r['condition'])}_seed{r['seed']}.pt"}
            for r in result["runs"] if r["condition"] in chosen]


def native_data(record, c, root):
    if record["native_kind"] == "ou":
        return None
    if record["native_path"] is not None:
        saved = torch.load(record["native_path"], map_location="cpu")
        return saved["data"]
    if record["native_kind"] == "random":
        return behavior.collect(c, "random", c.train_episodes, 10000 + record["seed"])
    saved = torch.load(root / "curriculum-controls" / "output" / f"behavior_seed{record['seed']}.pt", map_location="cpu")
    policy = behavior.Policy(); policy.load_state_dict(saved["policy"]); policy.eval()
    return behavior.collect(c, "icm", c.train_episodes, 10000 + record["seed"], policy)


def write_report(out, result):
    (out / "metrics.json").write_text(json.dumps(result, indent=2) + '\n')
    lines = ["# Planning evaluation", "", "Frozen encoders; observation/action-trained predictors; identical reachable goals and candidate sequences. "
             "Each run uses 32 cases, 12 replanning steps, horizon 8 and 128 candidate sequences. "
             "The planner receives only current and goal embeddings, never true distance.", ""]
    if result['quick']: lines += ["**Execution check only; not trained planning performance.**", ""]
    lines += ["| Encoder | Predictor data | Seeds | Terminal distance error ↓ | Within 0.25 ↑ | One-step latent MSE ↓ |", "| --- | --- | ---: | ---: | ---: | ---: |"]
    rows = result['runs']
    for name, data_kind in sorted({(r['condition'], r['predictor_data']) for r in rows}):
        records = [r for r in rows if r['condition'] == name and r['predictor_data'] == data_kind]
        values = [[r['metrics'][key] for r in records] for key in ['mean_terminal_distance_error','success_within_one_step','heldout_latent_mse']]
        lines.append(f"| {name} | {data_kind} | {len(records)} | " + ' | '.join(study.controls.summary(v) for v in values) + ' |')
    lines += ["", "Common data holds predictor-training coverage fixed to compare representations. Native data also changes coverage and tests the full pipeline. "
              "Native OU is omitted because passive OU pairs have no physical action labels. A lower latent prediction loss need not mean better physical planning. "
              "Goals are local and reachable; this is an initial short-horizon planning test, not arbitrary-distance or uncertainty-aware planning. "
              "Cases within a seed share a trained model; averages across seeds, not case-level significance tests, are reported.", ""]
    (out / 'RESULTS.md').write_text('\n'.join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', choices=['legacy','repairs','policies'], default='legacy')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--quick', action='store_true')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--evaluate-only', action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(1)
    root = Path(__file__).resolve().parents[1]
    out = args.output or Path(__file__).parent / f'output-{args.source}'
    out.mkdir(parents=True, exist_ok=True)
    result = {'source': args.source, 'quick': args.quick, 'predictor_updates': 20 if args.quick else 1000, 'planner_version': 2, 'runs': []}
    if args.resume or args.evaluate_only:
        result = json.loads((out / 'metrics.json').read_text())
        if result['source'] != args.source or result['quick'] != args.quick or result.get('planner_version') != 2:
            raise RuntimeError('Planning resume configuration differs')
    records = discover(root, args.source)
    if args.quick: records = records[:1]
    for record in records:
        seed = record['seed']
        saved = torch.load(record['checkpoint'], map_location='cpu')
        c = behavior.Config(**saved['config'])
        encoder = study.MonotoneEncoder() if record['condition'] == 'monotone' else behavior.Encoder()
        encoder.load_state_dict(saved['encoder']); encoder.eval()
        common = behavior.collect(c, 'random', c.train_episodes, 510000 + seed)
        heldout = behavior.collect(c, 'random', c.eval_episodes, 520000 + seed)
        native = native_data(record, c, root)
        for kind, data in [('common', common), ('native', native)]:
            if data is None: continue
            existing = next((r for r in result['runs'] if r['seed'] == seed and r['condition'] == record['condition'] and r['predictor_data'] == kind), None)
            if existing is not None and not args.evaluate_only: continue
            filename = f"predictor_{record['condition']}_{kind}_seed{seed}.pt"
            if args.evaluate_only:
                state = torch.load(out / filename, map_location='cpu')
                predictor = Predictor(); predictor.load_state_dict(state['predictor']); predictor.eval()
                mean, scale = state['mean'], state['scale']
            else:
                predictor, mean, scale = train_predictor(encoder, data, common, seed, result['predictor_updates'])
                torch.save({'predictor': predictor.state_dict(), 'mean': mean, 'scale': scale,
                            'encoder_checkpoint': str(record['checkpoint'].relative_to(root)), 'config': vars(c)}, out / filename)
            metrics, paths = rollout(encoder, predictor, mean, scale, c, seed)
            with torch.no_grad():
                y = encode(encoder, heldout['x'], mean, scale)
                yp = encode(encoder, heldout['xp'], mean, scale)
                metrics['heldout_latent_mse'] = float((predictor(y, heldout['action']) - yp).square().mean())
            if args.evaluate_only:
                for key in metrics:
                    if not np.isclose(metrics[key], existing['metrics'][key], rtol=0, atol=1e-7): raise RuntimeError('Planning evaluation changed')
            else:
                np.savez_compressed(out / filename.replace('.pt','.npz'), **paths)
                result['runs'].append({'seed': seed, 'condition': record['condition'], 'predictor_data': kind, 'metrics': metrics})
                write_report(out, result)
                print(json.dumps(result['runs'][-1]), flush=True)
    print(f"{'Reproduced' if args.evaluate_only else 'Completed'} {len(result['runs'])} planning evaluations.", flush=True)


if __name__ == '__main__': main()
