"""Measure policy attraction, reward geometry and matched training histories."""
import inspect
import json
import numpy as np
import torch
from run_study import old, OUT, HERE

B = old.trials.behavior


@torch.no_grad()
def diagnostic(policy, icm, c):
    d = torch.linspace(1, 16, 601)
    x = B.sensor(d, c)
    probs = policy(x)[0].probs
    successors = torch.stack([B.move(d, torch.full_like(d, a, dtype=torch.long), c) for a in range(3)], 1)
    errors = torch.stack([icm(x, B.sensor(successors[:, a], c), torch.full_like(d, a, dtype=torch.long))[0] for a in range(3)], 1)
    reward = (c.reward_scale * errors).clamp(max=1.)
    drift = (probs * (successors-d[:, None])).sum(1)
    # Frozen chain at the exact quarter-distance lattice, with reflected ends.
    lattice = torch.linspace(1, 16, 61)
    lp = policy(B.sensor(lattice, c))[0].probs.numpy().astype(np.float64)
    transition = np.zeros((61, 61))
    for i in range(61):
        for a in range(3):
            j = i+a-1
            j = -j if j < 0 else 120-j if j > 60 else j
            transition[i,j] += lp[i,a]
    transition /= transition.sum(1, keepdims=True)
    occupancy = np.ones(61)/61
    burn128 = None
    collected=[]
    for t in range(192):
        if t>=128:
            collected.append(occupancy.copy())
        occupancy = occupancy @ transition
        if t == 127:
            burn128 = occupancy.copy()
    matrix = transition.T - np.eye(61)
    matrix[-1] = 1
    rhs = np.zeros(61); rhs[-1] = 1
    stationary = np.linalg.solve(matrix, rhs)
    hitting = np.linalg.solve(np.eye(12)-transition[:12,:12], np.ones(12))
    # Instantaneous geometry is descriptive; reward is learned, not pixels.
    visual_step = torch.stack([(B.sensor(successors[:,a], c)-x).norm(dim=1) for a in range(3)], 1)
    return dict(d=d.numpy(), probs=probs.numpy(), drift=drift.numpy(),
        reward=reward.numpy(), visual_step=visual_step.numpy(),
        lattice=lattice.numpy(), chain_burn128=burn128,
        chain192=occupancy, stationary=stationary,
        chain_collection=np.mean(collected,axis=0),
        expected_steps_2_to_4=float(hitting[4]),
        stationary_residual=float(np.max(abs(stationary@transition-stationary))))


def main():
    torch.set_num_threads(1)
    c = B.Config(**json.loads((old.OUT/'settings.json').read_text())['config'])
    results = []
    for seed in range(130, 140):
        saved = old.trials.load(old.ORIGINAL/'sources'/f'icm-{seed}.pt')
        policy, icm = B.Policy(), B.ICM()
        policy.load_state_dict(saved['policy']); icm.load_state_dict(saved['icm'])
        state = diagnostic(policy.eval(), icm.eval(), c)
        distances = old.trials.distances(saved['data']['x']).numpy()
        state.update(seed=seed, stored_distances=distances, history=saved['history'])
        results.append(state)
    old.trials.save_torch(OUT/'policy-maps.pt', results)
    # Instrument the original trainer without changing any of its operations.
    # Every diagnostic preserves global RNG state; final equality is a gate.
    source = inspect.getsource(B.train_policy)
    anchor = '        if update % 25 == 0 or update == c.policy_updates - 1:'
    assert source.count(anchor) == 1
    instrumented = source.replace(anchor,
        '        if update in (0, 25, 100, 200, 400, 599):\n'
        '            capture(policy, icm, c, seed, update)\n' + anchor)
    for seed in (130, 131, 136):
        path = OUT / f'policy-trace-{seed}.pt'
        if path.exists():
            continue
        frames = []
        def capture(policy, icm, c, seed, update):
            with torch.random.fork_rng():
                state = diagnostic(policy, icm, c)
                data = B.collect(c, 'icm', 256, 10000+seed, policy)
                state.update(update=update, distances=old.trials.distances(data['x']).numpy())
                frames.append(state)
        namespace = dict(B.__dict__, capture=capture)
        exec(compile(instrumented, str(HERE/'instrumented-original-trainer'), 'exec'), namespace)
        policy, icm, history, _ = namespace['train_policy'](c, seed)
        saved = old.trials.load(old.ORIGINAL/'sources'/f'icm-{seed}.pt')
        maximum = max(float((v-saved['policy'][k]).abs().max()) for k,v in policy.state_dict().items())
        icm_error = max(float((v-saved['icm'][k]).abs().max()) for k,v in icm.state_dict().items())
        if maximum != 0 or icm_error != 0:
            raise RuntimeError(f'Instrumented policy did not reproduce {seed}: {maximum}, {icm_error}')
        old.trials.save_torch(path, dict(seed=seed, frames=frames,
            max_policy_error=maximum, max_icm_error=icm_error, original_source=source))
        print(f'policy trace {seed}: exact final reproduction', flush=True)


if __name__ == '__main__':
    main()
