"""Small float64 algebra checks for the note; not a neural-network certificate."""
import itertools
import json
from pathlib import Path
import torch

torch.set_default_dtype(torch.float64)

def h(theta, q):
    return theta[0] * q + theta[1].square() * q.square() + theta[2] * q.pow(3)


def check():
    theta = torch.tensor([0.8, 0.3, -0.2], requires_grad=True)
    q = torch.tensor([-0.7, 0.1, 1.2])
    alpha = torch.tensor([0.6, 0.3, 0.1])
    freq = torch.tensor([0.4, 1.1, 1.8])
    weights = torch.tensor([0.2, 0.5, 0.3])
    target = (-freq.square() / 2).exp()
    scale = 7.0

    def moments(y, p):
        phase = y[:, None] * freq
        return (p[:, None] * phase.cos()).sum(0), (p[:, None] * phase.sin()).sum(0)

    def objective(y, p):
        c, s = moments(y, p)
        return scale * (weights * ((c - target).square() + s.square())).sum()

    y = h(theta, q)
    c, s = moments(y, alpha)
    phase = y[:, None] * freq
    force = 2 * scale * (weights * freq * (
        -(c - target) * phase.sin() + s * phase.cos())).sum(1)
    jac = torch.autograd.functional.jacobian(lambda z: h(z, q), theta)
    grad = torch.autograd.grad(objective(y, alpha), theta, create_graph=True)[0]
    force_error = (grad - (alpha[:, None] * force[:, None] * jac).sum(0)).abs().max()
    query = torch.tensor(0.9, requires_grad=True)
    slope = torch.autograd.grad(h(theta, query), query, create_graph=True)[0]
    mixed = torch.autograd.grad(slope, theta, create_graph=True)[0]
    integral = (alpha * force * (jac @ mixed)).sum()
    kernel_error = (integral - mixed @ grad).abs()

    left, right = torch.tensor([0, 1, 2]), torch.tensor([1, 2, 0])
    pair_weights = torch.tensor([0.2, 0.5, 0.3])
    alignment = (pair_weights * (y[left] - y[right]).square()).sum()
    pair_gradient = torch.autograd.grad(alignment, theta, retain_graph=True)[0]
    pair_formula = 2 * (pair_weights * (y[left] - y[right]) * ((jac[left] - jac[right]) @ mixed)).sum()
    pair_error = (pair_formula - mixed @ pair_gradient).abs()

    expected = torch.zeros(())
    for i, j in itertools.product(range(3), repeat=2):
        expected = expected + alpha[i] * alpha[j] * objective(y[[i, j]], torch.tensor([0.5, 0.5]))
    formula = 0.5 * objective(y, alpha) + scale / 2 * (weights * (1 + target.square() - 2 * target * c)).sum()
    expectation_error = (expected - formula).abs()
    expectation_gradient_error = (torch.autograd.grad(expected, theta, retain_graph=True)[0]
                                  - torch.autograd.grad(formula, theta)[0]).abs().max()

    gradients = torch.tensor([[1., -2., 0.5], [-3., 1., 2.], [2., 0.2, -1.]])
    beta = 0.9
    moment = torch.zeros(3)
    for g in gradients:
        moment = beta * moment + (1 - beta) * g
    expanded = (1 - beta) * sum(beta ** (2 - j) * gradients[j] for j in range(3))
    memory_error = (moment - expanded).abs().max()

    # Analytic segment bound: Hessian of u(q) in theta is diag(0,4q,0).
    # On q in [0,1], H <= 4; the chosen finite step creates a sign reversal.
    initial = torch.tensor([1., 0.1, 0.])
    update = torch.tensor([0., 0., -0.6])
    def u(z, x):
        return z[0] + 2 * z[1].square() * x + 3 * z[2] * x ** 2
    certificate = {}
    for name, x in [('negative_witness', 1.), ('positive_anchor', 0.)]:
        a = torch.autograd.functional.jacobian(lambda z: u(z, x), initial)
        linear = u(initial, x) + a @ update
        rho = 0.5 * abs(4 * x) * float(update.square().sum())
        actual = u(initial + update, x)
        assert abs(float(actual - linear)) <= rho + 1e-14
        certificate[name] = dict(linear=float(linear), remainder_bound=rho, actual=float(actual))
    assert certificate['negative_witness']['linear'] + certificate['negative_witness']['remainder_bound'] < 0
    assert certificate['positive_anchor']['linear'] - certificate['positive_anchor']['remainder_bound'] > 0

    errors = dict(force=float(force_error), kernel=float(kernel_error), alignment=float(pair_error),
                  iid_batch_expectation=float(expectation_error),
                  iid_batch_expected_gradient=float(expectation_gradient_error), adam_memory=float(memory_error))
    assert max(errors.values()) < 1e-11, errors
    return dict(date='2026-11-17', dtype='float64', algebra_errors=errors,
                analytic_polynomial_example=certificate,
                scope='Algebra and a polynomial example only; no validated MLP bound or future-training prediction.')


if __name__ == '__main__':
    result = check()
    output = Path(__file__).with_name('checks.json')
    output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))
