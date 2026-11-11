"""Small tabular JEPA; gradients through both encoder endpoints, as in the paper."""
import torch
from torch import nn


class JEPA(nn.Module):
    def __init__(self, input_dim, latent_dim, weather_dim):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(input_dim, 64), nn.GELU(),
                                     nn.Linear(64, 64), nn.GELU(), nn.Linear(64, latent_dim))
        self.predictor = nn.Sequential(nn.Linear(latent_dim + weather_dim, 64), nn.GELU(),
                                       nn.Linear(64, 64), nn.GELU(), nn.Linear(64, latent_dim))

    def forward(self, x, y, weather):
        z, target = self.encoder(x), self.encoder(y)
        prediction = self.predictor(torch.cat([z, weather], dim=-1))
        return z, target, prediction


def sigreg(z, projections=32):
    """Sliced Gaussian characteristic-function penalty, separately per endpoint.

    Formula follows upstream jepa/losses.py at 2964fa8 (MIT; see NOTICE).
    z: (2, batch, latent). Reduced projections are a recorded pilot variant.
    """
    directions = torch.randn(z.shape[-1], projections, device=z.device)
    directions = directions / directions.norm(dim=0, keepdim=True).clamp_min(1e-8)
    t = torch.linspace(0, 3, 17, device=z.device)
    target = torch.exp(-t.square() / 2)
    weights = torch.full_like(t, 2 * 3 / 16)
    weights[[0, -1]] *= 0.5
    angles = (z @ directions).unsqueeze(-1) * t
    error = (angles.cos().mean(1) - target).square() + angles.sin().mean(1).square()
    return ((error @ (weights * target)) * z.shape[1]).mean()
