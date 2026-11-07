"""Scalar observation encoder used by the Predictive Closure extension.

The architecture and state-dictionary names match the saved study checkpoints.
"""
import torch
from torch import nn


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(6, 48), nn.Tanh(),
                                 nn.Linear(48, 48), nn.Tanh(), nn.Linear(48, 1))

    def forward(self, x):
        # Bounded sensor coordinates improve conditioning. This supplies neither
        # log-distance nor an inverse-distance formula to the network.
        return self.net(torch.tanh(4 * x)).flatten()
