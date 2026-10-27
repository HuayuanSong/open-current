"""The evaluated v1 residual networks, retained in the frozen ensemble."""
import torch
from torch import nn
L=168
H=48

class ResidualBlock(nn.Module):
    def __init__(self, inputs, hidden, outputs, dropout=0.1):
        super().__init__()
        self.main = nn.Sequential(nn.Linear(inputs, hidden), nn.ReLU(), nn.Linear(hidden, outputs), nn.Dropout(dropout))
        self.skip = nn.Linear(inputs, outputs)
        self.norm = nn.LayerNorm(outputs)

    def forward(self, x):
        return self.norm(self.main(x) + self.skip(x))

class WeatherMLP(nn.Module):
    """TiDE-inspired model adapted for origin-specific future NWP covariates.

    Unlike the paper, no past weather is fed to the encoder. Each sample has
    its own issued future weather; a stitched perfect-weather series is avoided.
    """
    def __init__(self, features, width, target):
        super().__init__()
        self.project = ResidualBlock(features, 32, 8)
        self.encoder = nn.Sequential(ResidualBlock(L + H * 8 + 2, width, width),
                                     ResidualBlock(width, width, width))
        self.decoder = ResidualBlock(width, width, H * 8)
        self.temporal = nn.Sequential(nn.Linear(16, 32), nn.ReLU(), nn.Linear(32, 1))
        self.linear = nn.Linear(L, H)
        # Start from an appropriate deterministic baseline; learn corrections.
        with torch.no_grad():
            self.linear.weight.zero_()
            self.linear.bias.zero_()
            for lead in range(H):
                index = L - 1 if target == 0 else (L - 24 + lead % 24 if target == 1 else lead)
                self.linear.weight[lead, index] = 1

    def forward(self, history, future, area):
        projected = self.project(future)
        encoded = self.encoder(torch.cat([history, projected.flatten(1), area], dim=1))
        decoded = self.decoder(encoded).reshape(-1, H, 8)
        residual = self.temporal(torch.cat([decoded, projected], dim=-1)).squeeze(-1)
        return self.linear(history) + residual
