import torch
import torch.nn as nn
from typing import Callable


def odeint_euler(func: Callable, x0: torch.Tensor, tau_span: torch.Tensor) -> torch.Tensor:
    device = x0.device
    tau_span = tau_span.to(device)
    x = x0
    trajectory = [x]
    for i in range(len(tau_span) - 1):
        dtau = tau_span[i + 1] - tau_span[i]
        x = x + dtau * func(tau_span[i], x)
        trajectory.append(x)
    return torch.stack(trajectory)


def odeint_rk4(func: Callable, x0: torch.Tensor, tau_span: torch.Tensor) -> torch.Tensor:
    device = x0.device
    tau_span = tau_span.to(device)
    x = x0
    trajectory = [x]
    for i in range(len(tau_span) - 1):
        tau = tau_span[i]
        dtau = tau_span[i + 1] - tau
        k1 = func(tau, x)
        k2 = func(tau + dtau / 2, x + dtau / 2 * k1)
        k3 = func(tau + dtau / 2, x + dtau / 2 * k2)
        k4 = func(tau + dtau, x + dtau * k3)
        x = x + (dtau / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
        trajectory.append(x)
    return torch.stack(trajectory)


class SinusoidalEmbedding(nn.Module):
    def __init__(self, tau_dim: int):
        super().__init__()
        if tau_dim < 4 or tau_dim % 2 != 0:
            raise ValueError("tau_dim must be even and >= 4")
        self.tau_dim = tau_dim

    def forward(self, tau: torch.Tensor) -> torch.Tensor:
        device = tau.device
        half = self.tau_dim // 2
        emb = torch.log(torch.tensor(10000.0, device=device)) / (half - 1)
        emb = torch.exp(torch.arange(half, device=device) * -emb)
        emb = tau.unsqueeze(1) * emb.unsqueeze(0)
        return torch.cat((torch.sin(emb), torch.cos(emb)), dim=1)


class FiLMLayer(nn.Module):
    def __init__(self, z_dim: int, h_dim: int, h_dims=None):
        super().__init__()
        if h_dims is None:
            h_dims = [max(z_dim, h_dim // 2)]
        layers = []
        prev = z_dim
        for d in h_dims:
            layers.extend([nn.Linear(prev, d), nn.SiLU()])
            prev = d
        layers.append(nn.Linear(prev, h_dim * 2))
        self.film_net = nn.Sequential(*layers)
        nn.init.zeros_(self.film_net[-1].weight)
        nn.init.zeros_(self.film_net[-1].bias)

    def forward(self, h: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        lambda_param, gamma = self.film_net(z).chunk(2, dim=-1)
        return h * (1 + lambda_param) + gamma


class FiLMResidualBlock(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, tau_dim: int, z_dim: int, dropout: float = 0.1):
        super().__init__()
        self.norm = nn.LayerNorm(in_dim)
        self.linear = nn.Linear(in_dim, out_dim)
        self.act = nn.SiLU()
        self.dropout = nn.Dropout(dropout)
        self.film = FiLMLayer(z_dim, in_dim)
        self.tau_proj = nn.Linear(tau_dim, in_dim * 3)
        self.residual_proj = nn.Linear(in_dim, out_dim) if in_dim != out_dim else None

        nn.init.normal_(self.tau_proj.weight, std=0.01)
        nn.init.zeros_(self.tau_proj.bias)
        if self.residual_proj is not None:
            nn.init.normal_(self.residual_proj.weight, std=1e-3)
            nn.init.zeros_(self.residual_proj.bias)

    def forward(self, h: torch.Tensor, tau_emb: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        h_norm = self.norm(h)
        h_film = self.film(h_norm, z)
        tau_params = self.tau_proj(tau_emb)
        scale, shift, gate_logits = tau_params.chunk(3, dim=1)
        gate = torch.sigmoid(gate_logits)
        h_time = h_film * (1 + torch.tanh(scale)) + shift
        h_mod = (1 - gate) * h_film + gate * h_time
        out = self.dropout(self.act(self.linear(h_mod)))
        residual = h if self.residual_proj is None else self.residual_proj(h)
        return residual + out


class ResidualEncoderBlock(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.0):
        super().__init__()
        self.norm = nn.LayerNorm(in_dim)
        self.linear = nn.Linear(in_dim, out_dim)
        self.act = nn.SiLU()
        self.dropout = nn.Dropout(dropout)
        self.residual_proj = nn.Linear(in_dim, out_dim) if in_dim != out_dim else None
        if self.residual_proj is not None:
            nn.init.normal_(self.residual_proj.weight, std=1e-3)
            nn.init.zeros_(self.residual_proj.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_norm = self.norm(x)
        out = self.dropout(self.act(self.linear(x_norm)))
        residual = x if self.residual_proj is None else self.residual_proj(x)
        return residual + out