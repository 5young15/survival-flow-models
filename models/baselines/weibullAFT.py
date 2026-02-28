import torch
import torch.nn as nn
from typing import Optional
from models.interface import TorchSurvivalModel


class WeibullAFT(TorchSurvivalModel):
    def __init__(self, in_dim: int, config: Optional[dict] = None, **kwargs):
        super().__init__()
        self.config = config or {}
        hidden_dims = self.config.get('hidden_dims', [64, 32])
        dropout = self.config.get('dropout', 0.0)
        layers = []
        prev = in_dim
        for h in hidden_dims:
            layers.extend([nn.Linear(prev, h), nn.SELU(), nn.Dropout(dropout)])
            prev = h
        layers.append(nn.Linear(prev, 2))
        self.net = nn.Sequential(*layers)
        self.is_log_space = False

    def forward_loss(self, features: torch.Tensor, times: torch.Tensor, events: torch.Tensor, **kwargs):
        params = self.net(features)
        log_k = params[:, 0]
        log_lambda = params[:, 1]
        k = torch.exp(torch.clamp(log_k, -5, 5))
        lam = torch.exp(torch.clamp(log_lambda, -5, 5))
        t = torch.clamp(times, min=1e-6)
        t_lam = t / lam
        term_pow = torch.pow(t_lam, k)
        log_pdf = torch.log(k) - torch.log(lam) + (k - 1) * torch.log(t_lam) - term_pow
        log_surv = -term_pow
        log_lik = events * log_pdf + (1 - events) * log_surv
        neg_ll = -log_lik.mean()
        return neg_ll, {"neg_loglik": neg_ll.item()}

    def predict_risk(self, features: torch.Tensor, **kwargs) -> torch.Tensor:
        with torch.no_grad():
            params = self.net(features)
            log_lambda = torch.clamp(params[:, 1], -5, 5)
            return -log_lambda

    def predict_survival_function(self, features: torch.Tensor, time_grid: torch.Tensor = None, **kwargs) -> torch.Tensor:
        device = features.device
        with torch.no_grad():
            params = self.net(features)
            k = torch.exp(torch.clamp(params[:, 0], -5, 5)).unsqueeze(-1)
            lam = torch.exp(torch.clamp(params[:, 1], -5, 5)).unsqueeze(-1)
            if time_grid is None:
                t_max = 10.0
                time_grid = torch.linspace(0, t_max, 100, device=device)
            else:
                time_grid = time_grid.to(device)
            t = time_grid.unsqueeze(0)
            t_lam = t / lam
            return torch.exp(-torch.pow(t_lam, k))

    def predict_time(self, features: torch.Tensor, **kwargs) -> torch.Tensor:
        with torch.no_grad():
            params = self.net(features)
            k = torch.exp(torch.clamp(params[:, 0], -5, 5))
            lam = torch.exp(torch.clamp(params[:, 1], -5, 5))
            return lam * (torch.log(torch.tensor(2.0, device=features.device))) ** (1.0 / k)

    def compute_hazard_rate(self, features: torch.Tensor, time_grid: torch.Tensor, **kwargs) -> torch.Tensor:
        device = features.device
        time_grid = time_grid.to(device) if time_grid is not None else torch.linspace(1e-2, 10.0, 100, device=device)
        time_grid = torch.clamp(time_grid, min=1e-2)
        with torch.no_grad():
            params = self.net(features)
            k = torch.exp(torch.clamp(params[:, 0], -5, 5)).unsqueeze(-1)
            lam = torch.exp(torch.clamp(params[:, 1], -5, 5)).unsqueeze(-1)
            t = time_grid.unsqueeze(0)
            t_lam = t / lam
            h = (k / lam) * torch.pow(t_lam, k - 1)
            return torch.clamp(h, min=0.0, max=1000.0)