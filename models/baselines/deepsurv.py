import torch
import torch.nn as nn
import warnings
from typing import Optional
from models.interface import TorchSurvivalModel


class DeepSurv(TorchSurvivalModel):
    def __init__(self, in_dim: int, config: Optional[dict] = None, **kwargs):
        super().__init__()
        self.config = config or {}
        hidden_dims = self.config.get("hidden_dims", [128, 128, 64, 32])
        dropout = self.config.get("dropout", 0.1)
        layers = []
        prev = in_dim
        for h in hidden_dims:
            layers.extend([nn.Linear(prev, h), nn.SELU(), nn.Dropout(dropout)])
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.risk_net = nn.Sequential(*layers)
        self._baseline_cum_haz = None
        self._baseline_times = None
        self._train_times = None
        self._train_events = None
        self._train_log_haz = None

    def forward_loss(self, features: torch.Tensor, times: torch.Tensor, events: torch.Tensor, **kwargs):
        log_haz = self.risk_net(features).squeeze(-1)
        sort_idx = torch.argsort(times)
        log_haz_s = log_haz[sort_idx]
        events_s = events[sort_idx]
        exp_haz = torch.exp(log_haz_s)
        risk_set = torch.flip(torch.cumsum(torch.flip(exp_haz, [0]), 0), [0])
        event_mask = events_s.bool()
        if not event_mask.any():
            return torch.tensor(0.0, device=features.device, requires_grad=True), {"neg_partial_ll": 0.0}
        log_lik = log_haz_s[event_mask] - torch.log(risk_set[event_mask] + 1e-10)
        neg_ll = -log_lik.mean()
        return neg_ll, {"neg_partial_ll": neg_ll.item()}

    def _fit_baseline_hazard(self, times: torch.Tensor, events: torch.Tensor, log_haz: torch.Tensor):
        self._baseline_times, self._baseline_cum_haz = super()._fit_breslow_baseline_hazard(times, events, log_haz)
        self._train_times = times
        self._train_events = events
        self._train_log_haz = log_haz

    def predict_risk(self, features: torch.Tensor, **kwargs) -> torch.Tensor:
        return self.risk_net(features).squeeze(-1)

    def predict_survival_function(self, features: torch.Tensor, time_grid: torch.Tensor = None, **kwargs) -> torch.Tensor:
        device = features.device
        with torch.no_grad():
            log_haz = self.predict_risk(features)
            exp_haz = torch.exp(torch.clamp(log_haz, max=88.0))
            if self._baseline_cum_haz is None:
                warnings.warn("Baseline hazard not fitted; returning S(t)=1", UserWarning)
                grid = time_grid.to(device) if time_grid is not None else torch.tensor([0.0], device=device)
                return torch.ones((features.shape[0], len(grid)), device=device)
            if self._baseline_cum_haz.device != device:
                self._baseline_cum_haz = self._baseline_cum_haz.to(device)
                self._baseline_times = self._baseline_times.to(device)
            grid = time_grid.to(device) if time_grid is not None else self._baseline_times
            indices = torch.searchsorted(self._baseline_times, grid)
            indices = torch.clamp(indices, 0, len(self._baseline_cum_haz) - 1)
            baseline_interp = self._baseline_cum_haz[indices]
            cum_haz = exp_haz.unsqueeze(1) * baseline_interp.unsqueeze(0)
            return torch.exp(-cum_haz)

    def predict_time(self, features: torch.Tensor, **kwargs) -> torch.Tensor:
        # 与 LinearCoxPH 完全一致的向量化实现
        device = features.device
        with torch.no_grad():
            if self._baseline_times is None:
                if self._train_times is not None:
                    self._fit_baseline_hazard(self._train_times, self._train_events, self._train_log_haz)
                else:
                    return torch.full((features.shape[0],), float('nan'), device=device)
            if self._baseline_times.device != device:
                self._baseline_times = self._baseline_times.to(device)
                self._baseline_cum_haz = self._baseline_cum_haz.to(device)
            t_max = self._baseline_times[-1] * 2
            grid = torch.linspace(0, t_max, 200, device=device)
            log_haz = self.predict_risk(features)
            exp_haz = torch.exp(log_haz)
            indices = torch.searchsorted(self._baseline_times, grid)
            indices = torch.clamp(indices, 0, len(self._baseline_cum_haz) - 1)
            baseline_interp = self._baseline_cum_haz[indices]
            S_grid = torch.exp(-(exp_haz.unsqueeze(1) * baseline_interp.unsqueeze(0)))
            idx = torch.searchsorted(-S_grid, torch.full((features.shape[0], 1), -0.5, device=device)).squeeze(1)
            medians = torch.zeros(features.shape[0], device=device)
            mid = (idx > 0) & (idx < 199)
            medians[idx == 0] = grid[0]
            medians[idx >= 199] = grid[-1]
            if mid.any():
                i_r = idx[mid]
                i_l = i_r - 1
                t1, t2 = grid[i_l], grid[i_r]
                s1, s2 = S_grid[mid, i_l], S_grid[mid, i_r]
                denom = s2 - s1
                mask_stable = torch.abs(denom) > 1e-10
                m_mid = torch.zeros_like(t1)
                m_mid[mask_stable] = t1[mask_stable] + (t2[mask_stable] - t1[mask_stable]) * (0.5 - s1[mask_stable]) / denom[mask_stable]
                m_mid[~mask_stable] = (t1[~mask_stable] + t2[~mask_stable]) / 2
                medians[mid] = m_mid
            return medians

    def compute_hazard_rate(self, features: torch.Tensor, time_grid: torch.Tensor, **kwargs) -> torch.Tensor:
        device = features.device
        time_grid = time_grid.to(device)
        with torch.no_grad():
            log_haz = self.predict_risk(features)
            exp_haz = torch.exp(torch.clamp(log_haz, max=88.0))
            if self._baseline_cum_haz is None:
                return torch.zeros((features.shape[0], len(time_grid)), device=device)
            if self._baseline_times.device != device:
                self._baseline_times = self._baseline_times.to(device)
                self._baseline_cum_haz = self._baseline_cum_haz.to(device)
            times_ext = torch.cat([torch.zeros(1, device=device), self._baseline_times])
            haz_ext = torch.cat([torch.zeros(1, device=device), self._baseline_cum_haz])
            dt = torch.diff(times_ext)
            dh = torch.diff(haz_ext)
            baseline_h = dh / torch.clamp(dt, min=1e-10)
            indices = torch.searchsorted(self._baseline_times, time_grid)
            indices = torch.clamp(indices, 0, len(baseline_h) - 1)
            h0_interp = baseline_h[indices]
            hazard = exp_haz.unsqueeze(1) * h0_interp.unsqueeze(0)
            return torch.clamp(hazard, min=0.0, max=1000.0)