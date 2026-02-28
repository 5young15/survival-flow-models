import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from models.interface import TorchSurvivalModel


class DeepHit(TorchSurvivalModel):
    def __init__(self, in_dim: int, config: Optional[dict] = None, **kwargs):
        super().__init__()
        self.config = config or {}
        self.num_bins = self.config.get('n_time_bins', 50)
        hidden_dims = self.config.get('hidden_dims', [128, 64, 32])
        dropout = self.config.get('dropout', 0.1)
        layers = []
        prev = in_dim
        for h in hidden_dims:
            layers.extend([nn.Linear(prev, h), nn.SELU(), nn.Dropout(dropout)])
            prev = h
        layers.append(nn.Linear(prev, self.num_bins))
        self.net = nn.Sequential(*layers)
        self.is_log_space = True

    def forward_loss(self, features: torch.Tensor, times: torch.Tensor, events: torch.Tensor, **kwargs):
        times_norm = self._to_normalized_time(times).float()
        times_01 = (times_norm + 4.0) / 8.0
        bin_indices = torch.floor(times_01 * self.num_bins).long().clamp(0, self.num_bins - 1)
        logits = self.net(features)
        probs = F.softmax(logits, dim=-1)
        event_mask = events.bool()
        event_prob = probs[range(len(times)), bin_indices] * event_mask.float()
        surv_prob = 1.0 - torch.cumsum(probs, dim=1)[:, bin_indices]
        censor_prob = surv_prob * (~event_mask).float()
        loglik = torch.log(event_prob + censor_prob + 1e-10)
        neg_ll = -loglik.mean()
        return neg_ll, {"neg_loglik": neg_ll.item()}

    def predict_risk(self, features: torch.Tensor, **kwargs) -> torch.Tensor:
        with torch.no_grad():
            logits = self.net(features)
            probs = F.softmax(logits, dim=-1)
            t_bins_norm = torch.linspace(-4.0, 4.0, self.num_bins, device=features.device)
            t_bins_raw = self._to_original_time(t_bins_norm)
            expected_t = torch.sum(probs * t_bins_raw[None, :], dim=1)
            return -expected_t

    def predict_survival_function(self, features: torch.Tensor, time_grid: torch.Tensor = None, **kwargs) -> torch.Tensor:
        device = features.device
        with torch.no_grad():
            logits = self.net(features)
            probs = F.softmax(logits, dim=-1)
            cum_probs = torch.cumsum(probs, dim=1)
            surv_points = F.pad(1.0 - cum_probs, (1, 0), value=1.0)
            if time_grid is None:
                return (1.0 - cum_probs).clamp(0.0, 1.0)
            time_grid = time_grid.to(device)
            time_grid_norm = self._to_normalized_time(time_grid)
            bin_edges_norm = torch.linspace(-4.0, 4.0, self.num_bins + 1, device=device)
            indices = torch.searchsorted(bin_edges_norm, time_grid_norm)
            idx_r = indices.clamp(1, self.num_bins)
            idx_l = idx_r - 1
            t_l = bin_edges_norm[idx_l]
            t_r = bin_edges_norm[idx_r]
            w_r = ((time_grid_norm - t_l) / (t_r - t_l + 1e-10)).clamp(0.0, 1.0)
            w_l = 1.0 - w_r
            idx_l_exp = idx_l.unsqueeze(0).expand(features.shape[0], -1)
            idx_r_exp = idx_r.unsqueeze(0).expand(features.shape[0], -1)
            S_l = torch.gather(surv_points, 1, idx_l_exp)
            S_r = torch.gather(surv_points, 1, idx_r_exp)
            S = S_l * w_l.unsqueeze(0) + S_r * w_r.unsqueeze(0)
            mask = time_grid_norm < bin_edges_norm[0]
            S[:, mask] = 1.0
            return S.clamp(0.0, 1.0)

    def predict_time(self, features: torch.Tensor, mode: str = 'median', **kwargs) -> torch.Tensor:
        device = features.device
        with torch.no_grad():
            grid_norm = torch.linspace(-4.0, 4.0, 200, device=device)
            grid_orig = self._to_original_time(grid_norm)
            S_grid = self.predict_survival_function(features, grid_orig)
            idx = torch.searchsorted(-S_grid, torch.full((features.shape[0], 1), -0.5, device=device)).squeeze(1)
            medians = torch.zeros(features.shape[0], device=device)
            at_start = (idx == 0)
            at_end = (idx >= 199)
            mid = ~at_start & ~at_end
            medians[at_start] = grid_orig[0]
            medians[at_end] = grid_orig[-1]
            if mid.any():
                i_r = idx[mid]
                i_l = i_r - 1
                t1, t2 = grid_orig[i_l], grid_orig[i_r]
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
            dt = 1e-2
            combined = torch.cat([time_grid - dt, time_grid, time_grid + dt])
            S = self.predict_survival_function(features, combined)
            n = len(time_grid)
            S_m, S_c, S_p = S[:, :n], S[:, n:2*n], S[:, 2*n:]
            pdf = (S_m - S_p) / (2 * dt)
            pdf = torch.clamp(pdf, min=0.0, max=1000.0)
            h = pdf / torch.clamp(S_c, min=1e-4)
            return torch.clamp(h, min=0.0, max=1000.0)