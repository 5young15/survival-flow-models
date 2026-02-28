import  numpy as np
import torch
import torch.nn as nn
import warnings
from abc import ABC, abstractmethod
from typing import Union, Optional


class SurvivalModelInterface(ABC):
    """
    生存分析模型统一接口 (Abstract Base Class)
    """

    @abstractmethod
    def forward_loss(self, features, times, events, **kwargs):
        pass

    @abstractmethod
    def predict_risk(self, features: torch.Tensor, **kwargs) -> torch.Tensor:
        pass

    def predict_survival_function(self, features: torch.Tensor, time_grid: Optional[torch.Tensor] = None,
                                  **kwargs) -> torch.Tensor:
        if time_grid is not None:
            time_grid = time_grid.to(features.device)

        if not hasattr(self, '_baseline_cumulative_hazard') or self._baseline_cumulative_hazard is None:
            warnings.warn("Baseline hazard not fitted; returning S(t)=1 for all samples.", UserWarning)
            grid = time_grid if time_grid is not None else torch.tensor([0.0], device=features.device)
            return torch.ones((features.shape[0], len(grid)), device=features.device)

        log_haz = self.predict_risk(features)
        exp_haz = torch.exp(torch.clamp(log_haz, max=88.0))

        baseline_times = self._baseline_times.to(features.device)
        baseline_cum_haz = self._baseline_cumulative_hazard.to(features.device)
        grid = time_grid if time_grid is not None else baseline_times

        indices = torch.searchsorted(baseline_times, grid)
        indices = torch.clamp(indices, 0, len(baseline_cum_haz) - 1)

        selected_cum_haz = baseline_cum_haz[indices]
        surv = torch.exp(-exp_haz.unsqueeze(1) * selected_cum_haz.unsqueeze(0))
        return surv

    @abstractmethod
    def predict_time(self, features: torch.Tensor, **kwargs) -> torch.Tensor:
        pass

    def compute_hazard_rate(self, features: torch.Tensor, time_grid: torch.Tensor, **kwargs) -> torch.Tensor:
        S = self.predict_survival_function(features, time_grid, **kwargs)
        log_S = torch.log(torch.clamp(S, min=1e-8, max=1.0))

        if len(time_grid) > 1:
            dt = time_grid[1:] - time_grid[:-1]
            dt = torch.cat([dt, dt[-1:]])

            d_log_S = -(log_S[:, 1:] - log_S[:, :-1])
            d_log_S = torch.cat([d_log_S, d_log_S[:, -1:]], dim=1)

            h = d_log_S / torch.clamp(dt.unsqueeze(0), min=1e-4)
            return torch.clamp(h, min=0.0, max=1000.0)
        else:
            return torch.full((features.shape[0], 1), float('nan'), device=features.device)


class TorchSurvivalModel(nn.Module, SurvivalModelInterface):
    def __init__(self):
        super().__init__()
        self.register_buffer('time_scaler_mean', torch.tensor(0.0))
        self.register_buffer('time_scaler_std', torch.tensor(1.0))
        self.is_log_space = True

    def set_time_scaler(self, mean_val: float, std_val: float, is_log_space: bool = True):
        self.time_scaler_mean.fill_(mean_val)
        self.time_scaler_std.fill_(std_val)
        self.is_log_space = is_log_space

    def _to_original_time(self, t_norm: Union[torch.Tensor, np.ndarray]) -> torch.Tensor:
        if isinstance(t_norm, np.ndarray):
            t_norm = torch.from_numpy(t_norm).to(self.time_scaler_mean.device)
        else:
            t_norm = torch.as_tensor(t_norm, device=self.time_scaler_mean.device).float()
        t_norm = t_norm.reshape(-1)

        if self.is_log_space:
            t_raw = torch.exp(t_norm * self.time_scaler_std + self.time_scaler_mean) - 1.0
        else:
            t_raw = t_norm * self.time_scaler_std + self.time_scaler_mean
        return torch.clamp(t_raw, min=0.0)

    def _to_normalized_time(self, t_raw: Union[torch.Tensor, np.ndarray]) -> torch.Tensor:
        if isinstance(t_raw, np.ndarray):
            t_raw = torch.from_numpy(t_raw).to(self.time_scaler_mean.device)
        else:
            t_raw = torch.as_tensor(t_raw, device=self.time_scaler_mean.device).float()

        if self.is_log_space:
            t_log = torch.log(torch.clamp(t_raw, min=0.0) + 1.0)
            t_norm = (t_log - self.time_scaler_mean) / (self.time_scaler_std + 1e-8)
        else:
            t_norm = (t_raw - self.time_scaler_mean) / (self.time_scaler_std + 1e-8)
        return t_norm

    def _fit_breslow_baseline_hazard(self, times: torch.Tensor, events: torch.Tensor, log_haz: torch.Tensor):
        device = self.time_scaler_mean.device
        times = torch.as_tensor(times, dtype=torch.float32, device=device).reshape(-1)
        events = torch.as_tensor(events, dtype=torch.float32, device=device).reshape(-1)
        log_haz = torch.as_tensor(log_haz, dtype=torch.float32, device=device).reshape(-1)

        times_s, sort_idx = torch.sort(times)
        events_s = events[sort_idx]
        log_haz_s = log_haz[sort_idx]

        exp_haz = torch.exp(torch.clamp(log_haz_s, max=88.0))
        risk_set_sum = torch.flip(torch.cumsum(torch.flip(exp_haz, dims=[0]), dim=0), dims=[0])

        event_mask = (events_s == 1)
        if not torch.any(event_mask):
            return torch.tensor([times.max().item()], device=device), torch.tensor([0.0], device=device)

        unique_times, inverse_indices, counts = torch.unique(
            times_s[event_mask], sorted=True, return_inverse=True, return_counts=True
        )

        event_indices = torch.where(event_mask)[0]
        first_occurrence_idx = torch.zeros(len(unique_times), dtype=torch.long, device=device)
        for i in range(len(unique_times)):
            first_occurrence_idx[i] = event_indices[inverse_indices == i][0]

        summed_contributions = counts.float() / torch.clamp(risk_set_sum[first_occurrence_idx], min=1e-10)
        baseline_cum_haz = torch.cumsum(summed_contributions, dim=0)

        return unique_times, baseline_cum_haz