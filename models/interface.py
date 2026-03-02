import numpy as np
import torch
import torch.nn as nn
import warnings
from abc import ABC, abstractmethod
from typing import Union, Optional


class SurvivalModelInterface(ABC):
    """
    生存分析模型统一接口 (Abstract Base Class)
    """

    # 计算损失函数 loss
    @abstractmethod
    def forward_loss(self, features, times, events, **kwargs):
        pass

    # 计算中位生存时间 t_median
    @abstractmethod
    def predict_time(self, features: torch.Tensor, **kwargs) -> torch.Tensor:
        pass

    # 预测风险分数 score(t) -> C-index
    @abstractmethod
    def predict_risk(self, features: torch.Tensor, **kwargs) -> torch.Tensor:
        pass

    # cox模型预测生存函数 S(t|X) -> IBS, Brter Score
    def predict_survival_function(self, features: torch.Tensor, time_grid: Optional[torch.Tensor] = None,
                                  **kwargs) -> torch.Tensor:
        if time_grid is not None:
            time_grid = time_grid.to(features.device)

        # 处理基线风险用于只能计算相对风险的模型
        if not hasattr(self, '_baseline_cum_haz') or self._baseline_cum_haz is None:
            warnings.warn("Baseline hazard not fitted; returning S(t)=1 for all samples.", UserWarning, stacklevel=2)
            grid = time_grid if time_grid is not None else torch.tensor([0.0], device=features.device)
            return torch.ones((features.shape[0], len(grid)), device=features.device)

        # cox 比例风险模型 计算风险比 HR = exp(h(t|X))
        log_haz = self.predict_risk(features)
        exp_haz = torch.exp(torch.clamp(log_haz, max=88.0))

        baseline_times = self._baseline_times.to(features.device)
        baseline_cum_haz = self._baseline_cum_haz.to(features.device)
        grid = time_grid if time_grid is not None else baseline_times

        indices = torch.searchsorted(baseline_times, grid)
        indices = torch.clamp(indices, 0, len(baseline_cum_haz) - 1)

        selected_cum_haz = baseline_cum_haz[indices]
        surv = torch.exp(-exp_haz.unsqueeze(1) * selected_cum_haz.unsqueeze(0))
        return surv

    # RSF 计算风险函数 log h(t|X) -> hazard_mse
    def compute_hazard_rate(self, features: torch.Tensor, time_grid: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        计算对数风险函数 log h(t|X)
        
        默认实现：从生存函数数值微分计算
        子类可覆盖使用更精确的解析公式
        """
        S = self.predict_survival_function(features, time_grid, **kwargs)
        log_S = torch.log(torch.clamp(S, min=1e-8, max=1.0))

        if len(time_grid) > 1:
            dt = time_grid[1:] - time_grid[:-1]
            dt = torch.cat([dt, dt[-1:]])

            d_log_S = -(log_S[:, 1:] - log_S[:, :-1])
            d_log_S = torch.cat([d_log_S, d_log_S[:, -1:]], dim=1)

            h = d_log_S / torch.clamp(dt.unsqueeze(0), min=1e-8)
            h = torch.clamp(h, min=0.0, max=1000.0)
            return torch.log(h + 1e-8)
        else:
            return torch.full((features.shape[0], 1), float('nan'), device=features.device)


class TorchSurvivalModel(nn.Module, SurvivalModelInterface):
    """
    PyTorch 生存模型基类
    
    提供:
    - 时间标准化/反标准化
    - Breslow 基线风险估计
    - Cox 模型通用的 predict_time 和 compute_hazard_rate
    """
    
    def __init__(self):
        super().__init__()
        self.register_buffer('time_scaler_mean', torch.tensor(0.0))
        self.register_buffer('time_scaler_std', torch.tensor(1.0))
        self.is_log_space = True
    
    # 初始化时间scaler
    def set_time_scaler(self, mean_val: float, std_val: float, is_log_space: bool = True):
        self.time_scaler_mean.fill_(mean_val)
        self.time_scaler_std.fill_(std_val)
        self.is_log_space = is_log_space

    # scale(log(t+1)) -> t
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
    
    # t -> scale(log(t+1))
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

# ===========================================以下为部分共有函数=============================================================
    
    # Cox模型 估计基线累积风险函数
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

    # cox模型 计算中位生存时间
    def _cox_predict_time(self, features: torch.Tensor) -> torch.Tensor:
        """
        Cox 模型通用的中位生存时间预测
        适用于 LinearCoxPH, DeepSurv 等 Cox 类模型
        """
        device = features.device
        with torch.no_grad():
            if self._baseline_times is None:
                if hasattr(self, '_train_times') and self._train_times is not None:
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
            at_start = (idx == 0)
            at_end = (idx >= 199)
            mid = ~at_start & ~at_end
            medians[at_start] = grid[0]
            medians[at_end] = grid[-1]
            
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
    
    # cox模型 计算对数风险函数 h(t|X) = exp(h(t|X)) - > hazard_mse
    def _cox_compute_hazard_rate(self, features: torch.Tensor, time_grid: torch.Tensor) -> torch.Tensor:
        """
        Cox 模型通用的对数风险函数计算
        使用基线风险函数的数值微分
        """
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
            hazard = torch.clamp(hazard, min=0.0, max=1000.0)
            return torch.log(hazard + 1e-8)
