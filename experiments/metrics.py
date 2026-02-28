import torch
import numpy as np
from typing import Dict, Tuple, Optional, List, Union
from dataclasses import dataclass


@dataclass
class MetricsResult:
    """评估指标结果容器"""
    c_index: float
    ibs: float
    brier_scores: Dict[str, float]
    median_mae: float
    median_rmse: float
    hazard_mse: Optional[float] = None
    hazard_mae: Optional[float] = None
    hazard_iae: Optional[float] = None
    density_mse: Optional[float] = None
    density_mae: Optional[float] = None
    wasserstein_1: Optional[float] = None


def concordance_index_pytorch(risk_scores: torch.Tensor,
                              event_times: torch.Tensor,
                              event_indicators: torch.Tensor) -> float:
    """
    快速计算 C-index (向量化实现, 基于 PyTorch GPU)
    
    参数:
        risk_scores: (N,) 风险分数 (越高代表风险越大)
        event_times: (N,) 发生时间张量
        event_indicators: (N,) 事件指示器 (1=事件, 0=删失)
    """
    n = len(event_times)
    if n == 0:
        return 0.5
    
    device = event_times.device
    
    # 按时间排序 (升序)
    order = torch.argsort(event_times)
    event_times = event_times[order]
    event_indicators = event_indicators[order]
    risk_scores = risk_scores[order]
    
    # 使用 tensor 累加，避免 GPU-CPU 同步
    concordant = torch.tensor(0.0, device=device)
    permissible = torch.tensor(0.0, device=device)
    
    # 向量化计算改进：
    # 只有当 event_times[i] < event_times[j] 且 event_indicators[i]==1 时, (i, j) 才是可比较对
    for i in range(n):
        if event_indicators[i] == 0:
            continue
            
        # 找到所有 event_times[j] > event_times[i] 的样本
        mask_j = event_times > event_times[i]
        if not torch.any(mask_j):
            continue
            
        valid_risks = risk_scores[mask_j]
        current_risk = risk_scores[i]
        
        n_valid = len(valid_risks)
        permissible += n_valid
        
        # 一致对 (concordant): 发生时间早的风险更高
        concordant += torch.sum(current_risk > valid_risks).float()
        # 风险相等对 (tied risk): 计入 0.5
        concordant += 0.5 * torch.sum(current_risk == valid_risks).float()
    
    if permissible == 0:
        return 0.5
    
    return (concordant / permissible).item()


def kaplan_meier_estimator(times: torch.Tensor, 
                           events: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Kaplan-Meier 生存函数估计 (PyTorch 实现)
    """
    device = times.device
    event_times = times[events == 1]
    if len(event_times) == 0:
        return torch.tensor([0.0], device=device), torch.tensor([1.0], device=device)
        
    unique_times = torch.unique(event_times)
    unique_times, _ = torch.sort(unique_times)
    
    n = len(times)
    S = torch.ones(len(unique_times) + 1, device=device)
    S_times = torch.cat([torch.tensor([0.0], device=device), unique_times])
    
    for i, t in enumerate(unique_times):
        n_at_risk = torch.sum(times >= t).float()
        n_events = torch.sum((times == t) & (events == 1)).float()
        
        if n_at_risk > 0:
            S[i + 1] = S[i] * (1 - n_events / n_at_risk)
        else:
            S[i + 1] = S[i]
    
    return S_times, S


def ipcw_weights(times: torch.Tensor,
                 events: torch.Tensor,
                 eval_times: torch.Tensor,
                 max_weight: float = 20.0) -> torch.Tensor:
    """
    计算逆概率删失加权 (IPCW) 权重 (PyTorch 实现)
    """
    device = times.device
    censoring_indicators = 1 - events
    G_times, G_km = kaplan_meier_estimator(times, censoring_indicators)
    
    weights = torch.zeros(len(eval_times), device=device)
    for i, t in enumerate(eval_times):
        idx = torch.searchsorted(G_times, t.unsqueeze(0) if t.dim()==0 else t, right=True) - 1
        idx = torch.clamp(idx, 0, len(G_km) - 1)
        G_t = G_km[idx]
        weights[i] = 1.0 / torch.clamp(G_t, min=1.0 / max_weight)
    
    return torch.clamp(weights, 1.0, max_weight)


def brier_score_at_time(times: torch.Tensor,
                        events: torch.Tensor,
                        pred_survival: torch.Tensor,
                        eval_time: float,
                        max_weight: float = 20.0,
                        G_times: Optional[torch.Tensor] = None,
                        G_km: Optional[torch.Tensor] = None,
                        G_eval_time: Optional[float] = None,
                        G_at_times: Optional[torch.Tensor] = None) -> float:
    """
    计算指定时间点的 Brier Score (PyTorch 实现)
    """
    n = len(times)
    if n == 0:
        return float('nan')
    
    device = times.device
    eval_time_tensor = torch.tensor(eval_time, device=device)
    
    if G_times is None or G_km is None:
        censoring_indicators = 1 - events
        G_times, G_km = kaplan_meier_estimator(times, censoring_indicators)
    
    if G_eval_time is None:
        idx_t = torch.searchsorted(G_times, eval_time_tensor.unsqueeze(0), right=True) - 1
        idx_t = torch.clamp(idx_t, 0, len(G_km) - 1)
        G_eval_time = torch.clamp(G_km[idx_t], min=1.0 / max_weight).item()
    
    if G_at_times is None:
        idx_all = torch.searchsorted(G_times, times, right=True) - 1
        idx_all = torch.clamp(idx_all, 0, len(G_km) - 1)
        G_at_times = G_km[idx_all]
    
    event_mask = (times <= eval_time_tensor) & (events == 1)
    survive_mask = times > eval_time_tensor
    
    if not (torch.any(event_mask) or torch.any(survive_mask)):
        return float('nan')
    
    event_weights = 1.0 / torch.clamp(G_at_times[event_mask], min=1.0 / max_weight)
    event_weights = torch.clamp(event_weights, max=max_weight)
    event_errors = pred_survival[event_mask] ** 2
    
    survive_weight = min(1.0 / G_eval_time, max_weight)
    survive_errors = (1 - pred_survival[survive_mask]) ** 2
    
    bs_sum = torch.sum(event_weights * event_errors) + survive_weight * torch.sum(survive_errors)
    
    return (bs_sum / n).item()

def integrated_brier_score(times: torch.Tensor,
                           events: torch.Tensor,
                           pred_survival: torch.Tensor,
                           time_grid: torch.Tensor,
                           max_weight: float = 20.0) -> float:
    """
    计算积分 Brier Score (IBS) (PyTorch 实现)
    """
    if len(time_grid) < 2:
        return float('nan')
    
    device = times.device
    censoring_indicators = 1 - events
    G_times, G_km = kaplan_meier_estimator(times, censoring_indicators)
    idx_all = torch.searchsorted(G_times, times, right=True) - 1
    idx_all = torch.clamp(idx_all, 0, len(G_km) - 1)
    G_at_times = G_km[idx_all]
    
    n_times = min(pred_survival.shape[1], len(time_grid))
    valid_grid = time_grid[:n_times]
    
    bs_values = torch.zeros(n_times, device=device)
    valid_mask = torch.ones(n_times, dtype=torch.bool, device=device)
    
    for i in range(n_times):
        bs = brier_score_at_time(
            times, events, pred_survival[:, i], valid_grid[i].item(), max_weight,
            G_times=G_times, G_km=G_km, G_at_times=G_at_times
        )
        if np.isnan(bs):
            valid_mask[i] = False
        else:
            bs_values[i] = bs
    
    if valid_mask.sum() < 2:
        return float('nan')
    
    bs_valid = bs_values[valid_mask]
    t_valid = valid_grid[valid_mask]
    
    ibs = torch.trapezoid(bs_valid, t_valid) / (t_valid[-1] - t_valid[0])
    
    return ibs.item()


def time_dependent_brier_score(times: torch.Tensor,
                                events: torch.Tensor,
                                pred_survival: torch.Tensor,
                                time_grid: torch.Tensor,
                                quantiles: Tuple[float, ...] = (0.25, 0.5, 0.75)) -> Dict[str, float]:
    """
    计算指定分位点的时间依赖 Brier Score (PyTorch 实现)
    """
    device = times.device
    t_max = torch.max(times)
    
    event_times = times[events == 1]
    if len(event_times) > 0:
        quantile_times = torch.quantile(event_times, torch.tensor(quantiles, device=device))
    else:
        quantile_times = t_max * torch.tensor(quantiles, device=device)
    
    censoring_indicators = 1 - events
    G_times, G_km = kaplan_meier_estimator(times, censoring_indicators)
    idx_all = torch.searchsorted(G_times, times, right=True) - 1
    idx_all = torch.clamp(idx_all, 0, len(G_km) - 1)
    G_at_times = G_km[idx_all]
    
    results = {}
    for q, t in zip(quantiles, quantile_times):
        idx = torch.argmin(torch.abs(time_grid - t))
        if idx < pred_survival.shape[1]:
            bs = brier_score_at_time(
                times, events, pred_survival[:, idx], t.item(),
                G_times=G_times, G_km=G_km, G_at_times=G_at_times
            )
            results[f"BS_{int(q*100)}%"] = bs
    
    return results


def median_time_error(true_medians: torch.Tensor,
                      pred_medians: torch.Tensor) -> Tuple[float, float]:
    """
    计算中位生存时间预测误差 (PyTorch 实现)
    """
    valid_mask = ~(torch.isnan(true_medians) | torch.isnan(pred_medians))
    if not torch.any(valid_mask):
        return float('nan'), float('nan')
    
    true_valid = true_medians[valid_mask]
    pred_valid = pred_medians[valid_mask]
    
    mae = torch.mean(torch.abs(true_valid - pred_valid))
    rmse = torch.sqrt(torch.mean((true_valid - pred_valid)**2))
    
    return mae.item(), rmse.item()


def hazard_mse(true_hazard: torch.Tensor,
               pred_hazard: torch.Tensor) -> float:
    """
    计算对数风险函数 MSE (Log-Hazard MSE)
    """
    if true_hazard is None or pred_hazard is None:
        return float('nan')
    
    if true_hazard.shape != pred_hazard.shape:
        min_cols = min(true_hazard.shape[1], pred_hazard.shape[1])
        true_hazard = true_hazard[:, :min_cols]
        pred_hazard = pred_hazard[:, :min_cols]
    
    # 增加 epsilon 防止 log(0), 并在对数空间计算
    eps = 1e-9
    true_log = torch.log(torch.clamp(true_hazard, min=0.0) + eps)
    pred_log = torch.log(torch.clamp(pred_hazard, min=0.0) + eps)
    
    return torch.mean((true_log - pred_log)**2).item()


def hazard_mae(true_hazard: torch.Tensor,
               pred_hazard: torch.Tensor) -> float:
    """计算对数风险函数 MAE (Log-Hazard MAE)"""
    if true_hazard is None or pred_hazard is None:
        return float('nan')
    
    if true_hazard.shape != pred_hazard.shape:
        min_cols = min(true_hazard.shape[1], pred_hazard.shape[1])
        true_hazard = true_hazard[:, :min_cols]
        pred_hazard = pred_hazard[:, :min_cols]
    
    eps = 1e-9
    true_log = torch.log(torch.clamp(true_hazard, min=0.0) + eps)
    pred_log = torch.log(torch.clamp(pred_hazard, min=0.0) + eps)
    
    return torch.mean(torch.abs(true_log - pred_log)).item()


def hazard_integrated_absolute_error(true_hazard: torch.Tensor,
                                     pred_hazard: torch.Tensor,
                                     time_grid: torch.Tensor) -> float:
    """
    计算对数风险函数积分绝对误差 (Log-IAE)
    """
    if true_hazard is None or pred_hazard is None:
        return float('nan')
    
    if true_hazard.shape != pred_hazard.shape:
        min_cols = min(true_hazard.shape[1], pred_hazard.shape[1])
        true_hazard = true_hazard[:, :min_cols]
        pred_hazard = pred_hazard[:, :min_cols]
        time_grid = time_grid[:min_cols]
    
    eps = 1e-9
    true_log = torch.log(torch.clamp(true_hazard, min=0.0) + eps)
    pred_log = torch.log(torch.clamp(pred_hazard, min=0.0) + eps)
    
    abs_diff = torch.abs(true_log - pred_log)
    
    iae_per_sample = torch.trapezoid(abs_diff, time_grid, dim=1)
    
    return torch.mean(iae_per_sample).item()


def density_mse(true_density: torch.Tensor,
                pred_density: torch.Tensor) -> float:
    """计算密度函数 MSE (带数值稳定性保护)"""
    if true_density is None or pred_density is None:
        return float('nan')
    
    if true_density.shape != pred_density.shape:
        min_cols = min(true_density.shape[1], pred_density.shape[1])
        true_density = true_density[:, :min_cols]
        pred_density = pred_density[:, :min_cols]
    
    true_clamped = torch.clamp(true_density, min=0.0, max=1000.0)
    pred_clamped = torch.clamp(pred_density, min=0.0, max=1000.0)
    
    return torch.mean((true_clamped - pred_clamped)**2).item()


def density_mae(true_density: torch.Tensor,
                pred_density: torch.Tensor) -> float:
    """计算密度函数 MAE (带数值稳定性保护)"""
    if true_density is None or pred_density is None:
        return float('nan')
    
    if true_density.shape != pred_density.shape:
        min_cols = min(true_density.shape[1], pred_density.shape[1])
        true_density = true_density[:, :min_cols]
        pred_density = pred_density[:, :min_cols]
    
    true_clamped = torch.clamp(true_density, min=0.0, max=1000.0)
    pred_clamped = torch.clamp(pred_density, min=0.0, max=1000.0)
    
    return torch.mean(torch.abs(true_clamped - pred_clamped)).item()


def wasserstein_1_distance(true_survival: torch.Tensor,
                           pred_survival: torch.Tensor,
                           time_grid: torch.Tensor) -> float:
    """
    计算 Wasserstein-1 距离
    """
    if true_survival is None or pred_survival is None:
        return float('nan')
    
    if true_survival.shape != pred_survival.shape:
        min_cols = min(true_survival.shape[1], pred_survival.shape[1])
        true_survival = true_survival[:, :min_cols]
        pred_survival = pred_survival[:, :min_cols]
        time_grid = time_grid[:min_cols]
    
    true_cdf = 1 - true_survival
    pred_cdf = 1 - pred_survival
    
    abs_diff = torch.abs(true_cdf - pred_cdf)
    
    w1_per_sample = torch.trapezoid(abs_diff, time_grid, dim=1)
    
    return torch.mean(w1_per_sample).item()


def compute_all_metrics(times: Union[torch.Tensor, np.ndarray, None],
                        events: Union[torch.Tensor, np.ndarray, None],
                        risk_scores: Union[torch.Tensor, np.ndarray, None],
                        pred_survival: Union[torch.Tensor, np.ndarray, None],
                        pred_medians: Union[torch.Tensor, np.ndarray, None],
                        time_grid: Union[torch.Tensor, np.ndarray, None],
                        true_hazard: Optional[torch.Tensor] = None,
                        true_density: Optional[torch.Tensor] = None,
                        true_survival: Optional[torch.Tensor] = None,
                        true_medians: Optional[torch.Tensor] = None,
                        pred_hazard: Optional[torch.Tensor] = None,
                        pred_density: Optional[torch.Tensor] = None,
                        quantiles: Tuple[float, ...] = (0.25, 0.5, 0.75),
                        max_weight: float = 20.0) -> MetricsResult:
    """
    计算所有评估指标 (统一使用 Tensor 输入)
    """
    def to_tensor(x, device):
        if x is None: return None
        if isinstance(x, np.ndarray):
            return torch.from_numpy(x).float().to(device)
        return x.float().to(device)

    device = times.device if isinstance(times, torch.Tensor) else torch.device('cpu')
    times = to_tensor(times, device)
    events = to_tensor(events, device)
    risk_scores = to_tensor(risk_scores, device)
    pred_survival = to_tensor(pred_survival, device)
    pred_medians = to_tensor(pred_medians, device)
    time_grid = to_tensor(time_grid, device)
    true_hazard = to_tensor(true_hazard, device)
    true_density = to_tensor(true_density, device)
    true_survival = to_tensor(true_survival, device)
    true_medians = to_tensor(true_medians, device)
    pred_hazard = to_tensor(pred_hazard, device)
    pred_density = to_tensor(pred_density, device)

    c_index = concordance_index_pytorch(risk_scores, times, events)
    
    ibs = integrated_brier_score(times, events, pred_survival, time_grid, max_weight)
    
    brier_scores = time_dependent_brier_score(
        times, events, pred_survival, time_grid, quantiles
    )
    
    if true_medians is not None:
        median_mae, median_rmse = median_time_error(true_medians, pred_medians)
    else:
        median_mae, median_rmse = float('nan'), float('nan')
    
    hazard_mse_val = None
    hazard_mae_val = None
    hazard_iae_val = None
    density_mse_val = None
    density_mae_val = None
    w1_val = None
    
    if true_hazard is not None and pred_hazard is not None:
        hazard_mse_val = hazard_mse(true_hazard, pred_hazard)
        hazard_mae_val = hazard_mae(true_hazard, pred_hazard)
        hazard_iae_val = hazard_integrated_absolute_error(true_hazard, pred_hazard, time_grid)
    
    if true_density is not None and pred_density is not None:
        density_mse_val = density_mse(true_density, pred_density)
        density_mae_val = density_mae(true_density, pred_density)
    
    if true_survival is not None and pred_survival is not None:
        w1_val = wasserstein_1_distance(true_survival, pred_survival, time_grid)
    
    return MetricsResult(
        c_index=c_index,
        ibs=ibs,
        brier_scores=brier_scores,
        median_mae=median_mae,
        median_rmse=median_rmse,
        hazard_mse=hazard_mse_val,
        hazard_mae=hazard_mae_val,
        hazard_iae=hazard_iae_val,
        density_mse=density_mse_val,
        density_mae=density_mae_val,
        wasserstein_1=w1_val
    )


def metrics_to_dict(result: MetricsResult) -> Dict[str, float]:
    """将 MetricsResult 转换为字典"""
    d = {
        'c_index': result.c_index,
        'ibs': result.ibs,
        'median_mae': result.median_mae,
        'median_rmse': result.median_rmse,
    }
    d.update(result.brier_scores)
    
    if result.hazard_mse is not None:
        d['hazard_mse'] = result.hazard_mse
    if result.hazard_mae is not None:
        d['hazard_mae'] = result.hazard_mae
    if result.hazard_iae is not None:
        d['hazard_iae'] = result.hazard_iae
    if result.density_mse is not None:
        d['density_mse'] = result.density_mse
    if result.density_mae is not None:
        d['density_mae'] = result.density_mae
    if result.wasserstein_1 is not None:
        d['wasserstein_1'] = result.wasserstein_1
    
    return d


if __name__ == "__main__":
    print("测试评估指标模块")
    print("=" * 60)
    
    torch.manual_seed(42)
    n = 100
    
    times = torch.exp(torch.randn(n) * 0.5 + 1.5)
    events = (torch.rand(n) > 0.3).float()
    risk_scores = torch.randn(n)
    
    c_index = concordance_index_fast(times, events, risk_scores)
    print(f"C-index: {c_index:.4f}")
    
    time_grid = torch.linspace(0.1, 15, 50)
    pred_survival = torch.rand(n, len(time_grid)) * 0.7 + 0.3
    pred_survival = torch.sort(pred_survival, dim=1)[0].flip(dims=[1])
    
    ibs = integrated_brier_score(times, events, pred_survival, time_grid)
    print(f"IBS: {ibs:.4f}")
    
    true_medians = torch.rand(n) * 5 + 3
    pred_medians = true_medians + torch.randn(n) * 0.5
    mae, rmse = median_time_error(true_medians, pred_medians)
    print(f"Median MAE: {mae:.4f}, RMSE: {rmse:.4f}")
