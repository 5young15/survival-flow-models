from __future__ import annotations

from typing import Callable, Dict, Tuple

import torch
from torch import Tensor
import torch.nn.functional as F


def safe_log(x: Tensor, eps: float = 1e-8) -> Tensor:
    """t -> logt"""
    return torch.log(torch.clamp(x, min=eps))


def solve_euler(
    y0: Tensor,
    field_fn: Callable[[Tensor, Tensor], Tensor],
    steps: int,
    tau_start: float = 0.0,
    tau_end: float = 1.0,
) -> Tensor:
    """欧拉法 ODE 求解器"""
    if steps <= 0:
        return y0
    y = y0
    dt = (tau_end - tau_start) / float(steps)
    tau = torch.full_like(y0, tau_start)
    for _ in range(steps):
        # 增加速度截断，防止数值爆炸
        v = field_fn(y, tau)
        v = torch.clamp(v, -10.0, 10.0)
        y = y + dt * v
        tau = tau + dt
    return y


def solve_rk4(
    y0: Tensor,
    field_fn: Callable[[Tensor, Tensor], Tensor],
    steps: int,
    tau_start: float = 0.0,
    tau_end: float = 1.0,
) -> Tensor:
    """四阶龙格-库塔 (RK4) ODE 求解器"""
    if steps <= 0:
        return y0
    y = y0
    dt = (tau_end - tau_start) / float(steps)
    tau = torch.full_like(y0, tau_start)
    half_dt = dt * 0.5
    for _ in range(steps):
        # 对 ODE 速度进行截断，防止数值爆炸
        k1 = torch.clamp(field_fn(y, tau), -10.0, 10.0)
        k2 = torch.clamp(field_fn(y + half_dt * k1, tau + half_dt), -10.0, 10.0)
        k3 = torch.clamp(field_fn(y + half_dt * k2, tau + half_dt), -10.0, 10.0)
        k4 = torch.clamp(field_fn(y + dt * k3, tau + dt), -10.0, 10.0)
        y = y + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        tau = tau + dt
    return y


def integrate_ode(
    y0: Tensor,
    field_fn: Callable[[Tensor, Tensor], Tensor],
    steps: int,
    method: str = "rk4",
    tau_start: float = 0.0,
    tau_end: float = 1.0,
) -> Tensor:
    """通用的 ODE 积分接口"""
    if method.lower() == "euler":
        return solve_euler(y0, field_fn, steps, tau_start, tau_end)
    elif method.lower() == "rk4":
        return solve_rk4(y0, field_fn, steps, tau_start, tau_end)
    else:
        raise ValueError(f"Unknown ODE method: {method}")


def sample_truncated_times(
    base_samples: Tensor,
    lower_bound: Tensor,
    max_trials: int = 8,
) -> Tensor:
    """
    对基础样本进行截断采样，确保样本大于等于下限。
    """
    lb = lower_bound.view(-1, 1)
    samples = base_samples
    for _ in range(max_trials):
        mask = samples <= lb
        if not mask.any():
            break
        repl = torch.where(mask, lb + torch.rand_like(samples) * torch.clamp(lb, min=0.1), samples)
        samples = repl
    return samples


def flow_matching_targets(y0: Tensor, y1: Tensor, tau: Tensor) -> Tuple[Tensor, Tensor]:
    """
    计算 Flow Matching 目标样本 y_tau 和速度 v。
    """

    y_tau = (1.0 - tau) * y0 + tau * y1
    v = y1 - y0
    return y_tau, v


def ranking_regularizer(
    risk_pred: Tensor,
    time_obs: Tensor,
    event: Tensor,
    margin: float = 0.05,
    eps: float = 1e-8,
) -> Tensor:
    """
    Pairwise ranking regularizer（只在事件样本之间比较）
    若 ti < tj，则要求 risk_i > risk_j + margin
    """
    # 展平，确保形状一致
    risk = risk_pred.view(-1)
    time = time_obs.view(-1)
    e = event.view(-1).float()

    # 只取事件样本
    event_mask = e > 0.5
    if event_mask.sum() < 2:
        return torch.tensor(0.0, device=risk.device, dtype=risk.dtype)

    risk_event = risk[event_mask]
    time_event = time[event_mask]

    # 生成所有事件样本对
    diff_time = time_event.unsqueeze(1) - time_event.unsqueeze(0)  # (N, N)
    diff_risk = risk_event.unsqueeze(1) - risk_event.unsqueeze(0)  # (N, N)

    # 有效对：ti < tj
    valid_pairs = (diff_time < -eps).float()
    num_valid = valid_pairs.sum() + eps

    if num_valid < 1:
        return torch.tensor(0.0, device=risk.device, dtype=risk.dtype)

    # 违反排序的 hinge loss
    violation = torch.relu(margin - diff_risk) * valid_pairs

    # 平均损失（或 sum / num_valid 都可，平均更稳定）
    return violation.sum() / num_valid


def build_time_grid(
    observed_t: Tensor,
    n_grid: int,
) -> Tensor:
    """
    根据观测时间构建时间网格（真实范围）。
    t_min = 0.5 * min(t_obs)
    t_max = max(p95, 2 * median)
    """
    t_min_raw = torch.min(observed_t)
    p95 = torch.quantile(observed_t, 0.95)
    med = torch.quantile(observed_t, 0.5)

    t_min = 0.5 * t_min_raw
    t_max = torch.maximum(p95, 2.0 * med)

    t_min = torch.clamp(t_min, min=1e-4)
    t_max = torch.clamp(t_max, min=t_min + 1e-4)

    # 保持网格相对于边界的可微性
    # 优化：在 log(t) 空间生成均匀网格，再映射回 t 空间
    # 这样在 t 较小时（风险剧烈波动期）网格更密，在 t 较大时（平滑期）网格更稀疏
    log_t_min = torch.log(t_min)
    log_t_max = torch.log(t_max)
    
    grid_log_norm = torch.linspace(0, 1, n_grid, device=observed_t.device, dtype=observed_t.dtype)
    grid_log = log_t_min + (log_t_max - log_t_min) * grid_log_norm
    grid = torch.exp(grid_log)
    
    return grid


def log_density_via_cnf_reverse(
    y_target: Tensor,
    velocity_fn: Callable[[Tensor, Tensor], Tensor],
    steps: int,
    method: str = "rk4",
    create_graph: bool = False,
) -> Tuple[Tensor, Tensor]:
    """
    通过 CNF 反向积分计算 log p(y_target)。
    d/dtau [y, l] = [v, -div(v)]
    从 tau=1 反向积分到 tau=0。
    y(1) = y_target, l(1) = 0
    y(0) = y_prior, l(0) = integral(div v)
    log p(y_target) = log p_prior(y(0)) - l(0)

    Args:
        y_target: 目标样本 [batch, 1]
        velocity_fn: 速度场函数 v(y, tau)
        steps: ODE 积分步数
        method: ODE 求解方法 ('euler' 或 'rk4')
        create_graph: 是否为散度计算创建计算图。训练阶段(如 Density Path 训练)设为 True，
                      推理阶段(如 evaluate_model)设为 False 以节省内存。

    Returns:
        y0: 积分终点 y(0)
        delta_logp: 积分得到的 log det jacobian 变化量 (即 l(0))
    """

    def augmented_field(state: Tensor, tau: Tensor) -> Tensor:
        # state: [batch, 2] -> col 0 is y, col 1 is delta_logp
        y = state[:, 0:1]
        
        # tau 与状态 state 的形状 [batch, 2] 匹配，但 velocity_fn 需要 [batch, 1]
        if tau.dim() > 1 and tau.shape[1] > 1:
            tau_in = tau[:, 0:1]
        else:
            tau_in = tau

        # 核心优化：一维潜空间的精确散度计算 (Exact Divergence)
        # 由于 y 是 1D, div v = dv/dy。直接使用 autograd 计算，避免 Hutchinson 估计带来的方差。
        with torch.set_grad_enabled(True):
            y_in = y.detach().requires_grad_(True)
            v = velocity_fn(y_in, tau_in)
            # 因为 y_in 和 v 都是 [B, 1], 显式计算 dv/dy
            grad_v = torch.autograd.grad(
                v, y_in, grad_outputs=torch.ones_like(v), create_graph=create_graph
            )[0]
        
        return torch.cat([v, -grad_v], dim=1)

    batch_size = y_target.shape[0]
    initial_state = torch.cat([y_target, torch.zeros_like(y_target)], dim=1)  # [batch, 2]

    # 反向积分：从 tau=1 到 tau=0
    # 注意 integrate_ode 默认是正向，我们需要反向
    # 可以通过 steps=-steps 或者 tau_start=1.0, tau_end=0.0 来实现
    # 这里我们修改 integrate_ode 调用方式
    
    final_state = integrate_ode(
        y0=initial_state,
        field_fn=augmented_field,
        steps=steps,
        method=method,
        tau_start=1.0,
        tau_end=0.0
    )

    y0 = final_state[:, 0:1]
    delta_logp = final_state[:, 1:2]
    
    return y0, delta_logp


def hazard_from_survival_curve(surv: Tensor, grid_t: Tensor, eps: float = 1e-8) -> Tensor:
    """
    从生存函数 S(t) 计算风险函数 h(t)。
    h(t) = - d/dt log S(t)
    使用数值微分。
    """
    # log S(t)
    log_surv = safe_log(surv, eps=eps)
    
    # 对中间部分使用中心差分，边界使用前向/后向差分
    # grid_t 形状: [batch, n_grid] 或 [n_grid]
    if grid_t.ndim == 1:
        dt = grid_t[1:] - grid_t[:-1] # [n-1]
        dt = dt.view(1, -1)
    else:
        dt = grid_t[:, 1:] - grid_t[:, :-1]
        
    d_log_s = log_surv[:, 1:] - log_surv[:, :-1]
    
    # hazard = - d_log_s / dt
    # 这计算了中点处的风险。我们可以插值回到网格点。
    haz_mid = - d_log_s / torch.clamp(dt, min=eps)
    
    # 填充以匹配形状
    haz = torch.zeros_like(surv)
    haz[:, 1:] = haz_mid
    haz[:, 0] = haz[:, 1] # 第一个点使用最近邻填充
    
    return torch.clamp(haz, min=0.0)


def build_prediction_bundle(
    surv: Tensor,
    haz: Tensor,
    dens: Tensor,
    cdf: Tensor,
) -> Dict[str, Tensor]:
    """
    构建预测结果包。
    """
    return {
        "survival": surv,
        "hazard": haz,
        "density": dens,
        "cdf": cdf
    }




def median_from_cdf(grid_t: Tensor, cdf: Tensor) -> Tensor:
    """
    从 CDF 计算中位生存时间。
    F(t) >= 0.5 的第一个点。
    当网格内未达到 0.5 时，使用常数尾部风险（指数外推）进行预测。
    """
    batch_size, n_grid = cdf.shape
    medians = torch.zeros(batch_size, device=cdf.device, dtype=cdf.dtype)
    
    # 如果需要，展开网格
    if grid_t.ndim == 1:
        grid_t = grid_t.unsqueeze(0).repeat(batch_size, 1)

    eps = torch.tensor(1e-8, device=cdf.device, dtype=cdf.dtype)
    half = torch.tensor(0.5, device=cdf.device, dtype=cdf.dtype)

    for i in range(batch_size):
        # 找到 F(t) >= 0.5 的索引
        idx = (cdf[i] >= 0.5).nonzero(as_tuple=True)[0]
        
        if len(idx) > 0:
            first_idx = idx[0].item()
            if first_idx == 0:
                medians[i] = grid_t[i, 0]
            else:
                # 线性插值
                t1, t2 = grid_t[i, first_idx-1], grid_t[i, first_idx]
                f1, f2 = cdf[i, first_idx-1], cdf[i, first_idx]
                # (0.5 - f1) / (f2 - f1) = (t - t1) / (t2 - t1)
                # t = t1 + (t2 - t1) * (0.5 - f1) / (f2 - f1)
                denom = torch.clamp(f2 - f1, min=eps)
                slope = (t2 - t1) / denom
                medians[i] = t1 + slope * (half - f1)
        else:
            f_last = cdf[i, -1]
            t_last = grid_t[i, -1]

            surv_last = torch.clamp(1.0 - f_last, min=eps)
            target_val = torch.log(surv_last / half)
            target_val = torch.clamp(target_val, min=0.0)

            if n_grid >= 2:
                t_prev = grid_t[i, -2]
                f_prev = cdf[i, -2]
                dt_last = torch.clamp(t_last - t_prev, min=eps)
                dens_last = torch.clamp((f_last - f_prev) / dt_last, min=0.0)
                h_ref = dens_last / torch.clamp(surv_last, min=eps)
            else:
                h_ref = eps

            h_ref = torch.clamp(h_ref, min=eps)
            t_med_est = t_last + target_val / h_ref
            medians[i] = torch.clamp(t_med_est, min=t_last, max=t_last * 50.0)

    return medians.view(-1, 1)


