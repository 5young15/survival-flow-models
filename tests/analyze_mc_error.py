"""
深入分析 MC 方法计算 log h(t) 的误差来源
"""
import torch
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.data_generation import SurvivalDataGenerator
from experiments.config import DataConfig
from models.flowmodel.base_flow import FlowSurv


def analyze_mc_error_sources():
    """
    分析 MC 方法的误差来源
    """
    print("=" * 70)
    print("深入分析 MC 方法计算 log h(t) 的误差来源")
    print("=" * 70)
    
    torch.manual_seed(42)
    np.random.seed(42)
    device = torch.device('cpu')
    
    config = DataConfig(
        n_samples=500,
        distribution_type="weibull_single",
        censoring_rate=0.3,
        random_seed=42
    )
    
    generator = SurvivalDataGenerator(config)
    data = generator.generate()
    
    in_dim = data.features.shape[1]
    
    time_mean = np.log(data.times + 1).mean()
    time_std = np.log(data.times + 1).std()
    
    features = torch.tensor(data.features[:1], dtype=torch.float32, device=device)
    time_grid = torch.tensor(data.time_grid, dtype=torch.float32, device=device)
    t_np = time_grid.cpu().numpy()
    
    true_hazard = data.true_hazard[0]
    true_log_h = np.log(np.maximum(true_hazard, 1e-100))
    true_S = data.true_survival[0]
    true_H = -np.log(np.maximum(true_S, 1e-100))
    
    # 测试不同采样数
    print("\n--- 测试不同采样数的影响 ---")
    for n_samples in [500, 1000, 2000, 5000, 10000]:
        model = FlowSurv(in_dim=in_dim, config={'use_mc': True, 'mc_samples': n_samples, 'ode_steps': 50})
        model.set_time_scaler(time_mean, time_std, is_log_space=True)
        
        with torch.no_grad():
            log_h = model.compute_hazard_rate_mc(features, time_grid, n_samples=n_samples)
        
        pred_log_h = log_h.cpu().numpy()[0]
        mse = np.mean((pred_log_h - true_log_h) ** 2)
        print(f"n_samples={n_samples:5d}: log h(t) MSE = {mse:.6f}")
    
    # 分析核心问题：MC 估计的 S(t) 在尾部不准确
    print("\n--- 分析 S(t) 在尾部的估计误差 ---")
    model = FlowSurv(in_dim=in_dim, config={'use_mc': True, 'mc_samples': 5000, 'ode_steps': 50})
    model.set_time_scaler(time_mean, time_std, is_log_space=True)
    
    with torch.no_grad():
        S_mc = model.predict_survival_function_mc(features, time_grid, n_samples=5000)
    
    S_mc_np = S_mc.cpu().numpy()[0]
    
    # 找出 S(t) 估计误差大的区域
    S_error = np.abs(S_mc_np - true_S)
    high_error_idx = np.where(S_error > 0.05)[0]
    
    print(f"S(t) 误差 > 0.05 的点数: {len(high_error_idx)}")
    if len(high_error_idx) > 0:
        print("高误差点示例:")
        for idx in high_error_idx[:5]:
            print(f"  t={t_np[idx]:.4f}: MC S={S_mc_np[idx]:.4f}, True S={true_S[idx]:.4f}, Error={S_error[idx]:.4f}")
    
    # 核心问题：h(t) = f(t)/S(t)，当 S(t) 接近 0 时，误差被放大
    print("\n--- 分析 h(t) = f(t)/S(t) 的误差放大 ---")
    H_mc = -np.log(np.maximum(S_mc_np, 1e-100))
    
    # 使用真实 H(t) 和 MC H(t) 计算导数
    dH_dt_true = np.gradient(true_H, t_np)
    dH_dt_mc = np.gradient(H_mc, t_np)
    
    h_true = np.clip(dH_dt_true, 1e-10, 1000)
    h_mc = np.clip(dH_dt_mc, 1e-10, 1000)
    
    log_h_true = np.log(h_true)
    log_h_mc_raw = np.log(h_mc)
    
    print(f"真实 h(t) 范围: [{h_true.min():.4f}, {h_true.max():.4f}]")
    print(f"MC h(t) 范围: [{h_mc.min():.4f}, {h_mc.max():.4f}]")
    
    # 分析 h(t) 的相对误差
    h_rel_error = np.abs(h_mc - h_true) / np.maximum(h_true, 1e-10)
    print(f"h(t) 相对误差均值: {h_rel_error.mean():.4f}")
    print(f"h(t) 相对误差最大: {h_rel_error.max():.4f}")
    
    # 问题定位：MC 方法估计的 S(t) 是离散的阶梯函数
    print("\n--- MC S(t) 的阶梯特性分析 ---")
    S_diff = np.diff(S_mc_np)
    unique_diffs = np.unique(np.round(S_diff, 4))
    print(f"S(t) 差分的唯一值数量: {len(unique_diffs)}")
    print(f"差分值示例: {unique_diffs[:10]}")
    
    # 阶梯函数导致导数不稳定
    print("\n--- 结论 ---")
    print("MC 方法的核心问题:")
    print("1. S(t) 是阶梯函数，导数在大部分点为 0")
    print("2. 在阶梯跳跃点，导数可能非常大")
    print("3. 即使使用平滑，也无法完全消除阶梯效应")
    print("4. 需要极大的采样数才能使 S(t) 足够平滑")


if __name__ == "__main__":
    analyze_mc_error_sources()
