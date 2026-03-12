from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from typing import Dict

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@dataclass
class NonPHSimulatedData:
    """
    专门为 Flow Matching 模型设计的非比例风险 (Non-Proportional Hazards) 数据生成器。
    模拟智能制造设备核心部件在不同工作温度和振动强度下的寿命。
    包含：
    1. 非比例风险 (Crossing Hazards): 材料 A (初期高风险) vs 材料 B (后期高风险)。
    2. 极度非线性效应: 温度 X1 的 U 型影响。
    3. 阈值效应: 振动 X2 的突变阈值。
    """
    num_features: int = 3

    def compute_true_hazard(self, x: np.ndarray, grid_t: np.ndarray) -> np.ndarray:
        """
        计算给定协变量和时间点的真实风险率 h(t|x)。
        x: (n, 3) 数组，特征为 [x0, x1, x2]
        grid_t: (m,) 时间点数组
        返回: (n, m) 风险率矩阵
        """
        n = x.shape[0]
        m = len(grid_t)
        h = np.zeros((n, m), dtype=np.float32)
        
        x1, x2, x3 = x[:, 0], x[:, 1], x[:, 2]
        mask_a = x3 == 0
        mask_b = x3 == 1
        
        # 处理时间 0，避免除以 0
        t = np.clip(grid_t, 1e-8, None)
        
        if np.any(mask_a):
            shape_a = 0.6
            scale_a = np.exp(3.0 - 2.0 * x1[mask_a]**2 - 1.5 * np.maximum(0, x2[mask_a] - 0.5))
            # Weibull hazard: h(t) = (k/lambda) * (t/lambda)^(k-1)
            # 为了计算效率，使用 exp(log)
            log_h_a = np.log(shape_a) - shape_a * np.log(scale_a[:, None]) + (shape_a - 1) * np.log(t[None, :])
            h[mask_a] = np.exp(log_h_a)
            
        if np.any(mask_b):
            shape_b = 2.5
            scale_b = np.exp(2.0 - 1.0 * x1[mask_b]**2 - 3.0 * x2[mask_b])
            log_h_b = np.log(shape_b) - shape_b * np.log(scale_b[:, None]) + (shape_b - 1) * np.log(t[None, :])
            h[mask_b] = np.exp(log_h_b)
            
        return h

    def compute_true_density(self, x: np.ndarray, grid_t: np.ndarray) -> np.ndarray:
        """
        计算给定协变量和时间点的真实概率密度 f(t|x)。
        Weibull density: f(t) = h(t) * S(t) = h(t) * exp(-(t/lambda)^k)
        """
        h = self.compute_true_hazard(x, grid_t)
        n = x.shape[0]
        m = len(grid_t)
        f = np.zeros((n, m), dtype=np.float32)
        
        x1, x2, x3 = x[:, 0], x[:, 1], x[:, 2]
        mask_a = x3 == 0
        mask_b = x3 == 1
        t = grid_t
        
        if np.any(mask_a):
            shape_a = 0.6
            scale_a = np.exp(3.0 - 2.0 * x1[mask_a]**2 - 1.5 * np.maximum(0, x2[mask_a] - 0.5))
            surv_a = np.exp(-np.power(t[None, :] / scale_a[:, None], shape_a))
            f[mask_a] = h[mask_a] * surv_a
            
        if np.any(mask_b):
            shape_b = 2.5
            scale_b = np.exp(2.0 - 1.0 * x1[mask_b]**2 - 3.0 * x2[mask_b])
            surv_b = np.exp(-np.power(t[None, :] / scale_b[:, None], shape_b))
            f[mask_b] = h[mask_b] * surv_b
            
        return f

    def generate(self, n: int, seed: int = 42, include_group: bool = True, censoring_rate: float = 0.3) -> pd.DataFrame:
        rng = np.random.default_rng(seed)
        
        # 1. 生成协变量
        # X1: 工作温度, Uniform(-1, 1)
        x1 = rng.uniform(-1, 1, size=n).astype(np.float32)
        # X2: 振动强度, Uniform(0, 1)
        x2 = rng.uniform(0, 1, size=n).astype(np.float32)
        # X3: 材料类型, Bernoulli(0.5)
        x3 = rng.binomial(1, 0.5, size=n).astype(np.float32)
        
        # 2. 生成真实的生存时间 T (Weibull 分布)
        # T = scale * (-ln(U))^(1/shape)
        death_times = np.zeros(n, dtype=np.float32)
        
        # 掩码：材料 A (X3=0) 和 材料 B (X3=1)
        mask_a = x3 == 0
        mask_b = x3 == 1
        
        # --- 材料 A: Shape=0.6 (初期高风险), Scale = exp(3.0 - 2.0*X1^2 - 1.5*max(0, X2-0.5))
        shape_a = 0.6
        scale_a = np.exp(3.0 - 2.0 * x1[mask_a]**2 - 1.5 * np.maximum(0, x2[mask_a] - 0.5))
        u_a = rng.uniform(0, 1, size=np.sum(mask_a))
        death_times[mask_a] = scale_a * np.power(-np.log(u_a), 1.0 / shape_a)
        
        # --- 材料 B: Shape=2.5 (后期高风险), Scale = exp(2.0 - 1.0*X1^2 - 3.0*X2)
        shape_b = 2.5
        scale_b = np.exp(2.0 - 1.0 * x1[mask_b]**2 - 3.0 * x2[mask_b])
        u_b = rng.uniform(0, 1, size=np.sum(mask_b))
        death_times[mask_b] = scale_b * np.power(-np.log(u_b), 1.0 / shape_b)
        
        # 3. 添加指数分布的删失时间 C ~ Exponential(lambda_c)
        # 我们通过迭代寻找合适的 lambda_c 以达到约 30% 的删失率
        # E[I(T > C)] = P(T > C)
        # 粗略估计：如果 T 的均值约为 E[T]，则 P(T > C) 约等于 E[T] / (E[T] + E[C])
        # 如果我们需要 30% 删失，即 P(T > C) = 0.3
        # 经验性地调整：对于目前的 Weibull 分布，scale 取 avg_t * 1.05 左右可以达到约 30% 删失
        avg_t = np.mean(death_times)
        c_scale = 1.05 * avg_t
        censoring_times = rng.exponential(scale=c_scale, size=n)
        
        # 4. 最终观测时间 Y = min(T, C), 状态指标 delta = I(T <= C)
        obs_time = np.minimum(death_times, censoring_times)
        event = (death_times <= censoring_times).astype(np.int32)
        
        # 5. 构建 DataFrame
        df = pd.DataFrame({
            "x0": x1,
            "x1": x2,
            "x2": x3,
            "time": obs_time.astype(np.float32),
            "event": event
        })
        
        if include_group:
            df["group"] = x3.astype(np.int32)
            
        return df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=3000, help="生成样本数量")
    parser.add_argument("--out_dir", type=str, default=os.path.join(ROOT, "results", "toy_datasets"))
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    gen = NonPHSimulatedData()
    df = gen.generate(n=args.n, seed=args.seed, include_group=True)
    
    os.makedirs(args.out_dir, exist_ok=True)
    out_csv = os.path.join(args.out_dir, "toy_non_ph_dataset.csv")
    df.to_csv(out_csv, index=False)
    
    censoring_rate = 1 - df['event'].mean()
    print(f"成功创建 Non-PH 玩具数据集: {out_csv}")
    print(f"样本总数: {len(df)}")
    print(f"发生事件数: {df['event'].sum()}, 删失率: {censoring_rate:.2%}")
    if "group" in df.columns:
        n_a = np.sum(df['group'] == 0)
        n_b = np.sum(df['group'] == 1)
        print(f"材料 A (早期高危, Shape=0.6) 样本数: {n_a}")
        print(f"材料 B (后期高危, Shape=2.5) 样本数: {n_b}")


if __name__ == "__main__":
    main()
