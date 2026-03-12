from __future__ import annotations

from typing import Dict

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from .components import FeatureEncoder, ModelOutputs, VectorFieldNet
from .compute_utils import (
    integrate_ode,  # ode求解器
    sample_truncated_times,  # 截断采样筛选合法截断样本
    safe_log,  # t -> logt
    flow_matching_targets,  # 最优传输路径
    ranking_regularizer,  # 风险排序损失
    hazard_from_survival_curve,  # mc法通过S(t)进行微分计算h
    build_prediction_bundle,  # 构建预测结果包
    log_density_via_cnf_reverse,  # CNF反向积分
    build_time_grid,


    median_from_cdf,
)


class GaussianFlowMatchingModel(nn.Module):
    def __init__(
        self,
        input_dim: int,
        encoder_hidden_dims: list[int],
        latent_dim: int,
        vf_hidden_dims: list[int],
        time_emb_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        if len(encoder_hidden_dims) == 0:
            raise ValueError("encoder_hidden_dims 不能为空")
        if len(vf_hidden_dims) == 0:
            raise ValueError("vf_hidden_dims 不能为空")
        latent_dim = int(latent_dim)
        if latent_dim <= 0:
            raise ValueError("latent_dim 必须为正整数")
        self.latent_dim = latent_dim

        # 特征编码器: z = encoder(x)
        self.encoder = FeatureEncoder(
            input_dim=input_dim,
            hidden_dims=encoder_hidden_dims,
            output_dim=latent_dim,
            dropout=dropout,
        )
        # 向量场: v = vector_field(t, t_tau, z)
        self.vector_field = VectorFieldNet(
            latent_dim=latent_dim,
            hidden_dims=vf_hidden_dims,
            time_emb_dim=time_emb_dim,
            dropout=dropout,
        )
        # 动态时间网格：记录训练集中生存时间的运行最大值，用于推理时的网格扩展
        self.register_buffer("running_max_t", torch.tensor(1.0))
        # 目标标准化参数 (y = log(t))
        self.register_buffer("y_mean", torch.tensor(0.0))
        self.register_buffer("y_std", torch.tensor(1.0))

    def set_target_normalization(self, t: Tensor, event: Tensor) -> None:
        """根据训练数据设置目标标准化参数"""
        with torch.no_grad():
            mask = event.view(-1) > 0.5
            if mask.sum() < 2:
                return
            t_event = t.view(-1)[mask]
            y = safe_log(t_event)
            self.y_mean.data = y.mean()
            self.y_std.data = y.std()
            # 避免 std 过小
            self.y_std.data = torch.clamp(self.y_std.data, min=1e-6)

    def _transform_target(self, t: Tensor) -> Tensor:
        """t -> y_std"""
        y = safe_log(t)
        return (y - self.y_mean) / self.y_std

    def _inverse_transform_target(self, y_std: Tensor) -> Tensor:
        """y_std -> t"""
        y = y_std * self.y_std + self.y_mean
        # 防止溢出
        y = torch.clamp(y, max=20.0)
        return torch.exp(y)

    def encode(self, x: Tensor) -> Tensor:
        """特征编码"""
        return self.encoder(x)

    def sample_prior(self, z: Tensor) -> Tensor:
        """先验采样"""
        # 对于 Gaussian, z 仅用于确定 batch size 和 device
        return torch.randn((z.shape[0], 1), device=z.device, dtype=z.dtype)


    def prior_log_prob(self, y0: Tensor, z: Tensor) -> Tensor:
        """计算对数先验密度,用于density方法计算生存时间密度"""
        # 对于 Gaussian, z 被忽略
        return -0.5 * (y0 ** 2 + torch.log(torch.tensor(2.0 * torch.pi, device=y0.device, dtype=y0.dtype)))


    def velocity(self, y_tau: Tensor, tau: Tensor, z: Tensor) -> Tensor:
        """向量场"""
        return self.vector_field(y_tau, tau, z)

    def forward_loss(
        self,
        x: Tensor,  # 特征协变量
        t_obs: Tensor,  # 观测到的生存时间
        event: Tensor,  # 事件
        rank_weight: float = 0.4,  # 风险排序损失权重
        rank_margin: float = 0.1,  # 风险排序损失 margin
        event_weight: float = 0.7,  # 事件样本权重
        truncation_samples: int = 16,  # 截断采样样本数
        truncation_max_trials: int = 8,  # 截断采样最大尝试次数
        truncation_ode_steps: int = 25,  # 截断采样的ode步数
        truncation_ode_method: str = "euler",  # 截断采样的ode算法
    ) -> ModelOutputs:
        """
        前向损失函数 L = L_flow + rw * L_rank
        """
        z = self.encode(x)  # 潜向量z

        # 动态更新时间网格边界 (仅在训练模式)
        if self.training:
            with torch.no_grad():
                batch_max = t_obs.max()
                # 采用指数移动平均或简单取最大值
                self.running_max_t.data = torch.max(self.running_max_t, batch_max)

        t_obs_2d = torch.clamp(t_obs.view(-1, 1), min=1e-6)
        t_target = t_obs_2d  # 目标时间

        event_mask = event.view(-1, 1) > 0.5  # 事件掩码
        cens_mask = ~event_mask  # 删失掩码

        # 如果当前批次存在删失样本
        if bool(cens_mask.any()):
            # ... (truncated for conciseness in thought, but using the actual code block for the replace)
            z_cens = z[cens_mask.view(-1)]
            n_cens = z_cens.shape[0]
            z_rep = z_cens.unsqueeze(1).repeat(1, truncation_samples, 1).reshape(n_cens * truncation_samples, -1)
            y0_cens = self.sample_prior(z_rep)
            # 一个闭包函数，用于计算向量场在当前时间tau下的所有截断采样样本的伪时间
            def field_fn(y_tau: Tensor, tau: Tensor) -> Tensor:
                return self.velocity(y_tau, tau, z_rep)

            # 使用 compute_utils 中封装的 ODE 求解器
            y1_cens = integrate_ode(
                y0=y0_cens,
                field_fn=field_fn,
                steps=truncation_ode_steps,
                method=truncation_ode_method,
                tau_start=0.0,
                tau_end=1.0,
            )

            # 伪时间
            y1_cens = torch.clamp(y1_cens, max=20.0) # 防止数值爆炸
            t_cens = self._inverse_transform_target(y1_cens).reshape(n_cens, truncation_samples)
            # 删失样本的观测时间
            lb = t_obs_2d[cens_mask].view(-1)
            # 筛选合法截断样本
            t_imp = sample_truncated_times(t_cens, lower_bound=lb, max_trials=truncation_max_trials)
            # 从合法样本中随机筛选作为伪时间
            choose_idx = torch.randint(0, truncation_samples, (n_cens,), device=t_target.device)
            t_imp_pick = t_imp[torch.arange(n_cens, device=t_target.device), choose_idx].view(-1, 1)
            # 替换删失样本目标时间为伪时间
            t_target = t_target.clone()
            t_target[cens_mask] = torch.clamp(t_imp_pick.view(-1), min=1e-8)

        # t -> logt -> std
        y1 = self._transform_target(t_target)
        y0 = self.sample_prior(z)
        tau = torch.rand_like(y0)
        y_tau, v_target = flow_matching_targets(y0, y1, tau)
        v_pred = self.velocity(y_tau, tau, z)

        # 计算 Flow Matching 损失
        mse = F.mse_loss(v_pred, v_target, reduction="none")
        # 针对事件和删失样本分别加权
        weights = torch.zeros_like(event.view(-1, 1))
        weights[event_mask] = event_weight
        weights[cens_mask] = 1.0 - event_weight
        flow_loss = (mse * weights).mean()

        # 修正风险排序损失：移除 torch.no_grad() 以便梯度回传
        # 用一个新的先验起点
        y0_for_rank = self.sample_prior(z)
        tau_zero = torch.zeros_like(y0_for_rank)

        # 在 τ=0 时刻预测速度，然后单步外推到 τ=1
        v_at_zero = self.velocity(y0_for_rank, tau_zero, z)
        y1_pred_from_zero = y0_for_rank + v_at_zero

        # risk 来自这个 τ=1 的预测
        risk = -y1_pred_from_zero.view(-1)
        
        rank_loss = rank_weight * ranking_regularizer(risk, t_obs.view(-1), event.view(-1), margin=rank_margin)

        total = flow_loss + rank_loss

        return ModelOutputs(loss=total, flow_loss=flow_loss, rank_loss=rank_loss, risk=risk)


    @torch.no_grad()
    def predict_via_mc_path(
        self,
        x: Tensor,
        time_grid: Tensor,
        n_samples: int = 100,
        ode_steps: int = 20,
        ode_method: str = "euler",
    ) -> Dict[str, Tensor]:
        """
        MC 路径（蒙特卡洛采样法）推理算法
        
        流程:
        1. 编码特征 z = Encoder(x)
        2. 从先验分布采样 y0 ~ p(y0) (batch * n_samples)
        3. 正向 ODE 积分求解 y1 (tau=0 -> tau=1)
        4. 转换到时间空间 t_pred = exp(y1)
        5. 计算经验生存函数 S(t) = Mean(t_pred > t)
        6. 计算风险函数 h(t) 和中位生存时间
        
        Args:
            x: 输入特征 [batch, input_dim]
            time_grid: 时间网格 [n_grid] (必须提供)
            n_samples: 每个样本的采样次数
            ode_steps: ODE 步数
            ode_method: ODE 求解方法 ('euler' or 'rk4')
            
        Returns:
            Dict: {
                "survival": S(t) [batch, n_grid],
                "hazard": h(t) [batch, n_grid],
                "density": f(t) [batch, n_grid] (通过数值微分获得),
                "cdf": F(t) [batch, n_grid],
                "median": median_time [batch, 1]
            }
        """
        batch_size = x.shape[0]
        n_grid = time_grid.shape[0]
        
        # 特征编码
        z = self.encode(x)
        # 准备相同潜变量的样本
        z_expanded = z.repeat_interleave(n_samples, dim=0)
        # 从先验中采样
        y0 = self.sample_prior(z_expanded)  # [batch * n_samples, 1]
        # ode求解
        def field_fn(y: Tensor, tau: Tensor) -> Tensor:
            return self.velocity(y, tau, z_expanded) 
        y1 = integrate_ode(
            y0=y0,
            field_fn=field_fn,
            steps=ode_steps,
            method=ode_method,
            tau_start=0.0,
            tau_end=1.0,
        )
        
        # y -> t
        # 防止数值溢出，设定上限已经在 _inverse_transform_target 中处理
        t_pred = self._inverse_transform_target(y1).view(batch_size, n_samples)  # [batch, n_samples]
        
        # S(t) = Mean(t_pred > t)
        grid_expanded = time_grid.view(1, 1, n_grid)
        t_pred_expanded = t_pred.unsqueeze(-1)
        surv = (t_pred_expanded > grid_expanded).float().mean(dim=1)  # [batch, n_grid]
        
        # F(t) = 1 - S(t)
        cdf = 1.0 - surv
        
        # h(t)
        haz = hazard_from_survival_curve(surv, time_grid)
        
        # 通过 CDF 的数值微分计算密度 f(t) = dF/dt
        dt = time_grid[1:] - time_grid[:-1]
        d_cdf = cdf[:, 1:] - cdf[:, :-1]
        dens_mid = d_cdf / torch.clamp(dt.view(1, -1), min=1e-8)
        dens = torch.zeros_like(cdf)
        dens[:, 1:] = dens_mid
        dens[:, 0] = dens[:, 1]
        dens = torch.clamp(dens, min=0.0)
        
        # 中位数: 样本中位数
        median_time = torch.median(t_pred, dim=1).values.view(-1, 1)
        
        bundle = build_prediction_bundle(surv, haz, dens, cdf)
        bundle["median"] = median_time
        return bundle


    def predict_via_density_path(
        self,
        x: Tensor,
        time_grid: Tensor,
        ode_steps: int = 20,
        ode_method: str = "rk4",
    ) -> Dict[str, Tensor]:
        """
        Density 路径（显式密度法）推理算法
        """
        # 核心优化：推理阶段全程开启 no_grad，彻底解决 ODE 积分导致的 CUDA OOM
        with torch.no_grad():
            batch_size = x.shape[0]
            n_grid_original = time_grid.shape[0]
            
            # 1. 扩展网格以实现虚拟最大值积分 (满足用户需求：预测时间范围 < 积分范围)
            # 使用运行最大值 running_max_t 作为动态基准，确保积分覆盖训练集的时间范围
            t_max_real = time_grid[-1]
            t_max_virtual = torch.max(t_max_real * 1.5, self.running_max_t * 1.2)
            
            # 在 log 空间一次性构造到虚拟上界的网格，避免线性拼接导致 dt 突变
            if t_max_virtual > t_max_real + 1e-4:
                log_t_min = torch.log(torch.clamp(time_grid[0], min=1e-8))
                log_t_max_real = torch.log(torch.clamp(t_max_real, min=1e-8))
                log_t_max_virtual = torch.log(torch.clamp(t_max_virtual, min=t_max_real + 1e-8))

                if n_grid_original >= 2:
                    log_span_real = torch.clamp(log_t_max_real - log_t_min, min=1e-8)
                    base_dlog = log_span_real / float(n_grid_original - 1)
                    extra_dlog = torch.clamp(log_t_max_virtual - log_t_max_real, min=0.0)
                    n_extra = int(torch.ceil(extra_dlog / torch.clamp(base_dlog, min=1e-8)).item())
                    n_extra = max(16, min(n_extra, max(32, n_grid_original * 2)))
                else:
                    n_extra = 16

                n_grid_full = n_grid_original + n_extra
                full_grid_norm = torch.linspace(0, 1, n_grid_full, device=time_grid.device, dtype=time_grid.dtype)
                full_grid_log = log_t_min + (log_t_max_virtual - log_t_min) * full_grid_norm
                full_grid = torch.exp(full_grid_log)
            else:
                full_grid = time_grid
                
            n_grid_full = full_grid.shape[0]
            
            # 特征Encode
            z = self.encode(x)
                
            # t -> y
            y_grid = self._transform_target(full_grid)
            
            # 展开以进行批处理
            z_expanded = z.repeat_interleave(n_grid_full, dim=0)
            y_target = y_grid.repeat(batch_size).view(-1, 1)
            
            # CNF 反向积分
            def velocity_fn(y: Tensor, tau: Tensor) -> Tensor:
                return self.velocity(y, tau, z_expanded)
                
            # 推理阶段不创建散度计算图，避免内存泄漏
            is_training = self.training
            y0, delta_logp = log_density_via_cnf_reverse(
                y_target,
                velocity_fn,
                steps=ode_steps,
                method=ode_method,
                create_graph=is_training
            )
            
            # 计算对数概率
            log_p0 = self.prior_log_prob(y0, z_expanded)
            log_py = log_p0 - delta_logp
            
            # 变回形状 [batch, n_grid_full]
            log_py = log_py.view(batch_size, n_grid_full)
            
            # Jacobian 修正
            y_grid_batch = y_grid.unsqueeze(0).expand(batch_size, n_grid_full)
            # log p(t) = log p(y_std) - log|dt/dy_std|
            # dt/dy_std = t * sigma = exp(y_raw) * sigma
            # log|dt/dy_std| = y_raw + log(sigma) = (y_std * sigma + mean) + log(sigma)
            # log_pt = log_py - (y_grid_batch * self.y_std + self.y_mean + safe_log(self.y_std))
            
            log_det_jacobian = y_grid_batch * self.y_std + self.y_mean + safe_log(self.y_std)
            log_pt = log_py - log_det_jacobian
            
            # 密度 f(t)
            ft = torch.exp(torch.clamp(log_pt, min=-50.0, max=50.0))
            
            # 直接由原始密度累计 CDF，不做强制归一化
            dt = full_grid[1:] - full_grid[:-1]
            dt = dt.view(1, -1)
            
            # 计算 CDF
            cdf = torch.zeros_like(ft)
            for i in range(1, n_grid_full):
                step_area = 0.5 * (ft[:, i] + ft[:, i-1]) * (full_grid[i] - full_grid[i-1])
                cdf[:, i] = cdf[:, i-1] + step_area
            
            cdf = torch.clamp(cdf, 0.0, 1.0)
            surv = 1.0 - cdf
            
            # 将全网格结果插值回原始 time_grid，保持下游接口不变
            if full_grid.shape[0] == n_grid_original and torch.allclose(full_grid, time_grid):
                cdf_ret = cdf
            else:
                idx_right = torch.searchsorted(full_grid, time_grid)
                idx_right = torch.clamp(idx_right, min=1, max=full_grid.shape[0] - 1)
                idx_left = idx_right - 1

                t_left = full_grid[idx_left]
                t_right = full_grid[idx_right]
                weight = (time_grid - t_left) / torch.clamp(t_right - t_left, min=1e-8)
                weight = weight.view(1, -1)

                cdf_left = cdf[:, idx_left]
                cdf_right = cdf[:, idx_right]
                cdf_ret = cdf_left + (cdf_right - cdf_left) * weight

            cdf_ret = torch.clamp(cdf_ret, 0.0, 1.0)
            surv_ret = 1.0 - cdf_ret

            ft_ret = torch.zeros_like(cdf_ret)
            if n_grid_original >= 2:
                dt_ret = torch.clamp(time_grid[1:] - time_grid[:-1], min=1e-8).view(1, -1)
                dens_mid = (cdf_ret[:, 1:] - cdf_ret[:, :-1]) / dt_ret
                ft_ret[:, 1:] = dens_mid
                ft_ret[:, 0] = ft_ret[:, 1]
            ft_ret = torch.clamp(ft_ret, min=0.0)

            # 风险函数
            haz_ret = hazard_from_survival_curve(surv_ret, time_grid)
            
            # 中位生存时间
            median_time = median_from_cdf(full_grid, cdf)
            
            bundle = build_prediction_bundle(surv_ret, haz_ret, ft_ret, cdf_ret)
            bundle["median"] = median_time
            return bundle


    def predict_bundle(
        self,
        x: Tensor,
        ode_solver: Callable,
        grid_t: Tensor,
        mc_samples: int = 100,
        method: str = "mc",
    ) -> Dict[str, Tensor]:
        """
        统一推理接口，兼容旧代码。
        根据 method 选择 'mc' 或 'density' 路径。
        """
        if method.lower() == "density":
            # 尽可能从求解器中提取 ode_steps 和 ode_method，否则使用默认值
            steps = getattr(ode_solver, "ode_steps", 20)
            solver_method = getattr(ode_solver, "ode_method", "rk4")
            return self.predict_via_density_path(x, grid_t, ode_steps=steps, ode_method=solver_method)
        else:
            # 默认为 MC 方法
            steps = getattr(ode_solver, "ode_steps", 20)
            solver_method = getattr(ode_solver, "ode_method", "euler")
            return self.predict_via_mc_path(x, grid_t, n_samples=mc_samples, ode_steps=steps, ode_method=solver_method)


    
