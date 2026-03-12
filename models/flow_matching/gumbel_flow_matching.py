from __future__ import annotations

from typing import Iterable, Tuple

import torch
from torch import Tensor
import torch.nn.functional as F

from .components import GumbelParamNet
from .gaussian_flow_matching import GaussianFlowMatchingModel
from .compute_utils import safe_log


class GumbelFlowMatchingModel(GaussianFlowMatchingModel):
    def __init__(
        self,
        input_dim: int,
        encoder_hidden_dims: Iterable[int],
        latent_dim: int,
        vf_hidden_dims: Iterable[int],
        gumbel_hidden_dims: Iterable[int],
        time_emb_dim: int,
        dropout: float = 0.0,
    ) -> None:
        gumbel_hidden_dims = list(gumbel_hidden_dims)
        super().__init__(
            input_dim=input_dim,
            encoder_hidden_dims=encoder_hidden_dims,
            latent_dim=latent_dim,
            vf_hidden_dims=vf_hidden_dims,
            time_emb_dim=time_emb_dim,
            dropout=dropout,
        )
        self.gumbel_head = GumbelParamNet(latent_dim=self.latent_dim, hidden_dims=gumbel_hidden_dims)

    def get_gumbel_params(self, z: Tensor) -> Tuple[Tensor, Tensor]:
        """预测 Gumbel 分布参数 (loc, scale)"""
        return self.gumbel_head(z)

    def sample_prior(self, z: Tensor) -> Tensor:
        """从 Gumbel 先验 (MinGumbel) 采样"""
        loc, scale = self.get_gumbel_params(z)
        u = torch.rand_like(loc)
        # 避免出现 log(0)
        u = torch.clamp(u, min=1e-8, max=1.0 - 1e-8)
        # MinGumbel 采样 g = log(-log(u))
        g = torch.log(-torch.log(u))
        return loc + scale * g

    def prior_log_prob(self, y0: Tensor, z: Tensor) -> Tensor:
        """计算 Gumbel 先验 (MinGumbel) 的对数密度"""
        loc, scale = self.get_gumbel_params(z)
        # 归一化: z_norm = (y - loc) / scale
        # 增加 eps 防止除零
        z_norm = (y0 - loc) / torch.clamp(scale, min=1e-8)
        # 密度公式 (MinGumbel): log p(y) = -log(scale) + z_norm - exp(z_norm)
        # 使用 safe_log
        log_prob = -safe_log(scale) + z_norm - torch.exp(torch.clamp(z_norm, max=20.0))
        return log_prob

    def initialize_gumbel_prior(self, time: Tensor, event: Tensor) -> None:
        """
        利用训练数据的事件分布初始化 Gumbel 先验参数 (Min-Gumbel)。
        目标：使初始先验分布尽可能接近对数生存时间的经验分布。
        
        Min-Gumbel 分布统计量:
        E[Y] = mu - gamma * beta
        Var(Y) = (pi^2 / 6) * beta^2
        其中 gamma ≈ 0.5772
        """
        with torch.no_grad():
            # 1. 筛选事件样本 (delta = 1)
            mask = event.view(-1) > 0.5
            if mask.sum() < 2:
                print("Warning: Not enough event samples to initialize Gumbel prior.")
                return
                
            t_event = time.view(-1)[mask]
            
            # 2. 计算目标值 (可能是标准化的)
            # 使用统一的 transform 接口
            y = self._transform_target(t_event)
            
            y_bar = y.mean().item()
            s_y = y.std().item()
            
            # 3. 反推先验参数
            # beta = (sqrt(6) / pi) * s_y ≈ 0.7797 * s_y
            import math
            beta_init = (math.sqrt(6) / math.pi) * s_y
            
            # mu = y_bar + gamma * beta
            gamma = 0.5772156649
            mu_init = y_bar + gamma * beta_init
            
            # 4. 初始化网络输出层
            # 权重初始化：使用统一标准差 (0.2) 保留足够梯度流，避免初始化阶段学习停滞
            final_layer = self.gumbel_head.net.net[-1]
            torch.nn.init.normal_(final_layer.weight, mean=0.0, std=0.2)
            
            # 计算偏置：使初始输出接近目标均值和方差
        # scale = softplus(raw_scale) + min_scale => raw_scale = inverse_softplus(scale - min_scale)
        target_scale = max(beta_init - self.gumbel_head.min_scale, 1e-6)
        raw_scale_init = torch.log(torch.expm1(torch.tensor(target_scale))).item()
        
        with torch.no_grad():
            final_layer.bias[0] = mu_init
            final_layer.bias[1] = raw_scale_init

    def stage1_loss(self, x: Tensor, t: Tensor, e: Tensor) -> Tensor:
        """第一阶段损失：最大化对数似然估计 MinGumbel 先验参数 (考虑删失)"""
        z = self.encode(x)
        
        # 动态更新时间网格边界
        if self.training:
            with torch.no_grad():
                self.running_max_t.data = torch.max(self.running_max_t, t.max())

        # 使用统一的目标变换 (t -> y_std)
        y = self._transform_target(t.view(-1, 1))
        
        loc, scale = self.get_gumbel_params(z)
        
        # 归一化: z_norm = (y - loc) / scale
        # 增加数值稳定性：限制 z_norm 的范围，防止 exp(z_norm) 爆炸
        z_norm = (y - loc) / torch.clamp(scale, min=1e-6)
        z_norm = torch.clamp(z_norm, min=-15.0, max=15.0)
        
        # u = exp(z_norm)
        u = torch.exp(z_norm)
        
        # 事件发生样本 (Event=1): log PDF
        # log p(y) = -log(scale) + z_norm - exp(z_norm)
        log_pdf = -safe_log(scale) + z_norm - u
        
        # 删失样本 (Event=0): log Survival
        # S(y) = exp(-exp(z_norm)) = exp(-u)
        # log S(y) = -u
        log_surv = -u
        
        # 组合损失: NLL
        event_mask = e.view(-1, 1)
        loss = -(event_mask * log_pdf + (1.0 - event_mask) * log_surv)
        return loss.mean()
