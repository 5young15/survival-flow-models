from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Tuple

import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F


def _ensure_2d(x: Tensor) -> Tensor:
    """确保输入形状为 (B, D)"""
    if x.dim() == 1:
        return x.unsqueeze(-1)
    if x.dim() > 2:
        while x.dim() > 2 and x.size(1) == 1:
            x = x.squeeze(1)
        if x.dim() > 2:
            x = x.view(x.size(0), -1)
    return x


class ResidualBlock(nn.Module):
    """简单残差块（无条件调制）"""
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        dropout: float,
    ):
        super().__init__()
        self.norm = nn.LayerNorm(input_dim)
        self.fc = nn.Linear(input_dim, output_dim)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        self.residual_proj = (
            nn.Linear(input_dim, output_dim, bias=False)
            if input_dim != output_dim
            else nn.Identity()
        )

        # Kaiming 初始化
        nn.init.kaiming_normal_(self.fc.weight, mode='fan_in', nonlinearity='relu')
        nn.init.zeros_(self.fc.bias)
        if isinstance(self.residual_proj, nn.Linear):
            nn.init.kaiming_normal_(self.residual_proj.weight, mode='fan_in', nonlinearity='linear')

    def forward(self, x: Tensor) -> Tensor:
        residual = self.residual_proj(x)
        x_norm = self.norm(x)
        x = F.gelu(self.fc(x_norm))
        x = self.dropout(x)
        return residual + x


class FeatureEncoder(nn.Module):
    """表格特征编码器（多层残差 MLP）"""

    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        output_dim: int,
        dropout: float,
    ):
        super().__init__()
        if not hidden_dims:
            raise ValueError("hidden_dims 不能为空")

        dims = [input_dim] + hidden_dims + [output_dim]

        self.blocks = nn.ModuleList([
            ResidualBlock(dims[i], dims[i + 1], dropout=dropout)
            for i in range(len(dims) - 1)
        ])

        self.final_norm = nn.LayerNorm(output_dim)

    def forward(self, x: Tensor) -> Tensor:
        x = _ensure_2d(x)
        for block in self.blocks:
            x = block(x)
        return self.final_norm(x)


class TimeEmbedding(nn.Module):
    """时间嵌入：正弦编码 + MLP 变换"""

    def __init__(self, emb_dim: int, hidden_dim: int):
        super().__init__()
        assert emb_dim % 2 == 0, "正弦编码维度 emb_dim 必须为偶数"
        self.emb_dim = emb_dim
        
        self.mlp = nn.Sequential(
            nn.Linear(emb_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Kaiming 初始化
        for m in self.mlp:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                nn.init.zeros_(m.bias)

    def forward(self, tau: Tensor) -> Tensor:
        tau = _ensure_2d(tau)  # (B, 1)
        half_dim = self.emb_dim // 2
        
        # 1. 正弦编码 (Sinusoidal Encoding)
        # 使用 exp( -log(10000) * i / half_dim ) 提高数值稳定性
        emb = torch.arange(half_dim, dtype=tau.dtype, device=tau.device)
        freq = torch.exp(-torch.log(torch.tensor(10000.0, device=tau.device)) * emb / half_dim)
        emb = tau * freq
        emb = torch.cat([emb.sin(), emb.cos()], dim=-1)  # (B, emb_dim)
        
        # 2. MLP 变换
        return self.mlp(emb)  # (B, hidden_dim)


class AdaLNZero(nn.Module):
    """AdaLN-Zero 调制参数生成器"""

    def __init__(self, cond_dim: int, input_dim: int, output_dim: int):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(cond_dim, input_dim * 2 + output_dim)
        )
        # 零初始化 (Zero-init)
        nn.init.zeros_(self.mlp[1].weight)
        nn.init.zeros_(self.mlp[1].bias)

        with torch.no_grad():
            self.mlp[1].bias[2 * input_dim:].fill_(-10.0)

    def forward(self, cond: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        params = self.mlp(cond)
        scale = params[:, :self.input_dim] + 1.0
        shift = params[:, self.input_dim:2 * self.input_dim]
        gate = torch.sigmoid(params[:, 2 * self.input_dim:])
        return scale, shift, gate


class ResidualAdaLNZeroBlock(nn.Module):
    """AdaLN-Zero 条件残差块"""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        cond_dim: int,
        dropout: float,
    ):
        super().__init__()
        self.norm = nn.LayerNorm(input_dim)
        self.adalnzero = AdaLNZero(cond_dim, input_dim, output_dim)
        self.fc = nn.Linear(input_dim, output_dim)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        self.residual_proj = (
            nn.Linear(input_dim, output_dim, bias=False)
            if input_dim != output_dim
            else nn.Identity()
        )

        # Kaiming 初始化 (除了 AdaLNZero 已经处理了它的 mlp)
        nn.init.kaiming_normal_(self.fc.weight, mode='fan_in', nonlinearity='relu')
        nn.init.zeros_(self.fc.bias)
        if isinstance(self.residual_proj, nn.Linear):
            nn.init.kaiming_normal_(self.residual_proj.weight, mode='fan_in', nonlinearity='linear')

    def forward(self, h: Tensor, cond: Tensor) -> Tensor:
        residual = self.residual_proj(h)
        h_norm = self.norm(h)
        scale, shift, gate = self.adalnzero(cond)
        h_mod = h_norm * scale + shift
        h_act = F.silu(h_mod)
        h_fc = self.fc(h_act)
        h_dp = self.dropout(h_fc)
        return residual + gate * h_dp


class VectorFieldNet(nn.Module):
    """向量场网络，用于估计 v_θ(τ, t_τ) ≈ dt/dτ"""

    def __init__(
        self,
        latent_dim: int,
        hidden_dims: List[int],
        time_emb_dim: int,
        dropout: float,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.time_emb_dim = time_emb_dim
        self.cond_dim = latent_dim + time_emb_dim

        # 1. 流时间嵌入：正弦编码 + MLP 变换
        # 使用 1/4 的维度进行正弦编码，然后 MLP 映射回 time_emb_dim
        # 或者直接使用 time_emb_dim 作为基础编码维度
        self.time_emb = TimeEmbedding(emb_dim=time_emb_dim, hidden_dim=time_emb_dim)

        input_dim = 1 + self.cond_dim  # t_τ + cond

        dims_in = [input_dim] + hidden_dims
        dims_out = hidden_dims + [hidden_dims[-1]]  # 最后一个 block 保持 hidden_dims[-1]

        self.blocks = nn.ModuleList([
            ResidualAdaLNZeroBlock(
                input_dim=dims_in[i],
                output_dim=dims_out[i],
                cond_dim=self.cond_dim,
                dropout=dropout,
            )
            for i in range(len(hidden_dims))
        ])

        self.output_layer = nn.Sequential(
            nn.LayerNorm(hidden_dims[-1]),
            nn.Linear(hidden_dims[-1], 1),
        )

        # 3. 零输出初始化 (Zero-init output layer)
        # 确保训练初期 v ≈ 0, 模型从认同先验分布开始平滑演化
        nn.init.zeros_(self.output_layer[1].weight)
        nn.init.zeros_(self.output_layer[1].bias)

    def forward(
        self,
        t_tau: Tensor,
        tau: Tensor,
        z: Tensor,
    ) -> Tensor:
        t_tau = _ensure_2d(t_tau)
        tau = _ensure_2d(tau)
        
        # 1. 时间嵌入 (内置 MLP)
        tau_emb = self.time_emb(tau)
        
        # 2. 构造条件向量: z (特征) + tau_emb (时间)
        cond = torch.cat([z, tau_emb], dim=-1)
        h = torch.cat([t_tau, cond], dim=-1)

        for block in self.blocks:
            h = block(h, cond)

        return self.output_layer(h)


class MLP(nn.Module):
    """通用 MLP（用于 Gumbel 等参数预测）"""

    def __init__(
        self,
        in_dim: int,
        hidden_dims: Iterable[int],
        out_dim: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        dims = [in_dim, *hidden_dims, out_dim]
        layers = []
        for i in range(len(dims) - 2):
            layers.extend([
                nn.Linear(dims[i], dims[i + 1]),
                nn.SiLU(),
            ])
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(dims[-2], dims[-1]))
        self.net = nn.Sequential(*layers)

        # Kaiming 初始化
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                nn.init.zeros_(m.bias)

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class GumbelParamNet(nn.Module):
    """Gumbel 分布参数预测网络"""

    def __init__(
        self,
        latent_dim: int,
        hidden_dims: Iterable[int],
        min_scale: float = 1e-2,
    ):
        super().__init__()
        self.net = MLP(latent_dim, hidden_dims, 2)
        self.min_scale = min_scale

    def forward(self, z: Tensor) -> Tuple[Tensor, Tensor]:
        loc, raw_scale = self.net(z).chunk(2, dim=-1)
        scale = F.softplus(raw_scale) + self.min_scale
        return loc, scale


@dataclass
class ModelOutputs:
    loss: Tensor
    flow_loss: Tensor
    rank_loss: Tensor
    risk: Tensor
