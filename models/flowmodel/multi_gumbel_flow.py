import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Union, Optional, Dict, Tuple

from models.flowmodel.base_flow import FlowSurv
from models.flowmodel.components import FiLMResidualBlock, odeint_euler, odeint_rk4


class MultiGumbelFlowSurv(FlowSurv):
    """
    二维Min-Gumbel流匹配生存模型
    
    同时建模失效时间T和删失时间C的联合分布
    
    核心思想:
    - 第一阶段: 预训练二元Min-Gumbel分布参数 (μ_T, μ_C, β)
    - 第二阶段: 流匹配学习 (y_T, y_C) → Min-Gumbel先验的映射
    
    数学推导:
    - 联合CDF: F(y_T,y_C) = exp(-(exp(-(y_T-μ_T)/β) + exp(-(y_C-μ_C)/β)))
    - 流匹配: z(s) = (1-s)z_0 + s*z_1, 损失 = ||v_θ(z(s),x) - (z_1-z_0)||²
    
    与GumbelFlowSurv的差异:
    - gumbel_head输出3个参数(μ_T, μ_C, β)而非2个(α, β)
    - vf_in_dim=2(二维向量场)而非1
    - 两阶段训练: pretrain + flow
    """
    
    _LN_2 = 0.6931471805599453
    _LN_LN_2 = -0.3665129205816643
    
    def __init__(self, in_dim: int, config: Optional[dict] = None, **kwargs):
        super().__init__(in_dim, config, **kwargs)
        self._stage = 'gumbel'
        self.use_joint_loss = self.config.get('use_joint_loss', False)
        
        z_dim = self.encoder_dims[-1]
        gumbel_layers = []
        prev = z_dim
        for d in self.config.get('gumbel_head_hidden', [16, 8]):
            gumbel_layers.extend([nn.Linear(prev, d), nn.SiLU()])
            prev = d
        gumbel_layers.append(nn.Linear(prev, 3))
        self.gumbel_head = nn.Sequential(*gumbel_layers)
        
        self.vf_in_dim = 2
        self.film_head = self._build_film_head()
        self.vector_field = self._build_vector_field()
        
        self._set_stage('gumbel')
    
    def _build_film_head(self) -> nn.Module:
        film_hidden = self.config.get('film_hidden', [16])
        film_layers = []
        prev = self.encoder_dims[-1]
        for d in film_hidden:
            film_layers.extend([nn.Linear(prev, d), nn.SiLU()])
            prev = d
        film_layers.append(nn.Linear(prev, self.vf_in_dim * 2))
        return nn.Sequential(*film_layers)
    
    def _build_vector_field(self) -> nn.Module:
        vf_hidden_dims = self.config.get('vf_hidden_dims', [32, 16, 8])
        dropout = self.config.get('dropout', 0.1)
        vf_layers = []
        prev_dim = self.vf_in_dim
        for d in vf_hidden_dims:
            vf_layers.append(FiLMResidualBlock(prev_dim, d, self.tau_dim, self.vf_in_dim * 2, dropout))
            prev_dim = d
        vf_layers.append(nn.Linear(prev_dim, self.vf_in_dim))
        return nn.ModuleList(vf_layers)
    
    def _set_stage(self, stage: str):
        if stage not in {'gumbel', 'flow'}:
            raise ValueError(f"Unknown stage: {stage}")
        self._stage = stage
        if stage == 'gumbel':
            self.encoder.requires_grad_(True)
            self.gumbel_head.requires_grad_(True)
            self.film_head.requires_grad_(False)
            self.vector_field.requires_grad_(False)
        else:
            self.encoder.requires_grad_(True)
            self.gumbel_head.requires_grad_(False)
            self.film_head.requires_grad_(True)
            self.vector_field.requires_grad_(True)
    
    def set_stage(self, stage: str):
        self._set_stage(stage)
    
    def _gumbel_params(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        params = self.gumbel_head(z)
        mu_T = torch.clamp(params[:, 0:1], -15.0, 15.0)
        mu_C = torch.clamp(params[:, 1:2], -15.0, 15.0)
        log_beta = torch.clamp(params[:, 2:3], -10.0, 5.0)
        beta = torch.exp(log_beta)
        beta = torch.clamp(beta, min=1e-5, max=100.0)
        return mu_T, mu_C, beta
    
    def get_gumbel_params(self, x: torch.Tensor, z: Optional[torch.Tensor] = None
                         ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if z is None:
            z = self.encoder(x)
        z_gumbel = z.detach() if self._stage == 'flow' else z
        return self._gumbel_params(z_gumbel)
    
    def init_gumbel_params(self, times: torch.Tensor, events: torch.Tensor = None):
        t_norm = self._to_normalized_time(times)
        mu_T_init = t_norm.mean().item()
        sigma = t_norm.std().item()
        
        mu_C_init = mu_T_init + 0.5
        
        sqrt_6 = math.sqrt(6.0)
        beta_init = max(sigma * sqrt_6 / math.pi, 0.1)
        
        with torch.no_grad():
            self.gumbel_head[-1].weight.fill_(0)
            self.gumbel_head[-1].bias[0].fill_(mu_T_init)
            self.gumbel_head[-1].bias[1].fill_(mu_C_init)
            self.gumbel_head[-1].bias[2].fill_(math.log(beta_init))
    
    def sample_min_gumbel(self, shape, device, mu_T, mu_C, beta) -> torch.Tensor:
        B = shape[0] if isinstance(shape, tuple) else shape
        u = torch.rand(B, 2, device=device).clamp_(1e-10, 1.0 - 1e-10)
        log_neg_log_u = -torch.log(-torch.log(u))
        z_std = torch.clamp(log_neg_log_u, min=-15.0, max=15.0)
        
        z_T = mu_T + beta * z_std[:, 0:1]
        z_C = mu_C + beta * z_std[:, 1:2]
        
        z_T = torch.clamp(z_T, min=-20.0, max=20.0)
        z_C = torch.clamp(z_C, min=-20.0, max=20.0)
        
        return torch.cat([z_T, z_C], dim=-1)
    
    def log_prob_min_gumbel(self, z_T: torch.Tensor, z_C: torch.Tensor,
                            mu_T: torch.Tensor, mu_C: torch.Tensor, beta: torch.Tensor
                            ) -> torch.Tensor:
        """
        计算 Min-Gumbel 联合分布的对数密度 log f_{T,C}(t, c)
        
        数学推导 (参考 docs/MultiGumbelFlow_Model.md):
        - 联合 CDF: F(t,c) = exp(-(exp((t-μ_T)/β) + exp((c-μ_C)/β)))
        - 联合 PDF: f(t,c) = F(t,c) * exp((t-μ_T)/β)/β * exp((c-μ_C)/β)/β * (exp((t-μ_T)/β) + exp((c-μ_C)/β))
        - 对数密度: log f = -sum_exp - std_T - log(β) - std_C - log(β) + log(sum_exp)
        """
        beta = torch.clamp(beta, min=1e-5, max=100.0)
        
        std_T = torch.clamp((z_T - mu_T) / beta, min=-20.0, max=10.0)
        std_C = torch.clamp((z_C - mu_C) / beta, min=-20.0, max=10.0)
        
        exp_std_T = torch.clamp(torch.exp(std_T), min=0.0, max=1e10)
        exp_std_C = torch.clamp(torch.exp(std_C), min=0.0, max=1e10)
        
        sum_exp = exp_std_T + exp_std_C
        log_F = -sum_exp
        
        log_beta = torch.log(beta)
        log_sum_exp = torch.log(torch.clamp(sum_exp, min=1e-10))
        
        log_f = log_F + (-std_T - log_beta) + (-std_C - log_beta) + log_sum_exp
        
        return torch.clamp(log_f, min=-100.0, max=88.0)
    
    def log_prob_gumbel_marginal(self, z: torch.Tensor, mu: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
        beta = torch.clamp(beta, min=1e-5, max=100.0)
        std = torch.clamp((z - mu) / beta, min=-20.0, max=10.0)
        exp_std = torch.clamp(torch.exp(std), min=0.0, max=1e10)
        log_prob = -torch.log(beta) + std - exp_std
        return torch.clamp(log_prob, min=-100.0, max=88.0)
    
    def log_surv_gumbel_marginal(self, z: torch.Tensor, mu: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
        beta = torch.clamp(beta, min=1e-5, max=100.0)
        std = torch.clamp((z - mu) / beta, min=-20.0, max=10.0)
        log_surv = -torch.exp(std)
        return torch.clamp(log_surv, min=-100.0, max=0.0)
    
    def sample_gumbel_truncated_above(self, t: torch.Tensor, mu: torch.Tensor, beta: torch.Tensor
                                      ) -> torch.Tensor:
        """
        从 Gumbel 条件分布 p(C | C > t) 采样
        
        数学推导:
        - Gumbel 生存函数: S_C(t) = exp(-exp((t - μ_C)/β))
        - 条件分布采样: c = μ_C - β * log(-log(S_C(t) * u + (1 - S_C(t))))
          其中 u ~ Uniform(0,1)
        
        简化形式:
        - 令 v = 1 - u, v ~ Uniform(0,1)
        - c = μ_C - β * log(-log(1 - S_C(t) * v))
        - 当 v ∈ (0,1), c ∈ (t, +∞)
        """
        beta = torch.clamp(beta, min=1e-5, max=100.0)
        std_t = torch.clamp((t - mu) / beta, min=-20.0, max=10.0)
        S_t = torch.exp(-torch.exp(std_t))
        S_t = torch.clamp(S_t, min=1e-10, max=1.0 - 1e-10)
        
        v = torch.rand_like(t)
        inner = 1.0 - S_t * v
        inner = torch.clamp(inner, min=1e-10, max=1.0 - 1e-10)
        
        c = mu - beta * torch.log(-torch.log(inner))
        c = torch.max(c, t)
        c = torch.clamp(c, max=20.0)
        return c
    
    def sample_gumbel_truncated_below(self, t: torch.Tensor, mu: torch.Tensor, beta: torch.Tensor
                                      ) -> torch.Tensor:
        """
        从 Gumbel 条件分布 p(C | C < t) 采样
        
        数学推导:
        - Gumbel CDF: F_C(t) = 1 - exp(-exp((t - μ_C)/β))
        - 条件分布采样: c = μ_C - β * log(-log(1 - F_C(t) * u))
          其中 u ~ Uniform(0,1)
        """
        beta = torch.clamp(beta, min=1e-5, max=100.0)
        std_t = torch.clamp((t - mu) / beta, min=-20.0, max=10.0)
        F_t = 1.0 - torch.exp(-torch.exp(std_t))
        F_t = torch.clamp(F_t, min=1e-10, max=1.0 - 1e-10)
        
        u = torch.rand_like(t)
        inner = 1.0 - F_t * u
        inner = torch.clamp(inner, min=1e-10, max=1.0 - 1e-10)
        
        c = mu - beta * torch.log(-torch.log(inner))
        c = torch.min(c, t)
        c = torch.clamp(c, min=-20.0)
        return c
    
    def gumbel_truncated_above_median(self, t: torch.Tensor, mu: torch.Tensor, beta: torch.Tensor
                                      ) -> torch.Tensor:
        """
        计算 Gumbel 条件分布 p(C | C > t) 的中位数（确定性）
        
        数学推导:
        - 中位数对应于 u = 0.5
        - c_median = μ_C - β * log(-log(1 - S_C(t) * 0.5))
        """
        beta = torch.clamp(beta, min=1e-5, max=100.0)
        std_t = torch.clamp((t - mu) / beta, min=-20.0, max=10.0)
        S_t = torch.exp(-torch.exp(std_t))
        S_t = torch.clamp(S_t, min=1e-10, max=1.0 - 1e-10)
        
        inner = 1.0 - S_t * 0.5
        inner = torch.clamp(inner, min=1e-10, max=1.0 - 1e-10)
        
        c = mu - beta * torch.log(-torch.log(inner))
        c = torch.max(c, t)
        c = torch.clamp(c, max=20.0)
        return c
    
    def forward_loss(self, features: torch.Tensor, times_raw: torch.Tensor, events: torch.Tensor, **kwargs):
        device = features.device
        t1 = self._to_normalized_time(times_raw).float().unsqueeze(-1)
        
        if self._stage == 'gumbel':
            return self._pretrain_loss(features, t1, events, device)
        else:
            return self._flow_loss(features, t1, events, device)
    
    def _pretrain_loss(self, features: torch.Tensor, t1: torch.Tensor, 
                       events: torch.Tensor, device: torch.device) -> Tuple[torch.Tensor, Dict]:
        z = self.encoder(features)
        mu_T, mu_C, beta = self._gumbel_params(z)
        
        event_mask = (events == 1)
        censored_mask = (events == 0)
        
        total_loss = torch.zeros(1, device=device, requires_grad=True)
        loss_dict = {'pretrain_loss': 0.0, 'event_loss': 0.0, 'censored_loss': 0.0}
        
        if self.use_joint_loss:
            if event_mask.any():
                t_event = t1[event_mask]
                mu_T_event = mu_T[event_mask]
                mu_C_event = mu_C[event_mask]
                beta_event = beta[event_mask]
                
                c_sample = self.sample_gumbel_truncated_above(t_event, mu_C_event, beta_event)
                log_joint_event = self.log_prob_min_gumbel(t_event, c_sample, mu_T_event, mu_C_event, beta_event)
                
                event_loss = -log_joint_event.mean()
                total_loss = total_loss + event_loss
                loss_dict['event_loss'] = event_loss.item()
            
            if censored_mask.any():
                t_cens = t1[censored_mask]
                mu_T_cens = mu_T[censored_mask]
                mu_C_cens = mu_C[censored_mask]
                beta_cens = beta[censored_mask]
                
                t_sample = self.sample_gumbel_truncated_above(t_cens, mu_T_cens, beta_cens)
                log_joint_cens = self.log_prob_min_gumbel(t_sample, t_cens, mu_T_cens, mu_C_cens, beta_cens)
                
                censored_loss = -log_joint_cens.mean()
                total_loss = total_loss + censored_loss
                loss_dict['censored_loss'] = censored_loss.item()
        else:
            if event_mask.any():
                t_event = t1[event_mask]
                mu_T_event = mu_T[event_mask]
                mu_C_event = mu_C[event_mask]
                beta_event = beta[event_mask]
                
                log_prob_T = self.log_prob_gumbel_marginal(t_event, mu_T_event, beta_event)
                log_surv_C = self.log_surv_gumbel_marginal(t_event, mu_C_event, beta_event)
                log_lik_event = log_prob_T + log_surv_C
                
                event_loss = -log_lik_event.mean()
                total_loss = total_loss + event_loss
                loss_dict['event_loss'] = event_loss.item()
            
            if censored_mask.any():
                t_cens = t1[censored_mask]
                mu_T_cens = mu_T[censored_mask]
                mu_C_cens = mu_C[censored_mask]
                beta_cens = beta[censored_mask]
                
                log_prob_C = self.log_prob_gumbel_marginal(t_cens, mu_C_cens, beta_cens)
                log_surv_T = self.log_surv_gumbel_marginal(t_cens, mu_T_cens, beta_cens)
                log_lik_cens = log_prob_C + log_surv_T
                
                censored_loss = -log_lik_cens.mean()
                total_loss = total_loss + censored_loss
                loss_dict['censored_loss'] = censored_loss.item()
        
        loss_dict['pretrain_loss'] = total_loss.item()
        return total_loss, loss_dict
    
    def _flow_loss(self, features: torch.Tensor, t1: torch.Tensor,
                   events: torch.Tensor, device: torch.device) -> Tuple[torch.Tensor, Dict]:
        z = self.encoder(features)
        mod_params = self.film_head(z)
        mu_T, mu_C, beta = self.get_gumbel_params(features, z=z)
        
        event_mask = (events == 1)
        censored_mask = (events == 0)
        
        if not event_mask.any() and not censored_mask.any():
            return torch.tensor(0.0, device=device, requires_grad=True), {}
        
        total_loss = torch.zeros(1, device=device, requires_grad=True)
        loss_dict = {'event_loss': 0.0, 'censored_loss': 0.0}
        
        if event_mask.any():
            t1_event = t1[event_mask]
            mod_event = mod_params[event_mask]
            mu_T_event = mu_T[event_mask]
            mu_C_event = mu_C[event_mask]
            beta_event = beta[event_mask]
            n_event = t1_event.size(0)
            
            tau = torch.rand(n_event, device=device)
            z0 = self.sample_min_gumbel((n_event, 1), device, mu_T_event, mu_C_event, beta_event)
            c_sample = self.sample_gumbel_truncated_above(t1_event, mu_C_event, beta_event)
            z1 = torch.cat([t1_event, c_sample], dim=-1)
            
            zt = (1 - tau.unsqueeze(-1)) * z0 + tau.unsqueeze(-1) * z1
            target_v = z1 - z0
            pred_v = self.vf_forward(tau, zt, mod_event)
            
            event_loss = F.mse_loss(pred_v, target_v)
            total_loss = total_loss + self.weight_event * event_loss
            loss_dict['event_loss'] = event_loss.item()
        
        if censored_mask.any():
            t_obs = t1[censored_mask]
            mod_cens = mod_params[censored_mask]
            mu_T_cens = mu_T[censored_mask]
            mu_C_cens = mu_C[censored_mask]
            beta_cens = beta[censored_mask]
            
            beta_rate = torch.clamp(1.0 / beta_cens, min=0.1, max=10.0)
            t_truncated = self._sample_truncated_exponential(t_obs, self.truncated_samples, rate=beta_rate)
            n_total = t_truncated.numel()
            t1_flat = t_truncated.view(-1, 1)
            
            tau = torch.rand(n_total, device=device)
            z0 = self.sample_min_gumbel(
                (n_total, 1), device,
                mu_T_cens.repeat_interleave(self.truncated_samples, dim=0),
                mu_C_cens.repeat_interleave(self.truncated_samples, dim=0),
                beta_cens.repeat_interleave(self.truncated_samples, dim=0)
            )
            mu_T_exp = mu_T_cens.repeat_interleave(self.truncated_samples, dim=0)
            beta_exp = beta_cens.repeat_interleave(self.truncated_samples, dim=0)
            t_sample = self.sample_gumbel_truncated_above(t1_flat, mu_T_exp, beta_exp)
            z1 = torch.cat([t_sample, t1_flat], dim=-1)
            
            zt = (1 - tau.unsqueeze(-1)) * z0 + tau.unsqueeze(-1) * z1
            target_v = z1 - z0
            mod_batched = mod_cens.repeat_interleave(self.truncated_samples, dim=0)
            pred_v = self.vf_forward(tau, zt, mod_batched)
            
            censored_loss = F.mse_loss(pred_v, target_v)
            total_loss = total_loss + self.weight_censored * censored_loss
            loss_dict['censored_loss'] = censored_loss.item()
        
        return total_loss, loss_dict
    
    def predict_time(self, features: torch.Tensor, mode: str = 'median', **kwargs) -> torch.Tensor:
        self.eval()
        with torch.no_grad():
            z = self.encoder(features)
            mod_params = self.film_head(z)
            mu_T, mu_C, beta = self.get_gumbel_params(features, z=z)
            B = features.size(0)
            device = features.device
            n_samples = self.mc_samples if self.use_mc else self.n_samples
            if mode == 'ode_step':
                n_samples = 1
            
            if mode == 'ode_step':
                log_log_2 = mu_T.new_tensor(self._LN_LN_2)
                t0_T = mu_T + beta * log_log_2
                t0_C = mu_C + beta * log_log_2
                z0 = torch.cat([t0_T, t0_C], dim=-1)
                z0 = torch.clamp(z0, -10.0, 10.0)
                mod_ext = mod_params
            else:
                z0 = self.sample_min_gumbel(
                    (B * n_samples, 1), device,
                    mu_T.repeat_interleave(n_samples, dim=0),
                    mu_C.repeat_interleave(n_samples, dim=0),
                    beta.repeat_interleave(n_samples, dim=0)
                )
                mod_ext = mod_params.repeat_interleave(n_samples, dim=0)
            
            tau_span = torch.linspace(0, 1, self.ode_steps, device=device)
            
            def func(tau, h):
                tau_batch = torch.full((h.size(0),), float(tau), device=device)
                v = self.vf_forward(tau_batch, h, mod_ext)
                v = torch.clamp(v, min=-100.0, max=100.0)
                return v
            
            traj = odeint_rk4(func, z0, tau_span) if self.solver == 'rk4' else odeint_euler(func, z0, tau_span)
            z1 = traj[-1]
            z1 = torch.clamp(z1, min=-20.0, max=20.0)
            
            t_norm = z1[:, 0:1].squeeze(-1)
            
            if mode == 'ode_step':
                return self._to_original_time(t_norm)
            
            t_raw = self._to_original_time(t_norm).view(B, n_samples)
            return t_raw.median(dim=1)[0] if mode == 'median' else t_raw.mean(dim=1)
    
    def _inverse_flow_with_integral_2d(self, z1: torch.Tensor, mod_params: torch.Tensor, 
                                       ode_steps: int = 100) -> Tuple[torch.Tensor, torch.Tensor]:
        tau_span = torch.linspace(1.0, 0.0, ode_steps, device=z1.device)
        dt = tau_span[1] - tau_span[0]
        
        curr_z = z1.clone()
        integral = torch.zeros(z1.size(0), 1, device=z1.device)
        
        for i in range(len(tau_span) - 1):
            tau = tau_span[i]
            z_in = curr_z.detach().requires_grad_(True)
            
            tau_batch = torch.full((z1.size(0),), float(tau), device=z1.device)
            v = self.vf_forward(tau_batch, z_in, mod_params)
            
            div_v = torch.zeros_like(curr_z[:, 0:1])
            for d in range(v.shape[1]):
                grad_d = torch.autograd.grad(
                    v[:, d], z_in,
                    grad_outputs=torch.ones_like(v[:, d]),
                    create_graph=False,
                    retain_graph=True,
                    allow_unused=True
                )[0]
                if grad_d is not None:
                    div_v = div_v + grad_d[:, d:d+1]
            
            v_val = torch.clamp(v.detach(), min=-50.0, max=50.0)
            div_v_val = torch.clamp(div_v.detach(), min=-50.0, max=50.0)
            
            curr_z = curr_z + v_val * dt
            curr_z = torch.clamp(curr_z, min=-15.0, max=15.0)
            
            integral = integral + div_v_val * dt
            integral = torch.clamp(integral, min=-100.0, max=100.0)
        
        return curr_z.detach(), integral
    
    def compute_log_density(self, features: torch.Tensor, time_grid: torch.Tensor,
                           ode_steps: int = 100, batch_size_limit: int = 50000) -> torch.Tensor:
        """
        计算边缘对数密度 log f_T(t_T|x)
        
        数学推导:
        - 先验：z0 ~ Min-Gumbel(μ_T, μ_C, β)
        - 边缘先验：z0_T ~ Gumbel(μ_T, β)
        - 流变换：t_T = Flow(z0_T; x)
        - 密度：log f_T(t_T|x) = log p_0(z0_T|x) - log|det J|
        """
        self.eval()
        device = features.device
        z = self.encoder(features)
        mod_params = self.film_head(z)
        mu_T, mu_C, beta = self.get_gumbel_params(features, z=z)
        time_grid = time_grid.to(device)
        num_times = time_grid.shape[0]
        
        if self.is_log_space:
            log_jacobians = -torch.log(self.time_scaler_std) - torch.log(
                torch.clamp(time_grid + 1.0, min=1e-2))
        else:
            log_jacobians = torch.full_like(
                time_grid, -torch.log(torch.tensor(self.time_scaler_std, device=device)))
        
        t_norm_grid = self._to_normalized_time(time_grid)
        all_log_densities = []
        N = features.size(0)
        chunk_size = max(1, batch_size_limit // N)
        
        for i in range(0, num_times, chunk_size):
            end_i = min(i + chunk_size, num_times)
            curr_norm = t_norm_grid[i:end_i]
            n_curr = end_i - i
            
            curr_mod = mod_params.unsqueeze(1).expand(-1, n_curr, -1).reshape(-1, mod_params.size(-1))
            curr_mu_T = mu_T.unsqueeze(1).expand(-1, n_curr, -1).reshape(-1, 1)
            curr_mu_C = mu_C.unsqueeze(1).expand(-1, n_curr, -1).reshape(-1, 1)
            curr_beta = beta.unsqueeze(1).expand(-1, n_curr, -1).reshape(-1, 1)
            
            t1_T = curr_norm.unsqueeze(0).expand(N, -1).reshape(-1, 1)
            
            with torch.enable_grad():
                c_median = self.gumbel_truncated_above_median(t1_T, curr_mu_C, curr_beta)
                z1 = torch.cat([t1_T, c_median], dim=-1)
                z0, integral = self._inverse_flow_with_integral_2d(z1, curr_mod, ode_steps)
                
                z0_T = torch.clamp(z0[:, 0:1], min=-15.0, max=15.0)
                
                std_T = torch.clamp((z0_T - curr_mu_T) / curr_beta, min=-20.0, max=10.0)
                log_p0_marginal = -torch.log(curr_beta) + std_T - torch.exp(std_T)
                log_p0_marginal = log_p0_marginal.squeeze(-1)
                
                log_f_norm = log_p0_marginal - integral.squeeze(-1)
                log_f_norm = torch.clamp(log_f_norm, min=-88.0, max=88.0)
            
            log_f_phys = log_f_norm.view(N, n_curr) + log_jacobians[i:end_i].unsqueeze(0)
            log_f_phys = torch.clamp(log_f_phys, min=-88.0, max=88.0)
            all_log_densities.append(log_f_phys)
        
        return torch.cat(all_log_densities, dim=1)
    
    def predict_survival_function(self, features: torch.Tensor, time_grid: torch.Tensor = None,
                                  ode_steps: int = 100, t_max: Optional[float] = None) -> torch.Tensor:
        self.eval()
        device = features.device
        if time_grid is None:
            if t_max is None:
                t_max = (self.time_scaler_mean + 3 * self.time_scaler_std).item()
                t_max = max(t_max, 10.0)
            time_grid = torch.linspace(0, t_max, 100, device=device)
        else:
            time_grid = time_grid.to(device)
        time_grid, _ = torch.sort(time_grid)
        
        if time_grid.numel() == 0 or time_grid[0] > 1e-6:
            full_grid = torch.cat([torch.zeros(1, device=device), time_grid])
            add_zero = True
        else:
            full_grid = time_grid
            add_zero = False
        
        log_f = self.compute_log_density(features, full_grid, ode_steps)
        dt = torch.diff(full_grid)
        dt = torch.clamp(dt, min=1e-10)
        
        log_areas = []
        for j in range(len(dt)):
            log_sum_f = torch.logsumexp(torch.stack([log_f[:, j], log_f[:, j+1]]), dim=0)
            log_area = log_sum_f + torch.log(dt[j]) - torch.log(torch.tensor(2.0, device=device))
            log_areas.append(log_area)
        
        log_areas = torch.stack(log_areas, dim=1)
        
        log_survival_list = []
        for i in range(log_areas.shape[1]):
            log_S_i = torch.logsumexp(log_areas[:, i:], dim=1, keepdim=True)
            log_survival_list.append(log_S_i)
        
        log_survival = torch.cat(log_survival_list, dim=1)
        log_S0 = torch.zeros((features.size(0), 1), device=device)
        log_survival_full = torch.cat([log_S0, log_survival], dim=1)
        
        s_full = torch.exp(torch.clamp(log_survival_full, min=-88, max=0))
        s_full = torch.clamp(s_full, min=0.0, max=1.0)
        
        return s_full[:, 1:] if add_zero else s_full

    def _forward_flow_samples_2d(self, z0: torch.Tensor, mod_params: torch.Tensor) -> torch.Tensor:
        """
        正向ODE: 从先验空间 z0 映射到生存时间空间 (二维版本)
        
        参数:
            z0: (B, 2) 先验采样 [z_T, z_C]
            mod_params: (B, vf_in_dim*2) FiLM调制参数
            
        返回:
            z1: (B, 2) 归一化生存时间 [t_T, t_C]
        """
        device = z0.device
        tau_span = torch.linspace(0, 1, self.ode_steps, device=device)
        
        def func(tau, h):
            tau_batch = torch.full((h.size(0),), float(tau), device=device)
            v = self.vf_forward(tau_batch, h, mod_params)
            v = torch.clamp(v, min=-100.0, max=100.0)
            return v
        
        traj = odeint_rk4(func, z0, tau_span) if self.solver == 'rk4' else odeint_euler(func, z0, tau_span)
        z1 = traj[-1]
        return torch.clamp(z1, min=-20.0, max=20.0)

    def predict_survival_function_mc(self, features: torch.Tensor, time_grid: torch.Tensor,
                                      n_samples: Optional[int] = None) -> torch.Tensor:
        """
        蒙特卡洛采样法计算生存函数 S(t|x) - Min-Gumbel先验版本
        
        数学原理:
            S_T(t|x) = P(T > t|x) = E_{z~MinGumbel}[1(Flow_T(z;x) > t)]
            
        参数:
            features: (N, D) 特征张量
            time_grid: (T,) 时间点网格
            n_samples: 采样数量
            
        返回:
            (N, T) 生存函数 S(t|x)
        """
        self.eval()
        device = features.device
        time_grid = time_grid.to(device)
        B = features.size(0)
        n_samples = n_samples or self.mc_samples
        
        with torch.no_grad():
            z = self.encoder(features)
            mod_params = self.film_head(z)
            mu_T, mu_C, beta = self.get_gumbel_params(features, z=z)
            
            mod_ext = mod_params.repeat_interleave(n_samples, dim=0)
            mu_T_ext = mu_T.repeat_interleave(n_samples, dim=0)
            mu_C_ext = mu_C.repeat_interleave(n_samples, dim=0)
            beta_ext = beta.repeat_interleave(n_samples, dim=0)
            
            z0 = self.sample_min_gumbel((B * n_samples,), device, mu_T_ext, mu_C_ext, beta_ext)
            
            z1 = self._forward_flow_samples_2d(z0, mod_ext)
            t_samples_norm = z1[:, 0]
            t_samples = self._to_original_time(t_samples_norm)
            t_samples = t_samples.view(B, n_samples)
            
            time_grid_exp = time_grid.unsqueeze(0).unsqueeze(2)
            t_samples_exp = t_samples.unsqueeze(1)
            S = (t_samples_exp > time_grid_exp).float().mean(dim=2)
            
        return S

    def predict_time_mc(self, features: torch.Tensor, n_samples: Optional[int] = None,
                        mode: str = 'median') -> torch.Tensor:
        """
        蒙特卡洛采样法计算中位生存时间 - Min-Gumbel先验版本
        
        参数:
            features: (N, D) 特征张量
            n_samples: 采样数量
            mode: 'median' 或 'mean'
            
        返回:
            (N,) 中位/平均生存时间
        """
        self.eval()
        device = features.device
        B = features.size(0)
        n_samples = n_samples or self.mc_samples
        
        with torch.no_grad():
            z = self.encoder(features)
            mod_params = self.film_head(z)
            mu_T, mu_C, beta = self.get_gumbel_params(features, z=z)
            
            mod_ext = mod_params.repeat_interleave(n_samples, dim=0)
            mu_T_ext = mu_T.repeat_interleave(n_samples, dim=0)
            mu_C_ext = mu_C.repeat_interleave(n_samples, dim=0)
            beta_ext = beta.repeat_interleave(n_samples, dim=0)
            
            z0 = self.sample_min_gumbel((B * n_samples,), device, mu_T_ext, mu_C_ext, beta_ext)
            
            z1 = self._forward_flow_samples_2d(z0, mod_ext)
            t_samples_norm = z1[:, 0]
            t_samples = self._to_original_time(t_samples_norm)
            t_samples = t_samples.view(B, n_samples)
            
            if mode == 'median':
                return t_samples.median(dim=1)[0]
            else:
                return t_samples.mean(dim=1)
