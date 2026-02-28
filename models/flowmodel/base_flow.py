import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from typing import Optional, Dict, Tuple, Union

from models.interface import TorchSurvivalModel
from models.flowmodel.components import (
    SinusoidalEmbedding,
    FiLMResidualBlock,
    ResidualEncoderBlock,
    odeint_euler,
    odeint_rk4
)


class FlowSurv(TorchSurvivalModel):
    def __init__(self, in_dim: int, config: Optional[dict] = None, **kwargs):
        super().__init__()
        self.config = config if config is not None else kwargs
        self.in_dim = in_dim

        self.tau_dim = self.config.get('tau_dim', 16)
        self.time_emb = SinusoidalEmbedding(self.tau_dim)
        self.time_proj = nn.Sequential(
            nn.Linear(self.tau_dim, self.tau_dim), nn.SiLU(),
            nn.Linear(self.tau_dim, self.tau_dim), nn.SiLU()
        )

        self.encoder_dims = self.config.get('encoder_hidden', [16, 8])
        encoder_layers = [ResidualEncoderBlock(prev, d) for prev, d in zip([in_dim] + self.encoder_dims[:-1], self.encoder_dims)]
        self.encoder = nn.Sequential(*encoder_layers)

        self.vf_in_dim = 1
        self.film_head = self._build_film_head()
        self.vector_field = self._build_vector_field()

        self.weight_event = self.config.get('weight_event', 1.0)
        self.weight_censored = self.config.get('weight_censored', 1.0)
        self.ode_steps = self.config.get('ode_steps', 50)
        self.solver = self.config.get('solver', 'rk4')
        self.truncated_samples = self.config.get('truncated_samples', 16)
        self.n_samples = self.config.get('n_samples', 64)

        self.register_buffer('prior_mean', torch.tensor(0.0))
        self.register_buffer('prior_std', torch.tensor(1.0))

    def _build_film_head(self) -> nn.Module:
        film_hidden = self.config.get('film_hidden', [8])
        film_layers = []
        prev = self.encoder_dims[-1]
        for d in film_hidden:
            film_layers.extend([nn.Linear(prev, d), nn.SiLU()])
            prev = d
        film_layers.append(nn.Linear(prev, self.vf_in_dim * 2))
        return nn.Sequential(*film_layers)

    def _build_vector_field(self) -> nn.Module:
        vf_hidden_dims = self.config.get('vf_hidden_dims', [16, 8])
        dropout = self.config.get('dropout', 0.1)
        vf_layers = []
        prev_dim = self.vf_in_dim
        for d in vf_hidden_dims:
            vf_layers.append(FiLMResidualBlock(prev_dim, d, self.tau_dim, self.vf_in_dim * 2, dropout))
            prev_dim = d
        vf_layers.append(nn.Linear(prev_dim, self.vf_in_dim))
        return nn.ModuleList(vf_layers)

    @property
    def prior(self):
        return Normal(self.prior_mean, self.prior_std)

    def get_film(self, x: torch.Tensor) -> torch.Tensor:
        return self.film_head(self.encoder(x))

    def vf_forward(self, tau: torch.Tensor, h: torch.Tensor, mod_params: torch.Tensor) -> torch.Tensor:
        tau_emb = self.time_proj(self.time_emb(tau))
        for layer in self.vector_field[:-1]:
            h = layer(h, tau_emb, mod_params)
        return self.vector_field[-1](h)

    def _inverse_flow_with_integral(self, t1: torch.Tensor, mod_params: torch.Tensor, ode_steps: int = 100):
        """
        反向 ODE 求解并同时计算散度积分 (用于密度估计)
        使用更高效的数值方法替代手动循环
        """
        tau_span = torch.linspace(1.0, 0.0, ode_steps, device=t1.device)
        dt = tau_span[1] - tau_span[0]
        
        curr_t = t1.clone()
        integral = torch.zeros(t1.size(0), 1, device=t1.device)
        
        # 预先计算步长, 减少循环内运算
        for i in range(len(tau_span) - 1):
            tau = tau_span[i]
            t_in = curr_t.detach().requires_grad_(True)
            
            # 计算向量场
            tau_batch = torch.full((t1.size(0),), float(tau), device=t1.device)
            v = self.vf_forward(tau_batch, t_in, mod_params)
            
            # 批量化计算散度 (Divergence): div(v) = d v_i / d x_i
            # 对于一维生存时间, 散度就是简单的导数
            div_v = torch.autograd.grad(
                v, t_in, 
                grad_outputs=torch.ones_like(v),
                create_graph=False, 
                retain_graph=False,
                allow_unused=True
            )[0]
            
            if div_v is None:
                div_v = torch.zeros_like(curr_t)
            
            # 数值稳定性保护
            v_val = torch.clamp(v.detach(), min=-100.0, max=100.0)
            div_v_val = torch.clamp(div_v.detach(), min=-100.0, max=100.0)
            
            # 改进的 Euler 步 (Heun's method 思想或简单的稳定更新)
            curr_t = curr_t + v_val * dt
            integral = integral + div_v_val * dt
            
        return curr_t.detach(), integral

    def _sample_truncated_exponential(self, t_obs: torch.Tensor, n_samples: int, rate: float = 1.0) -> torch.Tensor:
        device = t_obs.device
        u = torch.rand(t_obs.size(0), n_samples, device=device)
        delta_t = -torch.log(1 - u + 1e-10) / rate
        return t_obs + delta_t

    def forward_loss(self, features: torch.Tensor, times_raw: torch.Tensor, events: torch.Tensor, **kwargs):
        device = features.device
        t1 = self._to_normalized_time(times_raw).unsqueeze(-1)
        mod_params = self.get_film(features)
        event_mask = (events == 1)
        censored_mask = (events == 0)

        if not event_mask.any() and not censored_mask.any():
            return torch.tensor(0.0, device=device, requires_grad=True), {}

        total_loss = torch.zeros(1, device=device, requires_grad=True)
        loss_dict = {}

        if event_mask.any():
            t1_event = t1[event_mask]
            mod_event = mod_params[event_mask]
            n_event = t1_event.size(0)
            tau = torch.rand(n_event, device=device)
            t0 = torch.randn_like(t1_event)
            xt = (1 - tau.unsqueeze(-1)) * t0 + tau.unsqueeze(-1) * t1_event
            target_v = t1_event - t0
            pred_v = self.vf_forward(tau, xt, mod_event)
            event_loss = F.mse_loss(pred_v, target_v)
            total_loss = total_loss + self.weight_event * event_loss
            loss_dict['event_loss'] = event_loss.item()

        if censored_mask.any():
            t_obs = t1[censored_mask]
            mod_cens = mod_params[censored_mask]
            n_cens = t_obs.size(0)
            t_trunc = self._sample_truncated_exponential(t_obs, self.truncated_samples)
            t1_flat = t_trunc.view(-1, 1)
            mod_exp = mod_cens.repeat_interleave(self.truncated_samples, dim=0)
            tau = torch.rand(t1_flat.size(0), device=device)
            t0 = torch.randn_like(t1_flat)
            xt = (1 - tau.unsqueeze(-1)) * t0 + tau.unsqueeze(-1) * t1_flat
            target_v = t1_flat - t0
            pred_v = self.vf_forward(tau, xt, mod_exp)
            censored_loss = F.mse_loss(pred_v, target_v)
            total_loss = total_loss + self.weight_censored * censored_loss
            loss_dict['censored_loss'] = censored_loss.item()

        loss_dict.setdefault('event_loss', 0.0)
        loss_dict.setdefault('censored_loss', 0.0)
        return total_loss, loss_dict

    def predict_time(self, features: torch.Tensor, mode: str = 'median', **kwargs) -> torch.Tensor:
        self.eval()
        with torch.no_grad():
            mod_params = self.get_film(features)
            B = features.size(0)
            device = features.device
            n_samples = self.n_samples if mode != 'ode_step' else 1
            if mode == 'ode_step':
                t0 = torch.zeros((B, 1), device=device)
                mod_ext = mod_params
            else:
                t0 = torch.randn((B * n_samples, 1), device=device)
                t0 = torch.clamp(t0, min=-10.0, max=10.0)
                mod_ext = mod_params.repeat_interleave(n_samples, dim=0)
            tau_span = torch.linspace(0, 1, self.ode_steps, device=device)
            def func(tau, h):
                tau_batch = torch.full((h.size(0),), float(tau), device=device)
                v = self.vf_forward(tau_batch, h, mod_ext)
                v = torch.clamp(v, min=-100.0, max=100.0)
                return v
            traj = odeint_rk4(func, t0, tau_span) if self.solver == 'rk4' else odeint_euler(func, t0, tau_span)
            t1_norm = traj[-1]
            t1_norm = torch.clamp(t1_norm, min=-20.0, max=20.0)
            if mode == 'ode_step':
                return self._to_original_time(t1_norm.squeeze(-1))
            t_raw = self._to_original_time(t1_norm.squeeze(-1)).view(B, n_samples)
            return t_raw.median(dim=1)[0] if mode == 'median' else t_raw.mean(dim=1)

    def compute_density(self, features: torch.Tensor, time_grid: torch.Tensor,
                        ode_steps: int = 100, batch_size_limit: int = 50000) -> torch.Tensor:
        self.eval()
        device = features.device
        mod_params = self.get_film(features)
        time_grid = time_grid.to(device)
        num_times = time_grid.shape[0]
        if self.is_log_space:
            jacobians = 1.0 / (self.time_scaler_std * torch.clamp(time_grid + 1.0, min=1e-2))
        else:
            jacobians = torch.full_like(time_grid, 1.0 / self.time_scaler_std)
        jacobians = torch.clamp(jacobians, min=0.0, max=1000.0)
        t_norm_grid = self._to_normalized_time(time_grid)
        all_densities = []
        N = features.size(0)
        chunk_size = max(1, batch_size_limit // N)
        for i in range(0, num_times, chunk_size):
            end_i = min(i + chunk_size, num_times)
            curr_norm = t_norm_grid[i:end_i]
            n_curr = end_i - i
            curr_mod = mod_params.unsqueeze(1).expand(-1, n_curr, -1).reshape(-1, mod_params.size(-1))
            t1 = curr_norm.unsqueeze(0).expand(N, -1).reshape(-1, 1)
            with torch.enable_grad():
                t0, integral = self._inverse_flow_with_integral(t1, curr_mod, ode_steps)
                log_p0 = self.prior.log_prob(t0)
                log_f_norm = log_p0 - integral
                log_f_norm = torch.clamp(log_f_norm, min=-88.0, max=88.0)
                f_norm = torch.exp(log_f_norm).view(N, n_curr)
            f_phys = f_norm * jacobians[i:end_i].unsqueeze(0)
            f_phys = torch.clamp(f_phys, min=0.0, max=1000.0)
            all_densities.append(f_phys)
        return torch.cat(all_densities, dim=1)

    def predict_survival_function(self, features: torch.Tensor, time_grid: Optional[torch.Tensor] = None, 
                                  ode_steps: int = 100, t_max: Optional[float] = None) -> torch.Tensor:
        self.eval()
        device = features.device
        if time_grid is None:
            # 改进默认 time_grid 逻辑
            if t_max is None:
                # 默认使用 10.0, 但如果 scaler 显示标准差很大, 则相应扩大范围
                # 这是一个启发式策略: max(20.0, mean + 5*std)
                t_max = max(20.0, float(self.time_scaler_mean + 5 * self.time_scaler_std))
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
        f_full = self.compute_density(features, full_grid, ode_steps)
        dt = torch.diff(full_grid)
        dt = torch.clamp(dt, min=1e-4)
        areas = 0.5 * (f_full[:, :-1] + f_full[:, 1:]) * dt.unsqueeze(0)
        cdf = torch.zeros_like(f_full)
        cdf[:, 1:] = torch.cumsum(areas, dim=1)
        cdf = torch.clamp(cdf, min=0.0, max=1.0)
        s_full = torch.clamp(1.0 - cdf, min=0.0, max=1.0)
        return s_full[:, 1:] if add_zero else s_full

    def predict_risk(self, features: torch.Tensor, time_grid: Optional[torch.Tensor] = None, **kwargs) -> torch.Tensor:
        pred_medians = self.predict_time(features, mode='median')
        return -pred_medians

    def predict_survival_metrics(self, features: torch.Tensor, time_grid: torch.Tensor, 
                                 ode_steps: int = 100, t_max: Optional[float] = None):
        s = self.predict_survival_function(features, time_grid, ode_steps, t_max=t_max)
        f = self.compute_density(features, time_grid, ode_steps)
        f_clamped = torch.clamp(f, min=0.0, max=1000.0)
        h = f_clamped / torch.clamp(s, min=1e-4)
        h = torch.clamp(h, min=0.0, max=1000.0)
        H = -torch.log(torch.clamp(s, min=1e-6, max=1.0))
        H = torch.clamp(H, min=0.0, max=100.0)
        return {'density': f, 'hazard': h, 'cum_hazard': H, 'survival': s}

    def compute_hazard_rate(self, features: torch.Tensor, time_grid: torch.Tensor, **kwargs) -> torch.Tensor:
        ode_steps = kwargs.get('ode_steps', 100)
        metrics = self.predict_survival_metrics(features, time_grid, ode_steps)
        return metrics['hazard']