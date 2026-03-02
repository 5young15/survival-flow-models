import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from typing import Optional, Dict, Tuple, Union
import numpy as np

from models.interface import TorchSurvivalModel
from models.flowmodel.components import (
    SinusoidalEmbedding,
    FiLMResidualBlock,
    ResidualEncoderBlock,
    odeint_euler,
    odeint_rk4
)


class FlowSurv(TorchSurvivalModel):
    """
    Flow-based 生存模型
    
    统一接口:
    - compute_log_density: 返回 log f(t|x) (密度积分法)
    - compute_hazard_rate: 返回 log h(t|x)
    - predict_survival_function: 返回 S(t|x)
    
    计算方法选择 (通过 use_mc 参数):
    - use_mc=False (默认): 密度积分法，通过逆流积分计算Jacobian
    - use_mc=True: 蒙特卡洛采样法，通过正向ODE采样估计S(t)
    """
    
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
        self.mc_samples = self.config.get('mc_samples', 1000)
        self.use_mc = self.config.get('use_mc', False)

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
        tau_span = torch.linspace(1.0, 0.0, ode_steps, device=t1.device)
        dt = tau_span[1] - tau_span[0]
        
        curr_t = t1.clone()
        integral = torch.zeros(t1.size(0), 1, device=t1.device)
        
        for i in range(len(tau_span) - 1):
            tau = tau_span[i]
            t_in = curr_t.detach().requires_grad_(True)
            
            tau_batch = torch.full((t1.size(0),), float(tau), device=t1.device)
            v = self.vf_forward(tau_batch, t_in, mod_params)
            
            div_v = torch.autograd.grad(
                v, t_in, 
                grad_outputs=torch.ones_like(v),
                create_graph=False, 
                retain_graph=False,
                allow_unused=True
            )[0]
            
            if div_v is None:
                div_v = torch.zeros_like(curr_t)
            
            v_val = torch.clamp(v.detach(), min=-50.0, max=50.0)
            div_v_val = torch.clamp(div_v.detach(), min=-50.0, max=50.0)
            
            curr_t = curr_t + v_val * dt
            curr_t = torch.clamp(curr_t, min=-15.0, max=15.0)
            
            integral = integral + div_v_val * dt
            integral = torch.clamp(integral, min=-100.0, max=100.0)
            
        return curr_t.detach(), integral

    def _sample_truncated_exponential(self, t_obs: torch.Tensor, n_samples: int, rate: Union[float, torch.Tensor] = 1.0) -> torch.Tensor:
        device = t_obs.device
        u = torch.rand(t_obs.size(0), n_samples, device=device)
        if isinstance(rate, torch.Tensor):
            rate_expanded = rate.view(-1, 1).expand(-1, n_samples)
            delta_t = -torch.log(1 - u + 1e-10) / rate_expanded
        else:
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
            n_samples = self.mc_samples if self.use_mc else self.n_samples
            if mode == 'ode_step':
                n_samples = 1
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

    def compute_log_density(self, features: torch.Tensor, time_grid: torch.Tensor,
                            ode_steps: int = 100, batch_size_limit: int = 50000) -> torch.Tensor:
        """
        计算对数密度函数 log f(t|x)
        """
        self.eval()
        device = features.device
        mod_params = self.get_film(features)
        time_grid = time_grid.to(device)
        num_times = time_grid.shape[0]
        
        if self.is_log_space:
            log_jacobians = -torch.log(self.time_scaler_std) - torch.log(torch.clamp(time_grid + 1.0, min=1e-2))
        else:
            log_jacobians = torch.full_like(time_grid, -torch.log(self.time_scaler_std).item())

        t_norm_grid = self._to_normalized_time(time_grid)
        all_log_densities = []
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
                log_f_phys = log_f_norm.view(N, n_curr) + log_jacobians[i:end_i].unsqueeze(0)
            
            all_log_densities.append(log_f_phys)
        
        return torch.cat(all_log_densities, dim=1)

    def predict_survival_function(self, features: torch.Tensor, time_grid: Optional[torch.Tensor] = None, 
                                  ode_steps: int = 100, t_max: Optional[float] = None) -> torch.Tensor:
        if self.use_mc:
            if time_grid is None:
                if t_max is None:
                    t_max = max(20.0, float(self.time_scaler_mean + 5 * self.time_scaler_std))
                device = features.device
                time_grid = torch.linspace(0, t_max, 100, device=device)
            return self.predict_survival_function_mc(features, time_grid, self.mc_samples)
        
        self.eval()
        device = features.device
        if time_grid is None:
            if t_max is None:
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
        
        log_f = self.compute_log_density(features, full_grid, ode_steps)
        dt = torch.diff(full_grid)
        dt = torch.clamp(dt, min=1e-10)
        
        log_areas = []
        for j in range(len(dt)):
            log_sum_f = torch.logsumexp(torch.stack([log_f[:, j], log_f[:, j+1]]), dim=0)
            log_area = log_sum_f + torch.log(dt[j]) - torch.log(torch.tensor(2.0, device=device))
            log_areas.append(log_area)
        
        log_areas = torch.stack(log_areas, dim=1)
        
        log_areas_reversed = torch.flip(log_areas, dims=[1])
        log_survival_rev = torch.zeros_like(log_areas_reversed)
        log_survival_rev[:, 0] = log_areas_reversed[:, 0]
        for i in range(1, log_areas_reversed.shape[1]):
            log_survival_rev[:, i] = torch.logsumexp(
                torch.stack([log_survival_rev[:, i-1], log_areas_reversed[:, i]]), dim=0)
        log_survival = torch.flip(log_survival_rev, dims=[1])
        log_S0 = torch.zeros((features.size(0), 1), device=device)
        log_survival_full = torch.cat([log_S0, log_survival], dim=1)
        
        s_full = torch.exp(torch.clamp(log_survival_full, min=-88, max=0))
        s_full = torch.clamp(s_full, min=0.0, max=1.0)
        
        return s_full[:, 1:] if add_zero else s_full

    def predict_risk(self, features: torch.Tensor, _time_grid: Optional[torch.Tensor] = None, **_kwargs) -> torch.Tensor:
        """
        预测风险分数（用于 C-index）
        
        参数:
            features: (N, D) 特征张量
            _time_grid: 未使用，保留用于接口兼容性
            **_kwargs: 其他未使用参数
            
        返回:
            (N,) 风险分数（负的中位生存时间）
        """
        pred_medians = self.predict_time(features, mode='median')
        pred_medians = torch.nan_to_num(pred_medians, nan=0.0, posinf=20.0, neginf=-20.0)
        return -pred_medians.float()

    def predict_survival_metrics(self, features: torch.Tensor, time_grid: torch.Tensor, 
                                 ode_steps: int = 100, t_max: Optional[float] = None):
        if self.use_mc:
            return self.predict_survival_metrics_mc(features, time_grid, self.mc_samples)
        
        log_f = self.compute_log_density(features, time_grid, ode_steps)
        s = self.predict_survival_function(features, time_grid, ode_steps, t_max=t_max)
        log_s = torch.log(torch.clamp(s, min=1e-100, max=1.0))
        log_h = log_f - log_s
        log_h = torch.clamp(log_h, min=-20, max=20)
        H = -log_s
        H = torch.clamp(H, min=0.0, max=100.0)
        
        return {
            'log_density': log_f, 
            'log_hazard': log_h, 
            'cum_hazard': H, 
            'survival': s,
            'log_survival': log_s
        }

    def compute_hazard_rate(self, features: torch.Tensor, time_grid: torch.Tensor, **_kwargs) -> torch.Tensor:
        """
        计算对数风险函数 log h(t|X)
        
        注意: 返回值是对数空间的风险函数，而非原始风险值。
        这是为了与 hazard_mse/hazard_mae 等指标函数保持一致（它们期望对数空间输入）。
        
        参数:
            features: (N, D) 特征张量
            time_grid: (T,) 时间点网格
            
        返回:
            (N, T) 对数风险函数 log h(t|X)
        """
        if self.use_mc:
            return self.compute_hazard_rate_mc(features, time_grid, self.mc_samples)
        
        ode_steps = _kwargs.get('ode_steps', 100)
        metrics = self.predict_survival_metrics(features, time_grid, ode_steps)
        return metrics['log_hazard']

    def _forward_flow_samples(self, z0: torch.Tensor, mod_params: torch.Tensor) -> torch.Tensor:
        """
        正向ODE: 从先验空间 z0 映射到生存时间空间
        
        参数:
            z0: (B, 1) 先验采样
            mod_params: (B, vf_in_dim*2) FiLM调制参数
            
        返回:
            t_norm: (B, 1) 归一化生存时间
        """
        device = z0.device
        tau_span = torch.linspace(0, 1, self.ode_steps, device=device)
        
        def func(tau, h):
            tau_batch = torch.full((h.size(0),), float(tau), device=device)
            v = self.vf_forward(tau_batch, h, mod_params)
            v = torch.clamp(v, min=-100.0, max=100.0)
            return v
        
        traj = odeint_rk4(func, z0, tau_span) if self.solver == 'rk4' else odeint_euler(func, z0, tau_span)
        t_norm = traj[-1]
        return torch.clamp(t_norm, min=-20.0, max=20.0)

    def sample_prior(self, shape: tuple, device: torch.device) -> torch.Tensor:
        """
        从标准正态先验采样
        
        子类可覆盖此方法以使用不同先验（如Gumbel）
        """
        return torch.randn(shape, device=device)

    def predict_survival_function_mc(self, features: torch.Tensor, time_grid: torch.Tensor,
                                      n_samples: Optional[int] = None) -> torch.Tensor:
        """
        蒙特卡洛采样法计算生存函数 S(t|x)
        
        数学原理:
            S(t|x) = P(T > t|x) = E_{z~p(z)}[1(Flow(z;x) > t)]
            
        通过采样估计:
            S(t|x) ≈ (1/N) * sum_{i=1}^N 1(t_i > t)
            
        参数:
            features: (N, D) 特征张量
            time_grid: (T,) 时间点网格
            n_samples: 采样数量，默认使用 self.mc_samples
            
        返回:
            (N, T) 生存函数 S(t|x)
        """
        self.eval()
        device = features.device
        time_grid = time_grid.to(device)
        B = features.size(0)
        n_samples = n_samples or self.mc_samples
        
        with torch.no_grad():
            mod_params = self.get_film(features)
            mod_ext = mod_params.repeat_interleave(n_samples, dim=0)
            
            z0 = self.sample_prior((B * n_samples, self.vf_in_dim), device)
            z0 = torch.clamp(z0, min=-10.0, max=10.0)
            
            t_samples_norm = self._forward_flow_samples(z0, mod_ext)
            t_samples = self._to_original_time(t_samples_norm.squeeze(-1))
            t_samples = t_samples.view(B, n_samples)
            
            time_grid_exp = time_grid.unsqueeze(0).unsqueeze(2)
            t_samples_exp = t_samples.unsqueeze(1)
            S = (t_samples_exp > time_grid_exp).float().mean(dim=2)
            
        return S

    def predict_time_mc(self, features: torch.Tensor, n_samples: Optional[int] = None,
                        mode: str = 'median') -> torch.Tensor:
        """
        蒙特卡洛采样法计算中位生存时间
        
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
            mod_params = self.get_film(features)
            mod_ext = mod_params.repeat_interleave(n_samples, dim=0)
            
            z0 = self.sample_prior((B * n_samples, self.vf_in_dim), device)
            z0 = torch.clamp(z0, min=-10.0, max=10.0)
            
            t_samples_norm = self._forward_flow_samples(z0, mod_ext)
            t_samples = self._to_original_time(t_samples_norm.squeeze(-1))
            t_samples = t_samples.view(B, n_samples)
            
            if mode == 'median':
                return t_samples.median(dim=1)[0]
            else:
                return t_samples.mean(dim=1)

    def predict_risk_mc(self, features: torch.Tensor, n_samples: Optional[int] = None) -> torch.Tensor:
        """
        蒙特卡洛采样法计算风险分数
        
        参数:
            features: (N, D) 特征张量
            n_samples: 采样数量
            
        返回:
            (N,) 风险分数（负的中位生存时间）
        """
        pred_medians = self.predict_time_mc(features, n_samples, mode='median')
        pred_medians = torch.nan_to_num(pred_medians, nan=0.0, posinf=20.0, neginf=-20.0)
        return -pred_medians.float()

    def compute_hazard_rate_mc(self, features: torch.Tensor, time_grid: torch.Tensor,
                                n_samples: Optional[int] = None) -> torch.Tensor:
        """
        蒙特卡洛采样法计算对数风险函数 log h(t|x)
        
        改进方法:
        1. 使用累积风险函数 H(t) = -log(S(t))
        2. 使用 Savitzky-Golay 滤波器平滑 H(t)，减少阶梯函数噪声
        3. 使用中心差分计算导数，提高数值稳定性
        
        数学推导:
            h(t) = dH(t)/dt, 其中 H(t) = -log(S(t))
            log h(t) = log(dH/dt)
            
        参数:
            features: (N, D) 特征张量
            time_grid: (T,) 时间点网格
            n_samples: 采样数量
            
        返回:
            (N, T) 对数风险函数 log h(t|x)
        """
        S = self.predict_survival_function_mc(features, time_grid, n_samples)
        
        if len(time_grid) < 5:
            return torch.full((features.size(0), len(time_grid)), float('nan'), device=features.device)
        
        eps = 1e-100
        log_S = torch.log(torch.clamp(S, min=eps))
        H = -log_S
        
        H_np = H.cpu().numpy()
        t_np = time_grid.cpu().numpy()
        
        try:
            from scipy.signal import savgol_filter
            window_length = min(11, len(t_np) - 1)
            if window_length % 2 == 0:
                window_length -= 1
            window_length = max(5, window_length)
            polyorder = min(3, window_length - 1)
            use_savgol = True
        except ImportError:
            use_savgol = False
        
        h_list = []
        for i in range(H_np.shape[0]):
            H_i = H_np[i]
            
            if use_savgol:
                H_smooth = savgol_filter(H_i, window_length, polyorder)
                dH_dt = np.gradient(H_smooth, t_np)
            else:
                dH_dt = np.gradient(H_i, t_np)
            
            dH_dt = np.clip(dH_dt, 1e-10, 1000.0)
            h_list.append(dH_dt)
        
        h = torch.from_numpy(np.array(h_list)).float().to(features.device)
        log_h = torch.log(torch.clamp(h, min=eps))
        log_h = torch.clamp(log_h, min=-20.0, max=20.0)
        
        return log_h

    def predict_survival_metrics_mc(self, features: torch.Tensor, time_grid: torch.Tensor,
                                     n_samples: Optional[int] = None) -> Dict[str, torch.Tensor]:
        """
        蒙特卡洛采样法计算所有生存指标
        
        参数:
            features: (N, D) 特征张量
            time_grid: (T,) 时间点网格
            n_samples: 采样数量
            
        返回:
            dict: {
                'survival': S(t),
                'log_survival': log S(t),
                'log_hazard': log h(t),
                'cum_hazard': H(t)
            }
        """
        S = self.predict_survival_function_mc(features, time_grid, n_samples)
        log_S = torch.log(torch.clamp(S, min=1e-100, max=1.0))
        log_h = self.compute_hazard_rate_mc(features, time_grid, n_samples)
        H = -log_S
        H = torch.clamp(H, min=0.0, max=100.0)
        
        return {
            'survival': S,
            'log_survival': log_S,
            'log_hazard': log_h,
            'cum_hazard': H
        }
