import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from models.flowmodel.base_flow import FlowSurv
from models.flowmodel.components import odeint_euler, odeint_rk4


class GumbelFlowSurv(FlowSurv):
    """
    Gumbel Flow 生存模型
    
    继承 FlowSurv，使用 Gumbel 分布作为先验
    """
    
    _LN_2 = 0.6931471805599453
    _LN_LN_2 = -0.3665129205816643

    def __init__(self, in_dim: int, config: Optional[dict] = None, **kwargs):
        super().__init__(in_dim, config, **kwargs)
        self._stage = 'gumbel'

        z_dim = self.encoder_dims[-1]
        gumbel_layers = []
        prev = z_dim
        for d in self.config.get('gumbel_head_hidden', [16, 8]):
            gumbel_layers.extend([nn.Linear(prev, d), nn.SiLU()])
            prev = d
        gumbel_layers.append(nn.Linear(prev, 2))
        self.gumbel_head = nn.Sequential(*gumbel_layers)
        
        self._set_stage('gumbel')

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

    def _gumbel_params(self, z: torch.Tensor):
        params = self.gumbel_head(z)
        alpha = torch.clamp(params[:, 0:1], -15.0, 15.0)
        log_beta = torch.clamp(params[:, 1:2], -10.0, 5.0)
        beta = torch.exp(log_beta)
        beta = torch.clamp(beta, min=1e-5, max=100.0)
        return alpha, beta

    def get_gumbel_params(self, x: torch.Tensor, z: Optional[torch.Tensor] = None):
        if z is None:
            z = self.encoder(x)
        z_gumbel = z.detach() if self._stage == 'flow' else z
        alpha, beta = self._gumbel_params(z_gumbel)
        return alpha, beta

    def init_gumbel_params(self, times: torch.Tensor, events: torch.Tensor):
        t_norm = self._to_normalized_time(times)
        
        event_mask = (events == 1)
        if event_mask.any():
            t_event = t_norm[event_mask]
            mu = t_event.mean().item()
            sigma = t_event.std().item()
        else:
            mu = t_norm.mean().item()
            sigma = t_norm.std().item()
        
        gamma = 0.57721
        sqrt_6 = math.sqrt(6.0)
        
        beta_init = max(sigma * sqrt_6 / math.pi, 0.1)
        alpha_init = mu - beta_init * gamma
        
        with torch.no_grad():
            self.gumbel_head[-1].weight.fill_(0)
            self.gumbel_head[-1].bias[0].fill_(alpha_init)
            self.gumbel_head[-1].bias[1].fill_(math.log(beta_init + 1e-6))

    def sample_prior(self, shape, device, alpha=None, beta=None):
        u = torch.rand(shape, device=device).clamp_(1e-10, 1.0 - 1e-10)
        log_neg_log_u = -torch.log(-torch.log(u))
        z_std = torch.clamp(log_neg_log_u, min=-15.0, max=15.0)
        
        if alpha is not None and beta is not None:
            z = alpha + beta * z_std
        else:
            z = z_std
        return torch.clamp(z, min=-20.0, max=20.0)

    def log_prob_prior(self, z, alpha, beta):
        beta = torch.clamp(beta, min=1e-5, max=100.0)
        log_beta = torch.log(beta)
        std = torch.clamp((z - alpha) / beta, min=-20.0, max=10.0)
        exp_std = torch.exp(std)
        exp_std = torch.clamp(exp_std, min=0.0, max=1e10)
        log_prob = -log_beta + std - exp_std
        return torch.clamp(log_prob, min=-100.0, max=88.0)

    def forward_loss(self, features: torch.Tensor, times_raw: torch.Tensor, events: torch.Tensor, **kwargs):
        device = features.device
        t1 = self._to_normalized_time(times_raw).float().unsqueeze(-1)
        if self._stage == 'gumbel':
            z = self.encoder(features)
            alpha, beta = self._gumbel_params(z)
            log_prob = self.log_prob_prior(t1, alpha, beta)
            std = torch.clamp((t1 - alpha) / beta, min=-20.0, max=10.0)
            log_surv = -torch.exp(std)
            log_lik = events * log_prob + (1 - events) * log_surv
            neg_ll = -log_lik.mean()
            return neg_ll, {"neg_loglik": neg_ll.item()}

        z = self.encoder(features)
        mod_params = self.film_head(z)
        alpha, beta = self.get_gumbel_params(features, z=z)

        event_mask = (events == 1)
        censored_mask = (events == 0)

        if not event_mask.any() and not censored_mask.any():
            return torch.tensor(0.0, device=device, requires_grad=True), {}

        total_loss = torch.zeros(1, device=device, requires_grad=True)
        loss_dict = {'event_loss': 0.0, 'censored_loss': 0.0}

        if event_mask.any():
            t1_event = t1[event_mask]
            mod_event = mod_params[event_mask]
            alpha_event = alpha[event_mask]
            beta_event = beta[event_mask]
            n_event = t1_event.size(0)
            tau = torch.rand(n_event, device=device)
            t0 = self.sample_prior((n_event, 1), device, alpha_event, beta_event)
            xt = (1 - tau.unsqueeze(-1)) * t0 + tau.unsqueeze(-1) * t1_event
            target_v = t1_event - t0
            pred_v = self.vf_forward(tau, xt, mod_event)
            event_loss = F.mse_loss(pred_v, target_v)
            total_loss = total_loss + self.weight_event * event_loss
            loss_dict['event_loss'] = event_loss.item()

        if censored_mask.any():
            t_obs = t1[censored_mask]
            mod_cens = mod_params[censored_mask]
            alpha_cens = alpha[censored_mask]
            beta_cens = beta[censored_mask]
            beta_rate = torch.clamp(1.0 / beta_cens, min=0.1, max=10.0)
            t_truncated = self._sample_truncated_exponential(t_obs, self.truncated_samples, rate=beta_rate)
            n_total = t_truncated.numel()
            t1_flat = t_truncated.view(-1, 1)
            tau = torch.rand(n_total, device=device)
            t0 = self.sample_prior((n_total, 1), device,
                                   alpha_cens.repeat_interleave(self.truncated_samples, dim=0),
                                   beta_cens.repeat_interleave(self.truncated_samples, dim=0))
            mod_batched = mod_cens.repeat_interleave(self.truncated_samples, dim=0)
            xt = (1 - tau.unsqueeze(-1)) * t0 + tau.unsqueeze(-1) * t1_flat
            target_v = t1_flat - t0
            pred_v = self.vf_forward(tau, xt, mod_batched)
            censored_loss = F.mse_loss(pred_v, target_v)
            total_loss = total_loss + self.weight_censored * censored_loss
            loss_dict['censored_loss'] = censored_loss.item()

        return total_loss, loss_dict

    def predict_time(self, features: torch.Tensor, mode: str = 'median', **kwargs) -> torch.Tensor:
        self.eval()
        with torch.no_grad():
            z = self.encoder(features)
            mod_params = self.film_head(z)
            alpha, beta = self.get_gumbel_params(features, z=z)
            B = features.size(0)
            device = features.device
            n_samples = self.mc_samples if self.use_mc else self.n_samples
            if mode == 'ode_step':
                n_samples = 1
            if mode == 'ode_step':
                log_log_2 = alpha.new_tensor(self._LN_LN_2)
                t0 = alpha + beta * log_log_2
                t0 = torch.clamp(t0, -10.0, 10.0)
                mod_ext = mod_params
            else:
                t0 = self.sample_prior((B * n_samples, 1), device,
                                       alpha.repeat_interleave(n_samples, dim=0),
                                       beta.repeat_interleave(n_samples, dim=0))
                mod_ext = mod_params.repeat_interleave(n_samples, dim=0)
            tau_span = torch.linspace(0, 1, self.ode_steps, device=device)

            def func(tau, h):
                tau_batch = torch.full((h.size(0),), float(tau), device=device)
                v = self.vf_forward(tau_batch, h, mod_ext)
                v = torch.clamp(v, min=-100.0, max=100.0)
                return v

            traj = odeint_rk4(func, t0, tau_span) if self.solver == 'rk4' else odeint_euler(func, t0, tau_span)
            t1 = traj[-1]
            t1 = torch.clamp(t1, min=-20.0, max=20.0)
            if mode == 'ode_step':
                return self._to_original_time(t1.squeeze(-1))
            t_raw = self._to_original_time(t1.squeeze(-1)).view(B, n_samples)
            return t_raw.median(dim=1)[0] if mode == 'median' else t_raw.mean(dim=1)

    def compute_log_density(self, features: torch.Tensor, time_grid: torch.Tensor,
                            ode_steps: int = 100, batch_size_limit: int = 50000) -> torch.Tensor:
        """
        计算对数密度函数 log f(t|x)
        使用 Gumbel 先验
        """
        self.eval()
        device = features.device
        z = self.encoder(features)
        mod_params = self.film_head(z)
        alpha, beta = self.get_gumbel_params(features, z=z)
        time_grid = time_grid.to(device)
        num_times = time_grid.shape[0]
        
        if self.is_log_space:
            log_jacobians = -torch.log(self.time_scaler_std) - torch.log(torch.clamp(time_grid + 1.0, min=1e-2))
        else:
            log_jacobians = torch.full_like(time_grid, -torch.log(torch.tensor(self.time_scaler_std, device=device)))
        
        t_norm_grid = self._to_normalized_time(time_grid)
        all_log_densities = []
        N = features.size(0)
        chunk_size = max(1, batch_size_limit // N)
        
        for i in range(0, num_times, chunk_size):
            end_i = min(i + chunk_size, num_times)
            curr_norm = t_norm_grid[i:end_i]
            n_curr = end_i - i
            curr_mod = mod_params.unsqueeze(1).expand(-1, n_curr, -1).reshape(-1, mod_params.size(-1))
            curr_alpha = alpha.unsqueeze(1).expand(-1, n_curr, -1).reshape(-1, 1)
            curr_beta = beta.unsqueeze(1).expand(-1, n_curr, -1).reshape(-1, 1)
            t1 = curr_norm.unsqueeze(0).expand(N, -1).reshape(-1, 1)
            
            with torch.enable_grad():
                t0, integral = self._inverse_flow_with_integral(t1, curr_mod, ode_steps)
                t0 = torch.clamp(t0, min=-15.0, max=15.0)
                log_p0 = self.log_prob_prior(t0, curr_alpha, curr_beta).squeeze(-1)
                log_f_norm = log_p0 - integral.squeeze(-1)
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

    def predict_survival_function_mc(self, features: torch.Tensor, time_grid: torch.Tensor,
                                      n_samples: Optional[int] = None) -> torch.Tensor:
        """
        蒙特卡洛采样法计算生存函数 S(t|x) - Gumbel先验版本
        
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
            alpha, beta = self.get_gumbel_params(features, z=z)
            
            mod_ext = mod_params.repeat_interleave(n_samples, dim=0)
            alpha_ext = alpha.repeat_interleave(n_samples, dim=0)
            beta_ext = beta.repeat_interleave(n_samples, dim=0)
            
            z0 = self.sample_prior((B * n_samples, 1), device, alpha_ext, beta_ext)
            
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
        蒙特卡洛采样法计算中位生存时间 - Gumbel先验版本
        
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
            alpha, beta = self.get_gumbel_params(features, z=z)
            
            mod_ext = mod_params.repeat_interleave(n_samples, dim=0)
            alpha_ext = alpha.repeat_interleave(n_samples, dim=0)
            beta_ext = beta.repeat_interleave(n_samples, dim=0)
            
            z0 = self.sample_prior((B * n_samples, 1), device, alpha_ext, beta_ext)
            
            t_samples_norm = self._forward_flow_samples(z0, mod_ext)
            t_samples = self._to_original_time(t_samples_norm.squeeze(-1))
            t_samples = t_samples.view(B, n_samples)
            
            if mode == 'median':
                return t_samples.median(dim=1)[0]
            else:
                return t_samples.mean(dim=1)
