import torch
import torch.nn as nn
import torch.nn.functional as F
import math  # 用于init中的标量log
from models.flowmodel.base_flow import FlowSurv
from models.flowmodel.components import FiLMResidualBlock, odeint_euler, odeint_rk4
from typing import Union, Optional


class GumbelFlowSurv(FlowSurv):
    def __init__(self, in_dim: int, config: Optional[dict] = None, **kwargs):
        super().__init__(in_dim, config, **kwargs)
        self.weight_gumbel = self.config.get('weight_gumbel', 0.1)
        self.truncated_samples = self.config.get('truncated_samples', 32)

        z_dim = self.encoder_dims[-1]
        # film_head (复用)
        film_layers = []
        prev = z_dim
        for d in self.config.get('film_hidden', [8]):
            film_layers.extend([nn.Linear(prev, d), nn.SiLU()])
            prev = d
        film_layers.append(nn.Linear(prev, self.vf_in_dim * 2))
        self.film_head = nn.Sequential(*film_layers)

        # alpha / log_beta head
        alpha_layers = []
        prev = z_dim
        for d in self.config.get('gumbel_alpha_head_hidden', [16, 8]):
            alpha_layers.extend([nn.Linear(prev, d), nn.SiLU()])
            prev = d
        alpha_layers.append(nn.Linear(prev, 1))
        self.alpha_head = nn.Sequential(*alpha_layers)

        beta_layers = []
        prev = z_dim
        for d in self.config.get('gumbel_beta_head_hidden', [16, 8]):
            beta_layers.extend([nn.Linear(prev, d), nn.SiLU()])
            prev = d
        beta_layers.append(nn.Linear(prev, 1))
        self.log_beta_head = nn.Sequential(*beta_layers)

        # vector_field (复用父类结构)
        vf_layers = []
        prev_dim = self.vf_in_dim
        for d in self.vf_hidden_dims:
            vf_layers.append(FiLMResidualBlock(prev_dim, d, self.tau_dim, self.vf_in_dim * 2, self.dropout))
            prev_dim = d
        vf_layers.append(nn.Linear(prev_dim, self.vf_in_dim))
        self.vector_field = nn.ModuleList(vf_layers)

        self._gumbel_initialized = False

    def init_gumbel_params(self, times: torch.Tensor, events: Optional[torch.Tensor] = None, robust_scale: bool = True):
        """根据训练数据初始化 Gumbel 参数（纯tensor）"""
        if events is not None:
            event_mask = (events == 1)
            if torch.any(event_mask):
                times = times[event_mask]
        times_norm = self._to_normalized_time(times)
        median_norm = torch.median(times_norm).item()

        if robust_scale:
            q25 = torch.quantile(times_norm, 0.25).item()
            q75 = torch.quantile(times_norm, 0.75).item()
            iqr = q75 - q25
            scale_est = iqr / 1.34898
        else:
            scale_est = torch.std(times_norm).item()

        scale_est = max(scale_est, 0.1)
        beta_init = scale_est / 1.28255
        alpha_init = median_norm + beta_init * math.log(math.log(2.0))
        log_beta_init = math.log(beta_init)

        with torch.no_grad():
            self.alpha_head[-1].bias.fill_(alpha_init)
            self.log_beta_head[-1].bias.fill_(log_beta_init)
        self._gumbel_initialized = True

    def get_gumbel_params(self, x: torch.Tensor):
        z = self.encoder(x)
        alpha = self.alpha_head(z)
        log_beta = self.log_beta_head(z)
        beta = torch.exp(torch.clamp(log_beta, min=-10.0, max=6.0))
        return alpha, beta

    def sample_prior(self, shape, device, alpha=None, beta=None):
        u = torch.rand(shape, device=device).clamp_(1e-6, 1 - 1e-6)
        log_u = torch.log(u)
        z = -torch.log(-log_u)
        z = torch.clamp(z, -10.0, 10.0)
        if alpha is not None and beta is not None:
            z = alpha + beta * z
        return torch.clamp(z, -20.0, 20.0)

    def log_prob_prior(self, z, alpha, beta):
        beta = torch.clamp(beta, 1e-6, 1e6)
        log_beta = torch.log(beta)
        std = torch.clamp((z - alpha) / beta, -20.0, 20.0)
        exp_neg_std = torch.exp(-std)
        log_prob = -log_beta - std - exp_neg_std
        return torch.clamp(log_prob, min=-100.0, max=88.0)

    def forward_loss(self, features: torch.Tensor, times_raw: torch.Tensor, events: torch.Tensor, **kwargs):
        device = features.device
        t1 = self._to_normalized_time(times_raw).float().unsqueeze(-1)
        mod_params = self.get_film(features)
        alpha, beta = self.get_gumbel_params(features)

        event_mask = (events == 1)
        censored_mask = (events == 0)

        if not event_mask.any() and not censored_mask.any():
            return torch.tensor(0.0, device=device, requires_grad=True), {}

        total_loss = torch.zeros(1, device=device, requires_grad=True)
        loss_dict = {}

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
            logp = self.log_prob_prior(t1_event, alpha_event, beta_event)
            mle_loss = -logp.mean()
            total_loss = total_loss + self.weight_event * event_loss + self.weight_gumbel * mle_loss
            loss_dict['event_loss'] = event_loss.item()
            loss_dict['mle_loss'] = mle_loss.item()

        if censored_mask.any():
            t_obs = t1[censored_mask]
            mod_cens = mod_params[censored_mask]
            alpha_cens = alpha[censored_mask]
            beta_cens = beta[censored_mask]
            t_truncated = self._sample_truncated_exponential(t_obs, self.truncated_samples)
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

        loss_dict.setdefault('event_loss', 0.0)
        loss_dict.setdefault('censored_loss', 0.0)
        loss_dict.setdefault('mle_loss', 0.0)
        return total_loss, loss_dict

    def predict_time(self, features: torch.Tensor, mode: str = 'median', **kwargs) -> torch.Tensor:
        self.eval()
        with torch.no_grad():
            mod_params = self.get_film(features)
            alpha, beta = self.get_gumbel_params(features)
            B = features.size(0)
            device = features.device
            n_samples = self.n_samples if mode != 'ode_step' else 1
            if mode == 'ode_step':
                log_log_2 = torch.log(torch.log(torch.tensor(2.0, device=device)))
                t0 = alpha - beta * log_log_2
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

    def compute_density(self, features: torch.Tensor, time_grid: torch.Tensor,
                        ode_steps: int = 100, batch_size_limit: int = 50000) -> torch.Tensor:
        self.eval()
        device = features.device
        mod_params = self.get_film(features)
        alpha, beta = self.get_gumbel_params(features)
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
            curr_alpha = alpha.unsqueeze(1).expand(-1, n_curr, -1).reshape(-1, 1)
            curr_beta = beta.unsqueeze(1).expand(-1, n_curr, -1).reshape(-1, 1)
            t1 = curr_norm.unsqueeze(0).expand(N, -1).reshape(-1, 1)
            with torch.enable_grad():
                t0, integral = self._inverse_flow_with_integral(t1, curr_mod, ode_steps)
                log_p0 = self.log_prob_prior(t0, curr_alpha, curr_beta).squeeze(-1)
                log_f_norm = log_p0 - integral.squeeze(-1)
                log_f_norm = torch.clamp(log_f_norm, min=-88.0, max=88.0)
                f_norm = torch.exp(log_f_norm).view(N, n_curr)
            f_phys = f_norm * jacobians[i:end_i].unsqueeze(0)
            f_phys = torch.clamp(f_phys, min=0.0, max=1000.0)
            all_densities.append(f_phys)
        return torch.cat(all_densities, dim=1)

    def predict_survival_function(self, features: torch.Tensor, time_grid: torch.Tensor = None,
                                  ode_steps: int = 100) -> torch.Tensor:
        self.eval()
        device = features.device
        if time_grid is None:
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
        f_full = self.compute_density(features, full_grid, ode_steps)
        dt = torch.diff(full_grid)
        areas = 0.5 * (f_full[:, :-1] + f_full[:, 1:]) * dt.unsqueeze(0)
        cdf = torch.zeros_like(f_full)
        cdf[:, 1:] = torch.cumsum(areas, dim=1)
        s_full = torch.clamp(1.0 - cdf, min=0.0, max=1.0)
        return s_full[:, 1:] if add_zero else s_full

    # 其余方法（hazard, cum_hazard, metrics, risk）复用父类，无需重写