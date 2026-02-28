import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Union, Optional, Dict, Tuple

from models.flowmodel.base_flow import FlowSurv
from models.flowmodel.components import FiLMResidualBlock, odeint_euler, odeint_rk4


class GumbelFlowSurv(FlowSurv):
    # 数学常量
    _LN_2 = 0.6931471805599453      # ln(2)
    _LN_LN_2 = -0.3665129205816643  # ln(ln(2))

    def __init__(self, in_dim: int, config: Optional[dict] = None, **kwargs):
        super().__init__(in_dim, config, **kwargs)
        self.stage = self.config.get('stage', 'flow')

        z_dim = self.encoder_dims[-1]
        weibull_layers = []
        prev = z_dim
        for d in self.config.get('weibull_head_hidden', [16, 8]):
            weibull_layers.extend([nn.Linear(prev, d), nn.SiLU()])
            prev = d
        weibull_layers.append(nn.Linear(prev, 2))
        self.weibull_head = nn.Sequential(*weibull_layers)

        self.set_stage(self.stage)

    def set_stage(self, stage: str):
        if stage not in {'weibull', 'flow'}:
            raise ValueError(f"Unknown stage: {stage}")
        self.stage = stage
        if stage == 'weibull':
            self._set_requires_grad(self.encoder, True)
            self._set_requires_grad(self.weibull_head, True)
            self._set_requires_grad(self.film_head, False)
            self._set_requires_grad(self.vector_field, False)
        else:
            self._set_requires_grad(self.encoder, True)
            self._set_requires_grad(self.weibull_head, False)
            self._set_requires_grad(self.film_head, True)
            self._set_requires_grad(self.vector_field, True)

    def _set_requires_grad(self, module: nn.Module, flag: bool):
        for p in module.parameters():
            p.requires_grad = flag

    def _weibull_params(self, z: torch.Tensor):
        params = self.weibull_head(z)
        log_k = torch.clamp(params[:, 0:1], -5.0, 5.0)
        log_lam = torch.clamp(params[:, 1:2], -5.0, 5.0)
        k = torch.exp(log_k)
        lam = torch.exp(log_lam)
        return k, lam

    def _weibull_to_gumbel(self, k: torch.Tensor, lam: torch.Tensor):
        log_2 = lam.new_tensor(self._LN_2)
        t_median = lam * torch.pow(log_2, 1.0 / k)
        median_norm = self._to_normalized_time(t_median)
        beta = (1.0 / k) / (self.time_scaler_std + 1e-8)
        ln_ln2 = lam.new_tensor(self._LN_LN_2)
        alpha = median_norm - beta * ln_ln2
        alpha = torch.clamp(alpha, min=-10.0, max=10.0)
        log_beta = torch.log(torch.clamp(beta, min=1e-6))
        log_beta = torch.clamp(log_beta, min=-8.0, max=4.0)
        beta = torch.exp(log_beta)
        return alpha, beta

    def get_gumbel_params(self, x: torch.Tensor, z: Optional[torch.Tensor] = None):
        if z is None:
            z = self.encoder(x)
        z_weibull = z.detach() if self.stage == 'flow' else z
        k, lam = self._weibull_params(z_weibull)
        alpha, beta = self._weibull_to_gumbel(k, lam)
        return alpha, beta

    def sample_prior(self, shape, device, alpha=None, beta=None):
        """从 Gumbel 最小值分布采样 (Weibull 对应)"""
        # 更加数值稳定的 Gumbel 采样: z = α - β * log(-log U) 
        # 注意: 此处为 Gumbel 最小值分布, 对应 Weibull AFT 线性部分
        u = torch.rand(shape, device=device).clamp_(1e-10, 1.0 - 1e-10)
        # 使用 -log(-log(u)) 采样标准 Gumbel, 
        # 然后根据 Weibull AFT 对应的 Gumbel 最小值分布转换
        log_neg_log_u = torch.log(-torch.log(u))
        z = log_neg_log_u  # 标准 Gumbel 最小值分布的核心部分
        
        # 限制范围以防止溢出
        z = torch.clamp(z, min=-15.0, max=15.0)
        
        if alpha is not None and beta is not None:
            z = alpha + beta * z
        return torch.clamp(z, min=-20.0, max=20.0)

    def log_prob_prior(self, z, alpha, beta):
        """计算 Gumbel 最小值分布的对数似然, 带有数值保护"""
        # 确保 beta 为正且在一个合理范围内
        beta = torch.clamp(beta, min=1e-5, max=100.0)
        log_beta = torch.log(beta)
        # 限制标准化残差, 防止 exp(std) 溢出
        std = torch.clamp((z - alpha) / beta, min=-20.0, max=10.0)
        # Gumbel Min PDF: f(z) = 1/β * exp(std - exp(std))
        exp_std = torch.exp(std)
        log_prob = -log_beta + std - exp_std
        # 对最终对数似然进行截断保护
        return torch.clamp(log_prob, min=-100.0, max=88.0)

    def forward_loss(self, features: torch.Tensor, times_raw: torch.Tensor, events: torch.Tensor, **kwargs):
        device = features.device
        t1 = self._to_normalized_time(times_raw).float().unsqueeze(-1)
        if self.stage == 'weibull':
            z = self.encoder(features)
            k, lam = self._weibull_params(z)
            t = torch.clamp(times_raw, min=1e-6)
            t_lam = t / lam
            term_pow = torch.pow(t_lam, k)
            log_pdf = torch.log(k) - torch.log(lam) + (k - 1) * torch.log(t_lam) - term_pow
            log_surv = -term_pow
            log_lik = events * log_pdf + (1 - events) * log_surv
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

        return total_loss, loss_dict

    def predict_time(self, features: torch.Tensor, mode: str = 'median', **kwargs) -> torch.Tensor:
        self.eval()
        with torch.no_grad():
            z = self.encoder(features)
            mod_params = self.film_head(z)
            alpha, beta = self.get_gumbel_params(features, z=z)
            B = features.size(0)
            device = features.device
            n_samples = self.n_samples if mode != 'ode_step' else 1
            if mode == 'ode_step':
                # Gumbel 最小值分布的中位数: α + β * ln(ln 2)
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

    def compute_density(self, features: torch.Tensor, time_grid: torch.Tensor,
                        ode_steps: int = 100, batch_size_limit: int = 50000) -> torch.Tensor:
        self.eval()
        device = features.device
        z = self.encoder(features)
        mod_params = self.film_head(z)
        alpha, beta = self.get_gumbel_params(features, z=z)
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

    # 其余方法(hazard, cum_hazard, metrics, risk)复用父类,无需重写
