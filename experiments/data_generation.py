import numpy as np
import torch
from dataclasses import dataclass
from typing import Tuple, Optional, Dict, Union
from scipy import stats


@dataclass
class SurvivalData:
    features: Union[np.ndarray, torch.Tensor]
    times: Union[np.ndarray, torch.Tensor]
    events: Union[np.ndarray, torch.Tensor]
    true_times: Union[np.ndarray, torch.Tensor]
    true_hazard: Optional[Union[np.ndarray, torch.Tensor]] = None
    true_density: Optional[Union[np.ndarray, torch.Tensor]] = None
    true_survival: Optional[Union[np.ndarray, torch.Tensor]] = None
    time_grid: Optional[Union[np.ndarray, torch.Tensor]] = None
    feature_names: Optional[list] = None

    def to(self, device: Union[torch.device, str]) -> 'SurvivalData':
        def _to_tensor(x, dtype=torch.float32):
            if x is None:
                return None
            if isinstance(x, np.ndarray):
                return torch.from_numpy(x).to(dtype=dtype, device=device)
            if isinstance(x, torch.Tensor):
                return x.to(dtype=dtype, device=device)
            return x
        return SurvivalData(
            features=_to_tensor(self.features),
            times=_to_tensor(self.times),
            events=_to_tensor(self.events),
            true_times=_to_tensor(self.true_times),
            true_hazard=_to_tensor(self.true_hazard),
            true_density=_to_tensor(self.true_density),
            true_survival=_to_tensor(self.true_survival),
            time_grid=_to_tensor(self.time_grid),
            feature_names=self.feature_names
        )


class WeibullMixtureGenerator:
    """Weibull混合分布生成器"""
    
    def __init__(self, config):
        self.config = config
        np.random.seed(config.random_seed)
        
    def _compute_params(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """计算条件参数"""
        n = X.shape[0]
        
        if self.config.is_ph:
            lambda1 = self.config.weibull_lambda1_base * np.exp(
                self.config.beta_linear * X[:, 0]
            )
            lambda2 = self.config.weibull_lambda2_base * np.exp(
                self.config.beta_linear * X[:, 0]
            )
            pi1 = np.full(n, self.config.mixture_weight_base)
        else:
            lambda1 = self.config.weibull_lambda1_base * np.exp(
                self.config.beta_linear * X[:, 0] + 
                self.config.beta_nonlinear * X[:, 1]**2
            )
            lambda2 = self.config.weibull_lambda2_base * np.exp(
                self.config.beta_linear * X[:, 0] - 
                self.config.beta_nonlinear * X[:, 1]**2
            )
            pi1 = 1.0 / (1.0 + np.exp(-self.config.beta_linear * (X[:, 0] + X[:, 1])))
        
        return lambda1, lambda2, pi1
    
    def sample(self, X: np.ndarray) -> np.ndarray:
        """采样生存时间"""
        lambda1, lambda2, pi1 = self._compute_params(X)
        n = X.shape[0]
        
        component = np.random.binomial(1, pi1, size=n)
        
        t1 = np.random.weibull(self.config.weibull_k1, size=n) * lambda1
        t2 = np.random.weibull(self.config.weibull_k2, size=n) * lambda2
        
        times = np.where(component == 1, t1, t2)
        return np.maximum(times, 1e-6)
    
    def pdf(self, t: np.ndarray, X: np.ndarray) -> np.ndarray:
        """概率密度函数 f(t|x)"""
        lambda1, lambda2, pi1 = self._compute_params(X)
        
        k1, k2 = self.config.weibull_k1, self.config.weibull_k2
        
        t = t[np.newaxis, :]
        lambda1 = lambda1[:, np.newaxis]
        lambda2 = lambda2[:, np.newaxis]
        
        f1 = (k1 / lambda1) * (t / lambda1)**(k1 - 1) * np.exp(-(t / lambda1)**k1)
        f2 = (k2 / lambda2) * (t / lambda2)**(k2 - 1) * np.exp(-(t / lambda2)**k2)
        
        return pi1[:, np.newaxis] * f1 + (1 - pi1[:, np.newaxis]) * f2
    
    def survival(self, t: np.ndarray, X: np.ndarray) -> np.ndarray:
        """生存函数 S(t|x)"""
        lambda1, lambda2, pi1 = self._compute_params(X)
        k1, k2 = self.config.weibull_k1, self.config.weibull_k2
        
        S1 = np.exp(-(t / lambda1[:, np.newaxis])**k1)
        S2 = np.exp(-(t / lambda2[:, np.newaxis])**k2)
        
        return pi1[:, np.newaxis] * S1 + (1 - pi1[:, np.newaxis]) * S2
    
    def hazard(self, t: np.ndarray, X: np.ndarray) -> np.ndarray:
        """风险函数 h(t|x) = f(t|x) / S(t|x) (带数值稳定性保护)"""
        f = self.pdf(t, X)
        S = self.survival(t, X)
        h = f / np.maximum(S, 1e-6)
        return np.clip(h, 0.0, 1000.0)
    
    def median(self, X: np.ndarray) -> np.ndarray:
        """中位生存时间 (数值求解)"""
        lambda1, lambda2, pi1 = self._compute_params(X)
        k1, k2 = self.config.weibull_k1, self.config.weibull_k2
        
        t_grid = np.linspace(0.01, 20, 1000)
        medians = np.zeros(X.shape[0])
        
        for i in range(X.shape[0]):
            S = self.survival(t_grid, X[i:i+1])[0]
            idx = np.searchsorted(-S, -0.5)
            if idx >= len(t_grid):
                medians[i] = t_grid[-1]
            elif idx == 0:
                medians[i] = t_grid[0]
            else:
                medians[i] = t_grid[idx]
        
        return medians


class WeibullSingleGenerator:
    """Weibull单峰分布生成器 (PH场景)"""
    
    def __init__(self, config):
        self.config = config
        np.random.seed(config.random_seed)
        self.k = config.weibull_k1
        
    def _compute_lambda(self, X: np.ndarray) -> np.ndarray:
        """计算尺度参数"""
        return self.config.weibull_lambda1_base * np.exp(
            self.config.beta_linear * X[:, 0]
        )
    
    def sample(self, X: np.ndarray) -> np.ndarray:
        """采样"""
        lam = self._compute_lambda(X)
        times = np.random.weibull(self.k, size=X.shape[0]) * lam
        return np.maximum(times, 1e-6)
    
    def pdf(self, t: np.ndarray, X: np.ndarray) -> np.ndarray:
        """概率密度函数"""
        lam = self._compute_lambda(X)
        k = self.k
        f = (k / lam[:, np.newaxis]) * (t / lam[:, np.newaxis])**(k - 1) * \
            np.exp(-(t / lam[:, np.newaxis])**k)
        return f
    
    def survival(self, t: np.ndarray, X: np.ndarray) -> np.ndarray:
        """生存函数"""
        lam = self._compute_lambda(X)
        return np.exp(-(t / lam[:, np.newaxis])**self.k)
    
    def hazard(self, t: np.ndarray, X: np.ndarray) -> np.ndarray:
        """风险函数 (带数值稳定性保护)"""
        lam = self._compute_lambda(X)
        k = self.k
        h = (k / lam[:, np.newaxis]) * (t / lam[:, np.newaxis])**(k - 1)
        return np.clip(h, 0.0, 1000.0)
    
    def median(self, X: np.ndarray) -> np.ndarray:
        """中位生存时间"""
        lam = self._compute_lambda(X)
        return lam * (np.log(2))**(1 / self.k)


class GaussianMixtureGenerator:
    """高斯混合分布生成器 (对数时间域)"""
    
    def __init__(self, config):
        self.config = config
        np.random.seed(config.random_seed)
    
    def _compute_params(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """计算条件参数"""
        n = X.shape[0]
        
        mu1 = self.config.gaussian_mu1_base + self.config.beta_linear * X[:, 0]
        mu2 = self.config.gaussian_mu2_base + self.config.beta_linear * X[:, 0]
        
        if not self.config.is_ph:
            mu1 += self.config.beta_nonlinear * X[:, 1]**2
            mu2 -= self.config.beta_nonlinear * X[:, 1]**2
        
        pi1 = np.full(n, self.config.gaussian_mixture_weight)
        
        return mu1, mu2, pi1
    
    def sample(self, X: np.ndarray) -> np.ndarray:
        """采样生存时间"""
        mu1, mu2, pi1 = self._compute_params(X)
        n = X.shape[0]
        
        component = np.random.binomial(1, pi1, size=n)
        
        log_t1 = np.random.normal(mu1, self.config.gaussian_sigma1)
        log_t2 = np.random.normal(mu2, self.config.gaussian_sigma2)
        
        log_t = np.where(component == 1, log_t1, log_t2)
        return np.exp(log_t)
    
    def pdf(self, t: np.ndarray, X: np.ndarray) -> np.ndarray:
        """概率密度函数"""
        mu1, mu2, pi1 = self._compute_params(X)
        sigma1, sigma2 = self.config.gaussian_sigma1, self.config.gaussian_sigma2
        
        log_t = np.log(t + 1e-10)
        
        f1 = stats.norm.pdf(log_t, mu1[:, np.newaxis], sigma1) / (t + 1e-10)
        f2 = stats.norm.pdf(log_t, mu2[:, np.newaxis], sigma2) / (t + 1e-10)
        
        return pi1[:, np.newaxis] * f1 + (1 - pi1[:, np.newaxis]) * f2
    
    def survival(self, t: np.ndarray, X: np.ndarray) -> np.ndarray:
        """生存函数"""
        mu1, mu2, pi1 = self._compute_params(X)
        sigma1, sigma2 = self.config.gaussian_sigma1, self.config.gaussian_sigma2
        
        log_t = np.log(t + 1e-10)
        
        S1 = 1 - stats.norm.cdf(log_t, mu1[:, np.newaxis], sigma1)
        S2 = 1 - stats.norm.cdf(log_t, mu2[:, np.newaxis], sigma2)
        
        return pi1[:, np.newaxis] * S1 + (1 - pi1[:, np.newaxis]) * S2
    
    def hazard(self, t: np.ndarray, X: np.ndarray) -> np.ndarray:
        """风险函数 (带数值稳定性保护)"""
        f = self.pdf(t, X)
        S = self.survival(t, X)
        h = f / np.maximum(S, 1e-6)
        return np.clip(h, 0.0, 1000.0)
    
    def median(self, X: np.ndarray) -> np.ndarray:
        """中位生存时间"""
        t_grid = np.linspace(0.1, 200, 1000)
        medians = np.zeros(X.shape[0])
        
        for i in range(X.shape[0]):
            S = self.survival(t_grid, X[i:i+1])[0]
            idx = np.searchsorted(-S, -0.5)
            if idx >= len(t_grid):
                medians[i] = t_grid[-1]
            elif idx == 0:
                medians[i] = t_grid[0]
            else:
                medians[i] = t_grid[idx]
        
        return medians


class SurvivalDataGenerator:
    """生存数据生成器主类"""
    
    def __init__(self, config):
        self.config = config
        
        if config.distribution_type == "weibull_mixture":
            self.generator = WeibullMixtureGenerator(config)
        elif config.distribution_type == "weibull_single":
            self.generator = WeibullSingleGenerator(config)
        elif config.distribution_type == "gaussian_mixture":
            self.generator = GaussianMixtureGenerator(config)
        else:
            raise ValueError(f"Unknown distribution type: {config.distribution_type}")
    
    def generate_features(self, n: int, seed: int = None) -> np.ndarray:
        """生成协变量"""
        if seed is not None:
            np.random.seed(seed)
        
        X = np.zeros((n, self.config.n_features))
        
        for j in range(self.config.n_signal_features):
            X[:, j] = np.random.uniform(-1, 1, n)
        
        for j in range(self.config.n_signal_features, self.config.n_features):
            X[:, j] = np.random.normal(0, 1, n)
        
        if self.config.noise_std > 0:
            X += np.random.normal(0, self.config.noise_std, X.shape)
        
        return X
    
    def generate_censoring(self, n: int, true_times: np.ndarray, seed: int = None) -> np.ndarray:
        """生成删失时间"""
        if seed is not None:
            np.random.seed(seed)
        
        lambda_c = self.config.get_censoring_lambda()
        
        mean_time = np.mean(true_times)
        lambda_c_adjusted = lambda_c / mean_time
        
        censoring_times = np.random.exponential(1.0 / lambda_c_adjusted, n)
        
        return censoring_times
    
    def generate(self, n: int = None, seed: int = None) -> SurvivalData:
        """生成完整生存数据集"""
        if n is None:
            n = self.config.n_samples
        if seed is None:
            seed = self.config.random_seed
        
        np.random.seed(seed)
        
        X = self.generate_features(n, seed)
        
        true_times = self.generator.sample(X)
        
        censoring_times = self.generate_censoring(n, true_times, seed + 1000)
        
        observed_times = np.minimum(true_times, censoring_times)
        events = (true_times <= censoring_times).astype(int)
        
        actual_censoring_rate = 1 - events.mean()
        
        t_max = np.percentile(observed_times, 95)
        time_grid = np.linspace(0.01, t_max, 100)
        
        true_hazard = self.generator.hazard(time_grid, X)
        true_density = self.generator.pdf(time_grid, X)
        true_survival = self.generator.survival(time_grid, X)
        
        feature_names = [f"X{i+1}" for i in range(self.config.n_features)]
        for i in range(self.config.n_signal_features):
            feature_names[i] = f"X{i+1}_signal"
        
        return SurvivalData(
            features=X,
            times=observed_times,
            events=events,
            true_times=true_times,
            true_hazard=true_hazard,
            true_density=true_density,
            true_survival=true_survival,
            time_grid=time_grid,
            feature_names=feature_names
        )
    
    def compute_true_metrics(self, X: np.ndarray, time_grid: np.ndarray) -> Dict[str, np.ndarray]:
        """计算真实的风险、密度、生存函数"""
        return {
            'hazard': self.generator.hazard(time_grid, X),
            'density': self.generator.pdf(time_grid, X),
            'survival': self.generator.survival(time_grid, X),
            'median': self.generator.median(X)
        }


def generate_experiment_data(config, seed_offset: int = 0) -> Tuple[SurvivalData, SurvivalData, SurvivalData]:
    """生成实验用的训练/验证/测试集"""
    full_data = SurvivalDataGenerator(config).generate(
        n=config.n_samples,
        seed=config.random_seed + seed_offset
    )
    
    n = len(full_data.times)
    n_test = int(n * 0.2)
    n_val = int((n - n_test) * 0.1)
    n_train = n - n_test - n_val
    
    indices = np.random.permutation(n)
    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train + n_val]
    test_idx = indices[n_train + n_val:]
    
    def split_data(data: SurvivalData, idx: np.ndarray) -> SurvivalData:
        return SurvivalData(
            features=data.features[idx],
            times=data.times[idx],
            events=data.events[idx],
            true_times=data.true_times[idx],
            true_hazard=data.true_hazard[idx] if data.true_hazard is not None else None,
            true_density=data.true_density[idx] if data.true_density is not None else None,
            true_survival=data.true_survival[idx] if data.true_survival is not None else None,
            time_grid=data.time_grid,
            feature_names=data.feature_names
        )
    
    train_data = split_data(full_data, train_idx)
    val_data = split_data(full_data, val_idx)
    test_data = split_data(full_data, test_idx)
    
    return train_data, val_data, test_data


if __name__ == "__main__":
    from experiments.config import DataConfig, print_config_summary
    
    print("测试数据生成模块")
    print("=" * 60)
    
    config = DataConfig(
        n_samples=1000,
        distribution_type="weibull_mixture",
        censoring_rate=0.4
    )
    
    generator = SurvivalDataGenerator(config)
    data = generator.generate()
    
    print(f"样本量: {len(data.times)}")
    print(f"特征维度: {data.features.shape}")
    print(f"事件率: {data.events.mean():.2%}")
    print(f"时间范围: [{data.times.min():.2f}, {data.times.max():.2f}]")
    print(f"时间网格: {len(data.time_grid)} 点")
