from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
import torch
import numpy as np


@dataclass
class DataConfig:
    """数据生成配置"""
    n_samples: int = 2000
    n_features: int = 5
    n_signal_features: int = 2
    noise_std: float = 0.1
    censoring_rate: float = 0.4
    distribution_type: str = "weibull_mixture"
    is_ph: bool = True
    random_seed: int = 42
    
    # Weibull混合参数
    weibull_k1: float = 1.5
    weibull_k2: float = 3.0
    weibull_lambda1_base: float = 1.0
    weibull_lambda2_base: float = 2.0
    mixture_weight_base: float = 0.5
    
    # 高斯混合参数
    gaussian_mu1_base: float = 2.0
    gaussian_mu2_base: float = 4.0
    gaussian_sigma1: float = 0.5
    gaussian_sigma2: float = 0.8
    gaussian_mixture_weight: float = 0.5
    
    # 协变量效应系数
    beta_linear: float = 0.5
    beta_nonlinear: float = 0.3
    
    def get_censoring_lambda(self) -> float:
        """根据目标删失率计算指数删失参数"""
        return -np.log(1 - self.censoring_rate + 1e-6) / 3.0


@dataclass
class ModelConfig:
    """
    模型超参数配置
    
    【命名规范】
    - 网络架构参数: snake_case (hidden_dims, dropout, n_time_bins 等)
    - 训练参数: 大写 (LR, BATCH_SIZE, EPOCHS 等)
    - 与代码中 self.config.get() 保持一致
    """
    configs: Dict[str, Dict[str, Any]] = field(default_factory=lambda: {
        'LinearCoxPH': {
            'penalizer': 1e-5,
        },
        'DeepSurv': {
            'hidden_dims': [64, 32, 16, 8],
            'dropout': 0.1,
            'LR': 8e-5,
            'BATCH_SIZE': 64,
            'EPOCHS': 200,
            'PATIENCE': 10,
            'WEIGHT_DECAY': 1e-5,
        },
        'WeibullAFT': {
            'hidden_dims': [32, 16],
            'dropout': 0.0,
            'LR': 5e-8,
            'BATCH_SIZE': 64,
            'EPOCHS': 200,
            'PATIENCE': 15,
            'WEIGHT_DECAY': 1e-5,
        },
        'RSF': {
            'n_estimators': 100,
            'max_depth': None,
            'min_samples_split': 6,
            'min_samples_leaf': 3,
            'random_state': 42,
        },
        'DeepHit': {
            'hidden_dims': [32, 16, 8],
            'dropout': 0.1,
            'n_time_bins': 50,
            'LR': 5e-8,
            'BATCH_SIZE': 64,
            'EPOCHS': 200,
            'PATIENCE': 10,
            'WEIGHT_DECAY': 1e-5,
        },
        'FlowSurv': {
            'vf_hidden_dims': [32, 16, 8],
            'tau_dim': 32,
            'encoder_hidden': [32, 16],
            'film_hidden': [16],
            'sigma': 0.1,
            'dropout': 0.1,
            'LR': 8e-5,
            'BATCH_SIZE': 64,
            'EPOCHS': 200,
            'PATIENCE': 10,
            'WEIGHT_DECAY': 1e-5,
            'n_samples': 256,
            'ode_steps': 100,
            'solver': 'euler',
            'truncated_samples': 32,
            'weight_event': 2.0,
            'weight_censored': 1.0,
        },
        'GumbelFlowSurv': {
            'vf_hidden_dims': [32, 16, 8],
            'tau_dim': 32,
            'encoder_hidden': [32, 16],
            'film_hidden': [16],
            'weibull_head_hidden': [16],
            'sigma': 0.1,
            'dropout': 0.1,
            'LR': 1e-4,
            'BATCH_SIZE': 64,
            'EPOCHS': 200,
            'PATIENCE': 10,
            'WEIGHT_DECAY': 1e-5,
            'WEIBULL_LR': 5e-8,
            'WEIBULL_EPOCHS': 200,
            'WEIBULL_BATCH_SIZE': 64,
            'WEIBULL_PATIENCE': 15,
            'WEIBULL_WEIGHT_DECAY': 1e-5,
            'n_samples': 256,
            'ode_steps': 100,
            'solver': 'euler',
            'truncated_samples': 32,
            'weight_event': 1.0,
            'weight_censored': 2.0,
        },
    })


@dataclass
class PlotConfig:
    """绘图配置"""
    figsize_single: tuple = (8, 6)
    figsize_multi: tuple = (14, 10)
    dpi: int = 150
    font_size: int = 12
    n_representative_samples: int = 5
    time_grid_points: int = 100
    hazard_surface_points: int = 50
    colors: Dict[str, str] = field(default_factory=lambda: {
        'true': '#2E86AB',
        'LinearCoxPH': '#A23B72',
        'DeepSurv': '#F18F01',
        'WeibullAFT': '#C73E1D',
        'RSF': '#3B1F2B',
        'DeepHit': '#95C623',
        'FlowSurv': '#1B998B',
        'GumbelFlowSurv': '#E84855',
    })
    linestyles: Dict[str, str] = field(default_factory=lambda: {
        'true': '-',
        'LinearCoxPH': '--',
        'DeepSurv': '-.',
        'WeibullAFT': ':',
        'RSF': '--',
        'DeepHit': '-.',
        'FlowSurv': '-',
        'GumbelFlowSurv': '-',
    })
    save_format: str = 'png'
    save_dir: str = 'results/figures'


@dataclass 
class ExperimentGroup:
    """单个实验组配置"""
    name: str
    description: str
    data_config: DataConfig
    is_nph: bool = False
    is_high_censoring: bool = False
    is_small_sample: bool = False
    is_high_noise: bool = False


def _create_default_groups() -> List[ExperimentGroup]:
    """创建默认实验组"""
    groups = []
    
    # ==================== 基础场景 ====================
    e1_data = DataConfig(
        n_samples=2000, n_features=5, n_signal_features=2,
        noise_std=0.1, censoring_rate=0.3,
        distribution_type="weibull_single", is_ph=True,
        random_seed=42
    )
    groups.append(ExperimentGroup(
        name="E1",
        description="PH场景, Weibull单峰, 常规删失",
        data_config=e1_data, is_nph=False
    ))
    
    e2_data = DataConfig(
        n_samples=2000, n_features=5, n_signal_features=2,
        noise_std=0.1, censoring_rate=0.3,
        distribution_type="weibull_mixture", is_ph=False,
        weibull_k1=1.5, weibull_k2=3.5,
        random_seed=52
    )
    groups.append(ExperimentGroup(
        name="E2",
        description="NPH场景, Weibull混合, 常规删失",
        data_config=e2_data, is_nph=True
    ))
    
    # ==================== 高删失场景 ====================
    e3_data = DataConfig(
        n_samples=2000, n_features=5, n_signal_features=2,
        noise_std=0.1, censoring_rate=0.7,
        distribution_type="weibull_mixture", is_ph=False,
        weibull_k1=1.5, weibull_k2=3.5,
        random_seed=62
    )
    groups.append(ExperimentGroup(
        name="E3",
        description="NPH场景, Weibull混合, 高删失70%",
        data_config=e3_data, is_nph=True, is_high_censoring=True
    ))
    
    # ==================== 多峰密度场景 ====================
    e4_data = DataConfig(
        n_samples=2000, n_features=5, n_signal_features=2,
        noise_std=0.1, censoring_rate=0.3,
        distribution_type="gaussian_mixture", is_ph=False,
        gaussian_mu1_base=1.5, gaussian_mu2_base=4.0,
        gaussian_sigma1=0.4, gaussian_sigma2=0.6,
        random_seed=72
    )
    groups.append(ExperimentGroup(
        name="E4",
        description="多峰密度, 高斯混合",
        data_config=e4_data, is_nph=True
    ))
    
    # ==================== 小样本场景 ====================
    e5_data = DataConfig(
        n_samples=500, n_features=5, n_signal_features=2,
        noise_std=0.1, censoring_rate=0.3,
        distribution_type="weibull_mixture", is_ph=False,
        weibull_k1=1.5, weibull_k2=3.5,
        random_seed=82
    )
    groups.append(ExperimentGroup(
        name="E5",
        description="小样本500, NPH场景",
        data_config=e5_data, is_nph=True, is_small_sample=True
    ))
    
    # ==================== 高噪声场景 ====================
    e6_data = DataConfig(
        n_samples=2000, n_features=10, n_signal_features=2,
        noise_std=0.5, censoring_rate=0.3,
        distribution_type="weibull_mixture", is_ph=False,
        weibull_k1=1.5, weibull_k2=3.5,
        random_seed=92
    )
    groups.append(ExperimentGroup(
        name="E6",
        description="高维噪声(10维), 高噪声强度",
        data_config=e6_data, is_nph=True, is_high_noise=True
    ))
    
    # ==================== 大样本场景 (5000) ====================
    e7_data = DataConfig(
        n_samples=5000, n_features=5, n_signal_features=2,
        noise_std=0.1, censoring_rate=0.3,
        distribution_type="weibull_mixture", is_ph=False,
        weibull_k1=1.5, weibull_k2=3.5,
        random_seed=102
    )
    groups.append(ExperimentGroup(
        name="E7",
        description="大样本5000, NPH场景",
        data_config=e7_data, is_nph=True
    ))
    
    # ==================== 大样本场景 (10000) ====================
    e8_data = DataConfig(
        n_samples=10000, n_features=5, n_signal_features=2,
        noise_std=0.1, censoring_rate=0.3,
        distribution_type="weibull_mixture", is_ph=False,
        weibull_k1=1.5, weibull_k2=3.5,
        random_seed=112
    )
    groups.append(ExperimentGroup(
        name="E8",
        description="大样本10000, NPH场景",
        data_config=e8_data, is_nph=True
    ))
    
    # ==================== 大样本高删失场景 (5000, 70%) ====================
    e9_data = DataConfig(
        n_samples=5000, n_features=5, n_signal_features=2,
        noise_std=0.1, censoring_rate=0.7,
        distribution_type="weibull_mixture", is_ph=False,
        weibull_k1=1.5, weibull_k2=3.5,
        random_seed=122
    )
    groups.append(ExperimentGroup(
        name="E9",
        description="大样本5000, 高删失70%, NPH场景",
        data_config=e9_data, is_nph=True, is_high_censoring=True
    ))
    
    # ==================== 大样本高删失场景 (10000, 70%) ====================
    e10_data = DataConfig(
        n_samples=10000, n_features=5, n_signal_features=2,
        noise_std=0.1, censoring_rate=0.7,
        distribution_type="weibull_mixture", is_ph=False,
        weibull_k1=1.5, weibull_k2=3.5,
        random_seed=132
    )
    groups.append(ExperimentGroup(
        name="E10",
        description="大样本10000, 高删失70%, NPH场景",
        data_config=e10_data, is_nph=True, is_high_censoring=True
    ))
    
    return groups


@dataclass
class ExperimentConfig:
    """实验总配置"""
    base_seed: int = 42
    output_dir: str = 'results'
    save_predictions: bool = True
    save_models: bool = False
    n_repeats: int = 10
    val_ratio: float = 0.1
    test_ratio: float = 0.2
    
    groups: List[ExperimentGroup] = field(default_factory=_create_default_groups)
    
    time_quantiles: List[float] = field(default_factory=lambda: [0.25, 0.5, 0.75])
    ipcw_max_weight: float = 20.0
    calibration_bins: int = 10


def get_model_config(model_name: str, model_config: ModelConfig = None) -> Dict[str, Any]:
    """获取指定模型的超参数配置"""
    if model_config is None:
        model_config = ModelConfig()
    return model_config.configs.get(model_name, {})


@dataclass
class GlobalConfig:
    """全局配置汇总"""
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    plot: PlotConfig = field(default_factory=PlotConfig)
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)
    
    def get_experiment_group(self, name: str) -> Optional[ExperimentGroup]:
        """根据名称获取实验组"""
        for group in self.experiment.groups:
            if group.name == name:
                return group
        return None


CONFIG = GlobalConfig()


def print_config_summary():
    """打印配置摘要"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("=" * 60)
    print("FlowSurv / GumbelFlowSurv 实验配置摘要")
    print("=" * 60)
    print(f"\n设备: {device}")
    print(f"基础随机种子: {CONFIG.experiment.base_seed}")
    print(f"重复次数: {CONFIG.experiment.n_repeats}")
    print(f"\n实验组数量: {len(CONFIG.experiment.groups)}")
    
    print("\n" + "-" * 60)
    print("实验组列表:")
    print("-" * 60)
    for i, group in enumerate(CONFIG.experiment.groups, 1):
        flags = []
        if group.is_nph: flags.append("NPH")
        if group.is_high_censoring: flags.append("高删失")
        if group.is_small_sample: flags.append("小样本")
        if group.is_high_noise: flags.append("高噪声")
        flag_str = f" [{', '.join(flags)}]" if flags else ""
        
        print(f"  {i}. {group.name}{flag_str}")
        print(f"     样本量: {group.data_config.n_samples}, 特征: {group.data_config.n_features}")
        print(f"     删失率: {group.data_config.censoring_rate:.0%}, 分布: {group.data_config.distribution_type}")
    
    print("\n" + "-" * 60)
    print("模型列表:")
    print("-" * 60)
    for model_name in CONFIG.model.configs.keys():
        print(f"  - {model_name}")
    
    print("\n" + "=" * 60)


if __name__ == "__main__":
    print_config_summary()
