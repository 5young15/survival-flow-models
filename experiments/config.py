"""
生存分析实验配置模块

功能说明:
- 定义数据生成、模型超参数、实验设计、绘图等配置
- 支持 10 种不同实验场景 (PH/NPH、高删失、小样本、高噪声、大样本等)
- 使用 dataclass 进行类型安全的配置管理
- 提供配置摘要打印功能

配置项:
- DataConfig: 数据生成参数
- ModelConfig: 模型超参数
- PlotConfig: 绘图参数
- ExperimentGroup: 单个实验组配置
- ExperimentConfig: 实验总配置
- GlobalConfig: 全局配置汇总

作者：Statistical Modeling Team
日期：2026
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
import torch
import numpy as np


@dataclass
class DataConfig:
    """
    数据生成配置
    
    用于控制生存模拟数据的生成参数，包括:
    - 样本量、特征维度、噪声水平
    - 删失率、分布类型 (Weibull 单峰/混合、高斯混合)
    - PH/NPH 假设
    - 分布形状参数
    
    属性:
        n_samples: 样本数量
        n_features: 特征总数
        n_signal_features: 有预测能力的信号特征数量
        noise_std: 高斯噪声标准差
        censoring_rate: 目标删失率 (0-1 之间)
        distribution_type: 生存时间分布类型
        is_ph: 是否满足比例风险 (Proportional Hazards) 假设
        random_seed: 随机种子
        
        # Weibull 混合分布参数
        weibull_k1: 第一个 Weibull 成分的形状参数
        weibull_k2: 第二个 Weibull 成分的形状参数
        weibull_lambda1_base: 第一个 Weibull 成分的基准尺度参数
        weibull_lambda2_base: 第二个 Weibull 成分的基准尺度参数
        mixture_weight_base: 混合权重基准值
        
        # 高斯混合分布参数 (对数时间域)
        gaussian_mu1_base: 第一个高斯成分的均值基准
        gaussian_mu2_base: 第二个高斯成分的均值基准
        gaussian_sigma1: 第一个高斯成分的标准差
        gaussian_sigma2: 第二个高斯成分的标准差
        gaussian_mixture_weight: 混合权重
        
        # 协变量效应系数
        beta_linear: 线性协变量效应系数
        beta_nonlinear: 非线性协变量效应系数
    """
    n_samples: int = 2000  # 样本数量
    n_features: int = 5  # 特征总数
    n_signal_features: int = 2  # 信号特征数量 (有预测能力)
    noise_std: float = 0.1  # 高斯噪声标准差
    censoring_rate: float = 0.4  # 目标删失率 (0-1)
    distribution_type: str = "weibull_mixture"  # 分布类型: weibull_single/weibull_mixture/gaussian_mixture
    is_ph: bool = True  # 是否满足比例风险假设
    random_seed: int = 42  # 随机种子
    
    # Weibull 混合参数
    weibull_k1: float = 1.5  # 第一个 Weibull 成分形状参数
    weibull_k2: float = 3.0  # 第二个 Weibull 成分形状参数
    weibull_lambda1_base: float = 1.0  # 第一个 Weibull 成分基准尺度
    weibull_lambda2_base: float = 2.0  # 第二个 Weibull 成分基准尺度
    mixture_weight_base: float = 0.5  # 混合权重基准值
    
    # 高斯混合参数
    gaussian_mu1_base: float = 2.0  # 第一个高斯成分均值基准
    gaussian_mu2_base: float = 4.0  # 第二个高斯成分均值基准
    gaussian_sigma1: float = 0.5  # 第一个高斯成分标准差
    gaussian_sigma2: float = 0.8  # 第二个高斯成分标准差
    gaussian_mixture_weight: float = 0.5  # 高斯混合权重
    
    # 协变量效应系数
    beta_linear: float = 0.5  # 线性协变量效应系数
    beta_nonlinear: float = 0.3  # 非线性协变量效应系数
    
    def get_censoring_lambda(self) -> float:
        """
        根据目标删失率计算指数删失分布的参数 lambda
        
        原理:
        假设删失时间服从指数分布 C ~ Exp(λ_c), 则:
        P(C > t) = exp(-λ_c * t)
        
        为使平均删失率达到目标值，近似计算:
        λ_c ≈ -log(1 - censoring_rate) / E[T]
        
        返回:
            指数删失分布的参数 lambda
        """
        return -np.log(1 - self.censoring_rate + 1e-6) / 3.0


@dataclass
class ModelConfig:
    """
    模型超参数配置
    
    【命名规范】
    - 网络架构参数: snake_case (hidden_dims, dropout, n_time_bins 等)
    - 训练参数: 大写 (LR, BATCH_SIZE, EPOCHS 等)
    - 与代码中 self.config.get() 保持一致
    
    【蒙特卡洛算法选项】(仅适用于 Flow 系列模型)
    - use_mc: 是否使用蒙特卡洛采样法计算生存函数
      - False (默认): 密度积分法，通过逆流积分计算 Jacobian
      - True: 蒙特卡洛采样法，通过正向 ODE 采样估计 S(t)
    - mc_samples: 蒙特卡洛采样数量，越大越精确但计算越慢
    """
    configs: Dict[str, Dict[str, Any]] = field(default_factory=lambda: {
        'LinearCoxPH': {
            'penalizer': 1e-5,  # L2 正则化系数
        },
        'DeepSurv': {
            'hidden_dims': [16, 16],  # 隐藏层维度
            'dropout': 0.1,  # Dropout 比率
            'LR': 8e-5,  # 学习率
            'BATCH_SIZE': 64,  # 批大小
            'EPOCHS': 200,  # 训练轮数
            'PATIENCE': 10,  # 早停耐心值
            'WEIGHT_DECAY': 1e-5,  # 权重衰减
        },
        'WeibullAFT': {
            'hidden_dims': [32, 16],  # 隐藏层维度
            'dropout': 0.0,  # Dropout 比率
            'LR': 5e-5,  # 学习率
            'BATCH_SIZE': 64,  # 批大小
            'EPOCHS': 200,  # 训练轮数
            'PATIENCE': 15,  # 早停耐心值
            'WEIGHT_DECAY': 1e-5,  # 权重衰减
        },
        'RSF': {
            'n_estimators': 100,  # 决策树数量
            'max_depth': 10,  # 最大深度 (限制深度以减少大样本内存占用)
            'min_samples_split': 10,  # 节点分裂最小样本数
            'min_samples_leaf': 5,  # 叶节点最小样本数
            'random_state': 42,  # 随机种子
            'n_jobs': 4,  # 并行数 (限制以减少内存峰值)
        },
        'DeepHit': {
            'hidden_dims': [32, 16, 8],  # 隐藏层维度
            'dropout': 0.1,  # Dropout 比率
            'n_time_bins': 50,  # 时间离散化箱数
            'LR': 5e-8,  # 学习率
            'BATCH_SIZE': 64,  # 批大小
            'EPOCHS': 200,  # 训练轮数
            'PATIENCE': 10,  # 早停耐心值
            'WEIGHT_DECAY': 1e-5,  # 权重衰减
        },
        'FlowSurv': {
            'vf_hidden_dims': [16, 16],  # 向量场网络隐藏层维度
            'tau_dim': 32,  # 时间嵌入维度
            'encoder_hidden': [32, 16],  # 特征编码器隐藏层
            'film_hidden': [16],  # FiLM 调制网络隐藏层

            'dropout': 0.1,  # Dropout 比率

            'LR': 8e-5,  # 学习率
            'BATCH_SIZE': 64,  # 批大小
            'EPOCHS': 200,  # 训练轮数
            'PATIENCE': 10,  # 早停耐心值
            'WEIGHT_DECAY': 1e-5,  # 权重衰减
            'sigma': 0.1,  # 流模型噪声标准差

            'n_samples': 512,  # 预测t时采样数 (use_mc=False 时使用)
            'ode_steps': 100,  # ODE 积分步数
            'solver': 'rk4',  # ODE 求解器: euler/rk4/dopri5

            'truncated_samples': 32,  # 截断采样数 (删失样本训练)
            'weight_event': 1.0,  # 事件样本权重
            'weight_censored': 1.0,  # 删失样本权重

            'use_mc': False,  # 是否使用蒙特卡洛采样法 (True时用mc_samples)
            'mc_samples': 5000,  # 蒙特卡洛采样数量 (增加到5000以提高精度)
        },
        'GumbelFlowSurv': {
            'vf_hidden_dims': [16, 16],  # 向量场网络隐藏层维度
            'tau_dim': 32,  # 时间嵌入维度
            'encoder_hidden': [32, 16],  # 特征编码器隐藏层
            'film_hidden': [16],  # FiLM 调制网络隐藏层
            'gumbel_head_hidden': [16],  # Gumbel 头隐藏层

            'dropout': 0.1,  # Dropout 比率
            'LR': 8e-5,  # 学习率
            'BATCH_SIZE': 64,  # 批大小
            'EPOCHS': 200,  # 训练轮数
            'PATIENCE': 10,  # 早停耐心值
            'WEIGHT_DECAY': 1e-5,  # 权重衰减
            'sigma': 0.1,  # 流模型噪声标准差
            'GUMBEL_LR': 4e-5,  # Gumbel 头学习率
            'GUMBEL_EPOCHS': 200,  # Gumbel 头训练轮数
            'GUMBEL_BATCH_SIZE': 64,  # Gumbel 头批大小
            'GUMBEL_PATIENCE': 15,  # Gumbel 头早停耐心值
            'GUMBEL_WEIGHT_DECAY': 1e-5,  # Gumbel 头权重衰减

            'n_samples': 512,  # 预测时采样数 (use_mc=False 时使用)
            'ode_steps': 100,  # ODE 积分步数
            'solver': 'rk4',  # ODE 求解器: euler/rk4/dopri5

            'truncated_samples': 32,  # 截断采样数 (删失样本训练)
            'weight_event': 1.0,  # 事件样本权重
            'weight_censored': 1.0,  # 删失样本权重

            'use_mc': False,  # 是否使用蒙特卡洛采样法 (True时用mc_samples)
            'mc_samples': 5000,  # 蒙特卡洛采样数量 (增加到5000以提高精度)
        },
        'MultiGumbelFlowSurv': {
            'vf_hidden_dims': [16, 16],  # 向量场网络隐藏层维度
            'tau_dim': 32,  # 时间嵌入维度
            'encoder_hidden': [32, 16],  # 特征编码器隐藏层
            'film_hidden': [16],  # FiLM 调制网络隐藏层
            'gumbel_head_hidden': [16],  # Gumbel 头隐藏层

            'dropout': 0.1,  # Dropout 比率
            'LR': 8e-5,  # 学习率
            'BATCH_SIZE': 64,  # 批大小
            'EPOCHS': 200,  # 训练轮数
            'PATIENCE': 10,  # 早停耐心值
            'WEIGHT_DECAY': 1e-5,  # 权重衰减
            'sigma': 0.1,  # 流模型噪声标准差
            'GUMBEL_LR': 5e-5,  # Gumbel 头学习率
            'GUMBEL_EPOCHS': 200,  # Gumbel 头训练轮数
            'GUMBEL_BATCH_SIZE': 64,  # Gumbel 头批大小
            'GUMBEL_PATIENCE': 15,  # Gumbel 头早停耐心值
            'GUMBEL_WEIGHT_DECAY': 1e-5,  # Gumbel 头权重衰减

            'n_samples': 512,  # 预测时采样数 (use_mc=False 时使用)
            'ode_steps': 100,  # ODE 积分步数
            'solver': 'euler',  # ODE 求解器: euler/rk4/dopri5
            'truncated_samples': 32,  # 截断采样数 (删失样本训练)
            'weight_event': 1.0,  # 事件样本权重
            'weight_censored': 1.0,  # 删失样本权重

            'use_joint_loss': False,  # 是否使用联合损失
            'use_mc': False,  # 是否使用蒙特卡洛采样法 (True时用mc_samples)
            'mc_samples': 5000,  # 蒙特卡洛采样数量 (增加到5000以提高精度)
        },
    })


@dataclass
class PlotConfig:
    """绘图配置"""
    figsize_single: tuple = (8, 6)  # 单图尺寸
    figsize_multi: tuple = (14, 10)  # 多图尺寸
    dpi: int = 150  # 分辨率
    font_size: int = 12  # 字体大小
    n_representative_samples: int = 5  # 代表性样本数量
    time_grid_points: int = 100  # 时间网格点数
    hazard_surface_points: int = 50  # 风险曲面点数
    colors: Dict[str, str] = field(default_factory=lambda: {
        'true': '#2E86AB',  # 真实曲线颜色
        'LinearCoxPH': '#A23B72',
        'DeepSurv': '#F18F01',
        'WeibullAFT': '#C73E1D',
        'RSF': '#3B1F2B',
        'DeepHit': '#95C623',
        'FlowSurv': '#1B998B',
        'GumbelFlowSurv': '#E84855',
        'MultiGumbelFlowSurv': '#6B5B95',
    })
    linestyles: Dict[str, str] = field(default_factory=lambda: {
        'true': '-',  # 真实曲线线型
        'LinearCoxPH': '--',
        'DeepSurv': '-.',
        'WeibullAFT': ':',
        'RSF': '--',
        'DeepHit': '-.',
        'FlowSurv': '-',
        'GumbelFlowSurv': '-',
        'MultiGumbelFlowSurv': '-',
    })
    save_format: str = 'png'  # 保存格式
    save_dir: str = 'results/figures'  # 保存目录


@dataclass 
class ExperimentGroup:
    """单个实验组配置"""
    name: str  # 实验组名称
    description: str  # 实验描述
    data_config: DataConfig  # 数据配置
    is_nph: bool = False  # 是否为非比例风险场景
    is_high_censoring: bool = False  # 是否为高删失场景
    is_small_sample: bool = False  # 是否为小样本场景
    is_high_noise: bool = False  # 是否为高噪声场景


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
    base_seed: int = 42  # 基础随机种子
    output_dir: str = 'results'  # 输出目录
    save_predictions: bool = True  # 是否保存预测结果
    save_models: bool = False  # 是否保存模型
    n_repeats: int = 10  # 重复实验次数
    val_ratio: float = 0.1  # 验证集比例
    test_ratio: float = 0.2  # 测试集比例
    
    groups: List[ExperimentGroup] = field(default_factory=_create_default_groups)  # 实验组列表
    
    time_quantiles: List[float] = field(default_factory=lambda: [0.25, 0.5, 0.75])  # 时间分位数
    ipcw_max_weight: float = 20.0  # IPCW 最大权重
    calibration_bins: int = 10  # 校准箱数


def get_model_config(model_name: str, model_config: ModelConfig = None) -> Dict[str, Any]:
    """获取指定模型的超参数配置"""
    if model_config is None:
        model_config = ModelConfig()
    return model_config.configs.get(model_name, {})


@dataclass
class GlobalConfig:
    """全局配置汇总"""
    data: DataConfig = field(default_factory=DataConfig)  # 数据配置
    model: ModelConfig = field(default_factory=ModelConfig)  # 模型配置
    plot: PlotConfig = field(default_factory=PlotConfig)  # 绘图配置
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)  # 实验配置
    
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
