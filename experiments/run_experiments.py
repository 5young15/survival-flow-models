"""
生存分析模型实验运行脚本

功能说明:
- 提供统一的模型训练器接口 (ModelTrainer)
- 支持 PyTorch 模型 (PyTorchModelTrainer) 和随机生存森林 (RSFTrainer)
- 实现完整的实验流程：数据生成、模型训练、预测、评估
- 支持模型检查点保存与加载
- 批量运行所有实验并聚合结果

作者：Statistical Modeling Team
日期：2026
"""

import os
import sys
import json
import pickle
import csv
import numpy as np
import torch
from typing import Dict, List, Any, Optional, Tuple, Union
from datetime import datetime
from tqdm import tqdm

# 添加项目根目录到 Python 路径，以便导入其他模块
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from .config import (
    GlobalConfig, CONFIG, get_model_config,
    ExperimentGroup, DataConfig
)
from .data_generation import (
    SurvivalDataGenerator, SurvivalData, generate_experiment_data
)
from .metrics import (
    compute_all_metrics, metrics_to_dict, MetricsResult,
    concordance_index_fast, integrated_brier_score
)


class ModelTrainer:
    """
    模型训练器基类 (抽象接口)
    
    功能:
    - 定义模型训练器的标准接口
    - 管理模型配置和设备
    - 子类需实现：create_model, train, predict
    
    属性:
        model_name (str): 模型名称
        config (GlobalConfig): 全局配置对象
        model_config (Dict): 模型特定超参数配置
        device (torch.device): 计算设备 (CPU/GPU)
        model: 模型实例
    """

    def __init__(self, model_name: str, config: GlobalConfig):
        """
        初始化模型训练器
        
        参数:
            model_name: 模型名称标识符
            config: 全局配置对象
        """
        self.model_name = model_name
        self.config = config
        self.model_config = get_model_config(model_name, config.model)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None

    def create_model(self, in_dim: int):
        """
        创建模型实例 (纯虚函数)
        
        参数:
            in_dim: 输入特征维度
        
        异常:
            NotImplementedError: 子类必须实现此方法
        """
        raise NotImplementedError

    def train(self, train_data: SurvivalData, val_data: SurvivalData):
        """
        训练模型 (纯虚函数)
        
        参数:
            train_data: 训练数据集
            val_data: 验证数据集
        
        异常:
            NotImplementedError: 子类必须实现此方法
        """
        raise NotImplementedError

    def predict(self, test_data: SurvivalData, time_grid: Union[np.ndarray, torch.Tensor],
                use_mc: bool = False) -> Dict[str, torch.Tensor]:
        """
        模型预测 (纯虚函数)
        
        参数:
            test_data: 测试数据集
            time_grid: 时间点网格用于预测生存函数
            use_mc: 是否使用蒙特卡洛采样法 (仅流模型有效)
        
        返回:
            包含预测结果的字典，包括：
            - risk_scores: 风险评分
            - survival: 生存函数
            - medians: 中位生存时间
            - hazard: 风险函数 (可选)
            - density: 概率密度函数 (可选)
        
        异常:
            NotImplementedError: 子类必须实现此方法
        """
        raise NotImplementedError


class PyTorchModelTrainer(ModelTrainer):
    """
    PyTorch 模型训练器
    
    功能:
    - 支持多种 PyTorch 生存分析模型 (CoxPH, DeepSurv, FlowSurv, GumbelFlowSurv 等)
    - 实现两阶段训练策略 (Weibull 预训练 + Flow 主训练)
    - 支持模型检查点保存与加载
    - 自动早停和最优模型选择
    
    继承自:
        ModelTrainer: 模型训练器基类
    """

    def create_model(self, in_dim: int):
        """
        创建 PyTorch 模型实例
        
        参数:
            in_dim: 输入特征维度
        
        返回:
            模型实例
        
        异常:
            ValueError: 未知的模型名称
        """
        # 导入各类模型
        from models.baselines.coxph import LinearCoxPH
        from models.baselines.deepsurv import DeepSurv
        from models.baselines.deephit import DeepHit
        from models.baselines.weibullAFT import WeibullAFT
        from models.flowmodel.base_flow import FlowSurv
        from models.flowmodel.gumbel_flow import GumbelFlowSurv
        from models.flowmodel.multi_gumbel_flow import MultiGumbelFlowSurv

        model_classes = {
            'LinearCoxPH': LinearCoxPH,
            'DeepSurv': DeepSurv,
            'WeibullAFT': WeibullAFT,
            'DeepHit': DeepHit,
            'FlowSurv': FlowSurv,
            'GumbelFlowSurv': GumbelFlowSurv,
            'GumbelFlow': GumbelFlowSurv,
            'GFM': GumbelFlowSurv,
            'MultiGumbelFlowSurv': MultiGumbelFlowSurv,
            'MGFM': MultiGumbelFlowSurv,
        }

        if self.model_name not in model_classes:
            raise ValueError(f"Unknown model: {self.model_name}")

        # 创建模型并移动到计算设备
        self.model = model_classes[self.model_name](
            in_dim=in_dim,
            config=self.model_config
        ).to(self.device)

        return self.model

    def get_checkpoint_path(self, checkpoint_dir: str, repeat_id: int) -> str:
        """
        获取模型检查点保存路径
        
        参数:
            checkpoint_dir: 检查点根目录
            repeat_id: 实验重复次数 ID
        
        返回:
            完整的检查点文件路径
        """
        return os.path.join(checkpoint_dir, self.model_name, f"repeat_{repeat_id}", "best_model.pt")

    def save_checkpoint(self, checkpoint_path: str, best_val_loss: float,
                        time_scaler_mean: float, time_scaler_std: float, is_log_space: bool):
        """
        保存最优模型检查点
        
        保存内容:
        - 模型参数 (state_dict)
        - 最优验证损失
        - 时间标准化参数 (均值、标准差、是否对数空间)
        - 基线风险函数参数 (如适用)
        
        参数:
            checkpoint_path: 检查点文件路径
            best_val_loss: 最优验证损失值
            time_scaler_mean: 时间标准化均值
            time_scaler_std: 时间标准化标准差
            is_log_space: 是否在对数空间进行标准化
        """
        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'best_val_loss': best_val_loss,
            'time_scaler_mean': time_scaler_mean,
            'time_scaler_std': time_scaler_std,
            'is_log_space': is_log_space,
            'model_name': self.model_name,
            'baseline_times': getattr(self.model, '_baseline_times', None),
            'baseline_cum_haz': getattr(self.model, '_baseline_cum_haz', None),
            'train_times': getattr(self.model, '_train_times', None),
            'train_events': getattr(self.model, '_train_events', None),
            'train_log_haz': getattr(self.model, '_train_log_haz', None),
        }
        torch.save(checkpoint, checkpoint_path)

    def load_checkpoint(self, checkpoint_path: str) -> bool:
        """
        加载模型检查点
        
        参数:
            checkpoint_path: 检查点文件路径
        
        返回:
            bool: 加载成功返回 True，否则返回 False
        """
        if not os.path.exists(checkpoint_path):
            return False

        # 加载检查点
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        # 恢复模型参数
        self.model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        # 恢复时间标准化参数
        self.model.set_time_scaler(
            checkpoint['time_scaler_mean'],
            checkpoint['time_scaler_std'],
            checkpoint['is_log_space']
        )
        # 恢复基线风险函数相关参数
        if hasattr(self.model, '_baseline_times'):
            self.model._baseline_times = checkpoint.get('baseline_times', None)
            self.model._baseline_cum_haz = checkpoint.get('baseline_cum_haz', None)
            self.model._train_times = checkpoint.get('train_times', None)
            self.model._train_events = checkpoint.get('train_events', None)
            self.model._train_log_haz = checkpoint.get('train_log_haz', None)
        return True

    def train(self, train_data: SurvivalData, val_data: SurvivalData,
              checkpoint_dir: Optional[str] = None, repeat_id: int = 0) -> Tuple[float, bool]:
        """
        训练模型 (支持两阶段训练策略)
        
        训练流程:
        1. 检查并加载已有检查点 (如存在)
        2. 数据预处理：转换为 Tensor 并设置时间标准化参数
        3. 两阶段训练 (仅 GumbelFlowSurv):
           - Weibull 预训练阶段：稳定初始化
           - Flow 主训练阶段：流模型优化
        4. 拟合基线风险函数
        5. 保存最优模型检查点
        
        参数:
            train_data: 训练数据集 (包含 features, times, events)
            val_data: 验证数据集
            checkpoint_dir: 检查点保存目录
            repeat_id: 实验重复次数 ID
        
        返回:
            Tuple[float, bool]: (最优验证损失，是否从检查点加载)
        """
        if self.model is None:
            raise RuntimeError("Model not created. Call create_model first.")

        checkpoint_path = self.get_checkpoint_path(checkpoint_dir, repeat_id) if checkpoint_dir else None

        if checkpoint_path and self.load_checkpoint(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            return checkpoint['best_val_loss'], True

        model_config = self.model_config

        # 统一为 Tensor 并移动到设备，避免重复创建
        def to_device_tensor(x, dtype=torch.float32):
            """将输入转换为指定设备和类型的 Tensor"""
            if isinstance(x, torch.Tensor):
                return x.to(self.device, dtype=dtype)
            return torch.tensor(x, dtype=dtype, device=self.device)

        times_tensor = to_device_tensor(train_data.times)
        events_tensor = to_device_tensor(train_data.events)
        features_tensor = to_device_tensor(train_data.features)

        time_mean = torch.log(times_tensor + 1).mean().item()
        time_std = torch.log(times_tensor + 1).std().item()
        is_log_space = True
        self.model.set_time_scaler(time_mean, time_std, is_log_space=True)

        if hasattr(self.model, 'init_gumbel_params'):
            self.model.init_gumbel_params(times_tensor, events_tensor)

        val_times = to_device_tensor(val_data.times)
        val_events = to_device_tensor(val_data.events)
        val_features = to_device_tensor(val_data.features)

        def train_stage(stage: str, epochs: int, lr: float, batch_size: int, patience: int, weight_decay: float):
            """
            单阶段训练函数
            
            参数:
                stage: 训练阶段名称 ('weibull' 或 'flow')
                epochs: 训练轮数
                lr: 学习率
                batch_size: 批次大小
                patience: 早停耐心值
                weight_decay: L2 正则化系数
            
            返回:
                最优验证损失
            """
            if hasattr(self.model, 'set_stage'):
                self.model.set_stage(stage)

            params = [p for p in self.model.parameters() if p.requires_grad]
            optimizer = torch.optim.Adam(
                params,
                lr=lr,
                weight_decay=weight_decay
            )

            best_val_loss = float('inf')
            patience_counter = 0
            best_state = None

            n_samples = len(train_data.times)

            for _ in range(epochs):
                self.model.train()
                indices = torch.randperm(n_samples, device=self.device)

                for i in range(0, n_samples, batch_size):
                    batch_idx = indices[i:i + batch_size]

                    batch_features = features_tensor.index_select(0, batch_idx)
                    batch_times = times_tensor.index_select(0, batch_idx)
                    batch_events = events_tensor.index_select(0, batch_idx)

                    optimizer.zero_grad(set_to_none=True)
                    loss, _ = self.model.forward_loss(
                        batch_features, batch_times, batch_events
                    )
                    loss.backward()
                    optimizer.step()

                self.model.eval()
                with torch.no_grad():
                    val_loss_tensor, _ = self.model.forward_loss(
                        val_features, val_times, val_events
                    )
                    val_loss = val_loss_tensor.item()

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                else:
                    patience_counter += 1

                if patience_counter >= patience:
                    break

            if best_state is not None:
                self.model.load_state_dict(best_state)
            return best_val_loss

        if self.model_name in {'GumbelFlowSurv', 'GumbelFlow', 'GFM'} and hasattr(self.model, 'set_stage'):
            gumbel_epochs = model_config.get('GUMBEL_EPOCHS', 200)
            if gumbel_epochs > 0:
                train_stage(
                    stage='gumbel',
                    epochs=gumbel_epochs,
                    lr=model_config.get('GUMBEL_LR', 5e-8),
                    batch_size=model_config.get('GUMBEL_BATCH_SIZE', 64),
                    patience=model_config.get('GUMBEL_PATIENCE', 15),
                    weight_decay=model_config.get('GUMBEL_WEIGHT_DECAY', 1e-5),
                )

            best_val_loss = train_stage(
                stage='flow',
                epochs=model_config.get('EPOCHS', 200),
                lr=model_config.get('LR', 3e-4),
                batch_size=model_config.get('BATCH_SIZE', 64),
                patience=model_config.get('PATIENCE', 15),
                weight_decay=model_config.get('WEIGHT_DECAY', 1e-5),
            )
        elif self.model_name in {'MultiGumbelFlowSurv', 'MGFM'} and hasattr(self.model, 'set_stage'):
            gumbel_epochs = model_config.get('GUMBEL_EPOCHS', 200)
            if gumbel_epochs > 0:
                train_stage(
                    stage='gumbel',
                    epochs=gumbel_epochs,
                    lr=model_config.get('GUMBEL_LR', 5e-5),
                    batch_size=model_config.get('GUMBEL_BATCH_SIZE', 64),
                    patience=model_config.get('GUMBEL_PATIENCE', 15),
                    weight_decay=model_config.get('GUMBEL_WEIGHT_DECAY', 1e-5),
                )

            best_val_loss = train_stage(
                stage='flow',
                epochs=model_config.get('EPOCHS', 200),
                lr=model_config.get('LR', 1e-4),
                batch_size=model_config.get('BATCH_SIZE', 64),
                patience=model_config.get('PATIENCE', 10),
                weight_decay=model_config.get('WEIGHT_DECAY', 1e-5),
            )
        else:
            best_val_loss = train_stage(
                stage='flow',
                epochs=model_config.get('EPOCHS', 200),
                lr=model_config.get('LR', 3e-4),
                batch_size=model_config.get('BATCH_SIZE', 64),
                patience=model_config.get('PATIENCE', 15),
                weight_decay=model_config.get('WEIGHT_DECAY', 1e-5),
            )

        if hasattr(self.model, '_fit_baseline_hazard'):
            with torch.no_grad():
                train_log_haz = self.model.predict_risk(features_tensor)
            self.model._fit_baseline_hazard(
                times_tensor, events_tensor, train_log_haz
            )

        if checkpoint_path:
            self.save_checkpoint(checkpoint_path, best_val_loss, time_mean, time_std, is_log_space)

        return best_val_loss, False

    def predict(self, test_data: SurvivalData, time_grid: Union[np.ndarray, torch.Tensor],
                use_mc: bool = False) -> Dict[str, torch.Tensor]:
        """
        模型预测
        
        功能:
        - 统一处理 Tensor 和 Numpy 输入
        - 根据模型类型调用不同的预测方法
        - 返回完整的预测结果 (风险评分、生存函数、中位时间、风险函数、密度函数)
        
        参数:
            test_data: 测试数据集
            time_grid: 时间点网格用于预测生存函数
            use_mc: 是否使用蒙特卡洛采样法 (仅流模型有效)
        
        返回:
            包含预测结果的字典:
            - risk_scores: 风险评分 (越高风险越大)
            - survival: 生存函数 S(t|x)
            - medians: 中位生存时间
            - hazard: 风险函数 h(t|x) (如可用)
            - density: 概率密度函数 f(t|x) (如可用)
        """
        if self.model is None:
            raise RuntimeError("Model not trained.")

        self.model.eval()
        features = test_data.features
        if not isinstance(features, torch.Tensor):
            features = torch.tensor(features, dtype=torch.float32).to(self.device)
        else:
            features = features.to(self.device)

        if not isinstance(time_grid, torch.Tensor):
            time_grid = torch.tensor(time_grid, dtype=torch.float32).to(self.device)
        else:
            time_grid = time_grid.to(self.device)

        with torch.no_grad():
            if use_mc and hasattr(self.model, 'predict_survival_metrics_mc'):
                results = self.model.predict_survival_metrics_mc(features, time_grid)
                risk_scores = self.model.predict_risk_mc(features)
                survival = results['survival']
                log_hazard = results.get('log_hazard')
                log_density = None
                pred_medians = self.model.predict_time_mc(features, mode='median')
            elif hasattr(self.model, 'predict_survival_metrics'):
                results = self.model.predict_survival_metrics(features, time_grid)
                risk_scores = self.model.predict_risk(features)
                survival = results['survival']
                log_hazard = results.get('log_hazard')
                log_density = results.get('log_density')
                pred_medians = self.model.predict_time(features, mode='median')
            else:
                risk_scores = self.model.predict_risk(features)
                survival = self.model.predict_survival_function(features, time_grid)
                pred_medians = self.model.predict_time(features, mode='median')

                try:
                    log_hazard = self.model.compute_hazard_rate(features, time_grid)
                except Exception:
                    log_hazard = None

                try:
                    if hasattr(self.model, 'compute_density'):
                        density_raw = self.model.compute_density(features, time_grid)
                        if density_raw is not None:
                            log_density = torch.log(torch.clamp(density_raw, min=1e-100))
                        else:
                            log_density = None
                    else:
                        log_density = None
                except Exception:
                    log_density = None

        return {
            'risk_scores': risk_scores,
            'survival': survival,
            'medians': pred_medians,
            'log_hazard': log_hazard,
            'log_density': log_density
        }


class RSFTrainer(ModelTrainer):
    """
    随机生存森林 (Random Survival Forest) 训练器
    
    功能:
    - 封装 scikit-survival 的随机生存森林实现
    - 使用 pickle 保存/加载模型
    - 支持检查点机制
    
    继承自:
        ModelTrainer: 模型训练器基类
    """

    def create_model(self, in_dim: int):
        """
        创建随机生存森林模型
        
        参数:
            in_dim: 输入特征维度
        
        返回:
            RSF 模型实例
        """
        from models.baselines.RSF import RandomSurvivalForestWrapper

        self.model = RandomSurvivalForestWrapper(
            in_dim=in_dim,
            config=self.model_config
        )
        return self.model

    def get_checkpoint_path(self, checkpoint_dir: str, repeat_id: int) -> str:
        """
        获取模型检查点路径
        
        参数:
            checkpoint_dir: 检查点根目录
            repeat_id: 实验重复次数 ID
        
        返回:
            完整的检查点文件路径 (.pkl 格式)
        """
        return os.path.join(checkpoint_dir, self.model_name, f"repeat_{repeat_id}", "best_model.pkl")

    def save_checkpoint(self, checkpoint_path: str):
        """
        保存 RSF 模型 (使用 pickle 序列化)
        
        参数:
            checkpoint_path: 检查点文件路径
        """
        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
        with open(checkpoint_path, 'wb') as f:
            pickle.dump(self.model, f)

    def load_checkpoint(self, checkpoint_path: str) -> bool:
        """
        加载 RSF 模型
        
        参数:
            checkpoint_path: 检查点文件路径
        
        返回:
            bool: 加载成功返回 True，否则返回 False
        """
        if not os.path.exists(checkpoint_path):
            return False
        with open(checkpoint_path, 'rb') as f:
            self.model = pickle.load(f)
        return True

    def train(self, train_data: SurvivalData, val_data: SurvivalData,
              checkpoint_dir: Optional[str] = None, repeat_id: int = 0) -> Tuple[float, bool]:
        """
        训练随机生存森林模型
        
        参数:
            train_data: 训练数据集
            val_data: 验证数据集 (RSF 不使用)
            checkpoint_dir: 检查点目录
            repeat_id: 实验重复次数 ID
        
        返回:
            Tuple[float, bool]: (0.0, 是否从检查点加载)
        """
        checkpoint_path = self.get_checkpoint_path(checkpoint_dir, repeat_id) if checkpoint_dir else None

        if checkpoint_path and self.load_checkpoint(checkpoint_path):
            return 0.0, True

        # RSF 需要 numpy 数据，如果输入是 tensor 则转换
        X = train_data.features
        y_time = train_data.times
        y_event = train_data.events

        if isinstance(X, torch.Tensor):
            X = X.detach().cpu().numpy()
        if isinstance(y_time, torch.Tensor):
            y_time = y_time.detach().cpu().numpy()
        if isinstance(y_event, torch.Tensor):
            y_event = y_event.detach().cpu().numpy()

        # 拟合模型
        self.model.fit(X, y_time, y_event)

        if checkpoint_path:
            self.save_checkpoint(checkpoint_path)

        return 0.0, False

    def predict(self, test_data: SurvivalData, time_grid: Union[np.ndarray, torch.Tensor],
                _use_mc: bool = False) -> Dict[str, torch.Tensor]:
        """
        RSF 模型预测
        
        参数:
            test_data: 测试数据集
            time_grid: 时间点网格
            _use_mc: 未使用 (RSF不支持蒙特卡洛采样)
        
        返回:
            包含预测结果的字典
        """
        features = test_data.features
        if isinstance(time_grid, np.ndarray):
            time_grid = torch.from_numpy(time_grid).float().to(features.device if isinstance(features, torch.Tensor) else 'cpu')

        risk_scores = self.model.predict_risk(features)
        survival = self.model.predict_survival_function(features, time_grid)
        pred_medians = self.model.predict_time(features)

        try:
            log_hazard = self.model.compute_hazard_rate(features, time_grid)
        except Exception:
            log_hazard = None

        return {
            'risk_scores': risk_scores,
            'survival': survival,
            'medians': pred_medians,
            'log_hazard': log_hazard,
            'log_density': None
        }


def create_trainer(model_name: str, config: GlobalConfig) -> ModelTrainer:
    """
    创建模型训练器工厂函数
    
    参数:
        model_name: 模型名称 ('RSF' 或其他 PyTorch 模型)
        config: 全局配置对象
    
    返回:
        ModelTrainer: 对应的训练器实例
    """
    if model_name == 'RSF':
        return RSFTrainer(model_name, config)
    else:
        return PyTorchModelTrainer(model_name, config)


def run_single_experiment(
    group: ExperimentGroup,
    model_name: str,
    repeat_id: int,
    config: GlobalConfig,
    checkpoint_dir: Optional[str] = None,
    device: Optional[torch.device] = None,
    use_mc: bool = False
) -> Tuple[MetricsResult, Dict[str, np.ndarray], Dict[str, Any]]:
    """
    运行单次实验 (完整流程)
    
    实验流程:
    1. 设置随机种子以保证可复现性
    2. 生成生存数据
    3. 划分训练/验证/测试集
    4. 创建并训练模型
    5. 进行预测
    6. 计算评估指标
    
    参数:
        group: 实验组配置
        model_name: 模型名称
        repeat_id: 重复实验 ID
        config: 全局配置对象
        checkpoint_dir: 检查点目录
        device: 计算设备
        use_mc: 是否使用蒙特卡洛采样法 (仅流模型有效)
    
    返回:
        Tuple 包含:
        - metrics: 评估指标结果
        - predictions: 预测结果字典
        - info: 实验信息字典
    """
    
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 设置随机种子 (保证可复现性)
    seed = group.data_config.random_seed + repeat_id * 100
    np.random.seed(seed)
    torch.manual_seed(seed)

    data_config = group.data_config
    data_config.random_seed = seed

    # 生成数据
    generator = SurvivalDataGenerator(data_config)
    full_data = generator.generate(n=data_config.n_samples, seed=seed)

    # 计算测试集划分
    n = len(full_data.times)
    test_ratio = config.experiment.test_ratio
    val_ratio = config.experiment.val_ratio
    n_test = int(n * test_ratio)
    n_val = int((n - n_test) * val_ratio)

    # 随机划分数据集
    indices = np.random.permutation(n)
    test_idx = indices[:n_test]
    val_idx = indices[n_test:n_test + n_val]
    train_idx = indices[n_test + n_val:]

    # 计算真实中位生存时间 (用于评估)
    true_medians_np = generator.generator.median(full_data.features[test_idx])

    # 将数据移动到计算设备
    full_data = full_data.to(device)
    true_medians = torch.from_numpy(true_medians_np).float().to(device)

    def split_data(data: SurvivalData, idx: np.ndarray) -> SurvivalData:
        """
        数据划分辅助函数
        
        参数:
            data: 完整数据集
            idx: 索引数组
        
        返回:
            划分后的子集
        """
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

    # 创建并训练模型
    trainer = create_trainer(model_name, config)
    trainer.create_model(in_dim=train_data.features.shape[1])

    group_checkpoint_dir = os.path.join(checkpoint_dir, group.name) if checkpoint_dir else None
    best_val_loss, from_checkpoint = trainer.train(train_data, val_data, group_checkpoint_dir, repeat_id)

    # 预测
    time_grid = full_data.time_grid
    predictions = trainer.predict(test_data, time_grid, use_mc=use_mc)

    # 计算评估指标 (确保传入的是对数空间)
    eps = 1e-100
    
    # 提取预测结果并处理 None 值
    pred_hazard = predictions.get('log_hazard')
    pred_density_raw = predictions.get('log_density')
    
    # 安全地计算 exp(log_density)
    if pred_density_raw is not None:
        pred_density = torch.exp(pred_density_raw)
    else:
        pred_density = None

    metrics = compute_all_metrics(
        times=test_data.times,
        events=test_data.events,
        risk_scores=predictions['risk_scores'],
        pred_survival=predictions['survival'],
        pred_medians=predictions['medians'],
        time_grid=time_grid,
        true_hazard=torch.log(torch.clamp(test_data.true_hazard, min=eps)) if test_data.true_hazard is not None else None,
        true_density=test_data.true_density if test_data.true_density is not None else None,
        true_survival=test_data.true_survival,
        true_medians=true_medians,
        pred_hazard=pred_hazard,
        pred_density=pred_density,
        quantiles=config.experiment.time_quantiles,
        max_weight=config.experiment.ipcw_max_weight
    )

    # 实验信息记录
    info = {
        'group_name': group.name,
        'model_name': model_name,
        'repeat_id': repeat_id,
        'seed': seed,
        'n_train': len(train_data.times),
        'n_val': len(val_data.times),
        'n_test': len(test_data.times),
        'actual_censoring_rate': 1 - test_data.events.mean(),
        'from_checkpoint': from_checkpoint,
        'best_val_loss': best_val_loss,
    }

    return metrics, predictions, info


def run_all_experiments(
    config: GlobalConfig,
    model_names: Optional[List[str]] = None,
    group_names: Optional[List[str]] = None,
    n_repeats: Optional[int] = None,
    save_results: bool = True,
    output_dir: Optional[str] = None,
    checkpoint_dir: Optional[str] = None,
    device: Optional[torch.device] = None,
    use_mc: bool = False
) -> Dict[str, Any]:
    """
    运行所有实验 (批量执行)
    
    功能:
    - 遍历所有实验组和模型
    - 自动跳过已完成的实验 (通过检查点)
    - 显示进度条
    - 保存结果到 JSON 文件
    
    参数:
        config: 全局配置对象
        model_names: 要运行的模型列表 (默认全部)
        group_names: 要运行的实验组列表 (默认全部)
        n_repeats: 重复次数 (默认使用配置值)
        save_results: 是否保存结果
        output_dir: 输出目录
        checkpoint_dir: 检查点目录
        device: 计算设备
        use_mc: 是否使用蒙特卡洛采样法 (仅流模型有效)
    
    返回:
        包含所有实验结果的字典
    """

    if model_names is None:
        model_names = list(config.model.configs.keys())

    if group_names is None:
        groups = config.experiment.groups
    else:
        groups = [g for g in config.experiment.groups if g.name in group_names]

    if n_repeats is None:
        n_repeats = config.experiment.n_repeats

    if output_dir is None:
        output_dir = config.experiment.output_dir

    if checkpoint_dir is None:
        checkpoint_dir = 'checkpoints'

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)

    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    all_results = {}

    # 计算总实验次数
    total_runs = len(groups) * len(model_names) * n_repeats
    pbar = tqdm(total=total_runs, desc="Running experiments")

    for group in groups:
        group_results = {}

        for model_name in model_names:
            model_results = []

            for repeat_id in range(n_repeats):
                try:
                    metrics, predictions, info = run_single_experiment(
                        group, model_name, repeat_id, config, checkpoint_dir, device, use_mc
                    )

                    model_results.append({
                        'metrics': metrics_to_dict(metrics),
                        'info': info
                    })

                    if info.get('from_checkpoint'):
                        pbar.set_postfix_str(f"loaded from checkpoint")

                except Exception as e:
                    print(f"\nError in {group.name}/{model_name}/repeat_{repeat_id}: {e}")
                    model_results.append({
                        'metrics': None,
                        'error': str(e)
                    })

                pbar.update(1)

            group_results[model_name] = model_results

        all_results[group.name] = group_results

    pbar.close()

    if save_results:
        results_file = os.path.join(output_dir, 'experiment_results.json')

        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False, default=str)

        print(f"\nResults saved to {results_file}")
        
        # 保存为 CSV 格式
        save_results_to_csv(all_results, output_dir)

    return all_results


def run_cv_single_experiment(
    group,
    model_name: str,
    fold_id: int,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
    full_data: 'SurvivalData',
    config: 'GlobalConfig',
    checkpoint_dir: Optional[str] = None,
    device: Optional[torch.device] = None,
    use_mc: bool = False
) -> Tuple[MetricsResult, Dict[str, np.ndarray], Dict[str, Any]]:
    """
    运行单折交叉验证实验
    
    参数:
        group: 实验组配置
        model_name: 模型名称
        fold_id: 折ID
        train_idx: 训练集索引
        val_idx: 验证集索引
        test_idx: 测试集索引
        full_data: 完整数据
        config: 全局配置
        checkpoint_dir: 检查点目录
        device: 计算设备
        use_mc: 是否使用蒙特卡洛采样
    
    返回:
        Tuple 包含指标、预测结果、实验信息
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    seed = group.data_config.random_seed + fold_id
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    def split_data(data: 'SurvivalData', idx: np.ndarray) -> 'SurvivalData':
        idx_tensor = torch.from_numpy(idx).long() if isinstance(idx, np.ndarray) else torch.tensor(idx, dtype=torch.long)
        return SurvivalData(
            features=data.features[idx_tensor],
            times=data.times[idx_tensor],
            events=data.events[idx_tensor],
            true_times=data.true_times[idx_tensor] if data.true_times is not None else None,
            true_hazard=data.true_hazard[idx_tensor] if data.true_hazard is not None else None,
            true_density=data.true_density[idx_tensor] if data.true_density is not None else None,
            true_survival=data.true_survival[idx_tensor] if data.true_survival is not None else None,
            time_grid=data.time_grid,
            feature_names=data.feature_names
        )
    
    train_data = split_data(full_data, train_idx)
    val_data = split_data(full_data, val_idx)
    test_data = split_data(full_data, test_idx)
    
    generator = SurvivalDataGenerator(group.data_config)
    test_features_np = test_data.features.cpu().numpy() if isinstance(test_data.features, torch.Tensor) else test_data.features
    true_medians_np = generator.generator.median(test_features_np)
    true_medians = torch.from_numpy(true_medians_np).float().to(device)
    
    trainer = create_trainer(model_name, config)
    trainer.create_model(in_dim=train_data.features.shape[1])
    
    fold_checkpoint_dir = os.path.join(checkpoint_dir, group.name) if checkpoint_dir else None
    best_val_loss, from_checkpoint = trainer.train(train_data, val_data, fold_checkpoint_dir, fold_id)
    
    time_grid = full_data.time_grid
    predictions = trainer.predict(test_data, time_grid, use_mc=use_mc)
    
    eps = 1e-100
    pred_hazard = predictions.get('log_hazard')
    pred_density_raw = predictions.get('log_density')
    
    if pred_density_raw is not None:
        pred_density = torch.exp(pred_density_raw)
    else:
        pred_density = None
    
    metrics = compute_all_metrics(
        times=test_data.times,
        events=test_data.events,
        risk_scores=predictions['risk_scores'],
        pred_survival=predictions['survival'],
        pred_medians=predictions['medians'],
        time_grid=time_grid,
        true_hazard=torch.log(torch.clamp(test_data.true_hazard, min=eps)) if test_data.true_hazard is not None else None,
        true_density=test_data.true_density if test_data.true_density is not None else None,
        true_survival=test_data.true_survival,
        true_medians=true_medians,
        pred_hazard=pred_hazard,
        pred_density=pred_density,
        quantiles=config.experiment.time_quantiles,
        max_weight=config.experiment.ipcw_max_weight
    )
    
    info = {
        'group_name': group.name,
        'model_name': model_name,
        'fold_id': fold_id,
        'seed': seed,
        'n_train': len(train_data.times),
        'n_val': len(val_data.times),
        'n_test': len(test_data.times),
        'actual_censoring_rate': 1 - test_data.events.mean(),
        'from_checkpoint': from_checkpoint,
        'best_val_loss': best_val_loss,
    }
    
    return metrics, predictions, info


def run_cv_experiments(
    config: 'GlobalConfig',
    model_names: Optional[List[str]] = None,
    group_names: Optional[List[str]] = None,
    n_folds: int = 5,
    save_results: bool = True,
    output_dir: Optional[str] = None,
    checkpoint_dir: Optional[str] = None,
    device: Optional[torch.device] = None,
    use_mc: bool = False
) -> Dict[str, Any]:
    """
    运行K折交叉验证实验
    
    与重复随机实验不同，交叉验证:
    - 使用固定的一份数据
    - 按规则轮流划分训练/验证/测试集
    - 每个样本都会被作为测试集评估一次
    
    参数:
        config: 全局配置对象
        model_names: 要运行的模型列表
        group_names: 要运行的实验组列表
        n_folds: 折数 (默认5)
        save_results: 是否保存结果
        output_dir: 输出目录
        checkpoint_dir: 检查点目录
        device: 计算设备
        use_mc: 是否使用蒙特卡洛采样
    
    返回:
        包含所有实验结果的字典
    """
    from sklearn.model_selection import KFold
    
    if model_names is None:
        model_names = list(config.model.configs.keys())
    
    if group_names is None:
        groups = config.experiment.groups
    else:
        groups = [g for g in config.experiment.groups if g.name in group_names]
    
    if output_dir is None:
        output_dir = config.experiment.output_dir
    
    if checkpoint_dir is None:
        checkpoint_dir = 'cv_checkpoints'
    
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    all_results = {}
    kfold = KFold(n_splits=n_folds, shuffle=True, random_state=config.experiment.base_seed)
    
    total_runs = len(groups) * len(model_names) * n_folds
    pbar = tqdm(total=total_runs, desc=f"Running {n_folds}-fold CV")
    
    for group in groups:
        group_results = {}
        
        seed = group.data_config.random_seed
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        generator = SurvivalDataGenerator(group.data_config)
        full_data = generator.generate(n=group.data_config.n_samples, seed=seed)
        times_np = full_data.times.cpu().numpy() if isinstance(full_data.times, torch.Tensor) else full_data.times
        full_data = full_data.to(device)
        
        for model_name in model_names:
            model_results = []
            
            for fold_id, (train_val_idx, test_idx) in enumerate(kfold.split(times_np)):
                val_ratio = config.experiment.val_ratio
                n_train_val = len(train_val_idx)
                n_val = int(n_train_val * val_ratio)
                
                np.random.seed(seed + fold_id)
                shuffled_idx = np.random.permutation(train_val_idx)
                train_idx = shuffled_idx[n_val:]
                val_idx = shuffled_idx[:n_val]
                
                try:
                    metrics, predictions, info = run_cv_single_experiment(
                        group, model_name, fold_id, train_idx, val_idx, test_idx,
                        full_data, config, checkpoint_dir, device, use_mc
                    )
                    
                    model_results.append({
                        'metrics': metrics_to_dict(metrics),
                        'info': info
                    })
                    
                    if info.get('from_checkpoint'):
                        pbar.set_postfix_str(f"loaded from checkpoint")
                
                except Exception as e:
                    print(f"\nError in {group.name}/{model_name}/fold_{fold_id}: {e}")
                    model_results.append({
                        'metrics': None,
                        'error': str(e)
                    })
                
                pbar.update(1)
            
            group_results[model_name] = model_results
        
        all_results[group.name] = group_results
    
    pbar.close()
    
    if save_results:
        results_file = os.path.join(output_dir, 'cv_results.json')
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False, default=str)
        print(f"\nResults saved to {results_file}")
        save_results_to_csv(all_results, output_dir)
    
    return all_results


def aggregate_results(results: Dict[str, Any]) -> Dict[str, Dict[str, Dict[str, Tuple[float, float]]]]:
    """
    聚合实验结果：计算均值和标准差
    
    参数:
        results: run_all_experiments 返回的原始结果
    
    返回:
        聚合后的结果字典，格式:
        {
            '实验组名': {
                '模型名': {
                    '指标名': (均值，标准差)
                }
            }
        }
    """
    aggregated = {}

    for group_name, group_results in results.items():
        aggregated[group_name] = {}

        for model_name, model_results in group_results.items():
            # 过滤掉失败的实验
            valid_metrics = [r['metrics'] for r in model_results if r.get('metrics') is not None]

            if not valid_metrics:
                continue

            metric_names = list(valid_metrics[0].keys())
            aggregated[group_name][model_name] = {}

            for metric in metric_names:
                # 过滤掉 NaN 值
                values = [m[metric] for m in valid_metrics if metric in m and m[metric] is not None and not (isinstance(m[metric], float) and np.isnan(m[metric]))]
                if values:
                    aggregated[group_name][model_name][metric] = (
                        np.mean(values),
                        np.std(values)
                    )

    return aggregated


def save_results_to_csv(results: Dict[str, Any], output_dir: str):
    """
    将实验结果保存为 CSV 格式
    
    保存两个文件:
    1. detailed_results.csv: 每次重复的详细结果
    2. aggregated_results.csv: 聚合后的均值±标准差结果
    
    参数:
        results: run_all_experiments 返回的原始结果
        output_dir: 输出目录
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. 保存详细结果 (每次重复)
    detailed_file = os.path.join(output_dir, 'detailed_results.csv')
    detailed_rows = []
    
    for group_name, group_results in results.items():
        for model_name, model_results in group_results.items():
            for exp_result in model_results:
                if exp_result.get('metrics') is None:
                    continue
                    
                row = {
                    'group': group_name,
                    'model': model_name,
                    'repeat_id': exp_result.get('info', {}).get('repeat_id', 'N/A'),
                    'seed': exp_result.get('info', {}).get('seed', 'N/A'),
                    'n_train': exp_result.get('info', {}).get('n_train', 'N/A'),
                    'n_test': exp_result.get('info', {}).get('n_test', 'N/A'),
                    'censoring_rate': exp_result.get('info', {}).get('actual_censoring_rate', 'N/A'),
                }
                
                # 添加所有指标
                metrics = exp_result['metrics']
                for metric_name, metric_value in metrics.items():
                    if metric_value is not None and not (isinstance(metric_value, float) and np.isnan(metric_value)):
                        row[f'metric_{metric_name}'] = metric_value
                
                detailed_rows.append(row)
    
    if detailed_rows:
        # 收集所有可能的列名
        all_columns = set()
        for row in detailed_rows:
            all_columns.update(row.keys())
        
        # 确保指标列在后面
        metric_columns = sorted([col for col in all_columns if col.startswith('metric_')])
        basic_columns = sorted([col for col in all_columns if not col.startswith('metric_')])
        fieldnames = basic_columns + metric_columns
        
        with open(detailed_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
            writer.writeheader()
            writer.writerows(detailed_rows)
        print(f"详细结果已保存至：{detailed_file}")
    
    # 2. 保存聚合结果 (均值±标准差)
    aggregated = aggregate_results(results)
    aggregated_file = os.path.join(output_dir, 'aggregated_results.csv')
    
    if aggregated:
        # 获取所有实验组和模型
        group_names = list(aggregated.keys())
        model_names = list(aggregated[group_names[0]].keys()) if group_names else []
        
        # 获取所有指标名称
        all_metrics = set()
        for group_name in group_names:
            for model_name in aggregated[group_name]:
                all_metrics.update(aggregated[group_name][model_name].keys())
        all_metrics = sorted(all_metrics)
        
        # 构建 CSV 行
        aggregated_rows = []
        for group_name in group_names:
            for model_name in model_names:
                if model_name not in aggregated[group_name]:
                    continue
                    
                row = {
                    'group': group_name,
                    'model': model_name,
                }
                
                # 添加每个指标的均值±标准差
                for metric in all_metrics:
                    if metric in aggregated[group_name][model_name]:
                        mean, std = aggregated[group_name][model_name][metric]
                        row[f'{metric}'] = f"{mean:.6f}"
                        row[f'{metric}_std'] = f"{std:.6f}"
                    else:
                        # 如果某个模型没有这个指标，填充空值
                        row[f'{metric}'] = ''
                        row[f'{metric}_std'] = ''
                
                aggregated_rows.append(row)
        
        # 列顺序：基本信息 + 指标 (均值 + 标准差)
        metric_cols = []
        for metric in all_metrics:
            metric_cols.append(f'{metric}')
            metric_cols.append(f'{metric}_std')
        
        fieldnames = ['group', 'model', *metric_cols]
        
        with open(aggregated_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(aggregated_rows)
        
        print(f"聚合结果已保存至：{aggregated_file}")
    
    return detailed_file if detailed_rows else None, aggregated_file if aggregated_rows else None


def print_results_table(aggregated: Dict, metric_name: str = 'c_index'):
    """
    打印结果表格
    
    参数:
        aggregated: aggregate_results 返回的聚合结果
        metric_name: 要展示的指标名称 (默认 C-index)
    """
    print(f"\n{metric_name.upper()} Results")
    print("=" * 80)

    group_names = list(aggregated.keys())
    model_names = list(aggregated[group_names[0]].keys()) if group_names else []

    # 打印表头
    header = f"{'Group':<25}" + "".join([f"{m:<15}" for m in model_names])
    print(header)
    print("-" * 80)

    # 打印每一行
    for group_name in group_names:
        row = f"{group_name:<25}"
        for model_name in model_names:
            if model_name in aggregated[group_name]:
                result = aggregated[group_name][model_name].get(metric_name)
                if result:
                    mean, std = result
                    row += f"{mean:.4f}±{std:.4f} "
                else:
                    row += "N/A            "
            else:
                row += "N/A            "
        print(row)


# 注意：该脚本现已作为模块使用。
# 请使用 main.py 作为主程序入口运行实验。

if __name__ == "__main__":
    print("=" * 60)
    print("Survival Analysis Experiment Module")
    print("=" * 60)
    print("\n[提示] 该脚本现已重构为模块。")
    print("请使用 'python experiments/main.py' 来解析命令行参数并运行实验。")
    print("\n示例:")
    print("  python experiments/main.py --quick")
    print("  python experiments/main.py --models FlowSurv --groups E1 E2")
    print("=" * 60)
