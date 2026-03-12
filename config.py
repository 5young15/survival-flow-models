from __future__ import annotations

import os
from dataclasses import asdict, dataclass, fields, is_dataclass
from typing import Any, Dict, List, Optional


@dataclass
class DataConfig:
    csv_path: Optional[str] = None
    feature_cols: Optional[List[str]] = None
    time_col: Optional[str] = None
    event_col: Optional[str] = None
    train_ratio: Optional[float] = None
    random_seed: Optional[int] = None


@dataclass
class NetworkConfig:
    encoder_hidden_dims: Optional[List[int]] = None
    latent_dim: Optional[int] = None
    vf_hidden_dims: Optional[List[int]] = None
    time_emb_dim: Optional[int] = None
    dropout: Optional[float] = None
    gumbel_hidden_dims: Optional[List[int]] = None


@dataclass
class TrainConfig:
    batch_size: Optional[int] = None
    max_epochs_stage1: Optional[int] = None
    max_epochs_stage2: Optional[int] = None
    learning_rate: Optional[float] = None  # 第一阶段学习率
    stage2_learning_rate: Optional[float] = None  # 第二阶段向量场学习率
    stage2_encoder_lr_scale: Optional[float] = None  # 第二阶段编码器学习率衰减系数
    weight_decay: Optional[float] = None
    early_stop_patience: Optional[int] = None
    rank_loss_weight: Optional[float] = None
    rank_loss_margin: Optional[float] = None
    event_weight: Optional[float] = None
    grad_clip_norm: Optional[float] = None
    device: Optional[str] = None


@dataclass
class ODEConfig:
    ode_method: Optional[str] = None
    ode_steps: Optional[int] = None


@dataclass
class SamplingConfig:
    density_grid_size: Optional[int] = None
    mc_samples_train: Optional[int] = None
    mc_samples_eval: Optional[int] = None
    truncation_samples: Optional[int] = None
    survival_method: Optional[str] = None


@dataclass
class TuningConfig:
    n_trials: Optional[int] = None
    cv_folds: Optional[int] = None
    target_metric: Optional[str] = None
    direction: Optional[str] = None


@dataclass
class RuntimeConfig:
    models: Optional[List[str]] = None
    group: Optional[str] = None
    force_early_stop: Optional[str] = None
    do_cv_in_train: Optional[bool] = None


@dataclass
class ExperimentConfig:
    data: DataConfig
    network: NetworkConfig
    train: TrainConfig
    ode: ODEConfig
    sampling: SamplingConfig
    tuning: TuningConfig
    runtime: RuntimeConfig

    def validate_none(self) -> None:
        missing = _find_none_fields(self)
        if missing:
            text = ", ".join(missing)
            raise ValueError(f"以下配置仍为None，请显式设置: {text}")

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _find_none_fields(obj: Any, prefix: str = "") -> List[str]:
    missing: List[str] = []
    if is_dataclass(obj):
        for f in fields(obj):
            value = getattr(obj, f.name)
            next_prefix = f"{prefix}.{f.name}" if prefix else f.name
            missing.extend(_find_none_fields(value, next_prefix))
    elif isinstance(obj, list):
        if obj is None:
            missing.append(prefix)
    else:
        if obj is None:
            missing.append(prefix)
    return missing


def default_experiment_config() -> ExperimentConfig:
    return ExperimentConfig(
        data=DataConfig(),
        network=NetworkConfig(),
        train=TrainConfig(),
        ode=ODEConfig(),
        sampling=SamplingConfig(),
        tuning=TuningConfig(),
        runtime=RuntimeConfig(),
    )


def with_overrides(cfg: ExperimentConfig, overrides: Dict[str, Any]) -> ExperimentConfig:
    for section, payload in overrides.items():
        target = getattr(cfg, section)
        if not is_dataclass(target):
            continue
        for k, v in payload.items():
            if hasattr(target, k):
                setattr(target, k, v)
    return cfg


def preset_config(name: str) -> Dict[str, Dict[str, Any]]:
    root = os.path.dirname(os.path.abspath(__file__))
    base = {
        "data": {
            "csv_path": os.path.join(root, "results", "toy_datasets", "toy_non_ph_dataset.csv"),
            "feature_cols": [],
            "time_col": "time",
            "event_col": "event",
            "train_ratio": 0.8,
            "random_seed": 42,
        },
        "network": {
            "encoder_hidden_dims": [64],  # 编码器隐藏层维度
            "latent_dim": 32,  # 潜空间z维度
            "vf_hidden_dims": [32, 32],  # 向量场隐藏层维度
            "time_emb_dim": 32,  # 时间嵌入维度
            "dropout": 0.05,  # 随时失活
            "gumbel_hidden_dims": [32],  # gumbel头隐藏层维度
        },
        "train": {
            "batch_size": 128,
            "max_epochs_stage1": 200,
            "max_epochs_stage2": 200,
            "learning_rate": 5e-5,  # 第一阶段基础学习率
            "stage2_learning_rate": 3e-4,  # 第二阶段向量场学习率
            "stage2_encoder_lr_scale": 0.2,  # 二阶段编码器学习率衰减系数
            "weight_decay": 1e-5,
            "early_stop_patience": 10,
            "rank_loss_weight": 0.0,  # 排序损失权重
            "rank_loss_margin": 0.1,  # 排序损失 margin
            "event_weight": 0.7,  # 事件损失权重，删失损失权重: censor_event = 1-event_weight
            "grad_clip_norm": 5.0,  # 梯度裁剪
            "device": "cuda",
        },
        "ode": {
            "ode_method": "rk4",  # ode求解方法 "euler"/"rk4"
            "ode_steps": 50  # ode求解步数
        },
        "sampling": {
            "density_grid_size": 256,  # 密度积分网格大小
            "mc_samples_train": 256,  # mc训练采样
            "mc_samples_eval": 512,  # mc验证采样
            "truncation_samples": 16,  # 截断采样
            "survival_method": "density",  # 推断生存量算法
        },
        "tuning": {"n_trials": 20, "cv_folds": 3, "target_metric": "c_index", "direction": "maximize"},
        "runtime": {"models": [name], "group": "exp_default", "force_early_stop": "_", "do_cv_in_train": False},
    }
    if "gumbel" in name.lower():
        base["train"]["max_epochs_stage1"] = 100
    return base
