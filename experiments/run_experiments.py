import os
import sys
import json
import pickle
import numpy as np
import torch
from typing import Dict, List, Any, Optional, Tuple, Union
from datetime import datetime
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.config import (
    GlobalConfig, CONFIG, get_model_config,
    ExperimentGroup, DataConfig
)
from experiments.data_generation import (
    SurvivalDataGenerator, SurvivalData, generate_experiment_data
)
from experiments.metrics import (
    compute_all_metrics, metrics_to_dict, MetricsResult,
    concordance_index_fast, integrated_brier_score
)


class ModelTrainer:
    """模型训练器基类"""

    def __init__(self, model_name: str, config: GlobalConfig):
        self.model_name = model_name
        self.config = config
        self.model_config = get_model_config(model_name, config.model)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None

    def create_model(self, in_dim: int):
        """创建模型实例"""
        raise NotImplementedError

    def train(self, train_data: SurvivalData, val_data: SurvivalData):
        """训练模型"""
        raise NotImplementedError

    def predict(self, test_data: SurvivalData, time_grid: Union[np.ndarray, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """预测"""
        raise NotImplementedError


class PyTorchModelTrainer(ModelTrainer):
    """PyTorch模型训练器"""

    def create_model(self, in_dim: int):
        from models.baselines.coxph import LinearCoxPH
        from models.baselines.deepsurv import DeepSurv
        from models.baselines.deephit import DeepHit
        from models.baselines.weibullAFT import WeibullAFT
        from models.flowmodel.base_flow import FlowSurv
        from models.flowmodel.gumbel_flow import GumbelFlowSurv

        model_classes = {
            'LinearCoxPH': LinearCoxPH,
            'DeepSurv': DeepSurv,
            'WeibullAFT': WeibullAFT,
            'DeepHit': DeepHit,
            'FlowSurv': FlowSurv,
            'GumbelFlowSurv': GumbelFlowSurv,
            'GumbelFlow': GumbelFlowSurv,
            'GFM': GumbelFlowSurv,
        }

        if self.model_name not in model_classes:
            raise ValueError(f"Unknown model: {self.model_name}")

        self.model = model_classes[self.model_name](
            in_dim=in_dim,
            config=self.model_config
        ).to(self.device)

        return self.model

    def get_checkpoint_path(self, checkpoint_dir: str, repeat_id: int) -> str:
        """获取模型检查点路径"""
        return os.path.join(checkpoint_dir, self.model_name, f"repeat_{repeat_id}", "best_model.pt")

    def save_checkpoint(self, checkpoint_path: str, best_val_loss: float,
                        time_scaler_mean: float, time_scaler_std: float, is_log_space: bool):
        """保存最优模型"""
        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'best_val_loss': best_val_loss,
            'time_scaler_mean': time_scaler_mean,
            'time_scaler_std': time_scaler_std,
            'is_log_space': is_log_space,
            'model_name': self.model_name,
            'gumbel_initialized': getattr(self.model, '_gumbel_initialized', False),
            'baseline_times': getattr(self.model, '_baseline_times', None),
            'baseline_cum_haz': getattr(self.model, '_baseline_cum_haz', None),
            'train_times': getattr(self.model, '_train_times', None),
            'train_events': getattr(self.model, '_train_events', None),
            'train_log_haz': getattr(self.model, '_train_log_haz', None),
        }
        torch.save(checkpoint, checkpoint_path)

    def load_checkpoint(self, checkpoint_path: str) -> bool:
        """加载模型检查点, 返回是否成功"""
        if not os.path.exists(checkpoint_path):
            return False

        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        self.model.set_time_scaler(
            checkpoint['time_scaler_mean'],
            checkpoint['time_scaler_std'],
            checkpoint['is_log_space']
        )
        if hasattr(self.model, '_gumbel_initialized'):
            self.model._gumbel_initialized = checkpoint.get('gumbel_initialized', False)
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
        训练模型

        Args:
            train_data: 训练数据 (Tensor/Numpy)
            val_data: 验证数据 (Tensor/Numpy)
            checkpoint_dir: 检查点目录
            repeat_id: 重复次数ID

        Returns:
            best_val_loss: 最优验证损失
            from_checkpoint: 是否从检查点加载
        """
        if self.model is None:
            raise RuntimeError("Model not created. Call create_model first.")

        checkpoint_path = self.get_checkpoint_path(checkpoint_dir, repeat_id) if checkpoint_dir else None

        if checkpoint_path and self.load_checkpoint(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
            return checkpoint['best_val_loss'], True

        model_config = self.model_config

        # 统一为 Tensor 并移动到设备, 避免重复创建
        def to_device_tensor(x, dtype=torch.float32):
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

        if hasattr(self.model, 'init_gumbel_params') and hasattr(self.model, '_gumbel_initialized'):
            if not self.model._gumbel_initialized:
                # 某些模型初始化需要 numpy, 如果已经转换过, 传 Tensor 也是安全的
                self.model.init_gumbel_params(times_tensor, events_tensor)

        val_times = to_device_tensor(val_data.times)
        val_events = to_device_tensor(val_data.events)
        val_features = to_device_tensor(val_data.features)

        def train_stage(stage: str, epochs: int, lr: float, batch_size: int, patience: int, weight_decay: float):
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
            weibull_epochs = model_config.get('WEIBULL_EPOCHS', 200)
            if weibull_epochs > 0:
                train_stage(
                    stage='weibull',
                    epochs=weibull_epochs,
                    lr=model_config.get('WEIBULL_LR', 5e-8),
                    batch_size=model_config.get('WEIBULL_BATCH_SIZE', 64),
                    patience=model_config.get('WEIBULL_PATIENCE', 15),
                    weight_decay=model_config.get('WEIBULL_WEIGHT_DECAY', 1e-5),
                )

            best_val_loss = train_stage(
                stage='flow',
                epochs=model_config.get('EPOCHS', 200),
                lr=model_config.get('LR', 3e-4),
                batch_size=model_config.get('BATCH_SIZE', 64),
                patience=model_config.get('PATIENCE', 15),
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

    def predict(self, test_data: SurvivalData, time_grid: Union[np.ndarray, torch.Tensor]) -> Dict[str, torch.Tensor]:
        if self.model is None:
            raise RuntimeError("Model not trained.")

        self.model.eval()
        # 统一为 Tensor
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
            if hasattr(self.model, 'predict_survival_metrics'):
                results = self.model.predict_survival_metrics(features, time_grid)
                risk_scores = self.model.predict_risk(features)
                survival = results['survival']
                hazard = results['hazard']
                density = results['density']
                pred_medians = self.model.predict_time(features, mode='median')
            else:
                risk_scores = self.model.predict_risk(features)
                survival = self.model.predict_survival_function(features, time_grid)
                pred_medians = self.model.predict_time(features, mode='median')

                try:
                    hazard = self.model.compute_hazard_rate(features, time_grid)
                except:
                    hazard = None

                try:
                    if hasattr(self.model, 'compute_density'):
                        density = self.model.compute_density(features, time_grid)
                    else:
                        density = None
                except:
                    density = None

        return {
            'risk_scores': risk_scores,
            'survival': survival,
            'medians': pred_medians,
            'hazard': hazard,
            'density': density
        }


class RSFTrainer(ModelTrainer):
    """Random Survival Forest 训练器"""

    def create_model(self, in_dim: int):
        from models.baselines.RSF import RandomSurvivalForestWrapper

        self.model = RandomSurvivalForestWrapper(
            in_dim=in_dim,
            config=self.model_config
        )
        return self.model

    def get_checkpoint_path(self, checkpoint_dir: str, repeat_id: int) -> str:
        """获取模型检查点路径"""
        return os.path.join(checkpoint_dir, self.model_name, f"repeat_{repeat_id}", "best_model.pkl")

    def save_checkpoint(self, checkpoint_path: str):
        """保存RSF模型"""
        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
        with open(checkpoint_path, 'wb') as f:
            pickle.dump(self.model, f)

    def load_checkpoint(self, checkpoint_path: str) -> bool:
        """加载RSF模型"""
        if not os.path.exists(checkpoint_path):
            return False
        with open(checkpoint_path, 'rb') as f:
            self.model = pickle.load(f)
        return True

    def train(self, train_data: SurvivalData, val_data: SurvivalData,
              checkpoint_dir: Optional[str] = None, repeat_id: int = 0) -> Tuple[float, bool]:
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

        self.model.fit(X, y_time, y_event)

        if checkpoint_path:
            self.save_checkpoint(checkpoint_path)

        return 0.0, False

    def predict(self, test_data: SurvivalData, time_grid: Union[np.ndarray, torch.Tensor]) -> Dict[str, torch.Tensor]:
        features = test_data.features
        # 确保 time_grid 是 tensor
        if isinstance(time_grid, np.ndarray):
            time_grid = torch.from_numpy(time_grid).float().to(features.device if isinstance(features, torch.Tensor) else 'cpu')

        risk_scores = self.model.predict_risk(features, time_grid)
        survival = self.model.predict_survival_function(features, time_grid)
        pred_medians = self.model.predict_time(features)

        try:
            hazard = self.model.compute_hazard_rate(features, time_grid)
        except:
            hazard = None

        return {
            'risk_scores': risk_scores,
            'survival': survival,
            'medians': pred_medians,
            'hazard': hazard,
            'density': None
        }


def create_trainer(model_name: str, config: GlobalConfig) -> ModelTrainer:
    """创建模型训练器"""
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
    device: Optional[torch.device] = None
) -> Tuple[MetricsResult, Dict[str, np.ndarray], Dict[str, Any]]:
    """运行单次实验"""
    
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    seed = group.data_config.random_seed + repeat_id * 100
    np.random.seed(seed)
    torch.manual_seed(seed)

    data_config = group.data_config
    data_config.random_seed = seed

    generator = SurvivalDataGenerator(data_config)
    full_data = generator.generate(n=data_config.n_samples, seed=seed)

    n = len(full_data.times)
    test_ratio = config.experiment.test_ratio
    val_ratio = config.experiment.val_ratio
    n_test = int(n * test_ratio)
    n_val = int((n - n_test) * val_ratio)

    indices = np.random.permutation(n)
    test_idx = indices[:n_test]
    val_idx = indices[n_test:n_test + n_val]
    train_idx = indices[n_test + n_val:]

    true_medians_np = generator.generator.median(full_data.features[test_idx])

    full_data = full_data.to(device)
    true_medians = torch.from_numpy(true_medians_np).float().to(device)

    def split_data(data: SurvivalData, idx: np.ndarray) -> SurvivalData:
        # idx 是 numpy 数组，转换为 tensor 索引或直接使用
        # torch tensor 支持 numpy array indexing
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

    trainer = create_trainer(model_name, config)
    trainer.create_model(in_dim=train_data.features.shape[1])

    group_checkpoint_dir = os.path.join(checkpoint_dir, group.name) if checkpoint_dir else None
    best_val_loss, from_checkpoint = trainer.train(train_data, val_data, group_checkpoint_dir, repeat_id)

    time_grid = full_data.time_grid
    predictions = trainer.predict(test_data, time_grid)

    metrics = compute_all_metrics(
        times=test_data.times,
        events=test_data.events,
        risk_scores=predictions['risk_scores'],
        pred_survival=predictions['survival'],
        pred_medians=predictions['medians'],
        time_grid=time_grid,
        true_hazard=test_data.true_hazard,
        true_density=test_data.true_density,
        true_survival=test_data.true_survival,
        true_medians=true_medians,
        pred_hazard=predictions['hazard'],
        pred_density=predictions['density'],
        quantiles=config.experiment.time_quantiles,
        max_weight=config.experiment.ipcw_max_weight
    )

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
    device: Optional[torch.device] = None
) -> Dict[str, Any]:
    """运行所有实验"""

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

    total_runs = len(groups) * len(model_names) * n_repeats
    pbar = tqdm(total=total_runs, desc="Running experiments")

    for group in groups:
        group_results = {}

        for model_name in model_names:
            model_results = []

            for repeat_id in range(n_repeats):
                try:
                    metrics, predictions, info = run_single_experiment(
                        group, model_name, repeat_id, config, checkpoint_dir, device
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

    return all_results


def aggregate_results(results: Dict[str, Any]) -> Dict[str, Dict[str, Dict[str, Tuple[float, float]]]]:
    """聚合结果：计算均值和标准差"""
    aggregated = {}

    for group_name, group_results in results.items():
        aggregated[group_name] = {}

        for model_name, model_results in group_results.items():
            valid_metrics = [r['metrics'] for r in model_results if r.get('metrics') is not None]

            if not valid_metrics:
                continue

            metric_names = list(valid_metrics[0].keys())
            aggregated[group_name][model_name] = {}

            for metric in metric_names:
                values = [m[metric] for m in valid_metrics if metric in m and m[metric] is not None and not (isinstance(m[metric], float) and np.isnan(m[metric]))]
                if values:
                    aggregated[group_name][model_name][metric] = (
                        np.mean(values),
                        np.std(values)
                    )

    return aggregated


def print_results_table(aggregated: Dict, metric_name: str = 'c_index'):
    """打印结果表格"""
    print(f"\n{metric_name.upper()} Results")
    print("=" * 80)

    group_names = list(aggregated.keys())
    model_names = list(aggregated[group_names[0]].keys()) if group_names else []

    header = f"{'Group':<25}" + "".join([f"{m:<15}" for m in model_names])
    print(header)
    print("-" * 80)

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


if __name__ == "__main__":
    print("=" * 60)
    print("FlowSurv / GumbelFlowSurv Simulation Experiments")
    print("=" * 60)

    config = CONFIG

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    print(f"Number of repeats: {config.experiment.n_repeats}")
    print(f"Number of experiment groups: {len(config.experiment.groups)}")
    print(f"Number of models: {len(config.model.configs)}")

    model_names = ['LinearCoxPH', 'DeepSurv', 'WeibullAFT', 'RSF', 'DeepHit', 'FlowSurv', 'GumbelFlowSurv']

    results = run_all_experiments(
        config,
        model_names=model_names,
        n_repeats=3,
        save_results=True
    )

    aggregated = aggregate_results(results)

    print("\n" + "=" * 80)
    print("SUMMARY RESULTS")
    print("=" * 80)

    print_results_table(aggregated, 'c_index')
    print_results_table(aggregated, 'ibs')

    print("\nExperiment completed!")
