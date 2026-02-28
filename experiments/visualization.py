import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import Normalize
from typing import Dict, List, Optional, Tuple, Any

plt.rcParams['font.size'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['figure.dpi'] = 150

from experiments.config import PlotConfig


class SurvivalVisualizer:
    """生存分析可视化器"""
    
    def __init__(self, config: PlotConfig = None):
        self.config = config or PlotConfig()
        os.makedirs(self.config.save_dir, exist_ok=True)
    
    def _get_color(self, model_name: str) -> str:
        return self.config.colors.get(model_name, '#333333')
    
    def _get_linestyle(self, model_name: str) -> str:
        return self.config.linestyles.get(model_name, '-')
    
    def _save_figure(self, fig, filename: str):
        filepath = os.path.join(self.config.save_dir, f"{filename}.{self.config.save_format}")
        fig.savefig(filepath, dpi=self.config.dpi, bbox_inches='tight')
        print(f"Figure saved: {filepath}")
    
    def plot_hazard_curves(
        self,
        time_grid: np.ndarray,
        true_hazard: np.ndarray,
        pred_hazards: Dict[str, np.ndarray],
        sample_indices: List[int] = None,
        title: str = "Hazard Rate Comparison",
        save_name: str = None
    ):
        """
        绘制风险曲线对比图
        
        参数:
            time_grid: 时间网格
            true_hazard: 真实风险率 (n_samples, n_times)
            pred_hazards: 各模型预测的风险率 {model_name: (n_samples, n_times)}
            sample_indices: 要绘制的样本索引
            title: 图标题
            save_name: 保存文件名
        """
        n_samples = true_hazard.shape[0]
        if sample_indices is None:
            sample_indices = np.linspace(0, n_samples - 1, self.config.n_samples, dtype=int)
        
        n_plots = len(sample_indices)
        fig, axes = plt.subplots(1, n_plots, figsize=(4 * n_plots, 4))
        if n_plots == 1:
            axes = [axes]
        
        for ax_idx, sample_idx in enumerate(sample_indices):
            ax = axes[ax_idx]
            
            ax.plot(time_grid, true_hazard[sample_idx], 
                   color=self._get_color('true'), 
                   linestyle=self._get_linestyle('true'),
                   linewidth=2.5, label='True')
            
            for model_name, pred_h in pred_hazards.items():
                if pred_h is not None and sample_idx < pred_h.shape[0]:
                    ax.plot(time_grid, pred_h[sample_idx],
                           color=self._get_color(model_name),
                           linestyle=self._get_linestyle(model_name),
                           linewidth=1.5, alpha=0.8, label=model_name)
            
            ax.set_xlabel('Time')
            ax.set_ylabel('h(t)')
            ax.set_title(f'Sample {sample_idx}')
            ax.legend(loc='upper right', fontsize=8)
            ax.grid(True, alpha=0.3)
        
        fig.suptitle(title, fontsize=14, y=1.02)
        plt.tight_layout()
        
        if save_name:
            self._save_figure(fig, save_name)
        
        return fig
    
    def plot_density_curves(
        self,
        time_grid: np.ndarray,
        true_density: np.ndarray,
        pred_densities: Dict[str, np.ndarray],
        sample_indices: List[int] = None,
        title: str = "Density Function Comparison",
        save_name: str = None
    ):
        """
        绘制密度曲线对比图
        """
        n_samples = true_density.shape[0]
        if sample_indices is None:
            sample_indices = np.linspace(0, n_samples - 1, self.config.n_samples, dtype=int)
        
        n_plots = len(sample_indices)
        fig, axes = plt.subplots(1, n_plots, figsize=(4 * n_plots, 4))
        if n_plots == 1:
            axes = [axes]
        
        for ax_idx, sample_idx in enumerate(sample_indices):
            ax = axes[ax_idx]
            
            ax.plot(time_grid, true_density[sample_idx],
                   color=self._get_color('true'),
                   linestyle=self._get_linestyle('true'),
                   linewidth=2.5, label='True')
            
            for model_name, pred_d in pred_densities.items():
                if pred_d is not None and sample_idx < pred_d.shape[0]:
                    ax.plot(time_grid, pred_d[sample_idx],
                           color=self._get_color(model_name),
                           linestyle=self._get_linestyle(model_name),
                           linewidth=1.5, alpha=0.8, label=model_name)
            
            ax.set_xlabel('Time')
            ax.set_ylabel('f(t)')
            ax.set_title(f'Sample {sample_idx}')
            ax.legend(loc='upper right', fontsize=8)
            ax.grid(True, alpha=0.3)
        
        fig.suptitle(title, fontsize=14, y=1.02)
        plt.tight_layout()
        
        if save_name:
            self._save_figure(fig, save_name)
        
        return fig
    
    def plot_survival_curves(
        self,
        time_grid: np.ndarray,
        true_survival: np.ndarray,
        pred_survivals: Dict[str, np.ndarray],
        sample_indices: List[int] = None,
        title: str = "Survival Function Comparison",
        save_name: str = None
    ):
        """
        绘制生存曲线对比图
        """
        n_samples = true_survival.shape[0]
        if sample_indices is None:
            sample_indices = np.linspace(0, n_samples - 1, self.config.n_samples, dtype=int)
        
        n_plots = len(sample_indices)
        fig, axes = plt.subplots(1, n_plots, figsize=(4 * n_plots, 4))
        if n_plots == 1:
            axes = [axes]
        
        for ax_idx, sample_idx in enumerate(sample_indices):
            ax = axes[ax_idx]
            
            ax.plot(time_grid, true_survival[sample_idx],
                   color=self._get_color('true'),
                   linestyle=self._get_linestyle('true'),
                   linewidth=2.5, label='True')
            
            for model_name, pred_s in pred_survivals.items():
                if pred_s is not None and sample_idx < pred_s.shape[0]:
                    ax.plot(time_grid, pred_s[sample_idx],
                           color=self._get_color(model_name),
                           linestyle=self._get_linestyle(model_name),
                           linewidth=1.5, alpha=0.8, label=model_name)
            
            ax.axhline(y=0.5, color='gray', linestyle=':', alpha=0.5)
            ax.set_xlabel('Time')
            ax.set_ylabel('S(t)')
            ax.set_title(f'Sample {sample_idx}')
            ax.legend(loc='lower left', fontsize=8)
            ax.grid(True, alpha=0.3)
            ax.set_ylim(-0.05, 1.05)
        
        fig.suptitle(title, fontsize=14, y=1.02)
        plt.tight_layout()
        
        if save_name:
            self._save_figure(fig, save_name)
        
        return fig
    
    def plot_hazard_surface(
        self,
        time_grid: np.ndarray,
        feature_values: np.ndarray,
        hazard_matrix: np.ndarray,
        feature_name: str = "X1",
        title: str = "Hazard Surface",
        save_name: str = None
    ):
        """
        绘制风险曲面图
        
        参数:
            time_grid: 时间网格
            feature_values: 特征值网格
            hazard_matrix: 风险率矩阵 (n_features, n_times)
            feature_name: 特征名称
            title: 图标题
            save_name: 保存文件名
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        T, F = np.meshgrid(time_grid, feature_values)
        
        pcm = ax.pcolormesh(T, F, hazard_matrix, shading='auto', cmap='YlOrRd')
        
        cbar = fig.colorbar(pcm, ax=ax)
        cbar.set_label('h(t|x)')
        
        ax.set_xlabel('Time')
        ax.set_ylabel(feature_name)
        ax.set_title(title)
        
        plt.tight_layout()
        
        if save_name:
            self._save_figure(fig, save_name)
        
        return fig
    
    def plot_hazard_surface_comparison(
        self,
        time_grid: np.ndarray,
        feature_values: np.ndarray,
        true_hazard_surface: np.ndarray,
        pred_hazard_surfaces: Dict[str, np.ndarray],
        feature_name: str = "X1",
        title: str = "Hazard Surface Comparison",
        save_name: str = None
    ):
        """
        绘制风险曲面对比图 (真实 vs 各模型)
        """
        n_models = len(pred_hazard_surfaces) + 1
        fig, axes = plt.subplots(1, n_models, figsize=(4 * n_models, 4))
        
        T, F = np.meshgrid(time_grid, feature_values)
        
        ax = axes[0]
        pcm = ax.pcolormesh(T, F, true_hazard_surface, shading='auto', cmap='YlOrRd')
        fig.colorbar(pcm, ax=ax)
        ax.set_xlabel('Time')
        ax.set_ylabel(feature_name)
        ax.set_title('True')
        
        for idx, (model_name, pred_surface) in enumerate(pred_hazard_surfaces.items(), 1):
            ax = axes[idx]
            if pred_surface is not None:
                pcm = ax.pcolormesh(T, F, pred_surface, shading='auto', cmap='YlOrRd')
                fig.colorbar(pcm, ax=ax)
            ax.set_xlabel('Time')
            ax.set_ylabel(feature_name)
            ax.set_title(model_name)
        
        fig.suptitle(title, fontsize=14, y=1.02)
        plt.tight_layout()
        
        if save_name:
            self._save_figure(fig, save_name)
        
        return fig
    
    def plot_metrics_comparison(
        self,
        metrics_results: Dict[str, Dict[str, Tuple[float, float]]],
        metric_name: str = 'c_index',
        title: str = None,
        save_name: str = None
    ):
        """
        绘制指标对比柱状图
        
        参数:
            metrics_results: {group_name: {model_name: (mean, std)}}
            metric_name: 指标名称
            title: 图标题
            save_name: 保存文件名
        """
        group_names = list(metrics_results.keys())
        model_names = list(metrics_results[group_names[0]].keys()) if group_names else []
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        x = np.arange(len(group_names))
        width = 0.8 / len(model_names)
        
        for idx, model_name in enumerate(model_names):
            means = []
            stds = []
            for group_name in group_names:
                result = metrics_results[group_name].get(model_name, {}).get(metric_name)
                if result:
                    means.append(result[0])
                    stds.append(result[1])
                else:
                    means.append(0)
                    stds.append(0)
            
            bars = ax.bar(x + idx * width, means, width, 
                         yerr=stds, label=model_name,
                         color=self._get_color(model_name), alpha=0.8)
        
        ax.set_xlabel('Experiment Group')
        ax.set_ylabel(metric_name.upper())
        ax.set_title(title or f'{metric_name.upper()} Comparison')
        ax.set_xticks(x + width * (len(model_names) - 1) / 2)
        ax.set_xticklabels(group_names, rotation=45, ha='right')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        if save_name:
            self._save_figure(fig, save_name)
        
        return fig
    
    def plot_metrics_heatmap(
        self,
        metrics_results: Dict[str, Dict[str, Tuple[float, float]]],
        metric_name: str = 'c_index',
        title: str = None,
        save_name: str = None
    ):
        """
        绘制指标热力图
        """
        group_names = list(metrics_results.keys())
        model_names = list(metrics_results[group_names[0]].keys()) if group_names else []
        
        data = np.zeros((len(model_names), len(group_names)))
        
        for j, group_name in enumerate(group_names):
            for i, model_name in enumerate(model_names):
                result = metrics_results[group_name].get(model_name, {}).get(metric_name)
                if result:
                    data[i, j] = result[0]
                else:
                    data[i, j] = np.nan
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        im = ax.imshow(data, cmap='RdYlGn', aspect='auto')
        
        ax.set_xticks(np.arange(len(group_names)))
        ax.set_yticks(np.arange(len(model_names)))
        ax.set_xticklabels(group_names, rotation=45, ha='right')
        ax.set_yticklabels(model_names)
        
        for i in range(len(model_names)):
            for j in range(len(group_names)):
                if not np.isnan(data[i, j]):
                    text = ax.text(j, i, f'{data[i, j]:.3f}',
                                  ha='center', va='center', color='black', fontsize=9)
        
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label(metric_name.upper())
        
        ax.set_title(title or f'{metric_name.upper()} Heatmap')
        
        plt.tight_layout()
        
        if save_name:
            self._save_figure(fig, save_name)
        
        return fig
    
    def plot_boxplot(
        self,
        values_dict: Dict[str, List[float]],
        title: str = "Metrics Distribution",
        ylabel: str = "Value",
        save_name: str = None
    ):
        """
        绘制箱线图
        
        参数:
            values_dict: {model_name: [values]}
            title: 图标题
            ylabel: y轴标签
            save_name: 保存文件名
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        labels = list(values_dict.keys())
        data = [values_dict[k] for k in labels]
        
        bp = ax.boxplot(data, labels=labels, patch_artist=True)
        
        for idx, (label, box) in enumerate(zip(labels, bp['boxes'])):
            box.set_facecolor(self._get_color(label))
            box.set_alpha(0.7)
        
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        if save_name:
            self._save_figure(fig, save_name)
        
        return fig
    
    def plot_high_censoring_comparison(
        self,
        time_grid: np.ndarray,
        true_survival: np.ndarray,
        pred_survivals: Dict[str, np.ndarray],
        censoring_rate: float,
        sample_indices: List[int] = None,
        title: str = None,
        save_name: str = None
    ):
        """
        高删失场景下的生存曲线对比
        """
        n_samples = true_survival.shape[0]
        if sample_indices is None:
            sample_indices = [0, n_samples // 4, n_samples // 2, 3 * n_samples // 4]
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.flatten()
        
        for ax_idx, sample_idx in enumerate(sample_indices[:4]):
            ax = axes[ax_idx]
            
            ax.plot(time_grid, true_survival[sample_idx],
                   color=self._get_color('true'),
                   linestyle=self._get_linestyle('true'),
                   linewidth=2.5, label='True')
            
            for model_name, pred_s in pred_survivals.items():
                if pred_s is not None and sample_idx < pred_s.shape[0]:
                    ax.plot(time_grid, pred_s[sample_idx],
                           color=self._get_color(model_name),
                           linestyle=self._get_linestyle(model_name),
                           linewidth=1.5, alpha=0.8, label=model_name)
            
            ax.axhline(y=0.5, color='gray', linestyle=':', alpha=0.5)
            ax.set_xlabel('Time')
            ax.set_ylabel('S(t)')
            ax.set_title(f'Sample {sample_idx}')
            ax.legend(loc='lower left', fontsize=8)
            ax.grid(True, alpha=0.3)
            ax.set_ylim(-0.05, 1.05)
        
        title = title or f'Survival Curves (Censoring Rate: {censoring_rate:.0%})'
        fig.suptitle(title, fontsize=14, y=1.02)
        plt.tight_layout()
        
        if save_name:
            self._save_figure(fig, save_name)
        
        return fig


def generate_visualization_report(
    results: Dict[str, Any],
    predictions: Dict[str, Dict[str, np.ndarray]],
    test_data: Any,
    config: PlotConfig = None
):
    """
    生成完整的可视化报告
    """
    visualizer = SurvivalVisualizer(config)
    
    time_grid = test_data.time_grid
    
    pred_survivals = {}
    pred_hazards = {}
    pred_densities = {}
    
    for model_name, pred_dict in predictions.items():
        pred_survivals[model_name] = pred_dict.get('survival')
        pred_hazards[model_name] = pred_dict.get('hazard')
        pred_densities[model_name] = pred_dict.get('density')
    
    visualizer.plot_hazard_curves(
        time_grid, test_data.true_hazard, pred_hazards,
        save_name='hazard_curves'
    )
    
    visualizer.plot_density_curves(
        time_grid, test_data.true_density, pred_densities,
        save_name='density_curves'
    )
    
    visualizer.plot_survival_curves(
        time_grid, test_data.true_survival, pred_survivals,
        save_name='survival_curves'
    )
    
    print("Visualization report generated!")


if __name__ == "__main__":
    print("测试可视化模块")
    print("=" * 60)
    
    config = PlotConfig()
    visualizer = SurvivalVisualizer(config)
    
    np.random.seed(42)
    time_grid = np.linspace(0.1, 10, 100)
    n_samples = 5
    
    true_hazard = np.exp(-time_grid / 5) * (1 + 0.5 * np.sin(time_grid))
    true_hazard = np.tile(true_hazard, (n_samples, 1))
    true_hazard *= np.linspace(0.5, 1.5, n_samples)[:, np.newaxis]
    
    pred_hazards = {
        'FlowSurv': true_hazard * (1 + 0.1 * np.random.randn(*true_hazard.shape)),
        'DeepSurv': true_hazard * (1 + 0.2 * np.random.randn(*true_hazard.shape)),
    }
    
    fig = visualizer.plot_hazard_curves(
        time_grid, true_hazard, pred_hazards,
        title="Test Hazard Curves"
    )
    plt.show()
    
    print("\nVisualization test completed!")
