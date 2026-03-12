from __future__ import annotations

import os
from typing import Dict, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import cm
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def risk_from_bundle_log_hazard(bundle: Dict[str, np.ndarray], t_ref: Optional[float] = None, eps: float = 1e-8) -> np.ndarray:
    grid_t = np.asarray(bundle.get("grid_t", np.array([])))
    hazard = np.asarray(bundle.get("hazard", np.array([])))
    
    if len(grid_t) == 0 or len(hazard) == 0:
        return np.full(bundle.get("median", np.array([0])).shape, np.nan)
    
    # 强防护：去nan/inf + 合理 clip
    hazard = np.nan_to_num(hazard, nan=eps, posinf=1e6, neginf=eps)
    hazard = np.clip(hazard, eps, 1e6)  # 上限设死，避免爆炸
    
    n_samples = hazard.shape[0] if hazard.ndim > 1 else 1
    
    if t_ref is not None:
        idx = np.searchsorted(grid_t, t_ref, side="left")
        idx = np.clip(idx, 0, len(grid_t) - 1)
        h = hazard[:, idx] if hazard.ndim == 2 else hazard[idx]
    else:
        median = np.asarray(bundle.get("median", np.zeros(n_samples))).reshape(-1)
        h = np.full(len(median), eps)
        if grid_t.ndim == 2:
            for i in range(len(median)):
                current_grid = grid_t[i]
                idx = np.searchsorted(current_grid, median[i], side="left")
                idx = np.clip(idx, 0, len(current_grid) - 1)
                h[i] = hazard[i, idx] if hazard.ndim == 2 and idx < hazard.shape[1] else eps
        else:
            idx = np.searchsorted(grid_t, median, side="left")
            idx = np.clip(idx, 0, len(grid_t) - 1)
            valid = idx < hazard.shape[1] if hazard.ndim == 2 else True
            h[valid] = hazard[np.arange(len(median))[valid], idx[valid]] if hazard.ndim == 2 else hazard[idx[valid]]
    
    log_h = np.log(np.clip(h, eps, None))
    log_h = np.clip(log_h, -20.0, 10.0)  # 最终强制范围
    return log_h


def plot_two_stage_training_curve(loss_csv: str, out_png: str, best_epoch: Optional[int] = None) -> None:
    if not os.path.exists(loss_csv):
        return
    df = pd.read_csv(loss_csv)
    if df.empty or "epoch" not in df.columns:
        return

    train_cols = [c for c in df.columns if "train" in c.lower() and "loss" in c.lower()]
    val_cols = [c for c in df.columns if "val" in c.lower() and "loss" in c.lower()]

    if not train_cols and not val_cols:
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    epochs = df["epoch"].to_numpy()

    for c in train_cols:
        ax.plot(epochs, df[c].to_numpy(), linewidth=1.8, alpha=0.9, label=c)
    for c in val_cols:
        ax.plot(epochs, df[c].to_numpy(), linewidth=2.1, alpha=0.95, linestyle="--", label=c)

    if best_epoch is not None and best_epoch > 0:
        ax.axvline(float(best_epoch), color="tab:red", linestyle=":", linewidth=1.8, label=f"best_epoch={best_epoch}")

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Training Curve")
    ax.grid(True, linestyle="--", alpha=0.3)
    if train_cols or val_cols:
        ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(out_png, dpi=300)
    plt.close(fig)

    # 导出绘图数据为 CSV
    csv_path = out_png.replace(".png", "_data.csv")
    df.to_csv(csv_path, index=False)
    print(f"Training curve data exported to: {csv_path}")


def plot_crossing_survival_curves(
    grid_t: np.ndarray,
    km_a: np.ndarray,
    km_b: np.ndarray,
    flow_a: np.ndarray,
    flow_b: np.ndarray,
    cox_a: np.ndarray,
    cox_b: np.ndarray,
    out_png: str,
    title: str = "Crossing Survival Curves",
) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(grid_t, km_a, drawstyle="steps-post", linewidth=2.5, color="tab:blue", label="KM A")
    ax.plot(grid_t, km_b, drawstyle="steps-post", linewidth=2.5, color="tab:orange", label="KM B")
    ax.plot(grid_t, cox_a, linewidth=2.2, linestyle="--", color="tab:blue", alpha=0.7, label="Cox A")
    ax.plot(grid_t, cox_b, linewidth=2.2, linestyle="--", color="tab:orange", alpha=0.7, label="Cox B")
    ax.plot(grid_t, flow_a, linewidth=2.4, linestyle="-.", color="tab:green", label="Flow A")
    ax.plot(grid_t, flow_b, linewidth=2.4, linestyle="-.", color="tab:red", label="Flow B")
    ax.set_ylim(0.0, 1.02)
    ax.set_xlabel("Time")
    ax.set_ylabel("Survival Probability S(t)")
    ax.set_title(title)
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend(ncol=3, fontsize=10)
    fig.tight_layout()
    fig.savefig(out_png, dpi=300)
    plt.close(fig)

    # 导出绘图数据为 CSV
    csv_path = out_png.replace(".png", "_data.csv")
    pd.DataFrame({
        "time": grid_t,
        "km_a": km_a,
        "km_b": km_b,
        "flow_a": flow_a,
        "flow_b": flow_b,
        "cox_a": cox_a,
        "cox_b": cox_b
    }).to_csv(csv_path, index=False)
    print(f"Crossing survival curves data exported to: {csv_path}")


def plot_dynamic_metric(
    grid_t: np.ndarray,
    cox_vals: np.ndarray,
    flow_vals: np.ndarray,
    metric_label: str,
    out_png: str,
    title: str,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))

    ax.plot(grid_t, cox_vals, label=f"Cox {metric_label}", linestyle="--", linewidth=2.0, color="tab:orange")
    ax.plot(grid_t, flow_vals, label=f"Flow {metric_label}", linewidth=2.5, color="tab:blue")
    ax.set_xlabel("Time")
    ax.set_ylabel(metric_label)
    ax.set_title(title)
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.set_ylim(0.4, 1.02)

    fig.tight_layout()
    fig.savefig(out_png, dpi=300)
    plt.close(fig)

    # 导出绘图数据为 CSV
    csv_path = out_png.replace(".png", "_data.csv")
    pd.DataFrame({
        "time": grid_t,
        "cox_vals": cox_vals,
        "flow_vals": flow_vals
    }).to_csv(csv_path, index=False)
    print(f"Dynamic metric ({metric_label}) data exported to: {csv_path}")


def plot_time_varying_hazard_surface(
    grid_t: np.ndarray,
    temperature_grid: np.ndarray,
    hazard_surface: np.ndarray,
    out_png: str,
    title: str = "3D Time-varying Hazard Surface",
    true_hazard_surface: Optional[np.ndarray] = None,
    view_init: Optional[tuple] = None,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    survival_surface: Optional[np.ndarray] = None,
    survival_threshold: float = 0.1,
    z_label: str = "Hazard h(t)",
) -> None:
    t_mesh, temp_mesh = np.meshgrid(grid_t, temperature_grid)

    # Use hazard directly (no log)
    hazard_plot_data = hazard_surface.copy()
    
    # Apply survival mask if provided
    mask = None
    if survival_surface is not None:
        mask = survival_surface < survival_threshold
        hazard_plot_data[mask] = np.nan
    
    true_hazard_plot_data = None
    if true_hazard_surface is not None:
        if true_hazard_surface.shape != hazard_surface.shape:
            print(f"Warning: true_hazard_surface shape {true_hazard_surface.shape} mismatch with hazard_surface {hazard_surface.shape}")
        true_hazard_plot_data = true_hazard_surface.copy()
        if mask is not None:
            true_hazard_plot_data[mask] = np.nan

    # 在计算 vmin/vmax 前加防护
    all_vals_list = [hazard_plot_data[~np.isnan(hazard_plot_data)].flatten()]
    if true_hazard_plot_data is not None:
        all_vals_list.append(true_hazard_plot_data[~np.isnan(true_hazard_plot_data)].flatten())
    
    all_vals = np.concatenate(all_vals_list)
    
    if len(all_vals) > 0:
        if vmin is None:
            vmin = np.percentile(all_vals, 1)
        if vmax is None:
            vmax = np.percentile(all_vals, 99)
        # 防止 vmin == vmax 导致绘图崩溃
        if vmin >= vmax:
            vmin -= 0.5
            vmax += 0.5
    else:
        vmin, vmax = 0.0, 5.0  # 更安全的默认值 (Hazard通常>=0)

    # 强制 clip 绘图数据
    hazard_plot = np.clip(hazard_plot_data, vmin, vmax)
    
    if true_hazard_plot_data is not None:
        true_hazard_plot = np.clip(true_hazard_plot_data, vmin, vmax)
        
        fig = plt.figure(figsize=(16, 8), constrained_layout=True)
        elev, azim = 28, -132
        if view_init is not None:
            elev, azim = view_init

        ax1 = fig.add_subplot(121, projection="3d")
        surf1 = ax1.plot_surface(
            t_mesh, temp_mesh, true_hazard_plot,
            cmap=cm.viridis, linewidth=0, antialiased=True, alpha=0.9,
            vmin=vmin, vmax=vmax,
        )
        ax1.set_xlabel("Time t")
        ax1.set_ylabel("Temperature X1")
        ax1.set_zlabel(z_label)
        ax1.set_title("True Hazard Surface")
        ax1.view_init(elev=elev, azim=azim)
        ax1.set_zlim(vmin, vmax)
        fig.colorbar(surf1, ax=ax1, shrink=0.6, aspect=14, pad=0.08, label=z_label)

        ax2 = fig.add_subplot(122, projection="3d")
        surf2 = ax2.plot_surface(
            t_mesh, temp_mesh, hazard_plot,
            cmap=cm.viridis, linewidth=0, antialiased=True, alpha=0.9,
            vmin=vmin, vmax=vmax,
        )
        ax2.set_xlabel("Time t")
        ax2.set_ylabel("Temperature X1")
        ax2.set_zlabel(z_label)
        ax2.set_title("Predicted Hazard Surface")
        ax2.view_init(elev=elev, azim=azim)
        ax2.set_zlim(vmin, vmax)
        fig.colorbar(surf2, ax=ax2, shrink=0.6, aspect=14, pad=0.08, label=z_label)

        fig.suptitle(title, fontsize=16)
    else:
        fig = plt.figure(figsize=(10, 7), constrained_layout=True)
        ax = fig.add_subplot(111, projection="3d")
        
        elev, azim = 28, -132
        if view_init is not None:
            elev, azim = view_init
            
        surf = ax.plot_surface(
            t_mesh, temp_mesh, hazard_plot, 
            cmap=cm.viridis, linewidth=0, antialiased=True, alpha=0.95, 
            vmin=vmin, vmax=vmax
        )
        fig.colorbar(surf, shrink=0.65, aspect=16, pad=0.1, label=z_label)
        ax.set_xlabel("Time t")
        ax.set_ylabel("Temperature X1")
        ax.set_zlabel(z_label)
        ax.set_title(title)
        ax.view_init(elev=elev, azim=azim)
        ax.set_zlim(vmin, vmax)

    fig.savefig(out_png, dpi=300)
    plt.close(fig)


def _normalize_curve(y: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    y = np.asarray(y, dtype=np.float32)
    ymax = float(np.max(y))
    if ymax <= eps:
        return np.zeros_like(y)
    return y / ymax


def _effective_time_upper(grid_t: np.ndarray, curves: np.ndarray, mass_ratio: float = 0.995) -> float:
    grid = np.asarray(grid_t, dtype=np.float32)
    curves = np.asarray(curves, dtype=np.float32)
    if curves.ndim == 1:
        curves = curves[None, :]
    hit_times = []
    for i in range(curves.shape[0]):
        y = np.clip(curves[i], 0.0, None)
        if np.all(y <= 0):
            continue
        cdf = np.zeros_like(y)
        for j in range(1, len(grid)):
            cdf[j] = cdf[j - 1] + 0.5 * (y[j] + y[j - 1]) * max(float(grid[j] - grid[j - 1]), 1e-8)
        total = float(cdf[-1])
        if total <= 1e-12:
            continue
        ratio = cdf / total
        idx = int(np.searchsorted(ratio, mass_ratio, side="left"))
        idx = int(np.clip(idx, 1, len(grid) - 1))
        hit_times.append(float(grid[idx]))
    if not hit_times:
        return float(grid[-1])
    upper = float(np.percentile(np.asarray(hit_times), 90))
    return float(np.clip(upper, grid[min(3, len(grid) - 1)], grid[-1]))


def _short_label(s: str, max_len: int = 36) -> str:
    text = str(s)
    if len(text) <= max_len:
        return text
    return text[: max_len - 1] + "…"


def plot_flow_density_evolution(
    grid_t: np.ndarray,
    densities: np.ndarray,
    labels: Sequence[str],
    out_png: str,
    title: str = "Density Evolution (Ridge)",
    true_densities: Optional[np.ndarray] = None,
    colors: Optional[list[str]] = None,
) -> None:
    grid = np.asarray(grid_t, dtype=np.float32)
    pred = np.asarray(densities, dtype=np.float32)
    n = pred.shape[0]
    true = None if true_densities is None else np.asarray(true_densities, dtype=np.float32)

    if colors is None:
        colors = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple"]
    
    all_curves = pred if true is None else np.vstack([pred, true])
    x_upper = _effective_time_upper(grid, all_curves, mass_ratio=0.995)
    x_mask = grid <= x_upper
    x = grid[x_mask]
    pred = pred[:, x_mask]
    if true is not None:
        true = true[:, x_mask]

    fig_h = max(4.8, 1.2 * n + 2.0)
    fig, ax = plt.subplots(figsize=(9.0, fig_h), constrained_layout=True)

    step = 1.3
    for i in range(n):
        y0 = i * step
        c = colors[i % len(colors)]
        pred_norm = _normalize_curve(pred[i])
        ax.fill_between(x, y0, y0 + pred_norm, color=c, alpha=0.3)
        ax.plot(x, y0 + pred_norm, color=c, linewidth=2.0)
        if true is not None and i < true.shape[0]:
            true_norm = _normalize_curve(true[i])
            ax.plot(x, y0 + true_norm, color=c, linewidth=1.5, linestyle="--", alpha=0.8)

    ax.set_xlim(float(x[0]), float(x[-1]))
    ax.set_ylim(-0.2, (n - 1) * step + 1.5)
    ax.set_xlabel("Time t")
    ax.set_ylabel("Density Profiles")
    ax.set_title(title)
    ax.set_yticks([i * step + 0.4 for i in range(n)])
    
    # 动态调整字体大小并处理重叠
    ytick_fontsize = max(6, 10 - n // 5)
    ax.set_yticklabels([_short_label(labels[i]) for i in range(n)], fontsize=ytick_fontsize)
    
    if n > 15:
        plt.setp(ax.get_yticklabels(), rotation=30, ha="right")
        
    ax.grid(axis="x", linestyle="--", alpha=0.2)

    if n > 3:
        handles = [plt.Line2D([0], [0], color="tab:blue", lw=2.0, label="Predicted")]
        if true is not None:
            handles.append(plt.Line2D([0], [0], color="black", lw=1.5, linestyle="--", label="True"))
        ax.legend(handles=handles, loc="upper right", fontsize=9)
    else:
        # 当样本数少时，显示每个样本的图例，并区分预测和真实
        handles = []
        for i in range(n):
            c = colors[i % len(colors)]
            handles.append(plt.Line2D([0], [0], color=c, lw=2.0, label=f"Pred {i}"))
            if true is not None:
                handles.append(plt.Line2D([0], [0], color=c, lw=1.5, linestyle="--", label=f"True {i}"))
        ax.legend(handles=handles, loc="upper right", fontsize=8, ncol=2)

    fig.savefig(out_png, dpi=300)
    plt.close(fig)

    # 导出绘图数据为 CSV
    csv_path = out_png.replace(".png", "_data.csv")
    csv_data = {"time": x}
    for i in range(n):
        clean_label = _short_label(labels[i]).replace("|", "_").replace("=", "").replace(",", "").replace(" ", "")
        csv_data[f"pred_{clean_label}"] = pred[i]
        if true is not None:
            csv_data[f"true_{clean_label}"] = true[i]
    pd.DataFrame(csv_data).to_csv(csv_path, index=False)
    print(f"Density evolution data exported to: {csv_path}")


def plot_interactive_hazard_surface(
    grid_t: np.ndarray,
    temperature_grid: np.ndarray,
    hazard_surface: np.ndarray,
    out_html: str,
    title: str = "3D Hazard Surface",
    survival_surface: Optional[np.ndarray] = None,
    survival_threshold: float = 0.1,
    h_min: Optional[float] = None,
    h_max: Optional[float] = None,
    cmin: Optional[float] = None,
    cmax: Optional[float] = None,
) -> None:
    """
    绘制单个交互式 3D 风险曲面，支持风险区间过滤。
    """
    t_mesh, temp_mesh = np.meshgrid(grid_t, temperature_grid)
    
    h = hazard_surface.copy()
    if survival_surface is not None:
        mask = survival_surface < survival_threshold
        h[mask] = np.nan
    if h_min is not None:
        h[h < h_min] = np.nan
    if h_max is not None:
        h[h >= h_max] = np.nan

    if cmin is None or cmax is None:
        valid_h = h[~np.isnan(h)]
        if len(valid_h) > 0:
            cmin = cmin if cmin is not None else float(np.nanpercentile(hazard_surface, 1))
            cmax = cmax if cmax is not None else float(np.nanpercentile(hazard_surface, 99))
        else:
            cmin, cmax = 0.0, 1.0

    fig = go.Figure(data=[
        go.Surface(x=t_mesh, y=temp_mesh, z=h, colorscale='Viridis', cmin=cmin, cmax=cmax,
                   colorbar=dict(title='Hazard'))
    ])

    camera = dict(eye=dict(x=-1.5, y=-1.5, z=0.5))
    fig.update_layout(
        title=title,
        height=800, width=1000,
        scene=dict(
            xaxis_title='Time t',
            yaxis_title='Temp X1',
            zaxis_title='Hazard',
            camera=camera
        ),
        margin=dict(l=10, r=10, b=10, t=50)
    )

    fig.write_html(out_html)
    print(f"Interactive hazard surface saved to: {out_html}")

    # 导出 CSV 数据
    csv_path = out_html.replace(".html", "_data.csv")
    pd.DataFrame({
        "time": t_mesh.flatten(),
        "temperature": temp_mesh.flatten(),
        "hazard": h.flatten()
    }).to_csv(csv_path, index=False)


def plot_compare_true_pred_by_risk(
    grid_t: np.ndarray,
    temperature_grid: np.ndarray,
    true_hazard_surface: np.ndarray,
    pred_hazard_surface: np.ndarray,
    out_html: str,
    title: str,
    survival_surface: Optional[np.ndarray] = None,
    survival_threshold: float = 0.1,
    h_min: Optional[float] = None,
    h_max: Optional[float] = None,
    cmin: Optional[float] = None,
    cmax: Optional[float] = None,
) -> None:
    t_mesh, temp_mesh = np.meshgrid(grid_t, temperature_grid)
    t_true = true_hazard_surface.copy()
    t_pred = pred_hazard_surface.copy()
    if survival_surface is not None:
        mask = survival_surface < survival_threshold
        t_true[mask] = np.nan
        t_pred[mask] = np.nan
    if h_min is not None:
        t_true[t_true < h_min] = np.nan
        t_pred[t_pred < h_min] = np.nan
    if h_max is not None:
        t_true[t_true >= h_max] = np.nan
        t_pred[t_pred >= h_max] = np.nan
    if cmin is None or cmax is None:
        all_vals = np.concatenate([
            t_true[~np.isnan(t_true)].flatten(),
            t_pred[~np.isnan(t_pred)].flatten()
        ]) if np.any(~np.isnan(t_true)) or np.any(~np.isnan(t_pred)) else np.array([])
        if len(all_vals) > 0:
            cmin = cmin if cmin is not None else float(np.percentile(all_vals, 1))
            cmax = cmax if cmax is not None else float(np.percentile(all_vals, 99))
        else:
            cmin, cmax = 0.0, 1.0
    fig = make_subplots(
        rows=1, cols=2,
        specs=[[{'type': 'surface'}, {'type': 'surface'}]],
        subplot_titles=('True Hazard', 'Predicted Hazard'),
        horizontal_spacing=0.05
    )
    fig.add_trace(
        go.Surface(x=t_mesh, y=temp_mesh, z=t_true, colorscale='Viridis', cmin=cmin, cmax=cmax, showscale=False, name='True'),
        row=1, col=1
    )
    fig.add_trace(
        go.Surface(x=t_mesh, y=temp_mesh, z=t_pred, colorscale='Viridis', cmin=cmin, cmax=cmax, colorbar=dict(title='Hazard'), name='Pred'),
        row=1, col=2
    )
    camera = dict(eye=dict(x=-1.5, y=-1.5, z=0.5))
    fig.update_layout(
        title=title,
        height=800, width=1200,
        scene=dict(xaxis_title='Time t', yaxis_title='Temp X1', zaxis_title='Hazard', camera=camera),
        scene2=dict(xaxis_title='Time t', yaxis_title='Temp X1', zaxis_title='Hazard', camera=camera),
        margin=dict(l=10, r=10, b=10, t=50)
    )
    fig.write_html(out_html)
    csv_path = out_html.replace(".html", "_data.csv")
    pd.DataFrame({
        "time": t_mesh.flatten(),
        "temperature": temp_mesh.flatten(),
        "true_hazard": t_true.flatten(),
        "pred_hazard": t_pred.flatten()
    }).to_csv(csv_path, index=False)
