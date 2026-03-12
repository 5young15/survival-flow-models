from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from typing import Optional

import datetime
if not hasattr(datetime, "UTC"):
    import datetime as dt
    dt.UTC = dt.timezone.utc

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from lifelines import CoxPHFitter, KaplanMeierFitter
from tqdm.auto import tqdm

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from config import default_experiment_config, preset_config, with_overrides
from metrics.metrics_utils import dynamic_auc_score, dynamic_c_index_score
from models.flow_matching.compute_utils import build_time_grid, integrate_ode
from models.flow_matching.gaussian_flow_matching import GaussianFlowMatchingModel
from models.flow_matching.gumbel_flow_matching import GumbelFlowMatchingModel
from toy_experiments.generate_toy_data import NonPHSimulatedData
try:
    from plot.plot_utils import (
        ensure_dir,
        plot_crossing_survival_curves,
        plot_interactive_hazard_surface,
        plot_compare_true_pred_by_risk,
        plot_flow_density_evolution,
        plot_dynamic_metric,
    )
except ModuleNotFoundError:
    from plot_utils import (
        ensure_dir,
        plot_crossing_survival_curves,
        plot_interactive_hazard_surface,
        plot_compare_true_pred_by_risk,
        plot_flow_density_evolution,
        plot_dynamic_metric,
    )

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="non_ph", choices=["non_ph"])
    parser.add_argument("--model", type=str, default="gumbel_flow_matching")
    parser.add_argument("--plot_n", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--out_dir", type=str, default=os.path.join(ROOT, "results", "plot_results"))
    # 允许用户自定义检查点、训练结果与调参结果根目录或具体路径
    parser.add_argument("--ckpt_root", type=str, default=os.path.join(ROOT, "checkpoints"))
    parser.add_argument("--results_root", type=str, default=os.path.join(ROOT, "results", "train_results"))
    parser.add_argument("--tuning_root", type=str, default=os.path.join(ROOT, "results", "tuning_results"))
    parser.add_argument("--ckpt_path", type=str, default=None)
    parser.add_argument("--cfg_path", type=str, default=None)
    return parser.parse_args()


def load_tuned_params(group: str, model_name: str, tuning_root: str) -> dict:
    # 仅使用最新项目路径
    candidates = [
        os.path.join(tuning_root, group, model_name, "tuned_hparams.json"),
        os.path.join(tuning_root, group, model_name, "best_params.json"),
    ]
    best = {}
    for fp in candidates:
        if os.path.exists(fp):
            with open(fp, "r", encoding="utf-8") as f:
                data = json.load(f)
            best = data.get("best_params", {})
            break
    if not best:
        return {}
    network_keys = {"encoder_hidden_dims", "latent_dim", "vf_hidden_dims", "time_emb_dim", "dropout", "gumbel_hidden_dims"}
    train_keys = {"learning_rate", "weight_decay", "rank_loss_weight", "event_weight", "early_stop_patience"}
    ode_keys = {"ode_method", "ode_steps"}
    sampling_keys = {"mc_samples_eval", "density_grid_size"}
    merged = {"network": {}, "train": {}, "ode": {}, "sampling": {}}
    for k, v in best.items():
        if k in {"encoder_hidden_dims", "vf_hidden_dims", "gumbel_hidden_dims"} and isinstance(v, str):
            v = [int(x) for x in v.split(",") if x.strip()]
        if k in network_keys:
            merged["network"][k] = v
        elif k in train_keys:
            merged["train"][k] = v
        elif k in ode_keys:
            merged["ode"][k] = v
        elif k in sampling_keys:
            merged["sampling"][k] = v
    return merged


def infer_network_overrides_from_state_dict(state_dict: dict, model_name: str) -> dict:
    def _extract_block_index(key: str) -> int:
        parts = key.split(".")
        for i, p in enumerate(parts):
            if p == "blocks" and i + 1 < len(parts):
                try:
                    return int(parts[i + 1])
                except ValueError:
                    return 0
        return 0

    encoder_block_keys = [
        k for k in state_dict.keys() if k.startswith("encoder.blocks.") and k.endswith(".fc.weight") and len(k.split(".")) == 5
    ]
    encoder_block_keys = sorted(encoder_block_keys, key=_extract_block_index)
    
    # 从权重中提取输入维度
    input_dim = int(state_dict[encoder_block_keys[0]].shape[1]) if encoder_block_keys else 0
    
    # hidden_dims 是除最后一个 block 外的输出维度
    encoder_hidden_dims = [int(state_dict[k].shape[0]) for k in encoder_block_keys[:-1]]
    latent_dim = int(state_dict["encoder.final_norm.weight"].shape[0])

    vf_block_keys = [
        k
        for k in state_dict.keys()
        if k.startswith("vector_field.blocks.") and k.endswith(".fc.weight") and len(k.split(".")) == 5
    ]
    vf_block_keys = sorted(vf_block_keys, key=_extract_block_index)
    # 对于 VectorFieldNet, blocks 数量等于 hidden_dims 数量
    vf_hidden_dims = [int(state_dict[k].shape[0]) for k in vf_block_keys]
    vf_input_dim = int(state_dict[vf_block_keys[0]].shape[1]) if vf_block_keys else 0
    time_emb_dim = int(vf_input_dim - latent_dim - 1) if vf_input_dim else 32
    net = {
        "input_dim": input_dim,
        "encoder_hidden_dims": encoder_hidden_dims,
        "latent_dim": latent_dim,
        "vf_hidden_dims": vf_hidden_dims,
        "time_emb_dim": time_emb_dim,
    }
    if "gumbel" in model_name.lower():
        gumbel_linear_keys = sorted([k for k in state_dict.keys() if k.startswith("gumbel_head.net.net.") and k.endswith(".weight")])
        gumbel_hidden_dims = []
        if len(gumbel_linear_keys) >= 2:
            for k in gumbel_linear_keys[:-1]:
                gumbel_hidden_dims.append(int(state_dict[k].shape[0]))
        net["gumbel_hidden_dims"] = gumbel_hidden_dims
    return {"network": net}


def resolve_device(device: str) -> str:
    if device.startswith("cuda") and not torch.cuda.is_available():
        return "cpu"
    return device


def build_config(model_name: str, extra_overrides: dict):
    cfg = default_experiment_config()
    cfg = with_overrides(cfg, preset_config(model_name))
    cfg = with_overrides(cfg, extra_overrides)
    cfg.validate_none()
    return cfg


def build_model(model_name: str, input_dim: int, cfg):
    ncfg = cfg.network
    encoder_hidden_dims = list(ncfg.encoder_hidden_dims or [])
    vf_hidden_dims = list(ncfg.vf_hidden_dims or [])
    gumbel_hidden_dims = list(ncfg.gumbel_hidden_dims or [])
    if "gumbel" in model_name.lower():
        return GumbelFlowMatchingModel(
            input_dim=input_dim,
            encoder_hidden_dims=encoder_hidden_dims,
            latent_dim=ncfg.latent_dim,
            vf_hidden_dims=vf_hidden_dims,
            time_emb_dim=ncfg.time_emb_dim,
            gumbel_hidden_dims=gumbel_hidden_dims,
            dropout=ncfg.dropout,
        )
    return GaussianFlowMatchingModel(
        input_dim=input_dim,
        encoder_hidden_dims=encoder_hidden_dims,
        latent_dim=ncfg.latent_dim,
        vf_hidden_dims=vf_hidden_dims,
        time_emb_dim=ncfg.time_emb_dim,
        dropout=ncfg.dropout,
    )


def make_ode_solver(model, ode_method: str, ode_steps: int):
    def ode_solver(y0, field_fn):
        return integrate_ode(y0=y0, field_fn=field_fn, steps=ode_steps, method=ode_method)

    ode_solver.ode_method = ode_method
    ode_solver.ode_steps = ode_steps
    return ode_solver


def select_typical_masks(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    temp = df["x0"].to_numpy(dtype=np.float32)
    vib = df["x1"].to_numpy(dtype=np.float32)
    mat = df["x2"].to_numpy(dtype=np.int32)
    windows = [(0.20, 0.20), (0.30, 0.25), (0.40, 0.30), (0.50, 0.35)]
    for temp_thr, vib_thr in windows:
        base = (np.abs(temp) <= temp_thr) & (np.abs(vib - 0.5) <= vib_thr)
        mask_a = base & (mat == 0)
        mask_b = base & (mat == 1)
        if int(mask_a.sum()) >= 30 and int(mask_b.sum()) >= 30:
            return mask_a, mask_b
    return mat == 0, mat == 1


def km_curve_on_grid(times: np.ndarray, events: np.ndarray, grid_t: np.ndarray) -> np.ndarray:
    kmf = KaplanMeierFitter()
    kmf.fit(times, event_observed=events)
    surv_vals = kmf.survival_function_at_times(grid_t).to_numpy(dtype=np.float32)
    return surv_vals


def get_cox_survival_predictions(train_df: pd.DataFrame, test_df: pd.DataFrame, feature_cols: list[str], grid_t: np.ndarray) -> np.ndarray:
    cox_df = train_df[feature_cols + ["time", "event"]].copy()
    # 增加一个小 penalizer 防止奇异矩阵
    cox = CoxPHFitter(penalizer=1e-4)
    cox.fit(cox_df, duration_col="time", event_col="event", show_progress=False)
    surv_df = cox.predict_survival_function(test_df[feature_cols], times=grid_t)
    return surv_df.to_numpy(dtype=np.float32).T


def build_density_sample_ids(df: pd.DataFrame, hazard: np.ndarray) -> np.ndarray:
    material = df["x2"].to_numpy(dtype=np.int32)
    idx_a = np.where(material == 0)[0]
    idx_b = np.where(material == 1)[0]
    
    ids = []
    # 随机选取 Material A 中一个样本
    if len(idx_a) > 0:
        ids.append(np.random.choice(idx_a))
    # 随机选取 Material B 中一个样本
    if len(idx_b) > 0:
        ids.append(np.random.choice(idx_b))
        
    return np.array(ids, dtype=int)


def main() -> None:
    args = parse_args()
    group_name = args.dataset
    args.out_dir = os.path.join(args.out_dir, args.dataset, args.model)
    ensure_dir(args.out_dir)
    # 解析检查点与配置路径（支持新根目录与旧路径回退）
    ckpt_candidates = []
    if args.ckpt_path:
        ckpt_candidates.append(args.ckpt_path)
    ckpt_candidates.append(os.path.join(args.ckpt_root, group_name, args.model, "model.pt"))
    ckpt_path = next((p for p in ckpt_candidates if os.path.exists(p)), None)
    if not ckpt_path:
        msg = " 未找到可用的检查点文件，依次尝试：\n" + "\n".join([f"- {p}" for p in ckpt_candidates])
        raise FileNotFoundError(msg)

    cfg_candidates = []
    if args.cfg_path:
        cfg_candidates.append(args.cfg_path)
    cfg_candidates.append(os.path.join(args.results_root, group_name, args.model, "config_used.json"))
    cfg_path = next((p for p in cfg_candidates if os.path.exists(p)), None)

    tuned_overrides = load_tuned_params(group_name, args.model, args.tuning_root)
    cfg_overrides = {}
    if cfg_path and os.path.exists(cfg_path):
        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg_overrides = json.load(f)
    device = resolve_device(args.device)
    state_dict = torch.load(ckpt_path, map_location=device)
    inferred_overrides = infer_network_overrides_from_state_dict(state_dict, args.model)
    
    merged_overrides = dict(tuned_overrides)
    for key in ["data", "network", "train", "ode", "sampling", "tuning", "runtime"]:
        if key in cfg_overrides:
            merged_overrides[key] = {**merged_overrides.get(key, {}), **cfg_overrides.get(key, {})}
    merged_overrides["network"] = {**merged_overrides.get("network", {}), **inferred_overrides["network"]}

    inferred_input_dim = inferred_overrides["network"].get("input_dim", 0)
    
    cfg = build_config(model_name=args.model, extra_overrides=merged_overrides)
    cfg.train.device = device

    np.random.seed(args.seed + 1)
    data_gen = NonPHSimulatedData()
    # 生成训练集 (用于训练 Cox 和 IPCW)
    train_df = data_gen.generate(n=10000, seed=args.seed)
    # 生成测试/绘图集
    plot_df = data_gen.generate(n=args.plot_n, seed=args.seed + 1)
    
    plot_csv = os.path.join(args.out_dir, f"{group_name}_{args.model}_plot_dataset.csv")
    plot_df.to_csv(plot_csv, index=False)
    print(f"Plot dataset saved to {plot_csv}")
    print(f"Generated train_df: {len(train_df)} samples, plot_df: {len(plot_df)} samples")
    
    feature_cols = [c for c in plot_df.columns if c not in [cfg.data.time_col, cfg.data.event_col, "risk_true", "group"]]
    
    actual_input_dim = len(feature_cols)
    print(f"Inferred input dim: {inferred_input_dim}, Actual input dim: {actual_input_dim}")
    if inferred_input_dim > 0 and inferred_input_dim != actual_input_dim:
        raise ValueError(
            f"Checkpoint input_dim={inferred_input_dim} 与绘图特征维度={actual_input_dim} 不一致。"
            "请使用当前数据特征重新训练模型后再绘图。"
        )

    x = torch.tensor(plot_df[feature_cols].values, dtype=torch.float32, device=device)
    t_ref = torch.tensor(plot_df[cfg.data.time_col].values, dtype=torch.float32, device=device)
    model = build_model(args.model, input_dim=len(feature_cols), cfg=cfg).to(device)
    model.load_state_dict(state_dict)
    model.eval()
    grid_t = build_time_grid(t_ref, cfg.sampling.density_grid_size)
    ode_solver = make_ode_solver(model, cfg.ode.ode_method, cfg.ode.ode_steps)
    # 分批预测以避免 OOM (尤其在 CPU 上处理大量样本时)
    batch_size = 1000
    bundles = []
    print(f"Loaded model from {ckpt_path}")
    print(f"Starting prediction for {x.shape[0]} samples in batches of {batch_size}...")
    try:
        with torch.no_grad():
            for i in tqdm(range(0, x.shape[0], batch_size), desc="Predicting Risk Surface"):
                x_batch = x[i : i + batch_size]
                b = model.predict_bundle(
                    x_batch,
                    ode_solver=ode_solver,
                    grid_t=grid_t,
                    mc_samples=cfg.sampling.mc_samples_eval,
                    method=str(cfg.sampling.survival_method).lower(),
                )
                # 转为 numpy 并立即释放 tensor 以节省内存
                bundles.append({k: v.detach().cpu().numpy() for k, v in b.items()})
    except Exception as e:
        print(f"Error during prediction: {e}")
        import traceback
        traceback.print_exc()
        raise e
    print("Prediction finished, merging results...")
    sys.stdout.flush()
    
    # 合并结果
    if not bundles:
        print("No predictions were made. Check data loading or model.")
        return

    bundle_np = {}
    for k in bundles[0].keys():
        bundle_np[k] = np.concatenate([b[k] for b in bundles], axis=0)
    
    # 补充 grid_t 以满足 plot_utils 中的函数需求
    bundle_np["grid_t"] = grid_t.detach().cpu().numpy()

    grid_t_np = bundle_np["grid_t"]

    hazard_pred = np.asarray(bundle_np["hazard"], dtype=np.float32)
    survival_pred = np.asarray(bundle_np["survival"], dtype=np.float32)
    density_pred = np.asarray(bundle_np["density"], dtype=np.float32)

    mask_a, mask_b = select_typical_masks(plot_df)
    if int(mask_a.sum()) == 0 or int(mask_b.sum()) == 0:
        mask_a = plot_df["x2"].to_numpy(dtype=np.int32) == 0
        mask_b = ~mask_a

    km_a = km_curve_on_grid(
        plot_df.loc[mask_a, "time"].to_numpy(dtype=np.float32),
        plot_df.loc[mask_a, "event"].to_numpy(dtype=np.int32),
        grid_t_np,
    )
    km_b = km_curve_on_grid(
        plot_df.loc[mask_b, "time"].to_numpy(dtype=np.float32),
        plot_df.loc[mask_b, "event"].to_numpy(dtype=np.int32),
        grid_t_np,
    )
    flow_a = survival_pred[mask_a].mean(axis=0)
    flow_b = survival_pred[mask_b].mean(axis=0)
    
    # 获取 Cox 模型预测并计算平均生存曲线
    print("Fitting Cox model...")
    sys.stdout.flush()
    # 使用 train_df 训练 Cox 模型
    cox_surv = get_cox_survival_predictions(train_df, plot_df, feature_cols, grid_t_np)
    cox_a = cox_surv[mask_a].mean(axis=0)
    cox_b = cox_surv[mask_b].mean(axis=0)
    
    # 计算动态指标
    print("Calculating dynamic metrics for Flow model...")
    sys.stdout.flush()
    # 使用 train_df 作为 IPCW 的训练集分布
    time_train = train_df["time"].values
    event_train = train_df["event"].values
    
    # 测试集是 plot_df
    time_test = plot_df["time"].values
    event_test = plot_df["event"].values
    
    # 降采样 grid_t 用于指标计算以加速并避免潜在的边界问题
    # 只取中间部分的时间点，避开极端的开始和结束
    # 增加点数到 50 以获得更平滑的曲线
    eval_indices = np.linspace(10, len(grid_t_np)-10, 50, dtype=int)
    eval_times = grid_t_np[eval_indices]
    
    # 我们需要从 survival_pred 中提取对应时间点的预测
    # survival_pred 的列对应 grid_t_np
    surv_pred_eval = survival_pred[:, eval_indices]
    cox_surv_eval = cox_surv[:, eval_indices]

    try:
        # 恢复实际计算
        _, flow_auc_vals = dynamic_auc_score(
            time_train, event_train, time_test, event_test, surv_pred_eval, eval_times
        )
        _, flow_c_index_vals = dynamic_c_index_score(
            time_train, event_train, time_test, event_test, surv_pred_eval, eval_times
        )
        
        print(f"Dynamic metrics for Flow model calculated. Mean AUC: {np.nanmean(flow_auc_vals):.4f}")
        sys.stdout.flush()
        
        print("Calculating dynamic metrics for Cox model...")
        sys.stdout.flush()
        _, cox_auc_vals = dynamic_auc_score(
            time_train, event_train, time_test, event_test, cox_surv_eval, eval_times
        )
        _, cox_c_index_vals = dynamic_c_index_score(
            time_train, event_train, time_test, event_test, cox_surv_eval, eval_times
        )
        
        print(f"Dynamic metrics for Cox model calculated. Mean AUC: {np.nanmean(cox_auc_vals):.4f}")
        sys.stdout.flush()
        
        # 保存并绘制 Dynamic AUC
        auc_png = os.path.join(args.out_dir, f"{group_name}_{args.model}_dynamic_auc.png")
        plot_dynamic_metric(
            grid_t=eval_times,
            cox_vals=cox_auc_vals,
            flow_vals=flow_auc_vals,
            metric_label="AUC",
            out_png=auc_png,
            title=f"Cumulative Dynamic AUC: Cox vs {args.model}"
        )
        print(f"Dynamic AUC plot: {auc_png}")

        # 保存并绘制 Dynamic C-index
        cindex_png = os.path.join(args.out_dir, f"{group_name}_{args.model}_dynamic_cindex.png")
        plot_dynamic_metric(
            grid_t=eval_times,
            cox_vals=cox_c_index_vals,
            flow_vals=flow_c_index_vals,
            metric_label="C-index",
            out_png=cindex_png,
            title=f"Dynamic C-index: Cox vs {args.model}"
        )
        print(f"Dynamic C-index plot: {cindex_png}")
        sys.stdout.flush()
    except Exception as e:
        print(f"Error calculating dynamic metrics: {e}")
        import traceback
        traceback.print_exc()
        sys.stdout.flush()

    crossing_png = os.path.join(args.out_dir, f"{group_name}_{args.model}_crossing_survival_curves.png")
    plot_crossing_survival_curves(
        grid_t=grid_t_np,
        km_a=km_a,
        km_b=km_b,
        flow_a=flow_a,
        flow_b=flow_b,
        cox_a=cox_a,
        cox_b=cox_b,
        out_png=crossing_png,
        title=f"Crossing Survival Curves: KM vs Cox vs {args.model}",
    )
    print(f"Crossing survival curve plot: {crossing_png}")

    temp_grid = np.linspace(-1.0, 1.0, 61, dtype=np.float32)
    x_surface_a = np.column_stack(
        [
            temp_grid,
            np.full_like(temp_grid, 0.5, dtype=np.float32),
            np.zeros_like(temp_grid, dtype=np.float32),
        ]
    )
    x_surface_b = np.column_stack(
        [
            temp_grid,
            np.full_like(temp_grid, 0.5, dtype=np.float32),
            np.ones_like(temp_grid, dtype=np.float32),
        ]
    )
    with torch.no_grad():
        surface_bundle_a = model.predict_bundle(
            torch.tensor(x_surface_a, dtype=torch.float32, device=device),
            ode_solver=ode_solver,
            grid_t=grid_t,
            mc_samples=cfg.sampling.mc_samples_eval,
            method="density",
        )
        surface_bundle_b = model.predict_bundle(
            torch.tensor(x_surface_b, dtype=torch.float32, device=device),
            ode_solver=ode_solver,
            grid_t=grid_t,
            mc_samples=cfg.sampling.mc_samples_eval,
            method="density",
        )
    hazard_surface_a = surface_bundle_a["hazard"].detach().cpu().numpy()
    hazard_surface_b = surface_bundle_b["hazard"].detach().cpu().numpy()
    survival_surface_a = surface_bundle_a["survival"].detach().cpu().numpy()
    survival_surface_b = surface_bundle_b["survival"].detach().cpu().numpy()

    # 计算真实 3D 风险表面
    true_hazard_surface_a = data_gen.compute_true_hazard(x_surface_a, grid_t_np)
    true_hazard_surface_b = data_gen.compute_true_hazard(x_surface_b, grid_t_np)

    # -------------------------------------------------------------------------
    # 1. Interactive Hazard Surface: combine True & Pred by risk (Low/High)
    # -------------------------------------------------------------------------
    for mat_suffix, hazard_surface, true_hazard_surface, survival_surface in [
        ("material_a", hazard_surface_a, true_hazard_surface_a, survival_surface_a),
        ("material_b", hazard_surface_b, true_hazard_surface_b, survival_surface_b),
    ]:
        # 确定风险阈值 (使用预测风险的中位数)
        valid_h = hazard_surface[~np.isnan(hazard_surface)]
        threshold = np.median(valid_h) if len(valid_h) > 0 else 1.0
        
        # 统一颜色映射范围
        all_vals = [hazard_surface]
        if true_hazard_surface is not None:
            all_vals.append(true_hazard_surface)
        all_vals_flat = np.concatenate([v.flatten() for v in all_vals])
        all_vals_flat = all_vals_flat[~np.isnan(all_vals_flat)]
        cmin, cmax = None, None
        if len(all_vals_flat) > 0:
            cmin, cmax = float(np.percentile(all_vals_flat, 1)), float(np.percentile(all_vals_flat, 99))

        # Low risk: True & Pred side-by-side
        if true_hazard_surface is not None:
            plot_compare_true_pred_by_risk(
                grid_t=grid_t_np,
                temperature_grid=temp_grid,
                true_hazard_surface=true_hazard_surface,
                pred_hazard_surface=hazard_surface,
                out_html=os.path.join(args.out_dir, f"{group_name}_{args.model}_hazard_low_{mat_suffix}.html"),
                title=f"{mat_suffix} - Low Risk (True vs Pred)",
                survival_surface=survival_surface,
                h_max=threshold,
                cmin=cmin, cmax=cmax
            )
            # High risk: True & Pred side-by-side
            plot_compare_true_pred_by_risk(
                grid_t=grid_t_np,
                temperature_grid=temp_grid,
                true_hazard_surface=true_hazard_surface,
                pred_hazard_surface=hazard_surface,
                out_html=os.path.join(args.out_dir, f"{group_name}_{args.model}_hazard_high_{mat_suffix}.html"),
                title=f"{mat_suffix} - High Risk (True vs Pred)",
                survival_surface=survival_surface,
                h_min=threshold,
                cmin=cmin, cmax=cmax
            )
        print(f"Combined hazard surfaces for {mat_suffix} generated in {args.out_dir}")

    # -------------------------------------------------------------------------
    # 2. Density Evolution (Restore missing density plots)
    # -------------------------------------------------------------------------
    density_ids = build_density_sample_ids(plot_df, hazard_pred)
    density_curves = density_pred[density_ids]
    
    # 计算这些样本的真实密度
    x_density = plot_df.iloc[density_ids][feature_cols].to_numpy(dtype=np.float32)
    true_density_curves = data_gen.compute_true_density(x_density, grid_t_np)

    labels = [
        f"Material {'A' if plot_df.iloc[i]['x2'] == 0 else 'B'} (ID={i})"
        for i in density_ids
    ]
    density_png = os.path.join(args.out_dir, f"{group_name}_{args.model}_density_evolution.png")
    plot_flow_density_evolution(
        grid_t=grid_t_np,
        densities=density_curves,
        labels=labels,
        out_png=density_png,
        title=f"Density Evolution Comparison (True vs {args.model})",
        true_densities=true_density_curves,
    )
    print(f"Density evolution plot: {density_png}")


if __name__ == "__main__":
    main()
