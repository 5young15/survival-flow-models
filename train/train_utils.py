from __future__ import annotations

import json
import os
import shutil
from dataclasses import asdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import KFold, train_test_split
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset
from tqdm.auto import tqdm

from config import ExperimentConfig, default_experiment_config, preset_config, with_overrides
from metrics.metrics_utils import evaluate_all_metrics
from models.flow_matching.compute_utils import build_time_grid, integrate_ode
from models.flow_matching.gaussian_flow_matching import GaussianFlowMatchingModel
from models.flow_matching.gumbel_flow_matching import GumbelFlowMatchingModel
from plot.plot_utils import plot_two_stage_training_curve, risk_from_bundle_log_hazard


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_device(device: str) -> str:
    if device.startswith("cuda") and not torch.cuda.is_available():
        return "cpu"
    return device


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def ensure_parent(path: str) -> None:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DEFAULT_TOY_DATA_DIR = os.path.join(_PROJECT_ROOT, "results", "toy_datasets")


def default_toy_csv_path(dataset: str) -> str:
    return os.path.join(_DEFAULT_TOY_DATA_DIR, "toy_non_ph_dataset.csv")


def ensure_toy_dataset_csv(
    dataset: str,
    csv_path: str,
    n: int = 8000,
    seed: int = 42,
    **kwargs,
) -> str:
    if os.path.exists(csv_path):
        try:
            existing_df = pd.read_csv(csv_path)
            if "group" in existing_df.columns:
                existing_df = existing_df.drop(columns=["group"])
                existing_df.to_csv(csv_path, index=False)
        except Exception:
            pass
        return csv_path

    ensure_parent(csv_path)
    from toy_experiments.generate_toy_data import NonPHSimulatedData
    gen = NonPHSimulatedData()
    df = gen.generate(n=n, seed=seed, include_group=False)
    df.to_csv(csv_path, index=False)
    return csv_path



def build_config(model_name: str, extra_overrides: Optional[Dict] = None) -> ExperimentConfig:
    cfg = default_experiment_config()
    cfg = with_overrides(cfg, preset_config(model_name))
    if extra_overrides:
        cfg = with_overrides(cfg, extra_overrides)
    cfg.validate_none()
    return cfg


def parse_early_stop_tokens(raw: str, n_models: int) -> List[str]:
    tokens = raw.split() if raw.strip() else ["_"] * n_models
    if len(tokens) < n_models:
        tokens = tokens + ["_"] * (n_models - len(tokens))
    return tokens[:n_models]


def load_dataset(csv_path: str, time_col: str, event_col: str, feature_cols: Optional[List[str]] = None) -> pd.DataFrame:
    if not os.path.exists(csv_path):
        raise FileNotFoundError(csv_path)
    df = pd.read_csv(csv_path)
    forbidden_cols = {time_col, event_col, "risk_true", "group"}
    if not feature_cols:
        feature_cols = [c for c in df.columns if c not in forbidden_cols]
    else:
        feature_cols = [c for c in feature_cols if c not in forbidden_cols]
    need = [*feature_cols, time_col, event_col]
    miss = [c for c in need if c not in df.columns]
    if miss:
        raise ValueError(f"数据缺失字段: {miss}")
    return df[need + ([c for c in ["risk_true"] if c in df.columns])]


def split_train_test(df: pd.DataFrame, train_ratio: float, seed: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    idx = np.arange(len(df))
    tr_idx, te_idx = train_test_split(idx, train_size=train_ratio, random_state=seed, shuffle=True)
    return df.iloc[tr_idx].reset_index(drop=True), df.iloc[te_idx].reset_index(drop=True)


def dataframe_to_tensors(df: pd.DataFrame, feature_cols: List[str], time_col: str, event_col: str, device: str) -> Tuple[Tensor, Tensor, Tensor]:
    x = torch.tensor(df[feature_cols].values, dtype=torch.float32, device=device)
    t = torch.tensor(df[time_col].values, dtype=torch.float32, device=device)
    e = torch.tensor(df[event_col].values, dtype=torch.float32, device=device)
    return x, t, e


def build_model(model_name: str, input_dim: int, cfg: ExperimentConfig):
    ncfg = cfg.network
    encoder_hidden_dims = list(ncfg.encoder_hidden_dims or [])
    vf_hidden_dims = list(ncfg.vf_hidden_dims or [])
    gumbel_hidden_dims = list(ncfg.gumbel_hidden_dims or [])
    if "gumbel" in model_name.lower():
        model = GumbelFlowMatchingModel(
            input_dim=input_dim,
            encoder_hidden_dims=encoder_hidden_dims,
            latent_dim=ncfg.latent_dim,
            vf_hidden_dims=vf_hidden_dims,
            time_emb_dim=ncfg.time_emb_dim,
            gumbel_hidden_dims=gumbel_hidden_dims,
            dropout=ncfg.dropout,
        )
    else:
        model = GaussianFlowMatchingModel(
            input_dim=input_dim,
            encoder_hidden_dims=encoder_hidden_dims,
            latent_dim=ncfg.latent_dim,
            vf_hidden_dims=vf_hidden_dims,
            time_emb_dim=ncfg.time_emb_dim,
            dropout=ncfg.dropout,
        )
    return model


def make_ode_solver(model, ode_method: str, ode_steps: int):
    def ode_solver(y0, field_fn):
        return integrate_ode(y0=y0, field_fn=field_fn, steps=ode_steps, method=ode_method)

    ode_solver.ode_method = ode_method
    ode_solver.ode_steps = ode_steps
    return ode_solver


def parse_stage_force_tokens(raw: str) -> Tuple[str, str]:
    text = str(raw).strip()
    # 兼容 "/" 和 "," 分隔符
    for sep in ["/", ","]:
        if sep in text:
            parts = [p.strip() for p in text.split(sep, 1)]
            return parts[0] if parts[0] else "_", parts[1] if parts[1] else "_"
    return "_", text if text else "_"


def train_single_run(
    model_name: str,
    cfg: ExperimentConfig,
    train_df: pd.DataFrame,
    valid_df: Optional[pd.DataFrame],
    eval_df: Optional[pd.DataFrame],
    out_dir: str,
    result_dir: Optional[str] = None,
    force_early_stop: str = "_",
    save_weights: bool = True,
    save_results: bool = True,
    verbose: bool = True,
) -> Dict[str, float]:
    if result_dir is None:
        result_dir = out_dir
    if save_weights:
        ensure_dir(out_dir)
    if save_results:
        ensure_dir(result_dir)
    feature_cols = [c for c in train_df.columns if c not in [cfg.data.time_col, cfg.data.event_col, "risk_true", "group"]]
    device = resolve_device(cfg.train.device)
    cfg.train.device = device
    model = build_model(model_name=model_name, input_dim=len(feature_cols), cfg=cfg).to(device)
    if verbose:
        tqdm.write(f"[train] model={model_name} device={device} samples={len(train_df)} out_dir={out_dir}")
    tr_x, tr_t, tr_e = dataframe_to_tensors(train_df, feature_cols, cfg.data.time_col, cfg.data.event_col, device)
    tr_ds = TensorDataset(tr_x, tr_t, tr_e)
    
    # 设置目标标准化参数
    if hasattr(model, "set_target_normalization"):
        model.set_target_normalization(tr_t, tr_e)
        if verbose:
             # 安全获取 y_mean 和 y_std (如果是 tensor)
            mean_val = model.y_mean.item() if isinstance(model.y_mean, torch.Tensor) else model.y_mean
            std_val = model.y_std.item() if isinstance(model.y_std, torch.Tensor) else model.y_std
            tqdm.write(f"[train] Target normalization set: mean={mean_val:.4f}, std={std_val:.4f}")

    # 尝试使用训练数据初始化 Gumbel 先验参数
    if hasattr(model, "initialize_gumbel_prior"):
        model.initialize_gumbel_prior(tr_t, tr_e)

    tr_loader = DataLoader(tr_ds, batch_size=cfg.train.batch_size, shuffle=True)
    if valid_df is not None:
        va_x, va_t, va_e = dataframe_to_tensors(valid_df, feature_cols, cfg.data.time_col, cfg.data.event_col, device)
    else:
        va_x = va_t = va_e = None

    best_val = float("inf")
    best_epoch = 0
    best_state = None
    patience_count = 0
    loss_rows = []
    stage1_force, stage2_force = parse_stage_force_tokens(force_early_stop)
    best_stage1_epoch = 0
    stage1_last_epoch = 0
    is_two_stage = "gumbel" in model_name.lower() and hasattr(model, "stage1_loss")

    if is_two_stage:
        if verbose:
            tqdm.write(f"[train] stage1 epochs={cfg.train.max_epochs_stage1}")
        # 第一阶段优化器: 只优化 encoder 和 gumbel_head
        # 冻结 vector_field
        for p in model.vector_field.parameters():
            p.requires_grad = False
        for p in model.encoder.parameters():
            p.requires_grad = True
        for p in model.gumbel_head.parameters():
            p.requires_grad = True

        optimizer1 = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=cfg.train.learning_rate,
            weight_decay=cfg.train.weight_decay,
        )

        best_stage1_val = float("inf")
        best_stage1_state = None
        stage1_patience_count = 0
        
        # 第一阶段进度条：仅在 verbose=True 且非调优模式下显示
        stage1_iter = range(1, cfg.train.max_epochs_stage1 + 1)
        if verbose:
            stage1_iter = tqdm(stage1_iter, desc=f"{model_name} Stage1", leave=False)
            
        # 优化：提前创建验证集 DataLoader，避免循环内重复创建
        if valid_df is not None:
            va_loader_s1 = DataLoader(TensorDataset(va_x, va_t, va_e), batch_size=cfg.train.batch_size, shuffle=False)
        else:
            va_loader_s1 = None

        for epoch in stage1_iter:
            model.train()
            total_stage1_loss = torch.tensor(0.0, device=device)
            count = 0
            for bx, bt, be in tr_loader:
                optimizer1.zero_grad()
                stage1_loss = model.stage1_loss(bx, bt, be)
                stage1_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip_norm)
                optimizer1.step()
                total_stage1_loss += stage1_loss.detach()
                count += 1
            tr_stage1 = (total_stage1_loss / max(count, 1)).item()
            stage1_last_epoch = epoch

            if valid_df is not None:
                model.eval()
                with torch.no_grad():
                    total_va_stage1 = torch.tensor(0.0, device=device)
                    va_count = 0
                    for vbx, vbt, vbe in va_loader_s1:
                        v_loss = model.stage1_loss(vbx, vbt, vbe)
                        total_va_stage1 += v_loss.detach()
                        va_count += 1
                    val_stage1 = (total_va_stage1 / max(va_count, 1)).item()
                loss_rows.append({"epoch": epoch, "loss": tr_stage1, "val_loss": val_stage1, "stage": "stage1"})
            else:
                # 全量训练不记录验证损失
                loss_rows.append({"epoch": epoch, "loss": tr_stage1, "stage": "stage1"})
                val_stage1 = tr_stage1 # 仅用于逻辑判断，不记录

            # 决定用于早停和保存的指标
            monitor_score = val_stage1 if valid_df is not None else tr_stage1

            if monitor_score < best_stage1_val:
                best_stage1_val = monitor_score
                best_stage1_epoch = epoch
                best_stage1_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                stage1_patience_count = 0
            else:
                stage1_patience_count += 1
            
            # 早停逻辑：如果有强制指定的 epoch，则优先按强制值停止
            if stage1_force.isdigit() and epoch >= int(stage1_force):
                break
            # 只有在非强制模式且有验证集时，才执行自动早停
            if stage1_force == "_" and valid_df is not None and stage1_patience_count >= cfg.train.early_stop_patience:
                break
        if best_stage1_state is not None:
            model.load_state_dict(best_stage1_state)

    # 第二阶段开始
    if verbose:
        tqdm.write(f"[train] stage2 epochs={cfg.train.max_epochs_stage2}, early_stop={force_early_stop}")

    # 第二阶段优化器配置
    if is_two_stage:
        # 冻结 gumbel_head, 开启 vector_field 和 encoder
        for p in model.gumbel_head.parameters():
            p.requires_grad = False
        for p in model.vector_field.parameters():
            p.requires_grad = True
        for p in model.encoder.parameters():
            p.requires_grad = True

        # 编码器学习率衰减系数
        encoder_lr = cfg.train.stage2_learning_rate * cfg.train.stage2_encoder_lr_scale
        optimizer2 = torch.optim.AdamW(
            [
                {"params": model.vector_field.parameters(), "lr": cfg.train.stage2_learning_rate},
                {"params": model.encoder.parameters(), "lr": encoder_lr},
            ],
            weight_decay=cfg.train.weight_decay,
        )
    else:
        # 非两阶段模型直接训练全部
        for p in model.parameters():
            p.requires_grad = True
        optimizer2 = torch.optim.AdamW(
            model.parameters(),
            lr=cfg.train.learning_rate,
            weight_decay=cfg.train.weight_decay,
        )

    stage2_epoch_offset = stage1_last_epoch if is_two_stage else 0
    stage_label = "stage2" if is_two_stage else "single"
    
    # 第二阶段进度条：仅在 verbose=True 且非调优模式下显示
    stage2_iter = range(1, cfg.train.max_epochs_stage2 + 1)
    if verbose:
        stage2_iter = tqdm(stage2_iter, desc=f"{model_name} Stage2", leave=False)

    # 优化：提前创建验证集 DataLoader
    if valid_df is not None:
        va_loader_s2 = DataLoader(TensorDataset(va_x, va_t, va_e), batch_size=cfg.train.batch_size, shuffle=False)
    else:
        va_loader_s2 = None

    for epoch in stage2_iter:
        model.train()
        total_loss = torch.tensor(0.0, device=device)
        count = 0
        for bx, bt, be in tr_loader:
            optimizer2.zero_grad()
            out = model.forward_loss(
                bx,
                bt,
                be,
                rank_weight=cfg.train.rank_loss_weight,
                rank_margin=cfg.train.rank_loss_margin,
                event_weight=cfg.train.event_weight,
                truncation_samples=cfg.sampling.truncation_samples,
                truncation_ode_steps=cfg.ode.ode_steps, 
                truncation_ode_method=cfg.ode.ode_method,
            )
            out.loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip_norm)
            optimizer2.step()
            total_loss += out.loss.detach()
            count += 1
        tr_loss = (total_loss / max(count, 1)).item()
        
        if valid_df is not None:
            model.eval()
            with torch.no_grad():
                total_va_loss = torch.tensor(0.0, device=device)
                va_count = 0
                for vbx, vbt, vbe in va_loader_s2:
                    val_out = model.forward_loss(
                        vbx,
                        vbt,
                        vbe,
                        rank_weight=cfg.train.rank_loss_weight,
                        rank_margin=cfg.train.rank_loss_margin,
                        event_weight=cfg.train.event_weight,
                        truncation_samples=cfg.sampling.truncation_samples,
                        truncation_ode_steps=cfg.ode.ode_steps, 
                        truncation_ode_method=cfg.ode.ode_method,
                    )
                    total_va_loss += val_out.loss.detach()
                    va_count += 1
                val_loss = (total_va_loss / max(va_count, 1)).item()
            loss_rows.append({"epoch": stage2_epoch_offset + epoch, "loss": tr_loss, "val_loss": val_loss, "stage": stage_label})
        else:
            # 全量训练不记录验证损失
            loss_rows.append({"epoch": stage2_epoch_offset + epoch, "loss": tr_loss, "stage": stage_label})
            val_loss = tr_loss # 仅用于逻辑判断，不记录

        # 决定用于早停和保存的指标
        monitor_score = val_loss if valid_df is not None else tr_loss

        if monitor_score < best_val:
            best_val = monitor_score
            best_epoch = epoch
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_count = 0
        else:
            patience_count += 1

        # 早停逻辑：如果有强制指定的 epoch，则优先按强制值停止
        if stage2_force.isdigit() and epoch >= int(stage2_force):
            break
        # 只有在非强制模式且有验证集时，才执行自动早停
        if stage2_force == "_" and valid_df is not None and patience_count >= cfg.train.early_stop_patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    if save_results:
        loss_csv = os.path.join(result_dir, "loss_curve.csv")
        ensure_parent(loss_csv)
        loss_df = pd.DataFrame(loss_rows)
        if not loss_df.empty and "epoch" in loss_df.columns:
            loss_df = loss_df.sort_values("epoch", kind="stable").reset_index(drop=True)
        loss_df.to_csv(loss_csv, index=False)
        best_epoch_for_plot = stage2_epoch_offset + best_epoch if best_epoch > 0 else None
        plot_two_stage_training_curve(loss_csv, os.path.join(result_dir, "loss_curve.png"), best_epoch=best_epoch_for_plot)
    if save_weights:
        ckpt_path = os.path.join(out_dir, "model.pt")
        ensure_parent(ckpt_path)
        torch.save(model.state_dict(), ckpt_path)

    if is_two_stage:
        metrics = {
            "best_val_loss": best_val,
            "best_epoch_stage1": float(best_stage1_epoch),
            "best_epoch_stage2": float(best_epoch),
        }
    else:
        metrics = {"best_val_loss": best_val, "best_epoch": float(best_epoch)}
    target_eval_df = eval_df if eval_df is not None else (valid_df if valid_df is not None else train_df)
    metrics.update(evaluate_model(model, cfg, train_df, target_eval_df))
    if save_results:
        metrics_json = os.path.join(result_dir, "metrics.json")
        ensure_parent(metrics_json)
        with open(metrics_json, "w", encoding="utf-8") as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)
        cfg_json = os.path.join(result_dir, "config_used.json")
        ensure_parent(cfg_json)
        with open(cfg_json, "w", encoding="utf-8") as f:
            json.dump(asdict(cfg), f, ensure_ascii=False, indent=2)
    if verbose:
        tqdm.write(f"[train] done model={model_name} best_epoch={best_epoch} best_val_loss={best_val:.6f}")
    return metrics


def evaluate_model(model, cfg: ExperimentConfig, train_df: pd.DataFrame, test_df: pd.DataFrame) -> Dict[str, float]:
    feature_cols = [c for c in train_df.columns if c not in [cfg.data.time_col, cfg.data.event_col, "risk_true", "group"]]
    device = resolve_device(cfg.train.device)
    tr_x, tr_t, tr_e = dataframe_to_tensors(train_df, feature_cols, cfg.data.time_col, cfg.data.event_col, device)
    te_x, te_t, te_e = dataframe_to_tensors(test_df, feature_cols, cfg.data.time_col, cfg.data.event_col, device)
    ode_solver = make_ode_solver(model, cfg.ode.ode_method, cfg.ode.ode_steps)
    grid_t = build_time_grid(tr_t, cfg.sampling.density_grid_size)
    method = str(cfg.sampling.survival_method).lower()
    
    # 优化：评估阶段采用分批预测，防止大规模数据导致 GPU OOM
    batch_size = cfg.train.batch_size
    te_ds = TensorDataset(te_x)
    te_loader = DataLoader(te_ds, batch_size=batch_size, shuffle=False)
    
    all_bundles = []
    model.eval()
    with torch.no_grad():
        for (bx,) in te_loader:
            bundle = model.predict_bundle(
                bx,
                ode_solver=ode_solver,
                grid_t=grid_t,
                mc_samples=cfg.sampling.mc_samples_eval,
                method=method,
            )
            all_bundles.append(bundle)
            
    # 合并分批结果
    def merge_bundles(bundles):
        keys = bundles[0].keys()
        merged = {}
        for k in keys:
            merged[k] = torch.cat([b[k] for b in bundles], dim=0)
        return merged
        
    bundle = merge_bundles(all_bundles)
    risk = -bundle["median"]
    survival_matrix = bundle["survival"].detach().cpu().numpy()
    eval_times = grid_t.detach().cpu().numpy()

    # 调用综合评估
    metrics = evaluate_all_metrics(
        train_time=tr_t.detach().cpu().numpy(),
        train_event=tr_e.detach().cpu().numpy(),
        test_time=te_t.detach().cpu().numpy(),
        test_event=te_e.detach().cpu().numpy(),
        risk=risk.detach().cpu().numpy(),
        survival_matrix=survival_matrix,
        eval_times=eval_times,
    )
    
    return metrics


def run_cv(
    model_name: str,
    cfg: ExperimentConfig,
    train_df: pd.DataFrame,
    out_root: str,
    result_root: Optional[str] = None,
    force_early_stop: str = "_",
    save_weights: bool = True,
    save_results: bool = True,
    verbose: bool = True,
) -> Dict[str, float]:
    if save_weights:
        ensure_dir(out_root)
        for name in os.listdir(out_root):
            p = os.path.join(out_root, name)
            if os.path.isdir(p):
                shutil.rmtree(p, ignore_errors=True)
            else:
                try:
                    os.remove(p)
                except OSError:
                    pass
    if save_results and result_root is not None:
        ensure_dir(result_root)
        for name in os.listdir(result_root):
            p = os.path.join(result_root, name)
            if os.path.isdir(p):
                shutil.rmtree(p, ignore_errors=True)
            else:
                try:
                    os.remove(p)
                except OSError:
                    pass
    kf = KFold(n_splits=cfg.tuning.cv_folds, shuffle=True, random_state=cfg.data.random_seed)
    fold_metrics = []
    if verbose:
        tqdm.write(f"[cv] model={model_name} folds={cfg.tuning.cv_folds} out_root={out_root}")
    
    # 交叉验证循环：由外部控制是否显示进度条
    # 如果处于调优过程中，verbose 为 False，则不显示此进度条，避免与 Trial 进度条冲突
    cv_iter = kf.split(train_df)
    if verbose:
        cv_iter = tqdm(cv_iter, total=cfg.tuning.cv_folds, desc=f"{model_name} 交叉验证", leave=False)

    for fold, (tr_idx, va_idx) in enumerate(cv_iter, start=0):
        fold_dir = os.path.join(out_root, f"fold_{fold}")
        fold_result_dir = os.path.join(result_root, f"fold_{fold}") if result_root is not None else fold_dir
        if save_weights:
            ensure_dir(fold_dir)
        if save_results:
            ensure_dir(fold_result_dir)
        fold_tr = train_df.iloc[tr_idx].reset_index(drop=True)
        fold_va = train_df.iloc[va_idx].reset_index(drop=True)
        m = train_single_run(
            model_name,
            cfg,
            fold_tr,
            fold_va,
            fold_va,
            fold_dir,
            result_dir=fold_result_dir,
            force_early_stop=force_early_stop,
            save_weights=save_weights,
            save_results=save_results,
            verbose=verbose,
        )
        fold_metrics.append(m)
        if not verbose:
            target = cfg.tuning.target_metric
            score = m.get(target, 0.0)
            if "best_epoch_stage1" in m:
                s1 = int(m["best_epoch_stage1"])
                s2 = int(m["best_epoch_stage2"])
                tqdm.write(f"  Fold {fold}: {target}={score:.4f}, early_stop_stage1={s1}, stage2={s2}")
            else:
                ep = int(m["best_epoch"])
                tqdm.write(f"  Fold {fold}: {target}={score:.4f}, early_stop={ep}")
    keys = sorted(set().union(*[set(m.keys()) for m in fold_metrics]))
    # 记录所有指标的均值，包括早停 epoch
    mean_metrics = {
        f"mean_{k}": float(np.mean([m[k] for m in fold_metrics if k in m]))
        for k in keys
    }
    if save_results and result_root is not None:
        cv_metrics_csv = os.path.join(result_root, "cv_metrics.csv")
        ensure_parent(cv_metrics_csv)
        pd.DataFrame(fold_metrics).to_csv(cv_metrics_csv, index=False)
        cv_summary_json = os.path.join(result_root, "cv_summary.json")
        ensure_parent(cv_summary_json)
        with open(cv_summary_json, "w", encoding="utf-8") as f:
            json.dump(mean_metrics, f, ensure_ascii=False, indent=2)
    return mean_metrics
