from __future__ import annotations

import json
import os
from typing import Dict

import optuna
from tqdm.auto import tqdm

from train.train_utils import build_config, run_cv


def _parse_dims(raw: str) -> list[int]:
    return [int(x) for x in raw.split(",") if x.strip()]


def suggest_hparams(trial: optuna.trial.Trial) -> Dict:
    encoder_dims = _parse_dims(
        trial.suggest_categorical("encoder_hidden_dims", ["64", "32", "64, 32"])
    )
    vf_dims = _parse_dims(
        trial.suggest_categorical("vf_hidden_dims", ["32, 32", "32", "64"])
    )
    gumbel_dims = _parse_dims(
        trial.suggest_categorical("gumbel_hidden_dims", ["32"])
    )
    return {
        "network": {
            "encoder_hidden_dims": encoder_dims,
            "latent_dim": trial.suggest_categorical("latent_dim", [32, 64]),
            "vf_hidden_dims": vf_dims,
            "dropout": trial.suggest_float("dropout", 0.0, 0.1),
            "gumbel_hidden_dims": gumbel_dims,
        },
        "train": {
            "learning_rate": trial.suggest_float("learning_rate", 8e-5, 8e-4, log=True),
            "stage2_learning_rate": trial.suggest_float("stage2_learning_rate", 8e-5, 8e-4, log=True),
            "stage2_encoder_lr_scale": trial.suggest_float("stage2_encoder_lr_scale", 0.1, 0.5, log=True),
            "weight_decay": trial.suggest_float("weight_decay", 1e-6, 2e-5, log=True),
            # "rank_loss_weight": trial.suggest_float("rank_loss_weight", 0.01, 0.1),
            "event_weight": trial.suggest_float("event_weight", 0.6, 0.8),
        },
        "ode": {
            "ode_method": trial.suggest_categorical("ode_method", ["euler"]),
            "ode_steps": trial.suggest_categorical("ode_steps", [10]),  # 调优阶段减少 ODE 步数以加速
        },
        "sampling": {
            "mc_samples_eval": trial.suggest_categorical("mc_samples_eval", [256]),  # 调优阶段减少采样数以加速
            "density_grid_size": trial.suggest_categorical("density_grid_size", [100]),  # 调优阶段减少网格大小以加速
        },
    }


def metric_to_target(metrics: Dict[str, float], target_metric: str) -> float:
    key = f"mean_{target_metric}"
    if key not in metrics:
        raise KeyError(f"未找到目标指标: {key}")
    return float(metrics[key])


def tune_model(
    model_name: str,
    base_overrides: Dict,
    train_df,
    out_dir: str,
    cv_ckpt_dir: str,
    force_early_stop: str = "_",
) -> Dict:
    os.makedirs(out_dir, exist_ok=True)
    cfg = build_config(model_name=model_name, extra_overrides=base_overrides)
    direction = cfg.tuning.direction
    target_metric = cfg.tuning.target_metric
    n_trials = cfg.tuning.n_trials
    trial_records = []
    trial_params: Dict[int, Dict] = {}

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective(trial: optuna.trial.Trial) -> float:
        hparams = suggest_hparams(trial)
        cfg_trial = build_config(model_name=model_name, extra_overrides={**base_overrides, **hparams})
        cv_metrics = run_cv(
            model_name,
            cfg_trial,
            train_df,
            cv_ckpt_dir,
            result_root=None,
            force_early_stop=force_early_stop,
            save_weights=False,
            save_results=False,
            verbose=False,
        )
        score = metric_to_target(cv_metrics, target_metric)
        
        # 打印当前 Trial 的平均指标和早停信息
        if "mean_best_epoch_stage1" in cv_metrics:
            s1 = int(round(cv_metrics["mean_best_epoch_stage1"]))
            s2 = int(round(cv_metrics["mean_best_epoch_stage2"]))
            tqdm.write(f"[Trial {trial.number}] Avg {target_metric}: {score:.4f}, Avg EarlyStop: S1={s1}, S2={s2}")
        elif "mean_best_epoch" in cv_metrics:
            ep = int(round(cv_metrics["mean_best_epoch"]))
            tqdm.write(f"[Trial {trial.number}] Avg {target_metric}: {score:.4f}, Avg EarlyStop: {ep}")
        else:
            tqdm.write(f"[Trial {trial.number}] Avg {target_metric}: {score:.4f}")

        row = {"trial": trial.number, "score": score, **trial.params, **cv_metrics}
        trial_records.append(row)
        trial_params[trial.number] = dict(trial.params)
        return score

    study = optuna.create_study(direction=direction)
    pbar = tqdm(range(n_trials), desc=f"{model_name} tuning", leave=True, dynamic_ncols=True)
    for _ in pbar:
        study.optimize(objective, n_trials=1)
    
    best_row = next((r for r in trial_records if int(r["trial"]) == int(study.best_trial.number)), None)
    best_mean_target = float(study.best_value)
    best = {
        "best_trial": study.best_trial.number,
        f"best_mean_{target_metric}": best_mean_target,
        "best_params": trial_params.get(study.best_trial.number, dict(study.best_params)),
    }
    if best_row is not None:
        best["best_cv_metrics"] = {k: float(v) for k, v in best_row.items() if str(k).startswith("mean_")}
        # 为全量训练提供方便的早停字符串建议
        if "mean_best_epoch_stage1" in best["best_cv_metrics"]:
            s1 = int(round(best["best_cv_metrics"]["mean_best_epoch_stage1"]))
            s2 = int(round(best["best_cv_metrics"]["mean_best_epoch_stage2"]))
            best["suggested_early_stop"] = f"{s1}/{s2}"
        elif "mean_best_epoch" in best["best_cv_metrics"]:
            ep = int(round(best["best_cv_metrics"]["mean_best_epoch"]))
            best["suggested_early_stop"] = f"{ep}"
    if target_metric == "c_index":
        best["best_mean_cindex"] = best_mean_target

    # 打印最终总结结果
    tqdm.write("=" * 40)
    tqdm.write(f"[{model_name}] 调优完成!")
    tqdm.write(f"最佳 Trial: {best['best_trial']}")
    tqdm.write(f"最佳平均 {target_metric}: {best_mean_target:.4f}")
    if "suggested_early_stop" in best:
        tqdm.write(f"建议早停轮次 (EarlyStop): {best['suggested_early_stop']}")
    tqdm.write("最佳参数:")
    for k, v in best["best_params"].items():
        tqdm.write(f"  {k}: {v}")
    tqdm.write("=" * 40)

    best_json = os.path.join(out_dir, "tuned_hparams.json")
    os.makedirs(os.path.dirname(best_json), exist_ok=True)
    with open(best_json, "w", encoding="utf-8") as f:
        json.dump(best, f, ensure_ascii=False, indent=2)
    return best
