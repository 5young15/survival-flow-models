from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, List

import pandas as pd
from tqdm.auto import tqdm

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

try:
    from train.train_utils import (
        build_config,
        default_toy_csv_path,
        ensure_toy_dataset_csv,
        load_dataset,
        parse_early_stop_tokens,
        run_cv,
        split_train_test,
        train_single_run,
    )
except ModuleNotFoundError as e:
    print(f"Import from train.train_utils failed: {e}")
    try:
        from train_utils import (
            build_config,
            default_toy_csv_path,
            ensure_toy_dataset_csv,
            load_dataset,
            parse_early_stop_tokens,
            run_cv,
            split_train_test,
            train_single_run,
        )
    except ModuleNotFoundError as e2:
        print(f"Import from train_utils failed: {e2}")
        raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", required=True)
    parser.add_argument("--dataset", type=str, default="non_ph", choices=["non_ph"])
    parser.add_argument("--csv_path", type=str, default=None)
    parser.add_argument("--time_col", type=str, default="time")
    parser.add_argument("--event_col", type=str, default="event")
    parser.add_argument("--early_stop", type=str, default="_")
    parser.add_argument("--cv", type=int, default=0)
    parser.add_argument("--target_metric", type=str, default="c_index")
    return parser.parse_args()


def load_tuned_params(group: str, model_name: str) -> Dict:
    fp_new = os.path.join(ROOT, "results", "tuning_results", group, model_name, "tuned_hparams.json")
    fp_old = os.path.join(ROOT, "results", "tuning_results", group, model_name, "best_params.json")
    suggested_early_stop = "_"
    if os.path.exists(fp_new):
        with open(fp_new, "r", encoding="utf-8") as f:
            data = json.load(f)
        best = data.get("best_params", {})
        suggested_early_stop = data.get("suggested_early_stop", "_")
    elif os.path.exists(fp_old):
        with open(fp_old, "r", encoding="utf-8") as f:
            data = json.load(f)
        best = data.get("best_params", {})
        suggested_early_stop = data.get("suggested_early_stop", "_")
    else:
        return {}
    network_keys = {"encoder_hidden_dims", "latent_dim", "vf_hidden_dims", "time_emb_dim", "dropout", "gumbel_hidden_dims"}
    train_keys = {"learning_rate", "weight_decay", "rank_loss_weight", "event_weight", "early_stop_patience", "max_epochs_stage2"}
    ode_keys = {"ode_method", "ode_steps"}
    sampling_keys = {"mc_samples_eval", "density_grid_size"}
    merged = {"network": {}, "train": {}, "ode": {}, "sampling": {}, "suggested_early_stop": suggested_early_stop}
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


def main() -> None:
    args = parse_args()
    group_name = args.dataset
    csv_path = args.csv_path or default_toy_csv_path(args.dataset)
    if args.csv_path is None:
        ensure_toy_dataset_csv(dataset=args.dataset, csv_path=csv_path, n=8000, seed=42)
    os.makedirs(os.path.join(ROOT, "results", "train_results"), exist_ok=True)
    os.makedirs(os.path.join(ROOT, "cv_checkpoints"), exist_ok=True)
    os.makedirs(os.path.join(ROOT, "checkpoints"), exist_ok=True)
    early_tokens = parse_early_stop_tokens(args.early_stop, len(args.models))
    summary: List[Dict] = []
    tqdm.write(f"[train-main] dataset={args.dataset} models={args.models} cv={args.cv}")
    pairs = list(zip(args.models, early_tokens))
    # 如果只有一个模型，不显示外层进度条，保持输出整洁
    for model_name, token in tqdm(pairs, desc="models train", leave=True, disable=len(pairs) <= 1):
        tqdm.write(f"[train-main] start model={model_name}")
        tuned_overrides = load_tuned_params(group_name, model_name)
        
        # 如果用户没有指定早停轮次，且调优结果中有建议值，则使用建议值
        effective_token = token
        if token == "_" and tuned_overrides.get("suggested_early_stop", "_") != "_":
            effective_token = tuned_overrides["suggested_early_stop"]
            tqdm.write(f"[train-main] use suggested early stop: {effective_token}")

        base_overrides = {
            "data": {"csv_path": csv_path, "time_col": args.time_col, "event_col": args.event_col, "train_ratio": 0.8, "random_seed": 42},
            "tuning": {"target_metric": args.target_metric, "cv_folds": max(args.cv, 3)},
            "runtime": {"group": group_name, "models": args.models},
        }
        merged = {**base_overrides}
        for key in ["network", "train", "ode", "sampling"]:
            merged[key] = {**base_overrides.get(key, {}), **tuned_overrides.get(key, {})}
        cfg = build_config(model_name=model_name, extra_overrides=merged)
        df = load_dataset(cfg.data.csv_path, cfg.data.time_col, cfg.data.event_col, cfg.data.feature_cols)
        tr_df, te_df = split_train_test(df, train_ratio=cfg.data.train_ratio, seed=cfg.data.random_seed)
        if args.cv and args.cv > 1:
            cv_weights_dir = os.path.join(ROOT, "cv_checkpoints", group_name, model_name)
            cv_result_dir = os.path.join(ROOT, "results", "train_results", group_name, model_name, "cv")
            cv_metrics = run_cv(
                model_name,
                cfg,
                tr_df,
                cv_weights_dir,
                result_root=cv_result_dir,
                force_early_stop=effective_token,
                save_weights=True,
                save_results=True,
            )
        else:
            cv_metrics = {}
        out_dir = os.path.join(ROOT, "checkpoints", group_name, model_name)
        result_dir = os.path.join(ROOT, "results", "train_results", group_name, model_name)
        metrics = train_single_run(model_name, cfg, tr_df, None, te_df, out_dir, result_dir=result_dir, force_early_stop=effective_token)
        summary.append({"model": model_name, **cv_metrics, **metrics})
        tqdm.write(f"[train-main] done model={model_name}")

    out_dir = os.path.join(ROOT, "results")
    os.makedirs(out_dir, exist_ok=True)
    out_csv = os.path.join(out_dir, f"train_only_summary_{group_name}.csv")
    pd.DataFrame(summary).to_csv(out_csv, index=False)
    tqdm.write(out_csv)


if __name__ == "__main__":
    main()
