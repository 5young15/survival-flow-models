from __future__ import annotations

import argparse
import os
import sys
from tqdm.auto import tqdm

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from train.train_utils import build_config, load_dataset, parse_early_stop_tokens, split_train_test
from train.train_utils import default_toy_csv_path, ensure_toy_dataset_csv
try:
    from tuning.tuning_utils import tune_model
except ModuleNotFoundError:
    from tuning_utils import tune_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", required=True)
    parser.add_argument("--dataset", type=str, default="non_ph", choices=["non_ph"])
    parser.add_argument("--csv_path", type=str, default=None)
    parser.add_argument("--time_col", type=str, default="time")
    parser.add_argument("--event_col", type=str, default="event")
    parser.add_argument("--early_stop", type=str, default="_")
    parser.add_argument("--cv", type=int, default=3)
    parser.add_argument("--target_metric", type=str, default="c_index")
    parser.add_argument("--n_trials", type=int, default=20)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    group_name = args.dataset
    csv_path = args.csv_path or default_toy_csv_path(args.dataset)
    if args.csv_path is None:
        ensure_toy_dataset_csv(dataset=args.dataset, csv_path=csv_path, n=8000, seed=42)
    os.makedirs(os.path.join(ROOT, "results", "tuning_results"), exist_ok=True)
    os.makedirs(os.path.join(ROOT, "results", "train_results"), exist_ok=True)
    os.makedirs(os.path.join(ROOT, "cv_checkpoints"), exist_ok=True)
    os.makedirs(os.path.join(ROOT, "checkpoints"), exist_ok=True)
    base_overrides = {
        "data": {"csv_path": csv_path, "time_col": args.time_col, "event_col": args.event_col, "train_ratio": 0.8, "random_seed": 42},
        "tuning": {"cv_folds": args.cv, "target_metric": args.target_metric, "n_trials": args.n_trials},
        "runtime": {"group": group_name, "models": args.models},
    }
    probe_cfg = build_config(args.models[0], extra_overrides=base_overrides)
    df = load_dataset(
        csv_path=csv_path,
        time_col=probe_cfg.data.time_col,
        event_col=probe_cfg.data.event_col,
        feature_cols=probe_cfg.data.feature_cols,
    )
    train_df, _ = split_train_test(df, train_ratio=0.8, seed=42)

    early_tokens = parse_early_stop_tokens(args.early_stop, len(args.models))
    pairs = list(zip(args.models, early_tokens))
    for model_name, token in tqdm(pairs, desc="models tuning", leave=True):
        tune_out = os.path.join(ROOT, "results", "tuning_results", group_name, model_name)
        cv_ckpt_dir = os.path.join(ROOT, "cv_checkpoints", group_name, model_name)
        tune_model(
            model_name=model_name,
            base_overrides=base_overrides,
            train_df=train_df,
            out_dir=tune_out,
            cv_ckpt_dir=cv_ckpt_dir,
            force_early_stop=token,
        )


if __name__ == "__main__":
    main()
