"""
FlowSurv / GumbelFlowSurv Simulation Experiments
================================================
主入口脚本, 用于运行完整的仿真实验
"""

import torch
import os
import sys
import argparse
import json
from datetime import datetime
from typing import List, Optional

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from experiments.config import CONFIG, print_config_summary
from experiments.run_experiments import (
    run_all_experiments,
    run_cv_experiments,
    aggregate_results,
    print_results_table,
    save_results_to_csv
)


def parse_args():
    parser = argparse.ArgumentParser(description='FlowSurv Simulation Experiments')
    parser.add_argument('--quick', action='store_true', help='Quick test mode (1 repeat, 5 epochs)')
    parser.add_argument('--full', action='store_true', help='Full experiment mode (10 repeats)')
    parser.add_argument('--cv', type=int, nargs='?', const=5, default=None, help='K-fold cross validation (default: 5-fold if --cv specified without value)')
    parser.add_argument('--repeats', type=int, default=None, help='Number of repeats (overrides quick/full)')
    parser.add_argument('--models', nargs='+', default=None, help='Models to run (default: all)')
    parser.add_argument('--groups', nargs='+', default=None, help='Experiment groups to run (default: all)')
    parser.add_argument('--output', type=str, default='results', help='Output directory')
    parser.add_argument('--checkpoint', type=str, default=None, help='Checkpoint directory')
    parser.add_argument('--seed', type=int, default=None, help='Override base random seed')
    parser.add_argument('--device', type=str, default=None, help='Device to use (cuda/cpu)')
    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 70)
    print("FlowSurv / GumbelFlowSurv Simulation Experiments")
    print("=" * 70)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    if args.seed is not None:
        CONFIG.experiment.base_seed = args.seed

    # Device selection
    device = torch.device(args.device) if args.device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Quick mode: Reduce epochs for fast testing
    if args.quick:
        print("\n[QUICK MODE] Overriding configuration for fast testing:")
        for model_name, model_cfg in CONFIG.model.configs.items():
            if 'EPOCHS' in model_cfg:
                model_cfg['EPOCHS'] = min(model_cfg['EPOCHS'], 5)
                print(f"  - {model_name}: EPOCHS = {model_cfg['EPOCHS']}")
            if 'WEIBULL_EPOCHS' in model_cfg:
                model_cfg['WEIBULL_EPOCHS'] = min(model_cfg['WEIBULL_EPOCHS'], 5)
                print(f"  - {model_name}: WEIBULL_EPOCHS = {model_cfg['WEIBULL_EPOCHS']}")
    
    print_config_summary()

    model_names = args.models
    if model_names is None:
        model_names = ['LinearCoxPH', 'DeepSurv', 'WeibullAFT', 'RSF',
                      'DeepHit', 'FlowSurv', 'GumbelFlowSurv']

    if args.cv is not None:
        print(f"\n运行配置 (交叉验证模式):")
        print(f"  - 折数: {args.cv}")
        print(f"  - 模型列表: {model_names}")
        print(f"  - 输出目录: {args.output}")
        print(f"  - 检查点目录: {args.checkpoint or 'checkpoints'}")
        print()

        results = run_cv_experiments(
            config=CONFIG,
            model_names=model_names,
            group_names=args.groups,
            n_folds=args.cv,
            save_results=True,
            output_dir=args.output,
            checkpoint_dir=args.checkpoint,
            device=device
        )
    else:
        if args.repeats is not None:
            n_repeats = args.repeats
        elif args.quick:
            n_repeats = 1
        elif args.full:
            n_repeats = 10
        else:
            n_repeats = CONFIG.experiment.n_repeats

        print(f"\n运行配置:")
        print(f"  - 重复次数: {n_repeats}")
        print(f"  - 模型列表: {model_names}")
        print(f"  - 输出目录: {args.output}")
        print(f"  - 检查点目录: {args.checkpoint or 'checkpoints'}")
        print()

        results = run_all_experiments(
            config=CONFIG,
            model_names=model_names,
            group_names=args.groups,
            n_repeats=n_repeats,
            save_results=True,
            output_dir=args.output,
            checkpoint_dir=args.checkpoint,
            device=device
        )

    aggregated = aggregate_results(results)

    print("\n" + "=" * 80)
    print("实验结果汇总")
    print("=" * 80)

    print_results_table(aggregated, 'c_index')
    print()
    print_results_table(aggregated, 'ibs')
    print()

    try:
        print_results_table(aggregated, 'hazard_mse')
        print()
    except:
        pass

    try:
        print_results_table(aggregated, 'density_mse')
        print()
    except:
        pass

    aggregated_file = os.path.join(args.output, 'aggregated_results.json')
    with open(aggregated_file, 'w', encoding='utf-8') as f:
        json.dump({g: {m: {k: [v[0], v[1]] for k, v in metrics.items()}
                      for m, metrics in models.items()}
                 for g, models in aggregated.items()},
                 f, indent=2, ensure_ascii=False)
    print(f"\n聚合结果已保存至：{aggregated_file}")
    
    # 额外保存为 CSV 格式
    print("\n正在保存 CSV 格式结果...")
    detailed_csv, aggregated_csv = save_results_to_csv(results, args.output)
    if detailed_csv:
        print(f"详细结果 CSV: {detailed_csv}")
    if aggregated_csv:
        print(f"聚合结果 CSV: {aggregated_csv}")

    print("\n" + "=" * 70)
    print(f"实验完成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)


if __name__ == "__main__":
    main()
