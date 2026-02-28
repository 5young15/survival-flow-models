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
    aggregate_results,
    print_results_table
)


def parse_args():
    parser = argparse.ArgumentParser(description='FlowSurv Simulation Experiments')
    parser.add_argument('--quick', action='store_true', help='Quick test mode (2 repeats)')
    parser.add_argument('--full', action='store_true', help='Full experiment mode (10 repeats)')
    parser.add_argument('--models', nargs='+', default=None, help='Models to run (default: all)')
    parser.add_argument('--groups', nargs='+', default=None, help='Experiment groups to run (default: all)')
    parser.add_argument('--output', type=str, default='results', help='Output directory')
    parser.add_argument('--checkpoint', type=str, default=None, help='Checkpoint directory')
    parser.add_argument('--seed', type=int, default=None, help='Override base random seed')
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

    print_config_summary()

    n_repeats = 2 if args.quick else (10 if args.full else CONFIG.experiment.n_repeats)

    model_names = args.models
    if model_names is None:
        model_names = ['LinearCoxPH', 'DeepSurv', 'WeibullAFT', 'RSF',
                      'DeepHit', 'FlowSurv', 'GumbelFlowSurv']

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
        checkpoint_dir=args.checkpoint
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
    print(f"\n聚合结果已保存至: {aggregated_file}")

    print("\n" + "=" * 70)
    print(f"实验完成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)


if __name__ == "__main__":
    main()
