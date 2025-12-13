#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
分析消融实验结果，对比不同策略的性能
"""

import json
import argparse
from pathlib import Path
from collections import defaultdict


def load_training_log(log_file):
    """加载训练日志文件"""
    if not log_file.exists():
        return None
    
    with open(log_file, 'r') as f:
        data = json.load(f)
    
    return data


def extract_metrics(log_data, metric_keys):
    """从日志中提取指定指标"""
    metrics = {}
    for key in metric_keys:
        if key in log_data:
            values = log_data[key]
            if values:
                metrics[key] = {
                    'max': max(values),
                    'final': values[-1],
                    'values': values
                }
    return metrics


def analyze_experiment(work_dir, experiment_name):
    """分析单个实验的结果"""
    work_path = Path(work_dir)
    
    # 查找日志文件
    log_file = work_path / "vis_data" / "scalars.json"
    
    if not log_file.exists():
        print(f"⚠️  未找到日志文件: {log_file}")
        return None
    
    print(f"\n{'='*60}")
    print(f"分析实验: {experiment_name}")
    print(f"工作目录: {work_dir}")
    print(f"{'='*60}")
    
    log_data = load_training_log(log_file)
    if log_data is None:
        return None
    
    # 关键指标
    key_metrics = [
        'train/loss',
        'val/coco/AP',
        'val/coco/AP50',
        'val/coco/AP75',
        'val/coco/AR',
    ]
    
    metrics = extract_metrics(log_data, key_metrics)
    
    # 打印结果
    print("\n关键指标:")
    for key, value in metrics.items():
        print(f"  {key:30s} Max: {value['max']:.4f}, Final: {value['final']:.4f}")
    
    # 查找最佳checkpoint
    checkpoint_dir = work_path / "checkpoints"
    if checkpoint_dir.exists():
        checkpoints = list(checkpoint_dir.glob("*.pth"))
        if checkpoints:
            best_checkpoint = max(checkpoints, key=lambda x: x.stat().st_mtime)
            print(f"\n最佳checkpoint: {best_checkpoint.name}")
    
    return {
        'name': experiment_name,
        'work_dir': str(work_dir),
        'metrics': metrics,
        'log_data': log_data
    }


def compare_experiments(results):
    """对比多个实验的结果"""
    print(f"\n{'='*60}")
    print("实验结果对比")
    print(f"{'='*60}\n")
    
    # 对比表格
    comparison_metrics = [
        'val/coco/AP',
        'val/coco/AP50',
        'val/coco/AP75',
        'val/coco/AR',
    ]
    
    # 表头
    print(f"{'指标':<30s}", end="")
    for result in results:
        if result:
            print(f"{result['name']:>20s}", end="")
    print()
    print("-" * (30 + 20 * len(results)))
    
    # 数据行
    for metric_key in comparison_metrics:
        print(f"{metric_key:<30s}", end="")
        for result in results:
            if result and metric_key in result['metrics']:
                value = result['metrics'][metric_key]['max']
                print(f"{value:>20.4f}", end="")
            else:
                print(f"{'N/A':>20s}", end="")
        print()
    
    # 找出最佳策略
    print(f"\n{'='*60}")
    print("最佳策略分析")
    print(f"{'='*60}\n")
    
    best_ap = -1
    best_name = None
    
    for result in results:
        if result and 'val/coco/AP' in result['metrics']:
            ap = result['metrics']['val/coco/AP']['max']
            if ap > best_ap:
                best_ap = ap
                best_name = result['name']
    
    if best_name:
        print(f"最佳AP: {best_ap:.4f} ({best_name})")
    
    # 改进幅度
    if len(results) >= 2:
        baseline = results[0]
        if baseline and 'val/coco/AP' in baseline['metrics']:
            baseline_ap = baseline['metrics']['val/coco/AP']['max']
            print(f"\n相对于baseline的改进:")
            for result in results[1:]:
                if result and 'val/coco/AP' in result['metrics']:
                    ap = result['metrics']['val/coco/AP']['max']
                    improvement = ap - baseline_ap
                    improvement_pct = (improvement / baseline_ap) * 100 if baseline_ap > 0 else 0
                    print(f"  {result['name']:30s} "
                          f"AP: {ap:.4f} "
                          f"(+{improvement:.4f}, +{improvement_pct:.2f}%)")


def main():
    parser = argparse.ArgumentParser(description='分析消融实验结果')
    parser.add_argument(
        '--base-dir',
        type=str,
        default='work_dirs/ablation_experiments',
        help='实验结果基础目录')
    parser.add_argument(
        '--experiments',
        nargs='+',
        default=['loss_weight_only', 'weighted_sampling_only', 'combined'],
        help='要分析的实验目录列表')
    parser.add_argument(
        '--names',
        nargs='+',
        default=['Loss Weight Only', 'Weighted Sampling Only', 'Combined'],
        help='实验名称列表')
    
    args = parser.parse_args()
    
    base_dir = Path(args.base_dir)
    
    if len(args.experiments) != len(args.names):
        print("错误：实验目录和名称数量不匹配")
        return
    
    results = []
    for exp_dir, exp_name in zip(args.experiments, args.names):
        work_dir = base_dir / exp_dir
        result = analyze_experiment(work_dir, exp_name)
        results.append(result)
    
    # 对比结果
    valid_results = [r for r in results if r is not None]
    if len(valid_results) > 1:
        compare_experiments(valid_results)
    
    print(f"\n{'='*60}")
    print("分析完成！")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()

