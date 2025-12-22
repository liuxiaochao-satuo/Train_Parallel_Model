#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
绘制消融实验的AP和AR曲线
对比不同策略随训练轮次的变化
"""

import json
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib
import numpy as np

# 设置中文字体
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS']
matplotlib.rcParams['axes.unicode_minus'] = False


def load_scalars_jsonl(jsonl_file):
    """加载JSONL格式的scalars.json文件"""
    data = []
    if not jsonl_file.exists():
        return data
    
    with open(jsonl_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    data.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    
    return data


def extract_epoch_metrics(log_data, metric_key):
    """从日志数据中提取指定指标随epoch的变化"""
    epochs = []
    values = []
    
    # 首先尝试直接使用epoch字段
    for entry in log_data:
        if metric_key in entry and 'epoch' in entry:
            epoch = entry['epoch']
            value = entry[metric_key]
            epochs.append(epoch)
            values.append(value)
    
    # 如果没有找到，尝试通过step推断epoch或使用step作为x轴
    if not epochs:
        # 方法1: 通过查找验证点前后的训练数据来推断epoch
        for i, entry in enumerate(log_data):
            if metric_key in entry:
                # 向前查找最近的训练数据点
                epoch = None
                for j in range(i, max(-1, i-50), -1):
                    if j < len(log_data) and 'epoch' in log_data[j]:
                        epoch = log_data[j]['epoch']
                        break
                
                if epoch is not None:
                    epochs.append(epoch)
                    values.append(entry[metric_key])
                elif 'step' in entry:
                    # 如果找不到epoch，使用step（假设每10个step一个epoch验证）
                    # 或者直接使用step作为x轴
                    step = entry['step']
                    # 尝试从step推断epoch（需要根据实际情况调整）
                    # 这里先使用step，后续可以转换为epoch
                    epochs.append(step)
                    values.append(entry[metric_key])
    
    # 按epoch排序并去重（保留每个epoch的最后一个值）
    if epochs:
        epoch_value_dict = {}
        for e, v in zip(epochs, values):
            epoch_value_dict[e] = v
        
        sorted_epochs = sorted(epoch_value_dict.keys())
        sorted_values = [epoch_value_dict[e] for e in sorted_epochs]
        
        return sorted_epochs, sorted_values
    
    return [], []


def find_latest_scalars(work_dir):
    """查找最新的scalars.json文件"""
    work_path = Path(work_dir)
    
    # 先检查主目录下的vis_data
    main_scalars = work_path / "vis_data" / "scalars.json"
    if main_scalars.exists():
        return main_scalars
    
    # 查找所有日期子目录中的scalars.json
    scalars_files = list(work_path.glob("*/vis_data/scalars.json"))
    
    if not scalars_files:
        return None
    
    # 返回文件大小最大的（通常是最完整的）
    return max(scalars_files, key=lambda x: x.stat().st_size)


def load_experiment_data(work_dir, experiment_name):
    """加载单个实验的数据"""
    work_path = Path(work_dir)
    
    # 查找scalars.json文件
    scalars_file = find_latest_scalars(work_path)
    
    if scalars_file is None:
        print(f"⚠️  警告: 未找到 {experiment_name} 的scalars.json文件")
        return None
    
    print(f"✓ 加载 {experiment_name}: {scalars_file}")
    
    # 加载日志数据
    log_data = load_scalars_jsonl(scalars_file)
    
    if not log_data:
        print(f"⚠️  警告: {experiment_name} 的日志数据为空")
        return None
    
    # 提取AP和AR数据（尝试不同的键名）
    ap_epochs, ap_values = extract_epoch_metrics(log_data, 'coco/AP')
    if not ap_epochs:
        ap_epochs, ap_values = extract_epoch_metrics(log_data, 'val/coco/AP')
    
    ar_epochs, ar_values = extract_epoch_metrics(log_data, 'coco/AR')
    if not ar_epochs:
        ar_epochs, ar_values = extract_epoch_metrics(log_data, 'val/coco/AR')
    
    return {
        'name': experiment_name,
        'ap_epochs': ap_epochs,
        'ap_values': ap_values,
        'ar_epochs': ar_epochs,
        'ar_values': ar_values
    }


def plot_curves(experiments_data, output_dir):
    """绘制AP和AR曲线"""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 过滤掉None的数据
    valid_experiments = [e for e in experiments_data if e is not None]
    
    if not valid_experiments:
        print("错误: 没有有效的实验数据")
        return
    
    # 设置颜色和线型
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    linestyles = ['-', '--', '-.', ':', '-', '--']
    markers = ['o', 's', '^', 'D', 'v', 'p']
    
    # 绘制AP曲线
    plt.figure(figsize=(12, 6))
    for i, exp in enumerate(valid_experiments):
        if exp['ap_epochs'] and exp['ap_values']:
            plt.plot(
                exp['ap_epochs'],
                exp['ap_values'],
                label=exp['name'],
                color=colors[i % len(colors)],
                linestyle=linestyles[i % len(linestyles)],
                marker=markers[i % len(markers)],
                markersize=4,
                linewidth=2,
                alpha=0.8
            )
    
    plt.xlabel('训练轮次 (Epoch)', fontsize=12, fontweight='bold')
    plt.ylabel('AP (Average Precision)', fontsize=12, fontweight='bold')
    plt.title('不同策略的AP随训练轮次变化', fontsize=14, fontweight='bold')
    plt.legend(loc='best', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    ap_output = output_path / 'ap_curves.png'
    plt.savefig(ap_output, dpi=300, bbox_inches='tight')
    print(f"✓ AP曲线已保存: {ap_output}")
    plt.close()
    
    # 绘制AR曲线
    plt.figure(figsize=(12, 6))
    for i, exp in enumerate(valid_experiments):
        if exp['ar_epochs'] and exp['ar_values']:
            plt.plot(
                exp['ar_epochs'],
                exp['ar_values'],
                label=exp['name'],
                color=colors[i % len(colors)],
                linestyle=linestyles[i % len(linestyles)],
                marker=markers[i % len(markers)],
                markersize=4,
                linewidth=2,
                alpha=0.8
            )
    
    plt.xlabel('训练轮次 (Epoch)', fontsize=12, fontweight='bold')
    plt.ylabel('AR (Average Recall)', fontsize=12, fontweight='bold')
    plt.title('不同策略的AR随训练轮次变化', fontsize=14, fontweight='bold')
    plt.legend(loc='best', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    ar_output = output_path / 'ar_curves.png'
    plt.savefig(ar_output, dpi=300, bbox_inches='tight')
    print(f"✓ AR曲线已保存: {ar_output}")
    plt.close()
    
    # 打印统计信息
    print(f"\n{'='*60}")
    print("实验数据统计")
    print(f"{'='*60}")
    for exp in valid_experiments:
        print(f"\n{exp['name']}:")
        if exp['ap_epochs'] and exp['ap_values']:
            print(f"  AP: 最大值={max(exp['ap_values']):.4f}, "
                  f"最终值={exp['ap_values'][-1]:.4f}, "
                  f"数据点数={len(exp['ap_values'])}")
        if exp['ar_epochs'] and exp['ar_values']:
            print(f"  AR: 最大值={max(exp['ar_values']):.4f}, "
                  f"最终值={exp['ar_values'][-1]:.4f}, "
                  f"数据点数={len(exp['ar_values'])}")


def main():
    parser = argparse.ArgumentParser(description='绘制消融实验的AP和AR曲线')
    parser.add_argument(
        '--base-dir',
        type=str,
        default='/data/lxc/outputs/train_parallel_model/ablation_experiments',
        help='实验结果基础目录')
    parser.add_argument(
        '--experiments',
        nargs='+',
        default=['baseline', 'loss_weight_only', 'weighted_sampling_only', 'combined'],
        help='要分析的实验目录列表')
    parser.add_argument(
        '--names',
        nargs='+',
        default=['Baseline', 'Loss Weight Only', 'Weighted Sampling Only', 'Combined'],
        help='实验名称列表')
    parser.add_argument(
        '--output-dir',
        type=str,
        default='results/ablation_curves',
        help='输出目录')
    
    args = parser.parse_args()
    
    if len(args.experiments) != len(args.names):
        print("错误: 实验目录和名称数量不匹配")
        return
    
    base_dir = Path(args.base_dir)
    
    print(f"{'='*60}")
    print("加载实验数据")
    print(f"{'='*60}")
    
    experiments_data = []
    for exp_dir, exp_name in zip(args.experiments, args.names):
        work_dir = base_dir / exp_dir
        data = load_experiment_data(work_dir, exp_name)
        experiments_data.append(data)
    
    print(f"\n{'='*60}")
    print("绘制曲线")
    print(f"{'='*60}")
    
    plot_curves(experiments_data, args.output_dir)
    
    print(f"\n{'='*60}")
    print("完成！")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()

