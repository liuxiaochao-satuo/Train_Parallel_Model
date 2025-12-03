#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
模型评估与对比脚本

用于评估单个模型或对比训练前后的模型性能
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("错误: PyTorch未安装")
    sys.exit(1)


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='评估和对比模型性能')
    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='配置文件路径'
    )
    parser.add_argument(
        '--checkpoint',
        type=str,
        default=None,
        help='模型checkpoint路径（单个模型评估）'
    )
    parser.add_argument(
        '--checkpoint-pretrained',
        type=str,
        default=None,
        help='预训练模型checkpoint路径（对比模式）'
    )
    parser.add_argument(
        '--checkpoint-trained',
        type=str,
        default=None,
        help='训练后模型checkpoint路径（对比模式）'
    )
    parser.add_argument(
        '--ann-file',
        type=str,
        required=True,
        help='验证集标注文件路径'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='evaluation_results.json',
        help='输出结果文件路径'
    )
    parser.add_argument(
        '--compare',
        action='store_true',
        help='启用对比模式（需要提供pretrained和trained checkpoint）'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda:0',
        help='使用的设备（cuda:0 或 cpu）'
    )
    return parser.parse_args()


def evaluate_model(config_path: str, checkpoint_path: str, ann_file: str, device: str = 'cuda:0') -> Dict:
    """
    评估单个模型
    
    返回评估指标字典
    """
    print(f"正在评估模型: {checkpoint_path}")
    
    try:
        from mmengine import Config
        from mmengine.runner import Runner
        from mmengine.registry import init_default_scope
        
        # 初始化
        init_default_scope('mmpose')
        
        # 加载配置
        cfg = Config.fromfile(config_path)
        
        # 修改验证集标注文件路径
        if hasattr(cfg, 'val_evaluator'):
            cfg.val_evaluator.ann_file = ann_file
        
        # 创建runner
        runner = Runner.from_cfg(cfg)
        
        # 加载checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # 加载模型权重
        runner.model.load_state_dict(checkpoint.get('state_dict', checkpoint))
        runner.model.to(device)
        runner.model.eval()
        
        # 运行验证
        metrics = runner.val()
        
        return metrics
        
    except Exception as e:
        print(f"评估失败: {e}")
        import traceback
        traceback.print_exc()
        return {}


def evaluate_with_mmpose_tools(config_path: str, checkpoint_path: str, ann_file: str) -> Dict:
    """
    使用MMPose的test.py工具评估模型
    
    这是更可靠的方法
    """
    print(f"使用MMPose工具评估模型: {checkpoint_path}")
    
    import subprocess
    import tempfile
    
    # 构建测试命令
    cmd = [
        'python', 'tools/test.py',
        config_path,
        checkpoint_path,
        '--ann-file', ann_file
    ]
    
    # 检查是否在mmpose目录中
    if not os.path.exists('tools/test.py'):
        print("⚠️  警告: 未找到 tools/test.py")
        print("   请确保在MMPose根目录下运行此脚本")
        return {}
    
    # 执行测试
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=False
        )
        
        if result.returncode != 0:
            print(f"测试失败: {result.stderr}")
            return {}
        
        # 解析输出（MMPose会输出JSON格式的结果）
        # 这里需要根据实际输出格式解析
        output = result.stdout
        
        # 尝试从输出中提取指标
        metrics = {}
        
        # 查找AP指标
        import re
        ap_pattern = r'AP:\s*([\d.]+)'
        ap_match = re.search(ap_pattern, output)
        if ap_match:
            metrics['AP'] = float(ap_match.group(1))
        
        # 查找AR指标
        ar_pattern = r'AR:\s*([\d.]+)'
        ar_match = re.search(ar_pattern, output)
        if ar_match:
            metrics['AR'] = float(ar_match.group(1))
        
        return metrics
        
    except Exception as e:
        print(f"测试过程出错: {e}")
        return {}


def load_metrics_from_file(file_path: str) -> Optional[Dict]:
    """从文件加载之前保存的指标"""
    if os.path.exists(file_path):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"加载指标文件失败: {e}")
    return None


def save_metrics(metrics: Dict, file_path: str):
    """保存指标到文件"""
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    print(f"指标已保存到: {file_path}")


def compare_models(metrics_pretrained: Dict, metrics_trained: Dict) -> Dict:
    """对比两个模型的指标"""
    comparison = {
        'pretrained': metrics_pretrained,
        'trained': metrics_trained,
        'improvement': {}
    }
    
    # 计算改进百分比
    for key in metrics_pretrained:
        if key in metrics_trained and isinstance(metrics_pretrained[key], (int, float)):
            pretrained_val = metrics_pretrained[key]
            trained_val = metrics_trained[key]
            
            if pretrained_val != 0:
                improvement = ((trained_val - pretrained_val) / pretrained_val) * 100
                comparison['improvement'][key] = {
                    'absolute': trained_val - pretrained_val,
                    'percentage': improvement
                }
            else:
                comparison['improvement'][key] = {
                    'absolute': trained_val - pretrained_val,
                    'percentage': float('inf') if trained_val > 0 else 0
                }
    
    return comparison


def print_metrics(metrics: Dict, title: str = "评估结果"):
    """打印指标"""
    print(f"\n{'='*60}")
    print(f"{title}")
    print(f"{'='*60}")
    
    if not metrics:
        print("无评估结果")
        return
    
    # 打印主要指标
    main_metrics = ['AP', 'AR', 'AP@0.5', 'AP@0.75', 'AP (medium)', 'AP (large)']
    for metric in main_metrics:
        if metric in metrics:
            print(f"{metric}: {metrics[metric]:.4f}")
    
    # 打印其他指标
    other_metrics = {k: v for k, v in metrics.items() if k not in main_metrics}
    if other_metrics:
        print("\n其他指标:")
        for key, value in other_metrics.items():
            if isinstance(value, (int, float)):
                print(f"  {key}: {value:.4f}")
            else:
                print(f"  {key}: {value}")
    
    print()


def print_comparison(comparison: Dict):
    """打印对比结果"""
    print(f"\n{'='*60}")
    print("模型性能对比")
    print(f"{'='*60}\n")
    
    print("预训练模型指标:")
    print_metrics(comparison.get('pretrained', {}), "")
    
    print("训练后模型指标:")
    print_metrics(comparison.get('trained', {}), "")
    
    print("性能改进:")
    improvement = comparison.get('improvement', {})
    if improvement:
        for key, value in improvement.items():
            abs_change = value.get('absolute', 0)
            pct_change = value.get('percentage', 0)
            direction = "↑" if abs_change > 0 else "↓" if abs_change < 0 else "="
            print(f"  {key}: {abs_change:+.4f} ({pct_change:+.2f}%) {direction}")
    else:
        print("  无改进数据")
    
    print()


def main():
    """主函数"""
    args = parse_args()
    
    # 检查参数
    if args.compare:
        if not args.checkpoint_pretrained or not args.checkpoint_trained:
            print("错误: 对比模式需要提供 --checkpoint-pretrained 和 --checkpoint-trained")
            sys.exit(1)
    else:
        if not args.checkpoint:
            print("错误: 单模型评估需要提供 --checkpoint")
            sys.exit(1)
    
    # 检查文件是否存在
    if not os.path.exists(args.config):
        print(f"错误: 配置文件不存在: {args.config}")
        sys.exit(1)
    
    if not os.path.exists(args.ann_file):
        print(f"错误: 标注文件不存在: {args.ann_file}")
        sys.exit(1)
    
    if args.compare:
        # 对比模式
        if not os.path.exists(args.checkpoint_pretrained):
            print(f"错误: 预训练模型不存在: {args.checkpoint_pretrained}")
            sys.exit(1)
        if not os.path.exists(args.checkpoint_trained):
            print(f"错误: 训练后模型不存在: {args.checkpoint_trained}")
            sys.exit(1)
        
        print("=" * 60)
        print("模型性能对比")
        print("=" * 60)
        
        # 评估预训练模型
        print("\n1. 评估预训练模型...")
        metrics_pretrained = evaluate_with_mmpose_tools(
            args.config,
            args.checkpoint_pretrained,
            args.ann_file
        )
        
        # 评估训练后模型
        print("\n2. 评估训练后模型...")
        metrics_trained = evaluate_with_mmpose_tools(
            args.config,
            args.checkpoint_trained,
            args.ann_file
        )
        
        # 对比结果
        if metrics_pretrained and metrics_trained:
            comparison = compare_models(metrics_pretrained, metrics_trained)
            print_comparison(comparison)
            
            # 保存结果
            save_metrics(comparison, args.output)
        else:
            print("评估失败，无法进行对比")
            sys.exit(1)
    
    else:
        # 单模型评估模式
        if not os.path.exists(args.checkpoint):
            print(f"错误: 模型checkpoint不存在: {args.checkpoint}")
            sys.exit(1)
        
        print("=" * 60)
        print("模型评估")
        print("=" * 60)
        
        # 评估模型
        metrics = evaluate_with_mmpose_tools(
            args.config,
            args.checkpoint,
            args.ann_file
        )
        
        if metrics:
            print_metrics(metrics, "评估结果")
            save_metrics(metrics, args.output)
        else:
            print("评估失败")
            sys.exit(1)


if __name__ == "__main__":
    main()

