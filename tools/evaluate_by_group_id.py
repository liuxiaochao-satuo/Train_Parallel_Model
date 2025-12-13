#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
按group_id评估模型性能，特别关注group_id=1（运动员）的精度变化

使用方法：
    # 方法1：从work_dir自动查找预测结果
    python tools/evaluate_by_group_id.py \
        --ann-file data/coco_parallel/annotations_id/person_keypoints_val_parallel.json \
        --work-dirs work_dirs/ablation_experiments/loss_weight_only \
                    work_dirs/ablation_experiments/weighted_sampling_only \
                    work_dirs/ablation_experiments/combined \
        --experiment-names "Loss Weight" "Weighted Sampling" "Combined"
    
    # 方法2：直接指定预测结果文件
    python tools/evaluate_by_group_id.py \
        --ann-file data/coco_parallel/annotations_id/person_keypoints_val_parallel.json \
        --pred-files work_dirs/exp1/predictions/results.keypoints.json \
                    work_dirs/exp2/predictions/results.keypoints.json
"""

import json
import argparse
import tempfile
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np
from xtcocotools.coco import COCO
from xtcocotools.cocoeval import COCOeval


def find_prediction_file(work_dir: Path) -> Optional[Path]:
    """从work_dir中查找预测结果文件"""
    # CocoMetric通常将结果保存在以下位置：
    # 1. work_dir/predictions/results.keypoints.json
    # 2. work_dir/checkpoints/epoch_*.keypoints.json
    # 3. work_dir/*.keypoints.json
    
    search_paths = [
        work_dir / 'predictions' / 'results.keypoints.json',
        work_dir / 'predictions' / '*.keypoints.json',
        work_dir / '*.keypoints.json',
    ]
    
    for pattern in search_paths:
        if '*' in str(pattern):
            files = list(work_dir.glob(pattern.name))
            if files:
                # 选择最新的
                return max(files, key=lambda x: x.stat().st_mtime)
        else:
            if pattern.exists():
                return pattern
    
    return None


def create_filtered_coco(ann_file: str, group_id: int, output_file: str = None) -> Tuple[COCO, str]:
    """创建只包含指定group_id的COCO对象
    
    Returns:
        filtered_coco: 过滤后的COCO对象
        temp_file: 临时文件路径
    """
    # 加载原始COCO文件
    coco_gt = COCO(ann_file)
    
    # 获取指定group_id的annotation IDs
    target_ann_ids = []
    for ann_id, ann in coco_gt.anns.items():
        if ann.get('group_id') == group_id:
            target_ann_ids.append(ann_id)
    
    if not target_ann_ids:
        return None, None
    
    # 获取对应的image IDs
    target_img_ids = set()
    for ann_id in target_ann_ids:
        ann = coco_gt.anns[ann_id]
        target_img_ids.add(ann['image_id'])
    
    # 创建过滤后的COCO数据
    filtered_data = {
        'info': coco_gt.dataset.get('info', {}),
        'licenses': coco_gt.dataset.get('licenses', []),
        'categories': coco_gt.dataset['categories'],
        'images': [img for img in coco_gt.dataset['images'] if img['id'] in target_img_ids],
        'annotations': [ann for ann in coco_gt.dataset['annotations'] if ann['id'] in target_ann_ids],
    }
    
    # 保存到临时文件
    if output_file:
        temp_file = output_file
    else:
        temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False).name
    
    with open(temp_file, 'w') as f:
        json.dump(filtered_data, f)
    
    # 创建新的COCO对象
    filtered_coco = COCO(temp_file)
    
    return filtered_coco, temp_file


def filter_predictions_by_images(pred_file: str, target_img_ids: set) -> str:
    """过滤预测结果，只保留目标图像的预测"""
    with open(pred_file, 'r') as f:
        pred_data = json.load(f)
    
    # 过滤预测结果
    filtered_pred = [
        pred for pred in pred_data
        if pred['image_id'] in target_img_ids
    ]
    
    # 保存到临时文件
    temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False).name
    with open(temp_file, 'w') as f:
        json.dump(filtered_pred, f)
    
    return temp_file


def evaluate_group_id(
    ann_file: str,
    pred_file: str,
    group_id: int,
    dataset_meta: dict = None
) -> Dict:
    """评估指定group_id的性能
    
    Returns:
        包含AP, AP50, AP75, AR等指标的字典
    """
    # 创建过滤后的GT COCO对象
    filtered_coco_gt, temp_gt_file = create_filtered_coco(ann_file, group_id)
    
    if filtered_coco_gt is None:
        return None
    
    # 获取目标图像IDs
    target_img_ids = set(filtered_coco_gt.imgs.keys())
    
    # 过滤预测结果
    temp_pred_file = filter_predictions_by_images(pred_file, target_img_ids)
    
    try:
        # 加载预测结果
        filtered_coco_dt = filtered_coco_gt.loadRes(temp_pred_file)
        
        # 获取sigmas（用于OKS计算）
        if dataset_meta and 'sigmas' in dataset_meta:
            sigmas = np.array(dataset_meta['sigmas'])
        else:
            # 默认使用COCO的sigmas（17个关键点）
            sigmas = np.array([
                0.026, 0.025, 0.025, 0.035, 0.035, 0.079, 0.079, 0.072, 0.072,
                0.062, 0.062, 0.107, 0.107, 0.087, 0.087, 0.089, 0.089
            ])
            # 如果有21个关键点，需要扩展
            num_keypoints = len(filtered_coco_gt.loadCats(1)[0]['keypoints'])
            if num_keypoints > 17:
                # 扩展sigmas（新增的4个关键点使用ankle的sigma值）
                sigmas = np.concatenate([
                    sigmas,
                    np.array([0.089, 0.089, 0.089, 0.089])  # heel和foot
                ])
        
        # 进行评估
        coco_eval = COCOeval(filtered_coco_gt, filtered_coco_dt, 'keypoints', sigmas)
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
        
        # 提取结果
        results = {
            'AP': float(coco_eval.stats[0]),
            'AP50': float(coco_eval.stats[1]),
            'AP75': float(coco_eval.stats[2]),
            'APm': float(coco_eval.stats[3]),  # AP medium
            'APl': float(coco_eval.stats[4]),  # AP large
            'AR': float(coco_eval.stats[5]),
            'AR50': float(coco_eval.stats[6]),
            'AR75': float(coco_eval.stats[7]),
            'ARm': float(coco_eval.stats[8]),
            'ARl': float(coco_eval.stats[9]),
        }
        
        return results
    
    finally:
        # 清理临时文件
        import os
        if temp_gt_file and os.path.exists(temp_gt_file):
            os.unlink(temp_gt_file)
        if temp_pred_file and os.path.exists(temp_pred_file):
            os.unlink(temp_pred_file)


def analyze_experiments(
    ann_file: str,
    pred_files: List[str],
    experiment_names: List[str],
    group_ids: List[int] = [1, 2, 3, 4],
    dataset_meta: dict = None
):
    """分析多个实验的结果，按group_id分组"""
    print("=" * 80)
    print("按group_id分析实验结果")
    print("=" * 80)
    print(f"\n标注文件: {ann_file}")
    print(f"实验数量: {len(pred_files)}")
    print(f"分析的group_id: {group_ids}\n")
    
    # 加载标注并统计group_id
    coco_gt = COCO(ann_file)
    group_id_counts = defaultdict(int)
    
    for ann_id, ann in coco_gt.anns.items():
        group_id = ann.get('group_id')
        if group_id is not None:
            group_id_counts[group_id] += 1
    
    print("Group ID统计:")
    for gid in sorted(group_id_counts.keys()):
        print(f"  group_id={gid}: {group_id_counts[gid]} 个标注")
    print()
    
    # 对每个实验进行评估
    all_results = {}
    
    for exp_name, pred_file in zip(experiment_names, pred_files):
        if not Path(pred_file).exists():
            print(f"⚠️  预测文件不存在: {pred_file}")
            continue
        
        print(f"\n{'='*80}")
        print(f"分析实验: {exp_name}")
        print(f"预测文件: {pred_file}")
        print(f"{'='*80}")
        
        exp_results = {}
        
        for group_id in group_ids:
            if group_id not in group_id_counts:
                print(f"\n跳过 group_id={group_id}（无数据）")
                continue
            
            print(f"\n评估 group_id={group_id} ({group_id_counts[group_id]} 个标注)...")
            try:
                results = evaluate_group_id(ann_file, pred_file, group_id, dataset_meta)
                if results:
                    exp_results[group_id] = results
                    print(f"  AP:   {results['AP']:.4f}")
                    print(f"  AP50: {results['AP50']:.4f}")
                    print(f"  AP75: {results['AP75']:.4f}")
                    print(f"  AR:   {results['AR']:.4f}")
            except Exception as e:
                print(f"  ❌ 评估失败: {e}")
                import traceback
                traceback.print_exc()
        
        all_results[exp_name] = exp_results
    
    # 生成对比报告
    print(f"\n{'='*80}")
    print("实验结果对比（按group_id）")
    print(f"{'='*80}\n")
    
    # 重点对比group_id=1（运动员）
    if 1 in group_ids:
        print("=" * 80)
        print("🎯 重点分析：group_id=1（运动员）")
        print("=" * 80)
        print(f"\n{'实验名称':<30s} {'AP':>10s} {'AP50':>10s} {'AP75':>10s} {'AR':>10s}")
        print("-" * 80)
        
        baseline_ap = None
        for exp_name in experiment_names:
            if exp_name in all_results and 1 in all_results[exp_name]:
                r = all_results[exp_name][1]
                print(f"{exp_name:<30s} {r['AP']:>10.4f} {r['AP50']:>10.4f} "
                      f"{r['AP75']:>10.4f} {r['AR']:>10.4f}")
                if baseline_ap is None:
                    baseline_ap = r['AP']
        
        # 计算改进幅度
        if baseline_ap is not None and len(experiment_names) > 1:
            print(f"\n相对于第一个实验的改进（group_id=1，运动员）:")
            for exp_name in experiment_names[1:]:
                if exp_name in all_results and 1 in all_results[exp_name]:
                    ap = all_results[exp_name][1]['AP']
                    improvement = ap - baseline_ap
                    improvement_pct = (improvement / baseline_ap) * 100 if baseline_ap > 0 else 0
                    arrow = "↑" if improvement > 0 else "↓"
                    print(f"  {exp_name:30s} AP: {ap:.4f} "
                          f"({arrow}{abs(improvement):.4f}, {arrow}{abs(improvement_pct):.2f}%)")
    
    # 所有group_id的详细对比
    print(f"\n{'='*80}")
    print("所有group_id的详细对比")
    print(f"{'='*80}\n")
    
    for group_id in sorted(group_ids):
        if group_id not in group_id_counts:
            continue
        
        print(f"\ngroup_id={group_id} ({group_id_counts[group_id]} 个标注):")
        print(f"{'实验名称':<30s} {'AP':>10s} {'AP50':>10s} {'AP75':>10s} {'AR':>10s}")
        print("-" * 80)
        
        for exp_name in experiment_names:
            if exp_name in all_results and group_id in all_results[exp_name]:
                r = all_results[exp_name][group_id]
                print(f"{exp_name:<30s} {r['AP']:>10.4f} {r['AP50']:>10.4f} "
                      f"{r['AP75']:>10.4f} {r['AR']:>10.4f}")
    
    return all_results


def main():
    parser = argparse.ArgumentParser(
        description='按group_id评估模型性能，特别关注group_id=1（运动员）')
    parser.add_argument(
        '--ann-file',
        type=str,
        required=True,
        help='COCO格式的验证集标注文件（包含group_id）')
    parser.add_argument(
        '--pred-files',
        nargs='+',
        type=str,
        default=None,
        help='预测结果JSON文件列表（COCO格式，.keypoints.json）')
    parser.add_argument(
        '--work-dirs',
        nargs='+',
        type=str,
        default=None,
        help='实验工作目录列表（会自动查找预测文件）')
    parser.add_argument(
        '--experiment-names',
        nargs='+',
        type=str,
        default=None,
        help='实验名称列表')
    parser.add_argument(
        '--group-ids',
        nargs='+',
        type=int,
        default=[1, 2, 3, 4],
        help='要分析的group_id列表（默认：[1, 2, 3, 4]）')
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='保存结果到JSON文件（可选）')
    
    args = parser.parse_args()
    
    # 确定预测文件
    if args.pred_files:
        pred_files = args.pred_files
        if args.experiment_names:
            exp_names = args.experiment_names
        else:
            exp_names = [Path(f).stem for f in pred_files]
    elif args.work_dirs:
        results = []
        for work_dir in args.work_dirs:
            work_path = Path(work_dir)
            exp_name = work_path.name
            pred_file = find_prediction_file(work_path)
            if pred_file:
                results.append((exp_name, str(pred_file)))
            else:
                print(f"⚠️  未找到 {exp_name} 的预测文件，请先运行验证或测试")
        
        if not results:
            print("错误：未找到任何预测文件")
            print("\n提示：预测文件通常在以下位置：")
            print("  - work_dir/predictions/results.keypoints.json")
            print("  - work_dir/*.keypoints.json")
            print("\n请先运行验证或测试生成预测结果")
            return
        
        exp_names, pred_files = zip(*results)
        exp_names = list(exp_names)
        pred_files = list(pred_files)
    else:
        print("错误：必须提供 --pred-files 或 --work-dirs")
        return
    
    if args.experiment_names:
        if len(args.experiment_names) == len(pred_files):
            exp_names = args.experiment_names
        else:
            print("警告：实验名称数量与预测文件数量不匹配，使用默认名称")
    
    # 加载数据集元信息（用于sigmas）
    try:
        from mmpose.configs._base_.datasets.coco_parallel import dataset_info
        dataset_meta = dataset_info
    except:
        dataset_meta = None
    
    # 执行分析
    results = analyze_experiments(
        ann_file=args.ann_file,
        pred_files=pred_files,
        experiment_names=exp_names,
        group_ids=args.group_ids,
        dataset_meta=dataset_meta
    )
    
    # 保存结果
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n结果已保存到: {output_path}")


if __name__ == '__main__':
    main()

