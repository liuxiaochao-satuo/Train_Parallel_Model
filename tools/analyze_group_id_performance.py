#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
按group_id分析实验结果，特别关注group_id=1（运动员）的性能变化

使用方法：
    python tools/analyze_group_id_performance.py \
        --ann-file data/coco_parallel/annotations_id/person_keypoints_val_parallel.json \
        --pred-files work_dirs/ablation_experiments/*/predictions/*.json \
        --output-dir results/group_id_analysis
"""

import json
import argparse
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np
from xtcocotools.coco import COCO
from xtcocotools.cocoeval import COCOeval


def load_coco_annotations(ann_file: str) -> Tuple[COCO, Dict[int, int]]:
    """加载COCO标注文件并创建group_id映射
    
    Returns:
        coco: COCO对象
        ann_id_to_group_id: annotation_id到group_id的映射
    """
    coco = COCO(ann_file)
    
    # 创建annotation_id到group_id的映射
    ann_id_to_group_id = {}
    for ann_id, ann in coco.anns.items():
        group_id = ann.get('group_id')
        if group_id is not None:
            ann_id_to_group_id[ann_id] = group_id
    
    return coco, ann_id_to_group_id


def filter_by_group_id(coco: COCO, group_id: int, ann_id_to_group_id: Dict[int, int]) -> COCO:
    """根据group_id过滤COCO对象
    
    Args:
        coco: 原始COCO对象
        group_id: 要保留的group_id
        ann_id_to_group_id: annotation_id到group_id的映射
    
    Returns:
        过滤后的COCO对象（新创建的）
    """
    # 获取所有annotation IDs
    all_ann_ids = coco.getAnnIds()
    
    # 过滤出指定group_id的annotation IDs
    filtered_ann_ids = [
        ann_id for ann_id in all_ann_ids
        if ann_id_to_group_id.get(ann_id) == group_id
    ]
    
    # 获取对应的image IDs
    filtered_img_ids = set()
    for ann_id in filtered_ann_ids:
        ann = coco.anns[ann_id]
        filtered_img_ids.add(ann['image_id'])
    
    # 创建新的COCO对象（只包含过滤后的数据）
    # 注意：这里我们创建一个临时的COCO对象用于评估
    # 实际评估时，我们需要修改预测结果来匹配
    return filtered_ann_ids, list(filtered_img_ids)


def evaluate_by_group_id(
    ann_file: str,
    pred_file: str,
    group_id: int,
    output_dir: Path = None
) -> Dict:
    """按group_id评估性能
    
    Args:
        ann_file: COCO格式的标注文件
        pred_file: COCO格式的预测结果文件
        group_id: 要评估的group_id
        output_dir: 输出目录（可选）
    
    Returns:
        评估结果字典
    """
    # 加载标注
    coco_gt = COCO(ann_file)
    
    # 创建ann_id到group_id的映射
    ann_id_to_group_id = {}
    for ann_id, ann in coco_gt.anns.items():
        group_id_val = ann.get('group_id')
        if group_id_val is not None:
            ann_id_to_group_id[ann_id] = group_id_val
    
    # 获取指定group_id的annotation IDs
    target_ann_ids = [
        ann_id for ann_id in coco_gt.getAnnIds()
        if ann_id_to_group_id.get(ann_id) == group_id
    ]
    
    if not target_ann_ids:
        print(f"警告：未找到group_id={group_id}的标注")
        return None
    
    # 获取对应的image IDs
    target_img_ids = set()
    for ann_id in target_ann_ids:
        ann = coco_gt.anns[ann_id]
        target_img_ids.add(ann['image_id'])
    
    # 加载预测结果
    with open(pred_file, 'r') as f:
        pred_data = json.load(f)
    
    # 过滤预测结果：只保留对应group_id的GT所在的图像
    # 注意：对于bottom-up，预测结果按图像组织，我们需要保留所有预测
    # 但评估时只评估与target_ann_ids匹配的部分
    
    # 创建过滤后的GT COCO对象
    filtered_gt = {
        'images': [img for img in coco_gt.dataset['images'] if img['id'] in target_img_ids],
        'annotations': [ann for ann in coco_gt.dataset['annotations'] if ann['id'] in target_ann_ids],
        'categories': coco_gt.dataset['categories'],
    }
    
    # 保存临时GT文件
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        temp_gt_file = output_dir / f'temp_gt_group_{group_id}.json'
        with open(temp_gt_file, 'w') as f:
            json.dump(filtered_gt, f)
        
        # 创建临时COCO对象
        temp_coco_gt = COCO(str(temp_gt_file))
    else:
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(filtered_gt, f)
            temp_gt_file = f.name
        temp_coco_gt = COCO(temp_gt_file)
    
    # 过滤预测结果：只保留目标图像的预测
    filtered_pred = [
        pred for pred in pred_data
        if pred['image_id'] in target_img_ids
    ]
    
    # 保存临时预测文件
    if output_dir:
        temp_pred_file = output_dir / f'temp_pred_group_{group_id}.json'
    else:
        import tempfile
        temp_pred_file = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False).name
    
    with open(temp_pred_file, 'w') as f:
        json.dump(filtered_pred, f)
    
    # 加载预测结果到COCO对象
    coco_dt = temp_coco_gt.loadRes(str(temp_pred_file))
    
    # 进行评估
    coco_eval = COCOeval(temp_coco_gt, coco_dt, 'keypoints')
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    
    # 提取结果
    results = {
        'AP': coco_eval.stats[0],
        'AP50': coco_eval.stats[1],
        'AP75': coco_eval.stats[2],
        'APm': coco_eval.stats[3],  # AP medium
        'APl': coco_eval.stats[4],  # AP large
        'AR': coco_eval.stats[5],
        'AR50': coco_eval.stats[6],
        'AR75': coco_eval.stats[7],
        'ARm': coco_eval.stats[8],
        'ARl': coco_eval.stats[9],
    }
    
    # 清理临时文件
    if not output_dir:
        import os
        os.unlink(temp_gt_file)
        os.unlink(temp_pred_file)
    
    return results


def analyze_multiple_experiments(
    ann_file: str,
    pred_files: List[str],
    experiment_names: List[str],
    group_ids: List[int] = [1, 2, 3, 4],
    output_dir: Path = None
):
    """分析多个实验的结果，按group_id分组
    
    Args:
        ann_file: COCO格式的标注文件
        pred_files: 预测结果文件列表
        experiment_names: 实验名称列表
        group_ids: 要分析的group_id列表
        output_dir: 输出目录
    """
    print("=" * 80)
    print("按group_id分析实验结果")
    print("=" * 80)
    print(f"\n标注文件: {ann_file}")
    print(f"实验数量: {len(pred_files)}")
    print(f"分析的group_id: {group_ids}\n")
    
    # 加载标注并获取group_id统计
    coco_gt = COCO(ann_file)
    ann_id_to_group_id = {}
    group_id_counts = defaultdict(int)
    
    for ann_id, ann in coco_gt.anns.items():
        group_id = ann.get('group_id')
        if group_id is not None:
            ann_id_to_group_id[ann_id] = group_id
            group_id_counts[group_id] += 1
    
    print("Group ID统计:")
    for gid in sorted(group_id_counts.keys()):
        print(f"  group_id={gid}: {group_id_counts[gid]} 个标注")
    print()
    
    # 对每个实验和每个group_id进行评估
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
                results = evaluate_by_group_id(
                    ann_file, pred_file, group_id, output_dir
                )
                if results:
                    exp_results[group_id] = results
                    print(f"  AP: {results['AP']:.4f}")
                    print(f"  AP50: {results['AP50']:.4f}")
                    print(f"  AP75: {results['AP75']:.4f}")
                    print(f"  AR: {results['AR']:.4f}")
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
        print("重点分析：group_id=1（运动员）")
        print("=" * 80)
        print(f"\n{'实验名称':<30s} {'AP':>10s} {'AP50':>10s} {'AP75':>10s} {'AR':>10s}")
        print("-" * 80)
        
        for exp_name in experiment_names:
            if exp_name in all_results and 1 in all_results[exp_name]:
                r = all_results[exp_name][1]
                print(f"{exp_name:<30s} {r['AP']:>10.4f} {r['AP50']:>10.4f} "
                      f"{r['AP75']:>10.4f} {r['AR']:>10.4f}")
        
        # 计算改进幅度
        if len(experiment_names) >= 2:
            baseline_name = experiment_names[0]
            if baseline_name in all_results and 1 in all_results[baseline_name]:
                baseline_ap = all_results[baseline_name][1]['AP']
                print(f"\n相对于 {baseline_name} 的改进（group_id=1）:")
                for exp_name in experiment_names[1:]:
                    if exp_name in all_results and 1 in all_results[exp_name]:
                        ap = all_results[exp_name][1]['AP']
                        improvement = ap - baseline_ap
                        improvement_pct = (improvement / baseline_ap) * 100 if baseline_ap > 0 else 0
                        print(f"  {exp_name:30s} AP: {ap:.4f} "
                              f"(+{improvement:.4f}, +{improvement_pct:.2f}%)")
    
    # 所有group_id的对比
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
    
    # 保存结果到JSON
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        results_file = output_dir / 'group_id_analysis_results.json'
        with open(results_file, 'w') as f:
            json.dump(all_results, f, indent=2)
        print(f"\n结果已保存到: {results_file}")


def find_prediction_files(work_dirs: List[str]) -> List[Tuple[str, str]]:
    """从work_dirs中查找预测结果文件
    
    Returns:
        List of (experiment_name, pred_file_path) tuples
    """
    results = []
    
    for work_dir in work_dirs:
        work_path = Path(work_dir)
        exp_name = work_path.name
        
        # 查找预测文件（通常在predictions目录或checkpoints目录）
        pred_dirs = [
            work_path / 'predictions',
            work_path / 'checkpoints',
            work_path,
        ]
        
        pred_file = None
        for pred_dir in pred_dirs:
            if pred_dir.exists():
                # 查找最新的JSON文件（通常是预测结果）
                json_files = list(pred_dir.glob('*.json'))
                # 排除日志文件
                json_files = [f for f in json_files 
                             if 'scalars' not in f.name and 'vis_data' not in str(f)]
                if json_files:
                    # 选择最新的
                    pred_file = max(json_files, key=lambda x: x.stat().st_mtime)
                    break
        
        if pred_file and pred_file.exists():
            results.append((exp_name, str(pred_file)))
        else:
            print(f"⚠️  未找到 {exp_name} 的预测文件")
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description='按group_id分析实验结果，特别关注group_id=1（运动员）的性能')
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
        help='预测结果JSON文件列表（COCO格式）')
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
        help='实验名称列表（与pred-files或work-dirs对应）')
    parser.add_argument(
        '--group-ids',
        nargs='+',
        type=int,
        default=[1, 2, 3, 4],
        help='要分析的group_id列表（默认：[1, 2, 3, 4]）')
    parser.add_argument(
        '--output-dir',
        type=str,
        default='results/group_id_analysis',
        help='输出目录（默认：results/group_id_analysis）')
    
    args = parser.parse_args()
    
    # 确定预测文件
    if args.pred_files:
        pred_files = args.pred_files
        if args.experiment_names:
            exp_names = args.experiment_names
        else:
            exp_names = [Path(f).stem for f in pred_files]
    elif args.work_dirs:
        results = find_prediction_files(args.work_dirs)
        if not results:
            print("错误：未找到任何预测文件")
            return
        exp_names, pred_files = zip(*results)
        exp_names = list(exp_names)
        pred_files = list(pred_files)
    else:
        print("错误：必须提供 --pred-files 或 --work-dirs")
        return
    
    if len(exp_names) != len(pred_files):
        print("错误：实验名称和预测文件数量不匹配")
        return
    
    # 执行分析
    analyze_multiple_experiments(
        ann_file=args.ann_file,
        pred_files=pred_files,
        experiment_names=exp_names,
        group_ids=args.group_ids,
        output_dir=Path(args.output_dir) if args.output_dir else None
    )


if __name__ == '__main__':
    main()

