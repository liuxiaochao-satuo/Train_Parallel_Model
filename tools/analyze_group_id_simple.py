#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
简化版group_id分析脚本 - 直接使用预测结果和标注文件
"""

import json
import argparse
from pathlib import Path
from collections import defaultdict
import numpy as np

try:
    from xtcocotools.coco import COCO
    from xtcocotools.cocoeval import COCOeval
except ImportError:
    print("错误: 需要安装xtcocotools")
    print("请运行: pip install xtcocotools")
    exit(1)


def evaluate_by_group_id(ann_file, pred_file, group_id):
    """评估指定group_id的性能"""
    # 加载标注文件
    coco_gt = COCO(ann_file)
    
    # 加载预测结果
    with open(pred_file, 'r') as f:
        pred_data = json.load(f)
    
    # 创建临时COCO对象用于评估
    # 过滤出指定group_id的标注
    ann_ids = []
    for ann_id, ann in coco_gt.anns.items():
        if ann.get('group_id') == group_id:
            ann_ids.append(ann_id)
    
    if not ann_ids:
        print(f"警告: 未找到group_id={group_id}的标注")
        return None
    
    # 创建临时标注文件（仅包含指定group_id）
    # categories需要是列表格式
    temp_gt = {
        'images': [],
        'annotations': [],
        'categories': list(coco_gt.cats.values()) if isinstance(coco_gt.cats, dict) else coco_gt.cats
    }
    
    # 获取相关图像
    image_ids = set()
    for ann_id in ann_ids:
        ann = coco_gt.anns[ann_id]
        image_ids.add(ann['image_id'])
    
    # 添加图像信息
    for img_id in image_ids:
        if img_id in coco_gt.imgs:
            temp_gt['images'].append(coco_gt.imgs[img_id])
    
    # 添加标注信息
    for ann_id in ann_ids:
        temp_gt['annotations'].append(coco_gt.anns[ann_id])
    
    # 创建临时文件
    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(temp_gt, f)
        temp_gt_file = f.name
    
    # 过滤预测结果（仅包含相关图像）
    temp_pred = []
    for pred in pred_data:
        # 确保pred是字典格式
        if isinstance(pred, dict) and pred.get('image_id') in image_ids:
            temp_pred.append(pred)
    
    if not temp_pred:
        print(f"警告: 未找到group_id={group_id}相关的预测结果")
        return None
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(temp_pred, f)
        temp_pred_file = f.name
    
    try:
        # 加载临时标注和预测
        coco_gt_temp = COCO(temp_gt_file)
        # 确保预测文件格式正确
        with open(temp_pred_file, 'r') as f:
            pred_check = json.load(f)
            if not isinstance(pred_check, list):
                print(f"错误: 预测文件格式不正确，应为列表")
                return None
            # 检查第一个预测结果的格式
            if len(pred_check) > 0:
                first_pred = pred_check[0]
                required_keys = ['image_id', 'category_id', 'keypoints']
                missing_keys = [k for k in required_keys if k not in first_pred]
                if missing_keys:
                    print(f"错误: 预测结果缺少必需字段: {missing_keys}")
                    return None
        
        coco_dt = coco_gt_temp.loadRes(temp_pred_file)
        
        # 获取关键点数量并设置sigmas
        num_keypoints = len(coco_gt_temp.loadCats(1)[0]['keypoints'])
        
        # 默认COCO sigmas (17个关键点)
        default_sigmas = np.array([
            0.026, 0.025, 0.025, 0.035, 0.035, 0.079, 0.079, 0.072, 0.072,
            0.062, 0.062, 0.107, 0.107, 0.087, 0.087, 0.089, 0.089
        ])
        
        if num_keypoints == 17:
            sigmas = default_sigmas
        elif num_keypoints == 21:
            # 扩展sigmas（新增的4个关键点使用ankle的sigma值）
            sigmas = np.concatenate([
                default_sigmas,
                np.array([0.089, 0.089, 0.089, 0.089])  # heel和foot
            ])
        else:
            # 使用默认值，让COCOeval自动处理
            sigmas = None
        
        # 评估
        if sigmas is not None:
            coco_eval = COCOeval(coco_gt_temp, coco_dt, 'keypoints', sigmas)
        else:
            coco_eval = COCOeval(coco_gt_temp, coco_dt, 'keypoints')
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
        
        # 提取结果
        results = {
            'AP': float(coco_eval.stats[0]),
            'AP50': float(coco_eval.stats[1]),
            'AP75': float(coco_eval.stats[2]),
            'AR': float(coco_eval.stats[5]),
        }
        
        return results
    finally:
        # 清理临时文件
        import os
        if os.path.exists(temp_gt_file):
            os.unlink(temp_gt_file)
        if os.path.exists(temp_pred_file):
            os.unlink(temp_pred_file)


def main():
    parser = argparse.ArgumentParser(description='按group_id分析实验结果')
    parser.add_argument('--ann-file', required=True, help='标注文件路径')
    parser.add_argument('--pred-files', nargs='+', required=True, help='预测结果文件列表')
    parser.add_argument('--experiment-names', nargs='+', help='实验名称列表')
    parser.add_argument('--group-ids', nargs='+', type=int, default=[1], help='要分析的group_id列表')
    
    args = parser.parse_args()
    
    if args.experiment_names and len(args.experiment_names) != len(args.pred_files):
        print("错误: 实验名称数量必须与预测文件数量相同")
        return
    
    exp_names = args.experiment_names or [Path(f).parent.parent.name for f in args.pred_files]
    
    print("=" * 80)
    print("按group_id分析实验结果")
    print("=" * 80)
    print(f"\n标注文件: {args.ann_file}")
    print(f"实验数量: {len(args.pred_files)}")
    print(f"分析的group_id: {args.group_ids}\n")
    
    # 分析每个实验
    all_results = {}
    for exp_name, pred_file in zip(exp_names, args.pred_files):
        print(f"\n分析实验: {exp_name}")
        print(f"预测文件: {pred_file}")
        
        if not Path(pred_file).exists():
            print(f"⚠️  文件不存在，跳过")
            continue
        
        file_size = Path(pred_file).stat().st_size
        if file_size < 100:
            print(f"⚠️  文件太小({file_size} bytes)，可能为空，跳过")
            continue
        
        all_results[exp_name] = {}
        for group_id in args.group_ids:
            print(f"  分析 group_id={group_id}...", end=' ')
            try:
                results = evaluate_by_group_id(args.ann_file, pred_file, group_id)
                if results:
                    all_results[exp_name][group_id] = results
                    print(f"✓ AP={results['AP']:.4f}, AR={results['AR']:.4f}")
                else:
                    print("✗ 无结果")
            except Exception as e:
                import traceback
                print(f"✗ 错误: {e}")
                print(f"详细错误: {traceback.format_exc()}")
    
    # 打印对比结果
    print("\n" + "=" * 80)
    print("实验结果对比")
    print("=" * 80)
    
    for group_id in args.group_ids:
        print(f"\n【group_id={group_id}】")
        print(f"{'实验名称':<30s} {'AP':>10s} {'AP50':>10s} {'AP75':>10s} {'AR':>10s}")
        print("-" * 80)
        
        baseline_ap = None
        for exp_name in exp_names:
            if exp_name in all_results and group_id in all_results[exp_name]:
                r = all_results[exp_name][group_id]
                print(f"{exp_name:<30s} {r['AP']:>10.4f} {r['AP50']:>10.4f} "
                      f"{r['AP75']:>10.4f} {r['AR']:>10.4f}")
                if baseline_ap is None:
                    baseline_ap = r['AP']
        
        # 计算改进幅度
        if baseline_ap is not None and len(exp_names) > 1:
            print(f"\n相对于第一个实验的改进（group_id={group_id}）:")
            first_exp = exp_names[0]
            for exp_name in exp_names[1:]:
                if exp_name in all_results and group_id in all_results[exp_name]:
                    ap = all_results[exp_name][group_id]['AP']
                    improvement = ap - baseline_ap
                    improvement_pct = (improvement / baseline_ap) * 100 if baseline_ap > 0 else 0
                    arrow = "↑" if improvement > 0 else "↓"
                    print(f"  {exp_name:30s} AP: {ap:.4f} "
                          f"({arrow}{abs(improvement):.4f}, {arrow}{abs(improvement_pct):.2f}%)")


if __name__ == '__main__':
    main()

