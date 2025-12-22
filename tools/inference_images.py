#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
使用训练好的模型对图片进行推理
"""

import argparse
import cv2
import numpy as np
from pathlib import Path
import torch

# 修复torch.load的weights_only问题
_original_torch_load = torch.load
def patched_torch_load(*args, **kwargs):
    if 'weights_only' not in kwargs:
        kwargs['weights_only'] = False
    return _original_torch_load(*args, **kwargs)
torch.load = patched_torch_load

from mmengine.config import Config
from mmengine.runner import Runner
from mmpose.apis import inference_bottomup, init_model
from mmpose.visualization import PoseLocalVisualizer


def parse_args():
    parser = argparse.ArgumentParser(description='对图片进行姿态估计推理')
    parser.add_argument('config', help='配置文件路径')
    parser.add_argument('checkpoint', help='模型checkpoint路径')
    parser.add_argument('images', nargs='+', help='要推理的图片路径（可以是多个）')
    parser.add_argument('--output-dir', type=str, default='inference_results', 
                        help='输出目录（默认：inference_results）')
    parser.add_argument('--show', action='store_true', help='显示结果图片')
    parser.add_argument('--device', type=str, default='cpu', 
                        help='推理设备（默认：cpu，可选：cuda）')
    return parser.parse_args()


def main():
    args = parse_args()
    
    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"加载模型...")
    print(f"  配置文件: {args.config}")
    print(f"  Checkpoint: {args.checkpoint}")
    print(f"  设备: {args.device}")
    
    # 初始化模型
    model = init_model(args.config, args.checkpoint, device=args.device)
    
    # 创建可视化器
    visualizer = PoseLocalVisualizer()
    visualizer.set_dataset_meta(model.dataset_meta)
    
    print(f"\n开始推理 {len(args.images)} 张图片...")
    
    for img_path in args.images:
        img_path = Path(img_path)
        if not img_path.exists():
            print(f"⚠️  图片不存在: {img_path}")
            continue
        
        print(f"\n处理: {img_path.name}")
        
        # 推理
        results = inference_bottomup(model, str(img_path))
        
        if not results or len(results) == 0:
            print(f"  ⚠️  未检测到姿态")
            continue
        
        # 获取预测结果
        data_sample = results[0]
        pred_instances = data_sample.pred_instances
        
        print(f"  检测到 {len(pred_instances)} 个人体实例")
        
        # 可视化
        img = cv2.imread(str(img_path))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # 绘制关键点
        visualizer.add_datasample(
            'result',
            img_rgb,
            data_sample=data_sample,
            draw_gt=False,
            draw_bbox=True,
            draw_heatmap=False,
            show=False,
            wait_time=0,
            out_file=None
        )
        
        vis_img = visualizer.get_image()
        vis_img_bgr = cv2.cvtColor(vis_img, cv2.COLOR_RGB2BGR)
        
        # 保存结果
        output_path = output_dir / f"{img_path.stem}_result.jpg"
        cv2.imwrite(str(output_path), vis_img_bgr)
        print(f"  ✓ 结果已保存: {output_path}")
        
        # 打印关键点信息
        for i, instance in enumerate(pred_instances):
            keypoints = instance.keypoints
            scores = instance.keypoint_scores
            print(f"    实例 {i+1}: {len(keypoints)} 个关键点")
            print(f"      平均置信度: {scores.mean():.3f}")
            print(f"      最高置信度: {scores.max():.3f}")
        
        # 显示图片（如果指定）
        if args.show:
            cv2.imshow('Result', vis_img_bgr)
            print("  按任意键继续...")
            cv2.waitKey(0)
            cv2.destroyAllWindows()
    
    print(f"\n✓ 推理完成！结果保存在: {output_dir}")


if __name__ == '__main__':
    main()

