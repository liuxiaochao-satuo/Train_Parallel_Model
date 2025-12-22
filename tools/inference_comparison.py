#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
使用四种消融实验方法对图片进行对比推理，并将结果拼接在一起
"""

import sys
import argparse
from pathlib import Path

# 添加项目根目录到 Python 路径，以便导入自定义模块
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import cv2
import numpy as np
import torch

# 修复torch.load的weights_only问题
_original_torch_load = torch.load
def patched_torch_load(*args, **kwargs):
    if 'weights_only' not in kwargs:
        kwargs['weights_only'] = False
    return _original_torch_load(*args, **kwargs)
torch.load = patched_torch_load

from mmengine.config import Config
from mmpose.apis import inference_bottomup, init_model
from mmpose.visualization import PoseLocalVisualizer


# 四种实验方法的配置
EXPERIMENT_CONFIGS = {
    'baseline': {
        'config': 'configs/body_2d_keypoint/dekr/coco/dekr_hrnet-w32_parallel_baseline.py',
        'checkpoint': '/data/lxc/outputs/train_parallel_model/ablation_experiments/baseline/best_coco_AP_epoch_120.pth',
        'name': 'Baseline'
    },
    'loss_weight': {
        'config': 'configs/body_2d_keypoint/dekr/coco/dekr_hrnet-w32_parallel_ablation_loss_weight.py',
        'checkpoint': '/data/lxc/outputs/train_parallel_model/ablation_experiments/loss_weight_only/best_coco_AP_epoch_130.pth',
        'name': 'Loss Weight Only'
    },
    'weighted_sampling': {
        'config': 'configs/body_2d_keypoint/dekr/coco/dekr_hrnet-w32_parallel_ablation_weighted_sampling.py',
        'checkpoint': '/data/lxc/outputs/train_parallel_model/ablation_experiments/weighted_sampling_only/best_coco_AP_epoch_140.pth',
        'name': 'Weighted Sampling Only'
    },
    'combined': {
        'config': 'configs/body_2d_keypoint/dekr/coco/dekr_hrnet-w32_parallel_ablation_combined.py',
        'checkpoint': '/data/lxc/outputs/train_parallel_model/ablation_experiments/combined/best_coco_AP_epoch_110.pth',
        'name': 'Combined'
    }
}


def parse_args():
    parser = argparse.ArgumentParser(description='使用四种方法对图片进行对比推理')
    parser.add_argument(
        'inputs',
        nargs='+',
        help='要推理的路径：可以是图片文件或文件夹（会递归查找其中的图片）')
    parser.add_argument('--output-dir', type=str, default='inference_comparison_results', 
                        help='输出目录（默认：inference_comparison_results）')
    parser.add_argument('--show', action='store_true', help='显示结果图片')
    parser.add_argument('--device', type=str, default='cpu', 
                        help='推理设备（默认：cpu，可选：cuda）')
    parser.add_argument('--config-dir', type=str, default=None,
                        help='配置文件目录（默认：使用项目根目录）')
    return parser.parse_args()


def collect_image_paths(inputs):
    """从输入的文件/文件夹列表中收集所有图片路径."""
    image_exts = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
    image_paths = []

    for p in inputs:
        path = Path(p)
        if not path.exists():
            print(f"⚠️  路径不存在，跳过: {path}")
            continue

        if path.is_file():
            if path.suffix.lower() in image_exts:
                image_paths.append(path)
            else:
                print(f"⚠️  非图片文件，跳过: {path}")
        elif path.is_dir():
            # 递归遍历文件夹
            for sub_path in path.rglob('*'):
                if sub_path.is_file() and sub_path.suffix.lower() in image_exts:
                    image_paths.append(sub_path)

    # 去重并排序，保证结果稳定
    image_paths = sorted(set(image_paths))
    return image_paths


def add_text_label(img, text, position=(10, 30), font_scale=0.8, thickness=2):
    """
    在图片上添加文本标签（带背景框）
    
    Args:
        img: 输入图片（BGR格式）
        text: 要添加的文本
        position: 文本位置 (x, y)
        font_scale: 字体大小
        thickness: 字体粗细
    """
    font = cv2.FONT_HERSHEY_SIMPLEX
    (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    
    # 计算背景框位置
    x, y = position
    padding = 5
    bg_x1 = x - padding
    bg_y1 = y - text_height - padding
    bg_x2 = x + text_width + padding
    bg_y2 = y + baseline + padding
    
    # 绘制半透明背景框
    overlay = img.copy()
    cv2.rectangle(overlay, (bg_x1, bg_y1), (bg_x2, bg_y2), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.7, img, 0.3, 0, img)
    
    # 绘制文本
    cv2.putText(img, text, (x, y), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
    
    return img


def resize_to_same_size(images):
    """
    将所有图片调整为相同尺寸（使用最大尺寸）
    
    Args:
        images: 图片列表
        
    Returns:
        调整后的图片列表
    """
    if not images:
        return images
    
    # 找到最大宽度和高度
    max_h = max(img.shape[0] for img in images)
    max_w = max(img.shape[1] for img in images)
    
    resized_images = []
    for img in images:
        h, w = img.shape[:2]
        if h != max_h or w != max_w:
            # 保持宽高比，在周围填充黑色
            scale = min(max_w / w, max_h / h)
            new_w = int(w * scale)
            new_h = int(h * scale)
            
            resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            
            # 创建黑色背景
            padded = np.zeros((max_h, max_w, 3), dtype=img.dtype)
            
            # 计算居中位置
            y_offset = (max_h - new_h) // 2
            x_offset = (max_w - new_w) // 2
            
            padded[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
            resized_images.append(padded)
        else:
            resized_images.append(img)
    
    return resized_images


def concatenate_images(images, labels):
    """
    将四张图片拼接成2x2网格
    
    Args:
        images: 图片列表（BGR格式）
        labels: 标签列表
        
    Returns:
        拼接后的图片
    """
    # 调整所有图片到相同尺寸
    images = resize_to_same_size(images)
    
    # 添加标签到每张图片
    labeled_images = []
    for img, label in zip(images, labels):
        labeled_img = add_text_label(img.copy(), label, position=(10, 30))
        labeled_images.append(labeled_img)
    
    # 确保有4张图片
    while len(labeled_images) < 4:
        h, w = labeled_images[0].shape[:2] if labeled_images else (512, 512)
        black_img = np.zeros((h, w, 3), dtype=np.uint8)
        labeled_images.append(black_img)
    
    # 拼接成2x2网格
    top_row = np.hstack([labeled_images[0], labeled_images[1]])
    bottom_row = np.hstack([labeled_images[2], labeled_images[3]])
    result = np.vstack([top_row, bottom_row])
    
    return result


def main():
    args = parse_args()
    
    # 检查设备可用性，如果 CUDA 不可用则自动回退到 CPU
    actual_device = args.device
    if args.device == 'cuda':
        if not torch.cuda.is_available():
            print("⚠️  CUDA 不可用，自动切换到 CPU 模式")
            actual_device = 'cpu'
        elif torch.cuda.device_count() == 0:
            print("⚠️  CUDA 设备数量为 0，自动切换到 CPU 模式")
            actual_device = 'cpu'
    
    print(f"使用设备: {actual_device}")
    
    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 获取项目根目录
    project_root = Path(__file__).parent.parent
    if args.config_dir:
        project_root = Path(args.config_dir)
    
    print("=" * 60)
    print("加载四种实验方法的模型...")
    print("=" * 60)
    
    # 加载所有模型
    models = {}
    visualizers = {}
    
    for exp_key, exp_config in EXPERIMENT_CONFIGS.items():
        config_path = project_root / exp_config['config']
        checkpoint_path = Path(exp_config['checkpoint'])
        
        if not config_path.exists():
            print(f"⚠️  配置文件不存在: {config_path}")
            continue
        
        if not checkpoint_path.exists():
            print(f"⚠️  权重文件不存在: {checkpoint_path}")
            continue
        
        print(f"\n加载 {exp_config['name']}...")
        print(f"  配置: {config_path}")
        print(f"  权重: {checkpoint_path}")
        
        try:
            model = init_model(str(config_path), str(checkpoint_path), device=actual_device)
            visualizer = PoseLocalVisualizer()
            visualizer.set_dataset_meta(model.dataset_meta)
            
            models[exp_key] = model
            visualizers[exp_key] = visualizer
            print(f"  ✓ 加载成功")
        except Exception as e:
            error_msg = str(e)
            # 如果 CUDA 错误，尝试用 CPU 重新加载
            if 'cuda' in error_msg.lower() and actual_device == 'cuda':
                print(f"  ⚠️  CUDA 加载失败，尝试使用 CPU...")
                try:
                    model = init_model(str(config_path), str(checkpoint_path), device='cpu')
                    visualizer = PoseLocalVisualizer()
                    visualizer.set_dataset_meta(model.dataset_meta)
                    
                    models[exp_key] = model
                    visualizers[exp_key] = visualizer
                    print(f"  ✓ 使用 CPU 加载成功")
                except Exception as e2:
                    print(f"  ✗ CPU 加载也失败: {e2}")
                    continue
            else:
                print(f"  ✗ 加载失败: {e}")
                continue
    
    if not models:
        print("\n❌ 没有成功加载任何模型，退出")
        return
    
    print(f"\n成功加载 {len(models)} 个模型")

    # 收集要推理的所有图片
    image_paths = collect_image_paths(args.inputs)
    if not image_paths:
        print("\n❌ 未找到任何可用图片，请检查输入路径（支持文件和文件夹）")
        return

    print(f"\n开始推理 {len(image_paths)} 张图片...")
    
    for img_path in image_paths:
        img_path = Path(img_path)
        if not img_path.exists():
            print(f"\n⚠️  图片不存在: {img_path}")
            continue
        
        print(f"\n{'=' * 60}")
        print(f"处理图片: {img_path.name}")
        print(f"{'=' * 60}")
        
        # 读取原始图片
        original_img = cv2.imread(str(img_path))
        if original_img is None:
            print(f"  ⚠️  无法读取图片: {img_path}")
            continue
        
        img_rgb = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
        
        # 对每种方法进行推理
        result_images = []
        result_labels = []
        
        for exp_key, exp_config in EXPERIMENT_CONFIGS.items():
            if exp_key not in models:
                continue
            
            print(f"\n推理方法: {exp_config['name']}")
            
            try:
                model = models[exp_key]
                visualizer = visualizers[exp_key]
                
                # 推理
                results = inference_bottomup(model, str(img_path))
                
                if not results or len(results) == 0:
                    print(f"  ⚠️  未检测到姿态")
                    # 使用原图作为占位符
                    result_img = original_img.copy()
                else:
                    # 获取预测结果
                    data_sample = results[0]
                    pred_instances = data_sample.pred_instances
                    
                    print(f"  ✓ 检测到 {len(pred_instances)} 个人体实例")
                    
                    # 可视化
                    visualizer.add_datasample(
                        'result',
                        img_rgb.copy(),
                        data_sample=data_sample,
                        draw_gt=False,
                        draw_bbox=True,
                        draw_heatmap=False,
                        show=False,
                        wait_time=0,
                        out_file=None
                    )
                    
                    vis_img = visualizer.get_image()
                    result_img = cv2.cvtColor(vis_img, cv2.COLOR_RGB2BGR)
                
                result_images.append(result_img)
                result_labels.append(exp_config['name'])
                
            except Exception as e:
                print(f"  ✗ 推理失败: {e}")
                # 使用原图作为占位符
                result_images.append(original_img.copy())
                result_labels.append(f"{exp_config['name']} (Error)")
        
        # 拼接图片
        if result_images:
            print(f"\n拼接结果图片...")
            concatenated_img = concatenate_images(result_images, result_labels)
            
            # 保存结果
            output_path = output_dir / f"{img_path.stem}_comparison.jpg"
            cv2.imwrite(str(output_path), concatenated_img)
            print(f"✓ 对比结果已保存: {output_path}")
            
            # 显示图片（如果指定）
            if args.show:
                # 如果图片太大，先缩放显示
                display_img = concatenated_img.copy()
                h, w = display_img.shape[:2]
                max_display_size = 1920
                if w > max_display_size:
                    scale = max_display_size / w
                    new_w = int(w * scale)
                    new_h = int(h * scale)
                    display_img = cv2.resize(display_img, (new_w, new_h))
                
                cv2.imshow('Comparison Result', display_img)
                print("  按任意键继续...")
                cv2.waitKey(0)
                cv2.destroyAllWindows()
    
    print(f"\n{'=' * 60}")
    print(f"✓ 推理完成！结果保存在: {output_dir}")
    print(f"{'=' * 60}")


if __name__ == '__main__':
    main()

