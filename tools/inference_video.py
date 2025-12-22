#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
使用 Combined 方法（损失权重 + 加权采样）对视频进行姿态估计推理
"""

import sys
import argparse
from pathlib import Path

# 添加项目根目录到 Python 路径，以便导入自定义模块
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import os
import cv2
import numpy as np
import torch

# 设置CUDA环境变量（解决LXC容器中的CUDA初始化问题）
if 'CUDA_HOME' not in os.environ:
    cuda_paths = ['/usr/local/cuda-13.0', '/usr/local/cuda', '/usr/local/cuda-12.0']
    for cuda_path in cuda_paths:
        if os.path.exists(cuda_path):
            os.environ['CUDA_HOME'] = cuda_path
            if 'LD_LIBRARY_PATH' not in os.environ:
                os.environ['LD_LIBRARY_PATH'] = f'{cuda_path}/lib64'
            elif cuda_path not in os.environ['LD_LIBRARY_PATH']:
                os.environ['LD_LIBRARY_PATH'] = f'{cuda_path}/lib64:{os.environ["LD_LIBRARY_PATH"]}'
            break

# 修复torch.load的weights_only问题
_original_torch_load = torch.load
def patched_torch_load(*args, **kwargs):
    if 'weights_only' not in kwargs:
        kwargs['weights_only'] = False
    return _original_torch_load(*args, **kwargs)
torch.load = patched_torch_load

from mmpose.apis import inference_bottomup, init_model
from mmpose.visualization import PoseLocalVisualizer


# Combined 方法的配置
COMBINED_CONFIG = {
    'config': 'configs/body_2d_keypoint/dekr/coco/dekr_hrnet-w32_parallel_ablation_combined.py',
    'checkpoint': '/data/lxc/outputs/train_parallel_model/ablation_experiments/combined/best_coco_AP_epoch_110.pth',
    'name': 'Combined (Loss Weight + Weighted Sampling)'
}


def parse_args():
    parser = argparse.ArgumentParser(description='使用 Combined 方法对视频进行姿态估计推理')
    parser.add_argument('video', help='输入视频路径')
    parser.add_argument('--output', type=str, default=None, 
                        help='输出视频路径（默认：输入文件名_combined.mp4）')
    parser.add_argument('--device', type=str, default='cpu', 
                        help='推理设备（默认：cpu，可选：cuda）')
    parser.add_argument('--fps', type=int, default=None,
                        help='输出视频帧率（默认：使用输入视频帧率）')
    parser.add_argument('--skip-frames', type=int, default=0,
                        help='跳过的帧数（默认：0，处理所有帧）')
    parser.add_argument('--frame-interval', type=int, default=1,
                        help='每隔多少帧处理一帧（默认：1，处理所有帧；例如：10表示每隔10帧处理一帧）')
    parser.add_argument('--show', action='store_true', help='显示处理过程')
    parser.add_argument('--config-dir', type=str, default=None,
                        help='配置文件目录（默认：使用项目根目录）')
    parser.add_argument('--score-thr', type=float, default=None,
                        help='检测阈值（默认：使用模型配置，建议0.1-0.3，越低检测越多）')
    parser.add_argument('--nms-thr', type=float, default=None,
                        help='NMS阈值（默认：使用模型配置，建议0.3-0.5）')
    return parser.parse_args()


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


def main():
    args = parse_args()
    
    # 将 gpu 转换为 cuda（PyTorch 使用 cuda 而不是 gpu）
    device_input = args.device.lower()
    if device_input == 'gpu':
        device_input = 'cuda'
    
    # 检查设备可用性，如果 CUDA 不可用则自动回退到 CPU
    actual_device = device_input
    if device_input == 'cuda':
        try:
            if not torch.cuda.is_available():
                print("⚠️  CUDA 不可用，自动切换到 CPU 模式")
                actual_device = 'cpu'
            elif torch.cuda.device_count() == 0:
                print("⚠️  CUDA 设备数量为 0，自动切换到 CPU 模式")
                actual_device = 'cpu'
            else:
                print(f"✓ 检测到 {torch.cuda.device_count()} 个 CUDA 设备")
                print(f"  使用设备: {torch.cuda.get_device_name(0)}")
        except Exception as e:
            print(f"⚠️  CUDA 检测失败: {e}")
            print("⚠️  自动切换到 CPU 模式")
            actual_device = 'cpu'
    
    print(f"最终使用设备: {actual_device}")
    
    # 获取项目根目录
    project_root = Path(__file__).parent.parent
    if args.config_dir:
        project_root = Path(args.config_dir)
    
    # 检查输入视频文件
    video_path = Path(args.video)
    if not video_path.exists():
        print(f"❌ 视频文件不存在: {video_path}")
        return
    
    # 设置输出路径
    if args.output:
        output_path = Path(args.output)
        # 如果输出路径是目录，自动生成文件名
        if output_path.is_dir() or (not output_path.suffix):
            # 确保目录存在
            output_path.mkdir(parents=True, exist_ok=True)
            # 在目录中生成输出文件名
            output_path = output_path / f"{video_path.stem}_combined.mp4"
        else:
            # 如果是文件路径，确保父目录存在
            output_path.parent.mkdir(parents=True, exist_ok=True)
    else:
        output_path = video_path.parent / f"{video_path.stem}_combined.mp4"
    
    # 确保输出路径有正确的扩展名
    if not output_path.suffix:
        output_path = output_path.with_suffix('.mp4')
    
    print("=" * 60)
    print("加载 Combined 模型...")
    print("=" * 60)
    
    # 加载模型
    config_path = project_root / COMBINED_CONFIG['config']
    checkpoint_path = Path(COMBINED_CONFIG['checkpoint'])
    
    if not config_path.exists():
        print(f"❌ 配置文件不存在: {config_path}")
        return
    
    if not checkpoint_path.exists():
        print(f"❌ 权重文件不存在: {checkpoint_path}")
        return
    
    print(f"配置: {config_path}")
    print(f"权重: {checkpoint_path}")
    
    try:
        model = init_model(str(config_path), str(checkpoint_path), device=actual_device)
        visualizer = PoseLocalVisualizer()
        visualizer.set_dataset_meta(model.dataset_meta)
        
        # 调整检测阈值（如果指定）
        if args.score_thr is not None or args.nms_thr is not None:
            if hasattr(model, 'head') and hasattr(model.head, 'test_cfg'):
                test_cfg = model.head.test_cfg.copy()
                if args.score_thr is not None:
                    if 'score_thr' in test_cfg:
                        old_thr = test_cfg['score_thr']
                        test_cfg['score_thr'] = args.score_thr
                        print(f"  调整检测阈值: {old_thr} -> {args.score_thr}")
                    else:
                        test_cfg['score_thr'] = args.score_thr
                        print(f"  设置检测阈值: {args.score_thr}")
                if args.nms_thr is not None:
                    if 'nms_thr' in test_cfg:
                        old_thr = test_cfg['nms_thr']
                        test_cfg['nms_thr'] = args.nms_thr
                        print(f"  调整NMS阈值: {old_thr} -> {args.nms_thr}")
                    else:
                        test_cfg['nms_thr'] = args.nms_thr
                        print(f"  设置NMS阈值: {args.nms_thr}")
                model.head.test_cfg = test_cfg
                model.test_cfg = test_cfg
        
        print(f"✓ 模型加载成功")
    except Exception as e:
        error_msg = str(e)
        # 如果 CUDA 错误，尝试用 CPU 重新加载
        if 'cuda' in error_msg.lower() and actual_device == 'cuda':
            print(f"⚠️  CUDA 加载失败，尝试使用 CPU...")
            try:
                model = init_model(str(config_path), str(checkpoint_path), device='cpu')
                visualizer = PoseLocalVisualizer()
                visualizer.set_dataset_meta(model.dataset_meta)
                actual_device = 'cpu'
                print(f"✓ 使用 CPU 加载成功")
            except Exception as e2:
                print(f"❌ CPU 加载也失败: {e2}")
                return
        else:
            print(f"❌ 加载失败: {e}")
            return
    
    print("\n" + "=" * 60)
    print("开始处理视频...")
    print("=" * 60)
    
    # 打开输入视频
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"❌ 无法打开视频文件: {video_path}")
        return
    
    # 获取视频属性
    fps = int(cap.get(cv2.CAP_PROP_FPS)) if args.fps is None else args.fps
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"视频信息:")
    print(f"  分辨率: {width}x{height}")
    print(f"  帧率: {fps} FPS")
    print(f"  总帧数: {total_frames}")
    print(f"  跳过帧数: {args.skip_frames}")
    print(f"  帧间隔: {args.frame_interval} (每隔{args.frame_interval}帧处理一帧)")
    if args.frame_interval > 1:
        estimated_frames = (total_frames - args.skip_frames) // args.frame_interval
        print(f"  预计处理帧数: ~{estimated_frames} 帧")
    
    # 创建视频写入器
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    
    if not out.isOpened():
        print(f"❌ 无法创建输出视频文件: {output_path}")
        cap.release()
        return
    
    frame_count = 0
    processed_count = 0
    
    print(f"\n开始逐帧处理...")
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # 跳过指定数量的帧
            if frame_count <= args.skip_frames:
                continue
            
            # 每隔N帧处理一帧
            if (frame_count - args.skip_frames - 1) % args.frame_interval != 0:
                # 不处理这一帧，直接写入原帧到输出视频
                out.write(frame)
                continue
            
            # 处理当前帧
            # 确保图像数据类型正确（uint8）
            if frame.dtype != np.uint8:
                frame = frame.astype(np.uint8)
            
            # 转换为RGB（inference_bottomup期望RGB格式）
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # 确保图像连续存储（提高性能）
            if not frame_rgb.flags['C_CONTIGUOUS']:
                frame_rgb = np.ascontiguousarray(frame_rgb)
            
            try:
                # 推理
                results = inference_bottomup(model, frame_rgb)
                
                if results and len(results) > 0:
                    # 获取预测结果
                    data_sample = results[0]
                    pred_instances = data_sample.pred_instances
                    
                    # 可视化
                    visualizer.add_datasample(
                        'result',
                        frame_rgb.copy(),
                        data_sample=data_sample,
                        draw_gt=False,
                        draw_bbox=True,
                        draw_heatmap=False,
                        show=False,
                        wait_time=0,
                        out_file=None
                    )
                    
                    vis_img = visualizer.get_image()
                    result_frame = cv2.cvtColor(vis_img, cv2.COLOR_RGB2BGR)
                    
                    # 添加方法标签
                    result_frame = add_text_label(
                        result_frame, 
                        COMBINED_CONFIG['name'], 
                        position=(10, 30)
                    )
                    
                    # 添加帧信息
                    frame_info = f"Frame: {frame_count}/{total_frames}"
                    result_frame = add_text_label(
                        result_frame, 
                        frame_info, 
                        position=(10, 70)
                    )
                else:
                    # 未检测到姿态，使用原图
                    result_frame = frame.copy()
                    result_frame = add_text_label(
                        result_frame, 
                        COMBINED_CONFIG['name'], 
                        position=(10, 30)
                    )
                    frame_info = f"Frame: {frame_count}/{total_frames} (No detection)"
                    result_frame = add_text_label(
                        result_frame, 
                        frame_info, 
                        position=(10, 70)
                    )
                
                # 写入输出视频
                out.write(result_frame)
                processed_count += 1
                
                # 显示进度
                if processed_count % 10 == 0 or processed_count == 1:
                    progress = (frame_count / total_frames) * 100 if total_frames > 0 else 0
                    print(f"  处理进度: {processed_count} 帧 (总帧数: {frame_count}/{total_frames}, {progress:.1f}%)")
                
                # 显示当前帧（如果指定）
                if args.show:
                    display_frame = result_frame.copy()
                    # 如果图片太大，先缩放显示
                    max_display_size = 1280
                    if width > max_display_size:
                        scale = max_display_size / width
                        new_w = int(width * scale)
                        new_h = int(height * scale)
                        display_frame = cv2.resize(display_frame, (new_w, new_h))
                    
                    cv2.imshow('Video Inference', display_frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        print("\n用户中断处理")
                        break
                        
            except Exception as e:
                print(f"  ⚠️  处理第 {frame_count} 帧时出错: {e}")
                # 写入原帧
                out.write(frame)
                continue
    
    except KeyboardInterrupt:
        print("\n\n用户中断处理")
    finally:
        # 释放资源
        cap.release()
        out.release()
        if args.show:
            cv2.destroyAllWindows()
    
    print("\n" + "=" * 60)
    print(f"✓ 视频处理完成！")
    print(f"  输入视频: {video_path}")
    print(f"  输出视频: {output_path}")
    print(f"  总帧数: {total_frames}")
    print(f"  处理帧数: {processed_count} 帧")
    if args.frame_interval > 1:
        print(f"  帧间隔: 每隔 {args.frame_interval} 帧处理一帧")
        print(f"  处理比例: {processed_count}/{total_frames} ({processed_count/total_frames*100:.1f}%)")
    print("=" * 60)


if __name__ == '__main__':
    main()

