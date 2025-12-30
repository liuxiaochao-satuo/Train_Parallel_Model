#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
对比不同推理方式的结果差异
1. extract_frames.py 提取的图片 → inference_images.py
2. inference_video.py 直接推理
3. inference_video.py 使用临时文件推理
"""

import cv2
import numpy as np
from pathlib import Path
import argparse
from mmcv import imread


def parse_args():
    parser = argparse.ArgumentParser(description='对比不同推理方式')
    parser.add_argument('video', help='输入视频路径')
    parser.add_argument('--frame-idx', type=int, default=0,
                        help='要对比的帧索引（默认：0）')
    parser.add_argument('--extracted-frame', type=str, default=None,
                        help='extract_frames.py 提取的图片路径')
    parser.add_argument('--output-dir', type=str, default='compare_output',
                        help='输出对比结果目录')
    return parser.parse_args()


def compare_image_reading_methods(video_path, frame_idx, extracted_frame_path, output_dir):
    """对比不同方式读取的图像"""
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("对比不同推理方式的图像读取差异")
    print("=" * 60)
    
    # 方式1: 从视频直接读取（模拟 inference_video.py 直接方式）
    print(f"\n1. 从视频直接读取帧 {frame_idx}...")
    cap = cv2.VideoCapture(str(video_path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame_bgr_direct = cap.read()
    cap.release()
    
    if not ret:
        print(f"❌ 无法读取帧 {frame_idx}")
        return
    
    frame_rgb_direct = cv2.cvtColor(frame_bgr_direct, cv2.COLOR_BGR2RGB)
    print(f"✓ 直接读取成功: {frame_rgb_direct.shape}")
    
    # 方式2: 从extract_frames.py保存的图片读取（模拟 inference_images.py）
    if extracted_frame_path and Path(extracted_frame_path).exists():
        print(f"\n2. 从保存的图片读取: {extracted_frame_path}")
        img_mmcv = imread(str(extracted_frame_path))  # 模拟 inference_images.py
        print(f"✓ mmcv.imread 读取成功: {img_mmcv.shape}")
        
        # 对比差异
        if frame_rgb_direct.shape == img_mmcv.shape:
            diff = np.abs(frame_rgb_direct.astype(np.float32) - img_mmcv.astype(np.float32))
            mse = np.mean(diff ** 2)
            mae = np.mean(np.abs(diff))
            
            print(f"\n3. 对比分析:")
            print(f"  直接读取 vs 保存后读取 (mmcv.imread):")
            print(f"    MSE: {mse:.6f}")
            print(f"    MAE: {mae:.6f}")
            print(f"    最大差异: {np.abs(diff).max():.2f}")
            print(f"    有差异像素比例: {np.sum(diff != 0) / diff.size * 100:.2f}%")
            
            if mse > 100.0 or mae > 5.0:
                print(f"\n  ⚠️  差异较大！")
                print(f"    可能原因:")
                print(f"    1. JPEG压缩损失（如果保存的是JPEG）")
                print(f"    2. mmcv.imread() 的额外处理")
                print(f"    3. 颜色空间转换差异")
            else:
                print(f"\n  ✓ 差异很小，两种方式基本一致")
            
            # 保存对比图像
            cv2.imwrite(str(output_dir / "direct_read_bgr.jpg"), frame_bgr_direct)
            cv2.imwrite(str(output_dir / "saved_read_mmcv.jpg"), 
                       cv2.cvtColor(img_mmcv, cv2.COLOR_RGB2BGR))
            
            if mse > 0:
                diff_normalized = (diff / diff.max() * 255).astype(np.uint8)
                cv2.imwrite(str(output_dir / "difference.jpg"), 
                           cv2.cvtColor(diff_normalized, cv2.COLOR_RGB2BGR))
                print(f"\n  ✓ 差异图已保存: {output_dir / 'difference.jpg'}")
        else:
            print(f"\n  ⚠️  形状不匹配!")
            print(f"    直接读取: {frame_rgb_direct.shape}")
            print(f"    保存读取: {img_mmcv.shape}")
    else:
        print(f"\n⚠️  未提供保存的图片路径，跳过对比")
        print(f"  提示: 先运行 extract_frames.py 提取帧")
    
    # 方式3: 模拟临时文件方式（保存后再读取）
    print(f"\n4. 模拟临时文件方式（保存为PNG后读取）...")
    temp_png_path = output_dir / f"temp_frame_{frame_idx:06d}.png"
    cv2.imwrite(str(temp_png_path), frame_bgr_direct)
    img_temp_png = imread(str(temp_png_path))
    
    if frame_rgb_direct.shape == img_temp_png.shape:
        diff_temp = np.abs(frame_rgb_direct.astype(np.float32) - img_temp_png.astype(np.float32))
        mse_temp = np.mean(diff_temp ** 2)
        mae_temp = np.mean(np.abs(diff_temp))
        
        print(f"  直接读取 vs PNG临时文件读取:")
        print(f"    MSE: {mse_temp:.6f}")
        print(f"    MAE: {mae_temp:.6f}")
        
        if mse_temp < 1.0 and mae_temp < 0.1:
            print(f"  ✓ PNG格式完全无损")
        else:
            print(f"  ⚠️  PNG仍有差异（可能是视频帧本身的质量问题）")
    
    print(f"\n" + "=" * 60)
    print(f"✓ 对比完成！结果保存在: {output_dir}")
    print("=" * 60)


def main():
    args = parse_args()
    compare_image_reading_methods(
        args.video,
        args.frame_idx,
        args.extracted_frame,
        args.output_dir
    )


if __name__ == '__main__':
    main()

