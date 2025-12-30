#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
诊断视频帧读取质量和方式的问题
对比视频帧和保存的图片之间的差异
"""

import cv2
import numpy as np
from pathlib import Path
import argparse
from mmcv import imread, imfrombytes


def parse_args():
    parser = argparse.ArgumentParser(description='诊断视频帧质量')
    parser.add_argument('video', help='输入视频路径')
    parser.add_argument('--frame-idx', type=int, default=0,
                        help='要检查的帧索引（默认：0）')
    parser.add_argument('--saved-image', type=str, default=None,
                        help='对应的保存图片路径（用于对比）')
    parser.add_argument('--output-dir', type=str, default='diagnose_output',
                        help='输出诊断结果目录')
    return parser.parse_args()


def analyze_image(img, name):
    """分析图像属性"""
    info = {
        'name': name,
        'shape': img.shape,
        'dtype': str(img.dtype),
        'min': float(img.min()),
        'max': float(img.max()),
        'mean': float(img.mean()),
        'std': float(img.std()),
        'is_contiguous': img.flags['C_CONTIGUOUS'],
        'memory_size_mb': img.nbytes / (1024 * 1024),
    }
    return info


def compare_images(img1, img2, name1, name2):
    """对比两张图片的差异
    
    返回的指标说明:
    - MSE (Mean Squared Error): 均方误差，衡量像素差异的平方平均值
      值越大差异越大。对于uint8图像(0-255)，MSE > 100通常表示明显差异
    - MAE (Mean Absolute Error): 平均绝对误差，衡量像素差异的平均值
      值越大差异越大。对于uint8图像，MAE > 5通常表示明显差异
    - max_diff: 最大像素差异
    - pixel_diff_ratio: 有差异的像素比例
    """
    if img1.shape != img2.shape:
        return {
            'shape_match': False,
            'shape1': img1.shape,
            'shape2': img2.shape,
        }
    
    diff = img1.astype(np.float32) - img2.astype(np.float32)
    
    return {
        'shape_match': True,
        'mse': float(np.mean(diff ** 2)),
        'mae': float(np.mean(np.abs(diff))),
        'max_diff': float(np.abs(diff).max()),
        'pixel_diff_count': int(np.sum(diff != 0)),
        'pixel_diff_ratio': float(np.sum(diff != 0) / diff.size),
    }


def main():
    args = parse_args()
    
    video_path = Path(args.video)
    if not video_path.exists():
        print(f"❌ 视频文件不存在: {video_path}")
        return
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("视频帧质量诊断")
    print("=" * 60)
    
    # 1. 从视频读取帧
    print(f"\n1. 从视频读取第 {args.frame_idx} 帧...")
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"❌ 无法打开视频文件: {video_path}")
        return
    
    # 跳转到指定帧
    cap.set(cv2.CAP_PROP_POS_FRAMES, args.frame_idx)
    ret, frame_bgr = cap.read()
    cap.release()
    
    if not ret:
        print(f"❌ 无法读取第 {args.frame_idx} 帧")
        return
    
    print(f"✓ 成功读取视频帧")
    print(f"  原始形状: {frame_bgr.shape}")
    print(f"  数据类型: {frame_bgr.dtype}")
    
    # 转换为RGB（模拟视频推理脚本）
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    
    # 分析视频帧
    video_frame_info = analyze_image(frame_rgb, "视频帧(RGB)")
    print(f"\n视频帧信息:")
    for key, value in video_frame_info.items():
        print(f"  {key}: {value}")
    
    # 保存视频帧到输出目录（用于后续对比）
    print(f"\n2. 保存视频帧到输出目录...")
    # 同时保存JPEG和PNG格式进行对比
    video_frame_jpg_path = output_dir / f"video_frame_{args.frame_idx:06d}.jpg"
    video_frame_png_path = output_dir / f"video_frame_{args.frame_idx:06d}.png"
    
    # 保存为JPEG（质量95，模拟视频推理脚本的临时文件方式）
    cv2.imwrite(str(video_frame_jpg_path), frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])
    print(f"✓ 视频帧已保存为JPEG (质量95): {video_frame_jpg_path}")
    
    # 保存为PNG（无损格式，用于对比）
    cv2.imwrite(str(video_frame_png_path), frame_bgr)
    print(f"✓ 视频帧已保存为PNG (无损): {video_frame_png_path}")
    
    video_frame_saved_path = video_frame_jpg_path  # 默认使用JPEG进行测试
    
    # 2. 如果提供了保存的图片，进行对比
    if args.saved_image:
        saved_img_path = Path(args.saved_image)
        if not saved_img_path.exists():
            print(f"\n⚠️  保存的图片不存在: {saved_img_path}")
            print(f"  提示: 可以使用以下方式获取保存的图片:")
            print(f"  1. 运行抽帧脚本: python tools/extract_frames.py {args.video} --start-frame {args.frame_idx} --end-frame {args.frame_idx} --output-dir <output_dir>")
            print(f"  2. 或者使用已保存的视频帧进行自对比: {video_frame_saved_path}")
            print(f"\n  继续使用保存的视频帧进行自对比测试...")
            saved_img_path = video_frame_saved_path
        else:
            print(f"\n2. 读取保存的图片: {saved_img_path}")
            
            # 方式1: 使用 cv2.imread（模拟图片推理脚本的可视化部分）
            img_cv2 = cv2.imread(str(saved_img_path))
            img_cv2_rgb = cv2.cvtColor(img_cv2, cv2.COLOR_BGR2RGB)
            
            # 方式2: 使用 mmcv.imread（模拟 inference_bottomup 内部处理）
            img_mmcv = imread(str(saved_img_path))
            
            print(f"✓ cv2.imread 读取:")
            print(f"  形状: {img_cv2_rgb.shape}, 类型: {img_cv2_rgb.dtype}")
            print(f"✓ mmcv.imread 读取:")
            print(f"  形状: {img_mmcv.shape}, 类型: {img_mmcv.dtype}")
            
            # 分析保存的图片
            saved_img_info = analyze_image(img_mmcv, "保存的图片(mmcv)")
            print(f"\n保存的图片信息:")
            for key, value in saved_img_info.items():
                print(f"  {key}: {value}")
            
            # 对比视频帧和保存的图片
            print(f"\n3. 对比分析:")
            print(f"  视频帧 vs 保存的图片 (mmcv):")
            comparison = compare_images(frame_rgb, img_mmcv, "视频帧", "保存的图片")
            for key, value in comparison.items():
                print(f"    {key}: {value}")
            
            # 检查形状是否匹配
            if frame_rgb.shape != img_mmcv.shape:
                print(f"\n⚠️  警告: 形状不匹配!")
                print(f"  视频帧: {frame_rgb.shape}")
                print(f"  保存图片: {img_mmcv.shape}")
                print(f"  这可能导致推理结果不同!")
            
            # 保存对比图像
            print(f"\n4. 保存诊断图像到: {output_dir}")
            
            # 保存视频帧（如果还没保存）
            if not video_frame_saved_path.exists():
                cv2.imwrite(str(video_frame_saved_path), frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])
            cv2.imwrite(str(output_dir / "video_frame_rgb.jpg"), 
                       cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR))
            
            # 保存保存的图片
            cv2.imwrite(str(output_dir / "saved_image_cv2.jpg"), img_cv2)
            cv2.imwrite(str(output_dir / "saved_image_mmcv.jpg"), 
                       cv2.cvtColor(img_mmcv, cv2.COLOR_RGB2BGR))
            
            # 计算并保存差异图
            if frame_rgb.shape == img_mmcv.shape:
                diff = np.abs(frame_rgb.astype(np.float32) - img_mmcv.astype(np.float32))
                if diff.max() > 0:
                    diff_normalized = (diff / diff.max() * 255).astype(np.uint8)
                else:
                    diff_normalized = diff.astype(np.uint8)
                cv2.imwrite(str(output_dir / "difference.jpg"), 
                           cv2.cvtColor(diff_normalized, cv2.COLOR_RGB2BGR))
                print(f"  ✓ 差异图已保存")
            else:
                print(f"  ⚠️  形状不匹配，无法生成差异图")
    
    # 3. 如果没有提供保存的图片，进行自对比测试（视频帧保存后再读取）
    if not args.saved_image or not Path(args.saved_image).exists():
        print(f"\n3. 自对比测试（视频帧保存后再读取）:")
        
        # 测试JPEG格式
        if video_frame_jpg_path.exists():
            print(f"\n  3.1 测试JPEG格式 (质量95):")
            print(f"    读取: {video_frame_jpg_path}")
            img_jpg_read = imread(str(video_frame_jpg_path))
            
            print(f"    对比原始视频帧 vs JPEG保存后读取:")
            comparison_jpg = compare_images(frame_rgb, img_jpg_read, 
                                            "原始视频帧", "JPEG保存后读取")
            for key, value in comparison_jpg.items():
                if key != 'shape_match' or value:
                    print(f"      {key}: {value}")
            
            mse_jpg = comparison_jpg.get('mse', 0)
            mae_jpg = comparison_jpg.get('mae', 0)
            if mse_jpg > 100.0 or mae_jpg > 5.0:
                print(f"\n    ⚠️  警告: JPEG压缩导致明显质量损失!")
                print(f"      MSE: {mse_jpg:.2f} (理想值 < 10)")
                print(f"      MAE: {mae_jpg:.2f} (理想值 < 2)")
                print(f"      建议: 使用PNG格式或提高JPEG质量到100")
            else:
                print(f"\n    ✓ JPEG质量损失在可接受范围内")
        
        # 测试PNG格式（无损）
        if video_frame_png_path.exists():
            print(f"\n  3.2 测试PNG格式 (无损):")
            print(f"    读取: {video_frame_png_path}")
            img_png_read = imread(str(video_frame_png_path))
            
            print(f"    对比原始视频帧 vs PNG保存后读取:")
            comparison_png = compare_images(frame_rgb, img_png_read, 
                                            "原始视频帧", "PNG保存后读取")
            for key, value in comparison_png.items():
                if key != 'shape_match' or value:
                    print(f"      {key}: {value}")
            
            mse_png = comparison_png.get('mse', 0)
            mae_png = comparison_png.get('mae', 0)
            if mse_png < 0.01 and mae_png < 0.1:
                print(f"\n    ✓ PNG格式完全无损，质量损失可忽略")
            else:
                print(f"\n    ⚠️  PNG格式仍有轻微差异（可能是浮点精度问题）")
    
    # 4. 检查视频属性
    print(f"\n5. 视频属性:")
    cap = cv2.VideoCapture(str(video_path))
    if cap.isOpened():
        print(f"  编码格式: {cap.get(cv2.CAP_PROP_FOURCC)}")
        print(f"  帧率: {cap.get(cv2.CAP_PROP_FPS)}")
        print(f"  分辨率: {int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}")
        print(f"  总帧数: {int(cap.get(cv2.CAP_PROP_FRAME_COUNT))}")
        print(f"  亮度: {cap.get(cv2.CAP_PROP_BRIGHTNESS)}")
        print(f"  对比度: {cap.get(cv2.CAP_PROP_CONTRAST)}")
        print(f"  饱和度: {cap.get(cv2.CAP_PROP_SATURATION)}")
    cap.release()
    
    # 5. 测试不同的读取方式
    print(f"\n6. 测试不同的图像读取方式:")
    
    # 使用保存的视频帧进行测试（如果用户没有提供保存的图片）
    test_img_path = None
    if args.saved_image and Path(args.saved_image).exists():
        test_img_path = args.saved_image
    elif video_frame_saved_path.exists():
        test_img_path = str(video_frame_saved_path)
        print(f"  使用保存的视频帧进行测试: {test_img_path}")
    
    if test_img_path:
        
        # 方式1: 通过文件路径（模拟图片推理）
        print(f"  方式1: 通过文件路径 (mmcv.imread)")
        img_path_read = imread(test_img_path)
        print(f"    形状: {img_path_read.shape}, 类型: {img_path_read.dtype}")
        
        # 方式2: 通过numpy数组（模拟视频推理）
        print(f"  方式2: 通过numpy数组 (直接使用)")
        img_array = frame_rgb.copy()
        print(f"    形状: {img_array.shape}, 类型: {img_array.dtype}")
        
        # 检查两者是否相同（如果视频帧和保存的图片是同一帧）
        if frame_rgb.shape == img_path_read.shape:
            diff_path_vs_array = np.abs(img_path_read.astype(np.float32) - 
                                       img_array.astype(np.float32))
            mse = np.mean(diff_path_vs_array ** 2)
            mae = np.mean(np.abs(diff_path_vs_array))
            print(f"    路径读取 vs 数组读取的差异:")
            print(f"      MSE: {mse:.6f}")
            print(f"      MAE: {mae:.6f}")
            
            if mse > 10.0:
                print(f"    ⚠️  差异较大，说明文件读取和直接使用数组的处理方式不同")
            else:
                print(f"    ✓ 差异很小，两种方式基本一致")
            
            if mse > 10.0:
                print(f"    ⚠️  差异较大，说明文件读取和直接使用数组的处理方式不同")
            else:
                print(f"    ✓ 差异很小，两种方式基本一致")
    else:
        print(f"  ⚠️  无法测试（需要有效的图片文件）")
    
    print(f"\n" + "=" * 60)
    print(f"✓ 诊断完成！结果保存在: {output_dir}")
    print("=" * 60)
    
    # 总结可能的问题
    print(f"\n可能的问题总结:")
    print(f"1. 视频压缩损失: 视频帧可能经过压缩，质量低于原始图片")
    print(f"2. 解码器差异: cv2.VideoCapture 和 mmcv.imread 可能使用不同的解码器")
    print(f"3. 颜色空间: 确保 BGR->RGB 转换正确")
    print(f"4. 数据类型: 确保都是 uint8 类型")
    print(f"5. 图像连续性: 确保数组是 C_CONTIGUOUS")


if __name__ == '__main__':
    main()

