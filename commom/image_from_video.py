import cv2
import os
import sys
import numpy as np
from pathlib import Path


def extract_frames(video_path, output_dir, frame_interval=15, motion_threshold=15.0):
    """
    从视频中提取图片，结合固定间隔采样和运动检测采样
    
    参数:
        video_path (str): 输入视频文件路径
        output_dir (str): 输出图片保存目录
        frame_interval (int): 固定间隔，每隔多少帧提取一次（默认15帧）
        motion_threshold (float): 运动检测阈值，值越小越敏感（默认15.0）
    
    返回:
        提取的图片数量
    """
    # 创建输出目录
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # 打开视频文件
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"错误：无法打开视频文件 {video_path}")
        return 0
    
    # 获取视频信息
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"视频信息:")
    print(f"  分辨率: {width}x{height}")
    print(f"  帧率: {fps:.2f} fps")
    print(f"  总帧数: {total_frames}")
    print(f"  固定间隔: 每 {frame_interval} 帧提取一次")
    print(f"  运动阈值: {motion_threshold}")
    print(f"  预计提取: {total_frames // frame_interval} 张图片（仅固定间隔）")
    
    frame_count = 0
    saved_count = 0
    prev_gray = None
    
    # 获取视频文件名（不含扩展名）
    video_name = Path(video_path).stem
    
    # 读取视频帧
    while True:
        ret, frame = cap.read()
        
        if not ret:
            break
        
        # 转灰度，准备算差分
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        save_this = False
        
        # 规则A：固定间隔采样
        if frame_count % frame_interval == 0:
            save_this = True
        
        # 规则B：动作突变采样（运动能量高）
        if prev_gray is not None:
            diff = cv2.absdiff(gray, prev_gray)
            motion_score = float(np.mean(diff))
            if motion_score > motion_threshold:
                save_this = True
        
        if save_this:
            # 生成输出文件名：视频名_帧序号.jpg
            output_filename = f"{video_name}_frame_{frame_count:06d}.jpg"
            output_path = os.path.join(output_dir, output_filename)
            
            # 保存图片
            cv2.imwrite(output_path, frame)
            saved_count += 1
        
        prev_gray = gray
        frame_count += 1
    
    # 释放资源
    cap.release()
    
    print(f"\n提取完成！")
    print(f"总共保存了 {saved_count} 张图片到: {output_dir}")
    
    return saved_count


if __name__ == "__main__":
    # 检查命令行参数
    if len(sys.argv) < 2:
        print("使用方法: python image_from_video.py <视频路径>")
        print("示例: python image_from_video.py /path/to/video.mp4")
        sys.exit(1)
    
    # 从命令行获取视频路径
    video_path = sys.argv[1]
    
    # 验证视频文件是否存在
    if not os.path.exists(video_path):
        print(f"错误：视频文件不存在: {video_path}")
        sys.exit(1)
    
    # 固定参数
    output_dir = "/media/satuo/E05EEE285EEDF6E6/video_dataset/school/姿态估计数据集/全套后摆下"     # 输出目录
    frame_interval = 15               # 每隔15帧提取一次
    motion_threshold = 15.0           # 运动检测阈值
    
    # 调用函数
    extract_frames(video_path, output_dir, frame_interval, motion_threshold)

