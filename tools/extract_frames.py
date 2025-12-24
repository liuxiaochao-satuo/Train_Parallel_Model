#!/usr/bin/env python
# -*- coding: utf-8 -*-

import cv2
import os
import argparse
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description='视频抽帧脚本')
    parser.add_argument('video', help='输入视频路径')
    parser.add_argument('--output-dir', type=str, default='frames',
                        help='输出帧保存目录（默认：frames）')
    parser.add_argument('--interval', type=int, default=1,
                        help='抽帧间隔：每隔多少帧保存一张（默认：1，表示每帧都保存）')
    parser.add_argument('--start-frame', type=int, default=0,
                        help='从第几帧开始抽取（包含，默认：0）')
    parser.add_argument('--end-frame', type=int, default=-1,
                        help='到第几帧结束（包含，默认：-1，表示到视频结尾）')
    return parser.parse_args()


def main():
    args = parse_args()

    video_path = Path(args.video)
    if not video_path.exists():
        print(f'❌ 视频文件不存在: {video_path}')
        return

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f'❌ 无法打开视频文件: {video_path}')
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    print(f'视频信息:')
    print(f'  路径: {video_path}')
    print(f'  帧率: {fps:.2f} FPS')
    print(f'  总帧数: {total_frames}')

    start_frame = max(0, args.start_frame)
    end_frame = total_frames - 1 if args.end_frame < 0 else min(args.end_frame, total_frames - 1)
    interval = max(1, args.interval)

    print(f'抽帧设置:')
    print(f'  起始帧: {start_frame}')
    print(f'  结束帧: {end_frame}')
    print(f'  抽帧间隔: 每 {interval} 帧保存一张')
    print(f'  输出目录: {output_dir}')

    # 跳到起始帧
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    frame_idx = start_frame
    saved_count = 0

    while frame_idx <= end_frame:
        ret, frame = cap.read()
        if not ret:
            break

        # 判断是否需要保存当前帧
        if (frame_idx - start_frame) % interval == 0:
            # 文件名：videoName_frame_000001.jpg
            frame_name = f'{video_path.stem}_frame_{frame_idx:06d}.jpg'
            save_path = output_dir / frame_name
            cv2.imwrite(str(save_path), frame)
            saved_count += 1

            if saved_count % 50 == 0:
                print(f'  已保存 {saved_count} 帧（当前帧: {frame_idx}）')

        frame_idx += 1

    cap.release()
    print(f'\n✓ 抽帧完成！总共保存 {saved_count} 帧，保存在: {output_dir}')


if __name__ == '__main__':
    main()