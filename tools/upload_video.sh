#!/bin/bash
# 上传视频文件到远程服务器的脚本
# 使用方法：在本地Ubuntu电脑上运行此脚本

# ===== 配置区域 =====
REMOTE_USER="hello"
REMOTE_HOST="192.168.2.102"
REMOTE_PATH="/data/lxc/datasets/coco_paralel/test_video"

# ===== 脚本主体 =====
if [ $# -eq 0 ]; then
    echo "使用方法: $0 <视频文件路径>"
    echo "示例: $0 ~/Videos/my_video.mp4"
    exit 1
fi

VIDEO_FILE="$1"

if [ ! -f "$VIDEO_FILE" ]; then
    echo "错误: 文件不存在: $VIDEO_FILE"
    exit 1
fi

echo "=========================================="
echo "上传视频文件到远程服务器..."
echo "文件: $VIDEO_FILE"
echo "目标: $REMOTE_USER@$REMOTE_HOST:$REMOTE_PATH"
echo "=========================================="

# 使用 scp 上传
scp "$VIDEO_FILE" "$REMOTE_USER@$REMOTE_HOST:$REMOTE_PATH/"

if [ $? -eq 0 ]; then
    echo ""
    echo "✓ 上传完成！"
    FILENAME=$(basename "$VIDEO_FILE")
    echo "远程路径: $REMOTE_PATH/$FILENAME"
else
    echo ""
    echo "✗ 上传失败，请检查网络连接"
    exit 1
fi

