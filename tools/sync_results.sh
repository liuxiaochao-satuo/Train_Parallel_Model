#!/bin/bash
# 同步远程推理结果到本地的脚本
# 使用方法：在本地Ubuntu电脑上运行此脚本

# ===== 配置区域 =====
# 修改这些变量为你的实际值
REMOTE_USER="hello"
REMOTE_HOST="192.168.2.102"
REMOTE_PATH="/data/lxc/outputs/train_parallel_model/test_video"
LOCAL_PATH="$HOME/Downloads/inference_results"

# ===== 脚本主体 =====
echo "=========================================="
echo "开始同步推理结果..."
echo "远程路径: $REMOTE_USER@$REMOTE_HOST:$REMOTE_PATH"
echo "本地路径: $LOCAL_PATH"
echo "=========================================="

# 创建本地目录
mkdir -p "$LOCAL_PATH"

# 使用 rsync 同步
rsync -avz --progress \
    "$REMOTE_USER@$REMOTE_HOST:$REMOTE_PATH/" \
    "$LOCAL_PATH/"

if [ $? -eq 0 ]; then
    echo ""
    echo "✓ 同步完成！"
    echo "结果保存在: $LOCAL_PATH"
else
    echo ""
    echo "✗ 同步失败，请检查网络连接和路径"
    exit 1
fi

