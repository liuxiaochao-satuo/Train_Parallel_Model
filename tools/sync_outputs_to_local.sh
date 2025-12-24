#!/bin/bash
# 同步训练和推理结果到本地的脚本
# 使用方法: 
#   1. 在本地电脑执行: bash sync_outputs_to_local.sh
#   2. 或者: ./sync_outputs_to_local.sh [远程主机] [远程用户] [本地保存路径]

# 配置参数（请根据实际情况修改）
REMOTE_HOST="${1:-your_remote_host}"  # 远程主机地址或IP
REMOTE_USER="${2:-hello}"  # 远程用户名
REMOTE_PATH="/data/lxc/outputs/train_parallel_model"
LOCAL_PATH="${3:-$HOME/Downloads/train_parallel_model_outputs}"

echo "=========================================="
echo "训练结果同步脚本"
echo "=========================================="
echo "远程主机: ${REMOTE_USER}@${REMOTE_HOST}"
echo "远程路径: ${REMOTE_PATH}"
echo "本地路径: ${LOCAL_PATH}"
echo "=========================================="

# 创建本地目录（包括所有父目录）
mkdir -p "${LOCAL_PATH}"
echo "已创建本地目录: ${LOCAL_PATH}"

# 使用 rsync 同步（推荐：支持增量同步，只传输更改的文件）
echo "开始同步..."
rsync -avz --progress \
  --exclude='*.pth' \
  --exclude='*.ckpt' \
  --exclude='*.pth.tar' \
  --exclude='__pycache__' \
  --exclude='*.pyc' \
  --include='test_video/' \
  --include='test_video/**' \
  --include='*.mp4' \
  --include='*.avi' \
  --include='*.jpg' \
  --include='*.png' \
  --include='*.json' \
  --include='*.txt' \
  --include='*.log' \
  --include='*.csv' \
  --exclude='*' \
  ${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_PATH}/ ${LOCAL_PATH}/

if [ $? -eq 0 ]; then
    echo "=========================================="
    echo "✓ 同步完成！"
    echo "文件保存在: ${LOCAL_PATH}"
    echo "=========================================="
else
    echo "=========================================="
    echo "✗ 同步失败，请检查网络连接和权限"
    echo "=========================================="
    exit 1
fi

