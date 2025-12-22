#!/bin/bash
# 挂载远程目录到本地外置硬盘的脚本
# 在本地Ubuntu电脑上运行此脚本

# ===== 配置区域 =====
REMOTE_USER="hello"
REMOTE_HOST="192.168.2.102"
REMOTE_PATH="/data"
LOCAL_MOUNT="/media/satuo/E05EEE285EEDF6E6/remote_data"

# ===== 脚本主体 =====
echo "=========================================="
echo "挂载远程目录到本地..."
echo "远程: $REMOTE_USER@$REMOTE_HOST:$REMOTE_PATH"
echo "本地挂载点: $LOCAL_MOUNT"
echo "=========================================="

# 检查是否已安装 sshfs
if ! command -v sshfs &> /dev/null; then
    echo "⚠️  sshfs 未安装，正在安装..."
    sudo apt-get update
    sudo apt-get install -y sshfs
fi

# 创建挂载点
mkdir -p "$LOCAL_MOUNT"

# 检查是否已挂载
if mountpoint -q "$LOCAL_MOUNT"; then
    echo "⚠️  远程目录已挂载"
    echo "当前挂载内容:"
    ls -lh "$LOCAL_MOUNT" | head -10
    echo ""
    read -p "是否要卸载后重新挂载？(y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        umount "$LOCAL_MOUNT"
        echo "正在重新挂载..."
        sshfs "$REMOTE_USER@$REMOTE_HOST:$REMOTE_PATH" "$LOCAL_MOUNT"
    fi
else
    echo "正在挂载远程目录..."
    sshfs "$REMOTE_USER@$REMOTE_HOST:$REMOTE_PATH" "$LOCAL_MOUNT"
    
    if [ $? -eq 0 ]; then
        echo ""
        echo "✓ 挂载成功！"
        echo "访问路径: $LOCAL_MOUNT"
        echo ""
        echo "常用路径:"
        echo "  推理结果: $LOCAL_MOUNT/lxc/outputs/train_parallel_model/test_video/"
        echo "  测试视频: $LOCAL_MOUNT/lxc/datasets/coco_paralel/test_video/"
        echo ""
        echo "卸载命令: umount $LOCAL_MOUNT"
    else
        echo ""
        echo "✗ 挂载失败，请检查："
        echo "  1. SSH连接是否正常: ssh $REMOTE_USER@$REMOTE_HOST"
        echo "  2. 远程路径是否存在: $REMOTE_PATH"
        echo "  3. 本地挂载点是否有写权限"
        exit 1
    fi
fi

