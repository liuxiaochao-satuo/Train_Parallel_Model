#!/bin/bash
# ============================================================================
# 消融实验脚本：对比损失权重、加权采样和两者组合的效果
# ============================================================================
#
# 使用方法：
#   bash tools/run_ablation_experiments.sh                    # 运行所有实验（4张GPU）
#   bash tools/run_ablation_experiments.sh loss_weight        # 仅运行实验1
#   bash tools/run_ablation_experiments.sh weighted_sampling   # 仅运行实验2
#   bash tools/run_ablation_experiments.sh combined           # 仅运行实验3
#   GPUS=2 bash tools/run_ablation_experiments.sh             # 使用2张GPU运行
#   USE_AMP=1 bash tools/run_ablation_experiments.sh          # 启用混合精度训练（AMP）
# ============================================================================

set -e

# 激活 conda 环境（如果未激活）
if [ -z "$CONDA_DEFAULT_ENV" ] || [ "$CONDA_DEFAULT_ENV" != "openmmlab" ]; then
    # 初始化 conda（如果尚未初始化）
    if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
        source "$HOME/miniconda3/etc/profile.d/conda.sh"
    elif [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
        source "$HOME/anaconda3/etc/profile.d/conda.sh"
    fi
    # 激活 openmmlab 环境
    conda activate openmmlab || {
        echo "错误: 无法激活 openmmlab 环境，请确保已安装该环境"
        exit 1
    }
fi

# 默认运行所有实验
EXPERIMENT=${1:-all}

# 基础配置
# 输出目录：/data/lxc/outputs/train_parallel_model/ablation_experiments
BASE_DIR="/data/lxc/outputs/train_parallel_model/ablation_experiments"
CONFIG_DIR="configs/body_2d_keypoint/dekr/coco"

# GPU配置（默认使用4张GPU）
GPUS=${GPUS:-4}

# AMP配置（混合精度训练，可节省约50%显存）
USE_AMP=${USE_AMP:-0}

# 设置PyTorch CUDA内存分配配置，减少内存碎片
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# 构建AMP参数
AMP_ARG=""
if [ "$USE_AMP" == "1" ]; then
    AMP_ARG="--amp"
    echo "启用混合精度训练 (AMP) - 可节省约50%显存"
fi

# 创建结果目录
mkdir -p ${BASE_DIR}

echo "=========================================="
echo "开始消融实验"
echo "=========================================="
echo ""

# 实验1：仅使用损失权重
if [ "$EXPERIMENT" == "all" ] || [ "$EXPERIMENT" == "loss_weight" ]; then
    echo "----------------------------------------"
    echo "实验1: 仅使用损失权重 (Loss Weighting Only)"
    echo "----------------------------------------"
    bash tools/dist_train.sh \
        ${CONFIG_DIR}/dekr_hrnet-w32_parallel_ablation_loss_weight.py \
        ${GPUS} \
        --work-dir ${BASE_DIR}/loss_weight_only \
        --cfg-options randomness.seed=42 \
        ${AMP_ARG}
    
    echo "实验1完成！"
    echo ""
fi

# 实验2：仅使用加权采样
if [ "$EXPERIMENT" == "all" ] || [ "$EXPERIMENT" == "weighted_sampling" ]; then
    echo "----------------------------------------"
    echo "实验2: 仅使用加权采样 (Weighted Sampling Only)"
    echo "----------------------------------------"
    bash tools/dist_train.sh \
        ${CONFIG_DIR}/dekr_hrnet-w32_parallel_ablation_weighted_sampling.py \
        ${GPUS} \
        --work-dir ${BASE_DIR}/weighted_sampling_only \
        --cfg-options randomness.seed=42 \
        ${AMP_ARG}
    
    echo "实验2完成！"
    echo ""
fi

# 实验3：组合使用
if [ "$EXPERIMENT" == "all" ] || [ "$EXPERIMENT" == "combined" ]; then
    echo "----------------------------------------"
    echo "实验3: 损失权重 + 加权采样组合 (Combined)"
    echo "----------------------------------------"
    bash tools/dist_train.sh \
        ${CONFIG_DIR}/dekr_hrnet-w32_parallel_ablation_combined.py \
        ${GPUS} \
        --work-dir ${BASE_DIR}/combined \
        --cfg-options randomness.seed=42 \
        ${AMP_ARG}
    
    echo "实验3完成！"
    echo ""
fi

echo "=========================================="
echo "所有实验完成！"
echo "=========================================="
echo ""
echo "结果保存在: ${BASE_DIR}/"
echo ""
echo "实验结果对比："
echo "  1. loss_weight_only/     - 仅损失权重"
echo "  2. weighted_sampling_only/ - 仅加权采样"
echo "  3. combined/             - 两者组合"
echo ""
echo "可以使用以下命令查看训练日志："
echo "  tensorboard --logdir ${BASE_DIR}"

