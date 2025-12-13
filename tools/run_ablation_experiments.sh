#!/bin/bash
# ============================================================================
# 消融实验脚本：对比损失权重、加权采样和两者组合的效果
# ============================================================================
#
# 使用方法：
#   bash tools/run_ablation_experiments.sh
#
# 或者分别运行：
#   bash tools/run_ablation_experiments.sh --exp loss_weight
#   bash tools/run_ablation_experiments.sh --exp weighted_sampling
#   bash tools/run_ablation_experiments.sh --exp combined
# ============================================================================

set -e

# 默认运行所有实验
EXPERIMENT=${1:-all}

# 基础配置
BASE_DIR="work_dirs/ablation_experiments"
CONFIG_DIR="configs/body_2d_keypoint/dekr/coco"

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
    python tools/train.py \
        ${CONFIG_DIR}/dekr_hrnet-w32_parallel_ablation_loss_weight.py \
        --work-dir ${BASE_DIR}/loss_weight_only \
        --seed 42
    
    echo "实验1完成！"
    echo ""
fi

# 实验2：仅使用加权采样
if [ "$EXPERIMENT" == "all" ] || [ "$EXPERIMENT" == "weighted_sampling" ]; then
    echo "----------------------------------------"
    echo "实验2: 仅使用加权采样 (Weighted Sampling Only)"
    echo "----------------------------------------"
    python tools/train.py \
        ${CONFIG_DIR}/dekr_hrnet-w32_parallel_ablation_weighted_sampling.py \
        --work-dir ${BASE_DIR}/weighted_sampling_only \
        --seed 42
    
    echo "实验2完成！"
    echo ""
fi

# 实验3：组合使用
if [ "$EXPERIMENT" == "all" ] || [ "$EXPERIMENT" == "combined" ]; then
    echo "----------------------------------------"
    echo "实验3: 损失权重 + 加权采样组合 (Combined)"
    echo "----------------------------------------"
    python tools/train.py \
        ${CONFIG_DIR}/dekr_hrnet-w32_parallel_ablation_combined.py \
        --work-dir ${BASE_DIR}/combined \
        --seed 42
    
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

