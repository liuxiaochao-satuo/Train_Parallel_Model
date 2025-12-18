# 混合精度训练（AMP）命令指南

## 概述

混合精度训练（Automatic Mixed Precision, AMP）可以：
- **节省约50%显存**：使用FP16代替FP32，显存占用减半
- **加速训练**：在支持的GPU上（如RTX系列），FP16计算更快
- **保持精度**：自动管理精度，通常不会影响最终模型精度

## 使用方法

### 方法1：使用环境变量启用AMP（推荐）

```bash
# 运行所有实验（启用AMP）
USE_AMP=1 bash tools/run_ablation_experiments.sh

# 运行单个实验（启用AMP）
USE_AMP=1 bash tools/run_ablation_experiments.sh loss_weight

# 使用2张GPU + AMP
GPUS=2 USE_AMP=1 bash tools/run_ablation_experiments.sh
```

### 方法2：直接使用训练脚本（单GPU）

```bash
# 单GPU训练（启用AMP）
python tools/train.py \
    configs/body_2d_keypoint/dekr/coco/dekr_hrnet-w32_parallel_ablation_loss_weight.py \
    --work-dir /data/lxc/outputs/train_parallel_model/ablation_experiments/loss_weight_only \
    --amp \
    --cfg-options randomness.seed=42
```

### 方法3：使用分布式训练脚本（多GPU）

```bash
# 4张GPU训练（启用AMP）
bash tools/dist_train.sh \
    configs/body_2d_keypoint/dekr/coco/dekr_hrnet-w32_parallel_ablation_loss_weight.py \
    4 \
    --work-dir /data/lxc/outputs/train_parallel_model/ablation_experiments/loss_weight_only \
    --amp \
    --cfg-options randomness.seed=42
```

## 完整示例

### 示例1：运行所有消融实验（启用AMP）

```bash
cd /home/hello/lxc/Train_Parallel_Model

# 启用AMP运行所有实验
USE_AMP=1 bash tools/run_ablation_experiments.sh
```

### 示例2：仅运行损失权重实验（启用AMP）

```bash
cd /home/hello/lxc/Train_Parallel_Model

USE_AMP=1 bash tools/run_ablation_experiments.sh loss_weight
```

### 示例3：使用2张GPU + AMP（如果4张GPU显存仍不够）

```bash
cd /home/hello/lxc/Train_Parallel_Model

GPUS=2 USE_AMP=1 bash tools/run_ablation_experiments.sh loss_weight
```

## 显存优化对比

| 配置 | batch_size | AMP | 预估显存占用 | 有效batch_size |
|------|-----------|-----|------------|---------------|
| 原始配置 | 12 | 否 | ~14GB | 48 (4卡×12) |
| 优化配置1 | 4 | 否 | ~12GB | 16 (4卡×4) |
| 优化配置2 | 2 | 否 | ~11GB | 8 (4卡×2) |
| **优化配置3** | **2** | **是** | **~5.5GB** | **32 (4卡×2×4累积)** |

## 注意事项

1. **GPU要求**：AMP需要支持Tensor Core的GPU（如RTX系列、V100等）
2. **精度影响**：通常不会影响最终精度，但建议对比验证
3. **学习率**：使用AMP时，学习率可能需要微调（通常保持不变即可）
4. **梯度累积**：当前配置已启用梯度累积（`accumulative_counts=4`），与AMP兼容

## 验证AMP是否生效

训练开始时会看到类似输出：
```
[INFO] Using AmpOptimWrapper for mixed precision training
```

或者在日志中查找：
```bash
grep -i "amp\|mixed precision" /data/lxc/outputs/train_parallel_model/ablation_experiments/*/logs/*.log
```

## 故障排除

如果遇到问题：

1. **检查GPU是否支持AMP**：
   ```bash
   python -c "import torch; print(torch.cuda.get_device_capability())"
   # 输出应该是 (7, 5) 或更高（RTX系列）
   ```

2. **如果AMP导致训练不稳定**：
   - 可以尝试减小学习率
   - 或者不使用AMP，改用更小的batch_size

3. **显存仍然不足**：
   - 进一步减小batch_size到1
   - 减少GPU数量：`GPUS=2`
   - 减小输入尺寸：修改配置文件中的`input_size=(384, 384)`

