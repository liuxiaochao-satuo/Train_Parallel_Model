# 按group_id分析实验结果指南

## 概述

本指南说明如何分析消融实验结果，特别关注**group_id=1（运动员）**的识别精度变化。

## 训练过程中产生的指标

### 1. 整体性能指标（CocoMetric自动计算）

训练和验证过程中，`CocoMetric` 会自动计算并记录以下指标：

**验证集指标（每个val_interval记录一次）：**
- `val/coco/AP` - 平均精度（主指标，基于OKS）
- `val/coco/AP50` - OKS阈值0.5时的AP
- `val/coco/AP75` - OKS阈值0.75时的AP
- `val/coco/APm` - 中等尺寸目标的AP
- `val/coco/APl` - 大尺寸目标的AP
- `val/coco/AR` - 平均召回率
- `val/coco/AR50`, `AR75`, `ARm`, `ARl` - 对应的召回率指标

**训练指标：**
- `train/loss` - 训练损失
- `train/loss/heatmap` - 热图损失
- `train/loss/displacement` - 位移损失
- `lr` - 学习率

这些指标保存在：
- `work_dirs/.../vis_data/scalars.json` - JSON格式
- `work_dirs/.../vis_data/tf_event` - TensorBoard格式

### 2. 预测结果文件

验证或测试完成后，`CocoMetric` 会保存预测结果到：
- `work_dirs/.../predictions/results.keypoints.json` - COCO格式的预测结果

这个文件包含所有图像的预测关键点，用于后续的详细分析。

## 如何分析group_id=1（运动员）的精度变化

### 方法1：使用分析脚本（推荐）

运行按group_id分析脚本：

```bash
# 从work_dir自动查找预测结果
python tools/evaluate_by_group_id.py \
    --ann-file data/coco_parallel/annotations_id/person_keypoints_val_parallel.json \
    --work-dirs work_dirs/ablation_experiments/loss_weight_only \
                work_dirs/ablation_experiments/weighted_sampling_only \
                work_dirs/ablation_experiments/combined \
    --experiment-names "Loss Weight Only" "Weighted Sampling Only" "Combined" \
    --group-ids 1 2 3 4 \
    --output results/group_id_analysis.json
```

**输出示例：**
```
================================================================================
🎯 重点分析：group_id=1（运动员）
================================================================================

实验名称                        AP        AP50       AP75         AR
--------------------------------------------------------------------------------
Loss Weight Only            0.7234    0.8567    0.7891    0.8123
Weighted Sampling Only      0.7312    0.8612    0.7956    0.8198
Combined                    0.7456    0.8723    0.8089    0.8345

相对于第一个实验的改进（group_id=1，运动员）:
  Weighted Sampling Only     AP: 0.7312 (↑0.0078, ↑1.08%)
  Combined                   AP: 0.7456 (↑0.0222, ↑3.07%)
```

### 方法2：手动分析预测结果

如果预测文件已生成，可以手动分析：

```python
import json
from xtcocotools.coco import COCO
from xtcocotools.cocoeval import COCOeval

# 加载标注和预测
coco_gt = COCO('data/coco_parallel/annotations_id/person_keypoints_val_parallel.json')
coco_dt = coco_gt.loadRes('work_dirs/exp/predictions/results.keypoints.json')

# 获取group_id=1的annotation IDs
group_1_ann_ids = [
    ann_id for ann_id, ann in coco_gt.anns.items()
    if ann.get('group_id') == 1
]

# 创建过滤后的评估（需要自定义实现）
# ... 使用evaluate_by_group_id.py脚本更方便
```

## 完整分析流程

### 步骤1：训练完成后，确保有预测结果

如果验证时没有生成预测文件，可以运行测试：

```bash
python tools/test.py \
    configs/body_2d_keypoint/dekr/coco/dekr_hrnet-w32_parallel_ablation_loss_weight.py \
    work_dirs/ablation_experiments/loss_weight_only/checkpoints/best.pth \
    --out work_dirs/ablation_experiments/loss_weight_only/predictions/results.json
```

### 步骤2：运行group_id分析

```bash
python tools/evaluate_by_group_id.py \
    --ann-file data/coco_parallel/annotations_id/person_keypoints_val_parallel.json \
    --work-dirs work_dirs/ablation_experiments/loss_weight_only \
                work_dirs/ablation_experiments/weighted_sampling_only \
                work_dirs/ablation_experiments/combined \
    --experiment-names "Loss Weight" "Weighted Sampling" "Combined" \
    --group-ids 1 \
    --output results/athlete_analysis.json
```

### 步骤3：查看整体性能对比

使用之前的分析脚本查看整体性能：

```bash
python tools/analyze_ablation_results.py \
    --base-dir work_dirs/ablation_experiments \
    --experiments loss_weight_only weighted_sampling_only combined \
    --names "Loss Weight" "Weighted Sampling" "Combined"
```

### 步骤4：综合对比

对比两个结果：
- **整体AP**：所有样本的平均性能
- **group_id=1的AP**：运动员样本的性能

理想情况下：
- 整体AP应该保持或提升
- group_id=1的AP应该有明显提升

## 结果解读

### 成功案例

```
整体AP对比：
  Loss Weight:     0.7123
  Weighted Sampling: 0.7145 (+0.22%)
  Combined:        0.7189 (+0.66%)

group_id=1（运动员）AP对比：
  Loss Weight:     0.7034
  Weighted Sampling: 0.7212 (+2.53%)  ← 明显提升
  Combined:        0.7356 (+4.58%)  ← 显著提升
```

**解读：**
- ✅ 整体性能略有提升
- ✅ group_id=1的性能显著提升（+4.58%）
- ✅ 策略有效

### 需要注意的情况

```
整体AP对比：
  Loss Weight:     0.7123
  Combined:        0.7089 (-0.34%)  ← 整体略有下降

group_id=1（运动员）AP对比：
  Loss Weight:     0.7034
  Combined:        0.7456 (+6.00%)  ← 显著提升
```

**解读：**
- ⚠️ 整体性能略有下降（可能过度优化group_id=1）
- ✅ group_id=1性能显著提升
- 💡 建议：降低权重（如从2.0降到1.5）或调整策略

## 可视化结果

### 使用TensorBoard

```bash
tensorboard --logdir work_dirs/ablation_experiments
```

在TensorBoard中对比：
- `val/coco/AP` 曲线（整体性能）
- 训练损失曲线（观察是否过拟合）

### 创建对比图表

可以编写脚本从 `results/group_id_analysis.json` 生成对比图表：

```python
import json
import matplotlib.pyplot as plt

with open('results/group_id_analysis.json') as f:
    results = json.load(f)

# 提取group_id=1的AP
experiments = list(results.keys())
ap_values = [results[exp][1]['AP'] for exp in experiments]

plt.bar(experiments, ap_values)
plt.title('Group ID=1 (Athletes) AP Comparison')
plt.ylabel('AP')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('results/athlete_ap_comparison.png')
```

## 关键指标说明

### AP (Average Precision)
- **最重要**：综合评估指标，基于OKS计算
- **范围**：0-1，越高越好
- **含义**：在所有OKS阈值下的平均精度

### AP50 / AP75
- **含义**：OKS阈值分别为0.5和0.75时的精度
- **用途**：评估不同严格程度下的性能

### AR (Average Recall)
- **含义**：平均召回率
- **用途**：评估模型能找到多少正确的关键点

## 常见问题

### Q: 预测文件在哪里？

A: 通常在以下位置：
- `work_dirs/.../predictions/results.keypoints.json`
- `work_dirs/.../*.keypoints.json`

如果找不到，需要先运行测试生成预测结果。

### Q: 如何确保预测文件包含所有验证样本？

A: 运行完整的验证或测试：
```bash
python tools/test.py config.py checkpoint.pth
```

### Q: group_id=1的AP提升了，但整体AP下降了怎么办？

A: 可能过度优化了group_id=1，建议：
1. 降低权重（如从2.0降到1.5）
2. 使用课程学习（逐渐增加权重）
3. 检查是否有过拟合

## 总结

通过按group_id分析，您可以：
1. ✅ 明确知道group_id=1（运动员）的精度变化
2. ✅ 对比不同策略对运动员识别的效果
3. ✅ 平衡整体性能和特定group_id的性能
4. ✅ 做出数据驱动的决策

