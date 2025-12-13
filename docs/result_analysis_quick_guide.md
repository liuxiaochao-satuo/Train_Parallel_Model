# 实验结果分析快速指南

## 训练过程中产生的指标

### 自动记录的指标（保存在 `work_dirs/.../vis_data/scalars.json`）

**验证集指标（每个val_interval记录一次）：**
- `val/coco/AP` - **最重要**：平均精度（基于OKS）
- `val/coco/AP50` - OKS阈值0.5时的AP
- `val/coco/AP75` - OKS阈值0.75时的AP
- `val/coco/AR` - 平均召回率

**训练指标：**
- `train/loss` - 总训练损失
- `train/loss/heatmap` - 热图损失
- `train/loss/displacement` - 位移损失

### 预测结果文件（用于详细分析）

验证完成后，预测结果保存在：
- `work_dirs/.../predictions/results.keypoints.json`

**注意**：需要在配置文件中设置 `outfile_prefix` 才会保存预测结果。

## 如何分析group_id=1（运动员）的精度变化

### 快速分析（推荐）

```bash
# 分析所有实验的group_id性能
python tools/evaluate_by_group_id.py \
    --ann-file data/coco_parallel/annotations_id/person_keypoints_val_parallel.json \
    --work-dirs work_dirs/ablation_experiments/loss_weight_only \
                work_dirs/ablation_experiments/weighted_sampling_only \
                work_dirs/ablation_experiments/combined \
    --experiment-names "Loss Weight" "Weighted Sampling" "Combined" \
    --group-ids 1 \
    --output results/athlete_analysis.json
```

**输出会显示：**
- group_id=1（运动员）的AP、AP50、AP75、AR
- 相对于baseline的改进幅度
- 改进百分比

### 查看整体性能对比

```bash
python tools/analyze_ablation_results.py \
    --base-dir work_dirs/ablation_experiments \
    --experiments loss_weight_only weighted_sampling_only combined \
    --names "Loss Weight" "Weighted Sampling" "Combined"
```

## 结果解读示例

### 理想结果

```
整体AP：
  Loss Weight:     0.7123
  Combined:        0.7189 (+0.66%)  ✅ 整体提升

group_id=1（运动员）AP：
  Loss Weight:     0.7034
  Combined:        0.7356 (+4.58%)  ✅ 显著提升
```

**结论**：策略有效，既提升了整体性能，又显著提升了运动员识别精度。

### 需要调整的情况

```
整体AP：
  Loss Weight:     0.7123
  Combined:        0.7089 (-0.34%)  ⚠️ 整体略有下降

group_id=1（运动员）AP：
  Loss Weight:     0.7034
  Combined:        0.7456 (+6.00%)  ✅ 显著提升
```

**结论**：可能过度优化了group_id=1，建议降低权重或调整策略。

## 可视化

使用TensorBoard查看训练曲线：

```bash
tensorboard --logdir work_dirs/ablation_experiments
```

对比：
- `val/coco/AP` 曲线（整体性能）
- `train/loss` 曲线（训练稳定性）

## 完整分析流程

1. **训练完成后**，检查是否有预测结果文件
2. **运行group_id分析**，查看运动员精度变化
3. **运行整体分析**，查看整体性能
4. **综合对比**，做出决策

详细说明请参考：`docs/group_id_performance_analysis.md`

