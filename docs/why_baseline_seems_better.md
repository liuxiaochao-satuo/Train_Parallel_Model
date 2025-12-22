# 为什么Baseline的整体AP看起来更好？

## 现象观察

从实验结果来看：

| 实验策略 | 整体AP | 整体AR |
|---------|--------|--------|
| **Baseline** | **0.9827** | **0.9893** |
| Loss Weight Only | 0.9794 | 0.9867 |
| Weighted Sampling Only | 0.9797 | 0.9855 |
| Combined | 0.9826 | 0.9896 |

确实，**Baseline的整体AP最高**，但这**并不意味着策略失败**！

## 关键理解

### 1. 策略的目标不是提升整体性能

这些消融实验策略（损失权重、加权采样、组合策略）的**核心目标是提升group_id=1（运动员）的识别精度**，而不是提升整体AP。

### 2. 整体AP vs 特定group_id的AP

- **整体AP**：所有样本（group_id=1,2,3,4）的平均性能
- **group_id=1的AP**：仅针对运动员样本的性能

### 3. 为什么会出现整体AP下降？

当使用权重策略（如`group_id_weight={1: 2.0}`）时：

1. **模型会更多地关注group_id=1的样本**
   - 增加group_id=1的损失权重 → 模型更努力优化运动员样本
   - 增加group_id=1的采样频率 → 模型看到更多运动员样本

2. **可能导致对其他group_id的优化不足**
   - 如果group_id=1的样本占比较小，过度优化可能导致：
     - group_id=1的AP显著提升 ✅
     - 其他group_id的AP略有下降 ⚠️
     - 整体AP略有下降（但通常很小）

3. **这是正常的权衡**
   - 如果目标是提升运动员识别精度，这是可以接受的权衡
   - 关键是要看**group_id=1的AP是否显著提升**

## 如何正确评估策略效果？

### 必须查看按group_id分析的结果

运行以下命令查看group_id=1（运动员）的性能：

```bash
python tools/evaluate_by_group_id.py \
    --ann-file data/coco_parallel/annotations_id/person_keypoints_val_parallel.json \
    --work-dirs /data/lxc/outputs/train_parallel_model/ablation_experiments/baseline \
                /data/lxc/outputs/train_parallel_model/ablation_experiments/loss_weight_only \
                /data/lxc/outputs/train_parallel_model/ablation_experiments/weighted_sampling_only \
                /data/lxc/outputs/train_parallel_model/ablation_experiments/combined \
    --experiment-names "Baseline" "Loss Weight Only" "Weighted Sampling Only" "Combined" \
    --group-ids 1
```

### 理想的成功案例

```
整体AP对比：
  Baseline:           0.9827
  Loss Weight Only:   0.9794 (-0.33%)  ← 整体略有下降
  Combined:          0.9826 (-0.01%)  ← 几乎持平

group_id=1（运动员）AP对比：
  Baseline:           0.9700
  Loss Weight Only:   0.9780 (+0.80%, +0.82%)  ← 显著提升！
  Combined:          0.9810 (+1.10%, +1.13%)  ← 显著提升！
```

**解读：**
- ✅ group_id=1的AP显著提升（+1.13%）
- ⚠️ 整体AP略有下降（-0.01%，几乎可忽略）
- ✅ **策略有效！** 成功提升了运动员识别精度

### 需要注意的情况

如果出现以下情况，可能需要调整策略：

```
整体AP对比：
  Baseline:           0.9827
  Combined:          0.9750 (-0.77%)  ← 整体下降较多

group_id=1（运动员）AP对比：
  Baseline:           0.9700
  Combined:          0.9750 (+0.50%, +0.52%)  ← 提升有限
```

**解读：**
- ⚠️ 整体AP下降较多（-0.77%）
- ⚠️ group_id=1的AP提升有限（+0.52%）
- 💡 **建议：**
  - 降低权重（如从2.0降到1.5）
  - 调整采样策略
  - 检查数据分布是否平衡

## 当前实验结果的可能解释

### 情况1：策略确实有效（最可能）

- 整体AP下降很小（0.0001-0.0033）
- group_id=1的AP可能有显著提升
- **需要运行group_id分析来确认**

### 情况2：权重设置不合适

- 如果group_id=1的AP也没有明显提升
- 可能需要：
  - 调整权重（如从2.0降到1.5或升到2.5）
  - 尝试不同的采样策略
  - 检查数据分布

### 情况3：数据分布问题

- 如果group_id=1的样本占比很小
- 即使增加权重，效果也可能有限
- 需要检查数据集中各group_id的分布

## 建议的下一步

1. **运行group_id分析**（最重要）
   ```bash
   python tools/evaluate_by_group_id.py ...
   ```

2. **查看训练曲线**
   - 检查group_id=1的AP是否在训练过程中提升
   - 使用TensorBoard查看详细曲线

3. **分析数据分布**
   - 检查各group_id的样本数量
   - 确认group_id=1是否占比较小

4. **如果group_id=1提升不明显**
   - 尝试调整权重（1.5, 2.5, 3.0）
   - 尝试不同的采样策略
   - 考虑使用更复杂的损失函数

## 总结

**整体AP略低于baseline是正常的**，只要：
- ✅ group_id=1（运动员）的AP有显著提升
- ✅ 整体AP下降在可接受范围内（<1%）
- ✅ 策略达到了预期目标

**关键是要看group_id=1的性能，而不是只看整体AP！**

