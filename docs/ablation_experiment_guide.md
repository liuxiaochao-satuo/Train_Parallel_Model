# 消融实验指南：对比不同group_id提升策略

## 实验环境准备

### 数据路径配置

**重要**：如果数据集和程序放在不同盘，需要修改配置文件中的数据路径：

1. **数据集路径**：配置文件中的 `data_root` 需要指向data盘的数据集位置
   - 默认配置：`data_root = '/data/coco_parallel'`
   - 请根据实际data盘挂载路径修改（例如：`/mnt/data/coco_parallel`）

2. **输出路径**：训练结果（模型、日志）会保存到 `work_dir` 指定的目录
   - 项目输出目录：`/data/lxc/outputs/train_parallel_model/`
   - 消融实验结果保存在：`/data/lxc/outputs/train_parallel_model/ablation_experiments/`
   - 例如：`--work-dir /data/lxc/outputs/train_parallel_model/ablation_experiments/combined`

### 多GPU训练

如果使用多张GPU（例如4张显卡），请使用分布式训练脚本：

```bash
# 使用4张GPU训练
bash tools/dist_train.sh \
    configs/body_2d_keypoint/dekr/coco/dekr_hrnet-w32_parallel_ablation_combined.py \
    4 \
    --work-dir /data/work_dirs/ablation_experiments/combined
```

**注意**：
- 4张GPU时，总batch size = 单卡batch_size × 4 = 12 × 4 = 48
- 确保data盘有足够的空间存储训练输出（checkpoints、日志等）

## 实验目标

对比三种策略对group_id=1（运动员）精度提升的效果：
1. **仅使用损失权重**（Loss Weighting Only）
2. **仅使用加权采样**（Weighted Sampling Only）
3. **两者组合**（Combined）

## 实验配置

### 配置文件

已创建三个配置文件：

1. `configs/body_2d_keypoint/dekr/coco/dekr_hrnet-w32_parallel_ablation_loss_weight.py`
   - 仅使用损失权重：`group_id_weight={1: 2.0}`
   - 使用默认的`DefaultSampler`
   - Pipeline中包含`ApplyGroupWeight`

2. `configs/body_2d_keypoint/dekr/coco/dekr_hrnet-w32_parallel_ablation_weighted_sampling.py`
   - 仅使用加权采样：`WeightedGroupSampler` with `group_id_weights={1: 2.0}`
   - 不使用损失权重：`group_id_weight={}`
   - Pipeline中不包含`ApplyGroupWeight`

3. `configs/body_2d_keypoint/dekr/coco/dekr_hrnet-w32_parallel_ablation_combined.py`
   - 组合使用：同时使用损失权重和加权采样
   - `group_id_weight={1: 2.0}` + `WeightedGroupSampler` with `group_id_weights={1: 2.0}`
   - Pipeline中包含`ApplyGroupWeight`

## 运行实验

### 方法1：使用脚本批量运行（推荐）

```bash
# 运行所有实验
bash tools/run_ablation_experiments.sh

# 或分别运行
bash tools/run_ablation_experiments.sh --exp loss_weight
bash tools/run_ablation_experiments.sh --exp weighted_sampling
bash tools/run_ablation_experiments.sh --exp combined
```

### 方法2：手动运行

#### 单GPU训练

```bash
# 实验1：仅损失权重
python tools/train.py \
    configs/body_2d_keypoint/dekr/coco/dekr_hrnet-w32_parallel_ablation_loss_weight.py \
    --work-dir /data/lxc/outputs/train_parallel_model/ablation_experiments/loss_weight_only

# 实验2：仅加权采样
python tools/train.py \
    configs/body_2d_keypoint/dekr/coco/dekr_hrnet-w32_parallel_ablation_weighted_sampling.py \
    --work-dir /data/lxc/outputs/train_parallel_model/ablation_experiments/weighted_sampling_only

# 实验3：组合使用
python tools/train.py \
    configs/body_2d_keypoint/dekr/coco/dekr_hrnet-w32_parallel_ablation_combined.py \
    --work-dir /data/lxc/outputs/train_parallel_model/ablation_experiments/combined
```

#### 多GPU训练（推荐，4张显卡）

```bash
# 实验1：仅损失权重（4张GPU）
bash tools/dist_train.sh \
    configs/body_2d_keypoint/dekr/coco/dekr_hrnet-w32_parallel_ablation_loss_weight.py \
    4 \
    --work-dir /data/lxc/outputs/train_parallel_model/ablation_experiments/loss_weight_only

# 实验2：仅加权采样（4张GPU）
bash tools/dist_train.sh \
    configs/body_2d_keypoint/dekr/coco/dekr_hrnet-w32_parallel_ablation_weighted_sampling.py \
    4 \
    --work-dir /data/lxc/outputs/train_parallel_model/ablation_experiments/weighted_sampling_only

# 实验3：组合使用（4张GPU）
bash tools/dist_train.sh \
    configs/body_2d_keypoint/dekr/coco/dekr_hrnet-w32_parallel_ablation_combined.py \
    4 \
    --work-dir /data/lxc/outputs/train_parallel_model/ablation_experiments/combined
```

## 实验结果对比

### 关键指标

训练完成后，对比以下指标：

1. **整体性能**
   - AP (Average Precision)
   - AP50, AP75
   - AR (Average Recall)

2. **group_id=1的性能**（重点）
   - group_id=1的AP
   - group_id=1的AP50, AP75
   - group_id=1的AR

3. **训练过程**
   - 损失下降曲线
   - 验证集精度曲线
   - 训练时间

### 结果分析脚本

创建结果分析脚本：

```python
# tools/analyze_ablation_results.py
import json
from pathlib import Path

def analyze_experiment(work_dir):
    """分析单个实验的结果"""
    log_file = Path(work_dir) / "vis_data" / "scalars.json"
    if not log_file.exists():
        print(f"未找到日志文件: {log_file}")
        return None
    
    with open(log_file) as f:
        data = json.load(f)
    
    # 提取关键指标
    metrics = {}
    for key, values in data.items():
        if 'val/coco/AP' in key:
            metrics[key] = max(values) if values else 0
    
    return metrics

# 对比三个实验
experiments = {
    'loss_weight': 'work_dirs/ablation_experiments/loss_weight_only',
    'weighted_sampling': 'work_dirs/ablation_experiments/weighted_sampling_only',
    'combined': 'work_dirs/ablation_experiments/combined',
}

print("=" * 60)
print("消融实验结果对比")
print("=" * 60)

for name, path in experiments.items():
    print(f"\n{name}:")
    metrics = analyze_experiment(path)
    if metrics:
        for key, value in sorted(metrics.items()):
            print(f"  {key}: {value:.4f}")
```

## 预期结果

根据理论分析：

1. **仅损失权重**
   - ✅ 实现简单
   - ✅ 对训练速度影响小
   - ⚠️ 可能过拟合特定group_id

2. **仅加权采样**
   - ✅ 更自然的平衡方式
   - ✅ 不会直接改变损失函数
   - ⚠️ 可能增加训练时间

3. **两者组合**
   - ✅ 理论上效果最好
   - ✅ 既增加采样频率又增加损失权重
   - ⚠️ 可能过度优化特定group_id

## 实验记录模板

建议记录以下信息：

```markdown
## 实验记录

### 实验设置
- 数据集：coco_parallel (annotations_id)
- group_id=1权重：2.0
- 训练epochs：140
- Batch size：12
- 随机种子：42

### 实验结果

| 策略 | AP | AP50 | AP75 | AR | group_id=1 AP | 训练时间 |
|------|----|----|----|----|--------------|---------|
| Baseline | - | - | - | - | - | - |
| 仅损失权重 | - | - | - | - | - | - |
| 仅加权采样 | - | - | - | - | - | - |
| 两者组合 | - | - | - | - | - | - |

### 结论
[记录实验结论和观察]
```

## 注意事项

1. **使用相同的随机种子**：确保实验可复现
2. **使用相同的训练设置**：epochs、batch size等保持一致
3. **记录完整的训练日志**：便于后续分析
4. **多次运行取平均**：如果资源允许，建议运行3次取平均
5. **监控训练过程**：使用tensorboard实时监控

## 可视化结果

使用tensorboard对比训练过程：

```bash
tensorboard --logdir /data/lxc/outputs/train_parallel_model/ablation_experiments
```

在tensorboard中可以对比：
- 训练损失曲线
- 验证集精度曲线
- 不同group_id的性能

## 后续分析

实验完成后，可以：

1. **统计分析**：使用统计方法验证差异显著性
2. **可视化对比**：绘制对比图表
3. **错误分析**：分析不同策略下的错误模式
4. **参数调优**：基于最佳策略进一步调优权重

