# 使用group_id进行样本权重训练

## 概述

本功能允许您针对特定`group_id`的样本（如运动员）进行精度提升，通过在训练时对这些样本应用更大的损失权重来实现。

## 使用步骤

### 1. 生成group_id映射文件

由于COCO标注文件中不包含`group_id`信息，需要从原始labelme标注文件中恢复。运行以下命令生成映射文件：

```bash
python tools/generate_group_id_mapping.py \
    --coco-ann-file data/coco_parallel/annotations/person_keypoints_train_parallel.json \
    --labelme-dir /path/to/labelme/annotations \
    --output data/coco_parallel/annotations/person_keypoints_train_parallel_group_id_mapping.json
```

**参数说明：**
- `--coco-ann-file`: COCO格式的标注文件路径
- `--labelme-dir`: 原始labelme标注文件目录（可选，如果COCO文件中没有group_id）
- `--output`: 输出的映射文件路径（可选，默认与COCO文件同目录）

**映射文件格式：**
```json
{
  "0": 1,
  "1": 1,
  "2": 2,
  ...
}
```
其中key是annotation ID，value是group_id。

### 2. 配置数据集

在训练配置文件中，为`CocoParallelDataset`添加以下参数：

```python
dataset=dict(
    type='CocoParallelDataset',
    data_root='data/coco_parallel',
    data_mode='bottomup',
    ann_file='annotations/person_keypoints_train_parallel.json',
    data_prefix=dict(img='images/'),
    # 指定group_id映射文件
    group_id_mapping_file='annotations/person_keypoints_train_parallel_group_id_mapping.json',
    # 设置不同group_id的损失权重
    # {group_id: weight_multiplier}
    group_id_weight={1: 2.0},  # group_id=1的样本损失权重为2倍
    pipeline=[...],
)
```

**参数说明：**
- `group_id_mapping_file`: group_id映射文件路径（相对于data_root或绝对路径）
- `group_id_weight`: 字典，键为group_id，值为权重倍数。例如`{1: 2.0}`表示group_id=1的样本损失权重为2倍

### 3. 添加ApplyGroupWeight Transform

在训练pipeline中，在`GenerateTarget`和`BottomupGetHeatmapMask`之后添加`ApplyGroupWeight`：

```python
train_pipeline = [
    dict(type='LoadImage'),
    dict(type='BottomupRandomAffine', input_size=codec['input_size']),
    dict(type='RandomFlip', direction='horizontal'),
    dict(type='GenerateTarget', encoder=codec),
    dict(type='BottomupGetHeatmapMask'),
    dict(type='ApplyGroupWeight'),  # 添加这一行
    dict(type='PackPoseInputs'),
]
```

### 4. 开始训练

使用修改后的配置文件进行训练：

```bash
python tools/train.py configs/body_2d_keypoint/dekr/coco/dekr_hrnet-w32_parallel_with_group_weight.py
```

## 工作原理

1. **数据加载阶段**：`CocoParallelDataset`从映射文件中读取每个annotation的group_id，并根据`group_id_weight`设置计算样本权重。

2. **数据转换阶段**：`ApplyGroupWeight` transform将样本权重应用到`heatmap_weights`和`displacement_weights`上，使得group_id=1的样本在损失计算时具有更大的权重。

3. **损失计算阶段**：由于权重已经应用到heatmap_weights和displacement_weights上，损失函数会自动使用这些权重，从而对group_id=1的样本进行更强的监督。

## 权重设置建议

- **权重=1.0**：默认权重，不进行特殊处理
- **权重=1.5-2.0**：轻微提升，适用于样本数量较少的情况
- **权重=2.0-3.0**：中等提升，适用于需要重点关注的情况
- **权重>3.0**：强烈提升，可能导致过拟合，需谨慎使用

建议从较小的权重（如1.5-2.0）开始，根据训练效果逐步调整。

## 注意事项

1. 映射文件必须与COCO标注文件中的annotation ID对应
2. 如果某个annotation没有对应的group_id，其权重默认为1.0
3. 验证集通常不需要应用权重，但可以保留映射文件用于统计
4. 权重过大会导致模型过度关注特定group_id，可能影响整体性能

## 示例配置

完整示例请参考：
`configs/body_2d_keypoint/dekr/coco/dekr_hrnet-w32_parallel_with_group_weight.py`

