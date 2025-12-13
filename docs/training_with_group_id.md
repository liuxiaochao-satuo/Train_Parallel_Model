# 使用带group_id的COCO标注文件进行训练

## 快速开始

如果您已经将带有`group_id`的COCO标注文件保存在 `data/coco_parallel/annotations_id/` 目录下，可以按照以下步骤进行训练：

### 1. 确认文件结构

确保以下文件存在：
```
data/coco_parallel/
├── annotations_id/
│   ├── person_keypoints_train_parallel.json  # 训练集（包含group_id）
│   └── person_keypoints_val_parallel.json    # 验证集（包含group_id）
└── images/
    └── ...  # 图像文件
```

### 2. 使用配置文件进行训练

使用已配置好的配置文件：

```bash
python tools/train.py \
    configs/body_2d_keypoint/dekr/coco/dekr_hrnet-w32_parallel_group_id_training.py \
    --work-dir work_dirs/dekr_hrnet-w32_parallel_group_id
```

### 3. 配置文件说明

配置文件 `dekr_hrnet-w32_parallel_group_id_training.py` 的关键设置：

```python
# 数据集配置
dataset=dict(
    type='CocoParallelDataset',
    data_root='data/coco_parallel',
    ann_file='annotations_id/person_keypoints_train_parallel.json',  # 使用带group_id的文件
    # 不需要group_id_mapping_file，因为COCO文件中已包含group_id
    group_id_weight={1: 2.0},  # group_id=1的样本损失权重为2倍
    pipeline=[..., dict(type='ApplyGroupWeight'), ...],  # 应用权重
)
```

### 4. 调整权重

根据您的需求调整`group_id_weight`：

- **轻微提升**：`{1: 1.5}` - 适用于样本数量较多的情况
- **中等提升**：`{1: 2.0}` - 推荐起始值
- **强烈提升**：`{1: 3.0}` - 适用于样本数量较少的情况

**注意**：权重过大会导致模型过度关注特定group_id，可能影响整体性能。

### 5. 验证group_id是否正确加载

训练开始时会打印group_id映射信息：
```
Loaded group_id mapping from ...
Total mappings: ...
Group ID weights: {1: 2.0}
```

如果COCO文件中包含group_id，会显示：
```
Loaded group_id from COCO annotations
Group ID weights: {1: 2.0}
```

### 6. 监控训练效果

训练过程中，可以观察：
- 损失值的变化
- 验证集上的精度（特别是group_id=1的样本）
- 训练日志中的相关信息

## 自定义配置

如果需要自定义配置，可以基于 `dekr_hrnet-w32_parallel_group_id_training.py` 进行修改：

```python
# 修改权重
group_id_weight={1: 2.5, 2: 1.5}  # 可以同时设置多个group_id的权重

# 修改数据路径
ann_file='your_custom_path/train.json'

# 修改batch size等训练参数
batch_size=16
```

## 常见问题

### Q: 如何确认COCO文件中包含group_id？

A: 可以检查annotation中是否包含`group_id`字段：
```python
import json
with open('data/coco_parallel/annotations_id/person_keypoints_train_parallel.json') as f:
    data = json.load(f)
    print('group_id' in data['annotations'][0])  # 应该返回True
```

### Q: 训练时没有看到group_id相关信息？

A: 检查：
1. COCO文件中是否真的包含`group_id`字段
2. 配置文件中的路径是否正确
3. 是否使用了`CocoParallelDataset`而不是普通的`CocoDataset`

### Q: 如何调整权重？

A: 修改配置文件中的`group_id_weight`参数，然后重新开始训练或从checkpoint继续训练。

## 统计信息

根据您的数据，group_id分布如下：
- group_id=1: 4072 个标注（运动员）
- group_id=2: 4081 个标注
- group_id=3: 234 个标注
- group_id=4: 16 个标注

建议对group_id=1使用权重2.0，这样可以平衡不同group_id的样本数量差异。

