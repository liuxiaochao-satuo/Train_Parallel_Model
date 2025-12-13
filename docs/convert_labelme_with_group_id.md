# 从Labelme转换COCO格式并保留group_id

## 概述

本指南说明如何从Labelme标注文件转换到COCO格式，并保留`group_id`信息，以便在训练时针对特定group_id（如运动员）进行精度提升。

## 转换步骤

### 1. 转换单个Labelme文件为COCO格式（保留group_id）

使用修改后的`labelme2coco_bottomup.py`脚本：

```bash
python /home/satuo/code/Train_Parallel_Model/tools/labelme2coco_bottomup.py /path/to/labelme/annotations
```

这个脚本会：
- 读取每个labelme JSON文件
- 提取group_id信息
- 将group_id保存到COCO标注文件的每个annotation中
- 输出到指定目录（默认：`/home/satuo/code/Train_Parallel_Model/data/coco_annotations_only1`）

**输出格式：**
每个COCO JSON文件中的annotation会包含`group_id`字段：
```json
{
  "annotations": [
    {
      "id": 0,
      "image_id": 0,
      "category_id": 1,
      "bbox": [x, y, w, h],
      "keypoints": [...],
      "group_id": 1,  // 保留的group_id信息
      ...
    }
  ]
}
```

### 2. 合并所有COCO文件为完整数据集

使用`merge_coco_files.py`脚本合并所有单个COCO文件：

```bash
python /home/satuo/code/Train_Parallel_Model/tools/merge_coco_files.py \
    /home/satuo/code/Train_Parallel_Model/data/coco_annotations_id \
    --output-dir /home/satuo/code/Train_Parallel_Model/data \
    --split \
    --train-ratio 0.8
```

**参数说明：**
- `input_dir`: 包含单个COCO JSON文件的目录
- `--output-dir`: 输出目录（可选，默认与输入目录相同）
- `--split`: 是否自动分割为训练集和验证集
- `--train-ratio`: 训练集比例（默认0.8）
- `--seed`: 随机种子（默认42）

**输出文件：**
- `person_keypoints_merged.json`: 合并后的完整数据集（如果使用`--split`）
- `person_keypoints_train_parallel.json`: 训练集（如果使用`--split`）
- `person_keypoints_val_parallel.json`: 验证集（如果使用`--split`）

### 3. 配置训练使用group_id

由于COCO文件中已经包含`group_id`信息，您**不需要**额外的映射文件。只需在配置文件中设置`group_id_weight`：

```python
train_dataloader = dict(
    dataset=dict(
        type='CocoParallelDataset',
        data_root='data/coco_parallel',
        data_mode='bottomup',
        ann_file='annotations/person_keypoints_train_parallel.json',
        data_prefix=dict(img='images/'),
        # 不需要group_id_mapping_file，因为COCO文件中已有group_id
        # group_id_mapping_file=None,  # 可以省略
        # 设置group_id权重
        group_id_weight={1: 2.0},  # group_id=1的样本损失权重为2倍
        pipeline=[
            dict(type='LoadImage'),
            dict(type='BottomupRandomAffine', input_size=codec['input_size']),
            dict(type='RandomFlip', direction='horizontal'),
            dict(type='GenerateTarget', encoder=codec),
            dict(type='BottomupGetHeatmapMask'),
            dict(type='ApplyGroupWeight'),  # 应用权重
            dict(type='PackPoseInputs'),
        ],
    ))
```

## 完整工作流程示例

```bash
# 步骤1: 转换所有labelme文件为COCO格式（保留group_id）
python /home/satuo/code/Train_Parallel_Model/tools/labelme2coco_bottomup.py \
    /path/to/labelme/annotations

# 步骤2: 合并并分割数据集
python /home/satuo/code/Train_Parallel_Model/tools/merge_coco_files.py \
    /home/satuo/code/Train_Parallel_Model/data/coco_annotations_only1 \
    --output-dir data/coco_parallel/annotations \
    --split \
    --train-ratio 0.8

# 步骤3: 开始训练（使用包含group_id的COCO文件）
python tools/train.py configs/body_2d_keypoint/dekr/coco/dekr_hrnet-w32_parallel_with_group_weight.py
```

## 优势

相比使用映射文件的方式，直接在COCO文件中保留`group_id`有以下优势：

1. **更简单**：不需要额外的映射文件
2. **更可靠**：group_id直接存储在标注中，不会丢失
3. **更易维护**：所有信息都在一个文件中
4. **向后兼容**：如果COCO文件中没有group_id，仍然可以使用映射文件

## 注意事项

1. 确保labelme文件中的每个关键点都有`group_id`属性
2. 转换后的COCO文件会保留所有group_id信息
3. 如果某个annotation没有group_id，其权重默认为1.0
4. 建议在转换后检查group_id统计信息，确保数据正确

## 验证group_id

转换完成后，可以检查COCO文件中的group_id：

```python
import json

with open('data/coco_parallel/annotations/person_keypoints_train_parallel.json', 'r') as f:
    coco_data = json.load(f)

# 统计group_id
group_id_stats = {}
for ann in coco_data['annotations']:
    group_id = ann.get('group_id', 'None')
    group_id_stats[group_id] = group_id_stats.get(group_id, 0) + 1

print("Group ID统计:")
for group_id, count in sorted(group_id_stats.items()):
    print(f"  group_id={group_id}: {count} 个标注")
```

