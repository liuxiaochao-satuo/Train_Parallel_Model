# DEKR自底向上姿态估计模型训练完整指南

## 📋 目录

1. [概述](#概述)
2. [环境准备](#环境准备)
3. [数据准备](#数据准备)
   - [3.1 Labelme标注格式检查](#31-labelme标注格式检查)
   - [3.2 Labelme转COCO格式](#32-labelme转coco格式)
   - [3.3 COCO格式验证](#33-coco格式验证)
4. [模型配置准备](#模型配置准备)
5. [训练流程](#训练流程)
6. [模型评估与对比](#模型评估与对比)
7. [常见问题](#常见问题)
8. [附录](#附录)

---

## 概述

本指南将帮助您完成从Labelme标注数据到DEKR自底向上姿态估计模型训练的完整流程。

**训练目标**：使用您标注的数据集训练DEKR模型（`dekr_hrnet-w32_8xb10-140e_coco-512x512.py`），提升模型在您的应用场景（双杠系统）中的识别率。

**训练流程概览**：
```
Labelme标注数据
    ↓
格式检查 (check_json.py)
    ↓
转换为COCO格式 (labelme2coco_bottomup.py)
    ↓
COCO格式验证 (validate_coco_format.py)
    ↓
准备模型配置 (基于dekr_hrnet-w32_8xb10-140e_coco-512x512.py)
    ↓
开始训练 (train_dekr.py)
    ↓
模型评估与对比 (evaluate_and_compare.py)
```

---

## 环境准备

### 1. 安装MMPose

```bash
# 如果还没有安装MMPose，请先安装
cd /path/to/mmpose
pip install -e .
```

### 2. 安装依赖

```bash
# 安装必要的Python包
pip install pycocotools
pip install labelme
pip install numpy
pip install opencv-python
```

### 3. 检查GPU环境

```bash
# 检查CUDA和GPU
nvidia-smi

# 检查PyTorch是否支持CUDA
python -c "import torch; print(torch.cuda.is_available())"
```

---

## 数据准备

### 3.1 Labelme标注格式检查

在转换之前，首先检查您的Labelme标注文件是否符合要求。

#### 标注要求

1. **关键点标注**：
   - 每个关键点必须是`point`类型
   - 必须设置`group_id`来标识不同的人
   - 必须设置`description`字段表示可见性（"0"=完全遮挡, "1"=遮挡可推测, "2"=清晰可见）

2. **关键点标签**：
   - 必须使用标准的关键点名称（见下方列表）
   - 每个`group_id`应该包含所有必需的关键点

3. **标准关键点列表**（17个COCO标准关键点）：
   ```
   nose, left_eye, right_eye, left_ear, right_ear,
   left_shoulder, right_shoulder, left_elbow, right_elbow,
   left_wrist, right_wrist, left_hip, right_hip,
   left_knee, right_knee, left_ankle, right_ankle
   ```

#### 执行格式检查

```bash
# 检查单个文件
python check_json.py path/to/your/labelme_file.json

# 检查整个目录（递归搜索）
python check_json.py path/to/your/labelme_annotations/

# 只检查当前目录（不递归）
python check_json.py path/to/your/labelme_annotations/ --no-recursive
```

**检查通过标准**：
- ✅ 没有错误（errors）
- ⚠️ 警告（warnings）可以忽略，但建议修复

**如果检查失败**：
- 修复标注文件中的错误
- 确保所有关键点都有`group_id`
- 确保`description`字段值为"0"、"1"或"2"
- 确保关键点标签名称正确

---

### 3.2 Labelme转COCO格式

#### 准备数据目录结构

在开始转换之前，建议按以下方式组织数据：

```
Train_Parallel_Model/
├── labelme_annotations/          # Labelme JSON文件目录
│   ├── image001.json
│   ├── image002.json
│   └── ...
├── images/                        # 对应的图像文件
│   ├── image001.jpg
│   ├── image002.jpg
│   └── ...
└── labelme2coco_bottomup.py      # 转换脚本
```

**注意**：确保JSON文件和图像文件在同一目录，或者JSON文件中的`imagePath`字段指向正确的图像路径。

#### 执行转换

```bash
# 进入Train_Parallel_Model目录
cd /home/satuo/code/Train_Parallel_Model

# 将Labelme JSON文件复制到当前目录（或修改脚本中的路径）
# 然后运行转换脚本
python labelme2coco_bottomup.py
```

**转换输出**：
- 输出目录：`output_coco/`
- 输出文件：`output_coco/coco_bottomup.json`

**转换后的数据结构**：
```json
{
  "categories": [
    {
      "supercategory": "person",
      "id": 1,
      "name": "person",
      "keypoints": ["nose", "left_eye", ...],
      "skeleton": [[16, 14], [14, 12], ...]
    }
  ],
  "images": [
    {
      "file_name": "image001.jpg",
      "height": 480,
      "width": 640,
      "id": 0
    }
  ],
  "annotations": [
    {
      "id": 0,
      "image_id": 0,
      "category_id": 1,
      "bbox": [x, y, width, height],
      "area": width * height,
      "iscrowd": 0,
      "keypoints": [x1, y1, v1, x2, y2, v2, ...],
      "num_keypoints": 17
    }
  ]
}
```

---

### 3.3 COCO格式验证

转换完成后，必须验证COCO格式是否正确，以确保可以用于训练。

#### 使用验证脚本

```bash
# 验证COCO格式文件
python validate_coco_format.py output_coco/coco_bottomup.json
```

**验证内容**：
- ✅ JSON格式正确性
- ✅ 必需字段完整性
- ✅ 关键点格式正确性
- ✅ 图像文件存在性
- ✅ 数据统计信息

**验证通过标准**：
- 所有检查项通过
- 没有关键错误
- 数据统计信息合理

#### 手动验证（可选）

```python
# 使用pycocotools验证
from pycocotools.coco import COCO

coco = COCO('output_coco/coco_bottomup.json')
print(f'图像数量: {len(coco.imgs)}')
print(f'标注数量: {len(coco.anns)}')
print(f'类别数量: {len(coco.cats)}')
```

---

## 模型配置准备

### 4.1 复制配置文件

```bash
# 从brain目录复制配置文件到Train_Parallel_Model
cp /home/satuo/code/brain/algorithm/config/dekr_hrnet-w32_8xb10-140e_coco-512x512.py \
   /home/satuo/code/Train_Parallel_Model/configs/dekr_hrnet-w32_custom.py

# 复制default_runtime.py（如果不存在）
cp /home/satuo/code/brain/algorithm/config/default_runtime.py \
   /home/satuo/code/Train_Parallel_Model/configs/
```

### 4.2 修改配置文件

编辑 `configs/dekr_hrnet-w32_custom.py`，修改以下部分：

#### 修改数据路径

```python
# 原始配置
data_root = 'data/coco/'

# 修改为您的数据路径
data_root = '/home/satuo/code/Train_Parallel_Model/data/coco/'
```

#### 修改标注文件路径

```python
# 训练集标注
train_dataloader = dict(
    ...
    dataset=dict(
        ...
        ann_file='annotations/person_keypoints_train2017.json',  # 改为您的训练集标注
        data_prefix=dict(img='train2017/'),  # 改为您的训练图像目录
        ...
    )
)

# 验证集标注
val_dataloader = dict(
    ...
    dataset=dict(
        ...
        ann_file='annotations/person_keypoints_val2017.json',  # 改为您的验证集标注
        data_prefix=dict(img='val2017/'),  # 改为您的验证图像目录
        ...
    )
)
```

#### 修改评估器配置

```python
val_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + 'annotations/person_keypoints_val2017.json',  # 改为您的验证集标注
    nms_mode='none',
    score_mode='keypoint',
)
```

#### 调整训练参数（可选）

根据您的GPU内存和数据集大小调整：

```python
# 调整batch size
train_dataloader = dict(
    batch_size=10,  # 根据GPU内存调整（如果内存不足，可以减小到4或8）
    num_workers=2,  # 根据CPU核心数调整
    ...
)

# 调整学习率（如果batch size改变）
# 学习率通常与batch size成正比
# 如果batch size从10改为5，学习率可以从1e-3改为5e-4
optim_wrapper = dict(
    optimizer=dict(
        type='Adam',
        lr=1e-3,  # 根据batch size调整
    )
)

# 调整训练轮数
train_cfg = dict(max_epochs=140, val_interval=10)  # 可以根据需要调整
```

#### 移除rescore_cfg（如果使用自定义数据集）

如果您的数据集不是标准COCO数据集，建议移除`rescore_cfg`：

```python
head=dict(
    type='DEKRHead',
    ...
    # 注释掉或删除以下部分
    # rescore_cfg=dict(
    #     in_channels=74,
    #     norm_indexes=(5, 6),
    #     init_cfg=dict(
    #         type='Pretrained',
    #         checkpoint='https://download.openmmlab.com/mmpose/'
    #         'pretrain_models/kpt_rescore_coco-33d58c5c.pth')),
)
```

### 4.3 准备数据目录结构

创建符合MMPose要求的数据目录结构：

```bash
cd /home/satuo/code/Train_Parallel_Model

# 创建数据目录
mkdir -p data/coco/annotations
mkdir -p data/coco/train2017
mkdir -p data/coco/val2017

# 复制COCO格式标注文件
cp output_coco/coco_bottomup.json data/coco/annotations/person_keypoints_train2017.json

# 如果需要验证集，可以手动分割数据集
# 或者复制同一份文件作为验证集（仅用于测试）
cp output_coco/coco_bottomup.json data/coco/annotations/person_keypoints_val2017.json

# 复制图像文件到对应目录
# 假设您的图像在images/目录下
cp images/*.jpg data/coco/train2017/
# 如果需要验证集，可以手动分割
# cp images/val_*.jpg data/coco/val2017/
```

**数据目录结构**：
```
data/
└── coco/
    ├── annotations/
    │   ├── person_keypoints_train2017.json
    │   └── person_keypoints_val2017.json
    ├── train2017/
    │   ├── image001.jpg
    │   ├── image002.jpg
    │   └── ...
    └── val2017/
        ├── image100.jpg
        ├── image101.jpg
        └── ...
```

---

## 训练流程

### 5.1 保存预训练模型检查点

在开始训练之前，建议先保存预训练模型的性能指标，以便后续对比。

```bash
# 使用评估脚本保存预训练模型指标
python evaluate_and_compare.py \
    --config /home/satuo/code/brain/algorithm/config/dekr_hrnet-w32_8xb10-140e_coco-512x512.py \
    --checkpoint /home/satuo/code/brain/algorithm/checkpoints/dekr_hrnet-w32_8xb10-140e_coco-512x512_ac7c17bf-20221228.pth \
    --ann-file data/coco/annotations/person_keypoints_val2017.json \
    --output metrics_pretrained.json
```

### 5.2 开始训练

#### 方式1：使用训练脚本（推荐）

```bash
# 使用提供的训练脚本
python train_dekr.py \
    --config configs/dekr_hrnet-w32_custom.py \
    --work-dir work_dirs/dekr_custom \
    --gpus 1
```

#### 方式2：直接使用MMPose训练命令

```bash
# 单GPU训练
python tools/train.py \
    configs/dekr_hrnet-w32_custom.py \
    --work-dir work_dirs/dekr_custom

# 多GPU训练（例如4个GPU）
bash tools/dist_train.sh \
    configs/dekr_hrnet-w32_custom.py \
    4 \
    --work-dir work_dirs/dekr_custom

# 从checkpoint恢复训练
python tools/train.py \
    configs/dekr_hrnet-w32_custom.py \
    --resume work_dirs/dekr_custom/epoch_100.pth
```

### 5.3 监控训练过程

#### 使用TensorBoard

```bash
# 启动TensorBoard
tensorboard --logdir work_dirs/dekr_custom

# 在浏览器中打开 http://localhost:6006
```

#### 查看训练日志

```bash
# 实时查看训练日志
tail -f work_dirs/dekr_custom/train.log

# 或者查看最新的日志文件
cat work_dirs/dekr_custom/*.log | tail -100
```

### 5.4 训练输出

训练完成后，在`work_dirs/dekr_custom/`目录下会生成：

- `best.pth`：验证集上表现最好的模型
- `latest.pth`：最新的模型checkpoint
- `epoch_*.pth`：每个epoch的checkpoint
- `train.log`：训练日志
- `vis_data/`：可视化数据（用于TensorBoard）

---

## 模型评估与对比

### 6.1 评估训练后的模型

```bash
# 评估训练后的模型
python evaluate_and_compare.py \
    --config configs/dekr_hrnet-w32_custom.py \
    --checkpoint work_dirs/dekr_custom/best.pth \
    --ann-file data/coco/annotations/person_keypoints_val2017.json \
    --output metrics_trained.json
```

### 6.2 对比训练前后的模型

```bash
# 对比预训练模型和训练后模型
python evaluate_and_compare.py \
    --config configs/dekr_hrnet-w32_custom.py \
    --checkpoint-pretrained /home/satuo/code/brain/algorithm/checkpoints/dekr_hrnet-w32_8xb10-140e_coco-512x512_ac7c17bf-20221228.pth \
    --checkpoint-trained work_dirs/dekr_custom/best.pth \
    --ann-file data/coco/annotations/person_keypoints_val2017.json \
    --output comparison_report.json \
    --compare
```

**对比报告包含**：
- AP (Average Precision) 指标对比
- AR (Average Recall) 指标对比
- 各关键点的精度对比
- 识别率提升百分比

### 6.3 可视化对比结果

评估脚本会生成对比报告，包括：

1. **JSON格式报告**：包含详细的数值对比
2. **文本格式报告**：便于阅读的文本输出
3. **可视化图表**（如果启用）：关键点精度对比图

---

## 常见问题

### Q1: 训练时出现内存不足（OOM）错误

**解决方案**：
1. 减小batch size（例如从10改为4或8）
2. 减小输入图像尺寸（修改`input_size`）
3. 减小`decode_max_instances`（在codec配置中）
4. 使用梯度累积（在配置中添加`accumulative_counts`）

### Q2: 训练损失不下降或下降很慢

**可能原因和解决方案**：
1. **学习率过大或过小**：调整学习率
2. **数据质量问题**：检查标注是否正确
3. **数据量不足**：增加训练数据
4. **预训练模型不匹配**：确保使用正确的预训练模型

### Q3: 验证集指标为0或异常

**检查项**：
1. 验证集标注文件路径是否正确
2. 验证集图像文件是否存在
3. 验证集标注格式是否正确
4. 图像路径是否与标注文件中的`file_name`匹配

### Q4: 如何判断模型是否过拟合？

**判断方法**：
- 训练集损失持续下降，但验证集损失不再下降或上升
- 训练集精度远高于验证集精度

**解决方案**：
- 增加数据增强
- 使用dropout
- 减小模型容量
- 增加训练数据

### Q5: 训练中断后如何恢复？

```bash
# 从最新的checkpoint恢复
python tools/train.py \
    configs/dekr_hrnet-w32_custom.py \
    --resume work_dirs/dekr_custom/latest.pth
```

### Q6: 如何调整关键点数量？

如果您的数据集使用不同的关键点数量：

1. 修改配置文件中的`num_keypoints`：
   ```python
   head=dict(
       type='DEKRHead',
       num_keypoints=17,  # 改为您的关键点数量
       ...
   )
   ```

2. 修改`categories`中的`keypoints`列表：
   ```python
   class_list = {
       'keypoints': ['nose', 'left_eye', ...],  # 改为您的关键点列表
       ...
   }
   ```

3. 修改转换脚本中的`STANDARD_KEYPOINT_ORDER`

---

## 附录

### A. 完整训练命令示例

```bash
# 1. 检查Labelme格式
python check_json.py labelme_annotations/

# 2. 转换为COCO格式
python labelme2coco_bottomup.py

# 3. 验证COCO格式
python validate_coco_format.py output_coco/coco_bottomup.json

# 4. 准备数据目录
mkdir -p data/coco/annotations data/coco/train2017
cp output_coco/coco_bottomup.json data/coco/annotations/person_keypoints_train2017.json
cp images/*.jpg data/coco/train2017/

# 5. 保存预训练模型指标
python evaluate_and_compare.py \
    --config /home/satuo/code/brain/algorithm/config/dekr_hrnet-w32_8xb10-140e_coco-512x512.py \
    --checkpoint /home/satuo/code/brain/algorithm/checkpoints/dekr_hrnet-w32_8xb10-140e_coco-512x512_ac7c17bf-20221228.pth \
    --ann-file data/coco/annotations/person_keypoints_val2017.json \
    --output metrics_pretrained.json

# 6. 开始训练
python train_dekr.py \
    --config configs/dekr_hrnet-w32_custom.py \
    --work-dir work_dirs/dekr_custom \
    --gpus 1

# 7. 评估训练后的模型
python evaluate_and_compare.py \
    --config configs/dekr_hrnet-w32_custom.py \
    --checkpoint work_dirs/dekr_custom/best.pth \
    --ann-file data/coco/annotations/person_keypoints_val2017.json \
    --output metrics_trained.json

# 8. 对比训练前后模型
python evaluate_and_compare.py \
    --config configs/dekr_hrnet-w32_custom.py \
    --checkpoint-pretrained /home/satuo/code/brain/algorithm/checkpoints/dekr_hrnet-w32_8xb10-140e_coco-512x512_ac7c17bf-20221228.pth \
    --checkpoint-trained work_dirs/dekr_custom/best.pth \
    --ann-file data/coco/annotations/person_keypoints_val2017.json \
    --output comparison_report.json \
    --compare
```

### B. 配置文件关键参数说明

| 参数 | 说明 | 默认值 | 建议调整 |
|------|------|--------|----------|
| `batch_size` | 批次大小 | 10 | 根据GPU内存调整 |
| `num_workers` | 数据加载线程数 | 2 | 根据CPU核心数调整 |
| `lr` | 学习率 | 1e-3 | 与batch_size成正比 |
| `max_epochs` | 最大训练轮数 | 140 | 根据数据集大小调整 |
| `input_size` | 输入图像尺寸 | (512, 512) | 根据GPU内存调整 |
| `decode_max_instances` | 最大检测人数 | 30 | 根据场景调整 |

### C. 评估指标说明

- **AP (Average Precision)**：平均精度，衡量检测精度
- **AR (Average Recall)**：平均召回率，衡量检测完整性
- **AP@0.5**：IoU阈值为0.5时的AP
- **AP@0.75**：IoU阈值为0.75时的AP
- **AP (medium)**：中等大小目标的AP
- **AP (large)**：大目标的AP

### D. 文件清单

训练所需的所有文件：

```
Train_Parallel_Model/
├── check_json.py                    # Labelme格式检查脚本
├── labelme2coco_bottomup.py         # Labelme转COCO格式脚本
├── validate_coco_format.py          # COCO格式验证脚本
├── train_dekr.py                    # 训练脚本
├── evaluate_and_compare.py         # 评估对比脚本
├── configs/
│   ├── default_runtime.py           # 运行时配置
│   └── dekr_hrnet-w32_custom.py     # 自定义训练配置
├── data/
│   └── coco/
│       ├── annotations/
│       ├── train2017/
│       └── val2017/
├── work_dirs/
│   └── dekr_custom/                 # 训练输出目录
└── DEKR模型训练完整指南.md          # 本指南
```

---

## 总结

完成以上步骤后，您将：

1. ✅ 完成Labelme到COCO格式的转换
2. ✅ 验证数据格式正确性
3. ✅ 配置并开始模型训练
4. ✅ 获得训练后的模型
5. ✅ 对比训练前后的模型性能
6. ✅ 获得识别率提升的量化指标

**下一步**：将训练好的模型部署到您的双杠系统中，替换原有的预训练模型。

---

**祝训练顺利！如有问题，请参考常见问题部分或查看MMPose官方文档。**

