# MMPose 自底向上姿态估计模型训练指南

## 📋 目录
1. [MMPose中的自底向上方法](#mmpose中的自底向上方法)
2. [支持的模型](#支持的模型)
3. [配置文件结构](#配置文件结构)
4. [数据准备](#数据准备)
5. [训练步骤](#训练步骤)
6. [关键配置说明](#关键配置说明)

---

## MMPose中的自底向上方法

### 什么是自底向上方法？

**自底向上（Bottom-Up）方法**：
- 先检测图像中所有关键点
- 然后将关键点分组/关联到不同的人
- 不需要预先检测人的bounding box

**与自顶向下的区别**：
- **自顶向下**：先检测人（bounding box）→ 再检测关键点
- **自底向上**：先检测所有关键点 → 再分组到不同的人

### MMPose中的实现

在MMPose中，自底向上方法通过以下方式实现：
- **模型类型**：`BottomupPoseEstimator`
- **数据模式**：`data_mode = 'bottomup'`
- **Codec**：使用特定的codec（如`AssociativeEmbedding`）来编码/解码关键点和分组信息

---

## 支持的模型

MMPose在 `configs/body_2d_keypoint/` 目录下提供了多种自底向上方法：

### 1. Associative Embedding (AE)
- **目录**：`associative_embedding/`
- **原理**：为每个关键点预测一个tag，相同人的关键点tag相似，不同人的tag不同
- **配置文件**：`ae_hrnet-w32_8xb24-300e_coco-512x512.py`
- **README**：`associative_embedding/README.md`

### 2. RTMO (Real-Time Multi-Person One-stage)
- **目录**：`rtmo/`
- **原理**：单阶段实时多人姿态估计，集成到YOLO架构中
- **特点**：实时性能，适合多人场景
- **README**：`rtmo/README.md`

### 3. DEKR (Disentangled Keypoint Regression)
- **目录**：`dekr/`
- **支持数据集**：COCO, CrowdPose

### 4. CID (Center and Scale Invariant Detection)
- **目录**：`cid/`
- **支持数据集**：COCO

### 5. EDPose
- **目录**：`edpose/`
- **支持数据集**：COCO

---

## 配置文件结构

### 关键配置项

以 `associative_embedding/coco/ae_hrnet-w32_8xb24-300e_coco-512x512.py` 为例：

```python
# 1. 模型类型 - 必须是BottomupPoseEstimator
model = dict(
    type='BottomupPoseEstimator',  # ← 关键：自底向上模型
    ...
)

# 2. Codec设置 - 用于编码/解码关键点和分组信息
codec = dict(
    type='AssociativeEmbedding',  # ← 关键：AE codec
    input_size=(512, 512),
    heatmap_size=(128, 128),
    sigma=2,
    decode_topk=30,  # 最多解码30个实例
    decode_max_instances=30,  # 最多30个人
)

# 3. 数据模式 - 必须设置为bottomup
data_mode = 'bottomup'  # ← 关键：自底向上数据模式

# 4. 数据集配置
dataset = dict(
    type='CocoDataset',
    data_root='data/coco/',
    data_mode=data_mode,  # ← 必须设置
    ann_file='annotations/person_keypoints_train2017.json',
    ...
)

# 5. 数据变换 - 使用bottomup专用的transform
train_pipeline = [
    dict(type='LoadImage'),
    dict(type='BottomupRandomAffine', ...),  # ← bottomup专用
    dict(type='BottomupGetHeatmapMask', ...),  # ← bottomup专用
    ...
]
```

---

## 数据准备

### COCO格式要求

对于自底向上方法，COCO格式的标注文件需要：

1. **每个annotation代表一个人**
   ```json
   {
     "annotations": [
       {
         "id": 0,
         "image_id": 0,
         "category_id": 1,
         "bbox": [x, y, w, h],  // 可选，但建议有
         "keypoints": [x1, y1, v1, x2, y2, v2, ...],
         "num_keypoints": 17
       }
     ]
   }
   ```

2. **关键点格式**
   - 每3个数字一组：`[x坐标, y坐标, 可见性]`
   - 可见性：`0`=不存在, `1`=遮挡, `2`=可见
   - 必须按照`categories[].keypoints`定义的顺序排列

3. **不需要预先检测bounding box**
   - bbox可以从关键点计算（但COCO格式要求有bbox字段）
   - 训练时模型主要使用关键点信息

### 从Labelme转换

使用我们之前创建的 `labelme2coco_bottomup.py` 脚本：

```bash
python labelme2coco_bottomup.py
```

**输出**：`output_coco/coco_bottomup.json`

---

## 训练步骤

### 1. 准备数据

```bash
# 数据目录结构
data/
  coco/
    annotations/
      person_keypoints_train2017.json  # 训练集标注
      person_keypoints_val2017.json    # 验证集标注
    train2017/  # 训练图像
    val2017/    # 验证图像
```

### 2. 选择配置文件

以Associative Embedding为例：

```bash
# 配置文件路径
configs/body_2d_keypoint/associative_embedding/coco/ae_hrnet-w32_8xb24-300e_coco-512x512.py
```

### 3. 修改配置（如需要）

```python
# 修改数据路径
data_root = 'data/coco/'  # 改为你的数据路径
ann_file = 'annotations/person_keypoints_train2017.json'  # 改为你的标注文件

# 修改batch size（根据GPU内存调整）
train_dataloader = dict(batch_size=24, ...)

# 修改学习率（如果batch size改变）
optim_wrapper = dict(optimizer=dict(lr=1.5e-3, ...))
```

### 4. 开始训练

```bash
# 单GPU训练
python tools/train.py configs/body_2d_keypoint/associative_embedding/coco/ae_hrnet-w32_8xb24-300e_coco-512x512.py

# 多GPU训练（例如4个GPU）
bash tools/dist_train.sh configs/body_2d_keypoint/associative_embedding/coco/ae_hrnet-w32_8xb24-300e_coco-512x512.py 4

# 指定工作目录
python tools/train.py configs/.../xxx.py --work-dir work_dirs/my_experiment

# 从checkpoint恢复训练
python tools/train.py configs/.../xxx.py --resume work_dirs/xxx/epoch_100.pth
```

### 5. 验证和测试

```bash
# 测试
python tools/test.py configs/.../xxx.py work_dirs/xxx/best.pth

# 推理演示
python demo/bottomup_demo.py \
    configs/.../xxx.py \
    work_dirs/xxx/best.pth \
    --input path/to/image.jpg \
    --output-root output/
```

---

## 关键配置说明

### 1. Codec配置

不同的自底向上方法使用不同的codec：

#### Associative Embedding
```python
codec = dict(
    type='AssociativeEmbedding',
    input_size=(512, 512),
    heatmap_size=(128, 128),
    sigma=2,
    decode_topk=30,  # 解码时保留top-k个关键点
    decode_max_instances=30,  # 最多检测30个人
)
```

#### RTMO
```python
codec = dict(
    type='RTMOCodec',
    input_size=(640, 640),
    ...
)
```

### 2. 数据变换（Transforms）

自底向上方法使用专门的数据变换：

```python
train_pipeline = [
    dict(type='LoadImage'),
    dict(
        type='BottomupRandomAffine',  # 自底向上专用的仿射变换
        input_size=(512, 512),
        ...
    ),
    dict(
        type='BottomupGetHeatmapMask',  # 生成heatmap mask
        ...
    ),
    dict(type='PackPoseInputs'),
]
```

**关键变换**：
- `BottomupRandomAffine`：对整张图像进行仿射变换（不是对每个人）
- `BottomupGetHeatmapMask`：生成heatmap mask，用于处理遮挡
- `BottomupResize`：调整图像大小
- `BottomupRandomCrop`：随机裁剪

### 3. 评估器配置

```python
val_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + 'annotations/person_keypoints_val2017.json',
    nms_mode='none',  # 自底向上不需要NMS（或使用特定模式）
    score_mode='bbox',  # 或'keypoint'
)
```

### 4. 模型Head配置

#### Associative Embedding Head
```python
head=dict(
    type='AssociativeEmbeddingHead',
    in_channels=32,
    num_keypoints=17,
    tag_dim=1,  # tag维度
    tag_per_keypoint=True,  # 每个关键点一个tag
    keypoint_loss=dict(type='KeypointMSELoss', use_target_weight=True),
    tag_loss=dict(type='AssociativeEmbeddingLoss', loss_weight=0.001),  # tag损失
)
```

---

## 常见问题

### Q1: 如何选择自底向上还是自顶向下？

**A**: 
- **自底向上**：适合多人场景，推理速度快（不随人数线性增长）
- **自顶向下**：通常精度更高，但推理时间随人数线性增长

### Q2: 自底向上方法需要bounding box吗？

**A**: 
- **训练时**：COCO格式要求有bbox字段，但模型主要使用关键点信息
- **推理时**：不需要预先检测人，直接从图像检测所有关键点

### Q3: 如何调整检测的人数上限？

**A**: 修改codec配置：
```python
codec = dict(
    decode_max_instances=50,  # 改为50个人
    decode_topk=50,
)
```

### Q4: 训练时出现内存不足怎么办？

**A**: 
1. 减小batch size
2. 减小输入图像尺寸（`input_size`）
3. 减小`decode_max_instances`

### Q5: 如何在自己的数据集上训练？

**A**: 
1. 将数据转换为COCO格式（使用`labelme2coco_bottomup.py`）
2. 修改配置文件中的数据路径
3. 如果关键点数量不同，需要修改：
   - `num_keypoints`
   - `categories`中的`keypoints`列表
   - `dataset_info`中的关键点定义

---

## 推荐的训练流程

### 1. 数据准备阶段
```bash
# 1. 检查Labelme标注
python check_json.py your_annotations/

# 2. 转换为COCO格式
python labelme2coco_bottomup.py

# 3. 验证COCO格式
python -c "from pycocotools.coco import COCO; c = COCO('output_coco/coco_bottomup.json'); print(f'图像: {len(c.imgs)}, 标注: {len(c.anns)}')"
```

### 2. 配置准备阶段
```bash
# 1. 复制配置文件
cp configs/body_2d_keypoint/associative_embedding/coco/ae_hrnet-w32_8xb24-300e_coco-512x512.py \
   configs/my_custom_config.py

# 2. 修改配置文件
# - 数据路径
# - 关键点数量（如果不同）
# - batch size等超参数
```

### 3. 训练阶段
```bash
# 1. 开始训练
python tools/train.py configs/my_custom_config.py --work-dir work_dirs/my_experiment

# 2. 监控训练（使用TensorBoard）
tensorboard --logdir work_dirs/my_experiment
```

### 4. 评估和测试
```bash
# 1. 测试模型
python tools/test.py configs/my_custom_config.py work_dirs/my_experiment/best.pth

# 2. 推理演示
python demo/bottomup_demo.py \
    configs/my_custom_config.py \
    work_dirs/my_experiment/best.pth \
    --input test_image.jpg \
    --output-root output/
```

---

## 总结

MMPose提供了完整的自底向上姿态估计训练支持：

1. ✅ **多种方法**：AE, RTMO, DEKR, CID, EDPose等
2. ✅ **完整配置**：从数据加载到模型训练的完整配置
3. ✅ **专用工具**：bottomup专用的数据变换和codec
4. ✅ **易于使用**：标准的训练和测试接口

**关键点**：
- 使用`BottomupPoseEstimator`模型类型
- 设置`data_mode = 'bottomup'`
- 使用bottomup专用的数据变换
- 配置合适的codec（如`AssociativeEmbedding`）

现在你可以使用转换好的COCO格式数据开始训练自底向上姿态估计模型了！

