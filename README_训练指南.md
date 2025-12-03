# DEKR模型训练快速开始指南

## 📁 文件清单

本目录包含完整的DEKR自底向上姿态估计模型训练工具链：

### 核心文档
- **`DEKR模型训练完整指南.md`** - 详细的训练文档，包含所有步骤和说明

### 数据转换工具
- **`check_json.py`** - Labelme格式检查脚本
- **`labelme2coco_bottomup.py`** - Labelme转COCO格式转换脚本
- **`validate_coco_format.py`** - COCO格式验证脚本

### 训练工具
- **`train_dekr.py`** - DEKR模型训练脚本
- **`evaluate_and_compare.py`** - 模型评估与对比脚本

### 配置文件
- **`configs/dekr_hrnet-w32_custom.py`** - 自定义训练配置文件模板

### 其他文档
- **`mmpose自底向上训练指南.md`** - MMPose自底向上方法通用指南
- **`labelme2coco转换原理详解.md`** - 转换原理说明
- **`自底向上姿态估计转换注意事项.md`** - 转换注意事项

---

## 🚀 快速开始

### 步骤1: 检查Labelme标注格式

```bash
python check_json.py your_labelme_annotations/
```

### 步骤2: 转换为COCO格式

```bash
python labelme2coco_bottomup.py
```

输出：`output_coco/coco_bottomup.json`

### 步骤3: 验证COCO格式

```bash
python validate_coco_format.py output_coco/coco_bottomup.json
```

### 步骤4: 准备数据目录

```bash
mkdir -p data/coco/annotations data/coco/train2017
cp output_coco/coco_bottomup.json data/coco/annotations/person_keypoints_train2017.json
cp your_images/*.jpg data/coco/train2017/
```

### 步骤5: 修改配置文件

编辑 `configs/dekr_hrnet-w32_custom.py`，修改数据路径等配置。

### 步骤6: 开始训练

```bash
# 方式1: 使用训练脚本（推荐）
python train_dekr.py \
    --config configs/dekr_hrnet-w32_custom.py \
    --work-dir work_dirs/dekr_custom \
    --gpus 1

# 方式2: 直接使用MMPose命令
python tools/train.py configs/dekr_hrnet-w32_custom.py --work-dir work_dirs/dekr_custom
```

### 步骤7: 评估和对比

```bash
# 评估训练后的模型
python evaluate_and_compare.py \
    --config configs/dekr_hrnet-w32_custom.py \
    --checkpoint work_dirs/dekr_custom/best.pth \
    --ann-file data/coco/annotations/person_keypoints_val2017.json \
    --output metrics_trained.json

# 对比训练前后模型
python evaluate_and_compare.py \
    --config configs/dekr_hrnet-w32_custom.py \
    --checkpoint-pretrained /path/to/pretrained.pth \
    --checkpoint-trained work_dirs/dekr_custom/best.pth \
    --ann-file data/coco/annotations/person_keypoints_val2017.json \
    --output comparison_report.json \
    --compare
```

---

## 📖 详细文档

请参阅 **`DEKR模型训练完整指南.md`** 获取完整的训练流程和详细说明。

---

## ⚠️ 注意事项

1. **环境要求**：
   - Python 3.7+
   - PyTorch 1.8+
   - MMPose 1.0+
   - CUDA（推荐）

2. **数据要求**：
   - Labelme标注文件必须包含`group_id`字段
   - 关键点标签必须使用标准名称
   - 每个`group_id`应该包含所有必需的关键点

3. **配置文件**：
   - 使用自定义数据集时，建议注释掉`rescore_cfg`
   - 根据GPU内存调整`batch_size`和`input_size`

4. **训练建议**：
   - 首次训练建议使用较小的`batch_size`（如4或8）
   - 监控训练过程，使用TensorBoard查看训练曲线
   - 如果出现OOM错误，减小`batch_size`或`input_size`

---

## 🆘 获取帮助

- 查看 `DEKR模型训练完整指南.md` 中的"常见问题"部分
- 查看 `mmpose自底向上训练指南.md` 了解MMPose使用方法
- 查看MMPose官方文档：https://mmpose.readthedocs.io/

---

**祝训练顺利！**

