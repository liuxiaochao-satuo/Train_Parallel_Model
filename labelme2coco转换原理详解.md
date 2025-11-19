# Labelme转COCO格式转换原理详解

## 📋 目录
1. [格式对比](#格式对比)
2. [转换流程概览](#转换流程概览)
3. [核心转换函数详解](#核心转换函数详解)
4. [数据结构映射](#数据结构映射)
5. [关键点处理逻辑](#关键点处理逻辑)

---

## 格式对比

### Labelme格式结构
```json
{
  "version": "4.5.6",
  "flags": {},
  "imagePath": "image.jpg",
  "imageData": "...",
  "imageHeight": 480,
  "imageWidth": 640,
  "shapes": [
    {
      "label": "sjb_rect",
      "shape_type": "rectangle",
      "points": [[x1, y1], [x2, y2]]
    },
    {
      "label": "angle_30",
      "shape_type": "point",
      "points": [[x, y]]
    },
    {
      "label": "polygon",
      "shape_type": "polygon",
      "points": [[x1, y1], [x2, y2], ...]
    }
  ]
}
```

### COCO格式结构
```json
{
  "info": {...},
  "licenses": [...],
  "categories": [
    {
      "supercategory": "sjb_rect",
      "id": 1,
      "name": "sjb_rect",
      "keypoints": ["angle_30", "angle_60", "angle_90"],
      "skeleton": [[0,1], [0,2], [1,2]]
    }
  ],
  "images": [
    {
      "file_name": "image.jpg",
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
      "bbox": [x, y, w, h],
      "area": w * h,
      "iscrowd": 0,
      "segmentation": [[x1, y1, x2, y2, ...]],
      "num_keypoints": 3,
      "keypoints": [x1, y1, v1, x2, y2, v2, ...]
    }
  ]
}
```

---

## 转换流程概览

```
Labelme JSON文件
    ↓
1. 初始化COCO字典结构
    ↓
2. 定义categories（类别和关键点顺序）
    ↓
3. 遍历每个Labelme JSON文件
    ↓
4. 提取图像信息 → images数组
    ↓
5. 处理标注信息 → annotations数组
    ├─ 提取rectangle（边界框）
    ├─ 匹配polygon（分割掩码）
    └─ 匹配point（关键点）
    ↓
6. 保存为COCO格式JSON
```

---

## 核心转换函数详解

### 函数：`process_single_json(labelme, image_id)`

这是转换的核心函数，负责将一个Labelme格式的标注转换为COCO格式的annotations。

#### 步骤1：遍历所有标注，找到矩形框（rectangle）

```python
for each_ann in labelme['shapes']:
    if each_ann['shape_type'] == 'rectangle':
        # 处理这个矩形框
```

**作用**：每个rectangle代表一个目标对象（如一个人），是COCO格式中一个annotation的基础。

#### 步骤2：计算边界框（bbox）

```python
# 获取矩形框的两个对角点
bbox_left_top_x = min(int(each_ann['points'][0][0]), int(each_ann['points'][1][0]))
bbox_left_top_y = min(int(each_ann['points'][0][1]), int(each_ann['points'][1][1]))
bbox_right_bottom_x = max(int(each_ann['points'][0][0]), int(each_ann['points'][1][0]))
bbox_right_bottom_y = max(int(each_ann['points'][0][1]), int(each_ann['points'][1][1]))

# 计算宽度和高度
bbox_w = bbox_right_bottom_x - bbox_left_top_x
bbox_h = bbox_right_bottom_y - bbox_left_top_y

# COCO格式：左上角坐标 + 宽度 + 高度
bbox_dict['bbox'] = [bbox_left_top_x, bbox_left_top_y, bbox_w, bbox_h]
bbox_dict['area'] = bbox_w * bbox_h
```

**关键点**：
- Labelme的rectangle可能不是标准格式（左上-右下），需要取min/max确保正确
- COCO格式要求：`[左上角x, 左上角y, 宽度, 高度]`

#### 步骤3：匹配分割多边形（polygon）

```python
for each_ann in labelme['shapes']:
    if each_ann['shape_type'] == 'polygon':
        first_x = each_ann['points'][0][0]
        first_y = each_ann['points'][0][1]
        # 判断polygon是否在当前rectangle内部
        if (first_x > bbox_left_top_x) & (first_x < bbox_right_bottom_x) & 
           (first_y < bbox_right_bottom_y) & (first_y > bbox_left_top_y):
            # 将坐标保留两位小数
            bbox_dict['segmentation'] = list(map(
                lambda x: list(map(lambda y: round(y, 2), x)), 
                each_ann['points']
            ))
```

**匹配逻辑**：
- 通过判断polygon的第一个点是否在rectangle内部来匹配
- 一个rectangle对应一个polygon（如果有的话）
- segmentation格式：`[[x1, y1, x2, y2, x3, y3, ...]]`（一维数组）

#### 步骤4：匹配关键点（point）

```python
bbox_keypoints_dict = {}
for each_ann in labelme['shapes']:
    if each_ann['shape_type'] == 'point':
        x = int(each_ann['points'][0][0])
        y = int(each_ann['points'][0][1])
        label = each_ann['label']
        # 判断关键点是否在当前rectangle内部
        if (x > bbox_left_top_x) & (x < bbox_right_bottom_x) & 
           (y < bbox_right_bottom_y) & (y > bbox_left_top_y):
            bbox_keypoints_dict[label] = [x, y]
```

**匹配逻辑**：
- 通过判断point的坐标是否在rectangle内部来匹配
- 一个rectangle可以包含多个关键点
- 使用字典存储：`{关键点名称: [x, y]}`

#### 步骤5：按类别顺序排列关键点

```python
bbox_dict['keypoints'] = []
for each_class in class_list['keypoints']:  # ['angle_30', 'angle_60', 'angle_90']
    if each_class in bbox_keypoints_dict:
        # 存在：添加坐标 + 可见性标志
        bbox_dict['keypoints'].append(bbox_keypoints_dict[each_class][0])  # x
        bbox_dict['keypoints'].append(bbox_keypoints_dict[each_class][1])  # y
        bbox_dict['keypoints'].append(2)  # 可见性：2=可见，1=遮挡，0=不存在
    else:
        # 不存在：填充0
        bbox_dict['keypoints'].append(0)
        bbox_dict['keypoints'].append(0)
        bbox_dict['keypoints'].append(0)
```

**关键点格式**：
- COCO格式：`[x1, y1, v1, x2, y2, v2, ...]`（每3个数字一组）
- 可见性标志：
  - `2`：可见且不遮挡
  - `1`：遮挡但可推测
  - `0`：不存在或完全不可见
- **必须按照categories中定义的keypoints顺序排列**

---

## 数据结构映射

### 图像信息映射

| Labelme字段 | COCO字段 | 说明 |
|------------|---------|------|
| `imagePath` | `file_name` | 图像文件名 |
| `imageHeight` | `height` | 图像高度 |
| `imageWidth` | `width` | 图像宽度 |
| - | `id` | 图像ID（自动递增） |

### 标注信息映射

| Labelme | COCO | 说明 |
|---------|------|------|
| `shapes[].shape_type == 'rectangle'` | `annotations[].bbox` | 边界框 |
| `shapes[].shape_type == 'polygon'` | `annotations[].segmentation` | 分割掩码 |
| `shapes[].shape_type == 'point'` | `annotations[].keypoints` | 关键点坐标 |
| - | `annotations[].category_id` | 类别ID（固定为1） |
| - | `annotations[].iscrowd` | 是否拥挤（固定为0） |
| - | `annotations[].num_keypoints` | 关键点数量 |
| - | `annotations[].id` | 标注ID（自动递增） |
| - | `annotations[].image_id` | 对应的图像ID |

---

## 关键点处理逻辑

### 关键点匹配规则

1. **空间匹配**：关键点必须位于对应的rectangle内部
   ```python
   if (x > bbox_left_top_x) & (x < bbox_right_bottom_x) & 
      (y < bbox_right_bottom_y) & (y > bbox_left_top_y):
   ```

2. **名称匹配**：关键点的label必须与categories中定义的keypoints名称一致
   ```python
   class_list['keypoints'] = ['angle_30', 'angle_60', 'angle_90']
   # 只有label为这些名称的点才会被识别
   ```

3. **顺序排列**：必须按照categories中定义的顺序排列
   ```python
   # 如果定义顺序是 ['angle_30', 'angle_60', 'angle_90']
   # 那么keypoints数组必须是：[x30, y30, v30, x60, y60, v60, x90, y90, v90]
   ```

### 可见性处理

脚本中**硬编码可见性为2**（可见不遮挡）：
```python
bbox_dict['keypoints'].append(2)  # 2-可见不遮挡 1-遮挡 0-没有点
```

**改进建议**：可以从Labelme的`description`字段读取可见性：
```python
# 如果Labelme格式中有description字段
visibility = int(each_ann.get('description', '2'))
bbox_dict['keypoints'].append(visibility)
```

---

## 完整转换示例

### 输入：Labelme格式
```json
{
  "imagePath": "DSC_0281.jpg",
  "imageHeight": 480,
  "imageWidth": 640,
  "shapes": [
    {
      "label": "sjb_rect",
      "shape_type": "rectangle",
      "points": [[100, 100], [200, 200]]
    },
    {
      "label": "angle_30",
      "shape_type": "point",
      "points": [[150, 120]]
    },
    {
      "label": "angle_60",
      "shape_type": "point",
      "points": [[150, 150]]
    },
    {
      "label": "polygon",
      "shape_type": "polygon",
      "points": [[110, 110], [190, 110], [190, 190], [110, 190]]
    }
  ]
}
```

### 输出：COCO格式
```json
{
  "categories": [
    {
      "supercategory": "sjb_rect",
      "id": 1,
      "name": "sjb_rect",
      "keypoints": ["angle_30", "angle_60", "angle_90"],
      "skeleton": [[0,1], [0,2], [1,2]]
    }
  ],
  "images": [
    {
      "file_name": "DSC_0281.jpg",
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
      "bbox": [100, 100, 100, 100],
      "area": 10000,
      "iscrowd": 0,
      "segmentation": [[110.0, 110.0, 190.0, 110.0, 190.0, 190.0, 110.0, 190.0]],
      "num_keypoints": 2,
      "keypoints": [150, 120, 2, 150, 150, 2, 0, 0, 0]
    }
  ]
}
```

---

## 注意事项

1. **关键点顺序**：必须与categories中定义的顺序完全一致
2. **坐标匹配**：使用简单的边界框内判断，可能不够精确
3. **可见性**：当前脚本固定为2，可能需要从Labelme读取
4. **多边形匹配**：只匹配第一个点在矩形内的polygon，可能有多个polygon的情况
5. **ID管理**：使用全局变量`IMG_ID`和`ANN_ID`，确保ID唯一性

---

## 总结

转换的核心思想：
1. **以rectangle为中心**：每个rectangle生成一个COCO annotation
2. **空间匹配**：通过坐标判断polygon和point是否属于该rectangle
3. **顺序排列**：关键点必须按照预定义的顺序排列
4. **格式转换**：将Labelme的灵活格式转换为COCO的标准化格式

这个转换脚本适用于**目标检测 + 关键点检测 + 分割**的联合任务。

