# 视频预处理流程分析

## 问题描述

- `extract_frames.py` 提取的图片 → `inference_images.py` 推理：**效果好** ✅
- `inference_video.py` 直接提取帧推理：**效果差一点** ⚠️

## 关键差异分析

### 1. 数据流程对比

#### extract_frames.py → inference_images.py 流程：
```
视频 → cv2.VideoCapture.read() → cv2.imwrite(JPEG, 默认质量) 
→ 保存为文件 → inference_images.py → inference_bottomup(文件路径)
→ LoadImage(文件路径) → mmcv.imread() → pipeline处理
```

#### inference_video.py 流程（不使用临时文件）：
```
视频 → cv2.VideoCapture.read() → BGR转RGB → numpy数组
→ inference_bottomup(numpy数组) → LoadImage(numpy数组) → 直接使用 → pipeline处理
```

#### inference_video.py 流程（使用临时文件）：
```
视频 → cv2.VideoCapture.read() → cv2.imwrite(PNG/JPEG) 
→ 保存为文件 → inference_bottomup(文件路径)
→ LoadImage(文件路径) → mmcv.imread() → pipeline处理
```

### 2. 预处理 Pipeline 步骤

根据配置文件，`inference_bottomup` 使用的 pipeline 包括：

```python
val_pipeline = [
    dict(type='LoadImage'),              # 步骤1: 加载图片
    dict(
        type='BottomupResize',           # 步骤2: 调整大小
        input_size=codec['input_size'],
        size_factor=32,
        resize_mode='expand'),
    dict(
        type='PackPoseInputs',          # 步骤3: 打包输入
        meta_keys=(...))
]
```

### 3. LoadImage 的处理差异

**文件路径方式（extract_frames.py → inference_images.py）：**
```python
# inference_bottomup(model, str(img_path))
data_info = dict(img_path=img)  # 文件路径
# LoadImage.transform() 会调用：
# - mmcv.imread(img_path)  # 从文件读取
# - 可能使用不同的解码器/参数
```

**numpy数组方式（inference_video.py 直接方式）：**
```python
# inference_bottomup(model, frame_rgb)
data_info = dict(img=img)  # numpy数组
# LoadImage.transform() 会：
# - 直接使用 numpy 数组
# - 只做类型转换（如果需要float32）
# - 设置 img_shape, ori_shape
```

### 4. 可能的质量差异来源

#### A. 图像读取方式差异

**mmcv.imread() 可能：**
- 使用不同的解码后端（cv2, PIL等）
- 应用不同的颜色空间处理
- 可能有额外的预处理步骤

**直接使用numpy数组：**
- 完全依赖传入的数组
- 没有额外的解码/处理

#### B. JPEG 保存质量

**extract_frames.py：**
```python
cv2.imwrite(str(save_path), frame)  # 默认JPEG质量（通常是95）
```

**inference_video.py（临时文件模式）：**
```python
# 默认PNG（无损）
cv2.imwrite(str(temp_img_path), frame)  # PNG
# 或
cv2.imwrite(str(temp_img_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 100])  # JPEG质量100
```

**关键发现：**
- `extract_frames.py` 使用默认JPEG质量（可能是95）
- 如果 `inference_video.py` 使用PNG，理论上应该更好
- 但如果效果还是差，说明问题不在保存格式

#### C. 颜色空间处理

**extract_frames.py：**
```python
frame = cap.read()  # BGR格式
cv2.imwrite(..., frame)  # 直接保存BGR
# 读取时：mmcv.imread() → 可能内部转换为RGB
```

**inference_video.py（直接方式）：**
```python
frame = cap.read()  # BGR格式
frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # 手动转换
inference_bottomup(model, frame_rgb)  # 传入RGB
```

**inference_video.py（临时文件方式）：**
```python
frame = cap.read()  # BGR格式
cv2.imwrite(..., frame)  # 保存BGR
# 读取时：mmcv.imread() → 可能内部转换为RGB
```

### 5. BottomupResize 的处理

`BottomupResize` 会对图像进行resize，可能使用不同的插值方法：
- 文件读取的图像：可能经过mmcv的resize处理
- numpy数组：直接resize

## 可能的问题根源

### 假设1：mmcv.imread() 的额外处理

`mmcv.imread()` 可能：
1. 使用不同的解码器
2. 应用颜色空间转换
3. 进行额外的预处理

### 假设2：图像质量差异

即使使用临时文件，如果：
- 视频帧本身质量差（压缩损失）
- 保存/读取过程中有损失
- 都会影响最终结果

### 假设3：Pipeline处理的细微差异

虽然pipeline相同，但：
- 文件路径方式：`LoadImage` → `mmcv.imread()` → 可能有额外处理
- numpy数组方式：`LoadImage` → 直接使用数组 → 处理更直接

## 验证方法

### 方法1：对比两种方式读取的图片

```python
# 方式1：extract_frames.py 保存的图片
img1 = mmcv.imread('extracted_frame.jpg')

# 方式2：inference_video.py 临时保存的图片
img2 = mmcv.imread('temp_frame.png')

# 对比差异
diff = np.abs(img1.astype(float) - img2.astype(float))
print(f"MSE: {np.mean(diff**2)}")
```

### 方法2：检查LoadImage的处理

在 `LoadImage.transform()` 中添加日志，查看：
- 文件路径方式：实际读取的图像属性
- numpy数组方式：直接使用的数组属性

### 方法3：统一使用文件路径方式

确保 `inference_video.py` 使用 `--use-temp-files`，并且：
- 使用相同的保存格式（JPEG质量95）
- 使用相同的文件路径方式

## 建议的解决方案

### 方案1：统一使用文件路径方式（推荐）

修改 `inference_video.py`，默认使用临时文件：
```python
# 默认启用临时文件
use_temp_files = True
temp_format = 'jpg'  # 匹配 extract_frames.py
jpeg_quality = 95   # 匹配默认质量
```

### 方案2：检查mmcv.imread()的处理

对比 `mmcv.imread()` 和直接使用numpy数组的差异：
- 颜色空间
- 数据类型
- 图像属性

### 方案3：改进extract_frames.py

如果发现JPEG质量是问题，可以：
- 提高JPEG质量到100
- 或使用PNG格式

## 总结

**关键发现：**
1. 两种方式都使用相同的pipeline，但输入来源不同
2. `LoadImage` 对文件路径和numpy数组的处理可能有差异
3. `mmcv.imread()` 可能有额外的预处理步骤

**建议：**
1. 使用 `--use-temp-files` 并匹配 `extract_frames.py` 的保存方式
2. 对比两种方式读取的图像，找出具体差异
3. 如果差异很小，可能是视频帧本身的质量问题

