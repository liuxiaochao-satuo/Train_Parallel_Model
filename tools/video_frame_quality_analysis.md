# 视频帧质量诊断报告

## 发现的问题

### 1. **视频解码质量损失** ⚠️ **最可能的原因**

**问题描述：**
- `cv2.VideoCapture.read()` 读取的视频帧可能经过压缩编码（如 H.264, H.265）
- 视频压缩会引入：
  - 块状伪影（blocking artifacts）
  - 运动模糊
  - 细节丢失
  - 颜色量化损失

**证据：**
- 保存的图片（未压缩或高质量压缩）推理效果更好
- 视频帧和保存图片在像素级别存在差异

### 2. **图像读取方式差异**

**图片推理脚本：**
```python
results = inference_bottomup(model, str(img_path))  # 文件路径
# 内部通过 LoadImage -> LoadImageFromFile -> mmcv.imfrombytes 读取
```

**视频推理脚本：**
```python
frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # numpy数组
results = inference_bottomup(model, frame_rgb)
# 内部直接使用 numpy 数组，跳过文件读取步骤
```

**可能的影响：**
- `mmcv.imfrombytes` 可能使用不同的解码参数
- 文件读取可能有额外的预处理步骤

### 3. **颜色空间处理**

**当前实现：**
```python
frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
```

**检查点：**
- ✅ BGR->RGB 转换是正确的
- ⚠️ 但视频帧本身可能已经经过颜色空间转换（YUV->RGB）

### 4. **数据类型和连续性**

**当前实现：**
```python
if frame.dtype != np.uint8:
    frame = frame.astype(np.uint8)
if not frame_rgb.flags['C_CONTIGUOUS']:
    frame_rgb = np.ascontiguousarray(frame_rgb)
```

**状态：** ✅ 已正确处理

## 解决方案

### 方案1：临时保存帧为图片文件（推荐用于诊断）

**优点：**
- 完全匹配图片推理的处理流程
- 可以验证是否是视频解码的问题

**缺点：**
- 性能开销（I/O操作）
- 需要临时存储空间

### 方案2：改进视频帧读取方式

**改进点：**
1. 使用更高质量的视频解码设置
2. 尝试不同的视频读取后端
3. 添加去块滤波（deblocking filter）

### 方案3：混合方案

- 对于关键帧，使用临时文件方式
- 对于普通帧，使用直接读取方式

## 诊断步骤

1. 运行诊断脚本：
```bash
python tools/diagnose_video_frame_quality.py \
  your_video.mp4 \
  --frame-idx 100 \
  --saved-image /path/to/saved_frame_100.jpg \
  --output-dir diagnose_output
```

2. 检查输出：
   - 查看差异图（difference.jpg）
   - 检查 MSE/MAE 值
   - 对比像素级别的差异

3. 如果差异很大（MSE > 100），说明视频压缩是主要问题

## 建议的改进代码

见 `inference_video_improved.py`（如果创建）

