# demo_mmpose.py 视频帧处理方式分析

## 关键发现：颜色空间处理的差异！

### demo_mmpose.py 的处理流程

```python
# 1. 从视频读取帧（BGR格式）
ret, raw_frame = cap.read()  # BGR格式的numpy数组

# 2. 直接传入BGR数组给推理函数
pred_instances = self._process_frame(frame, model, visualizer, 0.001)

# 3. _process_frame 内部：
def _process_frame(self, img, pose_estimator, visualizer=None, show_interval=0):
    # 直接传入BGR数组（不做颜色转换）
    batch_results = inference_bottomup(pose_estimator, img)  # img是BGR格式
    
    # 可视化时才转换为RGB
    if isinstance(img, np.ndarray):
        img = mmcv.bgr2rgb(img)  # 仅用于可视化
```

### 关键差异对比

| 脚本 | 传入inference_bottomup的格式 | LoadImage处理 | 后续Pipeline |
|------|---------------------------|--------------|-------------|
| **demo_mmpose.py** | **BGR numpy数组** | 直接使用BGR数组 | 使用BGR格式处理 |
| **inference_video.py** | **RGB numpy数组** | 直接使用RGB数组 | 使用RGB格式处理 |
| **inference_images.py** | **文件路径** | mmcv.imread() → **BGR** | 可能内部转换为RGB |

### 重要发现

#### 1. **颜色空间不一致！** ⚠️

**demo_mmpose.py：**
- 传入 **BGR** 格式的 numpy 数组
- `LoadImage` 直接使用，不做转换
- Pipeline 中的 `BottomupResize` 等处理使用 **BGR** 格式

**inference_video.py：**
- 手动转换为 **RGB** 格式
- `LoadImage` 直接使用 RGB 数组
- Pipeline 中的处理使用 **RGB** 格式

**inference_images.py：**
- 通过文件路径，`mmcv.imread()` 默认读取为 **BGR**
- 但可能在 pipeline 内部有转换

#### 2. mmcv.imread() 的行为

从代码看：
```python
# demo_mmpose.py 可视化时
img = mmcv.imread(img, channel_order='rgb')  # 明确指定RGB

# 但 inference_bottomup 内部使用 LoadImage
# LoadImage → LoadImageFromFile → mmcv.imfrombytes
# 默认可能读取为 BGR
```

### 可能的问题根源

#### 假设：颜色空间不匹配导致效果差异

**如果模型训练时使用的是 RGB 格式：**
- `demo_mmpose.py` 传入 BGR → 模型看到的是错误的颜色通道顺序 → 效果差
- `inference_video.py` 传入 RGB → 模型看到正确的颜色 → 效果应该好
- `inference_images.py` 通过文件路径 → `mmcv.imread()` 可能内部转换为 RGB → 效果好

**但实际情况相反：**
- `demo_mmpose.py` 效果可能好（如果模型训练时用的是BGR）
- `inference_video.py` 效果差（如果传入RGB但模型期望BGR）
- `inference_images.py` 效果好（可能内部处理正确）

### 验证方法

检查模型训练时的数据格式：
1. 查看训练配置中的 `LoadImage` 参数
2. 检查是否有 `channel_order` 设置
3. 对比训练和推理时的颜色空间处理

### 建议的修复

#### 方案1：统一使用BGR格式（匹配demo_mmpose.py）

修改 `inference_video.py`，不进行颜色转换：

```python
# 不转换颜色空间，直接使用BGR
results = inference_bottomup(model, frame)  # frame是BGR格式
```

#### 方案2：检查模型期望的格式

查看训练配置，确认模型期望的颜色空间：
- 如果期望 BGR：使用方案1
- 如果期望 RGB：保持当前方式，但检查其他差异

#### 方案3：使用文件路径方式（最安全）

使用 `--use-temp-files`，让 `LoadImage` 通过文件读取，这样会使用和训练时一致的流程。

## 总结

**关键发现：**
1. `demo_mmpose.py` 直接传入 **BGR** 格式的 numpy 数组
2. `inference_video.py` 手动转换为 **RGB** 格式
3. 这可能导致颜色通道顺序不匹配，影响推理效果

**建议：**
1. 检查模型训练时使用的颜色空间格式
2. 统一使用相同的颜色空间格式
3. 或者使用文件路径方式（`--use-temp-files`），让 pipeline 自动处理

