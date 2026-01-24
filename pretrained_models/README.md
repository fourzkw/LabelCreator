# 预训练模型目录 (Pretrained Models)

此目录用于存放YOLO预训练模型文件。

## 目录说明

将预训练模型文件放置在此目录下，训练时会优先从本地加载，避免重复下载。

## 支持的模型

### YOLOv8 系列
- `yolov8n.pt` - Nano 版本（最小、最快）
- `yolov8s.pt` - Small 版本
- `yolov8m.pt` - Medium 版本
- `yolov8l.pt` - Large 版本
- `yolov8x.pt` - XLarge 版本（最大、最准确）

### YOLOv8 姿态检测
- `yolov8n-pose.pt` - Nano 姿态检测
- `yolov8s-pose.pt` - Small 姿态检测
- `yolov8m-pose.pt` - Medium 姿态检测
- `yolov8l-pose.pt` - Large 姿态检测
- `yolov8x-pose.pt` - XLarge 姿态检测

### YOLOv11 系列
- `yolo11n.pt` - Nano 版本
- `yolo11s.pt` - Small 版本
- `yolo11m.pt` - Medium 版本
- `yolo11l.pt` - Large 版本
- `yolo11x.pt` - XLarge 版本

### YOLO26 系列
- `yolo26n.pt` - Nano 版本（更快、更小、更适合边缘设备）
- `yolo26s.pt` - Small 版本
- `yolo26m.pt` - Medium 版本
- `yolo26l.pt` - Large 版本
- `yolo26x.pt` - XLarge 版本

### YOLO26 姿态检测
- `yolo26n-pose.pt` - Nano 姿态检测
- `yolo26s-pose.pt` - Small 姿态检测
- `yolo26m-pose.pt` - Medium 姿态检测
- `yolo26l-pose.pt` - Large 姿态检测
- `yolo26x-pose.pt` - XLarge 姿态检测

## 模型特点对比

### YOLOv8 vs YOLO11 vs YOLO26

| 特性 | YOLOv8 | YOLO11 | YOLO26 |
|-----|--------|--------|--------|
| 速度 | ⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| 精度 | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ |
| 模型大小 | 中等 | 中等 | 较小 |
| 适用场景 | 通用 | 通用/云端 | 边缘设备 |
| CPU推理 | 较慢 | 较慢 | 快 |
| GPU推理 | 快 | 快 | 非常快 |

**推荐使用场景**：
- **YOLO26**: 适合嵌入式设备、实时检测、资源受限环境
- **YOLO11**: 适合云端部署、高性能服务器
- **YOLOv8**: 平衡的选择，适合大多数场景

## 模型下载

### 方式一：通过程序自动下载
在"设置 → 模型设置"中首次选择模型时，程序会自动从 [Ultralytics](https://github.com/ultralytics/assets/releases) 下载。

### 方式二：手动下载
1. 访问 [Ultralytics Releases](https://github.com/ultralytics/assets/releases)
2. 下载所需模型文件
3. 将文件放入此目录

### 方式三：复制缓存
从 Ultralytics 缓存目录复制已下载的模型：
- **Linux/Mac**: `~/.cache/ultralytics/`
- **Windows**: `%USERPROFILE%\.cache\ultralytics\`

## 使用示例

### 1. 在图形界面中使用

1. 打开程序，点击 **"设置 → 模型设置"**
2. 选择模型版本（YOLOv8、YOLO11 或 YOLO26）
3. 选择模型格式（.pt 或 .onnx）
4. 点击"浏览"选择模型文件
5. 调整置信度阈值和其他参数
6. 点击"保存"

### 2. 使用 YOLO26 进行自动标注

```python
from ultralytics import YOLO

# 加载 YOLO26 模型
model = YOLO('pretrained_models/yolo26n.pt')

# 对图像进行预测
results = model.predict('image.jpg', conf=0.5, iou=0.45)

# YOLO26 特别适合：
# - 实时视频流检测
# - 移动设备部署
# - 资源受限环境
```

### 3. 训练时使用预训练模型

训练时，程序会自动在此目录查找模型：

```python
# 配置文件中设置
{
    "model_version": "yolo26",
    "model_size": "n",
    "pretrained": true,
    ...
}
```

程序会自动加载 `pretrained_models/yolo26n.pt`（如果存在）。

## 注意事项

- 模型文件较大（几MB到几百MB），已在 `.gitignore` 中忽略
- 不建议将模型文件提交到版本控制系统
- 定期清理不再使用的模型以节省磁盘空间

