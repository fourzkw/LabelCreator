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

## 模型下载

如果本地没有模型文件，训练时会自动从 [Ultralytics GitHub](https://github.com/ultralytics/assets/releases) 下载。

为了避免每次训练都下载，建议：
1. 手动下载常用模型到此目录
2. 或从 `~/.cache/ultralytics/` (Linux/Mac) 或 `%USERPROFILE%\.cache\ultralytics\` (Windows) 拷贝已下载的模型

## 使用示例

训练时，程序会自动在此目录查找模型：

```python
# 配置文件中设置
{
    "model_type": "yolov8n",
    "pretrained": true,
    ...
}
```

程序会自动加载 `pretrained_models/yolov8n.pt`（如果存在）。

## 注意事项

- 模型文件较大（几MB到几百MB），已在 `.gitignore` 中忽略
- 不建议将模型文件提交到版本控制系统
- 定期清理不再使用的模型以节省磁盘空间

