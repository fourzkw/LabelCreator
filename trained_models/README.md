# 训练输出模型目录 (Trained Models)

此目录用于存放训练完成后产出的模型文件。

## 目录说明

建议将训练好的最佳模型拷贝到此目录，便于后续使用和管理。

## 常见文件

- `best.pt` - 训练过程中验证集表现最好的模型
- `last.pt` - 最后一个epoch的模型
- `*.onnx` - 转换后的ONNX格式模型（用于部署）

## 模型命名建议

为便于管理，建议使用有意义的文件名：

```
<项目名>_<模型版本>_<数据集>_<指标>.pt
```

示例：
- `fruit_detection_yolov8n_v1_map95.pt`
- `pose_estimation_yolov8m_v2_kpts.pt`
- `custom_object_yolo11n_final.pt`

## 使用训练后的模型

### 1. 继续训练（微调）

```json
{
    "use_custom_model": true,
    "custom_model_path": "trained_models/best.pt",
    ...
}
```

### 2. 用于推理预测

在主窗口的"模型预测设置"中选择训练好的模型：
- 菜单 -> 设置 -> 模型预测设置
- 选择 `trained_models/best.pt`

### 3. 转换为ONNX

在主窗口选择：
- 菜单 -> 工具 -> PT模型转ONNX
- 选择输入模型：`trained_models/best.pt`
- 设置输出路径

## 模型备份建议

重要的模型文件应该：
1. 备份到云存储或外部硬盘
2. 记录模型的训练参数和性能指标
3. 保留训练日志和配置文件

## 目录结构示例

```
trained_models/
├── README.md
├── project_v1/
│   ├── best.pt
│   ├── last.pt
│   ├── training_config.json
│   └── metrics.txt
├── project_v2/
│   ├── best.pt
│   └── best.onnx
└── ...
```

## 注意事项

- 模型文件较大，已在 `.gitignore` 中忽略
- 定期清理旧版本模型以节省磁盘空间
- 重要模型记得备份和添加版本说明

