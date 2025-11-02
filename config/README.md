# 配置文件目录 (Configuration Files)

此目录存放项目的所有配置文件。

## 文件说明

### 1. settings.json
**主程序配置文件**

用于存储主程序的用户设置和首选项。

**配置内容：**
- 快捷键设置
- 界面偏好设置
- 最近使用的路径

**示例结构：**
```json
{
    "shortcuts": {
        "save_current": "Ctrl+S",
        "save_all": "Ctrl+Shift+S",
        "prev_image": "Left",
        "next_image": "Right",
        ...
    }
}
```

**说明：**
- 文件会在首次运行时自动创建
- 可以通过"设置 -> 首选项"菜单修改
- 删除此文件将恢复所有默认设置

---

### 2. yolo_train_settings.json
**YOLO训练配置文件**

用于存储YOLO模型训练的所有参数。

**配置内容：**
- 数据集路径
- 模型选择（yolov8n/yolov8s/yolov11等）
- 训练参数（轮数、批次大小等）
- 优化器设置
- 数据增强参数

**示例结构：**
```json
{
    "yaml_path": "path/to/data.yaml",
    "model_type": "yolov8n",
    "pretrained": true,
    "epochs": 100,
    "batch_size": 16,
    "img_size": 640,
    "optimizer": "SGD",
    "lr0": 0.01,
    "augment": true,
    ...
}
```

**使用方式：**
1. 在主窗口选择"训练 -> YOLO模型训练器"
2. 配置训练参数
3. 点击"保存设置"按钮，参数会保存到此文件
4. 点击"开始训练"会读取此文件进行训练

---

## 配置文件管理

### 备份配置
重要的配置建议定期备份：
```bash
cp config/yolo_train_settings.json config/yolo_train_settings.backup.json
```

### 恢复默认配置
删除对应的配置文件，程序会在下次运行时重新创建默认配置。

### 共享配置
可以将配置文件复制给团队成员，确保使用相同的训练参数：
```bash
# 导出配置
cp config/yolo_train_settings.json ~/shared/

# 导入配置
cp ~/shared/yolo_train_settings.json config/
```

---

## 配置文件位置

配置文件统一存放在项目根目录下的 `config/` 文件夹中：

```
label-creator/
├── config/
│   ├── README.md                    # 本说明文件
│   ├── settings.json                # 主程序配置
│   └── yolo_train_settings.json     # 训练配置
├── pretrained_models/               # 预训练模型
├── trained_models/                  # 训练输出
└── ...
```

---

## 配置优先级

1. **用户配置** - `config/` 目录中的文件（优先级最高）
2. **程序默认值** - 代码中定义的默认配置
3. **系统配置** - QSettings 存储的系统级配置

---

## 注意事项

1. **编码格式**：配置文件使用 UTF-8 编码
2. **JSON格式**：确保JSON格式正确，避免手动编辑时出错
3. **路径分隔符**：Windows使用 `\\` 或 `/`，建议使用 `/`
4. **版本控制**：
   - ✅ 可以提交默认配置模板
   - ❌ 不建议提交包含个人路径的配置

---

## 常见问题

### Q: 配置文件丢失怎么办？
A: 删除损坏的配置文件，程序会自动创建新的默认配置。

### Q: 修改配置后不生效？
A: 
1. 检查JSON格式是否正确
2. 重启程序
3. 查看日志文件 `app.log` 中的错误信息

### Q: 如何重置所有设置？
A: 删除 `config/` 目录下的所有配置文件，程序会重新创建默认配置。

### Q: 配置文件可以手动编辑吗？
A: 可以，但需要：
1. 确保JSON格式正确
2. 备份原文件
3. 注意路径分隔符
4. 建议使用程序界面修改

---

## 配置示例

### 基础训练配置
```json
{
    "model_type": "yolov8n",
    "pretrained": true,
    "epochs": 50,
    "batch_size": 16,
    "img_size": 640
}
```

### 高精度训练配置
```json
{
    "model_type": "yolov8l",
    "pretrained": true,
    "epochs": 300,
    "batch_size": 8,
    "img_size": 1280,
    "augment": true
}
```

### 从头训练配置
```json
{
    "model_type": "yolov8n",
    "pretrained": false,
    "epochs": 300,
    "batch_size": 32,
    "img_size": 640
}
```

---

## 相关文档

- [模型管理说明](../pretrained_models/README.md)
- [训练输出管理](../trained_models/README.md)
- [Ultralytics YOLO 文档](https://docs.ultralytics.com/)

