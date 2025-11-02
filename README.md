# YOLO标注工具 (YOLO Labeling Tool)

![Python 3.9+](https://img.shields.io/badge/Python-3.9%2B-blue)
![PyQt5 5.15.0+](https://img.shields.io/badge/PyQt5-5.15.0%2B-green)
![Ultralytics 8.0.0+](https://img.shields.io/badge/Ultralytics-8.0.0%2B-orange)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

## 📖 项目概述 (Project Overview)

这是一个功能完备的YOLO目标检测数据集标注、管理和训练工具，提供直观的图形界面，帮助用户高效地完成从数据标注到模型训练的全流程工作。

### 核心特性
- 🎯 **手动标注**：直观的可视化界面，支持边界框和关键点标注
- 🤖 **自动标注**：集成YOLOv8/YOLO11模型，实现智能预测和批量标注
- 🔧 **模型训练**：内置训练器，支持检测模型和姿态检测模型训练
- 🔄 **模型转换**：一键将PyTorch模型转换为ONNX格式，便于跨平台部署
- 📊 **数据管理**：数据集划分、类别管理等完整的数据处理工具链
- 🎨 **特征点支持**：完整支持关键点检测任务的标注和训练

## 📦 安装指南 (Installation Guide)

### 系统要求
- **Python**: 3.9（推荐，已知Python 3.13存在onnx依赖冲突）
- **操作系统**: Windows / Linux / MacOS
- **Windows系统**: 需要 [Visual Studio C++ Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/)

### 快速开始

#### 1. 克隆仓库
```bash
git clone <repository-url>
cd label-creator
```

#### 2. 安装依赖
   ```bash
   pip install -r requirements.txt
   ```

> **注意**: PyTorch需要根据系统CUDA版本单独安装，请访问 [PyTorch官网](https://pytorch.org/get-started/locally/) 获取正确的安装命令。

#### 3. 初始化配置
首次运行前，复制示例配置文件：
```bash
# Windows PowerShell
Copy-Item config\settings.json.example config\settings.json
Copy-Item config\yolo_train_settings.json.example config\yolo_train_settings.json

# Linux/MacOS
cp config/settings.json.example config/settings.json
cp config/yolo_train_settings.json.example config/yolo_train_settings.json
```

#### 4. 运行程序
   ```bash
   python main.py
   ```

### 训练环境配置（可选）
如需使用模型训练功能，请确保已安装：
- **Conda环境管理器**
- **CUDA工具包**（GPU训练必需）
- **PyTorch**（与CUDA版本匹配）
- 其他依赖见 `requirements.txt`

#### PyTorch安装示例
```bash
# CUDA 11.8
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# CUDA 12.1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# CPU版本
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

## 📂 项目结构 (Project Structure)

```
label-creator/
├── config/                      # 配置文件目录
│   ├── README.md               # 配置文件说明文档
│   ├── settings.json.example   # 主程序配置示例
│   └── yolo_train_settings.json.example  # 训练配置示例
├── doc/                         # 文档和截图
├── i18n/                        # 国际化支持
├── logs/                        # 日志文件目录
├── models/                      # 数据模型定义
│   └── bounding_box.py         # 边界框模型
├── pretrained_models/          # 预训练模型存放目录
│   └── README.md               # 模型管理说明
├── trained_models/             # 训练输出模型目录
│   └── README.md               # 训练输出说明
├── training/                    # 训练模块
│   ├── train_yolo.py           # YOLO训练脚本
│   ├── trainer_dialog.py       # 训练对话框
│   └── trainer_ui.py           # 训练界面
├── ui/                          # 用户界面模块
│   ├── canvas.py               # 画布组件
│   ├── class_manager_dialog.py # 类别管理对话框
│   ├── dataset_split_dialog.py # 数据集划分对话框
│   ├── main_window.py          # 主窗口
│   ├── model_converter_dialog.py  # 模型转换对话框
│   ├── model_inspector_dialog.py  # 模型查看器
│   ├── model_settings_dialog.py   # 模型设置对话框
│   └── settings_dialog.py      # 设置对话框
├── utils/                       # 工具模块
│   ├── logger.py               # 日志工具
│   ├── model_analyzer.py       # 模型分析器
│   ├── model_converter.py      # 模型转换工具
│   ├── settings.py             # 设置管理
│   └── yolo_predictor.py       # YOLO预测器
├── .gitignore                   # Git忽略文件
├── main.py                      # 程序入口
├── requirements.txt             # 依赖列表
├── LICENSE                      # 许可证
└── README.md                    # 项目说明文档
```

## ⚡ 主要功能 (Main Features)

### 🎨 标注功能
- ✅ 支持加载和浏览图像文件夹
- ✅ 手动绘制、编辑和删除边界框
- ✅ 支持多类别标注和类别切换
- ✅ 自动保存标注结果为YOLO格式
- ✅ 丰富的键盘快捷键支持
- ✅ 图像缩放、平移和重置功能
- ✅ 辅助线功能，精准定位目标

### 🎯 特征点标注功能
- ✅ 在边界框内添加关键点
- ✅ 左键单击绘制特征点
- ✅ 左键拖动移动特征点
- ✅ 右键双击删除特征点
- ✅ 完全兼容YOLO关键点检测格式
- ✅ 支持可变数量的关键点

### 🤖 自动标注
- ✅ 集成YOLOv8/YOLO11模型进行智能预测
- ✅ 支持单张图像预测和批量自动标注
- ✅ 可调整置信度阈值和IoU阈值
- ✅ 支持带特征点的模型自动标注
- ✅ 预测结果可手动调整优化

### 🚀 模型训练功能
- ✅ 内置YOLO训练器（支持YOLOv8和YOLO11）
- ✅ 多种模型规模：n/s/m/l/x
- ✅ 预训练权重或自定义模型训练
- ✅ 支持检测模型和姿态检测模型
- ✅ 丰富的训练参数配置：
  - 基本参数：批次大小、图像尺寸、训练轮数
  - 优化器设置：学习率、动量、权重衰减
  - 数据增强：HSV、旋转、平移、缩放、翻转、Mosaic
  - 高级设置：早停策略、学习率调度、设备选择
- ✅ Conda环境管理，一键启动训练
- ✅ 实时训练日志显示
- ✅ 数据集自动划分（训练/验证/测试）

### 🔄 模型转换功能
- ✅ PyTorch (.pt) 转 ONNX 格式
- ✅ 自定义输入尺寸和批次大小
- ✅ 可选ONNX操作集版本
- ✅ 模型简化（onnxslim）
- ✅ 半精度（FP16）转换
- ✅ 图形化界面，操作简便
- ✅ 支持多平台部署优化

### 🔧 其他辅助工具
- ✅ 类别管理系统
- ✅ 数据集划分工具
- ✅ 模型结构查看器
- ✅ 配置文件管理
- ✅ 详细的日志记录系统
- ✅ 键盘快捷键自定义

## 📖 使用指南 (Usage Guide)

### 1️⃣ 数据集准备

#### 加载图像
1. 点击 **"打开文件夹"** 按钮（或 `Ctrl+O`）
2. 选择包含图像的文件夹
3. 系统自动加载支持的图像格式：`.jpg`, `.jpeg`, `.png`, `.bmp`

#### 导入已有标注
- 将YOLO格式的 `.txt` 标注文件与图像放在同一目录
- 标注文件名必须与图像文件名相同（扩展名除外）
- 系统会自动加载并显示已有标注

---

### 2️⃣ 手动标注

#### 绘制边界框
1. 从右侧类别列表选择目标类别
2. 在图像上 **左键拖动** 绘制边界框
3. 松开鼠标自动保存

#### 编辑边界框
- **选择**：单击边界框
- **移动**：拖动边界框
- **调整大小**：拖动边界框角点
- **删除**：选中后按 `Delete` 键

#### 特征点标注
1. 点击 **"特征点模式"** 按钮（或 `Ctrl+K`）
2. 选择一个边界框
3. **左键单击**：添加特征点
4. **左键拖动**：移动特征点
5. **右键双击**：删除特征点

#### 视图操作
- **平移**：按住鼠标中键拖动
- **缩放**：滚轮滚动 或 `Ctrl +/-`
- **重置**：`Ctrl+0`

#### 快捷键列表
| 功能 | 快捷键 |
|------|--------|
| 上一张图像 | `←` |
| 下一张图像 | `→` |
| 保存当前标注 | `Ctrl+S` |
| 保存所有标注 | `Ctrl+Shift+S` |
| 删除边界框 | `Delete` |
| 自动标注当前图像 | `Ctrl+A` |
| 批量自动标注 | `Ctrl+Shift+A` |
| 特征点模式 | `Ctrl+K` |
| 放大 | `Ctrl++` |
| 缩小 | `Ctrl+-` |
| 重置缩放 | `Ctrl+0` |
| 打开文件夹 | `Ctrl+O` |
| 退出程序 | `Ctrl+Q` |

---

### 3️⃣ 自动标注

#### 配置模型
1. 在 **"设置 → 模型设置"** 中配置预测模型路径
2. 推荐使用 `pretrained_models/` 目录存放模型文件
3. 支持的模型：YOLOv8、YOLO11（检测/姿态）

#### 单张预测
1. 选择要标注的图像
2. 点击 **"自动标注"** 按钮（或 `Ctrl+A`）
3. 调整置信度阈值和IoU阈值
4. 点击 **"确认"** 开始预测
5. 检查结果并手动调整

#### 批量标注
1. 点击 **"批量自动标注"** 按钮（或 `Ctrl+Shift+A`）
2. 设置阈值参数
3. 点击 **"开始"** 自动处理所有图像
4. 完成后逐张检查和修正

> **提示**：自动标注结果需要人工审核和修正以确保质量。

---

### 4️⃣ 类别管理

1. 点击 **"类别管理"** 按钮
2. **添加类别**：输入类别名称，点击 "添加"
3. **删除类别**：选中类别，点击 "删除"
4. **编辑类别**：双击类别名称直接修改
5. 点击 **"保存"** 应用更改

---

### 5️⃣ 数据集划分

1. 点击 **"工具 → 数据集划分"**
2. 设置以下参数：
   - **源路径**：包含图像和标注的文件夹
   - **目标路径**：输出划分后的数据集
   - **训练集比例**：如 70%
   - **验证集比例**：如 20%
   - **测试集比例**：如 10%
   - **是否包含特征点**：根据数据类型选择
3. 点击 **"开始划分"**
4. 系统自动生成 `train/`, `val/`, `test/` 子目录

---

### 6️⃣ 模型训练

#### 训练前准备
1. 准备好已标注的数据集
2. 完成数据集划分
3. 确保已安装Conda并配置好训练环境

#### 开始训练
1. 点击 **"训练"** 按钮
2. 配置训练参数：

**基本设置**
- 数据集配置文件（`data.yaml`）路径
- 模型版本：YOLOv8 / YOLO11
- 模型类型：普通检测 / 姿态检测
- 模型规模：n / s / m / l / x
- 是否使用预训练权重

**训练参数**
- Epochs：训练轮数（如100）
- Batch Size：批次大小（如16）
- Image Size：输入图像尺寸（如640）
- Device：训练设备（0/cpu）

**优化器设置**
- Optimizer：SGD / Adam / AdamW
- Learning Rate：初始学习率（如0.01）
- Momentum：动量（如0.94）

**数据增强**
- HSV调整、旋转、平移、缩放
- 左右翻转、Mosaic增强等

3. 选择Conda环境
4. 点击 **"开始训练"**
5. 实时查看训练日志和进度

#### 训练输出
- 模型权重保存在 `trained_models/` 或指定目录
- 最佳权重：`best.pt`
- 最后权重：`last.pt`

---

### 7️⃣ 模型转换

#### PT 转 ONNX
1. 点击 **"工具 → PT模型转ONNX"**
2. 选择输入的 `.pt` 模型文件
3. 设置输出路径和文件名
4. 配置转换参数：
   - **输入尺寸**：宽度和高度（如640×640）
   - **Batch Size**：批次大小（通常为1）
   - **ONNX Opset**：操作集版本（如11）
   - **简化模型**：启用以减小模型大小
   - **半精度（FP16）**：启用以优化推理速度
5. 点击 **"转换"**
6. 转换完成后可用于ONNX Runtime、TensorRT等推理框架

---

### 8️⃣ 模型查看器

1. 点击 **"工具 → 查看模型结构"**
2. 选择 `.pt` 或 `.onnx` 模型文件
3. 查看模型详细信息：
   - 模型类型和任务
   - 输入/输出形状
   - 参数数量
   - 类别列表
4. 支持一键复制模型信息

## 🏗️ 技术架构 (Technical Architecture)

### 技术栈
- **界面框架**：PyQt5 5.15+
- **深度学习**：Ultralytics (YOLOv8/YOLO11)
- **模型推理**：PyTorch + ONNX Runtime
- **数据处理**：NumPy, Pandas
- **模型转换**：ONNX, onnxslim

### 架构设计
```
┌─────────────────────────────────────────┐
│          UI Layer (PyQt5)               │
│  ┌──────────┐  ┌──────────┐  ┌────────┐│
│  │ Canvas   │  │ Dialogs  │  │Settings││
│  └──────────┘  └──────────┘  └────────┘│
└─────────────────┬───────────────────────┘
                  │
┌─────────────────▼───────────────────────┐
│       Business Logic Layer              │
│  ┌──────────┐  ┌──────────┐  ┌────────┐│
│  │Predictor │  │ Trainer  │  │Analyzer││
│  └──────────┘  └──────────┘  └────────┘│
└─────────────────┬───────────────────────┘
                  │
┌─────────────────▼───────────────────────┐
│         Data Layer                      │
│  ┌──────────┐  ┌──────────┐  ┌────────┐│
│  │ Models   │  │ Settings │  │ Logger ││
│  └──────────┘  └──────────┘  └────────┘│
└─────────────────────────────────────────┘
```

### 模块说明
| 模块 | 功能 | 文件 |
|------|------|------|
| UI层 | 用户界面交互 | `ui/` |
| 业务逻辑层 | 核心功能实现 | `utils/`, `training/` |
| 数据层 | 数据模型和配置 | `models/`, `config/` |
| 国际化 | 多语言支持 | `i18n/` |
| 日志系统 | 运行日志记录 | `logs/` |

### 设计特点
- ✅ **模块化设计**：高内聚低耦合
- ✅ **可扩展性**：易于添加新功能
- ✅ **配置分离**：配置与代码分离
- ✅ **错误处理**：全局异常捕获和日志记录
- ✅ **代码规范**：清晰的注释和文档

---

## ❓ 常见问题 (FAQ)

### 安装相关

<details>
<summary><b>Q: 安装依赖时出现错误？</b></summary>

**A**: 常见解决方案：
1. 确保Python版本为3.9（推荐）
2. 升级pip：`python -m pip install --upgrade pip`
3. Windows系统需安装Visual Studio C++ Build Tools
4. 单独安装PyTorch：访问 [PyTorch官网](https://pytorch.org/)
</details>

<details>
<summary><b>Q: Python 3.13兼容性问题？</b></summary>

**A**: Python 3.13存在onnx依赖冲突，建议使用Python 3.9或3.10。
</details>

---

### 标注相关

<details>
<summary><b>Q: 图像加载失败？</b></summary>

**A**: 
- 检查图像格式是否为 `.jpg`, `.jpeg`, `.png`, `.bmp`
- 确认文件路径不包含特殊字符
- 查看 `logs/app.log` 获取详细错误信息
</details>

<details>
<summary><b>Q: 标注文件保存失败？</b></summary>

**A**:
- 检查文件夹是否有写入权限
- 确认磁盘空间充足
- 尝试以管理员权限运行程序
</details>

<details>
<summary><b>Q: 特征点无法添加？</b></summary>

**A**:
1. 确认已启用特征点模式（`Ctrl+K`）
2. 确保已选中一个边界框
3. 在边界框内部单击添加特征点
</details>

---

### 自动标注相关

<details>
<summary><b>Q: 自动标注效果不理想？</b></summary>

**A**:
- 调整置信度阈值（降低以获得更多检测）
- 使用更适合的预训练模型
- 尝试更大规模的模型（如yolov8m、yolov8l）
- 确保模型与数据集类别匹配
</details>

<details>
<summary><b>Q: 找不到模型文件？</b></summary>

**A**:
- 在"设置 → 模型设置"中配置模型路径
- 将模型放入 `pretrained_models/` 目录
- 确认模型文件完整且未损坏
</details>

---

### 训练相关

<details>
<summary><b>Q: 训练时内存不足？</b></summary>

**A**:
- 减小batch_size（如从16降至8）
- 减小img_size（如从640降至416）
- 使用更小的模型（如yolov8n）
- 关闭不必要的程序释放内存
</details>

<details>
<summary><b>Q: Conda环境找不到？</b></summary>

**A**:
1. 确认已安装Anaconda或Miniconda
2. 创建训练环境：
```bash
conda create -n yolo python=3.9
conda activate yolo
pip install ultralytics
```
3. 在训练界面选择对应的conda环境
</details>

<details>
<summary><b>Q: GPU训练不可用？</b></summary>

**A**:
- 检查CUDA是否正确安装：`nvidia-smi`
- 确认PyTorch支持CUDA：
```python
import torch
print(torch.cuda.is_available())
```
- 安装匹配CUDA版本的PyTorch
</details>

---

### 模型转换相关

<details>
<summary><b>Q: 模型转换失败？</b></summary>

**A**:
- 确保安装了onnx和onnxruntime：
```bash
pip install onnx==1.19.0 onnxruntime==1.19.2
```
- 检查模型文件是否完整
- 查看日志文件获取详细错误
- 尝试不启用"简化"和"FP16"选项
</details>

<details>
<summary><b>Q: 转换后的ONNX模型推理结果不一致？</b></summary>

**A**:
- 确认输入尺寸与训练时一致
- FP16转换可能影响精度，尝试使用FP32
- 检查前后处理流程是否一致
</details>

---

### 其他问题

<details>
<summary><b>Q: 如何查看详细日志？</b></summary>

**A**: 日志文件位于 `logs/app.log`，包含所有操作记录和错误信息。
</details>

<details>
<summary><b>Q: 配置文件在哪里？</b></summary>

**A**: 
- 主程序配置：`config/settings.json`
- 训练配置：`config/yolo_train_settings.json`
- 详见 `config/README.md`
</details>

<details>
<summary><b>Q: 如何重置所有设置？</b></summary>

**A**: 删除 `config/` 目录下的配置文件，程序会重新创建默认配置。
</details>

---

## 📝 更新日志 (Changelog)

### v2.3.0 (2025.11.02)
- 🎨 优化项目结构，配置文件统一管理
- 📁 新增 `config/`、`pretrained_models/`、`trained_models/` 目录
- 📄 新增配置文件示例（`.example`）
- 📖 完善项目文档和README
- 🔧 优化配置管理和路径处理

### v2.2.0 (2025.08.23)
- 🐛 修复模型训练已知bug
- 🐛 修复模型格式转换问题
- 🐛 修复模型结构查看功能
- ✨ 新增数据集配置扫描更新功能

### v2.1.0 (2025.07.03)
- ✨ 新增PT模型转ONNX功能
- ⚙️ 支持自定义输入尺寸和操作集版本
- 🎛️ 支持模型简化和FP16精度转换
- 👀 新增模型结构查看器
- 📋 支持一键复制模型信息

### v2.0.0 (2025.06.30)
- 🏗️ 重构项目架构，优化代码结构
- 🐛 修复标签文件读取bug
- 💬 优化代码注释和文档
- 🔧 改进异常处理机制

### v1.9.0 (2025.06.22)
- 🔄 改进数据集划分功能
- ✨ 支持带特征点的数据集划分
- 🎯 支持训练带特征点的姿态检测模型
- ⚙️ 增强训练配置选项

### v1.8.0 (2025.06.21)
- 🐛 修复特征点标注相关bug
- 🖱️ 新增图像拖动功能（鼠标中键）
- 🎨 优化用户交互体验

### v1.7.0 (2025.04.29)
- 🤖 新增带特征点的模型自动标注
- 🎯 新增特征点标注模式
- 🖱️ 支持特征点绘制、移动和删除
- 📐 完整支持YOLO关键点格式

---

## 🤝 贡献指南 (Contributing)

欢迎各种形式的贡献！无论是报告问题、提出建议还是提交代码。

### 如何贡献

#### 报告Bug
1. 在 [GitHub Issues](https://github.com/your-repo/label-creator/issues) 创建新issue
2. 清晰描述问题和复现步骤
3. 提供相关日志（`logs/app.log`）
4. 注明系统环境和Python版本

#### 功能建议
1. 创建Feature Request类型的issue
2. 详细描述期望的功能
3. 说明使用场景和必要性

#### 提交代码
1. Fork本项目
2. 创建特性分支：`git checkout -b feature/AmazingFeature`
3. 提交更改：`git commit -m 'Add some AmazingFeature'`
4. 推送到分支：`git push origin feature/AmazingFeature`
5. 创建Pull Request

### 代码规范
- 遵循PEP 8代码风格
- 添加必要的注释和文档字符串
- 确保新功能有适当的错误处理
- 更新相关文档

### 开发环境设置
```bash
# 克隆仓库
git clone https://github.com/your-repo/label-creator.git
cd label-creator

# 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 安装依赖
pip install -r requirements.txt

# 运行程序
python main.py
```

---

## 📄 许可证 (License)

本项目采用 [MIT License](LICENSE) 开源许可证。

```
MIT License

Copyright (c) 2025 YOLO Label Creator

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

---

## 🌟 致谢 (Acknowledgments)

感谢以下开源项目：
- [Ultralytics YOLO](https://github.com/ultralytics/ultralytics) - YOLO模型实现
- [PyQt5](https://www.riverbankcomputing.com/software/pyqt/) - GUI框架
- [ONNX](https://onnx.ai/) - 模型交换格式

---

## 📞 联系方式 (Contact)

- 📧 Email: your-email@example.com
- 🐛 Issues: [GitHub Issues](https://github.com/your-repo/label-creator/issues)
- 📖 Documentation: [Wiki](https://github.com/your-repo/label-creator/wiki)

---

<div align="center">

### ⭐ 如果这个项目对你有帮助，请给一个Star！⭐

**[↑ 返回顶部](#yolo标注工具-yolo-labeling-tool)**

</div>