import sys
import os
import json
import subprocess
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                             QLabel, QLineEdit, QPushButton, QFileDialog, QComboBox,
                             QSpinBox, QDoubleSpinBox, QGroupBox, QFormLayout, QTabWidget,
                             QTextEdit, QCheckBox, QGridLayout, QDialog)
from PyQt5.QtCore import Qt, QSettings

class YoloTrainerUI(QMainWindow):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("YOLOv8 训练器")
        self.setMinimumSize(800, 600)
        
        # 保存设置
        self.settings = QSettings("YoloTrainer", "YoloTrainerApp")
        
        # 创建主窗口部件
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        
        # 主布局
        self.main_layout = QVBoxLayout(self.central_widget)
        
        # 创建选项卡
        self.tabs = QTabWidget()
        self.main_layout.addWidget(self.tabs)
        
        # 创建基本设置选项卡
        self.basic_tab = QWidget()
        self.tabs.addTab(self.basic_tab, "基本设置")
        
        # 创建高级设置选项卡
        self.advanced_tab = QWidget()
        self.tabs.addTab(self.advanced_tab, "高级设置")
        
        # 设置基本选项卡布局
        self.setup_basic_tab()
        
        # 设置高级选项卡布局
        self.setup_advanced_tab()
        
        # 底部按钮区域
        self.button_layout = QHBoxLayout()
        self.save_button = QPushButton("保存设置")
        self.save_button.clicked.connect(self.save_settings)
        self.train_button = QPushButton("开始训练")
        self.train_button.clicked.connect(self.start_training)
        
        self.button_layout.addWidget(self.save_button)
        self.button_layout.addWidget(self.train_button)
        self.main_layout.addLayout(self.button_layout)
        
        # 日志区域
        self.log_group = QGroupBox("日志")
        self.log_layout = QVBoxLayout()
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_layout.addWidget(self.log_text)
        self.log_group.setLayout(self.log_layout)
        self.main_layout.addWidget(self.log_group)
        
        # 加载保存的设置
        self.load_settings()
    
    def setup_basic_tab(self):
        layout = QVBoxLayout(self.basic_tab)
        
        # Conda环境选择
        conda_group = QGroupBox("Conda环境")
        conda_layout = QFormLayout()
        
        self.conda_env_combo = QComboBox()
        self.refresh_conda_button = QPushButton("刷新")
        self.refresh_conda_button.clicked.connect(self.refresh_conda_envs)
        
        conda_env_layout = QHBoxLayout()
        conda_env_layout.addWidget(self.conda_env_combo)
        conda_env_layout.addWidget(self.refresh_conda_button)
        
        conda_layout.addRow("选择Conda环境:", conda_env_layout)
        conda_group.setLayout(conda_layout)
        layout.addWidget(conda_group)
        
        # 数据集和模型设置
        data_group = QGroupBox("数据集和模型")
        data_layout = QFormLayout()
        
        self.yaml_path = QLineEdit()
        self.yaml_browse = QPushButton("浏览...")
        self.yaml_browse.clicked.connect(lambda: self.browse_file(self.yaml_path, "YAML文件 (*.yaml *.yml)"))
        yaml_layout = QHBoxLayout()
        yaml_layout.addWidget(self.yaml_path)
        yaml_layout.addWidget(self.yaml_browse)
        data_layout.addRow("数据集YAML文件:", yaml_layout)
        
        self.model_type = QComboBox()
        self.model_type.addItems(["yolov8n", "yolov8s", "yolov8m", "yolov8l", "yolov8x"])
        data_layout.addRow("模型类型:", self.model_type)
        
        self.pretrained_checkbox = QCheckBox("使用预训练权重")
        self.pretrained_checkbox.setChecked(True)
        data_layout.addRow("", self.pretrained_checkbox)
        
        # 添加自定义预训练模型选择
        self.custom_model_checkbox = QCheckBox("使用自定义预训练模型")
        self.custom_model_checkbox.setChecked(False)
        self.custom_model_checkbox.stateChanged.connect(self.toggle_custom_model)
        data_layout.addRow("", self.custom_model_checkbox)
        
        self.custom_model_path = QLineEdit()
        self.custom_model_path.setEnabled(False)
        self.custom_model_browse = QPushButton("浏览...")
        self.custom_model_browse.setEnabled(False)
        self.custom_model_browse.clicked.connect(lambda: self.browse_file(self.custom_model_path, "PyTorch模型 (*.pt)"))
        custom_model_layout = QHBoxLayout()
        custom_model_layout.addWidget(self.custom_model_path)
        custom_model_layout.addWidget(self.custom_model_browse)
        data_layout.addRow("自定义模型路径:", custom_model_layout)
        
        data_group.setLayout(data_layout)
        layout.addWidget(data_group)
        
        # 基本训练参数
        train_group = QGroupBox("基本训练参数")
        train_layout = QFormLayout()
        
        self.epochs = QSpinBox()
        self.epochs.setRange(1, 1000)
        self.epochs.setValue(100)
        train_layout.addRow("训练轮数 (epochs):", self.epochs)
        
        self.batch_size = QSpinBox()
        self.batch_size.setRange(1, 128)
        self.batch_size.setValue(16)
        train_layout.addRow("批次大小 (batch size):", self.batch_size)
        
        self.img_size = QSpinBox()
        self.img_size.setRange(32, 1280)
        self.img_size.setValue(640)
        self.img_size.setSingleStep(32)
        train_layout.addRow("图像大小 (img size):", self.img_size)
        
        train_group.setLayout(train_layout)
        layout.addWidget(train_group)
        
        # 输出设置
        output_group = QGroupBox("输出设置")
        output_layout = QFormLayout()
        
        self.project_path = QLineEdit()
        self.project_browse = QPushButton("浏览...")
        self.project_browse.clicked.connect(lambda: self.browse_directory(self.project_path))
        project_layout = QHBoxLayout()
        project_layout.addWidget(self.project_path)
        project_layout.addWidget(self.project_browse)
        output_layout.addRow("项目路径:", project_layout)
        
        self.name = QLineEdit("exp")
        output_layout.addRow("实验名称:", self.name)
        
        output_group.setLayout(output_layout)
        layout.addWidget(output_group)
        
        # 添加弹性空间
        layout.addStretch()

    def toggle_custom_model(self, state):
        """启用或禁用自定义模型路径选择"""
        is_enabled = state == Qt.Checked
        self.custom_model_path.setEnabled(is_enabled)
        self.custom_model_browse.setEnabled(is_enabled)
        self.model_type.setEnabled(not is_enabled)
        self.pretrained_checkbox.setEnabled(not is_enabled)
        

    
    def setup_advanced_tab(self):
        layout = QVBoxLayout(self.advanced_tab)
        
        # 优化器设置
        optimizer_group = QGroupBox("优化器设置")
        optimizer_layout = QFormLayout()
        
        self.optimizer = QComboBox()
        self.optimizer.addItems(["SGD", "Adam", "AdamW"])
        optimizer_layout.addRow("优化器:", self.optimizer)
        
        self.lr0 = QDoubleSpinBox()
        self.lr0.setRange(0.00001, 0.1)
        self.lr0.setValue(0.01)
        self.lr0.setSingleStep(0.001)
        self.lr0.setDecimals(5)
        optimizer_layout.addRow("初始学习率 (lr0):", self.lr0)
        
        self.lrf = QDoubleSpinBox()
        self.lrf.setRange(0.01, 1.0)
        self.lrf.setValue(0.01)
        self.lrf.setSingleStep(0.01)
        self.lrf.setDecimals(2)
        optimizer_layout.addRow("最终学习率因子 (lrf):", self.lrf)
        
        self.momentum = QDoubleSpinBox()
        self.momentum.setRange(0.0, 1.0)
        self.momentum.setValue(0.937)
        self.momentum.setSingleStep(0.01)
        self.momentum.setDecimals(3)
        optimizer_layout.addRow("动量 (momentum):", self.momentum)
        
        self.weight_decay = QDoubleSpinBox()
        self.weight_decay.setRange(0.0, 0.001)
        self.weight_decay.setValue(0.0005)
        self.weight_decay.setSingleStep(0.0001)
        self.weight_decay.setDecimals(5)
        optimizer_layout.addRow("权重衰减 (weight decay):", self.weight_decay)
        
        optimizer_group.setLayout(optimizer_layout)
        layout.addWidget(optimizer_group)
        
        # 数据增强设置
        augmentation_group = QGroupBox("数据增强")
        augmentation_layout = QGridLayout()
        
        self.augment = QCheckBox("启用数据增强")
        self.augment.setChecked(True)
        augmentation_layout.addWidget(self.augment, 0, 0, 1, 2)
        
        self.hsv_h = QDoubleSpinBox()
        self.hsv_h.setRange(0.0, 1.0)
        self.hsv_h.setValue(0.015)
        self.hsv_h.setSingleStep(0.01)
        self.hsv_h.setDecimals(3)
        augmentation_layout.addWidget(QLabel("HSV-H:"), 1, 0)
        augmentation_layout.addWidget(self.hsv_h, 1, 1)
        
        self.hsv_s = QDoubleSpinBox()
        self.hsv_s.setRange(0.0, 1.0)
        self.hsv_s.setValue(0.7)
        self.hsv_s.setSingleStep(0.01)
        self.hsv_s.setDecimals(3)
        augmentation_layout.addWidget(QLabel("HSV-S:"), 2, 0)
        augmentation_layout.addWidget(self.hsv_s, 2, 1)
        
        self.hsv_v = QDoubleSpinBox()
        self.hsv_v.setRange(0.0, 1.0)
        self.hsv_v.setValue(0.4)
        self.hsv_v.setSingleStep(0.01)
        self.hsv_v.setDecimals(3)
        augmentation_layout.addWidget(QLabel("HSV-V:"), 3, 0)
        augmentation_layout.addWidget(self.hsv_v, 3, 1)
        
        self.degrees = QDoubleSpinBox()
        self.degrees.setRange(0.0, 180.0)
        self.degrees.setValue(0.0)
        self.degrees.setSingleStep(1.0)
        augmentation_layout.addWidget(QLabel("旋转角度:"), 1, 2)
        augmentation_layout.addWidget(self.degrees, 1, 3)
        
        self.translate = QDoubleSpinBox()
        self.translate.setRange(0.0, 1.0)
        self.translate.setValue(0.1)
        self.translate.setSingleStep(0.01)
        self.translate.setDecimals(2)
        augmentation_layout.addWidget(QLabel("平移:"), 2, 2)
        augmentation_layout.addWidget(self.translate, 2, 3)
        
        self.scale = QDoubleSpinBox()
        self.scale.setRange(0.0, 1.0)
        self.scale.setValue(0.5)
        self.scale.setSingleStep(0.01)
        self.scale.setDecimals(2)
        augmentation_layout.addWidget(QLabel("缩放:"), 3, 2)
        augmentation_layout.addWidget(self.scale, 3, 3)
        
        self.fliplr = QDoubleSpinBox()
        self.fliplr.setRange(0.0, 1.0)
        self.fliplr.setValue(0.5)
        self.fliplr.setSingleStep(0.01)
        self.fliplr.setDecimals(2)
        augmentation_layout.addWidget(QLabel("水平翻转概率:"), 4, 0)
        augmentation_layout.addWidget(self.fliplr, 4, 1)
        
        self.mosaic = QDoubleSpinBox()
        self.mosaic.setRange(0.0, 1.0)
        self.mosaic.setValue(1.0)
        self.mosaic.setSingleStep(0.01)
        self.mosaic.setDecimals(2)
        augmentation_layout.addWidget(QLabel("Mosaic概率:"), 4, 2)
        augmentation_layout.addWidget(self.mosaic, 4, 3)
        
        augmentation_group.setLayout(augmentation_layout)
        layout.addWidget(augmentation_group)
        
        # 其他高级设置
        advanced_group = QGroupBox("其他高级设置")
        advanced_layout = QFormLayout()
        
        # 添加特征点检测设置
        self.keypoints_checkbox = QCheckBox("启用特征点检测(姿态检测)")
        self.keypoints_checkbox.setChecked(False)
        advanced_layout.addRow("", self.keypoints_checkbox)
        
        # 添加特征点检测提示
        self.keypoints_note = QLabel("注意：启用特征点检测将使用pose任务类型，需要专用的姿态检测数据集。")
        self.keypoints_note.setWordWrap(True)
        advanced_layout.addRow("", self.keypoints_note)
        
        self.patience = QSpinBox()
        self.patience.setRange(0, 300)
        self.patience.setValue(100)
        advanced_layout.addRow("早停耐心值 (patience):", self.patience)
        
        self.workers = QSpinBox()
        self.workers.setRange(1, 16)
        self.workers.setValue(8)
        advanced_layout.addRow("数据加载线程数 (workers):", self.workers)
        
        self.device = QLineEdit("")
        self.device.setPlaceholderText("留空使用默认设备")
        advanced_layout.addRow("设备 (device):", self.device)
        
        self.cos_lr = QCheckBox("使用余弦学习率调度")
        self.cos_lr.setChecked(True)
        advanced_layout.addRow("", self.cos_lr)
        
        self.cache = QCheckBox("缓存图像以加速训练")
        self.cache.setChecked(False)
        advanced_layout.addRow("", self.cache)
        
        advanced_group.setLayout(advanced_layout)
        layout.addWidget(advanced_group)
        
        # 添加弹性空间
        layout.addStretch()
    
    def refresh_conda_envs(self):
        self.log_message("正在获取Conda环境列表...")
        self.conda_env_combo.clear()
        
        try:
            # 使用subprocess获取conda环境列表
            result = subprocess.run(['conda', 'env', 'list', '--json'], 
                                   capture_output=True, text=True, check=True)
            env_data = json.loads(result.stdout)
            
            # 提取环境名称
            envs = [os.path.basename(env) for env in env_data['envs']]
            self.conda_env_combo.addItems(envs)
            
            self.log_message(f"找到 {len(envs)} 个Conda环境")
        except Exception as e:
            self.log_message(f"获取Conda环境失败: {str(e)}")
    
    def browse_file(self, line_edit, file_filter):
        file_path, _ = QFileDialog.getOpenFileName(self, "选择文件", "", file_filter)
        if file_path:
            line_edit.setText(file_path)
    
    def browse_directory(self, line_edit):
        dir_path = QFileDialog.getExistingDirectory(self, "选择目录", "")
        if dir_path:
            line_edit.setText(dir_path)
    
    def log_message(self, message):
        self.log_text.append(message)
    
    def get_all_parameters(self):
        params = {
            # 基本设置
            "conda_env": self.conda_env_combo.currentText(),
            "yaml_path": self.yaml_path.text(),
            "model_type": self.model_type.currentText(),
            "pretrained": self.pretrained_checkbox.isChecked(),
            "use_custom_model": self.custom_model_checkbox.isChecked(),
            "custom_model_path": self.custom_model_path.text(),
            "epochs": self.epochs.value(),
            "batch_size": self.batch_size.value(),
            "img_size": self.img_size.value(),
            "project_path": self.project_path.text(),
            "name": self.name.text(),
            
            # 高级设置
            "optimizer": self.optimizer.currentText(),
            "lr0": self.lr0.value(),
            "lrf": self.lrf.value(),
            "momentum": self.momentum.value(),
            "weight_decay": self.weight_decay.value(),
            "augment": self.augment.isChecked(),
            "hsv_h": self.hsv_h.value(),
            "hsv_s": self.hsv_s.value(),
            "hsv_v": self.hsv_v.value(),
            "degrees": self.degrees.value(),
            "translate": self.translate.value(),
            "scale": self.scale.value(),
            "fliplr": self.fliplr.value(),
            "mosaic": self.mosaic.value(),
            "patience": self.patience.value(),
            "workers": self.workers.value(),
            "device": self.device.text(),
            "cos_lr": self.cos_lr.isChecked(),
            "cache": self.cache.isChecked(),
            
            # 特征点设置（姿态检测）
            "enable_keypoints": self.keypoints_checkbox.isChecked()
        }
        return params
    
    def save_settings(self):
        params = self.get_all_parameters()
        
        # 保存到QSettings
        for key, value in params.items():
            self.settings.setValue(key, value)
        
        self.log_message("设置已保存")
        
        # 同时保存到JSON文件以便训练脚本使用
        try:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            parent_dir = os.path.dirname(current_dir)  # 获取父目录路径
            settings_path = os.path.join(parent_dir, "yolo_train_settings.json")
            with open(settings_path, "w", encoding="utf-8") as f:
                json.dump(params, f, ensure_ascii=False, indent=4)
            self.log_message(f"设置已保存到{settings_path}")
        except Exception as e:
            self.log_message(f"保存设置到文件失败: {str(e)}")
    
    def load_settings(self):
        # 尝试从QSettings加载
        if self.settings.contains("yaml_path"):
            self.yaml_path.setText(self.settings.value("yaml_path", ""))
            self.model_type.setCurrentText(self.settings.value("model_type", "yolov8n"))
            self.pretrained_checkbox.setChecked(self.settings.value("pretrained", True, type=bool))
            
            # 加载自定义模型设置
            use_custom = self.settings.value("use_custom_model", False, type=bool)
            self.custom_model_checkbox.setChecked(use_custom)
            self.custom_model_path.setText(self.settings.value("custom_model_path", ""))
            self.toggle_custom_model(Qt.Checked if use_custom else Qt.Unchecked)
            
            self.epochs.setValue(self.settings.value("epochs", 100, type=int))
            self.batch_size.setValue(self.settings.value("batch_size", 16, type=int))
            self.img_size.setValue(self.settings.value("img_size", 640, type=int))
            self.project_path.setText(self.settings.value("project_path", ""))
            self.name.setText(self.settings.value("name", "exp"))
            
            # 高级设置
            self.optimizer.setCurrentText(self.settings.value("optimizer", "SGD"))
            self.lr0.setValue(self.settings.value("lr0", 0.01, type=float))
            self.lrf.setValue(self.settings.value("lrf", 0.01, type=float))
            self.momentum.setValue(self.settings.value("momentum", 0.937, type=float))
            self.weight_decay.setValue(self.settings.value("weight_decay", 0.0005, type=float))
            self.augment.setChecked(self.settings.value("augment", True, type=bool))
            self.hsv_h.setValue(self.settings.value("hsv_h", 0.015, type=float))
            self.hsv_s.setValue(self.settings.value("hsv_s", 0.7, type=float))
            self.hsv_v.setValue(self.settings.value("hsv_v", 0.4, type=float))
            self.degrees.setValue(self.settings.value("degrees", 0.0, type=float))
            self.translate.setValue(self.settings.value("translate", 0.1, type=float))
            self.scale.setValue(self.settings.value("scale", 0.5, type=float))
            self.fliplr.setValue(self.settings.value("fliplr", 0.5, type=float))
            self.mosaic.setValue(self.settings.value("mosaic", 1.0, type=float))
            self.patience.setValue(self.settings.value("patience", 100, type=int))
            self.workers.setValue(self.settings.value("workers", 8, type=int))
            self.device.setText(self.settings.value("device", ""))
            self.cos_lr.setChecked(self.settings.value("cos_lr", True, type=bool))
            self.cache.setChecked(self.settings.value("cache", False, type=bool))
            
            # 加载特征点设置
            enable_keypoints = self.settings.value("enable_keypoints", False, type=bool)
            self.keypoints_checkbox.setChecked(enable_keypoints)
            
            self.log_message("已加载保存的设置")
        
            # 刷新Conda环境列表
            self.refresh_conda_envs()
        
    def start_training(self):
        # 保存当前设置
        self.save_settings()
                
        # 检查必要参数
        if not self.yaml_path.text():
            self.log_message("错误: 请选择数据集YAML文件")
            return
                
        if not self.conda_env_combo.currentText():
            self.log_message("错误: 请选择Conda环境")
            return
                
        self.log_message("正在启动训练进程...")
                
        try:
            # 在Windows上使用start命令启动新的终端窗口
            conda_env = self.conda_env_combo.currentText()
                    
            # 获取训练脚本的绝对路径
            script_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "train_yolo.py")
                    
            # 构建命令
            cmd = f'start cmd.exe /k "conda activate {conda_env} && python "{script_path}""'
                    
            # 执行命令
            subprocess.Popen(cmd, shell=True)
                    
            self.log_message(f"已在新终端中启动训练进程，使用环境: {conda_env}")
        except Exception as e:
            self.log_message(f"启动训练进程失败: {str(e)}")
        
            # 如果直接运行此文件，则创建独立窗口
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = YoloTrainerUI()
    window.show()
    sys.exit(app.exec_())