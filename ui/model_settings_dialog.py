from PyQt5.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QLabel, 
                            QLineEdit, QDoubleSpinBox, QCheckBox, QPushButton, 
                            QFileDialog, QSpinBox, QGroupBox, QFormLayout, QComboBox)
from PyQt5.QtCore import Qt

from utils.config import Config
from i18n import tr

class ModelSettingsDialog(QDialog):
    """模型预测参数设置对话框"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.config = Config()
        self.model_params = self.config.get_model_params()
        
        self.setWindowTitle(tr("模型预测设置"))
        self.setMinimumWidth(450)
        self.setup_ui()
        
    def setup_ui(self):
        """设置UI界面"""
        layout = QVBoxLayout(self)
        
        # 模型文件设置
        model_group = QGroupBox(tr("模型文件"))
        model_layout = QHBoxLayout()
        
        self.model_path_edit = QLineEdit(self.model_params.get("model_path", ""))
        self.model_path_edit.setReadOnly(True)
        self.browse_btn = QPushButton(tr("浏览..."))
        self.browse_btn.clicked.connect(self.browse_model)
        
        model_layout.addWidget(self.model_path_edit)
        model_layout.addWidget(self.browse_btn)
        model_group.setLayout(model_layout)
        
        # 预测参数设置
        params_group = QGroupBox(tr("预测参数"))
        params_layout = QFormLayout()
        
        # 设备选择
        self.device_combo = QComboBox()
        self.device_combo.addItems(["cpu", "cuda"])
        current_device = self.model_params.get("device", "cpu")
        self.device_combo.setCurrentText(current_device)
        params_layout.addRow(tr("计算设备:"), self.device_combo)
        
        # 置信度阈值
        self.conf_threshold = QDoubleSpinBox()
        self.conf_threshold.setRange(0.01, 1.0)
        self.conf_threshold.setSingleStep(0.05)
        self.conf_threshold.setValue(self.model_params.get("confidence_threshold", 0.5))
        params_layout.addRow(tr("置信度阈值:"), self.conf_threshold)
        
        # IoU阈值
        self.iou_threshold = QDoubleSpinBox()
        self.iou_threshold.setRange(0.01, 1.0)
        self.iou_threshold.setSingleStep(0.05)
        self.iou_threshold.setValue(self.model_params.get("iou_threshold", 0.45))
        params_layout.addRow(tr("IoU阈值:"), self.iou_threshold)
        
        # 最大检测数量
        self.max_detections = QSpinBox()
        self.max_detections.setRange(1, 1000)
        self.max_detections.setValue(self.model_params.get("max_detections", 100))
        params_layout.addRow(tr("最大检测数量:"), self.max_detections)
        
        # 自动预测
        self.auto_predict = QCheckBox()
        self.auto_predict.setChecked(self.model_params.get("enable_auto_predict", False))
        params_layout.addRow(tr("启用自动预测:"), self.auto_predict)
        
        params_group.setLayout(params_layout)
        
        # 按钮
        btn_layout = QHBoxLayout()
        self.reset_btn = QPushButton(tr("重置为默认值"))
        self.reset_btn.clicked.connect(self.reset_params)
        
        self.save_btn = QPushButton(tr("保存"))
        self.save_btn.clicked.connect(self.save_params)
        self.save_btn.setDefault(True)
        
        self.cancel_btn = QPushButton(tr("取消"))
        self.cancel_btn.clicked.connect(self.reject)
        
        btn_layout.addWidget(self.reset_btn)
        btn_layout.addStretch()
        btn_layout.addWidget(self.cancel_btn)
        btn_layout.addWidget(self.save_btn)
        
        # 添加到主布局
        layout.addWidget(model_group)
        layout.addWidget(params_group)
        layout.addLayout(btn_layout)
        
    def browse_model(self):
        """浏览选择模型文件"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, 
            tr("选择模型文件"), 
            "", 
            tr("模型文件 (*.pt *.pth *.weights *.onnx);;所有文件 (*.*)")
        )
        if file_path:
            self.model_path_edit.setText(file_path)
    
    def reset_params(self):
        """重置为默认参数"""
        default_params = self.config.reset_model_params()
        self.model_path_edit.setText(default_params.get("model_path", ""))
        self.conf_threshold.setValue(default_params.get("confidence_threshold", 0.5))
        self.iou_threshold.setValue(default_params.get("iou_threshold", 0.45))
        self.max_detections.setValue(default_params.get("max_detections", 100))
        self.auto_predict.setChecked(default_params.get("enable_auto_predict", False))
        self.device_combo.setCurrentText(default_params.get("device", "cpu"))
    
    def save_params(self):
        """保存参数设置"""
        params = {
            "model_path": self.model_path_edit.text(),
            "confidence_threshold": self.conf_threshold.value(),
            "iou_threshold": self.iou_threshold.value(),
            "max_detections": self.max_detections.value(),
            "enable_auto_predict": self.auto_predict.isChecked(),
            "device": self.device_combo.currentText()
        }
        self.config.save_model_params(params)
        self.accept()