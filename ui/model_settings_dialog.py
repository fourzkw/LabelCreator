from PyQt5.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QFormLayout, 
                            QGroupBox, QLabel, QComboBox, QLineEdit, 
                            QPushButton, QFileDialog, QDoubleSpinBox, 
                            QSpinBox, QCheckBox, QRadioButton, QButtonGroup)
from PyQt5.QtCore import Qt
import os
import logging
from utils.config import Config
from i18n import tr

logger = logging.getLogger('YOLOLabelCreator.ModelSettings')

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
        model_group_layout = QVBoxLayout()
        
        # 模型版本选择
        version_layout = QVBoxLayout()
        version_label = QLabel(tr("模型版本"))
        version_layout.addWidget(version_label)
        
        self.version_group = QButtonGroup(self)
        
        # YOLOv5
        self.yolov5_radio = QRadioButton("YOLOv5")
        self.version_group.addButton(self.yolov5_radio)
        version_layout.addWidget(self.yolov5_radio)
        
        # YOLOv7
        self.yolov7_radio = QRadioButton("YOLOv7")
        self.version_group.addButton(self.yolov7_radio)
        version_layout.addWidget(self.yolov7_radio)
        
        # YOLOv8
        self.yolov8_radio = QRadioButton("YOLOv8")
        self.version_group.addButton(self.yolov8_radio)
        version_layout.addWidget(self.yolov8_radio)
        
        # YOLOv11
        self.yolov11_radio = QRadioButton("YOLOv11")
        self.version_group.addButton(self.yolov11_radio)
        version_layout.addWidget(self.yolov11_radio)
        
        # 设置默认选中的版本
        version = self.model_params.get("model_version", "yolov8")
        if version == "yolov5":
            self.yolov5_radio.setChecked(True)
        elif version == "yolov7":
            self.yolov7_radio.setChecked(True)
        elif version == "yolov11":
            self.yolov11_radio.setChecked(True)
        else:  # 默认 yolov8
            self.yolov8_radio.setChecked(True)
        
        # 模型格式选择
        format_layout = QVBoxLayout()
        format_label = QLabel(tr("模型格式"))
        format_layout.addWidget(format_label)
        
        self.format_group = QButtonGroup(self)
        
        # PT格式
        self.pt_radio = QRadioButton(".pt")
        self.format_group.addButton(self.pt_radio)
        format_layout.addWidget(self.pt_radio)
        
        # ONNX格式
        self.onnx_radio = QRadioButton(".onnx")
        self.format_group.addButton(self.onnx_radio)
        format_layout.addWidget(self.onnx_radio)
        
        # 设置默认选中的格式
        format_type = self.model_params.get("model_format", "pt")
        if format_type == "onnx":
            self.onnx_radio.setChecked(True)
        else:  # 默认 pt
            self.pt_radio.setChecked(True)
        
        # 模型路径选择
        path_layout = QHBoxLayout()
        path_label = QLabel(tr("模型路径"))
        
        self.model_path_edit = QLineEdit(self.model_params.get("model_path", ""))
        self.model_path_edit.setReadOnly(True)
        self.browse_btn = QPushButton(tr("浏览..."))
        self.browse_btn.clicked.connect(self.browse_model)
        
        path_layout.addWidget(self.model_path_edit)
        path_layout.addWidget(self.browse_btn)
        
        # 将所有布局添加到模型组
        model_group_layout.addLayout(version_layout)
        model_group_layout.addLayout(format_layout)
        model_group_layout.addWidget(path_label)
        model_group_layout.addLayout(path_layout)
        model_group.setLayout(model_group_layout)
        
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
        
        # 特征点数量
        self.keypoints_spinbox = QSpinBox()
        self.keypoints_spinbox.setRange(0, 100)
        self.keypoints_spinbox.setValue(self.model_params.get("keypoints_number", 0))
        self.keypoints_spinbox.setToolTip(tr("设置为0表示使用模型默认值"))
        params_layout.addRow(tr("特征点数量:"), self.keypoints_spinbox)
        
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
        # 根据选择的格式确定文件过滤器
        if self.pt_radio.isChecked():
            file_filter = tr("PyTorch模型 (*.pt *.pth);;所有文件 (*.*)")
        else:
            file_filter = tr("ONNX模型 (*.onnx);;所有文件 (*.*)")
        
        file_path, _ = QFileDialog.getOpenFileName(
            self, 
            tr("选择模型文件"), 
            "", 
            file_filter
        )
        if file_path:
            self.model_path_edit.setText(file_path)
            
            # 根据文件扩展名自动选择格式
            if file_path.lower().endswith('.onnx'):
                self.onnx_radio.setChecked(True)
            elif file_path.lower().endswith(('.pt', '.pth')):
                self.pt_radio.setChecked(True)
    
    def reset_params(self):
        """重置为默认参数"""
        default_params = self.config.reset_model_params()
        self.model_path_edit.setText(default_params.get("model_path", ""))
        self.conf_threshold.setValue(default_params.get("confidence_threshold", 0.5))
        self.iou_threshold.setValue(default_params.get("iou_threshold", 0.45))
        self.max_detections.setValue(default_params.get("max_detections", 100))
        self.auto_predict.setChecked(default_params.get("enable_auto_predict", False))
        self.device_combo.setCurrentText(default_params.get("device", "cpu"))
        self.keypoints_spinbox.setValue(default_params.get("keypoints_number", 0))
        
        # 重置模型版本和格式
        version = default_params.get("model_version", "yolov8")
        if version == "yolov5":
            self.yolov5_radio.setChecked(True)
        elif version == "yolov7":
            self.yolov7_radio.setChecked(True)
        elif version == "yolov11":
            self.yolov11_radio.setChecked(True)
        else:  # 默认 yolov8
            self.yolov8_radio.setChecked(True)
            
        format_type = default_params.get("model_format", "pt")
        if format_type == "onnx":
            self.onnx_radio.setChecked(True)
        else:  # 默认 pt
            self.pt_radio.setChecked(True)
    
    def get_model_version(self):
        """获取选择的模型版本"""
        if self.yolov5_radio.isChecked():
            return "yolov5"
        elif self.yolov7_radio.isChecked():
            return "yolov7"
        elif self.yolov8_radio.isChecked():
            return "yolov8"
        else:
            return "yolov11"
    
    def get_model_format(self):
        """获取选择的模型格式"""
        return "pt" if self.pt_radio.isChecked() else "onnx"
    
    def save_params(self):
        """保存参数设置"""
        params = {
            "model_path": self.model_path_edit.text(),
            "confidence_threshold": self.conf_threshold.value(),
            "iou_threshold": self.iou_threshold.value(),
            "max_detections": self.max_detections.value(),
            "enable_auto_predict": self.auto_predict.isChecked(),
            "device": self.device_combo.currentText(),
            "model_version": self.get_model_version(),
            "model_format": self.get_model_format()
        }
        self.config.save_model_params(params)
        self.accept()