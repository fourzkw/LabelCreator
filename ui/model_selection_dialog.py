from PyQt5.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QFormLayout, 
                            QGroupBox, QLabel, QComboBox, QLineEdit, 
                            QPushButton, QFileDialog, QRadioButton, QButtonGroup)
from PyQt5.QtCore import Qt
import os
import logging
from i18n import tr

logger = logging.getLogger('YOLOLabelCreator.ModelSelection')

class ModelSelectionDialog(QDialog):
    """模型选择对话框，用于选择模型版本、格式和路径"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle(tr("选择模型"))
        self.setMinimumWidth(500)
        self.setup_ui()
        
    def setup_ui(self):
        """设置UI界面"""
        layout = QVBoxLayout(self)
        
        # 模型版本选择
        version_group = QGroupBox(tr("模型版本"))
        version_layout = QVBoxLayout()
        
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
        self.yolov8_radio.setChecked(True)  # 默认选择YOLOv8
        self.version_group.addButton(self.yolov8_radio)
        version_layout.addWidget(self.yolov8_radio)
        
        # YOLOv11
        self.yolov11_radio = QRadioButton("YOLOv11")
        self.version_group.addButton(self.yolov11_radio)
        version_layout.addWidget(self.yolov11_radio)
        
        version_group.setLayout(version_layout)
        layout.addWidget(version_group)
        
        # 模型格式选择
        format_group = QGroupBox(tr("模型格式"))
        format_layout = QVBoxLayout()
        
        self.format_group = QButtonGroup(self)
        
        # PT格式
        self.pt_radio = QRadioButton(".pt")
        self.pt_radio.setChecked(True)  # 默认选择PT格式
        self.format_group.addButton(self.pt_radio)
        format_layout.addWidget(self.pt_radio)
        
        # ONNX格式
        self.onnx_radio = QRadioButton(".onnx")
        self.format_group.addButton(self.onnx_radio)
        format_layout.addWidget(self.onnx_radio)
        
        format_group.setLayout(format_layout)
        layout.addWidget(format_group)
        
        # 模型路径选择
        path_group = QGroupBox(tr("模型路径"))
        path_layout = QHBoxLayout()
        
        self.model_path_edit = QLineEdit()
        self.model_path_edit.setReadOnly(True)
        self.browse_btn = QPushButton(tr("浏览..."))
        self.browse_btn.clicked.connect(self.browse_model)
        
        path_layout.addWidget(self.model_path_edit)
        path_layout.addWidget(self.browse_btn)
        path_group.setLayout(path_layout)
        layout.addWidget(path_group)
        
        # 按钮
        btn_layout = QHBoxLayout()
        self.ok_btn = QPushButton(tr("确定"))
        self.ok_btn.clicked.connect(self.accept)
        self.ok_btn.setDefault(True)
        
        self.cancel_btn = QPushButton(tr("取消"))
        self.cancel_btn.clicked.connect(self.reject)
        
        btn_layout.addStretch()
        btn_layout.addWidget(self.cancel_btn)
        btn_layout.addWidget(self.ok_btn)
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
    
    def get_model_info(self):
        """获取选择的模型信息"""
        # 获取模型版本
        if self.yolov5_radio.isChecked():
            version = "yolov5"
        elif self.yolov7_radio.isChecked():
            version = "yolov7"
        elif self.yolov8_radio.isChecked():
            version = "yolov8"
        else:
            version = "yolov11"
        
        # 获取模型格式
        format_type = "pt" if self.pt_radio.isChecked() else "onnx"
        
        # 获取模型路径
        path = self.model_path_edit.text()
        
        return {
            "version": version,
            "format": format_type,
            "path": path
        }