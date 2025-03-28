import os
import sys
from PyQt5.QtWidgets import QDialog, QVBoxLayout, QPushButton, QWidget
from PyQt5.QtCore import Qt
from .trainer_ui import YoloTrainerUI

class YoloTrainerDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("YOLOv8 训练设置")
        self.setMinimumSize(800, 600)
        # 设置窗口标志，确保显示最小化和最大化按钮
        self.setWindowFlags(self.windowFlags() | Qt.WindowMinMaxButtonsHint)
        
        # 创建布局
        layout = QVBoxLayout(self)
        
        # 创建容器部件来包含训练器UI的内容
        container = QWidget()
        container_layout = QVBoxLayout(container)
        
        # 创建训练器UI实例
        self.trainer_ui = YoloTrainerUI(self)
        
        # 将训练器UI的中央部件内容添加到容器中
        container_layout.addWidget(self.trainer_ui.centralWidget())
        
        # 将容器添加到对话框布局中
        layout.addWidget(container)
        
        # 设置对话框按钮
        self.close_button = QPushButton("关闭")
        self.close_button.clicked.connect(self.accept)
        layout.addWidget(self.close_button)