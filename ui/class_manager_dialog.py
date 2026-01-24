from PyQt5.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QPushButton, 
                           QListWidget, QListWidgetItem, QInputDialog, QMessageBox)
from PyQt5.QtCore import Qt
import os
import yaml
from i18n import tr
import logging

logger = logging.getLogger('YOLOLabelCreator.ClassManager')

# 用于跟踪类别ID的映射关系（旧ID -> 新ID）
class_id_mapping = {}

class ClassManagerDialog(QDialog):
    def __init__(self, classes, data_yaml_path=None, parent=None):
        super().__init__(parent)
        self.classes = classes.copy()  # 复制类别列表，避免直接修改
        self.data_yaml_path = data_yaml_path
        self.original_classes = classes.copy()  # 保存原始类别列表，用于比较变化
        
        # 初始化类别ID映射（跟踪类别顺序变化）
        self.class_id_mapping = {i: i for i in range(len(self.classes))}  # 旧ID -> 新ID
        
        self.setWindowTitle(tr("类别管理"))
        self.setMinimumSize(500, 400)
        
        self.setup_ui()
        
    def setup_ui(self):
        # 主布局
        layout = QVBoxLayout(self)
        
        # 类别列表
        self.class_list = QListWidget()
        self.populate_class_list()
        layout.addWidget(self.class_list)
        
        # 按钮布局 - 第一行：基本操作
        button_layout = QHBoxLayout()
        
        # 添加类别按钮
        self.add_button = QPushButton(tr("添加类别"))
        self.add_button.clicked.connect(self.add_class)
        button_layout.addWidget(self.add_button)
        
        # 编辑类别按钮
        self.edit_button = QPushButton(tr("编辑类别"))
        self.edit_button.clicked.connect(self.edit_class)
        button_layout.addWidget(self.edit_button)
        
        # 删除类别按钮
        self.delete_button = QPushButton(tr("删除类别"))
        self.delete_button.clicked.connect(self.delete_class)
        button_layout.addWidget(self.delete_button)
        
        layout.addLayout(button_layout)
        
        # 按钮布局 - 第二行：排序操作
        order_layout = QHBoxLayout()
        
        # 上移按钮
        self.move_up_button = QPushButton(tr("↑ 上移"))
        self.move_up_button.clicked.connect(self.move_class_up)
        order_layout.addWidget(self.move_up_button)
        
        # 下移按钮
        self.move_down_button = QPushButton(tr("↓ 下移"))
        self.move_down_button.clicked.connect(self.move_class_down)
        order_layout.addWidget(self.move_down_button)
        
        # 添加伸缩空间
        order_layout.addStretch()
        
        layout.addLayout(order_layout)
        
        # 确定取消按钮
        dialog_buttons = QHBoxLayout()
        
        self.ok_button = QPushButton(tr("确定"))
        self.ok_button.clicked.connect(self.accept)
        dialog_buttons.addWidget(self.ok_button)
        
        self.cancel_button = QPushButton(tr("取消"))
        self.cancel_button.clicked.connect(self.reject)
        dialog_buttons.addWidget(self.cancel_button)
        
        layout.addLayout(dialog_buttons)
    
    def populate_class_list(self):
        """填充类别列表"""
        self.class_list.clear()
        for i, class_name in enumerate(self.classes):
            item = QListWidgetItem(f"{i}: {class_name}")
            self.class_list.addItem(item)
    
    def add_class(self):
        """添加新类别"""
        class_name, ok = QInputDialog.getText(self, tr("添加类别"), tr("类别名称:"))
        if ok and class_name:
            if class_name in self.classes:
                QMessageBox.warning(self, tr("警告"), tr("类别已存在"))
                return
            
            self.classes.append(class_name)
            self.populate_class_list()
    
    def edit_class(self):
        """编辑选中的类别"""
        current_item = self.class_list.currentItem()
        if not current_item:
            QMessageBox.warning(self, tr("警告"), tr("请先选择一个类别"))
            return
        
        # 获取当前选中的类别索引
        current_index = self.class_list.currentRow()
        current_class = self.classes[current_index]
        
        # 弹出编辑对话框
        new_name, ok = QInputDialog.getText(
            self, tr("编辑类别"), tr("类别名称:"), text=current_class
        )
        
        if ok and new_name:
            if new_name in self.classes and new_name != current_class:
                QMessageBox.warning(self, tr("警告"), tr("类别已存在"))
                return
            
            self.classes[current_index] = new_name
            self.populate_class_list()
    
    def delete_class(self):
        """删除选中的类别"""
        current_item = self.class_list.currentItem()
        if not current_item:
            QMessageBox.warning(self, tr("警告"), tr("请先选择一个类别"))
            return
        
        # 获取当前选中的类别索引
        current_index = self.class_list.currentRow()
        
        # 确认删除
        reply = QMessageBox.question(
            self, tr("确认删除"), 
            tr(f"确定要删除类别 '{self.classes[current_index]}' 吗？\n注意：这可能会影响已标注的数据。"),
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            del self.classes[current_index]
            self.populate_class_list()
    
    def move_class_up(self):
        """将选中的类别上移（降低索引）"""
        current_index = self.class_list.currentRow()
        
        # 检查是否选中且不是第一个
        if current_index <= 0:
            QMessageBox.warning(self, tr("警告"), tr("请选择一个类别，且不能是第一个"))
            return
        
        # 交换两个类别
        self.classes[current_index], self.classes[current_index - 1] = \
            self.classes[current_index - 1], self.classes[current_index]
        
        # 更新ID映射
        self._update_id_mapping_after_swap(current_index, current_index - 1)
        
        # 刷新列表并保持选中状态
        self.populate_class_list()
        self.class_list.setCurrentRow(current_index - 1)
    
    def move_class_down(self):
        """将选中的类别下移（增加索引）"""
        current_index = self.class_list.currentRow()
        
        # 检查是否选中且不是最后一个
        if current_index < 0 or current_index >= len(self.classes) - 1:
            QMessageBox.warning(self, tr("警告"), tr("请选择一个类别，且不能是最后一个"))
            return
        
        # 交换两个类别
        self.classes[current_index], self.classes[current_index + 1] = \
            self.classes[current_index + 1], self.classes[current_index]
        
        # 更新ID映射
        self._update_id_mapping_after_swap(current_index, current_index + 1)
        
        # 刷新列表并保持选中状态
        self.populate_class_list()
        self.class_list.setCurrentRow(current_index + 1)
    
    def _update_id_mapping_after_swap(self, old_idx1, old_idx2):
        """在交换两个类别后更新ID映射"""
        # 交换映射中的对应关系
        # 例如：原来 old_id1->new_id1, old_id2->new_id2
        # 交换后应该 old_id1->new_id2, old_id2->new_id1
        
        # 找到原始的ID（在original_classes中的索引）
        for old_id, new_id in list(self.class_id_mapping.items()):
            if new_id == old_idx1:
                self.class_id_mapping[old_id] = old_idx2
            elif new_id == old_idx2:
                self.class_id_mapping[old_id] = old_idx1
    
    def get_classes(self):
        """返回修改后的类别列表"""
        return self.classes
    
    def get_class_id_mapping(self):
        """返回类别ID映射（旧ID -> 新ID）"""
        return self.class_id_mapping
    
    def has_changes(self):
        """检查类别列表是否有变化"""
        return self.classes != self.original_classes
    
    def has_order_changes(self):
        """检查类别顺序是否有变化"""
        return any(self.class_id_mapping.get(i, i) != i for i in range(len(self.original_classes)))
    
    def accept(self):
        """确认按钮点击事件"""
        if len(self.classes) == 0:
            QMessageBox.warning(self, tr("警告"), tr("类别列表不能为空"))
            return
        
        # 如果有data.yaml路径，尝试更新
        if self.data_yaml_path and os.path.exists(os.path.dirname(self.data_yaml_path)):
            try:
                self.update_data_yaml()
            except Exception as e:
                logger.error(f"更新data.yaml失败: {str(e)}")
                QMessageBox.warning(self, tr("警告"), tr(f"更新data.yaml失败: {str(e)}"))
        
        super().accept()
    
    def update_data_yaml(self):
        """更新data.yaml文件"""
        if not self.data_yaml_path:
            return
        
        # 确保目录存在
        os.makedirs(os.path.dirname(self.data_yaml_path), exist_ok=True)
        
        # 读取现有的yaml文件（如果存在）
        data = {}
        if os.path.exists(self.data_yaml_path):
            try:
                with open(self.data_yaml_path, 'r', encoding='utf-8') as f:
                    data = yaml.safe_load(f) or {}
            except Exception as e:
                logger.error(f"读取data.yaml失败: {str(e)}")
        
        # 更新类别
        data['names'] = self.classes
        data['nc'] = len(self.classes)
        
        # 写入yaml文件
        try:
            with open(self.data_yaml_path, 'w', encoding='utf-8') as f:
                yaml.dump(data, f, default_flow_style=False, sort_keys=False, allow_unicode=True)
            logger.info(f"已更新data.yaml，包含{len(self.classes)}个类别")
        except Exception as e:
            logger.error(f"写入data.yaml失败: {str(e)}")
            raise