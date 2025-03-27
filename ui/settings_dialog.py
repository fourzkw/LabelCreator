from PyQt5.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QPushButton, 
                            QLabel, QTabWidget, QWidget, QTableWidget, 
                            QTableWidgetItem, QHeaderView, QMessageBox)
from PyQt5.QtGui import QKeySequence
from PyQt5.QtCore import Qt

from i18n import tr

class ShortcutEditor(QTableWidget):
    def __init__(self, shortcuts, parent=None):
        super().__init__(parent)
        self.shortcuts = shortcuts
        self.setup_ui()
        
    def setup_ui(self):
        # 设置表格列数和标题
        self.setColumnCount(2)
        self.setHorizontalHeaderLabels([tr("操作"), tr("快捷键")])
        self.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        
        # 填充表格数据
        self.setRowCount(len(self.shortcuts))
        
        # 操作名称映射
        action_names = {
            'save_current': tr("保存当前"),
            'save_all': tr("保存全部"),
            'prev_image': tr("上一张图像"),
            'next_image': tr("下一张图像"),
            'delete_box': tr("删除选中框"),
            'zoom_in': tr("放大"),
            'zoom_out': tr("缩小"),
            'reset_zoom': tr("重置缩放"),
            'open_directory': tr("打开目录"),
            'exit': tr("退出"),
            'auto_label': tr("自动标注")
        }
        
        row = 0
        for action, shortcut in self.shortcuts.items():
            # 操作名称
            name_item = QTableWidgetItem(action_names.get(action, action))
            name_item.setFlags(name_item.flags() & ~Qt.ItemIsEditable)  # 设为只读
            name_item.setData(Qt.UserRole, action)  # 存储原始操作名
            self.setItem(row, 0, name_item)
            
            # 快捷键
            shortcut_item = QTableWidgetItem(shortcut)
            self.setItem(row, 1, shortcut_item)
            
            row += 1
    
    def get_shortcuts(self):
        """获取修改后的快捷键"""
        result = {}
        for row in range(self.rowCount()):
            action = self.item(row, 0).data(Qt.UserRole)
            shortcut = self.item(row, 1).text()
            result[action] = shortcut
        return result

class SettingsDialog(QDialog):
    def __init__(self, settings, parent=None):
        super().__init__(parent)
        self.settings = settings
        self.setWindowTitle(tr("设置"))
        self.setMinimumSize(500, 400)
        self.setup_ui()
        
    def setup_ui(self):
        layout = QVBoxLayout(self)
        
        # 创建选项卡
        tab_widget = QTabWidget()
        
        # 快捷键选项卡
        shortcuts_tab = QWidget()
        shortcuts_layout = QVBoxLayout(shortcuts_tab)
        
        # 快捷键编辑器
        self.shortcut_editor = ShortcutEditor(self.settings.shortcuts)
        shortcuts_layout.addWidget(self.shortcut_editor)
        
        # 重置按钮
        reset_button = QPushButton(tr("重置为默认"))
        reset_button.clicked.connect(self.reset_shortcuts)
        shortcuts_layout.addWidget(reset_button)
        
        # 添加快捷键选项卡
        tab_widget.addTab(shortcuts_tab, tr("快捷键"))
        
        # 添加选项卡到主布局
        layout.addWidget(tab_widget)
        
        # 底部按钮
        button_layout = QHBoxLayout()
        save_button = QPushButton(tr("保存"))
        cancel_button = QPushButton(tr("取消"))
        
        save_button.clicked.connect(self.save_settings)
        cancel_button.clicked.connect(self.reject)
        
        button_layout.addStretch()
        button_layout.addWidget(save_button)
        button_layout.addWidget(cancel_button)
        
        layout.addLayout(button_layout)
    
    def reset_shortcuts(self):
        """重置快捷键为默认值"""
        from utils.settings import DEFAULT_SHORTCUTS
        reply = QMessageBox.question(self, tr("确认重置"), 
                                    tr("确定要将所有快捷键重置为默认值吗？"),
                                    QMessageBox.Yes | QMessageBox.No, 
                                    QMessageBox.No)
        if reply == QMessageBox.Yes:
            # 更新编辑器中的快捷键
            for row in range(self.shortcut_editor.rowCount()):
                action = self.shortcut_editor.item(row, 0).data(Qt.UserRole)
                if action in DEFAULT_SHORTCUTS:
                    self.shortcut_editor.item(row, 1).setText(DEFAULT_SHORTCUTS[action])
    
    def save_settings(self):
        """保存设置并关闭对话框"""
        # 获取修改后的快捷键
        new_shortcuts = self.shortcut_editor.get_shortcuts()
        
        # 更新设置
        for action, shortcut in new_shortcuts.items():
            self.settings.set_shortcut(action, shortcut)
        
        # 保存到文件
        if self.settings.save_settings():
            self.accept()  # 关闭对话框
        else:
            QMessageBox.warning(self, tr("错误"), tr("保存设置失败"))