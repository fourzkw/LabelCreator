import os
import yaml
import logging
import traceback
import shutil
import numpy as np
from PyQt5.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                             QPushButton, QLabel, QFileDialog, QListWidget, QMessageBox,
                             QComboBox, QLineEdit, QSplitter, QAction, QTreeView,
                             QGroupBox, QFrame, QStyle, QDialog, QApplication, QShortcut,
                             QScrollArea)
from PyQt5.QtGui import QStandardItemModel, QStandardItem, QKeySequence
from PyQt5.QtCore import Qt, QDir

from models.bounding_box import BoundingBox
from ui.canvas import ImageCanvas
from i18n import tr
from utils.yolo_predictor import YOLOPredictor
from utils.settings import Settings
from ui.settings_dialog import SettingsDialog
from ui.model_settings_dialog import ModelSettingsDialog
from training.trainer_dialog import YoloTrainerDialog
from ui.dataset_split_dialog import DatasetSplitDialog
from ui.class_manager_dialog import ClassManagerDialog
from ui.model_converter_dialog import ModelConverterDialog
from ui.model_inspector_dialog import ModelInspectorDialog

# 获取日志记录器
logger = logging.getLogger('YOLOLabelCreator.MainWindow')

class YOLOLabelCreator(QMainWindow):
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle(tr("YOLO Label Creator"))
        self.setGeometry(100, 100, 1200, 800)
        
        self.current_dir = ""
        self.image_files = []
        self.current_image_index = -1
        self.classes = []
        self.current_folder = ""
        
        # 初始化YOLO预测器
        self.yolo_predictor = YOLOPredictor()
        self.model_path = ""
        
        # 初始化设置
        app_dir = QDir.currentPath()
        self.settings = Settings(app_dir)
        
        # 应用样式表
        self.apply_stylesheet()
        
        self.init_ui()
        
        # 设置快捷键
        self.setup_shortcuts()
    
    def apply_stylesheet(self):
        """应用全局样式表"""
        self.setStyleSheet("""
            QMainWindow {
                background-color: #f5f5f5;
            }
            QWidget {
                font-family: 'Microsoft YaHei', 'Segoe UI', sans-serif;
                font-size: 10pt;
            }
            QPushButton {
                background-color: #2980b9;
                color: white;
                border: none;
                padding: 5px 15px;
                border-radius: 4px;
                min-height: 25px;
            }
            QPushButton:hover {
                background-color: #3498db;
            }
            QPushButton:pressed {
                background-color: #1c6ea4;
            }
            QPushButton:disabled {
                background-color: #cccccc;
                color: #999999;
            }
            QLabel {
                color: #333333;
            }
            QListWidget, QTreeView {
                background-color: white;
                border: 1px solid #dddddd;
                border-radius: 4px;
            }
            QListWidget::item:selected, QTreeView::item:selected {
                background-color: #3498db;
                color: white;
            }
            QComboBox {
                border: 1px solid #dddddd;
                border-radius: 4px;
                padding: 3px 15px 3px 5px;
                min-height: 25px;
            }
            QLineEdit {
                border: 1px solid #dddddd;
                border-radius: 4px;
                padding: 3px 5px;
                min-height: 25px;
            }
            QSplitter::handle {
                background-color: #dddddd;
            }
            QMenuBar {
                background-color: #f5f5f5;
                border-bottom: 1px solid #dddddd;
            }
            QMenuBar::item:selected {
                background-color: #3498db;
                color: white;
            }
            QMenu {
                background-color: white;
                border: 1px solid #dddddd;
            }
            QMenu::item:selected {
                background-color: #3498db;
                color: white;
            }
        """)
        
    def init_ui(self):
        # Create main widget and layout
        main_widget = QWidget()
        main_layout = QHBoxLayout()
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(10)
        
        # Create left panel for controls
        self.left_panel = QWidget()
        self.left_layout = QVBoxLayout(self.left_panel)
        self.left_layout.setContentsMargins(0, 0, 0, 0)
        self.left_layout.setSpacing(8)
        
        # Directory selection with icon
        dir_layout = QHBoxLayout()
        self.dir_label = QLabel(tr("No directory selected"))
        self.dir_label.setWordWrap(True)
        dir_button = QPushButton(tr("Select Directory"))
        dir_button.setIcon(self.style().standardIcon(QStyle.SP_DirOpenIcon))
        dir_button.clicked.connect(self.select_directory)
        dir_layout.addWidget(self.dir_label)
        dir_layout.addWidget(dir_button)
        
        # 添加文件夹树视图
        folder_group = QGroupBox(tr("Folders"))
        folder_layout = QVBoxLayout(folder_group)
        self.folder_tree = QTreeView()
        self.folder_tree.setHeaderHidden(True)
        self.folder_model = QStandardItemModel()
        self.folder_tree.setModel(self.folder_model)
        self.folder_tree.clicked.connect(self.folder_selected)
        self.folder_tree.setMinimumHeight(150)
        folder_layout.addWidget(self.folder_tree)
        
        # Image list in a group box
        image_group = QGroupBox(tr("Images in selected folder"))
        image_layout = QVBoxLayout(image_group)
        self.image_list = QListWidget()
        self.image_list.itemClicked.connect(self.load_selected_image)
        image_layout.addWidget(self.image_list)
        
        # Class management in a group box
        class_group = QGroupBox(tr("Class Management"))
        class_layout = QVBoxLayout(class_group)
        
        class_combo_layout = QHBoxLayout()
        class_combo_layout.addWidget(QLabel(tr("Class:")))
        self.class_combo = QComboBox()
        self.class_combo.addItems(self.classes)
        self.class_combo.currentIndexChanged.connect(self.update_current_class)
        class_combo_layout.addWidget(self.class_combo)
        
        class_add_layout = QHBoxLayout()
        self.class_input = QLineEdit()
        self.class_input.setPlaceholderText(tr("Enter new class name"))
        add_class_button = QPushButton(tr("Add Class"))
        add_class_button.setIcon(self.style().standardIcon(QStyle.SP_FileDialogNewFolder))
        add_class_button.clicked.connect(self.add_class)
        class_add_layout.addWidget(self.class_input)
        class_add_layout.addWidget(add_class_button)
        
        class_layout.addLayout(class_combo_layout)
        class_layout.addLayout(class_add_layout)

        # Box list in a group box
        box_group = QGroupBox(tr("Bounding Boxes"))
        box_layout = QVBoxLayout(box_group)
        self.box_list = QListWidget()
        self.box_list.itemClicked.connect(self.select_box)
        delete_box_button = QPushButton(tr("Delete Selected Box"))
        delete_box_button.setIcon(self.style().standardIcon(QStyle.SP_TrashIcon))
        delete_box_button.clicked.connect(self.delete_selected_box)
        box_layout.addWidget(self.box_list)
        box_layout.addWidget(delete_box_button)
        
        # Navigation buttons with icons
        nav_group = QGroupBox(tr("Navigation"))
        nav_layout = QHBoxLayout(nav_group)
        prev_button = QPushButton(tr("Previous Image"))
        prev_button.setIcon(self.style().standardIcon(QStyle.SP_ArrowLeft))
        next_button = QPushButton(tr("Next Image"))
        next_button.setIcon(self.style().standardIcon(QStyle.SP_ArrowRight))
        prev_button.clicked.connect(self.prev_image)
        next_button.clicked.connect(self.next_image)
        nav_layout.addWidget(prev_button)
        nav_layout.addWidget(next_button)
        
        # 保存按钮
        save_group = QGroupBox(tr("Save Options"))
        save_layout = QHBoxLayout(save_group)
        save_button = QPushButton(tr("Save Current"))
        save_button.setIcon(self.style().standardIcon(QStyle.SP_DialogSaveButton))
        save_all_button = QPushButton(tr("Save All"))
        save_all_button.setIcon(self.style().standardIcon(QStyle.SP_DialogSaveAllButton))
        save_button.clicked.connect(self.save_current)
        save_all_button.clicked.connect(self.save_all)
        save_layout.addWidget(save_button)
        save_layout.addWidget(save_all_button)
        
        # 自动标注控件放入组框
        auto_group = QGroupBox(tr("Auto Labeling"))
        auto_layout = QVBoxLayout(auto_group)
        
        model_layout = QHBoxLayout()
        self.auto_label_button = QPushButton(tr("Auto Label"))
        self.auto_label_button.setIcon(self.style().standardIcon(QStyle.SP_CommandLink))
        self.auto_label_button.clicked.connect(self.auto_label_current)
        self.auto_label_button.setEnabled(False)

        # 批量标注按钮
        self.auto_label_all_button = QPushButton(tr("Auto Label All"))
        self.auto_label_all_button.setIcon(self.style().standardIcon(QStyle.SP_DialogApplyButton))
        self.auto_label_all_button.clicked.connect(self.auto_label_all)
        self.auto_label_all_button.setEnabled(False)

        # 特征点编辑按钮
        self.keypoint_edit_button = QPushButton(tr("特征点编辑"))
        self.keypoint_edit_button.setIcon(self.style().standardIcon(QStyle.SP_ToolBarHorizontalExtensionButton))
        self.keypoint_edit_button.setCheckable(True)
        self.keypoint_edit_button.clicked.connect(self.toggle_keypoint_mode)

        # 数据集划分按钮
        self.dataset_split_button = QPushButton(tr("数据集划分"))
        self.dataset_split_button.setIcon(self.style().standardIcon(QStyle.SP_FileDialogListView))
        self.dataset_split_button.clicked.connect(self.open_dataset_split)


        model_layout.addWidget(self.auto_label_button)
        model_layout.addWidget(self.auto_label_all_button)
        model_layout.addWidget(self.keypoint_edit_button)
        model_layout.addWidget(self.dataset_split_button)
        
        self.model_label = QLabel(tr("No model selected"))
        self.model_label.setWordWrap(True)
        self.model_label.setStyleSheet("color: #666; font-size: 9pt;")
        
        auto_layout.addLayout(model_layout)
        auto_layout.addWidget(self.model_label)
        
        # 添加标签路径显示
        self.label_path_display = QLabel(tr("Label path: Not loaded"))
        self.label_path_display.setWordWrap(True)
        self.label_path_display.setStyleSheet("color: #666; font-size: 9pt;")
        
        # Add widgets to left layout
        self.left_layout.addLayout(dir_layout)
        self.left_layout.addWidget(folder_group)
        self.left_layout.addWidget(image_group)
        self.left_layout.addWidget(class_group)
        self.left_layout.addWidget(box_group)
        self.left_layout.addWidget(nav_group)
        self.left_layout.addWidget(save_group)
        self.left_layout.addWidget(auto_group)
        self.left_layout.addWidget(self.label_path_display)
        self.left_layout.addStretch(1)  # 添加弹性空间

        # Create right panel for image canvas
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(0, 0, 0, 0)
        
        # Image canvas with border
        canvas_frame = QFrame()
        canvas_frame.setFrameShape(QFrame.StyledPanel)
        canvas_frame.setFrameShadow(QFrame.Sunken)
        canvas_layout = QVBoxLayout(canvas_frame)
        canvas_layout.setContentsMargins(1, 1, 1, 1)
        self.canvas = ImageCanvas(self)
        canvas_layout.addWidget(self.canvas)
        
        # Zoom controls with icons
        zoom_layout = QHBoxLayout()
        zoom_in_button = QPushButton(tr("Zoom In"))
        zoom_in_button.setIcon(self.style().standardIcon(QStyle.SP_FileDialogListView))
        zoom_out_button = QPushButton(tr("Zoom Out"))
        zoom_out_button.setIcon(self.style().standardIcon(QStyle.SP_FileDialogDetailedView))
        reset_zoom_button = QPushButton(tr("Reset Zoom"))
        reset_zoom_button.setIcon(self.style().standardIcon(QStyle.SP_BrowserReload))
        
        # 辅助线切换按钮
        self.guide_lines_button = QPushButton(tr("辅助线"))
        self.guide_lines_button.setIcon(self.style().standardIcon(QStyle.SP_LineEditClearButton))
        self.guide_lines_button.setCheckable(True)
        self.guide_lines_button.setChecked(True)  # 默认启用
        self.guide_lines_button.clicked.connect(self.toggle_guide_lines)
        
        zoom_in_button.clicked.connect(self.canvas.zoom_in)
        zoom_out_button.clicked.connect(self.canvas.zoom_out)
        reset_zoom_button.clicked.connect(self.canvas.reset_zoom)
        zoom_layout.addWidget(zoom_in_button)
        zoom_layout.addWidget(zoom_out_button)
        zoom_layout.addWidget(reset_zoom_button)
        zoom_layout.addWidget(self.guide_lines_button)
        
        # Add widgets to right layout
        right_layout.addWidget(canvas_frame, 1)  # 让画布占据大部分空间
        right_layout.addLayout(zoom_layout)
        
        # Create scroll area for left panel
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setWidget(self.left_panel)
        self.scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        
        # Add panels to main layout
        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(self.scroll_area)
        splitter.addWidget(right_panel)
        splitter.setSizes([300, 900])  # Set initial sizes
        
        main_layout.addWidget(splitter)
        main_widget.setLayout(main_layout)
        self.setCentralWidget(main_widget)
        
        # Create menu bar with icons
        menubar = self.menuBar()
        file_menu = menubar.addMenu(tr("File"))
        
        open_dir_action = QAction(tr("Open Directory"), self)
        open_dir_action.setIcon(self.style().standardIcon(QStyle.SP_DirOpenIcon))
        open_dir_action.triggered.connect(self.select_directory)
        file_menu.addAction(open_dir_action)
        
        save_action = QAction(tr("Save Current"), self)
        save_action.setIcon(self.style().standardIcon(QStyle.SP_DialogSaveButton))
        save_action.triggered.connect(self.save_current)
        file_menu.addAction(save_action)
        
        save_all_action = QAction(tr("Save All"), self)
        save_all_action.setIcon(self.style().standardIcon(QStyle.SP_DialogSaveAllButton))
        save_all_action.triggered.connect(self.save_all)
        file_menu.addAction(save_all_action)
        file_menu.addSeparator()
        
        exit_action = QAction(tr("Exit"), self)
        exit_action.setIcon(self.style().standardIcon(QStyle.SP_DialogCloseButton))
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)
        
        settings_menu = menubar.addMenu(tr("设置"))
        
        # 首选项菜单项
        preferences_action = QAction(tr("首选项"), self)
        preferences_action.setIcon(self.style().standardIcon(QStyle.SP_FileDialogDetailedView))
        preferences_action.triggered.connect(self.show_settings)
        settings_menu.addAction(preferences_action)

        # 类别管理菜单项
        class_manager_action = QAction(tr("类别管理"), self)
        class_manager_action.setIcon(self.style().standardIcon(QStyle.SP_FileDialogListView))
        class_manager_action.triggered.connect(self.open_class_manager)
        settings_menu.addAction(class_manager_action)

        # 模型预测设置
        model_settings_action = QAction(tr("模型预测设置"), self)
        model_settings_action.setIcon(self.style().standardIcon(QStyle.SP_DesktopIcon))
        model_settings_action.triggered.connect(self.open_model_settings)
        settings_menu.addAction(model_settings_action)
        
        # 添加工具菜单
        tools_menu = menubar.addMenu(tr("工具"))
        
        # 添加数据集划分菜单项
        dataset_split_action = QAction(tr("数据集划分"), self)
        dataset_split_action.setIcon(self.style().standardIcon(QStyle.SP_FileDialogDetailedView))
        dataset_split_action.triggered.connect(self.open_dataset_split)
        tools_menu.addAction(dataset_split_action)
        
        # 添加模型转换菜单项
        model_converter_action = QAction(tr("PT模型转ONNX"), self)
        model_converter_action.setIcon(self.style().standardIcon(QStyle.SP_FileDialogContentsView))
        model_converter_action.triggered.connect(self.open_model_converter)
        tools_menu.addAction(model_converter_action)
        
        # 添加模型结构查看器菜单项
        model_inspector_action = QAction(tr("模型结构查看器"), self)
        model_inspector_action.setIcon(self.style().standardIcon(QStyle.SP_FileDialogInfoView))
        model_inspector_action.triggered.connect(self.open_model_inspector)
        tools_menu.addAction(model_inspector_action)
        
        # 添加训练菜单
        train_menu = menubar.addMenu(tr("训练"))
        
        # 添加YOLOv8训练器动作
        train_yolo_action = QAction(tr("YOLOv8训练器"), self)
        train_yolo_action.setIcon(self.style().standardIcon(QStyle.SP_ComputerIcon))
        train_yolo_action.triggered.connect(self.open_yolo_trainer)
        train_menu.addAction(train_yolo_action)
        
        # 添加状态栏
        self.statusBar().showMessage(tr("Ready"))

    def get_project_dir(self):
        return self.current_dir  # 即用户选择的目录

    def load_classes_from_yaml(self):
        if not self.current_dir:
            return None
        
        yaml_path = os.path.join(self.current_dir, "data.yaml")
        try:
            with open(yaml_path, 'r', encoding='utf-8') as f:  # 添加编码参数
                data = yaml.safe_load(f)
                return data.get('names', [])

        except FileNotFoundError:
            logger.warning(f"data.yaml not found at: {yaml_path}")
        except Exception as e:
            logger.error(f"YAML解析错误: {str(e)}")
        
        return None

    def select_directory(self):
        # 不再自动保存，直接切换目录
        dir_path = QFileDialog.getExistingDirectory(self, "Select Directory")
        if dir_path:
            self.current_dir = dir_path
            self.dir_label.setText(dir_path)
            self.populate_folder_tree()
            
            # 自动读取并初始化标签
            classes = self.load_classes_from_yaml()
            if classes:
                self.classes = classes
                logger.info(f"已从data.yaml加载{len(classes)}个标签类别")
            else:
                self.classes = ["Default"]
                logger.warning("使用默认标签类别")
            self.update_class_combo()
    
    def populate_folder_tree(self):
        """填充文件夹树视图"""
        self.folder_model.clear()
        root_item = QStandardItem(os.path.basename(self.current_dir))
        root_item.setData(self.current_dir, Qt.UserRole)
        self.folder_model.appendRow(root_item)
        
        # 递归添加子文件夹
        self.add_folders(root_item, self.current_dir)
        
        # 展开根节点
        self.folder_tree.expand(self.folder_model.indexFromItem(root_item))
        
        # 默认选择根目录
        self.folder_tree.setCurrentIndex(self.folder_model.indexFromItem(root_item))
        self.current_folder = self.current_dir
        self.load_images_from_directory(self.current_dir)
    
    def add_folders(self, parent_item, parent_path):
        """递归添加子文件夹到树视图"""
        try:
            for item in os.listdir(parent_path):
                item_path = os.path.join(parent_path, item)
                if os.path.isdir(item_path):
                    folder_item = QStandardItem(item)
                    folder_item.setData(item_path, Qt.UserRole)
                    parent_item.appendRow(folder_item)
                    self.add_folders(folder_item, item_path)
        except Exception as e:
            logger.error(f"Error adding folders: {str(e)}")
    
    def folder_selected(self, index):
        """当文件夹被选中时调用"""
        # 不再自动保存，直接切换文件夹
        item = self.folder_model.itemFromIndex(index)
        if item:
            folder_path = item.data(Qt.UserRole)
            if folder_path:
                self.current_folder = folder_path
                self.load_images_from_directory(folder_path)
    
    def load_images_from_directory(self, directory):
        """从指定目录加载图像"""
        self.image_files = []
        self.image_list.clear()
        
        # Get all image files
        try:
            for file in os.listdir(directory):
                if file.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
                    self.image_files.append(file)
            
            # Add to list widget
            self.image_list.addItems(self.image_files)
            
            # Load first image if available
            if self.image_files:
                self.current_image_index = 0
                self.image_list.setCurrentRow(0)
                self.load_image(os.path.join(directory, self.image_files[0]))
            else:
                # 清空画布
                self.canvas.image_path = None
                self.canvas.pixmap = None
                self.canvas.boxes = []
                self.canvas.update()
                self.update_box_list()
        except Exception as e:
            logger.error(f"Error loading images from directory: {str(e)}")
            QMessageBox.warning(self, tr("Error"), f"{tr('Failed to load images')}: {str(e)}")
    
    def load_selected_image(self, item):
        # 不再在切换图像时自动保存标签，只读取新图像的标签
        self.current_image_index = self.image_list.row(item)
        image_path = os.path.join(self.current_folder, item.text())
        self.load_image(image_path)
    
    def load_image(self, image_path):
        """
        加载图像并尝试读取关联的标签文件
        
        Args:
            image_path (str): 要加载的图像文件路径
        """
        logger.info(f"Loading image in main window: {image_path}")
        # Check if file exists
        if not os.path.exists(image_path):
            error_msg = tr("Image file not found") + f": {image_path}"
            logger.error(error_msg)
            QMessageBox.warning(self, tr("Error"), error_msg)
            return
        
        try:    
            # 首先加载图像到画布
            self.canvas.load_image(image_path)
            
            # 如果图像加载失败，pixmap将为None
            if not self.canvas.pixmap:
                error_msg = tr("Failed to load image") + f": {image_path}"
                logger.error(error_msg)
                QMessageBox.warning(self, tr("Error"), error_msg)
                return
            
            # 清空现有边界框以避免重复
            original_box_count = len(self.canvas.boxes)
            if original_box_count > 0:
                logger.info(f"清空加载新图像前的边界框，数量: {original_box_count}")
                self.canvas.boxes = []
            
            # 检查是否存在对应的标注文件
            label_path = self.get_label_path(image_path)
            if os.path.exists(label_path):
                logger.info(f"Found existing annotation file: {label_path}")
                self.load_annotations(label_path)
            else:
                logger.info(f"No existing annotation file found for: {image_path}")
                self.canvas.boxes = []
                self.update_box_list()
                
            # 验证加载后的边界框数量
            loaded_box_count = len(self.canvas.boxes)
            logger.info(f"图像加载完成后的边界框数量: {loaded_box_count}")
                
            logger.info(f"Successfully loaded image: {image_path}")
        except FileNotFoundError as e:
            error_msg = f"{tr('Image file not found')}: {str(e)}"
            logger.error(error_msg)
            QMessageBox.warning(self, tr("Error"), error_msg)
        except Exception as e:
            error_msg = f"{tr('Failed to load image')}: {str(e)}"
            logger.error(f"Unexpected error loading image: {str(e)}")
            logger.error(f"Exception details: {traceback.format_exc()}")
            QMessageBox.warning(self, tr("Error"), error_msg)
    
    def get_label_path(self, image_path):
        """根据图像路径生成对应的YOLO格式标签文件路径"""
        # 提取图像文件名（不含扩展名）
        image_filename = os.path.basename(image_path)
        base_name = os.path.splitext(image_filename)[0]
        
        # 创建标签目录结构
        image_dir = os.path.dirname(image_path)
        parent_dir = os.path.dirname(image_dir)
        labels_dir = os.path.join(parent_dir, "labels")
        label_subdir = labels_dir
        if not os.path.exists(label_subdir):
            os.makedirs(label_subdir, exist_ok=True)
            
        return os.path.normpath(os.path.join(label_subdir, f"{base_name}.txt"))
    
    def load_annotations(self, label_path):
        """从YOLO格式标签文件加载标注数据"""
        # 确保图像已正确加载
        if not self.canvas.pixmap or self.canvas.pixmap.isNull():
            QMessageBox.warning(self, tr("Error"), tr("Cannot load annotations: No valid image loaded"))
            return
            
        # 保存当前标签数量用于日志记录
        original_box_count = len(self.canvas.boxes)
        logger.info(f"加载标签前数量: {original_box_count}")
        
        # 先读取文件内容，确认能正确解析后再清空现有标签
        try:
            if os.path.exists(label_path):
                with open(label_path, 'r') as f:
                    lines = f.readlines()
                
                # 初始化临时列表
                temp_boxes = []
                
                # 原始图像尺寸
                img_width = self.canvas.pixmap.width()
                img_height = self.canvas.pixmap.height()
                
                # 解析每一行数据
                for line in lines:
                    line = line.strip()
                    if not line:  # 跳过空行
                        continue
                    
                    parts = line.split()
                    if len(parts) < 5:  # 至少需要类别和边界框坐标
                        logger.warning(f"格式错误的标注行: {line}")
                        continue
                        
                    try:
                        # 解析YOLO格式数据
                        class_id = int(parts[0])
                        x_center = float(parts[1])
                        y_center = float(parts[2])
                        width = float(parts[3])
                        height = float(parts[4])
                        
                        # 转换为像素坐标
                        x1 = (x_center - width / 2) * img_width
                        y1 = (y_center - height / 2) * img_height
                        x2 = (x_center + width / 2) * img_width
                        y2 = (y_center + height / 2) * img_height
                        
                        # 创建边界框对象
                        box = BoundingBox(x1, y1, x2, y2, class_id)
                        
                        # 检查是否有关键点数据（每个点有x、y两个坐标值）
                        if len(parts) > 5:
                            keypoints_data = parts[5:]
                            keypoints_count = len(keypoints_data) // 2
                            
                            if keypoints_count > 0 and len(keypoints_data) % 2 == 0:
                                keypoints = []
                                
                                # 解析关键点坐标
                                for i in range(keypoints_count):
                                    try:
                                        kp_x = float(keypoints_data[i*2]) * img_width
                                        kp_y = float(keypoints_data[i*2+1]) * img_height
                                        keypoints.append([kp_x, kp_y])
                                    except (ValueError, IndexError) as e:
                                        logger.warning(f"解析关键点坐标时出错 #{i}: {str(e)}")
                                
                                # 设置特征点
                                if keypoints:
                                    box.set_keypoints(np.array(keypoints))
                        
                        # 添加到临时列表
                        temp_boxes.append(box)
                        
                    except ValueError as e:
                        logger.warning(f"解析标注数据时出错: {str(e)}, 行: {line}")
                
                # 成功解析完成，现在更新画布的边界框列表
                self.canvas.boxes = temp_boxes
                self.update_box_list()
                logger.info(f"加载了 {len(temp_boxes)} 个边界框")
                
            else:
                # 文件不存在，清空边界框
                logger.warning(f"标注文件不存在: {label_path}")
                self.canvas.boxes = []
                self.update_box_list()
                
        except Exception as e:
            logger.error(f"加载标注失败: {str(e)}")
            logger.error(traceback.format_exc())
            QMessageBox.warning(self, tr("Error"), f"{tr('Failed to load annotations')}: {str(e)}")
            
        # 更新UI并触发重绘
        self.canvas.update()
    
    def update_box_list(self):
        self.box_list.clear()
        for i, box in enumerate(self.canvas.boxes):
            class_name = self.get_class_name(box.class_id)
            self.box_list.addItem(f"{tr('Box')} {i+1}: {class_name}")
    
    def select_box(self, item):
        # Get selected box index
        index = self.box_list.row(item)
        if 0 <= index < len(self.canvas.boxes):
            # Update class combo to match selected box
            box = self.canvas.boxes[index]
            self.class_combo.setCurrentIndex(box.class_id)
            
            # 更新画布中的选中边界框
            self.canvas.selected_box_index = index
            self.canvas.update()
    
    def delete_selected_box(self):
        selected_items = self.box_list.selectedItems()
        if selected_items:
            # 添加确认对话框
            reply = QMessageBox.question(self, tr("确认删除"), 
                                        tr("确定要删除选中的边界框吗？"),
                                        QMessageBox.Yes | QMessageBox.No, 
                                        QMessageBox.No)
            if reply == QMessageBox.Yes:
                index = self.box_list.row(selected_items[0])
                if 0 <= index < len(self.canvas.boxes):
                    # 记录删除前的标签数量
                    original_count = len(self.canvas.boxes)
                    
                    # 获取将被删除的标签类别信息
                    box_to_delete = self.canvas.boxes[index]
                    class_name = self.get_class_name(box_to_delete.class_id)
                    
                    # 记录删除操作
                    logger.info(f"删除标签: 索引 {index}，类别 '{class_name}'")
                    
                    # 执行删除操作
                    del self.canvas.boxes[index]
                    
                    # 记录删除后的标签数量
                    logger.info(f"删除后标签数量: {len(self.canvas.boxes)}")
                    
                    self.update_box_list()
                    self.canvas.update()
                    self.save_current()
    
    def add_class(self):
        class_name = self.class_input.text().strip()
        if class_name and class_name not in self.classes:
            self.classes.append(class_name)
            self.update_class_combo()
            self.class_input.clear()
    
    def update_class_combo(self):
        self.class_combo.clear()
        self.class_combo.addItems(self.classes)
        if self.classes:
            self.class_combo.setCurrentIndex(0)
    
    def update_current_class(self, index):
        self.canvas.set_current_class(index)
    
    def get_class_name(self, class_id):
        if 0 <= class_id < len(self.classes):
            return self.classes[class_id]
        return f"{tr('Unknown')} ({class_id})"
    
    def prev_image(self):
        if self.current_image_index > 0:
            # 不再自动保存，直接切换到上一张图像
            self.current_image_index -= 1
            self.image_list.setCurrentRow(self.current_image_index)
            image_path = os.path.join(self.current_folder, self.image_files[self.current_image_index])
            self.load_image(image_path)
    
    def next_image(self):
        if self.current_image_index < len(self.image_files) - 1:
            # 不再自动保存，直接切换到下一张图像
            self.current_image_index += 1
            self.image_list.setCurrentRow(self.current_image_index)
            image_path = os.path.join(self.current_folder, self.image_files[self.current_image_index])
            self.load_image(image_path)
    
    def save_current(self):
        if not self.canvas.image_path:
            QMessageBox.warning(self, tr("Warning"), tr("No image loaded"))
            return
        
        label_path = self.get_label_path(self.canvas.image_path)
        self.save_annotations(label_path)
    
    def save_all(self):
        if not self.image_files:
            QMessageBox.warning(self, tr("Warning"), tr("No images loaded"))
            return
        
        # Save current image first
        if self.canvas.image_path:
            self.save_current()
        
        # Save all other images
        current_index = self.current_image_index
        for i, image_file in enumerate(self.image_files):
            if i != current_index:  # Skip current image as it's already saved
                self.current_image_index = i
                self.image_list.setCurrentRow(i)
                image_path = os.path.join(self.current_folder, image_file)
                self.load_image(image_path)
                label_path = self.get_label_path(image_path)
                self.save_annotations(label_path)
        
        # Return to original image
        self.current_image_index = current_index
        self.image_list.setCurrentRow(current_index)
        image_path = os.path.join(self.current_folder, self.image_files[current_index])
        self.load_image(image_path)
        
        QMessageBox.information(self, tr("Success"), tr("All annotations saved successfully"))
    
    def save_annotations(self, label_path):
        """
        将当前标注保存为YOLO格式标签文件
        
        Args:
            label_path (str): 标签文件保存路径
            
        Returns:
            bool: 保存是否成功
            
        Note:
            同时会更新classes.txt和data.yaml文件
        """
        # 确保图像已正确加载
        if not self.canvas.pixmap or self.canvas.pixmap.isNull():
            QMessageBox.warning(self, tr("Error"), tr("Cannot save annotations: No valid image loaded"))
            return False
            
        # 检查是否存在标签，如果不存在但原文件存在标签，则给出警告
        if not self.canvas.boxes:
            if os.path.exists(label_path):
                try:
                    with open(label_path, 'r') as f:
                        lines = f.readlines()
                        if len(lines) > 0:
                            # 原文件有标签，但当前没有标签
                            reply = QMessageBox.question(
                                self, 
                                tr("警告"), 
                                tr("原标签文件包含{}个标注，但当前没有任何标签。确定要覆盖保存空标签文件吗？").format(len(lines)),
                                QMessageBox.Yes | QMessageBox.No
                            )
                            if reply == QMessageBox.No:
                                return False
                except Exception:
                    pass  # 如果读取失败，继续保存当前标签
                
            logger.warning(f"保存空标签文件: {label_path}")
            
        try:
            img_width = self.canvas.pixmap.width()
            img_height = self.canvas.pixmap.height()
            
            # 保存前记录即将保存的边界框数量
            pre_save_box_count = len(self.canvas.boxes)
            logger.info(f"即将保存的标签数量: {pre_save_box_count}")
            
            # 检查是否有边界框可以保存
            if pre_save_box_count == 0:
                logger.warning(f"注意：正在保存空标签文件 {label_path}")
            
            # 保存标签文件
            self._write_label_file(label_path, img_width, img_height)
            
            # 确认保存后标签数量未变化
            post_save_box_count = len(self.canvas.boxes)
            if pre_save_box_count != post_save_box_count:
                logger.error(f"警告：保存前边界框数量 {pre_save_box_count} 与保存后数量 {post_save_box_count} 不一致")
            
            # 保存类别名称到classes.txt
            classes_path = os.path.join(os.path.dirname(label_path), "classes.txt")
            with open(classes_path, 'w') as f:
                for class_name in self.classes:
                    f.write(f"{class_name}\n")
            
            # 记录保存的标签数量        
            logger.info(f"成功保存标签文件: {label_path}，共 {post_save_box_count} 个标签")
            
            return True
        except Exception as e:
            logger.error(f"Error saving annotations: {str(e)}")
            QMessageBox.warning(self, tr("Error"), f"{tr('Failed to save annotations')}: {str(e)}")
            return False
            
    def _write_label_file(self, path, img_width, img_height):
        """
        将边界框数据写入YOLO格式标签文件
        
        Args:
            path (str): 标签文件保存路径
            img_width (int): 图像宽度
            img_height (int): 图像高度
            
        Note:
            YOLO格式：每行表示一个边界框，格式为 "class_id x_center y_center width height [keypoints...]"
            所有坐标都是归一化的（0-1范围）
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # 记录即将写入的框数量
        boxes_to_save = len(self.canvas.boxes)
        logger.info(f"准备写入标签数量: {boxes_to_save}")
        
        with open(path, 'w', encoding='utf-8') as f:
            for box in self.canvas.boxes:
                yolo_box = box.to_yolo_format(img_width, img_height)
                line = f"{yolo_box[0]} {yolo_box[1]:.6f} {yolo_box[2]:.6f} {yolo_box[3]:.6f} {yolo_box[4]:.6f}"
                
                # 检查是否有特征点数据，如果有则添加到标签行
                if box.has_keypoints():
                    keypoints = box.keypoints
                    # 将特征点坐标转换为相对坐标并添加到行末
                    for kp in keypoints:
                        # 归一化坐标到[0,1]范围
                        kp_x_norm = kp[0] / img_width
                        kp_y_norm = kp[1] / img_height
                        line += f" {kp_x_norm:.6f} {kp_y_norm:.6f}"
                
                line += "\n"
                f.write(line)
        self.update_data_yaml()

    def update_data_yaml(self):
        if not self.current_dir:
            return
        
        yaml_path = os.path.join(self.current_dir, "data.yaml")
        data = {
            'train': '../train/images',
            'val': '../valid/images',
            'test': '../test/images',
            'nc': len(self.classes),
            'names': self.classes
        }
        try:
            with open(yaml_path, 'w', encoding='utf-8') as f:
                yaml.dump(data, f, default_flow_style=False)
        except Exception as e:
            logger.error(f"Failed to save data.yaml: {str(e)}")
    
    # 添加模型选择方法
    def select_model(self):
        """选择YOLO模型文件"""
        model_path, _ = QFileDialog.getOpenFileName(
            self, tr("选择YOLO模型"), "", tr("模型文件 (*.pt *.pth);;所有文件 (*)")
        )
        
        if model_path:
            logger.info(f"选择模型文件: {model_path}")
            self.model_path = model_path
            self.model_label.setText(os.path.basename(model_path))
            
            # 尝试加载模型
            if self.yolo_predictor.load_model(model_path):
                self.auto_label_button.setEnabled(True)
                self.auto_label_all_button.setEnabled(True)
                QMessageBox.information(self, tr("成功"), tr("模型加载成功"))
            else:
                self.auto_label_button.setEnabled(False)
                self.auto_label_all_button.setEnabled(False)
                QMessageBox.warning(self, tr("错误"), tr("模型加载失败"))

    # 添加自动标注方法
    def auto_label_current(self):
        """使用YOLO模型自动标注当前图像"""
        if not self.canvas.pixmap:
            QMessageBox.warning(self, tr("警告"), tr("请先加载图像"))
            return
            
        if not hasattr(self, 'model_path') or not self.model_path or not os.path.exists(self.model_path):
            QMessageBox.warning(self, tr("警告"), tr("请先选择有效的模型文件"))
            return
        
        try:
            # 显示进度对话框
            from PyQt5.QtWidgets import QProgressDialog
            progress = QProgressDialog(tr("处理中..."), tr("取消"), 0, 100, self)
            progress.setWindowTitle(tr("自动标注"))
            progress.setWindowModality(Qt.WindowModal)
            progress.setValue(10)
            
            # 获取当前图像路径
            current_image = self.image_files[self.current_image_index]
            image_path = os.path.join(self.current_folder, current_image)
            
            progress.setValue(30)
            
            # 执行预测
            boxes = self.yolo_predictor.predict(image_path)
            
            progress.setValue(70)
            
            # 记录自动标注前的标签数量
            original_count = len(self.canvas.boxes)
            logger.info(f"自动标注前标签数量: {original_count}")
            
            if boxes:
                # 不再弹出确认对话框，直接添加预测的边界框，保留现有标注
                self.canvas.boxes.extend(boxes)
                # 记录添加后的标签数量
                new_count = len(self.canvas.boxes)
                logger.info(f"自动标注后标签数量: {new_count}，新增 {new_count - original_count} 个标签")
                
                self.update_box_list()
                self.canvas.update()
                
                # 自动保存
                self.save_current()

                # 改为状态栏提示
                self.statusBar().showMessage(tr(f"自动标注完成，检测到{len(boxes)}个目标"), 3000)
            else:
                # 状态栏提示
                self.statusBar().showMessage(tr("未检测到任何目标"), 3000)
            progress.setValue(100)
            
        except Exception as e:
            logger.error(f"自动标注失败: {str(e)}")
            QMessageBox.warning(self, tr("错误"), tr(f"自动标注失败: {str(e)}"))

    def auto_label_all(self):
        """使用YOLO模型自动标注当前文件夹中的所有图像"""
        if not self.current_folder or not self.image_files:
            QMessageBox.warning(self, tr("警告"), tr("请先选择包含图像的文件夹"))
            return
            
        if not hasattr(self, 'model_path') or not self.model_path or not os.path.exists(self.model_path):
            QMessageBox.warning(self, tr("警告"), tr("请先选择有效的模型文件"))
            return
        
        try:
            # 创建进度对话框
            from PyQt5.QtWidgets import QProgressDialog
            progress = QProgressDialog(tr("正在处理图像..."), tr("取消"), 0, len(self.image_files), self)
            progress.setWindowTitle(tr("批量自动标注"))
            progress.setWindowModality(Qt.WindowModal)
            progress.setValue(0)
            
            # 保存当前图像索引，以便处理完成后恢复
            original_index = self.current_image_index
            
            # 处理每张图像
            processed_count = 0
            for i, image_file in enumerate(self.image_files):
                if progress.wasCanceled():
                    break
                    
                image_path = os.path.join(self.current_folder, image_file)
                progress.setLabelText(tr(f"正在处理 ({i+1}/{len(self.image_files)}): {image_file}"))
                
                # 加载图像
                self.current_image_index = i
                self.image_list.setCurrentRow(i)
                self.load_image(image_path)
                
                # 记录当前标签数量
                original_count = len(self.canvas.boxes)
                logger.info(f"批量自动标注 [{i+1}/{len(self.image_files)}] '{image_file}' - 初始标签数量: {original_count}")
                
                # 执行预测
                boxes = self.yolo_predictor.predict(image_path)
                
                if boxes:
                    # 添加预测的边界框，保留现有标注
                    self.canvas.boxes.extend(boxes)
                    new_count = len(self.canvas.boxes)
                    logger.info(f"批量自动标注 - 添加后标签数量: {new_count}，新增 {new_count - original_count} 个标签")
                    
                    self.update_box_list()
                    self.canvas.update()
                    
                    # 保存标注
                    self.save_current()
                    processed_count += 1
                
                # 更新进度
                progress.setValue(i + 1)
                QApplication.processEvents()  # 确保UI响应
            
            # 恢复到原始图像
            self.current_image_index = original_index
            self.image_list.setCurrentRow(original_index)
            image_path = os.path.join(self.current_folder, self.image_files[original_index])
            self.load_image(image_path)
            
            # 显示完成消息
            self.statusBar().showMessage(tr(f"批量标注完成，成功处理 {processed_count} 张图像"), 5000)
            
        except Exception as e:
            logger.error(f"批量自动标注失败: {str(e)}\n{traceback.format_exc()}")
            QMessageBox.warning(self, tr("错误"), tr(f"批量自动标注失败: {str(e)}"))


    # 添加快捷键设置方法
    def setup_shortcuts(self):
        """设置键盘快捷键"""
        # 保存标签
        save_shortcut = QShortcut(QKeySequence(self.settings.get_shortcut('save_current')), self)
        save_shortcut.activated.connect(self.save_current)
        
        # 保存所有标签
        save_all_shortcut = QShortcut(QKeySequence(self.settings.get_shortcut('save_all')), self)
        save_all_shortcut.activated.connect(self.save_all)
        
        # 上一张图像
        prev_shortcut = QShortcut(QKeySequence(self.settings.get_shortcut('prev_image')), self)
        prev_shortcut.activated.connect(self.prev_image)
        
        # 下一张图像
        next_shortcut = QShortcut(QKeySequence(self.settings.get_shortcut('next_image')), self)
        next_shortcut.activated.connect(self.next_image)
        
        # 删除选中的边界框
        delete_shortcut = QShortcut(QKeySequence(self.settings.get_shortcut('delete_box')), self)
        delete_shortcut.activated.connect(self.delete_selected_box)
        
        # 缩放控制
        zoom_in_shortcut = QShortcut(QKeySequence(self.settings.get_shortcut('zoom_in')), self)
        zoom_in_shortcut.activated.connect(self.canvas.zoom_in)
        
        zoom_out_shortcut = QShortcut(QKeySequence(self.settings.get_shortcut('zoom_out')), self)
        zoom_out_shortcut.activated.connect(self.canvas.zoom_out)
        
        zoom_reset_shortcut = QShortcut(QKeySequence(self.settings.get_shortcut('reset_zoom')), self)
        zoom_reset_shortcut.activated.connect(self.canvas.reset_zoom)
        
        # 打开目录
        open_dir_shortcut = QShortcut(QKeySequence(self.settings.get_shortcut('open_directory')), self)
        open_dir_shortcut.activated.connect(self.select_directory)
        
        # 退出
        exit_shortcut = QShortcut(QKeySequence(self.settings.get_shortcut('exit')), self)
        exit_shortcut.activated.connect(self.close)
        
        # 自动标注
        auto_label_shortcut = QShortcut(QKeySequence(self.settings.get_shortcut('auto_label')), self)
        auto_label_shortcut.activated.connect(self.auto_label_current)
        
        # 批量自动标注
        auto_label_all_shortcut = QShortcut(QKeySequence(self.settings.get_shortcut('auto_label_all')), self)
        auto_label_all_shortcut.activated.connect(self.auto_label_all)
        
        # 切换特征点编辑模式
        toggle_keypoint_shortcut = QShortcut(QKeySequence(self.settings.get_shortcut('toggle_keypoint_mode')), self)
        toggle_keypoint_shortcut.activated.connect(self.toggle_keypoint_mode)

    def show_settings(self):
        """显示设置对话框"""
        dialog = SettingsDialog(self.settings, self)  # Now this will work correctly
        if dialog.exec_() == QDialog.Accepted:
            # 如果设置已保存，重新应用快捷键
            self.setup_shortcuts()
            self.statusBar().showMessage(tr("设置已更新"), 3000)

    def open_yolo_trainer(self):
            """打开YOLOv8训练器对话框"""
            try:
                trainer_dialog = YoloTrainerDialog(self)
                trainer_dialog.exec_()
            except Exception as e:
                logger.error(f"打开训练器失败: {str(e)}")
                QMessageBox.warning(self, tr("错误"), f"{tr('打开训练器失败')}: {str(e)}")

    def open_model_settings(self):
        """打开模型设置对话框"""
        try:
            # 获取当前模型参数
            model_params = self.settings.get_model_params()
            
            # 创建并显示对话框
            dialog = ModelSettingsDialog(self, model_params, self.yolo_predictor.available_devices)
            result = dialog.exec_()
            
            # 处理结果
            if result == QDialog.Accepted:
                # 从对话框获取更新的参数
                new_params = dialog.get_updated_params()
                
                # 保存更新的参数
                self.settings.save_model_params(new_params)
                
                # 更新预测器设置
                self.yolo_predictor.set_params(
                    conf_threshold=new_params['confidence_threshold'],
                    iou_threshold=new_params['iou_threshold'],
                    max_detections=new_params['max_detections'],
                    device=new_params['device']
                )
                
                # 如果选择了新模型，加载它
                new_model_path = new_params.get('model_path')
                if new_model_path and (not self.model_path or new_model_path != self.model_path):
                    if os.path.exists(new_model_path):
                        logger.info(f"加载新模型: {new_model_path}")
                        if self.yolo_predictor.load_model(new_model_path):
                            self.model_path = new_model_path
                            self.auto_label_button.setEnabled(True)
                            self.auto_label_all_button.setEnabled(True)
                        else:
                            # 如果加载失败，重置模型路径
                            new_params['model_path'] = ""
                            self.settings.save_model_params(new_params)
                            self.auto_label_button.setEnabled(False)
                            self.auto_label_all_button.setEnabled(False)
                
                # 更新自动预测按钮的状态
                self.auto_label_button.setEnabled(bool(self.model_path))
                self.auto_label_all_button.setEnabled(bool(self.model_path))
                
                # 如果启用了自动预测并且当前有图像，则立即进行预测
                if new_params.get('enable_auto_predict') and self.canvas.pixmap:
                    self.auto_label_current()
                    
        except Exception as e:
            logger.error(f"打开模型设置对话框失败: {str(e)}")
            logger.error(traceback.format_exc())
            QMessageBox.warning(self, tr("Error"), f"{tr('Failed to open model settings')}: {str(e)}")

    def open_dataset_split(self):
        """打开数据集划分对话框"""
        try:
            if not self.current_folder:
                QMessageBox.warning(self, tr("警告"), tr("请先选择一个文件夹"))
                return
                
            dialog = DatasetSplitDialog(self, self.current_folder)
            dialog.exec_()
        except Exception as e:
            logger.error(f"打开数据集划分对话框失败: {str(e)}")
            QMessageBox.warning(self, tr("错误"), f"{tr('打开数据集划分对话框失败')}: {str(e)}")

    def open_class_manager(self):
        """打开类别管理对话框"""
        try:
            # 确定data.yaml的路径
            data_yaml_path = None
            if self.current_folder:
                data_yaml_path = os.path.join(self.current_folder, 'data.yaml')
            
            dialog = ClassManagerDialog(self.classes, data_yaml_path, self)
            if dialog.exec_() == QDialog.Accepted:
                # 获取修改后的类别列表
                new_classes = dialog.get_classes()
                
                # 检查是否有变化
                if dialog.has_changes():
                    # 更新类别列表
                    self.classes = new_classes
                    self.update_class_combo()
                    
                    # 如果当前有图像加载，可能需要更新标注
                    if self.canvas.pixmap:
                        # 更新类别下拉框
                        self.update_box_list()
                    
                    self.statusBar().showMessage(tr("类别已更新"), 3000)
        except Exception as e:
            logger.error(f"打开类别管理对话框失败: {str(e)}")
            QMessageBox.warning(self, tr("错误"), tr(f"打开类别管理对话框失败: {str(e)}"))

    def toggle_keypoint_mode(self):
        """切换特征点编辑模式"""
        is_checked = self.keypoint_edit_button.isChecked()
        
        # 保存当前窗口大小
        current_size = self.size()
        
        # 切换特征点编辑模式
        self.canvas.toggle_keypoint_mode(is_checked)
        
        # 更新按钮文本
        if is_checked:
            self.keypoint_edit_button.setText(tr("退出特征点编辑"))
            self.statusBar().showMessage(tr("特征点编辑模式：点击边界框内部添加特征点，双击特征点删除"))
        else:
            self.keypoint_edit_button.setText(tr("特征点编辑"))
            self.statusBar().showMessage(tr("Ready"))
            
        # 确保窗口大小不变
        self.resize(current_size)
        
    def toggle_guide_lines(self):
        """切换辅助线显示"""
        is_checked = self.guide_lines_button.isChecked()
        
        # 切换辅助线显示状态
        self.canvas.toggle_guide_lines(is_checked)
        
        # 更新按钮文本
        if is_checked:
            self.guide_lines_button.setText(tr("隐藏辅助线"))
            self.statusBar().showMessage(tr("辅助线已启用"), 2000)
        else:
            self.guide_lines_button.setText(tr("显示辅助线"))
            self.statusBar().showMessage(tr("辅助线已禁用"), 2000)

    def open_model_converter(self):
        """打开模型转换对话框"""
        try:
            converter_dialog = ModelConverterDialog(self)
            converter_dialog.exec_()
        except Exception as e:
            logger.error(f"打开模型转换对话框失败: {str(e)}")
            logger.error(f"异常详情: {traceback.format_exc()}")
            QMessageBox.warning(self, tr("错误"), f"{tr('打开模型转换对话框失败')}: {str(e)}")

    def open_model_inspector(self):
        """打开模型结构查看器对话框"""
        try:
            inspector_dialog = ModelInspectorDialog(self)
            inspector_dialog.exec_()
        except Exception as e:
            logger.error(f"打开模型结构查看器失败: {str(e)}")
            logger.error(f"异常详情: {traceback.format_exc()}")
            QMessageBox.warning(self, tr("错误"), f"{tr('打开模型结构查看器失败')}: {str(e)}")