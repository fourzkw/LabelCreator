import os
import yaml
import logging
import traceback
import shutil
from PyQt5.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                             QPushButton, QLabel, QFileDialog, QListWidget, QMessageBox,
                             QComboBox, QLineEdit, QSplitter, QAction, QTreeView,
                             QGroupBox, QFrame, QStyle, QDialog, QApplication)
from PyQt5.QtGui import QStandardItemModel, QStandardItem, QKeySequence
from PyQt5.QtCore import Qt, QDir

from models.bounding_box import BoundingBox
from ui.canvas import ImageCanvas
from i18n import tr
from utils.yolo_predictor import YOLOPredictor
from utils.settings import Settings
from ui.settings_dialog import SettingsDialog  # Import SettingsDialog directly
from ui.model_settings_dialog import ModelSettingsDialog
from training.trainer_dialog import YoloTrainerDialog

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
        self.settings = Settings(app_dir)  # Use the imported Settings class
        
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
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(8)
        
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
        
        # Save buttons with icons
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
        self.model_button = QPushButton(tr("Select Model"))
        self.model_button.setIcon(self.style().standardIcon(QStyle.SP_FileDialogDetailedView))
        self.model_button.clicked.connect(self.select_model)
        self.auto_label_button = QPushButton(tr("Auto Label"))
        self.auto_label_button.setIcon(self.style().standardIcon(QStyle.SP_CommandLink))
        self.auto_label_button.clicked.connect(self.auto_label_current)
        self.auto_label_button.setEnabled(False)

        # 批量标注按钮
        self.auto_label_all_button = QPushButton(tr("Auto Label All"))
        self.auto_label_all_button.setIcon(self.style().standardIcon(QStyle.SP_DialogApplyButton))
        self.auto_label_all_button.clicked.connect(self.auto_label_all)
        self.auto_label_all_button.setEnabled(False)


        model_layout.addWidget(self.model_button)
        model_layout.addWidget(self.auto_label_button)
        model_layout.addWidget(self.auto_label_all_button)
        
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
        left_layout.addLayout(dir_layout)
        left_layout.addWidget(folder_group)
        left_layout.addWidget(image_group)
        left_layout.addWidget(class_group)
        left_layout.addWidget(box_group)
        left_layout.addWidget(nav_group)
        left_layout.addWidget(save_group)
        left_layout.addWidget(auto_group)
        left_layout.addWidget(self.label_path_display)
        left_layout.addStretch(1)  # 添加弹性空间

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
        zoom_in_button.clicked.connect(self.canvas.zoom_in)
        zoom_out_button.clicked.connect(self.canvas.zoom_out)
        reset_zoom_button.clicked.connect(self.canvas.reset_zoom)
        zoom_layout.addWidget(zoom_in_button)
        zoom_layout.addWidget(zoom_out_button)
        zoom_layout.addWidget(reset_zoom_button)
        
        # Add widgets to right layout
        right_layout.addWidget(canvas_frame, 1)  # 让画布占据大部分空间
        right_layout.addLayout(zoom_layout)
        
        # Add panels to main layout
        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(left_panel)
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
        
        preferences_action = QAction(tr("首选项"), self)
        preferences_action.setIcon(self.style().standardIcon(QStyle.SP_FileDialogDetailedView))
        preferences_action.triggered.connect(self.show_settings)
        settings_menu.addAction(preferences_action)

        # 模型预测设置
        model_settings_action = QAction(tr("模型预测设置"), self)
        model_settings_action.setIcon(self.style().standardIcon(QStyle.SP_DesktopIcon))
        model_settings_action.triggered.connect(self.open_model_settings)
        settings_menu.addAction(model_settings_action)
        
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
        self.current_image_index = self.image_list.row(item)
        image_path = os.path.join(self.current_folder, item.text())
        self.load_image(image_path)
    
    def load_image(self, image_path):
        logger.info(f"Loading image in main window: {image_path}")
        # Check if file exists
        if not os.path.exists(image_path):
            error_msg = tr("Image file not found") + f": {image_path}"
            logger.error(error_msg)
            QMessageBox.warning(self, tr("Error"), error_msg)
            return
        
        try:    
            # Load image to canvas
            self.canvas.load_image(image_path)
            
            # If image loading failed, pixmap will be None
            if not self.canvas.pixmap:
                error_msg = tr("Failed to load image") + f": {image_path}"
                logger.error(error_msg)
                QMessageBox.warning(self, tr("Error"), error_msg)
                return
            
            # Check for existing annotation file
            label_path = self.get_label_path(image_path)
            if os.path.exists(label_path):
                logger.info(f"Found existing annotation file: {label_path}")
                self.load_annotations(label_path)
            else:
                logger.info(f"No existing annotation file found for: {image_path}")
                self.canvas.boxes = []
                self.update_box_list()
                
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
        """
        生成YOLO格式标签文件路径

        参数:
            image_path (str): 原始图像文件的绝对路径

        返回:
            str: 对应标签文件的绝对路径，保存在datasave/labels目录下

        说明:
            - 自动创建labels目录（如果不存在）
            - 标签文件名与图像文件名保持一致，扩展名为.txt
            - 路径转换示例：
              /images/train/dog.jpg → /datasave/labels/dog.txt
        """
        # 提取图像文件名（不含扩展名）
        image_filename = os.path.basename(image_path)  # 从完整路径中获取文件名
        base_name = os.path.splitext(image_filename)[0]  # 去除文件扩展名
        
        # 创建标签目录结构
        image_dir = os.path.dirname(image_path)
        parent_dir = os.path.dirname(image_dir)
        labels_dir = os.path.join(parent_dir, "labels")
        label_subdir = labels_dir
        if not os.path.exists(label_subdir):
            os.makedirs(label_subdir, exist_ok=True)
            
        # 生成标准化标签文件路径
        return os.path.normpath(  # 规范化路径分隔符
            os.path.join(label_subdir, f"{base_name}.txt")
        )
    
    def load_annotations(self, label_path):
        self.canvas.boxes = []
        
        # Check if pixmap is valid before proceeding
        if not self.canvas.pixmap or self.canvas.pixmap.isNull():
            QMessageBox.warning(self, tr("Error"), tr("Cannot load annotations: No valid image loaded"))
            return
        
        try:
            with open(label_path, 'r') as f:
                lines = f.readlines()
                
                for line in lines:
                    parts = line.strip().split()
                    if len(parts) == 5:
                        class_id = int(parts[0])
                        x_center = float(parts[1])
                        y_center = float(parts[2])
                        width = float(parts[3])
                        height = float(parts[4])
                        
                        # Ensure class exists
                        while class_id >= len(self.classes):
                            self.classes.append(f"Class {len(self.classes)}")
                        self.update_class_combo()
                        
                        # Convert from YOLO format to pixel coordinates
                        img_width = self.canvas.pixmap.width()
                        img_height = self.canvas.pixmap.height()
                        
                        x1 = (x_center - width/2) * img_width
                        y1 = (y_center - height/2) * img_height
                        x2 = (x_center + width/2) * img_width
                        y2 = (y_center + height/2) * img_height
                        
                        box = BoundingBox(x1, y1, x2, y2, class_id)
                        self.canvas.boxes.append(box)
                
                self.update_box_list()
                self.canvas.update()
        except Exception as e:
            QMessageBox.warning(self, tr("Error"), f"{tr('Failed to load annotations')}: {str(e)}")
    
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
                    del self.canvas.boxes[index]
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
        # 移除自动保存，避免仅切换类别就覆盖标签文件
        # self.save_current()  # 注释掉这行
    
    def get_class_name(self, class_id):
        if 0 <= class_id < len(self.classes):
            return self.classes[class_id]
        return f"{tr('Unknown')} ({class_id})"
    
    def prev_image(self):
        if self.current_image_index > 0:
            self.current_image_index -= 1
            self.image_list.setCurrentRow(self.current_image_index)
            image_path = os.path.join(self.current_folder, self.image_files[self.current_image_index])
            self.load_image(image_path)
    
    def next_image(self):
        if self.current_image_index < len(self.image_files) - 1:
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
        # Check if pixmap is valid before proceeding
        if not self.canvas.pixmap or self.canvas.pixmap.isNull():
            QMessageBox.warning(self, tr("Error"), tr("Cannot save annotations: No valid image loaded"))
            return False
            
        try:
            img_width = self.canvas.pixmap.width()
            img_height = self.canvas.pixmap.height()
            
            # 保存到项目目录和图片目录
            self._write_label_file(label_path, img_width, img_height)
            
            # 保存类别名称到classes.txt（项目目录）
            classes_path = os.path.join(os.path.dirname(label_path), "classes.txt")
            with open(classes_path, 'w') as f:
                for class_name in self.classes:
                    f.write(f"{class_name}\n")
                    
            return True
        except Exception as e:
            logger.error(f"Error saving annotations: {str(e)}")
            QMessageBox.warning(self, tr("Error"), f"{tr('Failed to save annotations')}: {str(e)}")
            return False

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
    
    def _write_label_file(self, path, img_width, img_height):
        """通用标签文件写入方法"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            for box in self.canvas.boxes:
                yolo_box = box.to_yolo_format(img_width, img_height)
                line = f"{yolo_box[0]} {yolo_box[1]:.6f} {yolo_box[2]:.6f} {yolo_box[3]:.6f} {yolo_box[4]:.6f}\n"
                f.write(line)
        self.update_data_yaml()
    
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
            
            if boxes:
                # 不再弹出确认对话框，直接添加预测的边界框，保留现有标注
                self.canvas.boxes.extend(boxes)
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
                
                # 执行预测
                boxes = self.yolo_predictor.predict(image_path)
                
                if boxes:
                    # 添加预测的边界框，保留现有标注
                    self.canvas.boxes.extend(boxes)
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
        """设置应用程序快捷键"""
        # 保存当前
        save_shortcut = QKeySequence(self.settings.get_shortcut('save_current'))
        save_action = QAction(tr("保存当前"), self)
        save_action.setShortcut(save_shortcut)
        save_action.triggered.connect(self.save_current)
        self.addAction(save_action)
        
        # 保存全部
        save_all_shortcut = QKeySequence(self.settings.get_shortcut('save_all'))
        save_all_action = QAction(tr("保存全部"), self)
        save_all_action.setShortcut(save_all_shortcut)
        save_all_action.triggered.connect(self.save_all)
        self.addAction(save_all_action)
        
        # 上一张图像
        prev_shortcut = QKeySequence(self.settings.get_shortcut('prev_image'))
        prev_action = QAction(tr("上一张图像"), self)
        prev_action.setShortcut(prev_shortcut)
        prev_action.triggered.connect(self.prev_image)
        self.addAction(prev_action)
        
        # 下一张图像
        next_shortcut = QKeySequence(self.settings.get_shortcut('next_image'))
        next_action = QAction(tr("下一张图像"), self)
        next_action.setShortcut(next_shortcut)
        next_action.triggered.connect(self.next_image)
        self.addAction(next_action)
        
        # 删除选中框
        delete_shortcut = QKeySequence(self.settings.get_shortcut('delete_box'))
        delete_action = QAction(tr("删除选中框"), self)
        delete_action.setShortcut(delete_shortcut)
        delete_action.triggered.connect(self.delete_selected_box)
        self.addAction(delete_action)
        
        # 放大
        zoom_in_shortcut = QKeySequence(self.settings.get_shortcut('zoom_in'))
        zoom_in_action = QAction(tr("放大"), self)
        zoom_in_action.setShortcut(zoom_in_shortcut)
        zoom_in_action.triggered.connect(self.canvas.zoom_in)
        self.addAction(zoom_in_action)
        
        # 缩小
        zoom_out_shortcut = QKeySequence(self.settings.get_shortcut('zoom_out'))
        zoom_out_action = QAction(tr("缩小"), self)
        zoom_out_action.setShortcut(zoom_out_shortcut)
        zoom_out_action.triggered.connect(self.canvas.zoom_out)
        self.addAction(zoom_out_action)
        
        # 重置缩放
        reset_zoom_shortcut = QKeySequence(self.settings.get_shortcut('reset_zoom'))
        reset_zoom_action = QAction(tr("重置缩放"), self)
        reset_zoom_action.setShortcut(reset_zoom_shortcut)
        reset_zoom_action.triggered.connect(self.canvas.reset_zoom)
        self.addAction(reset_zoom_action)
        
        # 打开目录
        open_dir_shortcut = QKeySequence(self.settings.get_shortcut('open_directory'))
        open_dir_action = QAction(tr("打开目录"), self)
        open_dir_action.setShortcut(open_dir_shortcut)
        open_dir_action.triggered.connect(self.select_directory)
        self.addAction(open_dir_action)
        
        # 退出
        exit_shortcut = QKeySequence(self.settings.get_shortcut('exit'))
        exit_action = QAction(tr("退出"), self)
        exit_action.setShortcut(exit_shortcut)
        exit_action.triggered.connect(self.close)
        self.addAction(exit_action)
        
        # 自动标注
        auto_label_shortcut = QKeySequence(self.settings.get_shortcut('auto_label'))
        auto_label_action = QAction(tr("自动标注"), self)
        auto_label_action.setShortcut(auto_label_shortcut)
        auto_label_action.triggered.connect(self.auto_label_current)
        self.addAction(auto_label_action)
        # 批量标注快捷键
        auto_label_all_shortcut = QKeySequence(self.settings.get_shortcut('auto_label_all'))
        auto_label_all_action = QAction(tr("批量自动标注"), self)
        auto_label_all_action.setShortcut(auto_label_all_shortcut)
        auto_label_all_action.triggered.connect(self.auto_label_all)
        self.addAction(auto_label_all_action)

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
        """打开模型预测设置对话框"""
        dialog = ModelSettingsDialog(self)
        if dialog.exec_() == QDialog.Accepted:
            # 如果设置已保存，更新预测器的参数
            from utils.config import Config
            config = Config()
            model_params = config.get_model_params()
            
            # 更新预测器参数
            if self.yolo_predictor:
                self.yolo_predictor.set_params(
                    conf_threshold=model_params.get("confidence_threshold", 0.5),
                    iou_threshold=model_params.get("iou_threshold", 0.45),
                    max_detections=model_params.get("max_detections", 100),
                    device=model_params.get("device", "cpu")
                )
                
                # 如果模型路径已设置且不同于当前路径，尝试加载新模型
                model_path = model_params.get("model_path", "")
                if model_path and model_path != self.model_path and os.path.exists(model_path):
                    if self.yolo_predictor.load_model(model_path):
                        self.model_path = model_path
                        self.model_label.setText(os.path.basename(model_path))
                        self.auto_label_button.setEnabled(True)
                        self.auto_label_all_button.setEnabled(True)
                        self.statusBar().showMessage(tr("模型已更新"), 3000)
                # 如果模型路径相同但设备改变，重新加载模型
                elif model_path and model_path == self.model_path and os.path.exists(model_path):
                    if self.yolo_predictor.load_model(model_path):
                        self.statusBar().showMessage(tr("模型已在新设备上重新加载"), 3000)
            
            # 检查是否启用自动预测
            if model_params.get("enable_auto_predict", False):
                # 这里可以添加自动预测的逻辑
                pass
                
            self.statusBar().showMessage(tr("模型设置已更新"), 3000)