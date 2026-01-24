import os
import random
import shutil
import yaml
from PyQt5.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QFormLayout, 
                            QLabel, QPushButton, QDoubleSpinBox, QGroupBox,
                            QRadioButton, QFileDialog, QLineEdit, QMessageBox,
                            QCheckBox, QProgressDialog)
from PyQt5.QtCore import Qt, QSettings
from utils.logger import setup_logger
from i18n import tr

logger = setup_logger('YOLOLabelCreator.DatasetSplit')

class DatasetSplitDialog(QDialog):
    """数据集划分对话框"""
    
    def __init__(self, parent=None, current_dir=None):
        super().__init__(parent)
        self.settings = QSettings()
        self.current_dir = current_dir  # 保存当前打开的目录
        self._image_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')
        
        self.setWindowTitle(tr("数据集划分"))
        self.setMinimumWidth(500)
        self.setup_ui()
        
        # 如果提供了当前目录，则自动设置源路径
        if self.current_dir and os.path.exists(self.current_dir):
            self.source_path.setText(self.current_dir)
            # 默认设置输出路径为源路径的父目录下的dataset_split
            parent_dir = os.path.dirname(self.current_dir)
            self.output_path.setText(os.path.join(parent_dir, "dataset_split"))
            self.update_split_button()
            self.update_current_total_count()
            
        self.load_settings()
        
    def setup_ui(self):
        """设置UI界面"""
        layout = QVBoxLayout(self)
        
        # 数据集路径设置
        dataset_group = QGroupBox(tr("数据集路径"))
        dataset_layout = QFormLayout()
        
        # 源数据集路径 - 只显示，不可选择
        self.source_path = QLineEdit()
        self.source_path.setReadOnly(True)
        dataset_layout.addRow(tr("源数据集路径:"), self.source_path)
        
        # 输出路径
        self.output_path = QLineEdit()
        self.output_path.setReadOnly(True)
        self.output_browse = QPushButton(tr("浏览..."))
        self.output_browse.clicked.connect(self.browse_output)
        output_layout = QHBoxLayout()
        output_layout.addWidget(self.output_path)
        output_layout.addWidget(self.output_browse)
        dataset_layout.addRow(tr("输出路径:"), output_layout)
        
        dataset_group.setLayout(dataset_layout)
        layout.addWidget(dataset_group)
        
        # 划分比例设置
        ratio_group = QGroupBox(tr("划分比例"))
        ratio_layout = QFormLayout()
        
        # 训练集比例
        self.train_ratio = QDoubleSpinBox()
        self.train_ratio.setRange(0.1, 0.9)
        self.train_ratio.setSingleStep(0.05)
        self.train_ratio.setValue(0.7)
        self.train_ratio.valueChanged.connect(self.update_test_ratio)
        ratio_layout.addRow(tr("训练集比例:"), self.train_ratio)
        
        # 验证集比例
        self.val_ratio = QDoubleSpinBox()
        self.val_ratio.setRange(0.1, 0.3)
        self.val_ratio.setSingleStep(0.05)
        self.val_ratio.setValue(0.2)
        self.val_ratio.valueChanged.connect(self.update_test_ratio)
        ratio_layout.addRow(tr("验证集比例:"), self.val_ratio)
        
        # 测试集比例（自动计算）
        self.test_ratio_label = QLabel("0.1")
        ratio_layout.addRow(tr("测试集比例:"), self.test_ratio_label)
        
        # 随机种子
        self.random_seed = QDoubleSpinBox()
        self.random_seed.setRange(0, 9999)
        self.random_seed.setDecimals(0)
        self.random_seed.setValue(42)
        ratio_layout.addRow(tr("随机种子:"), self.random_seed)
        
        # 创建YAML文件选项
        self.create_yaml = QCheckBox(tr("创建YAML配置文件"))
        self.create_yaml.setChecked(True)
        ratio_layout.addRow("", self.create_yaml)

        # 仅划分有标签图片（默认保持原行为）
        self.only_labeled = QCheckBox(tr("仅划分有标签的图片"))
        self.only_labeled.setChecked(True)
        self.only_labeled.stateChanged.connect(self.update_current_total_count)
        ratio_layout.addRow("", self.only_labeled)

        # 当前划分图片总数显示
        self.current_total_label = QLabel("0")
        ratio_layout.addRow(tr("当前划分图片总数:"), self.current_total_label)
        
        ratio_group.setLayout(ratio_layout)
        layout.addWidget(ratio_group)
        
        # 按钮区域
        button_layout = QHBoxLayout()
        self.split_button = QPushButton(tr("开始划分"))
        self.split_button.clicked.connect(self.split_dataset)
        self.cancel_button = QPushButton(tr("取消"))
        self.cancel_button.clicked.connect(self.reject)
        
        button_layout.addWidget(self.split_button)
        button_layout.addWidget(self.cancel_button)
        layout.addLayout(button_layout)
        
        # 初始化测试集比例
        self.update_test_ratio()
        self.update_current_total_count()
    
    def update_test_ratio(self):
        """更新测试集比例"""
        train = self.train_ratio.value()
        val = self.val_ratio.value()
        test = 1.0 - train - val
        
        # 确保测试集比例不为负
        if test < 0:
            test = 0
            val = 1.0 - train
            self.val_ratio.blockSignals(True)
            self.val_ratio.setValue(val)
            self.val_ratio.blockSignals(False)
        
        self.test_ratio_label.setText(f"{test:.2f}")

    def _has_label_content(self, label_path):
        """判断标签文件是否包含有效标注内容（非空白行）"""
        if not label_path or not os.path.exists(label_path):
            return False
        try:
            with open(label_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        return True
        except Exception as e:
            logger.error(f"读取标签文件失败: {label_path}, 错误: {str(e)}")
        return False

    def _scan_images_and_labels(self, source_path):
        """
        扫描目录（含子目录）收集图片与标签文件。

        Returns:
            tuple[list[str], dict[str, list[str]]]: (image_files, labels_by_basename)
        """
        image_files = []
        labels_by_basename = {}

        if not source_path or not os.path.exists(source_path):
            return image_files, labels_by_basename

        for root, _, files in os.walk(source_path):
            for file in files:
                file_path = os.path.join(root, file)
                file_lower = file.lower()

                # 图像
                if file_lower.endswith(self._image_extensions):
                    image_files.append(file_path)
                    continue

                # 标签
                if file_lower.endswith('.txt'):
                    # 排除classes.txt等非标签文件
                    if file_lower in ('classes.txt', 'data.yaml'):
                        continue
                    base = os.path.splitext(os.path.basename(file))[0]
                    labels_by_basename.setdefault(base, []).append(file_path)

        return image_files, labels_by_basename

    def update_current_total_count(self):
        """更新当前可划分图片总数显示"""
        source_path = self.source_path.text()
        if not source_path or not os.path.exists(source_path):
            self.current_total_label.setText("0")
            return

        image_files, labels_by_basename = self._scan_images_and_labels(source_path)
        only_labeled = self.only_labeled.isChecked() if hasattr(self, 'only_labeled') else True

        eligible = 0
        for img_path in image_files:
            img_name = os.path.splitext(os.path.basename(img_path))[0]
            # 优先同目录同名标签
            label_path = os.path.join(os.path.dirname(img_path), f"{img_name}.txt")
            if not os.path.exists(label_path):
                # 退化：全局同名标签
                candidates = labels_by_basename.get(img_name, [])
                label_path = candidates[0] if candidates else None

            if only_labeled:
                if self._has_label_content(label_path):
                    eligible += 1
            else:
                eligible += 1

        self.current_total_label.setText(str(eligible))
    
    def browse_output(self):
        """浏览输出路径"""
        folder_path = QFileDialog.getExistingDirectory(
            self, tr("选择输出文件夹"), ""
        )
        if folder_path:
            self.output_path.setText(folder_path)
            self.update_split_button()
    
    def update_split_button(self):
        """更新划分按钮状态"""
        source_path = self.source_path.text()
        output_path = self.output_path.text()
        
        self.split_button.setEnabled(
            bool(source_path) and bool(output_path)
        )
    
    def load_settings(self):
        """加载设置"""
        self.output_path.setText(self.settings.value("dataset_split/output_path", ""))
        self.train_ratio.setValue(float(self.settings.value("dataset_split/train_ratio", 0.7)))
        self.val_ratio.setValue(float(self.settings.value("dataset_split/val_ratio", 0.2)))
        self.random_seed.setValue(int(self.settings.value("dataset_split/random_seed", 42)))
        self.create_yaml.setChecked(self.settings.value("dataset_split/create_yaml", True, type=bool))
        self.only_labeled.setChecked(self.settings.value("dataset_split/only_labeled", True, type=bool))
        
        self.update_split_button()
        self.update_current_total_count()
    
    def save_settings(self):
        """保存设置"""
        self.settings.setValue("dataset_split/output_path", self.output_path.text())
        self.settings.setValue("dataset_split/train_ratio", self.train_ratio.value())
        self.settings.setValue("dataset_split/val_ratio", self.val_ratio.value())
        self.settings.setValue("dataset_split/random_seed", int(self.random_seed.value()))
        self.settings.setValue("dataset_split/create_yaml", self.create_yaml.isChecked())
        self.settings.setValue("dataset_split/only_labeled", self.only_labeled.isChecked())
        self.settings.sync()
    
    def split_dataset(self):
        """划分数据集"""
        source_path = self.source_path.text()
        output_path = self.output_path.text()
        train_ratio = self.train_ratio.value()
        val_ratio = self.val_ratio.value()
        test_ratio = 1.0 - train_ratio - val_ratio
        random_seed = int(self.random_seed.value())
        create_yaml = self.create_yaml.isChecked()
        only_labeled = self.only_labeled.isChecked()
        
        # 保存设置
        self.save_settings()
        
        try:
            # 检查源路径是否存在
            if not os.path.exists(source_path):
                QMessageBox.critical(self, tr("错误"), tr(f"源数据集路径不存在: {source_path}"))
                logger.error(f"源数据集路径不存在: {source_path}")
                return
                
            # 确保输出目录存在
            os.makedirs(output_path, exist_ok=True)
            
            # 获取所有图像和标签文件（包括子目录）
            image_files, labels_by_basename = self._scan_images_and_labels(source_path)
            logger.info(f"找到 {len(image_files)} 个图像文件")

            # 匹配图像和标签（允许无标签）
            all_pairs = []
            labeled_pairs = []  # 仅用于推断类别/姿态检测等需要标签内容的逻辑

            for img_path in image_files:
                img_dir = os.path.dirname(img_path)
                img_name = os.path.splitext(os.path.basename(img_path))[0]

                # 优先同一目录同名标签
                label_path = os.path.join(img_dir, f"{img_name}.txt")
                if not os.path.exists(label_path):
                    candidates = labels_by_basename.get(img_name, [])
                    label_path = candidates[0] if candidates else None

                has_content = self._has_label_content(label_path)

                if only_labeled and not has_content:
                    continue

                all_pairs.append((img_path, label_path))
                if has_content:
                    labeled_pairs.append((img_path, label_path))

            if not all_pairs:
                QMessageBox.warning(self, tr("警告"), tr("没有找到可用于划分的图片!"))
                logger.warning(f"在 {source_path} 中没有找到可用于划分的图片 (only_labeled={only_labeled})")
                return
            
            # 检查是否是姿态检测数据集（查看标签文件中是否有关键点数据）
            is_pose_dataset = False
            sample_label_path = labeled_pairs[0][1] if labeled_pairs else None
            if sample_label_path:
                try:
                    with open(sample_label_path, 'r', encoding='utf-8') as f:
                        label_content = f.readline().strip()
                        parts = label_content.split()
                        # 如果标签行包含的数值超过5个，则认为这是一个包含关键点的姿态检测数据集
                        # 标准YOLO格式为：class_id x_center y_center width height
                        # 姿态检测格式为：class_id x_center y_center width height x1 y1 x2 y2 ...
                        if len(parts) > 5:
                            is_pose_dataset = True
                            logger.info(f"检测到姿态检测数据集，包含关键点数据")
                except Exception as e:
                    logger.error(f"检查数据集类型时出错: {str(e)}")
                    # 即使出错也继续执行，当作普通数据集处理

            logger.info(f"可划分图片数量: {len(all_pairs)} (含标注内容: {len(labeled_pairs)})")
            
            # 设置随机种子
            random.seed(random_seed)
            # 随机打乱文件列表
            random.shuffle(all_pairs)
            
            # 计算每个集合的文件数量
            total_files = len(all_pairs)
            train_count = int(total_files * train_ratio)
            val_count = int(total_files * val_ratio)
            test_count = total_files - train_count - val_count
            
            # 划分文件
            train_files = all_pairs[:train_count]
            val_files = all_pairs[train_count:train_count+val_count]
            test_files = all_pairs[train_count+val_count:]
            
            # 创建进度对话框
            progress = QProgressDialog(tr("正在划分数据集..."), tr("取消"), 0, total_files, self)
            progress.setWindowTitle(tr("数据集划分"))
            progress.setWindowModality(Qt.WindowModal)
            progress.show()
            
            # 创建输出目录结构
            for split in ["train", "val", "test"]:
                for subdir in ["images", "labels"]:
                    split_dir = os.path.join(output_path, split, subdir)
                    try:
                        os.makedirs(split_dir, exist_ok=True)
                        logger.info(f"创建目录: {split_dir}")
                    except Exception as e:
                        QMessageBox.critical(self, tr("错误"), tr(f"无法创建目录: {split_dir}\n错误: {str(e)}"))
                        logger.error(f"无法创建目录: {split_dir}, 错误: {str(e)}")
                        return
            
            # 复制文件
            processed = 0
            
            # 复制训练集文件
            for img_path, label_path in train_files:
                if progress.wasCanceled():
                    break
                
                if not self._copy_file_pair(img_path, label_path, output_path, "train"):
                    continue
                
                processed += 1
                progress.setValue(processed)
            
            # 复制验证集文件
            for img_path, label_path in val_files:
                if progress.wasCanceled():
                    break
                
                if not self._copy_file_pair(img_path, label_path, output_path, "val"):
                    continue
                
                processed += 1
                progress.setValue(processed)
            
            # 复制测试集文件
            for img_path, label_path in test_files:
                if progress.wasCanceled():
                    break
                
                if not self._copy_file_pair(img_path, label_path, output_path, "test"):
                    continue
                
                processed += 1
                progress.setValue(processed)
            
            progress.setValue(total_files)
            
            # 如果需要创建YAML文件
            if create_yaml and not progress.wasCanceled():
                # 首先尝试从当前目录的data.yaml读取类别信息
                yaml_path = os.path.join(source_path, "data.yaml")
                classes = []
                
                if os.path.exists(yaml_path):
                    try:
                        with open(yaml_path, 'r', encoding='utf-8') as f:
                            yaml_data = yaml.safe_load(f)
                            if 'names' in yaml_data and isinstance(yaml_data['names'], list):
                                classes = yaml_data['names']
                                logger.info(f"从data.yaml读取到{len(classes)}个类别")
                    except Exception as e:
                        logger.error(f"读取data.yaml失败: {str(e)}")
                
                # 如果从yaml文件中没有读取到类别，尝试从classes.txt读取
                if not classes:
                    classes_file = os.path.join(source_path, "classes.txt")
                    if os.path.exists(classes_file):
                        try:
                            with open(classes_file, 'r', encoding='utf-8') as f:
                                classes = [line.strip() for line in f.readlines() if line.strip()]
                                logger.info(f"从classes.txt读取到{len(classes)}个类别")
                        except Exception as e:
                            logger.error(f"读取classes.txt失败: {str(e)}")
                
                # 如果仍然没有类别信息，尝试从标签文件中推断
                if not classes:
                    max_class = -1
                    for _, label_path in labeled_pairs:
                        if not label_path:
                            continue
                        try:
                            with open(label_path, 'r', encoding='utf-8') as f:
                                for line in f:
                                    parts = line.strip().split()
                                    if parts:
                                        class_id = int(parts[0])
                                        max_class = max(max_class, class_id)
                        except Exception as e:
                            logger.error(f"读取标签文件失败: {label_path}, 错误: {str(e)}")
                            continue
                    
                    if max_class >= 0:
                        classes = [f"class{i}" for i in range(max_class + 1)]
                        logger.info(f"从标签文件推断出{len(classes)}个类别")
                
                # 创建YAML文件
                yaml_path = os.path.join(output_path, "data.yaml")
                try:
                    with open(yaml_path, 'w', encoding='utf-8') as f:
                        f.write(f"# YOLOv8 数据集配置\n")
                        f.write(f"path: {output_path}\n")
                        f.write(f"train: train/images\n")
                        f.write(f"val: val/images\n")
                        f.write(f"test: test/images\n\n")
                        f.write(f"nc: {len(classes)}\n")
                        f.write(f"names: {classes}\n")
                        
                        # 如果是姿态检测数据集，添加关键点配置
                        if is_pose_dataset:
                            # 尝试确定关键点数量
                            try:
                                with open(sample_label_path, 'r', encoding='utf-8') as label_f:
                                    label_content = label_f.readline().strip()
                                    parts = label_content.split()
                                    # 计算关键点对数量：(总参数数量 - 5) / 2
                                    # 5个是class_id、x_center、y_center、width、height
                                    kpt_count = (len(parts) - 5) // 2
                                    
                                    # 检查是否有可见性参数 (dim = 3)
                                    has_visibility = (len(parts) - 5) % 3 == 0
                                    kpt_dim = 3 if has_visibility else 2
                                    
                                    if has_visibility:
                                        # 如果有可见性参数，重新计算关键点数量
                                        kpt_count = (len(parts) - 5) // 3
                                    
                                    f.write(f"\n# 关键点配置\n")
                                    f.write(f"kpt_shape: [{kpt_count}, {kpt_dim}]  # 关键点数量, 维度(2为x,y或3为x,y,visible)\n")
                                    
                                    # 添加默认的翻转索引（假设关键点是对称的）
                                    # 这里只是提供一个默认值，用户可能需要手动调整
                                    flip_idx = list(range(kpt_count))
                                    f.write(f"flip_idx: {flip_idx}  # 对称关键点的翻转索引，需要根据实际情况调整\n")
                                    
                                    logger.info(f"为姿态检测数据集添加了关键点配置: {kpt_count}个关键点, {kpt_dim}维")
                            except Exception as e:
                                logger.error(f"生成关键点配置时出错: {str(e)}")
                                # 添加一个通用的关键点配置，提示用户手动修改
                                f.write(f"\n# 关键点配置 (请根据实际数据集调整)\n")
                                f.write(f"kpt_shape: [17, 3]  # 关键点数量, 维度\n")
                                f.write(f"flip_idx: [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15]  # 对称关键点的翻转索引\n")
                    
                    logger.info(f"成功创建YAML文件: {yaml_path}")
                except Exception as e:
                    logger.error(f"创建YAML文件失败: {yaml_path}, 错误: {str(e)}")
                    QMessageBox.warning(self, tr("警告"), tr(f"创建YAML文件失败: {str(e)}"))
            
            if not progress.wasCanceled():
                # 构建完成消息
                complete_message = tr(f"数据集划分完成!\n"
                                      f"训练集: {len(train_files)} 文件\n"
                                      f"验证集: {len(val_files)} 文件\n"
                                      f"测试集: {len(test_files)} 文件")
                
                # 如果是姿态检测数据集，添加额外提示
                if is_pose_dataset and create_yaml:
                    complete_message += tr("\n\n检测到姿态检测数据集!\n"
                                          "已在data.yaml中添加了关键点配置。\n"
                                          "请检查data.yaml文件中的kpt_shape和flip_idx是否正确，\n"
                                          "并根据实际情况调整flip_idx（关键点对称翻转索引）。")
                
                QMessageBox.information(self, tr("完成"), complete_message)
                self.accept()
            
        except Exception as e:
            logger.error(f"划分数据集时出错: {str(e)}")
            logger.exception("详细错误信息")  # 添加详细的异常堆栈信息
            QMessageBox.critical(self, tr("错误"), tr(f"划分数据集时出错: {str(e)}"))
    
    def _copy_file_pair(self, img_path, label_path, output_path, split_type):
        """
        复制图像和标签文件对到指定的输出目录
        
        Args:
            img_path (str): 源图像文件路径
            label_path (str): 源标签文件路径
            output_path (str): 输出根目录
            split_type (str): 数据集类型 ("train", "val", "test")
            
        Returns:
            bool: 复制是否成功
        """
        img_file = os.path.basename(img_path)
        # label_path 可能为空（允许无标签图片）
        label_file = f"{os.path.splitext(img_file)[0]}.txt"
        
        # 复制图像
        dest_img_path = os.path.join(output_path, split_type, "images", img_file)
        try:
            shutil.copy2(img_path, dest_img_path)
        except Exception as e:
            logger.error(f"复制图像文件失败: {img_path} -> {dest_img_path}, 错误: {str(e)}")
            return False
        
        # 复制标签 - 使用读写方式确保完整复制特征点数据
        dest_label_path = os.path.join(output_path, split_type, "labels", label_file)
        try:
            # label_path 可能为空（允许无标签图片）
            content = ""
            if label_path and os.path.exists(label_path):
                # 使用读写模式复制文件内容，而不是直接复制文件
                with open(label_path, 'r', encoding='utf-8') as src_file:
                    content = src_file.read()

            # YOLO 训练通常期望 labels 下有同名 txt（无标注则为空文件）
            with open(dest_label_path, 'w', encoding='utf-8') as dst_file:
                dst_file.write(content)
        except Exception as e:
            logger.error(f"复制标签文件失败: {label_path} -> {dest_label_path}, 错误: {str(e)}")
            return False
            
        return True