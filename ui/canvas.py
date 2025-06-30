import os
import traceback
import logging
import numpy as np
from PyQt5.QtWidgets import QWidget, QMessageBox
from PyQt5.QtGui import QPainter, QPen, QColor, QPixmap, QCursor, QFont, QBrush
from PyQt5.QtCore import Qt, QPoint

from models.bounding_box import BoundingBox
from i18n import tr

# 获取日志记录器
logger = logging.getLogger('YOLOLabelCreator.Canvas')

class ImageCanvas(QWidget):
    """
    图像标注画布组件
    
    Attributes:
        pixmap (QPixmap): 当前显示的图像对象
        boxes (list): 存储当前图像的所有边界框
        current_box (BoundingBox): 正在绘制的临时边界框
        scale_factor (float): 图像缩放比例
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        # 初始化画布基础属性
        self.parent = parent
        self.pixmap = None
        self.image_path = None
        self.boxes = []
        self.current_box = None
        self.start_point = None
        self.current_class_id = 0
        self.scale_factor = 1.0
        
        # 添加编辑模式相关属性
        self.selected_box_index = -1  # 当前选中的边界框索引
        self.edit_mode = None  # 编辑模式：'move', 'resize-edge', 'resize-corner'
        self.edit_handle = None  # 当前编辑的边缘或角点
        self.last_cursor_pos = None  # 上一次鼠标位置
        
        # 添加图像拖动相关属性
        self.is_panning = False  # 是否正在拖动图像
        self.pan_start_pos = None  # 拖动开始位置
        self.offset_x = 0  # 图像X轴偏移量
        self.offset_y = 0  # 图像Y轴偏移量
        
        # 设置鼠标跟踪，以便接收mouseMoveEvent即使没有按下鼠标按钮
        self.setMouseTracking(True)
        
        # 设置画布最小尺寸
        self.setMinimumSize(600, 400)
        
        # 设置背景色和边框
        self.setStyleSheet("""
            background-color: #f0f0f0;
            border: 1px solid #cccccc;
        """)

        # 添加特征点编辑相关属性
        self.keypoint_edit_mode = False  # 特征点编辑模式标志
        self.current_keypoint = None  # 当前正在编辑的特征点
        self.keypoint_radius = 3  # 特征点半径
        
        # 添加特征点移动相关属性
        self.moving_keypoint = False  # 是否正在移动特征点
        self.moving_keypoint_box_index = -1  # 正在移动特征点所属的边界框索引
        self.moving_keypoint_index = -1  # 正在移动的特征点索引
        
        # 添加辅助线属性
        self.guide_lines_enabled = True  # 是否启用辅助线
        self.mouse_pos = None  # 当前鼠标位置
    def load_image(self, image_path):
        """
        加载图像文件到画布
        
        该方法仅负责图像加载，标签的读取由MainWindow类负责。
        加载过程中会保存当前状态，以便在出错时能够恢复。
        
        Args:
            image_path (str): 要加载的图像文件路径
            
        Raises:
            FileNotFoundError: 图像文件不存在时抛出
            ValueError: 图像格式不支持时抛出
            RuntimeError: 图像加载失败时抛出
        """
        self.image_path = image_path
        logger.info(f"开始加载图像: {image_path}")
        
        # 保留当前标注数据作为回滚点
        previous_boxes = self.boxes.copy() if self.boxes else []
        previous_pixmap = self.pixmap.copy() if self.pixmap else None
        
        try:
            # 文件存在性检查
            if not os.path.exists(image_path):
                logger.error(f"图像文件不存在: {image_path}")
                raise FileNotFoundError(tr("图像文件未找到"))

            # 文件格式校验
            valid_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
            if not any(image_path.lower().endswith(ext) for ext in valid_extensions):
                logger.error(f"不支持的图像格式: {image_path}")
                raise ValueError(tr("不支持的图像格式"))

            # 加载QPixmap并验证有效性
            self.pixmap = QPixmap(image_path)
            if self.pixmap.isNull():
                logger.error(f"QPixmap创建失败: {image_path}")
                raise RuntimeError(tr("图像加载失败"))
                
            # 更新标签路径显示（但不读取标签，由MainWindow负责）
            label_path = self.parent.get_label_path(image_path)
            self.parent.label_path_display.setText(
                tr("标签路径：") + f"{label_path}" + 
                (tr(" (不存在)") if not os.path.exists(label_path) else "")
            )
            
            # 触发界面更新
            self.update()
            logger.info(f"图像加载成功: {image_path}")

        except FileNotFoundError as e:
            # 文件不存在异常处理
            logger.error(f"文件未找到: {str(e)}")
            self._restore_previous_state(previous_pixmap, previous_boxes)
            raise

        except Exception as e:
            # 通用异常处理
            logger.error(f"图像加载失败: {str(e)}\n{traceback.format_exc()}")
            self._restore_previous_state(previous_pixmap, previous_boxes)
            raise

        finally:
            # 资源清理
            if previous_pixmap:
                del previous_pixmap

    def _restore_previous_state(self, previous_pixmap, previous_boxes):
        """恢复到之前的状态"""
        self.pixmap = previous_pixmap
        self.boxes = previous_boxes

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        # 绘制棋盘格背景
        painter.fillRect(self.rect(), QColor(240, 240, 240))
        
        # 绘制网格背景
        grid_size = 20
        painter.setPen(QPen(QColor(230, 230, 230), 1))
        for x in range(0, self.width(), grid_size):
            painter.drawLine(x, 0, x, self.height())
        for y in range(0, self.height(), grid_size):
            painter.drawLine(0, y, self.width(), y)
        
        if self.pixmap:
            # Calculate scaled dimensions while maintaining aspect ratio
            scaled_pixmap = self.pixmap.scaled(int(self.width() * self.scale_factor), 
                                              int(self.height() * self.scale_factor),
                                              Qt.KeepAspectRatio)
            
            # Center the image and apply pan offset
            x_offset = (self.width() - scaled_pixmap.width()) // 2 + self.offset_x
            y_offset = (self.height() - scaled_pixmap.height()) // 2 + self.offset_y
            
            # 绘制图像阴影
            shadow_offset = 5
            painter.fillRect(
                x_offset + shadow_offset, 
                y_offset + shadow_offset, 
                scaled_pixmap.width(), 
                scaled_pixmap.height(), 
                QColor(0, 0, 0, 30)
            )
            
            # Draw the image
            painter.drawPixmap(x_offset, y_offset, scaled_pixmap)
            
            # 绘制图像边框
            painter.setPen(QPen(QColor(180, 180, 180), 1))
            painter.drawRect(x_offset, y_offset, scaled_pixmap.width(), scaled_pixmap.height())
            
            # 绘制所有边界框
            for i, box in enumerate(self.boxes):
                # 选中的边界框使用不同的样式
                is_selected = (i == self.selected_box_index)
                self.draw_box(painter, box, x_offset, y_offset, scaled_pixmap.width(), scaled_pixmap.height(), is_selected)

                # 绘制特征点（如果有）
                if box.has_keypoints():
                    painter.save()  # 保存当前绘图状态
                    keypoints = box.get_keypoints()
                    # 设置特征点绘制样式
                    painter.setPen(QPen(QColor(255, 0, 255), 2))
                    painter.setBrush(QBrush(QColor(255, 0, 255, 180)))
                    
                    # 获取原始图像尺寸
                    orig_width = self.pixmap.width()
                    orig_height = self.pixmap.height()
                    
                    # 计算缩放比例
                    scale_x = scaled_pixmap.width() / orig_width
                    scale_y = scaled_pixmap.height() / orig_height
                    
                    # 设置特征点编号的字体
                    font = QFont()
                    font.setPointSize(8)
                    font.setBold(True)
                    painter.setFont(font)
                    
                    # 计算特征点在画布上的位置
                    for kp_idx, kp in enumerate(keypoints):
                        # 特征点数据只有 x, y 两个值
                        kp_x, kp_y = kp
                            
                        # 应用与边界框相同的缩放和偏移逻辑
                        kp_x = x_offset + kp_x * scale_x
                        kp_y = y_offset + kp_y * scale_y
                        
                        # 绘制特征点（小圆点）
                        point_radius = 3
                        painter.drawEllipse(QPoint(int(kp_x), int(kp_y)), point_radius, point_radius)
                        
                        # 绘制特征点编号
                        # 设置白色背景以增强可读性
                        text = str(kp_idx)
                        text_rect = painter.fontMetrics().boundingRect(text)
                        text_x = int(kp_x) + point_radius + 2
                        text_y = int(kp_y)
                        
                        # 绘制文本背景
                        bg_rect = text_rect.adjusted(-2, -2, 2, 2)
                        bg_rect.moveCenter(QPoint(int(text_x + text_rect.width()/2), int(text_y)))
                        painter.fillRect(bg_rect, QColor(255, 255, 255, 200))
                        
                        # 绘制编号文本
                        painter.drawText(int(text_x), int(text_y + text_rect.height()/2), text)
                    
                    painter.restore()  # 恢复绘图状态
            
            # Draw the box being created
            if self.current_box:
                self.draw_box(painter, self.current_box, x_offset, y_offset, scaled_pixmap.width(), scaled_pixmap.height())
                
            # 绘制辅助线（十字线）
            if self.guide_lines_enabled and self.mouse_pos and self.pixmap:
                mouse_x, mouse_y = self.mouse_pos.x(), self.mouse_pos.y()
                
                # 判断鼠标是否在图像范围内
                if (x_offset <= mouse_x <= x_offset + scaled_pixmap.width() and 
                    y_offset <= mouse_y <= y_offset + scaled_pixmap.height()):
                    
                    # 设置辅助线样式：半透明蓝色虚线
                    guide_pen = QPen(QColor(0, 120, 215, 180), 1, Qt.DashLine)
                    painter.setPen(guide_pen)
                    
                    # 绘制水平辅助线
                    painter.drawLine(x_offset, mouse_y, x_offset + scaled_pixmap.width(), mouse_y)
                    
                    # 绘制垂直辅助线
                    painter.drawLine(mouse_x, y_offset, mouse_x, y_offset + scaled_pixmap.height())
    
    def draw_box(self, painter, box, offset_x, offset_y, img_width, img_height, is_selected=False):
        # Get original image dimensions
        orig_width = self.pixmap.width()
        orig_height = self.pixmap.height()
        
        # Scale box coordinates to match displayed image
        scale_x = img_width / orig_width
        scale_y = img_height / orig_height
        
        x1 = offset_x + box.x1 * scale_x
        y1 = offset_y + box.y1 * scale_y
        x2 = offset_x + box.x2 * scale_x
        y2 = offset_y + box.y2 * scale_y
        
        # 更现代的颜色方案
        colors = [
            QColor(231, 76, 60),   # 红色
            QColor(46, 204, 113),  # 绿色
            QColor(52, 152, 219),  # 蓝色
            QColor(241, 196, 15),  # 黄色
            QColor(155, 89, 182),  # 紫色
            QColor(230, 126, 34)   # 橙色
        ]
        color = colors[box.class_id % len(colors)]
        
        # 选中的边界框使用更粗的线条
        pen_width = 3 if is_selected else 2
        pen = QPen(color, pen_width, Qt.SolidLine)
        painter.setPen(pen)
        
        # 无论是否选中，都添加半透明填充，但选中的透明度稍低（更明显）
        fill_color = QColor(color)
        fill_color.setAlpha(30 if not is_selected else 100)  # 设置透明度
        painter.fillRect(int(x1), int(y1), int(x2 - x1), int(y2 - y1), fill_color)
        
        # 绘制边框
        painter.drawRect(int(x1), int(y1), int(x2 - x1), int(y2 - y1))
        
        # 只有当is_selected为True时才绘制控制点
        if is_selected:
            painter.save()  # 保存当前状态
            handle_size = 6
            handle_color = Qt.white
            painter.setBrush(QColor(handle_color))
            
            # 绘制四个角点
            corners = [
                (x1, y1),  # 左上
                (x2, y1),  # 右上
                (x1, y2),  # 左下
                (x2, y2)   # 右下
            ]
            for cx, cy in corners:
                painter.drawRect(int(cx - handle_size/2), int(cy - handle_size/2), handle_size, handle_size)
            
            # 绘制四条边的中点
            edges = [
                (x1 + (x2-x1)/2, y1),  # 上边中点
                (x2, y1 + (y2-y1)/2),  # 右边中点
                (x1 + (x2-x1)/2, y2),  # 下边中点
                (x1, y1 + (y2-y1)/2)   # 左边中点
            ]
            for ex, ey in edges:
                painter.drawRect(int(ex - handle_size/2), int(ey - handle_size/2), handle_size, handle_size)
            
            painter.restore()  # 恢复保存的状态
        
        # Draw class label
        class_name = self.parent.get_class_name(box.class_id)
        painter.drawText(int(x1), int(y1) - 5, f"{class_name} (ID: {box.class_id})")
    
    def get_scaled_pos(self, pos):
        """将QPoint窗口坐标转换为考虑缩放因子的图像坐标"""
        if not self.pixmap:
            return QPoint(0, 0)
            
        scaled_pixmap = self.pixmap.scaled(int(self.width() * self.scale_factor), 
                                          int(self.height() * self.scale_factor),
                                          Qt.KeepAspectRatio)
        offset_x = (self.width() - scaled_pixmap.width()) // 2 + self.offset_x
        offset_y = (self.height() - scaled_pixmap.height()) // 2 + self.offset_y
        
        # 转换为原始图像坐标
        orig_width = self.pixmap.width()
        orig_height = self.pixmap.height()
        scale_x = orig_width / scaled_pixmap.width()
        scale_y = orig_height / scaled_pixmap.height()
        
        x = (pos.x() - offset_x) * scale_x
        y = (pos.y() - offset_y) * scale_y
        
        return QPoint(int(x), int(y))
    
    def get_image_coordinates(self, event_x, event_y):
        """将窗口坐标转换为原始图像坐标"""
        if not self.pixmap:
            return None, None
            
        scaled_pixmap = self.pixmap.scaled(int(self.width() * self.scale_factor), 
                                          int(self.height() * self.scale_factor),
                                          Qt.KeepAspectRatio)
        offset_x = (self.width() - scaled_pixmap.width()) // 2 + self.offset_x
        offset_y = (self.height() - scaled_pixmap.height()) // 2 + self.offset_y
        
        # 检查点是否在图像范围内
        if not (offset_x <= event_x <= offset_x + scaled_pixmap.width() and
                offset_y <= event_y <= offset_y + scaled_pixmap.height()):
            return None, None
            
        # 转换为原始图像坐标
        orig_width = self.pixmap.width()
        orig_height = self.pixmap.height()
        scale_x = orig_width / scaled_pixmap.width()
        scale_y = orig_height / scaled_pixmap.height()
        
        x = (event_x - offset_x) * scale_x
        y = (event_y - offset_y) * scale_y
        
        return x, y
    
    def get_image_position(self, pos):
        """将QPoint窗口坐标转换为原始图像坐标"""
        return self.get_image_coordinates(pos.x(), pos.y())
    
    def get_box_at_position(self, x, y):
        """获取指定位置的边界框索引和编辑模式"""
        for i, box in reversed(list(enumerate(self.boxes))):  # 从后往前检查，优先选择最上层的框
            # 检查是否在角点上
            corner = box.on_corner(x, y, margin=8)
            if corner:
                return i, 'resize-corner', corner
                
            # 检查是否在边缘上
            edge = box.on_edge(x, y, margin=8)
            if edge:
                return i, 'resize-edge', edge
                
            # 检查是否在框内
            if box.contains_point(x, y):
                return i, 'move', None
                
        return -1, None, None
    
    def update_cursor(self, edit_mode, handle):
        """根据编辑模式更新鼠标指针样式"""
        if edit_mode == 'move':
            self.setCursor(Qt.SizeAllCursor)
        elif edit_mode == 'resize-corner':
            if handle in ['top-left', 'bottom-right']:
                self.setCursor(Qt.SizeFDiagCursor)
            else:  # 'top-right', 'bottom-left'
                self.setCursor(Qt.SizeBDiagCursor)
        elif edit_mode == 'resize-edge':
            if handle in ['left', 'right']:
                self.setCursor(Qt.SizeHorCursor)
            else:  # 'top', 'bottom'
                self.setCursor(Qt.SizeVerCursor)
        else:
            self.setCursor(Qt.ArrowCursor)
    
    def mousePressEvent(self, event):
        """处理鼠标按下事件"""
        if not self.pixmap:
            return
            
        # 中键按下处理图像拖动
        if event.button() == Qt.MiddleButton:
            self.is_panning = True
            self.pan_start_pos = event.pos()
            self.setCursor(Qt.ClosedHandCursor)  # 设置为抓取光标
            return
            
        # 获取缩放后的鼠标位置
        pos = self.get_scaled_pos(event.pos())
        
        # 特征点编辑模式
        if self.keypoint_edit_mode:
            # 检查是否点击了已有特征点（用于移动或删除）
            if event.button() == Qt.LeftButton:
                # 先检查是否点击了已有特征点（用于移动）
                for box_idx, box in enumerate(self.boxes):
                    if box.has_keypoints():
                        for kp_idx, kp in enumerate(box.keypoints):
                            kp_x, kp_y = kp
                            # 检查鼠标是否在特征点上
                            if abs(kp_x - pos.x()) <= self.keypoint_radius * 2 and abs(kp_y - pos.y()) <= self.keypoint_radius * 2:
                                # 开始移动特征点
                                self.moving_keypoint = True
                                self.moving_keypoint_box_index = box_idx
                                self.moving_keypoint_index = kp_idx
                                self.setCursor(Qt.ClosedHandCursor)  # 设置为抓取光标
                                return
                
                # 如果没有点击已有特征点，且有选中的边界框，则添加新特征点
                if self.selected_box_index >= 0 and self.selected_box_index < len(self.boxes):
                    box = self.boxes[self.selected_box_index]
                    # 直接检查点击位置是否在边界框的确切边界内，不使用边缘容差
                    if box.x1 <= pos.x() <= box.x2 and box.y1 <= pos.y() <= box.y2:
                        # 添加特征点
                        if box.add_keypoint(pos.x(), pos.y()):
                            self.update()
                            self.parent.save_current()  # 保存修改
                            return
            return
        
        # 非特征点编辑模式的处理逻辑
        x, y = self.get_image_position(event.pos())
        if x is None or y is None:
            return
            
        if event.button() == Qt.LeftButton:
            # 检查是否点击了现有边界框
            box_index, edit_mode, handle = self.get_box_at_position(x, y)
            
            if box_index >= 0:
                # 选中边界框进行编辑
                self.selected_box_index = box_index
                self.edit_mode = edit_mode
                self.edit_handle = handle
                self.last_cursor_pos = (x, y)
                
                # 更新边界框列表选择
                self.parent.box_list.setCurrentRow(box_index)
            else:
                # 开始创建新边界框
                self.selected_box_index = -1
                self.start_point = QPoint(int(x), int(y))
                self.current_box = BoundingBox(x, y, x, y, self.current_class_id)
                self.edit_mode = None
                self.edit_handle = None
            
            self.update()
    
    def mouseDoubleClickEvent(self, event):
        """双击事件处理，用于删除特征点"""
        if not self.pixmap or not self.keypoint_edit_mode:
            return
            
        # 获取图像坐标系中的点击位置
        pos = self.get_image_position(event.pos())
        if pos is None:
            return
            
        x, y = pos
        
        # 检查是否双击了特征点
        for i, box in enumerate(self.boxes):
            if not box.has_keypoints():
                continue
                
            keypoints = box.get_keypoints()
            for j, kp in enumerate(keypoints):
                # 处理特征点数据可能只有2个值的情况
                if len(kp) >= 3:
                    kp_x, kp_y, _ = kp
                else:
                    kp_x, kp_y = kp
                    
                # 计算点击位置与特征点的距离
                distance = np.sqrt((x - kp_x)**2 + (y - kp_y)**2)
                
                # 如果距离小于阈值，删除该特征点
                if distance <= 5:  # 5像素的容差
                    # 删除特征点
                    new_keypoints = np.delete(keypoints, j, axis=0)
                    box.set_keypoints(new_keypoints)
                    self.update()
                    self.parent.save_current()  # 保存修改
                    return
    
    def mouseMoveEvent(self, event):
        """处理鼠标移动事件"""
        if not self.pixmap:
            return
            
        # 更新鼠标位置（用于辅助线绘制）
        self.mouse_pos = event.pos()
        self.update()  # 触发重绘
            
        # 处理图像拖动
        if self.is_panning and self.pan_start_pos:
            # 计算鼠标移动距离
            delta = event.pos() - self.pan_start_pos
            self.offset_x += delta.x()
            self.offset_y += delta.y()
            self.pan_start_pos = event.pos()
            self.update()  # 重绘画布
            return
            
        # 获取缩放后的鼠标位置
        pos = self.get_scaled_pos(event.pos())
        
        # 特征点编辑模式下的移动处理
        if self.keypoint_edit_mode:
            # 如果正在移动特征点
            if self.moving_keypoint and self.moving_keypoint_box_index >= 0 and self.moving_keypoint_index >= 0:
                # 更新特征点位置
                box = self.boxes[self.moving_keypoint_box_index]
                if box.has_keypoints():
                    # 确保特征点位置在图像范围内
                    x = max(0, min(pos.x(), self.pixmap.width()))
                    y = max(0, min(pos.y(), self.pixmap.height()))
                    
                    # 更新特征点坐标
                    box.keypoints[self.moving_keypoint_index] = [x, y]
                    self.update()  # 重绘画布
                return
            
            # 鼠标悬停在特征点上时改变光标
            cursor_changed = False
            for box in self.boxes:
                if box.has_keypoints():
                    for kp in box.keypoints:
                        kp_x, kp_y = kp
                        if abs(kp_x - pos.x()) <= self.keypoint_radius * 2 and abs(kp_y - pos.y()) <= self.keypoint_radius * 2:
                            self.setCursor(Qt.OpenHandCursor)  # 设置为手形光标
                            cursor_changed = True
                            break
                if cursor_changed:
                    break
            
            if not cursor_changed:
                self.setCursor(Qt.CrossCursor)  # 恢复十字光标
            
            return
            
        # 非特征点编辑模式的处理逻辑
        x, y = self.get_image_position(event.pos())
        if x is None or y is None:
            self.setCursor(Qt.ArrowCursor)
            return
            
        # 处理边界框编辑
        if self.edit_mode and self.selected_box_index >= 0 and self.last_cursor_pos:
            box = self.boxes[self.selected_box_index]
            dx = x - self.last_cursor_pos[0]
            dy = y - self.last_cursor_pos[1]
            
            # 获取图像边界
            img_width = self.pixmap.width()
            img_height = self.pixmap.height()
            
            if self.edit_mode == 'move':
                # 移动整个边界框
                new_x1 = max(0, min(box.x1 + dx, img_width - 5))
                new_y1 = max(0, min(box.y1 + dy, img_height - 5))
                new_x2 = max(5, min(box.x2 + dx, img_width))
                new_y2 = max(5, min(box.y2 + dy, img_height))
                
                # 确保边界框不会移出图像
                if new_x1 < new_x2 - 5 and new_y1 < new_y2 - 5:
                    box.x1 = new_x1
                    box.y1 = new_y1
                    box.x2 = new_x2
                    box.y2 = new_y2
                
            elif self.edit_mode == 'resize-corner':
                # 调整角点位置
                if self.edit_handle == 'top-left':
                    box.x1 = max(0, min(box.x1 + dx, box.x2 - 5))
                    box.y1 = max(0, min(box.y1 + dy, box.y2 - 5))
                elif self.edit_handle == 'top-right':
                    box.x2 = max(box.x1 + 5, min(box.x2 + dx, img_width))
                    box.y1 = max(0, min(box.y1 + dy, box.y2 - 5))
                elif self.edit_handle == 'bottom-left':
                    box.x1 = max(0, min(box.x1 + dx, box.x2 - 5))
                    box.y2 = max(box.y1 + 5, min(box.y2 + dy, img_height))
                elif self.edit_handle == 'bottom-right':
                    box.x2 = max(box.x1 + 5, min(box.x2 + dx, img_width))
                    box.y2 = max(box.y1 + 5, min(box.y2 + dy, img_height))
                
            elif self.edit_mode == 'resize-edge':
                # 调整边缘位置
                if self.edit_handle == 'left':
                    box.x1 = max(0, min(box.x1 + dx, box.x2 - 5))
                elif self.edit_handle == 'right':
                    box.x2 = max(box.x1 + 5, min(box.x2 + dx, img_width))
                elif self.edit_handle == 'top':
                    box.y1 = max(0, min(box.y1 + dy, box.y2 - 5))
                elif self.edit_handle == 'bottom':
                    box.y2 = max(box.y1 + 5, min(box.y2 + dy, img_height))
            
            self.last_cursor_pos = (x, y)
            self.update()
            
        # 处理新边界框创建
        elif self.start_point and self.current_box:
            # 限制在图像边界内
            x = max(0, min(x, self.pixmap.width()))
            y = max(0, min(y, self.pixmap.height()))
            
            self.current_box.x2 = x
            self.current_box.y2 = y
            self.update()
    
    def mouseReleaseEvent(self, event):
        """处理鼠标释放事件"""
        if not self.pixmap:
            return
            
        # 处理中键释放，结束图像拖动
        if event.button() == Qt.MiddleButton:
            self.is_panning = False
            self.pan_start_pos = None
            self.setCursor(Qt.ArrowCursor)
            return
            
        # 特征点编辑模式下的释放处理
        if self.keypoint_edit_mode:
            if event.button() == Qt.LeftButton and self.moving_keypoint:
                # 停止移动特征点
                self.moving_keypoint = False
                self.setCursor(Qt.CrossCursor)  # 恢复十字光标
                
                # 保存更新的特征点位置
                self.parent.save_current()  # 保存修改
                
                # 更新界面
                self.update()
            return
            
        # 非特征点编辑模式的处理逻辑
        x, y = self.get_image_position(event.pos())
        if x is None or y is None:
            self.setCursor(Qt.ArrowCursor)
            return
        
        if event.button() == Qt.LeftButton:
            # 处理边界框编辑
            if self.edit_mode and self.selected_box_index >= 0 and self.last_cursor_pos:
                box = self.boxes[self.selected_box_index]
                dx = x - self.last_cursor_pos[0]
                dy = y - self.last_cursor_pos[1]
                
                # 获取图像边界
                img_width = self.pixmap.width()
                img_height = self.pixmap.height()
                
                if self.edit_mode == 'move':
                    # 移动整个边界框
                    new_x1 = max(0, min(box.x1 + dx, img_width - 5))
                    new_y1 = max(0, min(box.y1 + dy, img_height - 5))
                    new_x2 = max(5, min(box.x2 + dx, img_width))
                    new_y2 = max(5, min(box.y2 + dy, img_height))
                    
                    # 确保边界框不会移出图像
                    if new_x1 < new_x2 - 5 and new_y1 < new_y2 - 5:
                        box.x1 = new_x1
                        box.y1 = new_y1
                        box.x2 = new_x2
                        box.y2 = new_y2
                    
                elif self.edit_mode == 'resize-corner':
                    # 调整角点位置
                    if self.edit_handle == 'top-left':
                        box.x1 = max(0, min(box.x1 + dx, box.x2 - 5))
                        box.y1 = max(0, min(box.y1 + dy, box.y2 - 5))
                    elif self.edit_handle == 'top-right':
                        box.x2 = max(box.x1 + 5, min(box.x2 + dx, img_width))
                        box.y1 = max(0, min(box.y1 + dy, box.y2 - 5))
                    elif self.edit_handle == 'bottom-left':
                        box.x1 = max(0, min(box.x1 + dx, box.x2 - 5))
                        box.y2 = max(box.y1 + 5, min(box.y2 + dy, img_height))
                    elif self.edit_handle == 'bottom-right':
                        box.x2 = max(box.x1 + 5, min(box.x2 + dx, img_width))
                        box.y2 = max(box.y1 + 5, min(box.y2 + dy, img_height))
                    
                elif self.edit_mode == 'resize-edge':
                    # 调整边缘位置
                    if self.edit_handle == 'left':
                        box.x1 = max(0, min(box.x1 + dx, box.x2 - 5))
                    elif self.edit_handle == 'right':
                        box.x2 = max(box.x1 + 5, min(box.x2 + dx, img_width))
                    elif self.edit_handle == 'top':
                        box.y1 = max(0, min(box.y1 + dy, box.y2 - 5))
                    elif self.edit_handle == 'bottom':
                        box.y2 = max(box.y1 + 5, min(box.y2 + dy, img_height))
                
                # 完成编辑后清除编辑状态
                self.edit_mode = None
                self.edit_handle = None
                self.last_cursor_pos = None
                self.parent.save_current()  # 保存修改
                
            # 处理新边界框创建
            elif self.start_point and self.current_box:
                # 限制在图像边界内
                x = max(0, min(x, self.pixmap.width()))
                y = max(0, min(y, self.pixmap.height()))
                
                self.current_box.x2 = x
                self.current_box.y2 = y
                
                # 只有当边界框达到最小尺寸时才添加
                width = abs(self.current_box.x2 - self.current_box.x1)
                height = abs(self.current_box.y2 - self.current_box.y1)
                
                box_added = False
                if width > 5 and height > 5:  # 最小尺寸阈值
                    self.boxes.append(self.current_box)
                    self.parent.update_box_list()
                    box_added = True
                
                self.current_box = None
                self.start_point = None
                
                # 只有在添加了新边界框时才自动保存
                if box_added:
                    self.parent.save_current()
            
            self.update()
    
    def set_current_class(self, class_id):
        self.current_class_id = class_id
        
        # 如果有选中的边界框，更新其类别
        if self.selected_box_index >= 0 and self.selected_box_index < len(self.boxes):
            self.boxes[self.selected_box_index].class_id = class_id
            self.parent.update_box_list()
            self.update()
            self.parent.save_current()  # 保存修改
    
    def zoom_in(self):
        self.scale_factor *= 1.2
        self.update()
    
    def zoom_out(self):
        self.scale_factor /= 1.2
        self.update()
    
    def reset_zoom(self):
        self.scale_factor = 1.0
        self.offset_x = 0  # 重置X轴偏移
        self.offset_y = 0  # 重置Y轴偏移
        self.update()

    def toggle_keypoint_mode(self, enabled=None):
        """切换特征点编辑模式"""
        if enabled is not None:
            self.keypoint_edit_mode = enabled
        else:
            self.keypoint_edit_mode = not self.keypoint_edit_mode
        
        # 更新鼠标指针
        if self.keypoint_edit_mode:
            self.setCursor(Qt.CrossCursor)
        else:
            self.setCursor(Qt.ArrowCursor)
        
        # 取消当前选择
        self.selected_box_index = -1
        
        # 仅更新画布，不触发布局变化
        self.update()
        
    def toggle_guide_lines(self, enabled=None):
        """切换辅助线显示"""
        if enabled is not None:
            self.guide_lines_enabled = enabled
        else:
            self.guide_lines_enabled = not self.guide_lines_enabled
            
        # 更新画布
        self.update()