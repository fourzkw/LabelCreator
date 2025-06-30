import numpy as np

class BoundingBox:
    """边界框类，用于存储和操作边界框数据"""
    
    def __init__(self, x1, y1, x2, y2, class_id, confidence=1.0):
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2
        self.class_id = class_id
        self.confidence = confidence
        self.keypoints = None  # 存储特征点数据
        
    def to_yolo_format(self, img_width, img_height):
        """将坐标转换为YOLO格式（归一化坐标）"""
        # 计算中心点坐标（归一化）
        x_center = (self.x1 + self.x2) / (2 * img_width)
        y_center = (self.y1 + self.y2) / (2 * img_height)
        
        # 计算宽度和高度（归一化）
        width = (self.x2 - self.x1) / img_width
        height = (self.y2 - self.y1) / img_height
        
        return [self.class_id, x_center, y_center, width, height]
        
    def contains_point(self, x, y, margin=5):
        """检查点是否在边界框内部"""
        return (self.x1 + margin <= x <= self.x2 - margin and 
                self.y1 + margin <= y <= self.y2 - margin)
    
    def on_edge(self, x, y, margin=5):
        """
        检查点是否在边界框的边缘上
        
        Returns:
            str: 返回边缘位置 ('left', 'right', 'top', 'bottom') 或 None
        """
        if abs(x - self.x1) <= margin and self.y1 <= y <= self.y2:
            return 'left'
        if abs(x - self.x2) <= margin and self.y1 <= y <= self.y2:
            return 'right'
        if abs(y - self.y1) <= margin and self.x1 <= x <= self.x2:
            return 'top'
        if abs(y - self.y2) <= margin and self.x1 <= x <= self.x2:
            return 'bottom'
        return None
    
    def on_corner(self, x, y, margin=5):
        """
        检查点是否在边界框的角点上
        
        Returns:
            str: 返回角点位置 ('top-left', 'top-right', 'bottom-left', 'bottom-right') 或 None
        """
        if abs(x - self.x1) <= margin and abs(y - self.y1) <= margin:
            return 'top-left'
        if abs(x - self.x2) <= margin and abs(y - self.y1) <= margin:
            return 'top-right'
        if abs(x - self.x1) <= margin and abs(y - self.y2) <= margin:
            return 'bottom-left'
        if abs(x - self.x2) <= margin and abs(y - self.y2) <= margin:
            return 'bottom-right'
        return None
    
    def set_keypoints(self, keypoints):
        """设置特征点数据"""
        self.keypoints = keypoints

    def add_keypoint(self, x, y):
        """添加一个新的特征点到边界框"""
        # 确保坐标在边界框内，允许点位于边缘上
        if not (self.x1 <= x <= self.x2 and self.y1 <= y <= self.y2):
            return False
            
        # 如果特征点列表不存在，创建一个新的
        if not hasattr(self, 'keypoints') or self.keypoints is None:
            self.keypoints = np.array([[x, y]])
        else:
            # 添加新的特征点
            self.keypoints = np.append(self.keypoints, [[x, y]], axis=0)
            
        return True
        
    def has_keypoints(self):
        """检查是否有特征点数据"""
        return self.keypoints is not None and len(self.keypoints) > 0
        
    def get_keypoints(self):
        """获取特征点数据"""
        return self.keypoints