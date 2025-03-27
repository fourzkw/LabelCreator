import os
import logging
import torch
import numpy as np
from PIL import Image
from models.bounding_box import BoundingBox

logger = logging.getLogger('YOLOLabelCreator.YOLOPredictor')

class YOLOPredictor:
    """
    使用YOLO模型进行目标检测的预测器类
    """
    def __init__(self):
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"使用设备: {self.device}")
        
    def load_model(self, model_path):
        """
        加载YOLO模型
        
        Args:
            model_path (str): 模型文件路径
            
        Returns:
            bool: 加载成功返回True，否则返回False
        """
        try:
            if not os.path.exists(model_path):
                logger.error(f"模型文件不存在: {model_path}")
                return False
                
            logger.info(f"正在加载YOLO模型: {model_path}")
            # 修改为正确的YOLOv8加载方式
            from ultralytics import YOLO
            self.model = YOLO(model_path)
            logger.info("YOLO模型加载成功")
            return True
            
        except Exception as e:
            logger.error(f"加载YOLO模型失败: {str(e)}")
            return False
            
    def predict(self, image_path, conf_threshold=0.25):
        """
        对图像进行目标检测
        
        Args:
            image_path (str): 图像文件路径
            conf_threshold (float): 置信度阈值
            
        Returns:
            list: BoundingBox对象列表，如果失败则返回空列表
        """
        if self.model is None:
            logger.error("模型未加载，无法进行预测")
            return []
            
        try:
            logger.info(f"对图像进行预测: {image_path}")
            
            # 加载图像
            img = Image.open(image_path)
            img_width, img_height = img.size
            
            # 进行预测 - 修改为YOLOv8的预测方式
            results = self.model(image_path, conf=conf_threshold)
            
            # 转换为BoundingBox对象
            boxes = []
            for result in results:
                # YOLOv8返回的是一个Results对象，需要从中提取边界框
                for box in result.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    class_id = int(box.cls[0].item())
                    boxes.append(BoundingBox(x1, y1, x2, y2, class_id))
                    
            logger.info(f"检测到{len(boxes)}个目标")
            return boxes
            
        except Exception as e:
            logger.error(f"预测失败: {str(e)}")
            return []