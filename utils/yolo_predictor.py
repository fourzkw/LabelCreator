import os
import torch
import logging
import traceback
from i18n import tr
import numpy as np
from PIL import Image
from models.bounding_box import BoundingBox

# 尝试导入 ultralytics 包
try:
    from ultralytics import YOLO
    ULTRALYTICS_AVAILABLE = True
except ImportError:
    ULTRALYTICS_AVAILABLE = False

logger = logging.getLogger('YOLOLabelCreator.YOLOPredictor')

class YOLOPredictor:
    def __init__(self):
        self.model = None
        self.conf_threshold = 0.5
        self.iou_threshold = 0.45
        self.max_detections = 100
        self.device = "cpu"  # 默认使用CPU
        self.model_type = None  # 'yolov5', 'yolov8', 'onnx'
        
        # 检测可用设备
        self.available_devices = ["cpu"]
        if torch.cuda.is_available():
            self.available_devices.append("cuda")
            logger.info(f"使用设备: cuda")
        else:
            logger.info(f"CUDA不可用，使用设备: cpu")
    
    def set_params(self, conf_threshold=None, iou_threshold=None, max_detections=None, device=None):
        """设置预测参数"""
        if conf_threshold is not None:
            self.conf_threshold = conf_threshold
        if iou_threshold is not None:
            self.iou_threshold = iou_threshold
        if max_detections is not None:
            self.max_detections = max_detections
        if device is not None and device in self.available_devices:
            self.device = device
            # 如果模型已加载，则将其移动到新设备
            if self.model is not None and self.model_type != 'onnx':
                try:
                    self.model.to(self.device)
                    logger.info(f"模型已移动到设备: {self.device}")
                except Exception as e:
                    logger.error(f"移动模型到设备 {self.device} 失败: {str(e)}")
    
    def load_model(self, model_path):
        """加载YOLO模型"""
        if not os.path.exists(model_path):
            logger.error(f"模型文件不存在: {model_path}")
            return False
        
        try:
            logger.info(f"正在加载YOLO模型: {model_path}")
            
            # 根据文件扩展名确定模型类型
            file_ext = os.path.splitext(model_path)[1].lower()
            
            # ONNX 模型
            if file_ext == '.onnx':
                import onnxruntime as ort
                
                # 创建 ONNX 运行时会话
                providers = ['CPUExecutionProvider']
                if self.device == 'cuda' and 'CUDAExecutionProvider' in ort.get_available_providers():
                    providers.insert(0, 'CUDAExecutionProvider')
                
                self.model = ort.InferenceSession(model_path, providers=providers)
                self.model_type = 'onnx'
                logger.info(f"ONNX模型加载成功，使用提供程序: {providers}")
                return True
                
            # YOLOv8 模型 (使用 ultralytics 包)
            elif ULTRALYTICS_AVAILABLE:
                self.model = YOLO(model_path)
                self.model_type = 'yolov8'
                logger.info("YOLOv8模型加载成功")
                return True
                
            # 尝试使用 torch.hub 加载 YOLOv5 模型
            else:
                try:
                    self.model = torch.hub.load('ultralytics/yolov5', 'custom', 
                                               path=model_path, device=self.device)
                    
                    # 设置NMS参数
                    self.model.conf = self.conf_threshold
                    self.model.iou = self.iou_threshold
                    self.model.max_det = self.max_detections
                    
                    self.model_type = 'yolov5'
                    logger.info("YOLOv5模型加载成功")
                    return True
                except Exception as e:
                    logger.error(f"使用torch.hub加载YOLOv5模型失败: {str(e)}")
                    logger.error(f"请尝试将模型转换为ONNX格式或安装ultralytics包")
                    return False
                
        except Exception as e:
            logger.error(f"加载模型失败: {str(e)}")
            logger.error(f"异常详情: {traceback.format_exc()}")
            return False
    
    def predict(self, image_path):
        """对图像进行预测"""
        if self.model is None:
            logger.error("模型未加载")
            return []
        
        if not os.path.exists(image_path):
            logger.error(f"图像文件不存在: {image_path}")
            return []
        
        try:
            logger.info(f"对图像进行预测: {image_path}")
            
            # 根据模型类型选择不同的预测方法
            if self.model_type == 'onnx':
                return self._predict_onnx(image_path)
            elif self.model_type == 'yolov8':
                return self._predict_yolov8(image_path)
            else:  # yolov5
                return self._predict_yolov5(image_path)
                
        except Exception as e:
            logger.error(f"预测失败: {str(e)}")
            logger.error(f"异常详情: {traceback.format_exc()}")
            return []
    
    def _predict_yolov5(self, image_path):
        """使用YOLOv5模型预测"""
        # 确保模型在正确的设备上
        self.model.to(self.device)
        
        # 进行预测
        results = self.model(image_path)
        
        # 提取预测结果
        predictions = []
        for pred in results.xyxy[0]:
            x1, y1, x2, y2, conf, cls = pred.cpu().numpy()
            
            # 将字典改为BoundingBox对象
            predictions.append(BoundingBox(
                x1=float(x1),
                y1=float(y1),
                x2=float(x2),
                y2=float(y2),
                class_id=int(cls),
                confidence=float(conf)
            ))
        
        return predictions
    
    def _predict_yolov8(self, image_path):
        """使用YOLOv8模型预测"""
        # 设置参数
        results = self.model.predict(
            source=image_path,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            max_det=self.max_detections,
            device=self.device
        )
        
        # 提取预测结果
        predictions = []
        if len(results) > 0:
            result = results[0]
            boxes = result.boxes
            
            for i in range(len(boxes)):
                box = boxes[i]
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = box.conf[0].cpu().numpy()
                cls = box.cls[0].cpu().numpy()
                
                # 将字典改为BoundingBox对象
                predictions.append(BoundingBox(
                    x1=float(x1),
                    y1=float(y1),
                    x2=float(x2),
                    y2=float(y2),
                    class_id=int(cls),
                    confidence=float(conf)
                ))
        
        return predictions
    
    def _predict_onnx(self, image_path):
        """使用ONNX模型预测"""
        # 加载并预处理图像
        image = Image.open(image_path).convert('RGB')
        img = np.array(image)
        
        # 获取输入名称
        input_name = self.model.get_inputs()[0].name
        
        # 预处理图像 (调整大小、归一化等)
        # 注意：这里的预处理步骤可能需要根据模型的具体要求进行调整
        img = img.transpose(2, 0, 1)  # HWC -> CHW
        img = img.astype('float32') / 255.0
        img = np.expand_dims(img, axis=0)  # 添加批次维度
        
        # 进行推理
        outputs = self.model.run(None, {input_name: img})
        
        # 解析输出 (具体解析方式取决于模型输出格式)
        # 这里假设输出格式为 [batch_id, x1, y1, x2, y2, confidence, class_id]
        predictions = []
        
        # 获取原始图像尺寸
        img_height, img_width = image.height, image.width
        
        if len(outputs) > 0 and len(outputs[0]) > 0:
            detections = outputs[0]
            
            # 应用置信度阈值
            valid_detections = detections[detections[:, 4] > self.conf_threshold]
            
            for detection in valid_detections:
                x1, y1, x2, y2, conf = detection[:5]
                cls_id = int(detection[5])
                
                # 将坐标转换为原始图像尺寸
                x1 = float(x1 * img_width)
                y1 = float(y1 * img_height)
                x2 = float(x2 * img_width)
                y2 = float(y2 * img_height)
                
                predictions.append({
                    'x1': x1,
                    'y1': y1,
                    'x2': x2,
                    'y2': y2,
                    'confidence': float(conf),
                    'class_id': cls_id
                })
        
        return predictions