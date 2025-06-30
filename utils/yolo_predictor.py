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
    """
    YOLO模型预测器类
    
    用于加载YOLO模型并对图像进行目标检测预测。
    支持YOLOv8和ONNX格式的模型。
    """
    
    def __init__(self):
        self.model = None
        self.conf_threshold = 0.5
        self.iou_threshold = 0.45
        self.max_detections = 100
        self.device = "cpu"  # 默认使用CPU
        self.model_type = None  # 'yolov8', 'onnx'
        self.keypoints_number = 0  # 特征点数量，0表示使用模型默认值
        
        # 检测可用设备
        self.available_devices = ["cpu"]
        if torch.cuda.is_available():
            self.available_devices.append("cuda")
            logger.info(f"使用设备: cuda")
        else:
            logger.info(f"CUDA不可用，使用设备: cpu")
    
    def set_params(self, conf_threshold=None, iou_threshold=None, max_detections=None, device=None, keypoints_number=None):
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
        if keypoints_number is not None:
            self.keypoints_number = keypoints_number
            logger.info(f"设置特征点数量: {self.keypoints_number}")
    
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
                
            # 不支持的模型类型
            else:
                logger.error("不支持的模型类型或缺少必要依赖")
                logger.error("请使用 ONNX 格式或安装 ultralytics 包使用 YOLOv8")
                return False
                
        except Exception as e:
            logger.error(f"加载模型失败: {str(e)}")
            logger.error(f"异常详情: {traceback.format_exc()}")
            return False
    
    def predict(self, image_path):
        """
        对图像进行目标检测预测
        
        Args:
            image_path (str): 图像文件路径
            
        Returns:
            list: 检测到的边界框列表，每个边界框为BoundingBox对象
        """
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
            else:
                logger.error(f"不支持的模型类型: {self.model_type}")
                return []
                
        except Exception as e:
            logger.error(f"预测失败: {str(e)}")
            logger.error(f"异常详情: {traceback.format_exc()}")
            return []
    
    def _predict_yolov8(self, image_path):
        """使用YOLOv8模型预测"""
        # 设置参数
        predict_args = {
            "source": image_path,
            "conf": self.conf_threshold,
            "iou": self.iou_threshold,
            "max_det": self.max_detections,
            "device": self.device
        }
        
        # 如果设置了特征点数量且大于0，则添加到预测参数中
        if self.keypoints_number > 0:
            predict_args["kpt_num"] = self.keypoints_number
            
        results = self.model.predict(**predict_args)
        
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
                
                # 创建边界框对象
                bbox = BoundingBox(
                    x1=float(x1),
                    y1=float(y1),
                    x2=float(x2),
                    y2=float(y2),
                    class_id=int(cls),
                    confidence=float(conf)
                )
                
                # 检查是否有关键点数据
                if hasattr(result, 'keypoints') and result.keypoints is not None:
                    try:
                        # 提取关键点数据
                        keypoints = result.keypoints[i].data[0].cpu().numpy()
                        # 只保留 x, y 坐标，去掉置信度
                        if len(keypoints) > 0:
                            # 转换为只包含 x, y 的数组
                            keypoints_xy = keypoints[:, :2]
                            # 设置边界框的关键点
                            bbox.set_keypoints(keypoints_xy)
                            logger.info(f"检测到 {len(keypoints_xy)} 个特征点")
                    except Exception as e:
                        logger.error(f"提取特征点时出错: {str(e)}")
                
                predictions.append(bbox)
        
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