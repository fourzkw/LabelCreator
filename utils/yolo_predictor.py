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

# 导入TensorRT路径设置函数
try:
    from utils.model_converter.tensorrt_converter import TensorRTConverter
    TENSORRT_CONVERTER_AVAILABLE = True
except ImportError:
    TENSORRT_CONVERTER_AVAILABLE = False

logger = logging.getLogger('YOLOLabelCreator.YOLOPredictor')

class YOLOPredictor:
    """
    YOLO模型预测器类
    
    用于加载YOLO模型并对图像进行目标检测预测。
    支持YOLOv5、YOLOv7、YOLOv8、YOLO11、YOLO26、ONNX和TensorRT格式的模型。
    """
    
    def __init__(self):
        self.model = None
        self.conf_threshold = 0.5
        self.iou_threshold = 0.45
        self.max_detections = 100
        self.device = "cpu"  # 默认使用CPU
        self.model_type = None  # 'ultralytics', 'onnx', 'tensorrt'
        self.model_version = "yolov8"  # 模型版本: 'yolov5', 'yolov7', 'yolov8', 'yolov11', 'yolo26'
        self.keypoints_number = 0  # 特征点数量，0表示使用模型默认值
        self.class_mapping = {}  # 标注映射：{识别类别ID: 标注类别ID}
        
        # 检测可用设备
        self.available_devices = ["cpu"]
        if torch.cuda.is_available():
            self.available_devices.append("cuda")
            logger.info(f"使用设备: cuda")
        else:
            logger.info(f"CUDA不可用，使用设备: cpu")
    
    def set_params(self, conf_threshold=None, iou_threshold=None, max_detections=None, device=None, keypoints_number=None, model_version=None, class_mapping=None):
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
        if model_version is not None:
            self.model_version = model_version
            logger.info(f"设置模型版本: {self.model_version}")
        if class_mapping is not None:
            self.class_mapping = class_mapping
            logger.info(f"设置标注映射: {class_mapping}")
    
    def load_model(self, model_path, model_version=None):
        """
        加载YOLO模型
        
        Args:
            model_path (str): 模型文件路径
            model_version (str): 模型版本 ('yolov5', 'yolov7', 'yolov8', 'yolov11', 'yolo26')
            
        Returns:
            bool: 加载是否成功
        """
        if not os.path.exists(model_path):
            logger.error(f"模型文件不存在: {model_path}")
            return False
        
        # 更新模型版本
        if model_version is not None:
            self.model_version = model_version
        
        try:
            logger.info(f"正在加载YOLO模型: {model_path} (版本: {self.model_version})")
            
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
            
            # TensorRT 模型 (.engine)
            elif file_ext == '.engine':
                if not ULTRALYTICS_AVAILABLE:
                    logger.error("加载TensorRT模型需要ultralytics包")
                    return False
                
                # 检查CUDA是否可用（TensorRT需要CUDA）
                if not torch.cuda.is_available():
                    logger.error("TensorRT模型需要CUDA支持，但CUDA不可用")
                    return False
                
                # 在加载TensorRT模型之前，先设置TensorRT的PATH
                # 这很重要，因为GUI应用可能没有TensorRT在PATH中
                if TENSORRT_CONVERTER_AVAILABLE:
                    TensorRTConverter._add_tensorrt_to_path()
                
                # ultralytics YOLO可以直接加载TensorRT引擎文件
                try:
                    self.model = YOLO(model_path)
                    self.model_type = 'tensorrt'
                    logger.info(f"TensorRT模型加载成功: {model_path}")
                    
                    # TensorRT模型必须在CUDA设备上运行
                    if self.device != 'cuda':
                        logger.warning("TensorRT模型需要CUDA设备，自动切换到cuda")
                        self.device = 'cuda'
                    
                    # 将模型移动到CUDA设备
                    try:
                        self.model.to(self.device)
                        logger.info(f"TensorRT模型已移动到设备: {self.device}")
                    except Exception as e:
                        logger.warning(f"无法将TensorRT模型移动到 {self.device}: {str(e)}")
                    
                    return True
                except Exception as e:
                    logger.error(f"加载TensorRT模型失败: {str(e)}")
                    logger.error(f"异常详情: {traceback.format_exc()}")
                    return False
                
            # Ultralytics YOLO 模型 (支持 YOLOv5、YOLOv7、YOLOv8、YOLO11、YOLO26)
            elif ULTRALYTICS_AVAILABLE:
                self.model = YOLO(model_path)
                self.model_type = 'ultralytics'
                logger.info(f"{self.model_version.upper()} 模型加载成功")
                
                # 将模型移动到指定设备
                if self.device != 'cpu':
                    try:
                        self.model.to(self.device)
                        logger.info(f"模型已移动到设备: {self.device}")
                    except Exception as e:
                        logger.warning(f"无法将模型移动到 {self.device}，使用CPU: {str(e)}")
                        self.device = 'cpu'
                
                return True
                
            # 不支持的模型类型
            else:
                logger.error("不支持的模型类型或缺少必要依赖")
                logger.error("支持的格式: .pt, .pth (PyTorch), .onnx (ONNX), .engine (TensorRT)")
                logger.error("请安装 ultralytics 包以支持PyTorch和TensorRT格式")
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
            logger.info(f"使用 {self.model_version.upper()} 模型对图像进行预测: {image_path}")
            
            # 根据模型类型选择不同的预测方法
            if self.model_type == 'onnx':
                return self._predict_onnx(image_path)
            elif self.model_type == 'ultralytics' or self.model_type == 'tensorrt':
                # TensorRT模型使用与ultralytics相同的预测接口
                return self._predict_ultralytics(image_path)
            else:
                logger.error(f"不支持的模型类型: {self.model_type}")
                return []
                
        except Exception as e:
            logger.error(f"预测失败: {str(e)}")
            logger.error(f"异常详情: {traceback.format_exc()}")
            return []
    
    def _predict_ultralytics(self, image_path):
        """
        使用 Ultralytics YOLO 模型预测
        支持 YOLOv5、YOLOv7、YOLOv8、YOLO11、YOLO26
        """
        # 如果是TensorRT模型，确保PATH已设置（以防万一）
        if self.model_type == 'tensorrt' and TENSORRT_CONVERTER_AVAILABLE:
            TensorRTConverter._add_tensorrt_to_path()
        
        # 设置参数
        predict_args = {
            "source": image_path,
            "conf": self.conf_threshold,
            "iou": self.iou_threshold,
            "max_det": self.max_detections,
            "device": self.device,
            "verbose": False  # 减少输出日志
        }
        
        # 如果设置了特征点数量且大于0，则添加到预测参数中
        if self.keypoints_number > 0:
            predict_args["kpt_num"] = self.keypoints_number
            
        logger.info(f"预测参数: conf={self.conf_threshold}, iou={self.iou_threshold}, max_det={self.max_detections}, device={self.device}")
        
        results = self.model.predict(**predict_args)
        
        # 提取预测结果
        predictions = []
        if len(results) > 0:
            result = results[0]
            boxes = result.boxes
            
            logger.info(f"检测到 {len(boxes)} 个目标")
            
            for i in range(len(boxes)):
                box = boxes[i]
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = box.conf[0].cpu().numpy()
                cls = box.cls[0].cpu().numpy()
                
                # 应用标注映射
                original_class_id = int(cls)
                mapped_class_id = self.class_mapping.get(original_class_id, original_class_id)
                
                # 创建边界框对象
                bbox = BoundingBox(
                    x1=float(x1),
                    y1=float(y1),
                    x2=float(x2),
                    y2=float(y2),
                    class_id=mapped_class_id,
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
                            logger.debug(f"边界框 {i} 包含 {len(keypoints_xy)} 个特征点")
                    except Exception as e:
                        logger.error(f"提取特征点时出错: {str(e)}")
                
                predictions.append(bbox)
        else:
            logger.info("未检测到任何目标")
        
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
                original_cls_id = int(detection[5])
                
                # 应用标注映射
                mapped_cls_id = self.class_mapping.get(original_cls_id, original_cls_id)
                
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
                    'class_id': mapped_cls_id
                })
        
        return predictions