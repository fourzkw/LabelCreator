import os
import logging
import traceback
import torch
import numpy as np
from ultralytics import YOLO

# 尝试导入onnx库
try:
    import onnx
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False

from i18n import tr

logger = logging.getLogger('YOLOLabelCreator.ModelAnalyzer')

class ModelAnalyzer:
    """
    模型分析工具类，用于分析ONNX和PyTorch模型的输入输出格式
    """
    
    @staticmethod
    def analyze_model(model_path):
        """
        分析模型结构，获取输入输出格式信息
        
        Args:
            model_path: 模型文件路径
        
        Returns:
            dict: 包含模型信息的字典，包括输入、输出格式和其他元数据
        """
        if not os.path.exists(model_path):
            logger.error(f"模型文件不存在: {model_path}")
            return {"error": tr("模型文件不存在")}
        
        try:
            # 根据扩展名确定模型类型
            ext = os.path.splitext(model_path)[1].lower()
            
            if ext == '.onnx':
                return ModelAnalyzer.analyze_onnx_model(model_path)
            elif ext in ['.pt', '.pth']:
                return ModelAnalyzer.analyze_pytorch_model(model_path)
            else:
                logger.error(f"不支持的模型格式: {ext}")
                return {"error": tr(f"不支持的模型格式: {ext}")}
        
        except Exception as e:
            logger.error(f"分析模型时出错: {str(e)}\n{traceback.format_exc()}")
            return {"error": tr(f"分析模型失败: {str(e)}")}
    
    @staticmethod
    def analyze_onnx_model(model_path):
        """
        分析ONNX模型结构
        """
        if not ONNX_AVAILABLE:
            return {"error": tr("ONNX库未安装，请安装onnx和onnxruntime")}
        
        try:
            # 加载ONNX模型
            model = onnx.load(model_path)
            
            # 获取模型元数据
            metadata = {}
            if model.metadata_props:
                for prop in model.metadata_props:
                    metadata[prop.key] = prop.value
            
            # 获取模型版本信息
            model_info = {
                "ir_version": f"ONNX IR version: {model.ir_version}",
                "producer": f"Producer: {model.producer_name} {model.producer_version}",
                "model_version": f"Model version: {model.model_version}",
                "domain": f"Domain: {model.domain}",
                "metadata": metadata,
            }
            
            # 分析输入
            inputs = []
            for input_info in model.graph.input:
                input_dict = {"name": input_info.name}
                
                # 获取形状
                shape = []
                if input_info.type.tensor_type.shape:
                    for dim in input_info.type.tensor_type.shape.dim:
                        if dim.dim_param:
                            shape.append(dim.dim_param)
                        else:
                            shape.append(dim.dim_value)
                input_dict["shape"] = shape
                
                # 获取数据类型
                data_type = input_info.type.tensor_type.elem_type
                input_dict["data_type"] = onnx.TensorProto.DataType.Name(data_type)
                
                inputs.append(input_dict)
            
            # 分析输出
            outputs = []
            for output_info in model.graph.output:
                output_dict = {"name": output_info.name}
                
                # 获取形状
                shape = []
                if output_info.type.tensor_type.shape:
                    for dim in output_info.type.tensor_type.shape.dim:
                        if dim.dim_param:
                            shape.append(dim.dim_param)
                        else:
                            shape.append(dim.dim_value)
                output_dict["shape"] = shape
                
                # 获取数据类型
                data_type = output_info.type.tensor_type.elem_type
                output_dict["data_type"] = onnx.TensorProto.DataType.Name(data_type)
                
                outputs.append(output_dict)
            
            # 获取节点数量和操作类型统计
            ops_count = {}
            for node in model.graph.node:
                op_type = node.op_type
                if op_type not in ops_count:
                    ops_count[op_type] = 0
                ops_count[op_type] += 1
            
            # 统计节点总数
            total_nodes = len(model.graph.node)
            
            result = {
                "model_type": "ONNX",
                "model_info": model_info,
                "inputs": inputs,
                "outputs": outputs,
                "ops_count": ops_count,
                "total_nodes": total_nodes,
            }
            
            return result
            
        except Exception as e:
            logger.error(f"分析ONNX模型失败: {str(e)}\n{traceback.format_exc()}")
            return {"error": tr(f"分析ONNX模型失败: {str(e)}")}
    
    @staticmethod
    def analyze_pytorch_model(model_path):
        """
        分析PyTorch模型结构
        """
        try:
            # 尝试使用ultralytics加载模型
            try:
                # 对于YOLOv8模型，使用YOLO类加载
                yolo_model = YOLO(model_path)
                model_type = "YOLOv8"
                
                # 获取YOLOv8模型信息
                task = yolo_model.task
                stride = yolo_model.model.stride
                pt_path = yolo_model.ckpt_path
                
                # 获取模型输入信息
                input_names = ["images"]
                input_shape = list(yolo_model.model.args['imgsz']) if hasattr(yolo_model.model, 'args') else [640, 640]
                if len(input_shape) == 1:
                    input_shape = [input_shape[0], input_shape[0]]
                
                inputs = [{"name": input_names[0], "shape": [1, 3, input_shape[0], input_shape[1]], "data_type": "float32"}]
                
                # 获取输出信息（基于任务类型推断）
                if task == 'detect':
                    outputs = [{"name": "output", "shape": ["batch_size", "num_boxes", 5 + yolo_model.model.nc], "data_type": "float32"}]
                elif task == 'segment':
                    outputs = [
                        {"name": "boxes", "shape": ["batch_size", "num_boxes", 5 + yolo_model.model.nc], "data_type": "float32"},
                        {"name": "masks", "shape": ["batch_size", "num_masks", "height", "width"], "data_type": "float32"}
                    ]
                elif task == 'pose':
                    num_keypoints = yolo_model.model.nm if hasattr(yolo_model.model, 'nm') else "未知"
                    outputs = [{"name": "output", "shape": ["batch_size", "num_boxes", 5 + yolo_model.model.nc + 2 * num_keypoints], "data_type": "float32"}]
                else:
                    outputs = [{"name": "output", "shape": ["未知"], "data_type": "float32"}]
                
                # 尝试获取节点数量
                try:
                    total_params = sum(p.numel() for p in yolo_model.model.parameters())
                except:
                    total_params = "未知"
                
                model_info = {
                    "task": f"Task: {task}",
                    "stride": f"Stride: {stride}",
                    "path": f"Path: {pt_path}",
                    "parameters": f"Parameters: {total_params:,}" if isinstance(total_params, int) else f"Parameters: {total_params}"
                }
                
                ops_count = {}  # YOLOv8模型不容易获取操作类型统计
                
                return {
                    "model_type": model_type,
                    "model_info": model_info,
                    "inputs": inputs,
                    "outputs": outputs,
                    "total_params": total_params if isinstance(total_params, int) else "未知",
                    "ops_count": ops_count
                }
            
            except Exception as yolo_error:
                logger.warning(f"使用YOLO加载模型失败: {str(yolo_error)}，尝试使用PyTorch直接加载")
                
                # 尝试使用PyTorch直接加载
                model = torch.load(model_path, map_location="cpu")
                
                if isinstance(model, dict) and "model" in model:
                    # 有些PyTorch模型保存为字典格式，模型在'model'键中
                    if hasattr(model["model"], "names") and hasattr(model["model"], "yaml"):
                        # 可能是旧版YOLO模型
                        model_type = "YOLOv5 or older"
                        model_info = {
                            "names": f"Classes: {model['model'].names}",
                            "yaml": f"Config: {model['model'].yaml}"
                        }
                    else:
                        model_type = "PyTorch (dict)"
                        model_info = {k: str(v) for k, v in model.items() if not isinstance(v, torch.Tensor) and not isinstance(v, dict)}
                else:
                    model_type = "PyTorch"
                    model_info = {}
                
                # 由于没有可靠的方法获取一般PyTorch模型的输入/输出格式，我们提供有限的信息
                return {
                    "model_type": model_type,
                    "model_info": model_info,
                    "warning": tr("无法完全解析此PyTorch模型的输入/输出格式，仅提供基本信息"),
                    "suggestion": tr("请考虑将模型转换为ONNX格式以获取更详细的信息")
                }
                
        except Exception as e:
            logger.error(f"分析PyTorch模型失败: {str(e)}\n{traceback.format_exc()}")
            return {"error": tr(f"分析PyTorch模型失败: {str(e)}")} 