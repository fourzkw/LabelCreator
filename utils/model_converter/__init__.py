"""
模型转换工具模块

提供 PyTorch 模型转换为 ONNX 和 TensorRT 格式的功能
"""

from .onnx_converter import ONNXConverter
from .tensorrt_converter import TensorRTConverter

class ModelConverter:
    """
    模型转换工具类
    提供统一的接口进行模型格式转换
    """
    
    @staticmethod
    def pt_to_onnx(
        input_path,
        output_path=None,
        img_size=(640, 640),
        simplify=True,
        opset=12,
        half=False
    ):
        """
        Convert PyTorch model to ONNX format
        
        Args:
            input_path (str): Path to the PT model file
            output_path (str, optional): Output path for ONNX model. If None, uses input path with .onnx extension.
            img_size (tuple, optional): Input image size. Defaults to (640, 640).
            simplify (bool, optional): Whether to simplify the ONNX model. Defaults to True.
            opset (int, optional): ONNX opset version. Defaults to 12.
            half (bool, optional): Whether to use half precision (FP16). Defaults to False.
            
        Returns:
            tuple: (bool, str) - (success, output_path_or_error_message)
        """
        return ONNXConverter.convert(
            input_path=input_path,
            output_path=output_path,
            img_size=img_size,
            simplify=simplify,
            opset=opset,
            half=half
        )
    
    @staticmethod
    def pt_to_tensorrt(
        input_path,
        output_path=None,
        img_size=(640, 640),
        half=False,
        int8=False,
        workspace=4,
        device=0
    ):
        """
        Convert PyTorch model to TensorRT format
        
        Args:
            input_path (str): Path to the PT model file
            output_path (str, optional): Output path for TensorRT model. If None, uses input path with .engine extension.
            img_size (tuple, optional): Input image size. Defaults to (640, 640).
            half (bool, optional): Whether to use half precision (FP16). Defaults to False.
            int8 (bool, optional): Whether to use INT8 quantization. Defaults to False.
            workspace (int, optional): TensorRT workspace size in GB. Defaults to 4.
            device (int, optional): CUDA device ID. Defaults to 0.
            
        Returns:
            tuple: (bool, str) - (success, output_path_or_error_message)
        """
        return TensorRTConverter.convert(
            input_path=input_path,
            output_path=output_path,
            img_size=img_size,
            half=half,
            int8=int8,
            workspace=workspace,
            device=device
        )

__all__ = ['ModelConverter', 'ONNXConverter', 'TensorRTConverter']

