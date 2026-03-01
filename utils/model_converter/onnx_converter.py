"""
ONNX 模型转换器
"""

import os
import logging
import traceback
from ultralytics import YOLO
from .base import BaseConverter

logger = logging.getLogger('YOLOLabelCreator.ModelConverter.ONNX')


class ONNXConverter(BaseConverter):
    """ONNX 模型转换器"""
    
    @staticmethod
    def convert(
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
        try:
            logger.info(f"Starting PT to ONNX conversion for {input_path}")
            
            # Validate input path
            is_valid, error_msg = ONNXConverter.validate_input_path(input_path)
            if not is_valid:
                return False, error_msg
            
            # Set default output path if not provided
            if output_path is None:
                output_path = ONNXConverter.get_default_output_path(input_path, '.onnx')
            
            # Load the model using ultralytics
            model = YOLO(input_path)
            
            # Export the model to ONNX format
            model.export(format='onnx', imgsz=img_size, simplify=simplify, opset=opset, half=half)
            
            # The YOLO export function saves the model in the same directory as the input
            # with a .onnx extension. Let's move it if necessary.
            default_output = ONNXConverter.get_default_output_path(input_path, '.onnx')
            ONNXConverter.move_output_file(default_output, output_path)
            
            logger.info(f"Successfully converted model to ONNX: {output_path}")
            return True, output_path
            
        except Exception as e:
            error_msg = f"Error converting model to ONNX: {str(e)}\n{traceback.format_exc()}"
            logger.error(error_msg)
            return False, error_msg

