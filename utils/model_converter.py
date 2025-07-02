import os
import logging
import torch
import traceback
from ultralytics import YOLO
from i18n import tr

logger = logging.getLogger('YOLOLabelCreator.ModelConverter')

class ModelConverter:
    """
    Model conversion utilities for YOLO models
    Currently supports converting PyTorch (.pt) models to ONNX format
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
            bool: True if conversion was successful, False otherwise
            str: Path to the output file if successful, error message otherwise
        """
        try:
            logger.info(f"Starting PT to ONNX conversion for {input_path}")
            
            # Validate input path
            if not os.path.exists(input_path):
                error_msg = f"Input model file not found: {input_path}"
                logger.error(error_msg)
                return False, error_msg
            
            # Set default output path if not provided
            if output_path is None:
                output_path = os.path.splitext(input_path)[0] + '.onnx'
            
            # Load the model using ultralytics
            model = YOLO(input_path)
            
            # Export the model to ONNX format
            model.export(format='onnx', imgsz=img_size, simplify=simplify, opset=opset, half=half)
            
            # The YOLO export function saves the model in the same directory as the input
            # with a .onnx extension. Let's move it if necessary.
            default_output = os.path.splitext(input_path)[0] + '.onnx'
            if default_output != output_path and os.path.exists(default_output):
                os.rename(default_output, output_path)
            
            logger.info(f"Successfully converted model to ONNX: {output_path}")
            return True, output_path
            
        except Exception as e:
            error_msg = f"Error converting model to ONNX: {str(e)}\n{traceback.format_exc()}"
            logger.error(error_msg)
            return False, error_msg 