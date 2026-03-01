"""
模型转换器基类
"""

import os
import logging
from abc import ABC, abstractmethod

logger = logging.getLogger('YOLOLabelCreator.ModelConverter')


class BaseConverter(ABC):
    """模型转换器基类"""
    
    @staticmethod
    def validate_input_path(input_path):
        """
        验证输入文件路径
        
        Args:
            input_path (str): 输入文件路径
            
        Returns:
            tuple: (bool, str) - (is_valid, error_message)
        """
        if not os.path.exists(input_path):
            error_msg = f"Input model file not found: {input_path}"
            logger.error(error_msg)
            return False, error_msg
        return True, None
    
    @staticmethod
    def get_default_output_path(input_path, extension):
        """
        生成默认输出路径
        
        Args:
            input_path (str): 输入文件路径
            extension (str): 输出文件扩展名（包含点号，如 '.onnx'）
            
        Returns:
            str: 默认输出路径
        """
        return os.path.splitext(input_path)[0] + extension
    
    @staticmethod
    def move_output_file(source_path, target_path):
        """
        移动输出文件到目标位置
        
        Args:
            source_path (str): 源文件路径
            target_path (str): 目标文件路径
        """
        if source_path != target_path and os.path.exists(source_path):
            os.rename(source_path, target_path)
    
    @abstractmethod
    def convert(self, input_path, output_path=None, **kwargs):
        """
        执行模型转换（抽象方法）
        
        Args:
            input_path (str): 输入文件路径
            output_path (str, optional): 输出文件路径
            **kwargs: 其他转换参数
            
        Returns:
            tuple: (bool, str) - (success, output_path_or_error_message)
        """
        pass

