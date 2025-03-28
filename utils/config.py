import os
import json
from PyQt5.QtCore import QSettings

class Config:
    """应用程序配置管理类"""
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(Config, cls).__new__(cls)
            cls._instance._init_config()
        return cls._instance
    
    def _init_config(self):
        """初始化配置"""
        self.settings = QSettings()
        
        # 默认模型预测参数
        self.default_model_params = {
            "confidence_threshold": 0.5,
            "iou_threshold": 0.45,
            "model_path": "",
            "enable_auto_predict": False,
            "max_detections": 100,
            "device": "cpu"  # 默认使用CPU设备
        }
        
        # 如果没有保存过模型参数，则使用默认值初始化
        if not self.settings.contains("model_params"):
            self.save_model_params(self.default_model_params)
    
    def get_model_params(self):
        """获取模型预测参数"""
        if self.settings.contains("model_params"):
            params_json = self.settings.value("model_params")
            try:
                return json.loads(params_json)
            except:
                return self.default_model_params
        return self.default_model_params
    
    def save_model_params(self, params):
        """保存模型预测参数"""
        params_json = json.dumps(params)
        self.settings.setValue("model_params", params_json)
        self.settings.sync()
    
    def reset_model_params(self):
        """重置模型预测参数为默认值"""
        self.save_model_params(self.default_model_params)
        return self.default_model_params