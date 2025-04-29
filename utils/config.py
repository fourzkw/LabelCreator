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
    
    # 在 get_model_params 方法中添加新的参数
    def get_model_params(self):
        """获取模型参数"""
        params = {
            "model_path": self.settings.value("model/path", ""),
            "confidence_threshold": float(self.settings.value("model/confidence_threshold", 0.5)),
            "iou_threshold": float(self.settings.value("model/iou_threshold", 0.45)),
            "max_detections": int(self.settings.value("model/max_detections", 100)),
            "enable_auto_predict": self.settings.value("model/enable_auto_predict", False, type=bool),
            "device": self.settings.value("model/device", "cpu"),
            # 添加新的参数
            "model_version": self.settings.value("model/version", "yolov8"),
            "model_format": self.settings.value("model/format", "pt")
        }
        return params
    
    # 在 save_model_params 方法中保存新的参数
    def save_model_params(self, params):
        """保存模型参数"""
        self.settings.setValue("model/path", params.get("model_path", ""))
        self.settings.setValue("model/confidence_threshold", params.get("confidence_threshold", 0.5))
        self.settings.setValue("model/iou_threshold", params.get("iou_threshold", 0.45))
        self.settings.setValue("model/max_detections", params.get("max_detections", 100))
        self.settings.setValue("model/enable_auto_predict", params.get("enable_auto_predict", False))
        self.settings.setValue("model/device", params.get("device", "cpu"))
        # 保存新的参数
        self.settings.setValue("model/version", params.get("model_version", "yolov8"))
        self.settings.setValue("model/format", params.get("model_format", "pt"))
        self.settings.sync()
        return True
    
    # 在 reset_model_params 方法中重置新的参数
    def reset_model_params(self):
        """重置模型参数为默认值"""
        default_params = {
            "model_path": "",
            "confidence_threshold": 0.5,
            "iou_threshold": 0.45,
            "max_detections": 100,
            "enable_auto_predict": False,
            "device": "cpu",
            # 添加新的默认参数
            "model_version": "yolov8",
            "model_format": "pt"
        }
        self.save_model_params(default_params)
        return default_params
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