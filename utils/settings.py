import os
import json
import logging
from PyQt5.QtCore import QSettings

logger = logging.getLogger('YOLOLabelCreator.Settings')

DEFAULT_SHORTCUTS = {
    'save_current': 'Ctrl+S',
    'save_all': 'Ctrl+Shift+S',
    'prev_image': 'Left',
    'next_image': 'Right',
    'delete_box': 'Delete',
    'zoom_in': 'Ctrl++',
    'zoom_out': 'Ctrl+-',
    'reset_zoom': 'Ctrl+0',
    'open_directory': 'Ctrl+O',
    'exit': 'Ctrl+Q',
    'auto_label': 'Ctrl+A',
    'auto_label_all': 'Ctrl+Shift+A',
    'toggle_keypoint_mode': 'Ctrl+K'
}

# 默认模型预测参数
DEFAULT_MODEL_PARAMS = {
    "model_path": "",
    "confidence_threshold": 0.5,
    "iou_threshold": 0.45,
    "max_detections": 100,
    "enable_auto_predict": False,
    "auto_label_include_labeled": True,
    "device": "cpu",
    "model_version": "yolov8",
    "model_format": "pt",
    "class_mapping": {}  # 标注映射：{识别类别ID: 标注类别ID}
}

class Settings:
    """统一的应用程序设置管理类，负责所有配置的读写操作"""
    
    _instance = None
    
    def __new__(cls, app_dir=None):
        if cls._instance is None:
            cls._instance = super(Settings, cls).__new__(cls)
            cls._instance._init_settings(app_dir)
        return cls._instance
    
    def _init_settings(self, app_dir):
        """初始化设置"""
        self.app_dir = app_dir
        self.settings_file = os.path.join(app_dir, 'config', 'settings.json')
        self.shortcuts = DEFAULT_SHORTCUTS.copy()
        self.qsettings = QSettings()
        
        # 加载设置
        self.load_settings()
        
        # 初始化模型参数设置
        for key, value in DEFAULT_MODEL_PARAMS.items():
            param_key = f"model/{key}"
            if not self.qsettings.contains(param_key):
                self.qsettings.setValue(param_key, value)
    
    def load_settings(self):
        """从文件加载设置"""
        try:
            if os.path.exists(self.settings_file):
                with open(self.settings_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    if 'shortcuts' in data:
                        for key, value in data['shortcuts'].items():
                            if key in self.shortcuts:
                                self.shortcuts[key] = value
                logger.info(f"已从 {self.settings_file} 加载设置")
            else:
                logger.info("未找到设置文件，使用默认设置")
                self.save_settings()  # 创建默认设置文件
        except Exception as e:
            logger.error(f"加载设置时出错: {str(e)}")
    
    def save_settings(self):
        """保存设置到文件"""
        try:
            data = {
                'shortcuts': self.shortcuts
            }
            with open(self.settings_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=4, ensure_ascii=False)
            logger.info(f"设置已保存到 {self.settings_file}")
            self.qsettings.sync()
            return True
        except Exception as e:
            logger.error(f"保存设置时出错: {str(e)}")
            return False
    
    def get_shortcut(self, action_name):
        """获取指定操作的快捷键"""
        return self.shortcuts.get(action_name, '')
    
    def set_shortcut(self, action_name, shortcut):
        """设置指定操作的快捷键"""
        if action_name in self.shortcuts:
            self.shortcuts[action_name] = shortcut
            return True
        return False
    
    def reset_shortcuts(self):
        """重置所有快捷键为默认值"""
        self.shortcuts = DEFAULT_SHORTCUTS.copy()
        return True
    
    # 整合自Config类的方法
    def get_model_params(self):
        """获取模型参数"""
        # 读取标注映射
        class_mapping_str = self.qsettings.value("model/class_mapping", "")
        class_mapping = {}
        if class_mapping_str:
            try:
                class_mapping = json.loads(class_mapping_str)
                # 确保键和值都是整数
                class_mapping = {int(k): int(v) for k, v in class_mapping.items()}
            except Exception as e:
                logger.warning(f"解析标注映射失败: {str(e)}")
                class_mapping = {}
        
        params = {
            "model_path": self.qsettings.value("model/model_path", ""),
            "confidence_threshold": float(self.qsettings.value("model/confidence_threshold", 0.5)),
            "iou_threshold": float(self.qsettings.value("model/iou_threshold", 0.45)),
            "max_detections": int(self.qsettings.value("model/max_detections", 100)),
            "enable_auto_predict": self._to_bool(self.qsettings.value("model/enable_auto_predict", False)),
            "auto_label_include_labeled": self._to_bool(self.qsettings.value("model/auto_label_include_labeled", True)),
            "device": self.qsettings.value("model/device", "cpu"),
            "model_version": self.qsettings.value("model/model_version", "yolov8"),
            "model_format": self.qsettings.value("model/model_format", "pt"),
            "keypoints_number": int(self.qsettings.value("model/keypoints_number", 0)),
            "class_mapping": class_mapping
        }
        return params
    
    def _to_bool(self, value):
        """将各种类型的值转换为布尔值"""
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.lower() == 'true'
        # 对于数字，0被视为False，其他为True
        if isinstance(value, (int, float)):
            return bool(value)
        # 默认返回False
        return False
    
    def save_model_params(self, params):
        """保存模型参数"""
        self.qsettings.setValue("model/model_path", params.get("model_path", ""))
        self.qsettings.setValue("model/confidence_threshold", float(params.get("confidence_threshold", 0.5)))
        self.qsettings.setValue("model/iou_threshold", float(params.get("iou_threshold", 0.45)))
        self.qsettings.setValue("model/max_detections", int(params.get("max_detections", 100)))
        self.qsettings.setValue("model/enable_auto_predict", bool(params.get("enable_auto_predict", False)))
        # 兼容旧调用方：如果没传该字段，则保留现有值不变
        if "auto_label_include_labeled" in params:
            self.qsettings.setValue("model/auto_label_include_labeled", bool(params.get("auto_label_include_labeled", True)))
        self.qsettings.setValue("model/device", params.get("device", "cpu"))
        self.qsettings.setValue("model/model_version", params.get("model_version", "yolov8"))
        self.qsettings.setValue("model/model_format", params.get("model_format", "pt"))
        self.qsettings.setValue("model/keypoints_number", int(params.get("keypoints_number", 0)))
        # 保存标注映射
        class_mapping = params.get("class_mapping", {})
        if isinstance(class_mapping, dict):
            # 确保键和值都是整数
            class_mapping = {int(k): int(v) for k, v in class_mapping.items()}
            self.qsettings.setValue("model/class_mapping", json.dumps(class_mapping))
        self.qsettings.sync()
        return True
    
    def reset_model_params(self):
        """重置模型参数为默认值"""
        self.save_model_params(DEFAULT_MODEL_PARAMS)
        return DEFAULT_MODEL_PARAMS.copy()