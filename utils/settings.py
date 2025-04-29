import os
import json
import logging

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
    'auto_label_all': 'Ctrl+Shift+A',  # 添加批量标注的默认快捷键
    'toggle_keypoint_mode': 'Ctrl+K'   # 添加特征点编辑模式的默认快捷键
}

class Settings:
    def __init__(self, app_dir):
        self.app_dir = app_dir
        self.settings_file = os.path.join(app_dir, 'settings.json')
        self.shortcuts = DEFAULT_SHORTCUTS.copy()
        self.load_settings()
    
    def load_settings(self):
        """从文件加载设置"""
        try:
            if os.path.exists(self.settings_file):
                with open(self.settings_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    if 'shortcuts' in data:
                        # 更新快捷键，保留默认值作为备份
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