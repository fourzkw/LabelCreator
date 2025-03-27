# 界面文本翻译

# 在translations字典中添加新的翻译项
translations = {
    # 窗口标题
    "YOLO Label Creator": "YOLO标注工具",
    
    # 目录选择
    "No directory selected": "未选择目录",
    "Select Directory": "选择目录",
    
    # 类别管理
    "Class:": "类别：",
    "Add Class": "添加类别",
    
    # 图像和边界框
    "Images:": "图像：",
    "Bounding Boxes:": "边界框：",
    "Delete Selected Box": "删除选中的框",
    "Box": "框",
    "Unknown": "未知",
    
    # 导航按钮
    "Previous Image": "上一张图像",
    "Next Image": "下一张图像",
    
    # 保存按钮
    "Save Current": "保存当前",
    "Save All": "保存全部",
    
    # 缩放控制
    "Zoom In": "放大",
    "Zoom Out": "缩小",
    "Reset Zoom": "重置缩放",
    
    # 菜单
    "File": "文件",
    "Open Directory": "打开目录",
    "Exit": "退出",
    
    # 消息框
    "Warning": "警告",
    "Error": "错误",
    "Success": "成功",
    "No image loaded": "未加载图像",
    "No images loaded": "未加载任何图像",
    "Failed to load annotations": "加载标注失败",
    "Failed to save annotations": "保存标注失败",
    "All annotations saved successfully": "所有标注已成功保存",
    "Failed to load image": "加载图像失败",
    "Image file not found": "图像文件未找到",
    "Cannot load annotations: No valid image loaded": "无法加载标注：未加载有效图像",
    "Cannot save annotations: No valid image loaded": "无法保存标注：未加载有效图像",
    "Project structure created successfully": "项目结构创建成功",
    "Failed to create project structure": "创建项目结构失败",
    "Folders": "文件夹",
    
    # 设置相关
    "Settings": "设置",
    "Preferences": "首选项",
    "Shortcuts": "快捷键",
    "Action": "操作",
    "Shortcut": "快捷键",
    "Reset to Default": "重置为默认",
    "Save": "保存",
    "Cancel": "取消",
    "Confirm Reset": "确认重置",
    "Are you sure you want to reset all shortcuts to default?": "确定要将所有快捷键重置为默认值吗？",
    "Error": "错误",
    "Failed to save settings": "保存设置失败",
    "Settings updated": "设置已更新",
    "操作": "操作",
    "快捷键": "快捷键",
    "保存当前": "保存当前",
    "保存全部": "保存全部",
    "上一张图像": "上一张图像",
    "下一张图像": "下一张图像",
    "删除选中框": "删除选中框",
    "放大": "放大",
    "缩小": "缩小",
    "重置缩放": "重置缩放",
    "打开目录": "打开目录",
    "退出": "退出",
    "自动标注": "自动标注",
    "首选项": "首选项",
    "设置": "设置",
    "重置为默认": "重置为默认",
    "确认重置": "确认重置",
    "设置已更新": "设置已更新"
}

def tr(text):
    """翻译函数，如果找不到翻译则返回原文"""
    return translations.get(text, text)