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
    "Failed to save settings": "保存设置失败",
    "Settings updated": "设置已更新",
    
    # 批处理相关
    "Auto Label": "自动标注",
    "Auto Label All": "批量自动标注",
    "Images in selected folder": "所选文件夹中的图像",
    "Class Management": "类别管理",
    "Enter new class name": "输入新类别名称",
    "批量自动标注": "批量自动标注",
    "正在处理图像...": "正在处理图像...",
    "批量标注完成，成功处理 {0} 张图像": "批量标注完成，成功处理 {0} 张图像",
    "正在处理 ({0}/{1}): {2}": "正在处理 ({0}/{1}): {2}",
    "取消": "取消",
    "特征点编辑": "特征点编辑",
    
    # 数据集划分相关
    "数据集划分": "数据集划分",
    "源数据集路径:": "源数据集路径:",
    "输出路径:": "输出路径:",
    "浏览...": "浏览...",
    "划分比例": "划分比例",
    "训练集比例:": "训练集比例:",
    "验证集比例:": "验证集比例:",
    "测试集比例:": "测试集比例:",
    "随机种子:": "随机种子:",
    "创建YAML配置文件": "创建YAML配置文件",
    "开始划分": "开始划分",
    "选择输出文件夹": "选择输出文件夹",
    "错误": "错误",
    
    # 模型转换相关
    "PT模型转ONNX": "PT模型转ONNX",
    "PT to ONNX Model Converter": "PT模型转ONNX转换器",
    "Input Model (PyTorch)": "输入模型 (PyTorch)",
    "Output Model (ONNX)": "输出模型 (ONNX)",
    "Browse...": "浏览...",
    "Conversion Parameters": "转换参数",
    "Image Size:": "图像尺寸:",
    "Width:": "宽度:",
    "Height:": "高度:",
    "ONNX Opset:": "ONNX操作集:",
    "Simplify Model": "简化模型",
    "Half Precision (FP16)": "半精度 (FP16)",
    "Convert": "转换",
    "Cancel": "取消",
    "Select PyTorch Model": "选择PyTorch模型",
    "PyTorch Models (*.pt *.pth);;All Files (*.*)": "PyTorch模型 (*.pt *.pth);;所有文件 (*.*)",
    "Save ONNX Model As": "ONNX模型另存为",
    "ONNX Models (*.onnx);;All Files (*.*)": "ONNX模型 (*.onnx);;所有文件 (*.*)",
    "Converting model...": "正在转换模型...",
    "Model Conversion": "模型转换",
    "Conversion Complete": "转换完成",
    "Model successfully converted to ONNX format at:\n{0}": "模型已成功转换为ONNX格式，保存于:\n{0}",
    "Conversion Error": "转换错误",
    "Error converting model:\n{0}": "转换模型时出错:\n{0}",
    "打开模型转换对话框失败": "打开模型转换对话框失败",

    # 模型检查器相关
    "模型结构查看器": "模型结构查看器",
    "模型文件:": "模型文件:",
    "未选择文件": "未选择文件",
    "模型信息": "模型信息",
    "模型类型: 未知": "模型类型: 未知",
    "模型类型: {0}": "模型类型: {0}",
    "属性": "属性",
    "值": "值",
    "输入格式": "输入格式",
    "输出格式": "输出格式",
    "操作统计": "操作统计",
    "名称": "名称",
    "形状": "形状",
    "数据类型": "数据类型",
    "操作类型": "操作类型",
    "数量": "数量",
    "总节点数": "总节点数",
    "总参数数": "总参数数",
    "关闭": "关闭",
    "转换为ONNX": "转换为ONNX",
    "选择模型文件": "选择模型文件",
    "模型文件 (*.pt *.pth *.onnx);;所有文件 (*.*)": "模型文件 (*.pt *.pth *.onnx);;所有文件 (*.*)",
    "正在分析模型...": "正在分析模型...",
    "模型分析": "模型分析",
    "取消": "取消",
    "模型文件不存在": "模型文件不存在",
    "不支持的模型格式: {0}": "不支持的模型格式: {0}",
    "分析模型失败: {0}": "分析模型失败: {0}",
    "ONNX库未安装，请安装onnx和onnxruntime": "ONNX库未安装，请安装onnx和onnxruntime",
    "分析ONNX模型失败: {0}": "分析ONNX模型失败: {0}",
    "分析PyTorch模型失败: {0}": "分析PyTorch模型失败: {0}",
    "无法完全解析此PyTorch模型的输入/输出格式，仅提供基本信息": "无法完全解析此PyTorch模型的输入/输出格式，仅提供基本信息",
    "请考虑将模型转换为ONNX格式以获取更详细的信息": "请考虑将模型转换为ONNX格式以获取更详细的信息",
    "打开模型结构查看器失败": "打开模型结构查看器失败",
    "警告": "警告",
    "请先选择有效的模型文件": "请先选择有效的模型文件",
    
    # 复制所有属性功能相关
    "复制所有属性": "复制所有属性",
    "没有可复制的模型信息": "没有可复制的模型信息",
    "复制成功": "复制成功",
    "所有模型属性已复制到剪贴板": "所有模型属性已复制到剪贴板",
    "复制模型信息失败: {0}": "复制模型信息失败: {0}"
}

def tr(text):
    """翻译函数，如果找不到翻译则返回原文"""
    return translations.get(text, text)