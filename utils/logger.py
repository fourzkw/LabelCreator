import logging
import os
import sys

def setup_logger(name='YOLOLabelCreator', log_file='app.log', level=logging.DEBUG):
    """
    配置并返回一个日志记录器
    
    Args:
        name (str): 日志记录器名称
        log_file (str): 日志文件路径（如果不是绝对路径，会自动保存到logs目录）
        level (int): 日志级别
        
    Returns:
        logging.Logger: 配置好的日志记录器
    """
    # 创建日志记录器
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # 如果不是绝对路径，则将日志文件保存到logs目录
    if not os.path.isabs(log_file):
        # 获取项目根目录
        current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        logs_dir = os.path.join(current_dir, 'logs')
        
        # 确保logs目录存在
        os.makedirs(logs_dir, exist_ok=True)
        
        # 构建完整的日志文件路径
        log_file = os.path.join(logs_dir, log_file)
    
    # 创建文件处理器
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(level)
    
    # 创建控制台处理器
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    
    # 创建格式化器
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # 添加处理器到日志记录器
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger