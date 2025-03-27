import sys
import os
import traceback
from PyQt5.QtWidgets import QApplication, QMessageBox

# 导入自定义模块
from ui.main_window import YOLOLabelCreator
from utils.logger import setup_logger
from i18n import tr

# 配置日志记录
logger = setup_logger('YOLOLabelCreator', 'app.log')

def exception_hook(exctype, value, traceback_obj):
    """全局异常处理函数"""
    error_msg = ''.join(traceback.format_exception(exctype, value, traceback_obj))
    logger.critical(f"未捕获的异常: {error_msg}")
    print(f"未捕获的异常: {error_msg}")
    # 显示错误消息框
    QMessageBox.critical(None, "Error", f"发生了意外错误:\n{str(value)}\n\n详细信息已记录到app.log")

if __name__ == "__main__":
    try:
        # 设置全局异常处理
        sys.excepthook = exception_hook
        
        logger.info("启动应用程序")
        app = QApplication(sys.argv)
        
        # 设置应用程序名称和组织名称，用于QSettings
        app.setApplicationName("YOLOLabelCreator")
        app.setOrganizationName("YOLOLabelCreator")
        
        window = YOLOLabelCreator()
        window.show()
        logger.info("应用程序窗口已显示")
        exit_code = app.exec_()
        logger.info(f"应用程序退出，代码: {exit_code}")
        sys.exit(exit_code)
    except Exception as e:
        logger.critical(f"主循环中的致命错误: {str(e)}")
        logger.critical(f"异常详情: {traceback.format_exc()}")
        print(f"致命错误: {str(e)}")
        sys.exit(1)