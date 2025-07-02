import os
from PyQt5.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
                           QTreeWidget, QTreeWidgetItem, QFileDialog, QMessageBox,
                           QGroupBox, QSplitter, QApplication, QProgressDialog,
                           QStyle)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QFont, QClipboard

from utils.model_analyzer import ModelAnalyzer
from i18n import tr
import logging

logger = logging.getLogger('YOLOLabelCreator.ModelInspectorDialog')

class ModelAnalysisThread(QThread):
    """后台线程用于分析模型，避免UI阻塞"""
    analysis_complete = pyqtSignal(dict)
    
    def __init__(self, model_path):
        super().__init__()
        self.model_path = model_path
    
    def run(self):
        """运行模型分析"""
        result = ModelAnalyzer.analyze_model(self.model_path)
        self.analysis_complete.emit(result)

class ModelInspectorDialog(QDialog):
    """模型检查对话框，用于分析和显示模型的输入输出格式"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle(tr("模型结构查看器"))
        self.resize(800, 600)
        self.setup_ui()
        self.model_result = None  # 存储模型分析结果
    
    def setup_ui(self):
        """设置对话框UI"""
        layout = QVBoxLayout(self)
        
        # 文件选择部分
        file_layout = QHBoxLayout()
        file_label = QLabel(tr("模型文件:"))
        self.file_path_label = QLabel(tr("未选择文件"))
        self.file_path_label.setWordWrap(True)
        
        browse_button = QPushButton(tr("浏览..."))
        browse_button.clicked.connect(self.browse_model)
        
        file_layout.addWidget(file_label)
        file_layout.addWidget(self.file_path_label, 1)
        file_layout.addWidget(browse_button)
        
        layout.addLayout(file_layout)
        
        # 创建分割器，上半部分显示模型信息，下半部分显示输入输出
        splitter = QSplitter(Qt.Vertical)
        
        # 模型基本信息区域
        self.info_group = QGroupBox(tr("模型信息"))
        info_layout = QVBoxLayout(self.info_group)
        
        self.model_type_label = QLabel(tr("模型类型: 未知"))
        self.model_type_label.setFont(QFont("Arial", 10, QFont.Bold))
        info_layout.addWidget(self.model_type_label)
        
        self.info_tree = QTreeWidget()
        self.info_tree.setHeaderLabels([tr("属性"), tr("值")])
        self.info_tree.setColumnWidth(0, 200)
        info_layout.addWidget(self.info_tree)
        
        # 输入输出信息区域
        io_splitter = QSplitter(Qt.Horizontal)
        
        # 输入格式
        self.input_group = QGroupBox(tr("输入格式"))
        input_layout = QVBoxLayout(self.input_group)
        self.input_tree = QTreeWidget()
        self.input_tree.setHeaderLabels([tr("名称"), tr("形状"), tr("数据类型")])
        input_layout.addWidget(self.input_tree)
        
        # 输出格式
        self.output_group = QGroupBox(tr("输出格式"))
        output_layout = QVBoxLayout(self.output_group)
        self.output_tree = QTreeWidget()
        self.output_tree.setHeaderLabels([tr("名称"), tr("形状"), tr("数据类型")])
        output_layout.addWidget(self.output_tree)
        
        # 操作统计
        self.ops_group = QGroupBox(tr("操作统计"))
        ops_layout = QVBoxLayout(self.ops_group)
        self.ops_tree = QTreeWidget()
        self.ops_tree.setHeaderLabels([tr("操作类型"), tr("数量")])
        ops_layout.addWidget(self.ops_tree)
        
        # 添加到IO分割器
        io_splitter.addWidget(self.input_group)
        io_splitter.addWidget(self.output_group)
        io_splitter.addWidget(self.ops_group)
        io_splitter.setSizes([250, 250, 250])
        
        # 添加到主分割器
        splitter.addWidget(self.info_group)
        splitter.addWidget(io_splitter)
        splitter.setSizes([200, 400])
        
        layout.addWidget(splitter)
        
        # 警告/错误信息
        self.warning_label = QLabel()
        self.warning_label.setStyleSheet("color: orange")
        self.warning_label.setWordWrap(True)
        self.warning_label.setVisible(False)
        layout.addWidget(self.warning_label)
        
        # 按钮区域
        button_layout = QHBoxLayout()
        
        # 一键复制所有属性按钮
        self.copy_all_button = QPushButton(tr("复制所有属性"))
        self.copy_all_button.setIcon(self.style().standardIcon(QStyle.SP_DialogSaveButton))
        self.copy_all_button.clicked.connect(self.copy_all_info)
        self.copy_all_button.setEnabled(False)  # 初始时禁用
        
        self.close_button = QPushButton(tr("关闭"))
        self.close_button.clicked.connect(self.accept)
        
        self.convert_button = QPushButton(tr("转换为ONNX"))
        self.convert_button.clicked.connect(self.convert_to_onnx)
        self.convert_button.setEnabled(False)
        
        button_layout.addWidget(self.copy_all_button)
        button_layout.addStretch()
        button_layout.addWidget(self.convert_button)
        button_layout.addWidget(self.close_button)
        
        layout.addLayout(button_layout)
    
    def browse_model(self):
        """浏览选择模型文件"""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            tr("选择模型文件"),
            "",
            tr("模型文件 (*.pt *.pth *.onnx);;所有文件 (*.*)")
        )
        
        if file_path:
            self.file_path_label.setText(file_path)
            self.analyze_model(file_path)
    
    def analyze_model(self, model_path):
        """分析模型结构"""
        # 清空之前的分析结果
        self.clear_ui()
        
        # 显示进度对话框
        progress = QProgressDialog(
            tr("正在分析模型..."),
            tr("取消"),
            0,
            0,
            self
        )
        progress.setWindowTitle(tr("模型分析"))
        progress.setWindowModality(Qt.WindowModal)
        progress.setAutoReset(False)
        progress.setAutoClose(False)
        progress.setValue(0)
        progress.show()
        QApplication.processEvents()
        
        # 创建分析线程
        self.analysis_thread = ModelAnalysisThread(model_path)
        self.analysis_thread.analysis_complete.connect(self.display_analysis_result)
        self.analysis_thread.finished.connect(progress.cancel)
        self.analysis_thread.start()
    
    def display_analysis_result(self, result):
        """显示模型分析结果"""
        if "error" in result:
            QMessageBox.critical(self, tr("错误"), result["error"])
            return
        
        # 保存分析结果，用于复制
        self.model_result = result
        
        # 更新模型类型
        model_type = result.get("model_type", "未知")
        self.model_type_label.setText(tr(f"模型类型: {model_type}"))
        
        # 如果是PT模型，启用转换按钮
        self.convert_button.setEnabled(model_type != "ONNX")
        
        # 启用复制按钮
        self.copy_all_button.setEnabled(True)
        
        # 显示警告信息（如果有）
        if "warning" in result:
            self.warning_label.setText(result["warning"])
            self.warning_label.setVisible(True)
            if "suggestion" in result:
                self.warning_label.setText(f"{result['warning']}\n{result['suggestion']}")
        else:
            self.warning_label.setVisible(False)
        
        # 填充模型信息树
        model_info = result.get("model_info", {})
        for key, value in model_info.items():
            if key == "metadata" and isinstance(value, dict):
                metadata_item = QTreeWidgetItem(self.info_tree, ["Metadata"])
                for meta_key, meta_value in value.items():
                    QTreeWidgetItem(metadata_item, [meta_key, str(meta_value)])
            else:
                QTreeWidgetItem(self.info_tree, [key, str(value)])
        
        # 填充输入树
        inputs = result.get("inputs", [])
        for input_info in inputs:
            name = input_info.get("name", "")
            shape = str(input_info.get("shape", []))
            data_type = input_info.get("data_type", "")
            QTreeWidgetItem(self.input_tree, [name, shape, data_type])
        
        # 填充输出树
        outputs = result.get("outputs", [])
        for output_info in outputs:
            name = output_info.get("name", "")
            shape = str(output_info.get("shape", []))
            data_type = output_info.get("data_type", "")
            QTreeWidgetItem(self.output_tree, [name, shape, data_type])
        
        # 填充操作统计树
        ops_count = result.get("ops_count", {})
        for op_type, count in ops_count.items():
            QTreeWidgetItem(self.ops_tree, [op_type, str(count)])
        
        # 添加总节点数（如果有）
        if "total_nodes" in result:
            QTreeWidgetItem(self.ops_tree, [tr("总节点数"), str(result["total_nodes"])])
        elif "total_params" in result:
            QTreeWidgetItem(self.ops_tree, [tr("总参数数"), str(result["total_params"])])
        
        # 展开所有树
        self.info_tree.expandAll()
        self.input_tree.expandAll()
        self.output_tree.expandAll()
        self.ops_tree.expandAll()
    
    def clear_ui(self):
        """清空UI中的数据"""
        self.info_tree.clear()
        self.input_tree.clear()
        self.output_tree.clear()
        self.ops_tree.clear()
        self.warning_label.setText("")
        self.warning_label.setVisible(False)
        self.model_type_label.setText(tr("模型类型: 未知"))
        self.copy_all_button.setEnabled(False)
        self.model_result = None
    
    def convert_to_onnx(self):
        """将模型转换为ONNX格式"""
        model_path = self.file_path_label.text()
        if not os.path.exists(model_path):
            QMessageBox.warning(self, tr("警告"), tr("请先选择有效的模型文件"))
            return
        
        # 这里我们使用已经实现的模型转换对话框
        try:
            from ui.model_converter_dialog import ModelConverterDialog
            converter_dialog = ModelConverterDialog(self)
            
            # 预先填充输入路径
            converter_dialog.input_path_edit.setText(model_path)
            
            # 自动生成输出路径
            output_path = os.path.splitext(model_path)[0] + '.onnx'
            converter_dialog.output_path_edit.setText(output_path)
            
            # 更新转换按钮状态
            converter_dialog.update_convert_button()
            
            if converter_dialog.exec_() == QDialog.Accepted:
                # 如果转换成功并且用户接受，重新分析新的ONNX模型
                new_model_path = converter_dialog.output_path_edit.text()
                if os.path.exists(new_model_path):
                    self.file_path_label.setText(new_model_path)
                    self.analyze_model(new_model_path)
            
        except Exception as e:
            logger.error(f"打开模型转换对话框失败: {str(e)}")
            QMessageBox.warning(self, tr("错误"), tr(f"打开模型转换对话框失败: {str(e)}"))
    
    def copy_all_info(self):
        """复制所有模型属性信息到剪贴板"""
        if not self.model_result:
            QMessageBox.warning(self, tr("警告"), tr("没有可复制的模型信息"))
            return
        
        try:
            # 构建信息文本
            info_text = self._format_model_info_as_text()
            
            # 复制到剪贴板
            clipboard = QApplication.clipboard()
            clipboard.setText(info_text)
            
            # 显示成功提示
            QMessageBox.information(self, tr("复制成功"), tr("所有模型属性已复制到剪贴板"))
            
        except Exception as e:
            logger.error(f"复制模型信息失败: {str(e)}")
            QMessageBox.warning(self, tr("错误"), tr(f"复制模型信息失败: {str(e)}"))
    
    def _format_model_info_as_text(self):
        """将模型信息格式化为文本"""
        result = self.model_result
        text_lines = []
        
        # 模型文件路径
        text_lines.append(f"模型文件: {self.file_path_label.text()}")
        text_lines.append("")
        
        # 模型类型
        model_type = result.get("model_type", "未知")
        text_lines.append(f"模型类型: {model_type}")
        text_lines.append("")
        
        # 警告信息
        if "warning" in result:
            text_lines.append(f"警告: {result['warning']}")
            if "suggestion" in result:
                text_lines.append(f"建议: {result['suggestion']}")
            text_lines.append("")
        
        # 模型信息
        text_lines.append("=== 模型信息 ===")
        model_info = result.get("model_info", {})
        for key, value in model_info.items():
            if key == "metadata" and isinstance(value, dict):
                text_lines.append("Metadata:")
                for meta_key, meta_value in value.items():
                    text_lines.append(f"  {meta_key}: {meta_value}")
            else:
                text_lines.append(f"{key}: {value}")
        text_lines.append("")
        
        # 输入信息
        text_lines.append("=== 输入格式 ===")
        inputs = result.get("inputs", [])
        if inputs:
            for input_info in inputs:
                name = input_info.get("name", "")
                shape = input_info.get("shape", [])
                data_type = input_info.get("data_type", "")
                text_lines.append(f"名称: {name}")
                text_lines.append(f"形状: {shape}")
                text_lines.append(f"数据类型: {data_type}")
                text_lines.append("")
        else:
            text_lines.append("无输入信息")
            text_lines.append("")
        
        # 输出信息
        text_lines.append("=== 输出格式 ===")
        outputs = result.get("outputs", [])
        if outputs:
            for output_info in outputs:
                name = output_info.get("name", "")
                shape = output_info.get("shape", [])
                data_type = output_info.get("data_type", "")
                text_lines.append(f"名称: {name}")
                text_lines.append(f"形状: {shape}")
                text_lines.append(f"数据类型: {data_type}")
                text_lines.append("")
        else:
            text_lines.append("无输出信息")
            text_lines.append("")
        
        # 操作统计
        text_lines.append("=== 操作统计 ===")
        ops_count = result.get("ops_count", {})
        if ops_count:
            for op_type, count in ops_count.items():
                text_lines.append(f"{op_type}: {count}")
            
            # 添加总节点数或总参数数
            if "total_nodes" in result:
                text_lines.append(f"总节点数: {result['total_nodes']}")
            elif "total_params" in result:
                text_lines.append(f"总参数数: {result['total_params']}")
        else:
            text_lines.append("无操作统计信息")
        
        # 生成最终文本
        return "\n".join(text_lines) 