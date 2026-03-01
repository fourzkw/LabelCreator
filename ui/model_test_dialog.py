import os
import json
import subprocess
import threading
import time
import sys

from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtWidgets import (
    QDialog,
    QVBoxLayout,
    QHBoxLayout,
    QGroupBox,
    QFormLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QFileDialog,
    QComboBox,
    QDoubleSpinBox,
    QSpinBox,
    QTextEdit,
    QTableWidget,
    QTableWidgetItem,
    QMessageBox,
    QProgressBar,
    QTabWidget,
    QCheckBox,
    QWidget,
)

from i18n import tr


class TestConfigThread(QThread):
    """测试配置执行线程"""
    progress_updated = pyqtSignal(int, int)  # current, total
    config_started = pyqtSignal(int, dict)  # index, config
    config_completed = pyqtSignal(int, dict, dict)  # index, config, result
    log_message = pyqtSignal(str)
    all_completed = pyqtSignal(list)  # all results

    def __init__(self, configs, conda_env, script_path, logs_dir):
        super().__init__()
        self.configs = configs
        self.conda_env = conda_env
        self.script_path = script_path
        self.logs_dir = logs_dir
        self._stop_flag = False

    def stop(self):
        self._stop_flag = True

    def run(self):
        results = []
        total = len(self.configs)
        
        for idx, config in enumerate(self.configs):
            if self._stop_flag:
                break
                
            self.config_started.emit(idx, config)
            self.progress_updated.emit(idx, total)  # 发送当前索引和总数
            
            try:
                result = self._run_single_test(config, idx)
                if result:
                    results.append(result)
                    self.config_completed.emit(idx, config, result)
                else:
                    results.append(None)
            except Exception as e:
                self.log_message.emit(tr(f"配置 {idx+1} 测试失败: {str(e)}"))
                results.append(None)
        
        # 发送最终进度
        self.progress_updated.emit(total, total)
        self.all_completed.emit(results)

    def _run_single_test(self, config, idx):
        """运行单个测试配置 - 完全参考单个测试的方式"""
        model_path = os.path.abspath(config['model_path'])
        images_dir = os.path.abspath(config['images_dir'])
        labels_dir = os.path.abspath(config['labels_dir']) if config.get('labels_dir') else ""
        device = config.get('device', 'cpu')
        conf = config.get('conf', 0.25)
        iou = config.get('iou', 0.5)
        max_det = config.get('max_det', 300)
        
        out_path = os.path.join(self.logs_dir, f"model_test_result_{idx+1}.json")
        
        # 构建Python命令参数 - 和单个测试完全一致
        py_cmd_parts = [
            f'--model "{model_path}"',
            f'--images "{images_dir}"',
            f'--device {device}',
            f'--conf {conf}',
            f'--iou {iou}',
            f'--max-det {max_det}',
            f'--out "{out_path}"',
        ]
        if labels_dir:
            py_cmd_parts.append(f'--labels "{labels_dir}"')
        
        py_cmd = " ".join(py_cmd_parts)
        
        model_name = os.path.basename(model_path)
        self.log_message.emit(tr(f"[{idx+1}/{len(self.configs)}] 正在启动测试进程: {model_name}"))
        
        try:
            # 使用和单个测试完全相同的命令格式
            # 在新终端窗口中启动测试
            cmd = f'start cmd.exe /k "conda init cmd.exe && conda activate {self.conda_env} && python "{self.script_path}" {py_cmd}"'
            
            # 执行命令（不等待，立即返回）
            subprocess.Popen(cmd, shell=True)
            
            self.log_message.emit(tr(f"[{idx+1}/{len(self.configs)}] 已在新终端中启动测试进程，使用环境: {self.conda_env}"))
            self.log_message.emit(tr(f"[{idx+1}/{len(self.configs)}] 请在新打开的终端窗口中查看测试进度和日志"))
            self.log_message.emit(tr(f"[{idx+1}/{len(self.configs)}] 等待测试完成，结果将保存到: {out_path}"))
            
            # 等待结果文件生成（轮询检查）
            max_wait_time = 3600 * 24  # 最多等待24小时
            check_interval = 2  # 每2秒检查一次
            waited_time = 0
            
            while waited_time < max_wait_time:
                if self._stop_flag:
                    self.log_message.emit(tr(f"[{idx+1}/{len(self.configs)}] 测试已取消"))
                    return None
                
                if os.path.exists(out_path):
                    # 等待文件写入完成（文件大小稳定）
                    time.sleep(1)
                    try:
                        with open(out_path, 'r', encoding='utf-8') as f:
                            result = json.load(f)
                        self.log_message.emit(tr(f"[{idx+1}/{len(self.configs)}] 测试完成: {model_name}"))
                        return result
                    except (json.JSONDecodeError, IOError):
                        # 文件可能还在写入中，继续等待
                        pass
                
                time.sleep(check_interval)
                waited_time += check_interval
            
            # 超时
            self.log_message.emit(tr(f"[{idx+1}/{len(self.configs)}] 测试超时: {model_name}"))
            return None
            
        except Exception as e:
            self.log_message.emit(tr(f"[{idx+1}/{len(self.configs)}] 启动测试失败: {str(e)}"))
            return None


class ModelTestDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle(tr("模型测试"))
        self.setMinimumSize(1200, 800)

        self._result_json_path = None
        self._test_configs = []  # 测试配置列表
        self._test_results = []  # 测试结果列表
        self._test_thread = None  # 测试线程
        self._is_testing = False

        self._build_ui()
        self.refresh_conda_envs()

    def _build_ui(self):
        layout = QVBoxLayout(self)

        # 使用TabWidget分离单个测试和列表测试
        tabs = QTabWidget()
        
        # Tab 1: 单个测试
        single_tab = self._build_single_test_tab()
        tabs.addTab(single_tab, tr("单个测试"))
        
        # Tab 2: 列表测试
        batch_tab = self._build_batch_test_tab()
        tabs.addTab(batch_tab, tr("列表测试"))
        
        layout.addWidget(tabs)

    def _build_single_test_tab(self):
        """构建单个测试标签页"""
        widget = QVBoxLayout()
        container = QWidget()
        
        # settings
        settings_group = QGroupBox(tr("测试设置"))
        form = QFormLayout()

        # conda env
        self.conda_env_combo = QComboBox()
        self.refresh_env_btn = QPushButton(tr("刷新"))
        self.refresh_env_btn.clicked.connect(self.refresh_conda_envs)
        env_row = QHBoxLayout()
        env_row.addWidget(self.conda_env_combo, 1)
        env_row.addWidget(self.refresh_env_btn)
        form.addRow(tr("Conda环境:"), env_row)

        # model path
        self.model_path_edit = QLineEdit()
        self.model_browse_btn = QPushButton(tr("浏览..."))
        self.model_browse_btn.clicked.connect(self._browse_model)
        model_row = QHBoxLayout()
        model_row.addWidget(self.model_path_edit, 1)
        model_row.addWidget(self.model_browse_btn)
        form.addRow(tr("模型文件:"), model_row)

        # images dir
        self.images_dir_edit = QLineEdit()
        self.images_browse_btn = QPushButton(tr("浏览..."))
        self.images_browse_btn.clicked.connect(self._browse_images_dir)
        images_row = QHBoxLayout()
        images_row.addWidget(self.images_dir_edit, 1)
        images_row.addWidget(self.images_browse_btn)
        form.addRow(tr("图片目录:"), images_row)

        # labels dir optional
        self.labels_dir_edit = QLineEdit()
        self.labels_browse_btn = QPushButton(tr("浏览..."))
        self.labels_browse_btn.clicked.connect(self._browse_labels_dir)
        labels_row = QHBoxLayout()
        labels_row.addWidget(self.labels_dir_edit, 1)
        labels_row.addWidget(self.labels_browse_btn)
        form.addRow(tr("标签目录(可选):"), labels_row)

        # device
        self.device_combo = QComboBox()
        self.device_combo.addItems(["cpu", "cuda"])
        form.addRow(tr("device:"), self.device_combo)

        # thresholds
        self.conf_spin = QDoubleSpinBox()
        self.conf_spin.setRange(0.0, 1.0)
        self.conf_spin.setSingleStep(0.05)
        self.conf_spin.setValue(0.25)
        form.addRow(tr("置信度阈值(conf):"), self.conf_spin)

        self.iou_spin = QDoubleSpinBox()
        self.iou_spin.setRange(0.0, 1.0)
        self.iou_spin.setSingleStep(0.05)
        self.iou_spin.setValue(0.5)
        form.addRow(tr("IoU阈值(iou):"), self.iou_spin)

        self.max_det_spin = QSpinBox()
        self.max_det_spin.setRange(1, 9999)
        self.max_det_spin.setValue(300)
        form.addRow(tr("最大检测数(max_det):"), self.max_det_spin)

        settings_group.setLayout(form)
        widget.addWidget(settings_group)

        # actions
        action_row = QHBoxLayout()
        self.run_btn = QPushButton(tr("开始测试"))
        self.run_btn.clicked.connect(self.start_single_test)
        action_row.addWidget(self.run_btn)
        action_row.addStretch(1)
        widget.addLayout(action_row)

        # progress + log
        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        widget.addWidget(self.progress)

        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setMinimumHeight(160)
        widget.addWidget(self.log_text, 1)

        # results
        result_group = QGroupBox(tr("统计结果"))
        result_layout = QVBoxLayout(result_group)

        self.summary_text = QTextEdit()
        self.summary_text.setReadOnly(True)
        self.summary_text.setMinimumHeight(120)
        result_layout.addWidget(self.summary_text)

        self.table = QTableWidget(0, 8)
        self.table.setHorizontalHeaderLabels([
            tr("class"),
            tr("gt"),
            tr("pred"),
            tr("tp"),
            tr("fp"),
            tr("fn"),
            tr("precision"),
            tr("recall"),
        ])
        self.table.horizontalHeader().setStretchLastSection(True)
        result_layout.addWidget(self.table, 1)

        widget.addWidget(result_group, 2)
        
        container.setLayout(widget)
        return container

    def _build_batch_test_tab(self):
        """构建列表测试标签页"""
        widget = QVBoxLayout()
        container = QWidget()
        
        # 配置列表管理
        config_group = QGroupBox(tr("测试配置列表"))
        config_layout = QVBoxLayout(config_group)
        
        # 配置列表表格
        self.config_table = QTableWidget(0, 7)
        self.config_table.setHorizontalHeaderLabels([
            tr("模型文件"),
            tr("图片目录"),
            tr("标签目录"),
            tr("device"),
            tr("conf"),
            tr("iou"),
            tr("max_det"),
        ])
        self.config_table.horizontalHeader().setStretchLastSection(True)
        self.config_table.setSelectionBehavior(QTableWidget.SelectRows)
        config_layout.addWidget(self.config_table, 1)
        
        # 配置列表操作按钮
        config_btn_row = QHBoxLayout()
        self.add_config_btn = QPushButton(tr("添加当前配置"))
        self.add_config_btn.clicked.connect(self._add_current_config)
        self.edit_config_btn = QPushButton(tr("编辑选中"))
        self.edit_config_btn.clicked.connect(self._edit_selected_config)
        self.remove_config_btn = QPushButton(tr("删除选中"))
        self.remove_config_btn.clicked.connect(self._remove_selected_config)
        self.clear_configs_btn = QPushButton(tr("清空列表"))
        self.clear_configs_btn.clicked.connect(self._clear_configs)
        self.load_configs_btn = QPushButton(tr("加载配置列表"))
        self.load_configs_btn.clicked.connect(self._load_configs)
        self.save_configs_btn = QPushButton(tr("保存配置列表"))
        self.save_configs_btn.clicked.connect(self._save_configs)
        
        config_btn_row.addWidget(self.add_config_btn)
        config_btn_row.addWidget(self.edit_config_btn)
        config_btn_row.addWidget(self.remove_config_btn)
        config_btn_row.addWidget(self.clear_configs_btn)
        config_btn_row.addWidget(self.load_configs_btn)
        config_btn_row.addWidget(self.save_configs_btn)
        config_btn_row.addStretch(1)
        config_layout.addLayout(config_btn_row)
        
        widget.addWidget(config_group)
        
        # 测试控制
        test_control_row = QHBoxLayout()
        self.batch_run_btn = QPushButton(tr("开始列表测试"))
        self.batch_run_btn.clicked.connect(self.start_batch_test)
        self.batch_stop_btn = QPushButton(tr("停止测试"))
        self.batch_stop_btn.clicked.connect(self.stop_batch_test)
        self.batch_stop_btn.setEnabled(False)
        test_control_row.addWidget(self.batch_run_btn)
        test_control_row.addWidget(self.batch_stop_btn)
        test_control_row.addStretch(1)
        widget.addLayout(test_control_row)
        
        # 进度显示
        self.batch_progress = QProgressBar()
        self.batch_progress.setRange(0, 100)
        self.batch_progress.setValue(0)
        self.batch_progress_label = QLabel(tr("等待开始..."))
        progress_layout = QVBoxLayout()
        progress_layout.addWidget(self.batch_progress_label)
        progress_layout.addWidget(self.batch_progress)
        widget.addLayout(progress_layout)
        
        # 日志
        self.batch_log_text = QTextEdit()
        self.batch_log_text.setReadOnly(True)
        self.batch_log_text.setMinimumHeight(120)
        widget.addWidget(self.batch_log_text, 1)
        
        # 结果汇总表格
        result_summary_group = QGroupBox(tr("测试结果汇总"))
        result_summary_layout = QVBoxLayout(result_summary_group)
        
        self.result_summary_table = QTableWidget(0, 10)
        self.result_summary_table.setHorizontalHeaderLabels([
            tr("序号"),
            tr("模型"),
            tr("Precision"),
            tr("Recall"),
            tr("F1"),
            tr("TP"),
            tr("FP"),
            tr("FN"),
            tr("平均时间(s/img)"),
            tr("状态"),
        ])
        self.result_summary_table.horizontalHeader().setStretchLastSection(True)
        result_summary_layout.addWidget(self.result_summary_table, 1)
        
        widget.addWidget(result_summary_group, 2)
        
        container.setLayout(widget)
        return container

    def _append_log(self, text: str):
        self.log_text.append(text.rstrip("\n"))

    def refresh_conda_envs(self):
        self.conda_env_combo.clear()
        self._append_log(tr("正在获取Conda环境列表..."))
        try:
            # 使用subprocess获取conda环境列表
            result = subprocess.run(['conda', 'env', 'list', '--json'], 
                                   capture_output=True, text=True, check=True)
            
            # 尝试找到JSON内容的实际起始和结束位置
            stdout = result.stdout
            json_start = stdout.find('{')
            json_end = stdout.rfind('}') + 1
            
            if json_start >= 0 and json_end > json_start:
                json_content = stdout[json_start:json_end]
                env_data = json.loads(json_content)
                
                # 提取环境名称
                envs = [os.path.basename(env) for env in env_data['envs']]
                self.conda_env_combo.addItems(envs)
                
                self._append_log(tr(f"找到 {len(envs)} 个Conda环境"))
            else:
                self._append_log(tr("无法解析Conda环境列表"))
        except Exception as e:
            self._append_log(tr(f"获取Conda环境失败: {str(e)}"))

    def _browse_model(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            tr("选择模型文件"),
            "",
            tr("模型文件 (*.pt *.pth *.onnx *.engine);;所有文件 (*.*)"),
        )
        if path:
            self.model_path_edit.setText(path)

    def _browse_images_dir(self):
        path = QFileDialog.getExistingDirectory(self, tr("选择图片目录"), "")
        if path:
            self.images_dir_edit.setText(path)

    def _browse_labels_dir(self):
        path = QFileDialog.getExistingDirectory(self, tr("选择标签目录"), "")
        if path:
            self.labels_dir_edit.setText(path)

    def _validate_inputs(self) -> bool:
        model_path = self.model_path_edit.text().strip()
        images_dir = self.images_dir_edit.text().strip()
        conda_env = self.conda_env_combo.currentText().strip()

        if not conda_env:
            QMessageBox.warning(self, tr("警告"), tr("请选择Conda环境"))
            return False
        if not model_path or not os.path.exists(model_path):
            QMessageBox.warning(self, tr("警告"), tr("请先选择有效的模型文件"))
            return False
        if not images_dir or not os.path.isdir(images_dir):
            QMessageBox.warning(self, tr("警告"), tr("请选择有效的图片目录"))
            return False
        labels_dir = self.labels_dir_edit.text().strip()
        if labels_dir and not os.path.isdir(labels_dir):
            QMessageBox.warning(self, tr("警告"), tr("标签目录无效"))
            return False
        return True

    def start_single_test(self):
        """开始单个测试"""
        if not self._validate_inputs():
            return

        self.summary_text.clear()
        self.table.setRowCount(0)
        self.progress.setValue(0)
        self._result_json_path = None

        conda_env = self.conda_env_combo.currentText().strip()
        model_path = os.path.abspath(self.model_path_edit.text().strip())
        images_dir = os.path.abspath(self.images_dir_edit.text().strip())
        labels_dir = self.labels_dir_edit.text().strip()
        labels_dir = os.path.abspath(labels_dir) if labels_dir else ""
        device = self.device_combo.currentText().strip()

        self._append_log(tr("正在启动测试进程..."))
        
        try:
            # 获取测试脚本的绝对路径
            script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repo root
            script_path = os.path.join(script_dir, "training", "model_test.py")
            out_path = os.path.join(script_dir, "logs", "model_test_result.json")

            # 构建Python命令参数
            py_cmd_parts = [
                f'--model "{model_path}"',
                f'--images "{images_dir}"',
                f'--device {device}',
                f'--conf {self.conf_spin.value()}',
                f'--iou {self.iou_spin.value()}',
                f'--max-det {self.max_det_spin.value()}',
                f'--out "{out_path}"',
            ]
            if labels_dir:
                py_cmd_parts.append(f'--labels "{labels_dir}"')
            
            py_cmd = " ".join(py_cmd_parts)
            
            # 构建命令，先初始化conda，然后激活环境并运行脚本
            cmd = f'start cmd.exe /k "conda init cmd.exe && conda activate {conda_env} && python "{script_path}" {py_cmd}"'
            
            # 执行命令
            subprocess.Popen(cmd, shell=True)
            
            self._append_log(tr(f"已在新终端中启动测试进程，使用环境: {conda_env}"))
            self._append_log(tr("请在新打开的终端窗口中查看测试进度和日志"))
            self._append_log(tr("测试完成后，结果将保存到: ") + out_path)
        except Exception as e:
            self._append_log(tr(f"启动测试进程失败: {str(e)}"))
            QMessageBox.warning(self, tr("错误"), tr(f"启动测试进程失败: {str(e)}"))

    def _get_current_config(self):
        """获取当前表单的配置"""
        model_path = self.model_path_edit.text().strip()
        images_dir = self.images_dir_edit.text().strip()
        labels_dir = self.labels_dir_edit.text().strip()
        device = self.device_combo.currentText().strip()
        conf = self.conf_spin.value()
        iou = self.iou_spin.value()
        max_det = self.max_det_spin.value()
        
        return {
            'model_path': model_path,
            'images_dir': images_dir,
            'labels_dir': labels_dir,
            'device': device,
            'conf': conf,
            'iou': iou,
            'max_det': max_det,
        }

    def _add_current_config(self):
        """添加当前配置到列表"""
        if not self._validate_inputs():
            return
        
        config = self._get_current_config()
        self._test_configs.append(config)
        self._update_config_table()

    def _edit_selected_config(self):
        """编辑选中的配置"""
        selected = self.config_table.selectedItems()
        if not selected:
            QMessageBox.warning(self, tr("警告"), tr("请先选择一个配置"))
            return
        
        row = selected[0].row()
        if row < 0 or row >= len(self._test_configs):
            return
        
        config = self._test_configs[row]
        
        # 将配置填入表单
        self.model_path_edit.setText(config.get('model_path', ''))
        self.images_dir_edit.setText(config.get('images_dir', ''))
        self.labels_dir_edit.setText(config.get('labels_dir', ''))
        self.device_combo.setCurrentText(config.get('device', 'cpu'))
        self.conf_spin.setValue(config.get('conf', 0.25))
        self.iou_spin.setValue(config.get('iou', 0.5))
        self.max_det_spin.setValue(config.get('max_det', 300))
        
        # 删除旧配置，用户可以通过"添加当前配置"重新添加
        self._test_configs.pop(row)
        self._update_config_table()
        
        QMessageBox.information(self, tr("提示"), tr("配置已加载到表单，修改后请点击\"添加当前配置\"重新添加"))

    def _remove_selected_config(self):
        """删除选中的配置"""
        selected = self.config_table.selectedItems()
        if not selected:
            QMessageBox.warning(self, tr("警告"), tr("请先选择一个配置"))
            return
        
        row = selected[0].row()
        if row < 0 or row >= len(self._test_configs):
            return
        
        self._test_configs.pop(row)
        self._update_config_table()

    def _clear_configs(self):
        """清空配置列表"""
        reply = QMessageBox.question(
            self, tr("确认"), tr("确定要清空所有配置吗？"),
            QMessageBox.Yes | QMessageBox.No
        )
        if reply == QMessageBox.Yes:
            self._test_configs.clear()
            self._update_config_table()

    def _update_config_table(self):
        """更新配置列表表格"""
        self.config_table.setRowCount(len(self._test_configs))
        for r, config in enumerate(self._test_configs):
            values = [
                os.path.basename(config.get('model_path', '')),
                os.path.basename(config.get('images_dir', '')),
                os.path.basename(config.get('labels_dir', '')) if config.get('labels_dir') else tr("(无)"),
                config.get('device', 'cpu'),
                str(config.get('conf', 0.25)),
                str(config.get('iou', 0.5)),
                str(config.get('max_det', 300)),
            ]
            for c, v in enumerate(values):
                item = QTableWidgetItem(v)
                if c >= 3:  # device, conf, iou, max_det 居中
                    item.setTextAlignment(Qt.AlignCenter)
                self.config_table.setItem(r, c, item)

    def _save_configs(self):
        """保存配置列表到文件"""
        if not self._test_configs:
            QMessageBox.warning(self, tr("警告"), tr("配置列表为空"))
            return
        
        path, _ = QFileDialog.getSaveFileName(
            self,
            tr("保存配置列表"),
            "",
            tr("JSON文件 (*.json);;所有文件 (*.*)")
        )
        if path:
            try:
                with open(path, 'w', encoding='utf-8') as f:
                    json.dump(self._test_configs, f, ensure_ascii=False, indent=2)
                QMessageBox.information(self, tr("成功"), tr(f"配置列表已保存到: {path}"))
            except Exception as e:
                QMessageBox.warning(self, tr("错误"), tr(f"保存失败: {str(e)}"))

    def _load_configs(self):
        """从文件加载配置列表"""
        path, _ = QFileDialog.getOpenFileName(
            self,
            tr("加载配置列表"),
            "",
            tr("JSON文件 (*.json);;所有文件 (*.*)")
        )
        if path:
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    configs = json.load(f)
                if isinstance(configs, list):
                    self._test_configs = configs
                    self._update_config_table()
                    QMessageBox.information(self, tr("成功"), tr(f"已加载 {len(configs)} 个配置"))
                else:
                    QMessageBox.warning(self, tr("错误"), tr("配置文件格式错误"))
            except Exception as e:
                QMessageBox.warning(self, tr("错误"), tr(f"加载失败: {str(e)}"))

    def start_batch_test(self):
        """开始列表测试"""
        if not self._test_configs:
            QMessageBox.warning(self, tr("警告"), tr("请先添加测试配置"))
            return
        
        conda_env = self.conda_env_combo.currentText().strip()
        if not conda_env:
            QMessageBox.warning(self, tr("警告"), tr("请选择Conda环境"))
            return
        
        if self._is_testing:
            QMessageBox.warning(self, tr("警告"), tr("测试正在进行中"))
            return
        
        # 验证所有配置
        for i, config in enumerate(self._test_configs):
            if not config.get('model_path') or not os.path.exists(config['model_path']):
                QMessageBox.warning(self, tr("警告"), tr(f"配置 {i+1} 的模型文件无效"))
                return
            if not config.get('images_dir') or not os.path.isdir(config['images_dir']):
                QMessageBox.warning(self, tr("警告"), tr(f"配置 {i+1} 的图片目录无效"))
                return
        
        # 初始化
        self._test_results = [None] * len(self._test_configs)  # 预分配结果列表
        self.batch_log_text.clear()
        self.result_summary_table.setRowCount(len(self._test_configs))  # 预分配表格行
        self._update_result_summary_table()  # 初始化表格显示
        self._is_testing = True
        self.batch_run_btn.setEnabled(False)
        self.batch_stop_btn.setEnabled(True)
        
        # 获取脚本路径
        script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        script_path = os.path.join(script_dir, "training", "model_test.py")
        logs_dir = os.path.join(script_dir, "logs")
        os.makedirs(logs_dir, exist_ok=True)
        
        # 创建并启动测试线程
        self._test_thread = TestConfigThread(
            self._test_configs,
            conda_env,
            script_path,
            logs_dir
        )
        self._test_thread.progress_updated.connect(self._on_batch_progress_updated)
        self._test_thread.config_started.connect(self._on_config_started)
        self._test_thread.config_completed.connect(self._on_config_completed)
        self._test_thread.log_message.connect(self._on_batch_log_message)
        self._test_thread.all_completed.connect(self._on_all_completed)
        self._test_thread.start()
        
        self._append_batch_log(tr(f"开始执行 {len(self._test_configs)} 个测试配置..."))

    def stop_batch_test(self):
        """停止列表测试"""
        if self._test_thread and self._test_thread.isRunning():
            self._test_thread.stop()
            self._append_batch_log(tr("正在停止测试..."))
            self.batch_stop_btn.setEnabled(False)

    def _on_batch_progress_updated(self, current, total):
        """更新批量测试进度"""
        progress = int((current / total) * 100) if total > 0 else 0
        self.batch_progress.setValue(progress)
        self.batch_progress_label.setText(tr(f"进度: {current+1}/{total}"))

    def _on_config_started(self, index, config):
        """配置测试开始"""
        model_name = os.path.basename(config.get('model_path', ''))
        self._append_batch_log(tr(f"[{index+1}/{len(self._test_configs)}] 开始测试: {model_name}"))

    def _on_config_completed(self, index, config, result):
        """配置测试完成"""
        # 确保结果列表长度与配置列表一致
        while len(self._test_results) <= index:
            self._test_results.append(None)
        
        self._test_results[index] = result
        self._update_result_summary_table()
        
        if result:
            model_name = os.path.basename(config.get('model_path', ''))
            precision = result.get('precision', 0)
            recall = result.get('recall', 0)
            f1 = result.get('f1', 0)
            self._append_batch_log(tr(f"[{index+1}/{len(self._test_configs)}] 完成: {model_name} - P:{precision:.4f} R:{recall:.4f} F1:{f1:.4f}"))
        else:
            model_name = os.path.basename(config.get('model_path', ''))
            self._append_batch_log(tr(f"[{index+1}/{len(self._test_configs)}] 失败: {model_name}"))

    def _on_batch_log_message(self, message):
        """批量测试日志消息"""
        self._append_batch_log(message)

    def _on_all_completed(self, results):
        """所有测试完成"""
        self._is_testing = False
        self.batch_run_btn.setEnabled(True)
        self.batch_stop_btn.setEnabled(False)
        self._append_batch_log(tr(f"所有测试完成！共 {len(results)} 个配置，成功 {sum(1 for r in results if r)} 个"))
        self.batch_progress_label.setText(tr("测试完成"))

    def _append_batch_log(self, text: str):
        """添加批量测试日志"""
        self.batch_log_text.append(text.rstrip("\n"))

    def _update_result_summary_table(self):
        """更新结果汇总表格"""
        self.result_summary_table.setRowCount(len(self._test_configs))
        for r in range(len(self._test_configs)):
            config = self._test_configs[r]
            result = self._test_results[r] if r < len(self._test_results) else None
            model_name = os.path.basename(config.get('model_path', ''))
            
            if result:
                values = [
                    str(r + 1),
                    model_name,
                    f"{result.get('precision', 0):.4f}" if result.get('precision') is not None else "N/A",
                    f"{result.get('recall', 0):.4f}" if result.get('recall') is not None else "N/A",
                    f"{result.get('f1', 0):.4f}" if result.get('f1') is not None else "N/A",
                    str(result.get('tp', 0)),
                    str(result.get('fp', 0)),
                    str(result.get('fn', 0)),
                    f"{result.get('avg_time_s_per_image', 0):.4f}",
                    tr("成功"),
                ]
            else:
                status = tr("失败") if r < len(self._test_results) else tr("等待中")
                values = [
                    str(r + 1),
                    model_name,
                    "N/A",
                    "N/A",
                    "N/A",
                    "N/A",
                    "N/A",
                    "N/A",
                    "N/A",
                    status,
                ]
            
            for c, v in enumerate(values):
                item = QTableWidgetItem(v)
                if c > 0:  # 除序号外都居中
                    item.setTextAlignment(Qt.AlignCenter)
                self.result_summary_table.setItem(r, c, item)

    def _render_result(self, result: dict):
        # summary
        def fmt(x):
            if x is None:
                return "N/A"
            if isinstance(x, float):
                return f"{x:.4f}"
            return str(x)

        lines = []
        lines.append(f"backend: {result.get('backend')}")
        lines.append(f"model: {result.get('model')}")
        lines.append(f"images_dir: {result.get('images_dir')}")
        lines.append(f"labels_dir: {result.get('labels_dir')}")
        lines.append("")
        lines.append(f"total_images: {fmt(result.get('total_images'))}")
        lines.append(f"images_with_labels: {fmt(result.get('images_with_labels'))}")
        lines.append(f"total_pred: {fmt(result.get('total_pred'))}")
        lines.append(f"total_gt: {fmt(result.get('total_gt'))}")
        lines.append(f"avg_time_s_per_image: {fmt(result.get('avg_time_s_per_image'))}")
        lines.append("")
        lines.append(f"TP: {fmt(result.get('tp'))}  FP: {fmt(result.get('fp'))}  FN: {fmt(result.get('fn'))}")
        lines.append(f"Precision: {fmt(result.get('precision'))}  Recall: {fmt(result.get('recall'))}  F1: {fmt(result.get('f1'))}")
        self.summary_text.setPlainText("\n".join(lines))

        # table
        per_class = result.get("per_class") or {}
        keys = sorted(per_class.keys(), key=lambda k: int(k) if str(k).isdigit() else str(k))
        self.table.setRowCount(len(keys))
        for r, k in enumerate(keys):
            row = per_class.get(k, {})
            values = [
                k,
                str(row.get("gt", 0)),
                str(row.get("pred", 0)),
                str(row.get("tp", 0)),
                str(row.get("fp", 0)),
                str(row.get("fn", 0)),
                fmt(row.get("precision")),
                fmt(row.get("recall")),
            ]
            for c, v in enumerate(values):
                item = QTableWidgetItem(v)
                if c != 0:
                    item.setTextAlignment(Qt.AlignCenter)
                self.table.setItem(r, c, item)


