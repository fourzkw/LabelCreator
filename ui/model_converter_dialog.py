import os
import json
import subprocess
from PyQt5.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QGroupBox, 
                             QLabel, QLineEdit, QPushButton, QFileDialog,
                             QSpinBox, QCheckBox, QMessageBox, QFormLayout,
                             QComboBox, QApplication, QProgressDialog)
from PyQt5.QtCore import Qt, QThread, pyqtSignal

from utils.model_converter import ModelConverter
from i18n import tr
import logging

logger = logging.getLogger('YOLOLabelCreator.ModelConverterDialog')

class ConversionThread(QThread):
    """Thread for running model conversion in background"""
    conversion_complete = pyqtSignal(bool, str)
    
    def __init__(self, format_type, input_path, output_path, img_size, **kwargs):
        super().__init__()
        self.format_type = format_type
        self.input_path = input_path
        self.output_path = output_path
        self.img_size = img_size
        self.kwargs = kwargs
    
    def run(self):
        """Run conversion process in background"""
        if self.format_type == 'onnx':
            success, message = ModelConverter.pt_to_onnx(
                self.input_path,
                self.output_path,
                self.img_size,
                self.kwargs.get('simplify', True),
                self.kwargs.get('opset', 12),
                self.kwargs.get('half', False)
            )
        elif self.format_type == 'tensorrt':
            success, message = ModelConverter.pt_to_tensorrt(
                self.input_path,
                self.output_path,
                self.img_size,
                self.kwargs.get('half', False),
                self.kwargs.get('int8', False),
                self.kwargs.get('workspace', 4),
                self.kwargs.get('device', 0)
            )
        else:
            success, message = False, f"Unsupported format: {self.format_type}"
        
        self.conversion_complete.emit(success, message)

class ModelConverterDialog(QDialog):
    """Dialog for converting PyTorch models to ONNX or TensorRT format"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle(tr("PT Model Converter"))
        self.setMinimumWidth(500)
        self.setup_ui()
    
    def setup_ui(self):
        """Set up the dialog UI"""
        layout = QVBoxLayout(self)
        
        # Output format selection
        format_group = QGroupBox(tr("Output Format"))
        format_layout = QHBoxLayout()
        
        format_layout.addWidget(QLabel(tr("Format:")))
        self.format_combo = QComboBox()
        self.format_combo.addItems([tr("ONNX"), tr("TensorRT")])
        self.format_combo.currentIndexChanged.connect(self.on_format_changed)
        format_layout.addWidget(self.format_combo)
        format_layout.addStretch()
        
        format_group.setLayout(format_layout)
        layout.addWidget(format_group)
        
        # Input model selection
        input_group = QGroupBox(tr("Input Model (PyTorch)"))
        input_layout = QHBoxLayout()
        
        self.input_path_edit = QLineEdit()
        self.input_path_edit.setReadOnly(True)
        self.input_browse_btn = QPushButton(tr("Browse..."))
        self.input_browse_btn.clicked.connect(self.browse_input_model)
        
        input_layout.addWidget(self.input_path_edit)
        input_layout.addWidget(self.input_browse_btn)
        input_group.setLayout(input_layout)
        layout.addWidget(input_group)
        
        # Output model selection
        self.output_group = QGroupBox(tr("Output Model (ONNX)"))
        output_layout = QHBoxLayout()
        
        self.output_path_edit = QLineEdit()
        self.output_path_edit.setReadOnly(True)
        self.output_browse_btn = QPushButton(tr("Browse..."))
        self.output_browse_btn.clicked.connect(self.browse_output_model)
        
        output_layout.addWidget(self.output_path_edit)
        output_layout.addWidget(self.output_browse_btn)
        self.output_group.setLayout(output_layout)
        layout.addWidget(self.output_group)
        
        # Conversion parameters
        params_group = QGroupBox(tr("Conversion Parameters"))
        params_layout = QFormLayout()
        
        # Image size
        size_layout = QHBoxLayout()
        self.width_spinbox = QSpinBox()
        self.width_spinbox.setRange(32, 1280)
        self.width_spinbox.setValue(640)
        self.width_spinbox.setSingleStep(32)
        
        self.height_spinbox = QSpinBox()
        self.height_spinbox.setRange(32, 1280)
        self.height_spinbox.setValue(640)
        self.height_spinbox.setSingleStep(32)
        
        size_layout.addWidget(QLabel(tr("Width:")))
        size_layout.addWidget(self.width_spinbox)
        size_layout.addWidget(QLabel(tr("Height:")))
        size_layout.addWidget(self.height_spinbox)
        params_layout.addRow(tr("Image Size:"), size_layout)
        
        # ONNX-specific parameters
        # ONNX opset
        opset_label = QLabel(tr("ONNX Opset:"))
        self.opset_combo = QComboBox()
        self.opset_combo.addItems(["12", "13", "14", "15", "16", "17"])
        self.opset_combo.setCurrentIndex(0)  # Default to 12
        params_layout.addRow(opset_label, self.opset_combo)
        self.opset_label = opset_label
        
        # Simplify checkbox
        self.simplify_checkbox = QCheckBox(tr("Simplify Model"))
        self.simplify_checkbox.setChecked(True)
        params_layout.addRow("", self.simplify_checkbox)
        
        # TensorRT-specific parameters
        # Workspace size
        workspace_label = QLabel(tr("Workspace Size:"))
        self.workspace_spinbox = QSpinBox()
        self.workspace_spinbox.setRange(1, 32)
        self.workspace_spinbox.setValue(4)
        self.workspace_spinbox.setSuffix(tr(" GB"))
        params_layout.addRow(workspace_label, self.workspace_spinbox)
        workspace_label.setVisible(False)
        self.workspace_spinbox.setVisible(False)
        
        # Device ID
        device_label = QLabel(tr("CUDA Device ID:"))
        self.device_spinbox = QSpinBox()
        self.device_spinbox.setRange(0, 7)
        self.device_spinbox.setValue(0)
        params_layout.addRow(device_label, self.device_spinbox)
        device_label.setVisible(False)
        self.device_spinbox.setVisible(False)
        
        # Store labels for later use
        self.workspace_label = workspace_label
        self.device_label = device_label
        
        # Common precision options
        self.half_checkbox = QCheckBox(tr("Half Precision (FP16)"))
        self.half_checkbox.setChecked(False)
        self.half_checkbox.stateChanged.connect(self.on_half_changed)
        params_layout.addRow("", self.half_checkbox)
        
        # INT8 checkbox (only for TensorRT)
        self.int8_checkbox = QCheckBox(tr("INT8 Quantization"))
        self.int8_checkbox.setChecked(False)
        self.int8_checkbox.setVisible(False)
        self.int8_checkbox.stateChanged.connect(self.on_int8_changed)
        params_layout.addRow("", self.int8_checkbox)
        
        params_group.setLayout(params_layout)
        layout.addWidget(params_group)
        
        # Conda environment selection (optional)
        conda_group = QGroupBox(tr("Conda Environment (Optional)"))
        conda_layout = QFormLayout()
        
        self.use_conda_checkbox = QCheckBox(tr("Use Conda Environment"))
        self.use_conda_checkbox.setChecked(False)
        self.use_conda_checkbox.stateChanged.connect(self.on_use_conda_changed)
        conda_layout.addRow("", self.use_conda_checkbox)
        
        conda_env_layout = QHBoxLayout()
        self.conda_env_combo = QComboBox()
        self.conda_env_combo.setEnabled(False)
        self.refresh_conda_btn = QPushButton(tr("Refresh"))
        self.refresh_conda_btn.setEnabled(False)
        self.refresh_conda_btn.clicked.connect(self.refresh_conda_envs)
        
        conda_env_layout.addWidget(self.conda_env_combo)
        conda_env_layout.addWidget(self.refresh_conda_btn)
        conda_layout.addRow(tr("Conda Environment:"), conda_env_layout)
        
        conda_group.setLayout(conda_layout)
        layout.addWidget(conda_group)
        
        # Buttons
        btn_layout = QHBoxLayout()
        self.convert_btn = QPushButton(tr("Convert"))
        self.convert_btn.clicked.connect(self.start_conversion)
        self.convert_btn.setEnabled(False)  # Disabled until input model is selected
        
        self.cancel_btn = QPushButton(tr("Cancel"))
        self.cancel_btn.clicked.connect(self.reject)
        
        btn_layout.addStretch()
        btn_layout.addWidget(self.cancel_btn)
        btn_layout.addWidget(self.convert_btn)
        layout.addLayout(btn_layout)
    
    def on_format_changed(self, index):
        """Handle format selection change"""
        is_tensorrt = (index == 1)  # TensorRT is index 1
        
        # Update output group title
        if is_tensorrt:
            self.output_group.setTitle(tr("Output Model (TensorRT)"))
        else:
            self.output_group.setTitle(tr("Output Model (ONNX)"))
        
        # Show/hide format-specific parameters
        self.opset_label.setVisible(not is_tensorrt)
        self.opset_combo.setVisible(not is_tensorrt)
        self.simplify_checkbox.setVisible(not is_tensorrt)
        self.workspace_label.setVisible(is_tensorrt)
        self.workspace_spinbox.setVisible(is_tensorrt)
        self.device_label.setVisible(is_tensorrt)
        self.device_spinbox.setVisible(is_tensorrt)
        self.int8_checkbox.setVisible(is_tensorrt)
        
        # Update output file extension if path exists
        current_path = self.output_path_edit.text()
        if current_path:
            base_path = os.path.splitext(current_path)[0]
            new_ext = '.engine' if is_tensorrt else '.onnx'
            self.output_path_edit.setText(base_path + new_ext)
    
    def on_half_changed(self, state):
        """Handle FP16 checkbox change - disable INT8 if FP16 is enabled"""
        if state == Qt.Checked:
            self.int8_checkbox.setChecked(False)
    
    def on_int8_changed(self, state):
        """Handle INT8 checkbox change - disable FP16 if INT8 is enabled"""
        if state == Qt.Checked:
            self.half_checkbox.setChecked(False)
    
    def browse_input_model(self):
        """Browse for input PyTorch model file"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, 
            tr("Select PyTorch Model"), 
            "", 
            tr("PyTorch Models (*.pt *.pth);;All Files (*.*)")
        )
        if file_path:
            self.input_path_edit.setText(file_path)
            
            # Auto-generate output path if it's empty
            if not self.output_path_edit.text():
                is_tensorrt = (self.format_combo.currentIndex() == 1)
                ext = '.engine' if is_tensorrt else '.onnx'
                output_path = os.path.splitext(file_path)[0] + ext
                self.output_path_edit.setText(output_path)
            
            # Enable convert button if both paths are set
            self.update_convert_button()
    
    def browse_output_model(self):
        """Browse for output model location"""
        is_tensorrt = (self.format_combo.currentIndex() == 1)
        
        if is_tensorrt:
            file_filter = tr("TensorRT Models (*.engine);;All Files (*.*)")
            dialog_title = tr("Save TensorRT Model As")
            default_ext = '.engine'
        else:
            file_filter = tr("ONNX Models (*.onnx);;All Files (*.*)")
            dialog_title = tr("Save ONNX Model As")
            default_ext = '.onnx'
        
        file_path, _ = QFileDialog.getSaveFileName(
            self, 
            dialog_title, 
            self.output_path_edit.text() or "", 
            file_filter
        )
        if file_path:
            # Add extension if not present
            if not file_path.lower().endswith(default_ext):
                file_path += default_ext
            self.output_path_edit.setText(file_path)
            
            # Enable convert button if both paths are set
            self.update_convert_button()
    
    def update_convert_button(self):
        """Update the state of the convert button"""
        self.convert_btn.setEnabled(
            bool(self.input_path_edit.text()) and bool(self.output_path_edit.text())
        )
    
    def on_use_conda_changed(self, state):
        """Handle use conda checkbox change"""
        enabled = (state == Qt.Checked)
        self.conda_env_combo.setEnabled(enabled)
        self.refresh_conda_btn.setEnabled(enabled)
        if enabled and self.conda_env_combo.count() == 0:
            self.refresh_conda_envs()
    
    def refresh_conda_envs(self):
        """Refresh conda environment list"""
        try:
            self.conda_env_combo.clear()
            result = subprocess.run(
                ['conda', 'env', 'list', '--json'],
                capture_output=True,
                text=True,
                check=True
            )
            
            stdout = result.stdout
            json_start = stdout.find('{')
            json_end = stdout.rfind('}') + 1
            
            if json_start >= 0 and json_end > json_start:
                json_content = stdout[json_start:json_end]
                env_data = json.loads(json_content)
                envs = [os.path.basename(env) for env in env_data['envs']]
                self.conda_env_combo.addItems(envs)
                logger.info(f"Found {len(envs)} conda environments")
            else:
                logger.warning("Failed to parse conda environment list")
        except Exception as e:
            logger.error(f"Failed to get conda environments: {str(e)}")
            QMessageBox.warning(
                self,
                tr("Warning"),
                tr(f"Failed to get conda environments:\n{str(e)}\n\n"
                   f"Please ensure Conda is installed and available in PATH.")
            )
    
    def start_conversion(self):
        """Start the model conversion process"""
        input_path = self.input_path_edit.text()
        output_path = self.output_path_edit.text()
        
        # Check if using conda environment
        use_conda = self.use_conda_checkbox.isChecked()
        if use_conda:
            conda_env = self.conda_env_combo.currentText()
            if not conda_env:
                QMessageBox.warning(
                    self,
                    tr("Warning"),
                    tr("Please select a Conda environment or disable Conda option.")
                )
                return
            
            # Use conda environment to run conversion
            self.start_conversion_with_conda(input_path, output_path, conda_env)
        else:
            # Use direct conversion (current process)
            self.start_conversion_direct(input_path, output_path)
    
    def start_conversion_direct(self, input_path, output_path):
        """Start conversion in current process"""
        # Get format type
        format_type = 'tensorrt' if self.format_combo.currentIndex() == 1 else 'onnx'
        
        # Get common conversion parameters
        img_size = (self.width_spinbox.value(), self.height_spinbox.value())
        half = self.half_checkbox.isChecked()
        
        # Get format-specific parameters
        kwargs = {}
        if format_type == 'onnx':
            kwargs['simplify'] = self.simplify_checkbox.isChecked()
            kwargs['opset'] = int(self.opset_combo.currentText())
            kwargs['half'] = half
        else:  # tensorrt
            kwargs['half'] = half
            kwargs['int8'] = self.int8_checkbox.isChecked()
            kwargs['workspace'] = self.workspace_spinbox.value()
            kwargs['device'] = self.device_spinbox.value()
        
        # Create progress dialog
        self.progress = QProgressDialog(
            tr("Converting model..."), 
            tr("Cancel"), 
            0, 
            0, 
            self
        )
        self.progress.setWindowTitle(tr("Model Conversion"))
        self.progress.setWindowModality(Qt.WindowModal)
        self.progress.setMinimumDuration(0)
        self.progress.setValue(0)
        self.progress.show()
        QApplication.processEvents()
        
        # Create and start conversion thread
        self.conversion_thread = ConversionThread(
            format_type, input_path, output_path, img_size, **kwargs
        )
        self.conversion_thread.conversion_complete.connect(self.on_conversion_complete)
        self.conversion_thread.start()
    
    def start_conversion_with_conda(self, input_path, output_path, conda_env):
        """Start conversion using conda environment"""
        try:
            # Get format type
            format_type = 'tensorrt' if self.format_combo.currentIndex() == 1 else 'onnx'
            
            # Get conversion parameters
            img_size = [self.width_spinbox.value(), self.height_spinbox.value()]
            half = self.half_checkbox.isChecked()
            
            # Prepare settings dictionary
            settings = {
                'format_type': format_type,
                'input_path': os.path.abspath(input_path),
                'output_path': os.path.abspath(output_path),
                'img_size': img_size,
            }
            
            if format_type == 'onnx':
                settings['simplify'] = self.simplify_checkbox.isChecked()
                settings['opset'] = int(self.opset_combo.currentText())
                settings['half'] = half
            else:  # tensorrt
                settings['half'] = half
                settings['int8'] = self.int8_checkbox.isChecked()
                settings['workspace'] = self.workspace_spinbox.value()
                settings['device'] = self.device_spinbox.value()
            
            # Create temporary settings file
            script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            temp_dir = os.path.join(script_dir, 'logs')
            os.makedirs(temp_dir, exist_ok=True)
            
            settings_file = os.path.join(temp_dir, 'conversion_settings.json')
            with open(settings_file, 'w', encoding='utf-8') as f:
                json.dump(settings, f, indent=2, ensure_ascii=False)
            
            # Get conversion script path
            script_path = os.path.join(script_dir, 'utils', 'model_converter', 'convert_model.py')
            
            # Build command
            cmd = f'start cmd.exe /k "conda init cmd.exe && conda activate {conda_env} && python "{script_path}" --settings "{settings_file}" && pause"'
            
            # Execute command
            subprocess.Popen(cmd, shell=True)
            
            QMessageBox.information(
                self,
                tr("Conversion Started"),
                tr(f"Conversion started in new terminal window using Conda environment: {conda_env}\n\n"
                   f"Please check the terminal window for progress and results.")
            )
            self.accept()
            
        except Exception as e:
            logger.error(f"Failed to start conversion with conda: {str(e)}")
            QMessageBox.critical(
                self,
                tr("Error"),
                tr(f"Failed to start conversion:\n{str(e)}")
            )
    
    def on_conversion_complete(self, success, message):
        """Handle conversion completion"""
        self.progress.cancel()
        
        if success:
            format_name = tr("TensorRT") if self.format_combo.currentIndex() == 1 else tr("ONNX")
            QMessageBox.information(
                self,
                tr("Conversion Complete"),
                tr("Model successfully converted to {0} format at:\n{1}").format(format_name, message)
            )
            self.accept()
        else:
            QMessageBox.critical(
                self,
                tr("Conversion Error"),
                tr("Error converting model:\n{0}").format(message)
            ) 