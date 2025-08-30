import os
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
    
    def __init__(self, input_path, output_path, img_size, simplify, opset, half):
        super().__init__()
        self.input_path = input_path
        self.output_path = output_path
        self.img_size = img_size
        self.simplify = simplify
        self.opset = opset
        self.half = half
    
    def run(self):
        """Run conversion process in background"""
        success, message = ModelConverter.pt_to_onnx(
            self.input_path,
            self.output_path,
            self.img_size,
            self.simplify,
            self.opset,
            self.half
        )
        self.conversion_complete.emit(success, message)

class ModelConverterDialog(QDialog):
    """Dialog for converting PyTorch models to ONNX format"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle(tr("PT to ONNX Model Converter"))
        self.setMinimumWidth(500)
        self.setup_ui()
    
    def setup_ui(self):
        """Set up the dialog UI"""
        layout = QVBoxLayout(self)
        
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
        output_group = QGroupBox(tr("Output Model (ONNX)"))
        output_layout = QHBoxLayout()
        
        self.output_path_edit = QLineEdit()
        self.output_path_edit.setReadOnly(True)
        self.output_browse_btn = QPushButton(tr("Browse..."))
        self.output_browse_btn.clicked.connect(self.browse_output_model)
        
        output_layout.addWidget(self.output_path_edit)
        output_layout.addWidget(self.output_browse_btn)
        output_group.setLayout(output_layout)
        layout.addWidget(output_group)
        
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
        
        # ONNX opset
        self.opset_combo = QComboBox()
        self.opset_combo.addItems(["12", "13", "14", "15", "16", "17"])
        self.opset_combo.setCurrentIndex(0)  # Default to 12
        params_layout.addRow(tr("ONNX Opset:"), self.opset_combo)
        
        # Simplify checkbox
        self.simplify_checkbox = QCheckBox(tr("Simplify Model"))
        self.simplify_checkbox.setChecked(True)
        params_layout.addRow("", self.simplify_checkbox)
        
        # Half precision checkbox
        self.half_checkbox = QCheckBox(tr("Half Precision (FP16)"))
        self.half_checkbox.setChecked(False)
        params_layout.addRow("", self.half_checkbox)
        
        params_group.setLayout(params_layout)
        layout.addWidget(params_group)
        
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
                output_path = os.path.splitext(file_path)[0] + '.onnx'
                self.output_path_edit.setText(output_path)
            
            # Enable convert button if both paths are set
            self.update_convert_button()
    
    def browse_output_model(self):
        """Browse for output ONNX model location"""
        file_path, _ = QFileDialog.getSaveFileName(
            self, 
            tr("Save ONNX Model As"), 
            self.output_path_edit.text() or "", 
            tr("ONNX Models (*.onnx);;All Files (*.*)")
        )
        if file_path:
            # Add .onnx extension if not present
            if not file_path.lower().endswith('.onnx'):
                file_path += '.onnx'
            self.output_path_edit.setText(file_path)
            
            # Enable convert button if both paths are set
            self.update_convert_button()
    
    def update_convert_button(self):
        """Update the state of the convert button"""
        self.convert_btn.setEnabled(
            bool(self.input_path_edit.text()) and bool(self.output_path_edit.text())
        )
    
    def start_conversion(self):
        """Start the model conversion process"""
        input_path = self.input_path_edit.text()
        output_path = self.output_path_edit.text()
        
        # Get conversion parameters
        img_size = (self.width_spinbox.value(), self.height_spinbox.value())
        simplify = self.simplify_checkbox.isChecked()
        opset = int(self.opset_combo.currentText())
        half = self.half_checkbox.isChecked()
        
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
            input_path, output_path, img_size, simplify, opset, half
        )
        self.conversion_thread.conversion_complete.connect(self.on_conversion_complete)
        self.conversion_thread.start()
    
    def on_conversion_complete(self, success, message):
        """Handle conversion completion"""
        self.progress.cancel()
        
        if success:
            QMessageBox.information(
                self,
                tr("Conversion Complete"),
                tr(f"Model successfully converted to ONNX format at:\n{message}")
            )
            self.accept()
        else:
            QMessageBox.critical(
                self,
                tr("Conversion Error"),
                tr(f"Error converting model:\n{message}")
            ) 