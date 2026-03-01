"""
TensorRT 模型转换器

依赖要求：
- NVIDIA GPU（支持 CUDA）
- CUDA Toolkit (11.0+，推荐 11.8+)
- TensorRT 库 (8.0+，推荐 8.5+)
- PyTorch（CUDA 版本）
- ultralytics>=8.4.7

详细安装说明请参考：TENSORRT_REQUIREMENTS.md
"""

import os
import logging
import traceback
import subprocess
import platform
import json
import sys
from datetime import datetime, timezone
import torch
from ultralytics import YOLO
from .base import BaseConverter

logger = logging.getLogger('YOLOLabelCreator.ModelConverter.TensorRT')


class TensorRTConverter(BaseConverter):
    """TensorRT 模型转换器"""

    @staticmethod
    def _safe_run(cmd, timeout=5):
        """Run a command and return stdout (str) or None on failure."""
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout,
                shell=False
            )
            if result.returncode == 0:
                out = (result.stdout or "").strip()
                return out if out else None
        except Exception:
            pass
        return None

    @staticmethod
    def _get_tensorrt_version():
        """Best-effort TensorRT version detection (string or None)."""
        # Ensure PATH contains TensorRT bin so import can find dependent DLLs
        TensorRTConverter._add_tensorrt_to_path()
        try:
            import tensorrt as trt  # type: ignore
            ver = getattr(trt, "__version__", None)
            if ver:
                return str(ver)
        except Exception:
            pass
        # Fallback: try to infer from DLL location (Windows)
        if platform.system() == "Windows":
            dll = TensorRTConverter._safe_run(["where", "nvinfer_10.dll"], timeout=5)
            if dll:
                return f"unknown (dll={dll.splitlines()[0]})"
        return None

    @staticmethod
    def _collect_env_info():
        """Collect environment info for TensorRT export (JSON-serializable dict)."""
        info = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "python": {
                "version": sys.version.split()[0],
                "executable": sys.executable,
            },
            "os": {
                "platform": platform.platform(),
                "system": platform.system(),
                "release": platform.release(),
                "version": platform.version(),
                "machine": platform.machine(),
            },
            "tensorrt": {
                "version": TensorRTConverter._get_tensorrt_version(),
                "tensorrt_bin": TensorRTConverter._find_tensorrt_bin_path(),
                "TENSORRT_HOME": os.environ.get("TENSORRT_HOME"),
            },
            "torch": {
                "version": getattr(torch, "__version__", None),
                "cuda_available": bool(torch.cuda.is_available()),
                "cuda_version": getattr(getattr(torch, "version", None), "cuda", None),
                "cudnn_version": torch.backends.cudnn.version() if hasattr(torch.backends, "cudnn") else None,
            },
            "ultralytics": {},
            "gpu": {},
            "nvidia_smi": {},
        }

        # ultralytics version
        try:
            import ultralytics  # type: ignore
            info["ultralytics"]["version"] = getattr(ultralytics, "__version__", None)
        except Exception as e:
            info["ultralytics"]["version_error"] = str(e)

        # GPU info via torch
        if torch.cuda.is_available():
            try:
                dev = torch.cuda.current_device()
                info["gpu"] = {
                    "device_id": int(dev),
                    "name": torch.cuda.get_device_name(dev),
                    "capability": ".".join(map(str, torch.cuda.get_device_capability(dev))),
                    "total_memory_gb": round(torch.cuda.get_device_properties(dev).total_memory / (1024 ** 3), 3),
                }
            except Exception as e:
                info["gpu"]["error"] = str(e)

        # NVIDIA driver info via nvidia-smi (best effort; may not exist in PATH)
        smi = TensorRTConverter._safe_run(
            ["nvidia-smi", "--query-gpu=driver_version,name", "--format=csv,noheader"],
            timeout=5
        )
        if smi:
            # If multiple GPUs, keep all lines
            info["nvidia_smi"]["query_gpu_driver_version_name"] = smi.splitlines()

        return info

    @staticmethod
    def _get_metadata_path(output_path: str) -> str:
        base, _ext = os.path.splitext(output_path)
        return base + ".engine.metadata.json"

    @staticmethod
    def _write_metadata(output_path: str, conversion_params: dict):
        """Write metadata JSON next to the engine. Never raises."""
        try:
            metadata = {
                "format": "tensorrt_engine",
                "engine_path": os.path.abspath(output_path),
                "conversion_params": conversion_params or {},
                "environment": TensorRTConverter._collect_env_info(),
            }
            meta_path = TensorRTConverter._get_metadata_path(output_path)
            os.makedirs(os.path.dirname(os.path.abspath(meta_path)), exist_ok=True)
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)
            logger.info(f"Wrote TensorRT export metadata: {meta_path}")
        except Exception as e:
            logger.warning(f"Failed to write TensorRT export metadata: {e}")
    
    @staticmethod
    def _find_tensorrt_bin_path():
        """
        查找TensorRT的bin目录路径
        
        Returns:
            str or None: TensorRT bin目录路径，如果找不到则返回None
        """
        # 方法1: 检查常见的TensorRT安装路径
        common_paths = [
            r"E:\Program Files\TensorRT-10.14.1.48\bin",
            r"C:\Program Files\TensorRT-10.14.1.48\bin",
            r"D:\Program Files\TensorRT-10.14.1.48\bin",
            r"E:\Program Files\TensorRT-10.0\bin",
            r"C:\Program Files\TensorRT-10.0\bin",
            r"D:\Program Files\TensorRT-10.0\bin",
            r"E:\Program Files\TensorRT-8.6\bin",
            r"C:\Program Files\TensorRT-8.6\bin",
            r"D:\Program Files\TensorRT-8.6\bin",
        ]
        
        for path in common_paths:
            dll_path = os.path.join(path, "nvinfer_10.dll")
            if os.path.exists(dll_path):
                logger.info(f"Found TensorRT bin directory: {path}")
                return path
        
        # 方法2: 通过where命令查找nvinfer_10.dll (Windows)
        if platform.system() == "Windows":
            try:
                result = subprocess.run(
                    ["where", "nvinfer_10.dll"],
                    capture_output=True,
                    text=True,
                    timeout=5
                )
                if result.returncode == 0 and result.stdout.strip():
                    dll_path = result.stdout.strip().split('\n')[0]
                    bin_path = os.path.dirname(dll_path)
                    if os.path.exists(bin_path):
                        logger.info(f"Found TensorRT bin directory via 'where' command: {bin_path}")
                        return bin_path
            except Exception as e:
                logger.debug(f"Failed to find TensorRT via 'where' command: {e}")
        
        # 方法3: 检查环境变量
        tensorrt_home = os.environ.get("TENSORRT_HOME")
        if tensorrt_home:
            bin_path = os.path.join(tensorrt_home, "bin")
            if os.path.exists(bin_path):
                logger.info(f"Found TensorRT bin directory via TENSORRT_HOME: {bin_path}")
                return bin_path
        
        # 方法4: 检查PATH环境变量中是否已有TensorRT路径
        path_env = os.environ.get("PATH", "")
        for path in path_env.split(os.pathsep):
            dll_path = os.path.join(path, "nvinfer_10.dll")
            if os.path.exists(dll_path):
                logger.info(f"Found TensorRT bin directory in PATH: {path}")
                return path
        
        logger.warning("Could not find TensorRT bin directory automatically")
        return None
    
    @staticmethod
    def _add_tensorrt_to_path():
        """
        将TensorRT的bin目录添加到PATH环境变量中
        这需要在导入tensorrt模块之前调用
        """
        bin_path = TensorRTConverter._find_tensorrt_bin_path()
        if bin_path:
            current_path = os.environ.get("PATH", "")
            if bin_path not in current_path:
                # 添加到PATH的最前面，优先使用
                os.environ["PATH"] = bin_path + os.pathsep + current_path
                logger.info(f"Added TensorRT bin directory to PATH: {bin_path}")
                return True
            else:
                logger.info(f"TensorRT bin directory already in PATH: {bin_path}")
                return True
        else:
            logger.warning("Could not add TensorRT to PATH: bin directory not found")
            return False
    
    @staticmethod
    def convert(
        input_path,
        output_path=None,
        img_size=(640, 640),
        half=False,
        int8=False,
        workspace=4,
        device=0
    ):
        """
        Convert PyTorch model to TensorRT format
        
        Args:
            input_path (str): Path to the PT model file
            output_path (str, optional): Output path for TensorRT model. If None, uses input path with .engine extension.
            img_size (tuple, optional): Input image size. Defaults to (640, 640).
            half (bool, optional): Whether to use half precision (FP16). Defaults to False.
            int8 (bool, optional): Whether to use INT8 quantization. Defaults to False. Note: int8 and half are mutually exclusive.
            workspace (int, optional): TensorRT workspace size in GB. Defaults to 4.
            device (int, optional): CUDA device ID. Defaults to 0.
            
        Returns:
            tuple: (bool, str) - (success, output_path_or_error_message)
        """
        try:
            logger.info(f"Starting PT to TensorRT conversion for {input_path}")
            
            # Add TensorRT bin directory to PATH before importing tensorrt
            # This is necessary because GUI applications may not have TensorRT in PATH
            TensorRTConverter._add_tensorrt_to_path()
            
            # Validate input path
            is_valid, error_msg = TensorRTConverter.validate_input_path(input_path)
            if not is_valid:
                return False, error_msg
            
            # Check if CUDA is available
            if not torch.cuda.is_available():
                error_msg = "CUDA is not available. TensorRT conversion requires NVIDIA GPU."
                logger.error(error_msg)
                return False, error_msg
            
            # Set default output path if not provided
            if output_path is None:
                output_path = TensorRTConverter.get_default_output_path(input_path, '.engine')
            
            # Load the model using ultralytics
            model = YOLO(input_path)
            
            # Determine precision (for logging purposes)
            if int8:
                precision = 'int8'
            elif half:
                precision = 'fp16'
            else:
                precision = 'fp32'
            
            logger.info(f"Converting with precision: {precision}, workspace: {workspace}GB, device: {device}")
            
            # Export the model to TensorRT format
            # Note: ultralytics YOLO.export supports TensorRT export
            model.export(
                format='engine',
                imgsz=img_size,
                half=half,
                int8=int8,
                workspace=workspace,
                device=device
            )
            
            # The YOLO export function saves the model in the same directory as the input
            # with a .engine extension. Let's move it if necessary.
            default_output = TensorRTConverter.get_default_output_path(input_path, '.engine')
            TensorRTConverter.move_output_file(default_output, output_path)

            # Write metadata next to the exported engine (best effort; does not affect success)
            TensorRTConverter._write_metadata(
                output_path=output_path,
                conversion_params={
                    "input_path": os.path.abspath(input_path),
                    "output_path": os.path.abspath(output_path),
                    "img_size": list(img_size),
                    "half": bool(half),
                    "int8": bool(int8),
                    "workspace_gb": workspace,
                    "device": device,
                },
            )
            
            logger.info(f"Successfully converted model to TensorRT: {output_path}")
            return True, output_path
            
        except Exception as e:
            error_msg = f"Error converting model to TensorRT: {str(e)}\n{traceback.format_exc()}"
            logger.error(error_msg)
            return False, error_msg

