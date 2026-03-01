"""
模型转换脚本
用于在 conda 环境中执行模型转换
"""

import os
import json
import argparse
import sys


def convert_model(settings_path=None):
    """根据设置执行模型转换"""
    try:
        # 加载设置
        if settings_path and os.path.exists(settings_path):
            with open(settings_path, 'r', encoding='utf-8') as f:
                settings = json.load(f)
        else:
            print("错误: 未找到转换设置文件")
            return False
        
        # 获取转换参数
        format_type = settings.get('format_type', 'onnx')
        input_path = settings.get('input_path')
        output_path = settings.get('output_path')
        img_size = settings.get('img_size', [640, 640])
        
        if not input_path or not os.path.exists(input_path):
            print(f"错误: 输入模型文件不存在: {input_path}")
            return False
        
        # 导入转换器
        from utils.model_converter import ModelConverter
        
        print(f"\n开始转换模型...")
        print(f"输入文件: {input_path}")
        print(f"输出文件: {output_path}")
        print(f"格式: {format_type.upper()}")
        print(f"图像尺寸: {img_size[0]}x{img_size[1]}\n")
        
        # 执行转换
        if format_type == 'onnx':
            success, result = ModelConverter.pt_to_onnx(
                input_path=input_path,
                output_path=output_path,
                img_size=tuple(img_size),
                simplify=settings.get('simplify', True),
                opset=settings.get('opset', 12),
                half=settings.get('half', False)
            )
        elif format_type == 'tensorrt':
            success, result = ModelConverter.pt_to_tensorrt(
                input_path=input_path,
                output_path=output_path,
                img_size=tuple(img_size),
                half=settings.get('half', False),
                int8=settings.get('int8', False),
                workspace=settings.get('workspace', 4),
                device=settings.get('device', 0)
            )
        else:
            print(f"错误: 不支持的格式类型: {format_type}")
            return False
        
        if success:
            print(f"\n✓ 转换成功!")
            print(f"输出文件: {result}")
            if format_type == 'tensorrt':
                try:
                    from utils.model_converter.tensorrt_converter import TensorRTConverter
                    meta_path = TensorRTConverter._get_metadata_path(result)
                    if os.path.exists(meta_path):
                        print(f"Metadata: {meta_path}")
                except Exception:
                    pass
            return True
        else:
            print(f"\n✗ 转换失败!")
            print(f"错误信息: {result}")
            return False
            
    except Exception as e:
        print(f"转换过程中出错: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='模型转换脚本')
    parser.add_argument('--settings', type=str, required=True,
                       help='转换设置 JSON 文件路径')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("模型转换脚本")
    print("=" * 60)
    
    success = convert_model(args.settings)
    
    if success:
        print("\n" + "=" * 60)
        print("转换完成!")
        print("=" * 60)
        sys.exit(0)
    else:
        print("\n" + "=" * 60)
        print("转换失败!")
        print("=" * 60)
        sys.exit(1)


if __name__ == "__main__":
    main()

