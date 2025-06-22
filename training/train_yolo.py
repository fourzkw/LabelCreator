import os
import json
import argparse
from ultralytics import YOLO

def load_settings(settings_path="yolo_train_settings.json"):
    """从JSON文件加载训练设置"""
    try:
        with open(settings_path, "r", encoding="utf-8") as f:
            settings = json.load(f)
        print("成功加载训练设置")
        return settings
    except Exception as e:
        print(f"加载设置失败: {str(e)}")
        return None

def print_settings(settings):
    """打印训练设置"""
    print("\n" + "=" * 50)
    print("训练设置:")
    print("=" * 50)
    
    # 基本设置
    print(f"\n基本设置:")
    print(f"  数据集YAML: {settings['yaml_path']}")
    
    # 显示模型信息
    if settings.get('use_custom_model', False) and settings.get('custom_model_path', ''):
        print(f"  使用自定义预训练模型: {settings['custom_model_path']}")
    else:
        print(f"  模型类型: {settings['model_type']}")
        print(f"  使用预训练权重: {settings['pretrained']}")
    
    print(f"  训练轮数: {settings['epochs']}")
    print(f"  批次大小: {settings['batch_size']}")
    print(f"  图像大小: {settings['img_size']}")
    print(f"  项目路径: {settings['project_path'] or '当前目录'}")
    print(f"  实验名称: {settings['name']}")
    
    # 高级设置
    print(f"\n高级设置:")
    print(f"  优化器: {settings['optimizer']}")
    print(f"  初始学习率: {settings['lr0']}")
    print(f"  最终学习率因子: {settings['lrf']}")
    print(f"  动量: {settings['momentum']}")
    print(f"  权重衰减: {settings['weight_decay']}")
    print(f"  数据增强: {settings['augment']}")
    print(f"  早停耐心值: {settings['patience']}")
    print(f"  数据加载线程数: {settings['workers']}")
    print(f"  设备: {settings['device'] or '默认'}")
    print(f"  余弦学习率调度: {settings['cos_lr']}")
    print(f"  缓存图像: {settings['cache']}")
    
    # 特征点设置
    if settings.get('enable_keypoints', False):
        print(f"\n特征点设置:")
        print(f"  启用特征点检测: {settings.get('enable_keypoints', False)}")
        print(f"  任务类型: pose (姿态检测)")
    print("=" * 50 + "\n")

def train_yolo(settings):
    """使用设置训练YOLOv8模型"""
    try:
        # 先确定任务类型
        task = "detect"  # 默认任务类型
        if settings.get('enable_keypoints', False):
            task = "pose"
            print(f"启用特征点检测训练，使用pose任务类型")

        # 构建模型
        if settings.get('use_custom_model', False) and settings.get('custom_model_path', ''):
            # 使用自定义预训练模型
            custom_model_path = settings['custom_model_path']
            print(f"使用自定义预训练模型: {custom_model_path}")
            model = YOLO(custom_model_path)
        else:
            # 使用标准模型
            model_name = settings['model_type']
            
            # 根据任务类型选择模型
            if task == "pose":
                model_name = f"{model_name}-pose"
                print(f"选择姿态检测模型: {model_name}")
                
            if settings['pretrained']:
                # 使用预训练模型
                model = YOLO(f"{model_name}.pt")
            else:
                # 从头开始训练
                model = YOLO(f"{model_name}.yaml")
        
        # 准备训练参数
        train_args = {
            'data': settings['yaml_path'],
            'epochs': settings['epochs'],
            'batch': settings['batch_size'],
            'imgsz': settings['img_size'],
            'optimizer': settings['optimizer'].lower(),
            'lr0': settings['lr0'],
            'lrf': settings['lrf'],
            'momentum': settings['momentum'],
            'weight_decay': settings['weight_decay'],
            'warmup_epochs': 3.0,
            'warmup_momentum': 0.8,
            'warmup_bias_lr': 0.1,
            'patience': settings['patience'],
            'workers': settings['workers'],
            'augment': settings['augment'],
            'cos_lr': settings['cos_lr'],
            'cache': settings['cache'],
            'exist_ok': True,
        }
        
        # 添加可选参数
        if settings['project_path']:
            train_args['project'] = settings['project_path']
        
        if settings['name']:
            train_args['name'] = settings['name']
            
        if settings['device']:
            train_args['device'] = settings['device']
        
        # 如果启用了数据增强，添加相关参数
        if settings['augment']:
            train_args['hsv_h'] = settings['hsv_h']
            train_args['hsv_s'] = settings['hsv_s']
            train_args['hsv_v'] = settings['hsv_v']
            train_args['degrees'] = settings['degrees']
            train_args['translate'] = settings['translate']
            train_args['scale'] = settings['scale']
            train_args['fliplr'] = settings['fliplr']
            train_args['mosaic'] = settings['mosaic']
            
        # 开始训练
        print("\n开始训练...\n")
        
        # 确保数据集YAML格式正确
        if task == "pose" and settings.get('enable_keypoints', False):
            print("注意：使用姿态检测(pose)任务训练需要特殊格式的数据集YAML文件！")
            print("数据集YAML必须包含关键点(keypoints)定义和正确的标注格式。")
            print("详情请参考：https://docs.ultralytics.com/datasets/keypoints/")
            
        # 执行训练，指定任务类型
        model.train(task=task, **train_args)
        
        print("\n训练完成!")
        print(f"模型保存在: {os.path.join(train_args.get('project', ''), train_args.get('name', 'exp'))}")
        
    except Exception as e:
        print(f"训练过程中出错: {str(e)}")

def main(settings_path="yolo_train_settings.json"):
    print("YOLOv8 训练脚本启动")
    
    # 加载设置
    settings = load_settings(settings_path)
    if not settings:
        print("无法加载设置，训练终止")
        return
    
    # 打印设置
    print_settings(settings)
    
    # 确认是否继续
    confirm = input("是否开始训练? (y/n): ")
    if confirm.lower() != 'y':
        print("训练已取消")
        return
    
    # 开始训练
    train_yolo(settings)

if __name__ == "__main__":
    main()