"""训练手势数字识别 YOLO 模型 (0-9 digit gestures)"""
import os, sys

from ultralytics import YOLO


def main():
    BASE = os.path.dirname(os.path.abspath(__file__))

    # ===== 配置参数 =====
    config = {
        'data': os.path.join(BASE, 'data.yaml'),
        'epochs': 200,           # 训练轮数
        'imgsz': 640,            # 图片尺寸
        'batch': 16,             # 批次大小（CPU 用 8-16，GPU 用 32+）
        'device': 0,              # RTX 4070 GPU
        'patience': 50,          # 早停：50 轮无改善就停止
        'lr0': 0.001,            # 初始学习率（迁移学习用低学习率）
        'lrf': 0.01,             # 最终学习率 = lr0 * lrf
        'momentum': 0.937,
        'weight_decay': 0.0005,
        'warmup_epochs': 3,
        'mosaic': 0.5,           # 小数据集减少 mosaic 概率
        'mixup': 0.0,            # 关闭 mixup（数据太少意义不大）
        'hsv_h': 0.015,
        'hsv_s': 0.7,
        'hsv_v': 0.4,
        'degrees': 10.0,         # ±10° 旋转
        'translate': 0.1,
        'scale': 0.5,
        'fliplr': 0.5,           # 50% 水平翻转
        'project': os.path.join(BASE, 'runs'),
        'name': 'digit_gesture',
        'exist_ok': True,
        'pretrained': True,
        'verbose': True,
    }

    print('=' * 60)
    print(f'训练配置:')
    print(f'  数据集: {config["data"]}')
    print(f'  轮数: {config["epochs"]}')
    print(f'  设备: {config["device"]}')
    print(f'  批次: {config["batch"]}')
    print(f'  图片尺寸: {config["imgsz"]}')
    print('=' * 60)

    # ===== 选择模型 =====
    # 方案 A: 用 Hagrid 预训练模型做迁移学习（推荐）
    pretrained_path = os.path.join(BASE, '_internal', 'best.pt')
    if os.path.exists(pretrained_path):
        print(f'\n检测到预训练模型: _internal/best.pt (Hagrid 34类)')
        print('使用迁移学习模式：基于 Hagrid 权重微调')
        model = YOLO(pretrained_path)
        
        # 注意: YOLO 要求新 data.yaml 的 nc 与预训练模型匹配或从头训练
        # 由于类别数不同(4 vs 34)，实际会保留 backbone 权重重新训练 head
    else:
        print('\n未找到预训练模型，使用 yolo11n 从头训练')
        model = YOLO('yolo11n.pt')

    # ===== 开始训练 =====
    print('\n开始训练...\n')
    try:
        results = model.train(**config)
        print('\n' + '=' * 60)
        print('训练完成!')
        print(f'最佳模型: {results.save_dir}/weights/best.pt')
        print('=' * 60)

        # 提示部署
        print('\n部署方法:')
        print(f'  复制 runs/digit_gesture/weights/best.pt 到 _internal/best.pt')
        print('  然后重新启动 Gesture.exe 即可')

    except Exception as e:
        print(f'\n训练出错: {e}')
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
