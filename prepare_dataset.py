"""使用 MediaPipe 自动标注手部边界框，将原始采集图片转换为 YOLO 格式数据集。"""
import os, shutil, sys, random

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
from tqdm import tqdm


def imread_unicode(path):
    """cv2.imread 不支持中文路径，用 imdecode 替代"""
    data = np.fromfile(path, dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def imwrite_unicode(path, img):
    """cv2.imwrite 不支持中文路径，用 imencode 替代"""
    ext = os.path.splitext(path)[1]
    _, buf = cv2.imencode(ext, img)
    buf.tofile(path)

# ===== 配置 =====
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SAMPLES_DIR = os.path.join(BASE_DIR, 'training_samples')
DATASET_DIR = os.path.join(BASE_DIR, 'dataset')
IMG_SIZE = 640
TRAIN_RATIO = 0.85

# 手势类别映射: 文件夹名 -> 类别ID
CLASS_MAP = {
    'zero': 0,
    'one': 1,
    'two': 2,
    'three': 3,
    'four': 4,
    'five': 5,
    'six': 6,
    'seven': 7,
    'eight': 8,
    'nine': 9,
}

# ===== 初始化 MediaPipe Tasks API =====
# 模型需放在纯英文路径（MediaPipe C++ 后端不兼容中文路径）
model_path = 'C:/temp_models/hand_landmarker.task'
base_options = mp_python.BaseOptions(model_asset_path=model_path)
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=1,
    min_hand_detection_confidence=0.2,
    min_hand_presence_confidence=0.2,
    min_tracking_confidence=0.2,
    running_mode=vision.RunningMode.IMAGE,
)
detector = vision.HandLandmarker.create_from_options(options)


def get_hand_bbox(image):
    """用 MediaPipe 检测手部，返回 YOLO 格式边界框 (xc, yc, w, h)，归一化到 [0,1]"""
    h, w = image.shape[:2]
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = detector.detect(mp_image)

    if result.hand_landmarks:
        lm = result.hand_landmarks[0]
        xs = [p.x for p in lm]
        ys = [p.y for p in lm]
        x_min, x_max = min(xs), max(xs)
        y_min, y_max = min(ys), max(ys)

        # 扩大 25% 作为边界框，确保包含完整手部
        bw = (x_max - x_min) * 1.25
        bh = (y_max - y_min) * 1.25
        xc = (x_min + x_max) / 2
        yc = (y_min + y_max) / 2

        # Clamp 到 [0,1]
        x1 = max(0, xc - bw / 2)
        y1 = max(0, yc - bh / 2)
        x2 = min(1, xc + bw / 2)
        y2 = min(1, yc + bh / 2)

        xc_final = (x1 + x2) / 2
        yc_final = (y1 + y2) / 2
        bw_final = x2 - x1
        bh_final = y2 - y1

        return True, xc_final, yc_final, bw_final, bh_final
    else:
        return False, 0.5, 0.5, 1.0, 1.0


def main():
    print('=' * 60)
    print('手势数据集准备 — MediaPipe 自动标注 YOLO 格式')
    print('=' * 60)

    all_samples = []

    # 扫描所有样本
    for cls_name, cls_id in CLASS_MAP.items():
        cls_dir = os.path.join(SAMPLES_DIR, cls_name)
        if not os.path.isdir(cls_dir):
            print(f'  [跳过] 文件夹不存在: {cls_dir}')
            continue

        images = [f for f in os.listdir(cls_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
        for img_name in images:
            all_samples.append((os.path.join(cls_dir, img_name), cls_id, cls_name, img_name))

        print(f'  {cls_name} (id={cls_id}): {len(images)} 张图片')

    if not all_samples:
        print('\n错误: 没有找到任何训练样本!')
        print(f'请确保 {SAMPLES_DIR} 下有子文件夹（如 six/, seven/ 等）')
        return

    # 打乱并分割训练/验证集
    random.seed(42)
    random.shuffle(all_samples)
    split_idx = int(len(all_samples) * TRAIN_RATIO)
    train_samples = all_samples[:split_idx]
    val_samples = all_samples[split_idx:]

    print(f'\n总计: {len(all_samples)} 张图片')
    print(f'训练集: {len(train_samples)} 张, 验证集: {len(val_samples)} 张')

    # 创建输出目录
    for split in ['train', 'val']:
        img_dir = os.path.join(DATASET_DIR, 'images', split)
        lbl_dir = os.path.join(DATASET_DIR, 'labels', split)
        for d in [img_dir, lbl_dir]:
            if os.path.exists(d):
                shutil.rmtree(d)
            os.makedirs(d)

    # 处理并保存
    success_count = 0
    fail_count = 0

    for split_name, samples in [('train', train_samples), ('val', val_samples)]:
        img_out = os.path.join(DATASET_DIR, 'images', split_name)
        lbl_out = os.path.join(DATASET_DIR, 'labels', split_name)

        for src_path, cls_id, cls_name, fname in tqdm(samples, desc=f'处理 {split_name}'):
            img = imread_unicode(src_path)
            if img is None:
                fail_count += 1
                continue

            # 检测手部边界框
            ok, xc, yc, bw, bh = get_hand_bbox(img)

            if not ok:
                fail_count += 1
                continue

            # Resize 到统一尺寸
            img_resized = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

            # 保存图片
            out_name = f'{cls_name}_{fname}'
            imwrite_unicode(os.path.join(img_out, out_name), img_resized)

            # 保存 YOLO 标签
            label_name = out_name.rsplit('.', 1)[0] + '.txt'
            with open(os.path.join(lbl_out, label_name), 'w') as f:
                f.write(f'{cls_id} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}\n')

            success_count += 1

    print(f'\n完成! 成功: {success_count} 张, 跳过(未检测到手): {fail_count} 张')
    print(f'数据集路径: {DATASET_DIR}')


if __name__ == '__main__':
    main()
