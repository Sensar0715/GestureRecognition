# GestureRecognition V5.0 — 手势数字识别系统

基于 YOLO11 + MediaPipe 的实时手势数字（0–9）识别系统。

---

## 1. 项目结构

```
Gesture/
├── Gesture.exe                  # 打包后的主程序 (PyInstaller)
├── Gesture.exe_extracted/       # 主程序反编译产物
├── _internal/                   # 运行时依赖 (MediaPipe, OpenCV, PyTorch 等)
│   └── best.pt                  # 预训练模型 (Hagrid 34类手势)
├── train.py                     # YOLO 模型训练脚本
├── prepare_dataset.py           # MediaPipe 自动标注 → YOLO 数据集
├── data.yaml                    # 数据集配置 (类别定义)
├── training_samples/            # 原始采集的手势图片 (按类别分文件夹)
│   ├── zero/                    # 手势 0 的原始图片
│   ├── one/                     # 手势 1 的原始图片
│   ├── ...                      #
│   └── nine/                    # 手势 9 的原始图片
├── dataset/                     # YOLO 格式数据集 (由 prepare_dataset.py 生成)
│   ├── images/
│   │   ├── train/               # 训练图片 (85%)
│   │   └── val/                 # 验证图片 (15%)
│   └── labels/
│       ├── train/               # YOLO 标签文件
│       └── val/                 # YOLO 标签文件
├── runs/digit_gesture/          # 训练输出
│   ├── weights/best.pt          # 训练得到的最佳模型
│   ├── results.csv              # 训练指标记录
│   ├── confusion_matrix.png     # 混淆矩阵
│   └── results.png              # 训练曲线图
├── non_digit_gestures/          # 已排除的非数字手势类别清单
├── yolo26n.pt                   # YOLO11 nano 基础权重
├── smartvision_data.db          # 程序运行数据库 (SQLite)
└── 数据集缺陷报告.txt            # 数据集质量分析报告
```

---

## 2. 识别原理

### 2.1 整体流程

```
摄像头输入 → MediaPipe 手部检测 → 手部区域裁剪/归一化 → YOLO11 分类 → 输出数字 (0-9)
```

### 2.2 MediaPipe 手部关键点检测

系统使用 Google MediaPipe 的 Hand Landmarker 模型，实时检测手部的 **21 个关键点**（指尖、指关节、手腕等）。这些关键点用于：

- **手部定位**：根据关键点的最小外接矩形 (bounding box) 定位手部在画面中的位置
- **区域裁剪**：将检测框放大 25%（确保完整包含手指），裁剪出手部 ROI（感兴趣区域）
- **归一化**：将裁剪区域缩放到 640×640，作为 YOLO 模型的输入

### 2.3 YOLO11 目标检测与分类

采用 **Ultralytics YOLO11 nano** 作为核心识别模型：

| 参数 | 值 |
|------|-----|
| 模型架构 | YOLO11 nano |
| 输入尺寸 | 640×640 |
| 输出类别 | 10 类 (数字 0–9) |
| 任务类型 | 目标检测 (detect) |

YOLO 模型同时完成 **定位** 和 **分类** 两个任务：

- **定位**：输出手部边界框 (bounding box)，格式为 `(x_center, y_center, width, height)`，归一化到 [0,1]
- **分类**：输出每个检测框对应的手势类别 (0–9) 及置信度分数

### 2.4 迁移学习策略

训练采用 **两阶段迁移学习**：

1. **第一阶段**：从 `yolo11n.pt`（ImageNet + COCO 预训练）继承 backbone 权重，获得通用视觉特征提取能力
2. **第二阶段**：从 `_internal/best.pt`（Hagrid 数据集预训练，34 类手势）加载权重，backbone 已具备手部特征理解能力，仅需重新训练检测头适应新的 10 类输出

由于预训练模型类别数 (34) 与当前任务 (10) 不同，YOLO 自动保留 backbone 的迁移权重，重新初始化并训练检测头 (head) 部分。

---

## 3. 训练方案

### 3.1 训练配置

| 超参数 | 值 | 说明 |
|--------|-----|------|
| epochs | 200 | 最大训练轮数 |
| imgsz | 640 | 输入图片尺寸 |
| batch | 16 | 批次大小 (可根据显存调整) |
| lr0 | 0.001 | 初始学习率 (迁移学习用低学习率) |
| lrf | 0.01 | 最终学习率因子 (最终 lr = lr0 × lrf = 1e-5) |
| momentum | 0.937 | SGD 动量 |
| weight_decay | 0.0005 | 权重衰减 (L2 正则化) |
| warmup_epochs | 3 | 学习率预热轮数 |
| patience | 50 | 早停 — 50 轮无改善则停止 |
| optimizer | auto | 自动选择优化器 (AdamW) |

### 3.2 数据增强

为提升模型在真实场景中的泛化能力，训练时对每张图片随机施加以下增强：

| 增强方式 | 参数 | 效果 |
|----------|------|------|
| HSV 色相抖动 | ±0.015 | 模拟不同光照色温 |
| HSV 饱和度抖动 | ±0.7 | 模拟不同色彩饱和度 |
| HSV 明度抖动 | ±0.4 | 模拟不同亮度环境 |
| 旋转 | ±10° | 模拟手部倾斜 |
| 平移 | ±10% | 模拟手部位置偏移 |
| 缩放 | 50%–150% | 模拟手部远近变化 |
| 水平翻转 | 50% 概率 | 模拟左右手 |
| Mosaic 拼接 | 50% 概率 | 4 张图拼接训练，提升小目标检测 |

### 3.3 训练步骤

**第一步：采集原始样本**

用手势采集工具（或摄像头拍照），为每个数字采集 100+ 张手势图片，放入 `training_samples/<类别名>/` 文件夹。建议覆盖：
- 不同角度（正面、倾斜 15°、倾斜 30°）
- 不同距离（近、中、远）
- 不同光照（自然光、室内灯、暗光）
- 不同手型（大人手、小孩手、胖瘦）

**第二步：自动标注数据集**

```bash
python prepare_dataset.py
```

该脚本执行：
1. 遍历 `training_samples/` 下每个类别的文件夹
2. 用 MediaPipe Hand Landmarker 检测每张图片中的手部
3. 根据手部 21 个关键点计算边界框（扩展 25% 确保完整包含）
4. 将图片 resize 到 640×640、生成 YOLO 格式标签 (`class_id xc yc w h`)
5. 按 85:15 随机分割训练集/验证集
6. 输出到 `dataset/` 目录

**第三步：训练模型**

```bash
python train.py
```

训练过程监控指标：
- `train/box_loss`：边界框回归损失 (越低越好)
- `train/cls_loss`：分类损失 (越低越好)
- `metrics/mAP50`：IoU=0.5 时的平均精度 (越高越好)
- `metrics/recall`：召回率 (越高越好)

**第四步：部署模型**

```bash
# 将训练好的模型复制到程序目录
cp runs/digit_gesture/weights/best.pt _internal/best.pt
```

重新启动 `Gesture.exe` 即生效。

---

## 4. 当前数据集状况

根据 `数据集缺陷报告.txt` 的分析：

| 类别 | 训练样本 | 占比 | mAP50 | 召回率 |
|------|----------|------|-------|--------|
| three | 887 | 32.5% | 0.993 | 0.980 |
| four | 372 | 13.6% | 0.995 | 1.000 |
| two | 371 | 13.6% | 0.901 | 0.838 |
| five | 272 | 10.0% | 0.982 | 0.862 |
| eight | 200 | 7.3% | 0.955 | 0.958 |
| one | 171 | 6.3% | 0.892 | 0.903 |
| zero | 170 | 6.2% | 0.886 | 0.800 |
| six | 100 | 3.7% | 0.957 | 0.857 |
| nine | 99 | 3.6% | 0.916 | 0.713 |
| seven | 87 | 3.2% | 0.995 | 1.000 |

**已知问题**：
- **类别严重不均衡**：three (887 张) 是 seven (87 张) 的 10.2 倍
- **nine、zero、six 召回率低**，需要补充样本
- **seven 验证集仅 3 张**，评估结果无统计意义

---

## 5. 环境要求

### 5.1 运行环境 (使用 Gesture.exe)

| 组件 | 要求 |
|------|------|
| 操作系统 | Windows 10/11 (64 位) |
| GPU | NVIDIA GPU (CUDA 12.x)，推荐 RTX 3060 及以上 |
| CPU | Intel Core i5 或同等性能 (回退推理) |
| 内存 | 8 GB+ |
| 摄像头 | 720p 或以上 USB/内置摄像头 |
| 存储 | 2 GB 可用空间 |

### 5.2 开发/训练环境

| 组件 | 版本 |
|------|------|
| Python | 3.11 |
| PyTorch | 2.x (CUDA 版本) |
| Ultralytics | 8.x (`pip install ultralytics`) |
| MediaPipe | 0.10.x (`pip install mediapipe`) |
| OpenCV | 4.9+ (`pip install opencv-python`) |
| NumPy | 1.26+ |
| tqdm | 4.x |

**一键安装训练依赖**：

```bash
pip install ultralytics mediapipe opencv-python numpy tqdm
```

### 5.3 MediaPipe 模型文件

`prepare_dataset.py` 需要 MediaPipe 的 `hand_landmarker.task` 模型文件。放置路径：`C:/temp_models/hand_landmarker.task`（需纯英文路径，MediaPipe C++ 后端不支持中文路径）。

从 [Google MediaPipe 官方](https://developers.google.com/mediapipe/solutions/vision/hand_landmarker) 下载。

---

## 6. 核心依赖与许可

| 库 | 用途 | 许可 |
|----|------|------|
| Ultralytics YOLO11 | 手势检测与分类 | AGPL-3.0 |
| Google MediaPipe | 手部关键点检测 | Apache 2.0 |
| OpenCV | 图像处理 | Apache 2.0 |
| PyTorch | 深度学习框架 | BSD |
| PyInstaller | 程序打包 | GPL |

---

## 7. 关于非数字手势

项目包含 Hagrid 数据集中的 30 种非数字手势类别（如 fist、ok、peace、like、call 等），这些已排除在训练之外，列表见 `non_digit_gestures/non_digit_classes.txt`。如需扩展到通用手势识别，可利用 `_internal/best.pt`（34 类预训练权重）进行多类别微调。
