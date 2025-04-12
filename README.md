# 🐞 Insect Detection & Classification System

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![Pytorch](https://img.shields.io/badge/PyTorch-2.0+-red)](https://pytorch.org/)
[![YOLO](https://img.shields.io/badge/YOLOv11-ultralytics-green)](https://github.com/ultralytics/ultralytics)
[![Flask](https://img.shields.io/badge/Flask-API%20Ready-lightgrey)](https://flask.palletsprojects.com/)

This project implements a dual-model insect detection system using &zwnj;**Faster R-CNN**&zwnj; and &zwnj;**YOLOv11**&zwnj;, featuring dataset conversion, model training/evaluation, and a web application for practical deployment.

---

## 🚀 Core Features
### 1. &zwnj;**Data Preparation**&zwnj;
- 🛠️ `trans_data_format.py`  
  Converts XML annotations to YOLO format:insects/yolo11_dataset
## 📂 项目目录结构详解

### 核心目录树（使用`tree`命令生成样式）
```bash
insects2/
├── apply/               # 应用交互目录
│   ├── input/           # ▶️ 用户上传的待检测文件
│   └── output/          # ✅ 模型处理后的输出结果
├── evaluate/            # 📊 模型评估数据存储
│   ├── faster_rcnn/     # 🔍 Faster R-CNN评估结果
│   │   ├── roc_curve.png
│   │   └── confusion_matrix.png
│   └── yolo11/          # 🚀 YOLOv11评估结果
├── insects/             # 🐛 原始数据集存储（XML标注）
├── model/               # 🤖 预训练模型存储
│   ├── faster_rcnn.pth  # 🔥 PyTorch格式模型权重
│   └── yolo11n.pt       # ⚡ Ultralytics格式模型文件
├── web_app/             # 🌐 Flask网页模板（可选）
├── faster_rcnn_apply.py    # 🎯 Faster R-CNN推理脚本
├── faster_rcnn_evaluate.py  # 📈 Faster R-CNN评估脚本
├── faster_rcnn_train.py     # 🏋️ Faster R-CNN训练脚本
├── trans_data_format.py     # 🔄 数据格式转换主工具
├── trans_data_format2.py    # 🔧 备用数据转换脚本
├── web.app.py               # 🖥️ Flask应用入口
├── yolo11_apply.py          # 💡 YOLOv11推理脚本
├── yolo11_evaluate.py       # 📉 YOLOv11评估脚本
└── yolo11_train.py          # 🚂 YOLOv11训练脚本

