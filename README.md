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
  ├── images/
│   ├── train/       # 训练集图像
│   │   ├── img1.jpg
│   │   └── ...
│   ├── val/         # 验证集图像
│   │   ├── img2.jpg
│   │   └── ...
│   └── test/        # （可选）测试集图像
├── labels/
│   ├── train/       # 训练集标签
│   │   ├── img1.txt
│   │   └── ...
│   ├── val/         # 验证集标签
│   │   ├── img2.txt
│   │   └── ...
│   └── test/        # （可选）测试集标签
└── data.yaml        # 配置文件
