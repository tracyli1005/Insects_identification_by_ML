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
  Converts XML annotations to YOLO format with:
  ```bash
  python trans_data_format.py --input insects/ --output insects/yolo11_dataset
