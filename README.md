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
## 📂 Project File Structure

### Core Directory Tree
```bash
insects2/
├── apply/               # Application Interaction Catalog
│   ├── input/           # ▶️ Files to be tested uploaded by the user
│   └── output/          # ✅ Output results after model processing
├── evaluate/            # 📊 Model evaluation data storage
│   ├── faster_rcnn/     # 🔍 Faster R-CNN evaluation results
│   │   ├── roc_curve.png
│   │   └── confusion_matrix.png
│   └── yolo11/          # 🚀 YOLOv11 Evaluation Results
├── insects/             # 🐛 Original data set storage (XML annotation)
├── model/               # 🤖 Pre-trained model 
│   ├── faster_rcnn.pth  # 🔥 PyTorch format model weights
│   └── yolo11n.pt       # ⚡ Ultralytics format model file
├── web_app/             # 🌐 Flask web page template (optional)
├── faster_rcnn_apply.py    # 🎯 Faster R-CNN inference script
├── faster_rcnn_evaluate.py  # 📈 Faster R-CNN evaluation script
├── faster_rcnn_train.py     # 🏋️ Faster R-CNN training script
├── trans_data_format.py     # 🔄 Data format conversion main tool
├── trans_data_format2.py    # 🔧 Alternative data conversion scripts
├── web.app.py               # 🖥️ Flask application entry
├── yolo11_apply.py          # 💡 YOLOv11 inference script
├── yolo11_evaluate.py       # 📉 YOLOv11 evaluation script
└── yolo11_train.py          # 🚂 YOLOv11 training script

