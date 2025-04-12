# 🐞 Insect Detection & Classification System

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![Pytorch](https://img.shields.io/badge/PyTorch-2.0+-red)](https://pytorch.org/)
[![YOLO](https://img.shields.io/badge/YOLOv11-ultralytics-green)](https://github.com/ultralytics/ultralytics)
[![Flask](https://img.shields.io/badge/Flask-API%20Ready-lightgrey)](https://flask.palletsprojects.com/)

This project implements a dual-model insect detection system using &zwnj;**Faster R-CNN**&zwnj; and &zwnj;**YOLOv11**&zwnj;, featuring dataset conversion, model training/evaluation, and a web application for practical deployment.

---

## 🚀 Core Features
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
### 1. &zwnj;**Data Preparation**&zwnj;
- 🛠️ `trans_data_format.py`  
  Converts XML annotations to YOLO format:insects/yolo11_dataset

1. &zwnj;**数据预处理**&zwnj;  
   - `trans_data_format.py`  
     ```python
     # 示例调用命令
     python trans_data_format.py --input insects/ --output insects/yolo_dataset
     ```
     ▸ 支持多线程转换  
     ▸ 自动生成`data.yaml`配置文件

2. &zwnj;**模型训练双引擎**&zwnj;  
   | 脚本 | 框架 | 加速支持 |
   |------|------|----------|
   | `faster_rcnn_train.py` | PyTorch | CUDA/MPL |
   | `yolo11_train.py` | Ultralytics | CUDA/DDP |

3. &zwnj;**Web服务部署**&zwnj;  
   ```bash
   # 启动命令（生产环境建议使用gunicorn）
   flask run --host=0.0.0.0 --port=5000 --debug

