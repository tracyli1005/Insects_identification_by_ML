# 🐞 Insect Detection & Classification System

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![Pytorch](https://img.shields.io/badge/PyTorch-2.0+-red)](https://pytorch.org/)
[![YOLO](https://img.shields.io/badge/YOLOv11-ultralytics-green)](https://github.com/ultralytics/ultralytics)
[![Flask](https://img.shields.io/badge/Flask-API%20Ready-lightgrey)](https://flask.palletsprojects.com/)
## 🚀 Core Features
This project implements a dual-model insect detection system using &zwnj;**Faster R-CNN**&zwnj; and &zwnj;**YOLOv11**&zwnj;, featuring dataset conversion, model training/evaluation, and a web application for practical deployment.

---
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

```

1. &zwnj;**Data Preprocessing**&zwnj;  
   - `trans_data_format.py`
   Converts XML annotations to YOLO format:insects/yolo11_dataset
     ```python
     # example code
     python trans_data_format.py --input insects/ --output insects/yolo_dataset
     ```
     ▸ Generate `data.yaml` configuration file

2. &zwnj;**Dual model training**&zwnj;  
   | Script | Framework | Acceleration |
   |------|------|----------|
   | `faster_rcnn_train.py` | PyTorch | CUDA/MPL |
   | `yolo11_train.py` | Ultralytics | CUDA/DDP |
  ```python
     # Faster R-CNN training
     python faster_rcnn_train.py --epochs 15 --batch_size 4
     # YOLO training
     python yolo11_train.py --epochs 100 --imgsz 640
  ```
3. &zwnj;**Performance evaluation**&zwnj;
  ```python
      python faster_rcnn_evaluate.py --model model/faster_rcnn.pth
      python yolo11_evaluate.py --model model/yolo11n.pt
  ```

   | Metrics| Faster R-CNN | YOLOv11 |
   |------|------|----------|
   | mAP@0.5 | 89.2%| 81.5% |
   | FPS | 23.6 | 58.4 |
   | Model Size | 187MB | 13.7MB |
   | Memory Usage | 4.8GB | 2.1GB |
   
4. &zwnj;**Testing with agriculture pests dataset**&zwnj;
   7 categories: 'Boerner', 'Leconte', 'Linnaeus', 'acuminatus', 'armandi', 'coleoptera', 'linnaeus'
   Testing flow:
   ![QQ_1744441745076](https://github.com/user-attachments/assets/99412546-9a78-4711-a28d-dc60f8431559)

   ![QQ_1744441902480](https://github.com/user-attachments/assets/a5bced5b-9b6d-4e07-97d4-80f7de0e43a6)

   ![QQ_1744444522382](https://github.com/user-attachments/assets/bab01a85-e350-451c-bd90-f79fcf96531b)

   ![QQ_1744443357123](https://github.com/user-attachments/assets/76a7f76d-aec2-4eee-b06a-1c27eb6e9bfd)

   
5. &zwnj;**Web Service Deployment**&zwnj;  
   ```bash
   # Command(Gunicorn is recommended for production environments)
   flask run --host=0.0.0.0 --port=5000 --debug
   ```
---
## 🚀 Next step 
   As existing insect datasets (e.g., Kaggle) often lack diverse, high-quality images of insects in real-world settings (e.g., urban environments, natural habitats),I plan to collect insect data from Tiktok user generated videos to augment training data to improve model robustness across varied environments, validate the model’s performance on "in-the-wild" images, and explore potential usage scenario of the model, such as: mapping the geographic distribution of insects tagged in #insects videos by cross-referencing video metadata (creator location), assess risks posed by invasive species or harmful interactions observed in user-generated content.

 



   
   
