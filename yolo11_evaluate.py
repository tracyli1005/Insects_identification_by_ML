import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from ultralytics import YOLO
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve, auc, log_loss
)
import pandas as pd
from tqdm import tqdm
from PIL import Image
import yaml
import glob
plt.rcParams['font.sans-serif'] = ['/System/Library/Fonts/AppleSDGothicNeo.ttc','Yuanti SC']
plt.rcParams['axes.unicode_minus'] = False

def load_validation_data(dataset_path):
    """Load validation set image paths and corresponding labels"""
    # Get validation set image paths
    val_images_dir = os.path.join(dataset_path, 'images', 'val')
    val_labels_dir = os.path.join(dataset_path, 'labels', 'val')
    
    # Get all validation set image files
    image_paths = glob.glob(os.path.join(val_images_dir, '*.jpg')) + \
                  glob.glob(os.path.join(val_images_dir, '*.jpeg')) + \
                  glob.glob(os.path.join(val_images_dir, '*.png'))
    
    data = []
    for img_path in image_paths:
        # Get corresponding label file path
        base_name = os.path.splitext(os.path.basename(img_path))[0]
        label_path = os.path.join(val_labels_dir, base_name + '.txt')
        
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                labels = [line.strip().split() for line in f]
                # Each label format: [class_id, x_center, y_center, width, height]
                labels = [[int(l[0]), float(l[1]), float(l[2]), float(l[3]), float(l[4])] for l in labels]
            
            data.append((img_path, labels))
    
    return data

def evaluate_yolo11(model_path, val_data, class_names, output_dir):
    """Evaluate YOLO11 model performance"""
    print(f"Evaluating using model {model_path}...")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load model
    model = YOLO(model_path)
    
    # Store ground truth and predictions
    all_preds = []
    all_true = []
    all_scores = []
    
    # Predict for each image
    for img_path, true_labels in tqdm(val_data, desc="Evaluation progress"):
        # Make predictions
        results = model.predict(img_path, conf=0.25)
        
        # Get ground truth classes
        true_classes = [label[0] for label in true_labels]
        
        # Get prediction results
        pred_classes = []
        pred_scores = []
        
        for r in results:
            boxes = r.boxes
            if len(boxes) > 0:
                for box in boxes:
                    cls = int(box.cls.item())
                    conf = box.conf.item()
                    pred_classes.append(cls)
                    pred_scores.append(conf)
        
        # Match each ground truth object with predictions
        # Simplified approach: if prediction count doesn't match ground truth, take minimum count for comparison
        min_len = min(len(true_classes), len(pred_classes))
        
        if min_len > 0:
            all_true.extend(true_classes[:min_len])
            all_preds.extend(pred_classes[:min_len])
            all_scores.extend(pred_scores[:min_len])
    
    # Exit early if no predictions
    if len(all_preds) == 0:
        print("Not enough predictions for evaluation")
        return
    
    # Calculate metrics
    accuracy = accuracy_score(all_true, all_preds)
    precision = precision_score(all_true, all_preds, average='macro', zero_division=0)
    recall = recall_score(all_true, all_preds, average='macro', zero_division=0)
    f1 = f1_score(all_true, all_preds, average='macro', zero_division=0)
    
    # Print results
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    
    # Save metrics to text file
    with open(os.path.join(output_dir, 'metrics.txt'), 'w') as f:
        f.write(f"Accuracy: {accuracy:.4f}\n")
        f.write(f"Precision: {precision:.4f}\n")
        f.write(f"Recall: {recall:.4f}\n")
        f.write(f"F1 Score: {f1:.4f}\n")
    
    # Confusion matrix
    cm = confusion_matrix(all_true, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", 
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Prediction class')
    plt.ylabel('True class')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'))
    plt.close()
    
    # Calculate ROC curve and AUC for each class
    plt.figure(figsize=(10, 8))
    
    # Convert labels to one-hot encoding
    n_classes = len(class_names)
    y_true_onehot = np.zeros((len(all_true), n_classes))
    for i, cls in enumerate(all_true):
        y_true_onehot[i, cls] = 1
    
    # Convert prediction scores to class probabilities
    y_scores = np.zeros((len(all_preds), n_classes))
    for i, (cls, score) in enumerate(zip(all_preds, all_scores)):
        y_scores[i, cls] = score
    
    # Calculate ROC curve and AUC for each class
    for i, class_name in enumerate(class_names):
        if i in all_true:  # Only calculate for classes present in dataset
            try:
                fpr, tpr, _ = roc_curve(y_true_onehot[:, i], y_scores[:, i])
                roc_auc = auc(fpr, tpr)
                plt.plot(fpr, tpr, lw=2,
                        label=f'{class_name} (AUC = {roc_auc:.2f})')
            except:
                print(f"Cannot calculate ROC curve for class {class_name}")
    
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(output_dir, 'roc_curve.png'))
    plt.close()
    
    # Calculate log loss
    try:
        # Since log_loss requires probabilities, we use prediction scores as approximation
        log_loss_value = log_loss(y_true_onehot, y_scores, eps=1e-15)
        print(f"Log Loss: {log_loss_value:.4f}")
        
        with open(os.path.join(output_dir, 'metrics.txt'), 'a') as f:
            f.write(f"Log Loss: {log_loss_value:.4f}\n")
    except:
        print("Cannot calculate log loss")
    
    # Visualize some prediction results
    visualize_predictions(model, val_data[:5], class_names, output_dir)
    
    print(f"Evaluation completed, results saved to {output_dir}")

def visualize_predictions(model, sample_data, class_names, output_dir):
    """Visualize some prediction results"""
    vis_dir = os.path.join(output_dir, 'visualizations')
    os.makedirs(vis_dir, exist_ok=True)
    
    for i, (img_path, _) in enumerate(sample_data):
        try:
            # Make predictions and save visualization
            results = model.predict(img_path, conf=0.25, save=True, save_conf=True)
            
            # Get YOLO auto-generated visualization
            pred_img_path = results[0].save_dir
            pred_img_name = os.path.basename(img_path)
            pred_img_path = os.path.join(pred_img_path, pred_img_name)
            
            # If visualization exists, copy to our output directory
            if os.path.exists(pred_img_path):
                import shutil
                shutil.copy(pred_img_path, os.path.join(vis_dir, f'sample_{i+1}_prediction.jpg'))
        except Exception as e:
            print(f"Error visualizing sample {i+1}: {e}")

if __name__ == "__main__":
    # Dataset path
    script_dir = Path(__file__).absolute().parent  # Get script directory
    dataset_path = os.path.join(script_dir, 'insects/yolo11_dataset')  # Combine paths
    
    # Load class names
    with open(os.path.join(dataset_path, 'data.yaml'), 'r') as f:
        data_config = yaml.safe_load(f)
        class_names = data_config['names']
    
    # Load validation data
    val_data = load_validation_data(dataset_path)
    
    # Ensure output directory exists
    output_dir = os.path.join(script_dir,'evaluate/yolo11')
    os.makedirs(output_dir, exist_ok=True)
    
    # Model path, using trained model
    model_path = os.path.join(script_dir,'model/yolo11_insects_final.pt')
    if not os.path.exists(model_path):
        # If final model doesn't exist, try using best model saved during training
        model_path = os.path.join(script_dir,'model/yolo11_insects/weights/best.pt')
        if not os.path.exists(model_path):
            print("Trained model not found, please run training script first")
            exit(1)
    
    # Evaluate model
    evaluate_yolo11(model_path, val_data, class_names, output_dir)