import os
import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torch.utils.data import DataLoader
import yaml
from PIL import Image
import cv2
from torchvision import transforms as T
from sklearn.metrics import precision_recall_curve, roc_curve, auc, confusion_matrix, log_loss, precision_score, recall_score, f1_score, accuracy_score
import seaborn as sns
from tqdm import tqdm
from pathlib import Path

plt.rcParams['font.sans-serif'] = ['Yuanti SC']
plt.rcParams['axes.unicode_minus'] = False

# Updated InsectDataset class for YOLOv11 format
class InsectDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, split='val', transforms=None):
        self.root_dir = root_dir
        self.split = split
        self.transforms = transforms
        
        # Define image and label directories based on split
        self.img_dir = os.path.join(root_dir, 'images', split)
        self.label_dir = os.path.join(root_dir, 'labels', split)
        
        # Get list of all images
        self.image_files = [f for f in os.listdir(self.img_dir) 
                           if f.endswith(('.jpg', '.jpeg', '.png'))]
            
        # Load class names from data.yaml
        with open(os.path.join(root_dir, 'data.yaml'), 'r') as f:
            self.class_names = yaml.safe_load(f)['names']
            
        # Default transforms if none provided
        if self.transforms is None:
            self.transforms = T.Compose([
                T.ToTensor(),
            ])
            
    def __len__(self):
        return len(self.image_files)
        
    def __getitem__(self, idx):
        img_file = self.image_files[idx]
        img_path = os.path.join(self.img_dir, img_file)
            
        img = Image.open(img_path).convert("RGB")
        
        # Get corresponding label file
        label_path = os.path.join(self.label_dir, 
                                  os.path.splitext(img_file)[0] + '.txt')
        
        # Read annotations
        boxes = []
        labels = []
        
        with open(label_path, 'r') as f:
            for line in f.readlines():
                class_id, x_center, y_center, width, height = map(float, line.strip().split())
                
                # Convert YOLO format to (x1, y1, x2, y2)
                img_width, img_height = img.size
                x1 = (x_center - width/2) * img_width
                y1 = (y_center - height/2) * img_height
                x2 = (x_center + width/2) * img_width
                y2 = (y_center + height/2) * img_height
                
                boxes.append([x1, y1, x2, y2])
                labels.append(int(class_id) + 1)  # +1 because background is class 0
                
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        
        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = torch.tensor([idx])
        
        # Convert PIL image to tensor
        img = self.transforms(img)
            
        return img, target

def get_model(num_classes):
    # Load a pre-trained model
    model = fasterrcnn_resnet50_fpn(weights=None)
    
    # Get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    
    # Replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    
    return model

def calculate_iou(box1, box2):
    """
    Calculate IoU between box1 and box2
    box format: [x1, y1, x2, y2]
    """
    # Determine coordinates of intersection
    x1_inter = max(box1[0], box2[0])
    y1_inter = max(box1[1], box2[1])
    x2_inter = min(box1[2], box2[2])
    y2_inter = min(box1[3], box2[3])
    
    # Calculate area of intersection
    width_inter = max(0, x2_inter - x1_inter)
    height_inter = max(0, y2_inter - y1_inter)
    area_inter = width_inter * height_inter
    
    # Calculate area of both boxes
    area_box1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area_box2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    # Calculate area of union
    area_union = area_box1 + area_box2 - area_inter
    
    # Calculate IoU
    iou = area_inter / area_union if area_union > 0 else 0
    
    return iou

def evaluate_model(model, data_loader, device, iou_threshold=0.5):
    model.eval()
    
    all_true_labels = []
    all_pred_labels = []
    all_pred_scores = []
    
    with torch.no_grad():
        for images, targets in tqdm(data_loader, desc="Evaluating"):
            images = list(img.to(device) for img in images)
            
            # Get predictions
            outputs = model(images)
            
            # Process each image in the batch
            for i, (output, target) in enumerate(zip(outputs, targets)):
                pred_boxes = output['boxes'].cpu().numpy()
                pred_scores = output['scores'].cpu().numpy()
                pred_labels = output['labels'].cpu().numpy()
                
                true_boxes = target['boxes'].cpu().numpy()
                true_labels = target['labels'].cpu().numpy()
                
                # For each true box, find the best matching predicted box
                for true_box, true_label in zip(true_boxes, true_labels):
                    best_iou = 0
                    best_pred_idx = -1
                    
                    for j, pred_box in enumerate(pred_boxes):
                        iou = calculate_iou(true_box, pred_box)
                        if iou > best_iou:
                            best_iou = iou
                            best_pred_idx = j
                    
                    # If we found a good match
                    if best_iou >= iou_threshold:
                        all_true_labels.append(true_label)
                        all_pred_labels.append(pred_labels[best_pred_idx])
                        all_pred_scores.append(pred_scores[best_pred_idx])
                    else:
                        # No good match found, count as false negative
                        all_true_labels.append(true_label)
                        all_pred_labels.append(0)  # Background
                        all_pred_scores.append(0.0)
                
                # For each prediction that didn't match a true box, count as false positive
                for j, pred_box in enumerate(pred_boxes):
                    matched = False
                    for true_box in true_boxes:
                        if calculate_iou(pred_box, true_box) >= iou_threshold:
                            matched = True
                            break
                    
                    if not matched and pred_scores[j] >= 0.5:  # Only count confident predictions
                        all_true_labels.append(0)  # Background
                        all_pred_labels.append(pred_labels[j])
                        all_pred_scores.append(pred_scores[j])
    
    return np.array(all_true_labels), np.array(all_pred_labels), np.array(all_pred_scores)

def calculate_metrics(true_labels, pred_labels, pred_scores, class_names, output_dir):
    # Convert to one-hot encoding for multi-class metrics
    num_classes = len(class_names) + 1  # +1 for background
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Calculate basic metrics
    accuracy = accuracy_score(true_labels, pred_labels)
    precision = precision_score(true_labels, pred_labels, average='weighted', zero_division=0)
    recall = recall_score(true_labels, pred_labels, average='weighted', zero_division=0)
    f1 = f1_score(true_labels, pred_labels, average='weighted', zero_division=0)
    
    # Print metrics
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    
    # Save metrics to file
    with open(os.path.join(output_dir, 'metrics.txt'), 'w') as f:
        f.write(f"Accuracy: {accuracy:.4f}\n")
        f.write(f"Precision: {precision:.4f}\n")
        f.write(f"Recall: {recall:.4f}\n")
        f.write(f"F1 Score: {f1:.4f}\n")
    
    # Generate confusion matrix
    cm = confusion_matrix(true_labels, pred_labels)
    
    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Background'] + class_names,
                yticklabels=['Background'] + class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'))
    plt.close()
    
    # Calculate ROC curve and AUC for each class
    plt.figure(figsize=(10, 8))
    
    # One-hot encode the true labels for ROC calculation
    true_one_hot = np.zeros((len(true_labels), num_classes))
    for i, label in enumerate(true_labels):
        true_one_hot[i, label] = 1
    
    # Create score matrix for each class
    pred_scores_matrix = np.zeros((len(pred_labels), num_classes))
    for i, (label, score) in enumerate(zip(pred_labels, pred_scores)):
        pred_scores_matrix[i, label] = score
    
    # Calculate ROC for each class
    for i in range(1, num_classes):  # Skip background class
        if i >= len(class_names) + 1:
            continue
            
        # Check if we have any samples of this class
        if np.sum(true_one_hot[:, i]) == 0:
            continue
            
        fpr, tpr, _ = roc_curve(true_one_hot[:, i], pred_scores_matrix[:, i])
        roc_auc = auc(fpr, tpr)
        
        plt.plot(fpr, tpr, lw=2, label=f'{class_names[i-1]} (AUC = {roc_auc:.2f})')
    
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(output_dir, 'roc_curve.png'))
    plt.close()
    
    # Calculate Precision-Recall curve
    plt.figure(figsize=(10, 8))
    
    for i in range(1, num_classes):  # Skip background class
        if i >= len(class_names) + 1:
            continue
            
        # Check if we have any samples of this class
        if np.sum(true_one_hot[:, i]) == 0:
            continue
            
        precision_values, recall_values, _ = precision_recall_curve(true_one_hot[:, i], pred_scores_matrix[:, i])
        plt.plot(recall_values, precision_values, lw=2, label=f'{class_names[i-1]}')
    
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc="best")
    plt.savefig(os.path.join(output_dir, 'precision_recall_curve.png'))
    plt.close()
    
    # Calculate log loss
    # We need to ensure we have predictions for each class
    # Create a matrix of predicted probabilities
    y_prob = np.zeros((len(true_labels), num_classes))
    for i, (label, score) in enumerate(zip(pred_labels, pred_scores)):
        # Assign the prediction score to the predicted class
        y_prob[i, label] = score
        
        # Distribute the remaining probability mass
        remaining = 1.0 - score
        other_classes = [j for j in range(num_classes) if j != label]
        for j in other_classes:
            y_prob[i, j] = remaining / (num_classes - 1)
    
    # Calculate log loss
    logloss = log_loss(true_one_hot, y_prob)
    print(f"Log Loss: {logloss:.4f}")
    
    with open(os.path.join(output_dir, 'metrics.txt'), 'a') as f:
        f.write(f"Log Loss: {logloss:.4f}\n")
    
    return accuracy, precision, recall, f1, logloss


def save_example_detections(model, dataset, device, output_dir, num_examples=5, class_names=None):
    """
    Save visualization of example detections
    """
    model.eval()
    
    # Create directory for example detections
    examples_dir = os.path.join(output_dir, 'example_detections')
    os.makedirs(examples_dir, exist_ok=True)
    
    # Choose random samples
    indices = np.random.choice(len(dataset), min(num_examples, len(dataset)), replace=False)
    
    with torch.no_grad():
        for i, idx in enumerate(indices):
            # Get image and target
            img, target = dataset[idx]
            
            # Convert to device
            img_tensor = img.unsqueeze(0).to(device)
            
            # Get prediction
            prediction = model(img_tensor)[0]
            
            # Convert image back to numpy for visualization
            img_np = img.permute(1, 2, 0).cpu().numpy()
            
            # Normalize image for display
            img_np = (img_np * 255).astype(np.uint8)
            
            # Create a copy for drawing
            img_draw = img_np.copy()
            
            # Draw ground truth boxes (green)
            for box, label in zip(target['boxes'], target['labels']):
                box = box.numpy().astype(np.int32)
                label = label.item()
                cv2.rectangle(img_draw, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
                
                # Add label text
                label_text = class_names[label-1] if class_names and label > 0 and label <= len(class_names) else f"Class {label}"
                cv2.putText(img_draw, f"GT: {label_text}", (box[0], box[1] - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # Draw predicted boxes (red)
            for box, label, score in zip(prediction['boxes'], prediction['labels'], prediction['scores']):
                if score >= 0.5:  # Only show confident predictions
                    box = box.cpu().numpy().astype(np.int32)
                    label = label.cpu().item()
                    cv2.rectangle(img_draw, (box[0], box[1]), (box[2], box[3]), (0, 0, 255), 2)
                    
                    # Add label text
                    label_text = class_names[label-1] if class_names and label > 0 and label <= len(class_names) else f"Class {label}"
                    cv2.putText(img_draw, f"Pred: {label_text} ({score:.2f})", (box[0], box[3] + 20), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            
            # Save the image
            save_path = os.path.join(examples_dir, f'example_{i+1}.jpg')
            cv2.imwrite(save_path, cv2.cvtColor(img_draw, cv2.COLOR_RGB2BGR))
            
            # Also save a side-by-side comparison
            # Original image with ground truth
            img_gt = img_np.copy()
            for box, label in zip(target['boxes'], target['labels']):
                box = box.numpy().astype(np.int32)
                label = label.item()
                cv2.rectangle(img_gt, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
            
            # Image with predictions
            img_pred = img_np.copy()
            for box, label, score in zip(prediction['boxes'], prediction['labels'], prediction['scores']):
                if score >= 0.5:
                    box = box.cpu().numpy().astype(np.int32)
                    label = label.cpu().item()
                    cv2.rectangle(img_pred, (box[0], box[1]), (box[2], box[3]), (0, 0, 255), 2)
            
            # Create side-by-side comparison
            comparison = np.hstack((img_gt, img_pred))
            save_path_comparison = os.path.join(examples_dir, f'comparison_{i+1}.jpg')
            cv2.imwrite(save_path_comparison, cv2.cvtColor(comparison, cv2.COLOR_RGB2BGR))


def main():

    script_dir = Path(__file__).absolute().parent  # 获取脚本所在目录
    # Configuration
    dataset_path = os.path.join(script_dir, 'insects/yolo11_dataset')  # Updated to YOLOv11 dataset path
    model_path = os.path.join(script_dir, 'model/faster_rcnn_insect.pth')
    output_dir = os.path.join(script_dir, 'evaluate/faster_rcnn')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f'Using device: {device}')
    
    # Create dataset and dataloader for validation
    print(f'Loading validation data from: {os.path.join(dataset_path, "images/val")}')
    
    val_dataset = InsectDataset(dataset_path, split='val')
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, 
                            collate_fn=lambda x: tuple(zip(*x)))
    
    # Get number of classes from data.yaml
    yaml_file = os.path.join(dataset_path, 'data.yaml')
    print(f'Loading class definitions from: {yaml_file}')
    
    with open(yaml_file, 'r') as f:
        data_config = yaml.safe_load(f)
        class_names = data_config['names']
        num_classes = len(class_names) + 1  # +1 for background
    
    print(f'Number of classes: {num_classes}')
    print(f'Class names: {class_names}')
    
    # Get model
    model = get_model(num_classes)
    
    # Load model weights
    print(f'Loading model from: {model_path}')
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    
    # Evaluate model
    print('Evaluating model...')
    true_labels, pred_labels, pred_scores = evaluate_model(model, val_loader, device)
    
    # Calculate and save metrics
    print('Calculating metrics...')
    os.makedirs(output_dir, exist_ok=True)
    calculate_metrics(true_labels, pred_labels, pred_scores, class_names, output_dir)
    
    # Generate and save some example visualizations
    save_example_detections(model, val_dataset, device, output_dir, num_examples=5, class_names=class_names)
    
    print(f'Evaluation complete. Results saved to {output_dir}')

if __name__ == '__main__':
    main()