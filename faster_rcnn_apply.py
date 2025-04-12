import os
import torch
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import yaml
from PIL import Image, ImageDraw, ImageFont
import cv2
import numpy as np
from torchvision import transforms as T
import matplotlib.pyplot as plt
import glob
from tqdm import tqdm
from pathlib import Path

script_dir = Path(__file__).absolute().parent  # 获取脚本所在目录


def get_model(num_classes):
    # Load a pre-trained model
    model = fasterrcnn_resnet50_fpn(weights=None)
    
    # Get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    
    # Replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    
    return model

def load_image(image_path):
    """Load and preprocess image for Faster R-CNN"""
    image = Image.open(image_path).convert("RGB")
    transform = T.Compose([
        T.ToTensor(),
    ])
    return image, transform(image)

def detect_insects(model, image_tensor, image_pil, class_names, device, confidence_threshold=0.5):
    """Detect insects in an image using Faster R-CNN model"""
    model.eval()
    with torch.no_grad():
        image_tensor = image_tensor.to(device)
        predictions = model([image_tensor])
    
    # Get predictions
    pred_boxes = predictions[0]['boxes'].cpu().numpy()
    pred_scores = predictions[0]['scores'].cpu().numpy()
    pred_labels = predictions[0]['labels'].cpu().numpy()
    
    # Filter predictions by confidence threshold
    keep = pred_scores >= confidence_threshold
    boxes = pred_boxes[keep]
    scores = pred_scores[keep]
    labels = pred_labels[keep]
    
    # Draw bounding boxes on image
    draw = ImageDraw.Draw(image_pil)
    
    # Try to load a font, use default if not available
    try:
        font = ImageFont.truetype("arial.ttf", 14)
    except IOError:
        font = ImageFont.load_default()
    
    width, height = image_pil.size
    
    # Generate random colors for each class
    np.random.seed(42)  # For reproducibility
    colors = {i+1: tuple(np.random.randint(0, 256, 3).tolist()) for i in range(len(class_names))}
    
    detections = []
    
    for box, score, label in zip(boxes, scores, labels):
        # Convert box coordinates to integers
        box = box.astype(np.int32)
        x1, y1, x2, y2 = box
        
        # Ensure box is within image boundaries
        x1 = max(0, min(x1, width-1))
        y1 = max(0, min(y1, height-1))
        x2 = max(0, min(x2, width-1))
        y2 = max(0, min(y2, height-1))
        
        # Get class name
        class_name = class_names[label-1]  # Subtract 1 because class 0 is background
        
        # Get color for this class
        color = colors[label]
        
        # Draw bounding box
        draw.rectangle([(x1, y1), (x2, y2)], outline=color, width=3)
        
        # Draw label
        text = f"{class_name}: {score:.2f}"
        text_size = draw.textbbox((0, 0), text, font=font)[2:4]
        
        # Draw text background
        draw.rectangle([(x1, y1), (x1 + text_size[0], y1 + text_size[1])], fill=color)
        
        # Draw text
        draw.text((x1, y1), text, fill=(255, 255, 255), font=font)
        
        # Save detection information
        detections.append({
            'class': class_name,
            'confidence': float(score),
            'bbox': [int(x1), int(y1), int(x2), int(y2)]
        })
    
    return image_pil, detections

def main():
    # Configuration
    model_path = os.path.join(script_dir, 'model/faster_rcnn_insect.pth')
    yaml_file = os.path.join(script_dir, 'insects/yolo11_dataset/data.yaml')
    input_dir = os.path.join(script_dir,'apply/input')  # Directory with images to process
    output_dir = os.path.join(script_dir,'apply/output')  # Directory to save results
    confidence_threshold = 0.5
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Determine device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Load class names from data.yaml
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
    
    # Get list of image files
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
    image_files = []
    for ext in image_extensions:
        image_files.extend(glob.glob(os.path.join(input_dir, ext)))
        image_files.extend(glob.glob(os.path.join(input_dir, ext.upper())))
    
    print(f'Found {len(image_files)} images to process')
    
    # Process each image
    for image_file in tqdm(image_files, desc="Processing images"):
        # Get base filename
        base_name = os.path.basename(image_file)
        
        # Load image
        image_pil, image_tensor = load_image(image_file)
        
        # Detect insects
        result_image, detections = detect_insects(
            model, image_tensor, image_pil, class_names, device, confidence_threshold
        )
        
        # Save result image
        output_image_path = os.path.join(output_dir, f"detected_{base_name}")
        result_image.save(output_image_path)
        
        # Save detection results as text
        output_text_path = os.path.join(output_dir, f"{os.path.splitext(base_name)[0]}_detections.txt")
        with open(output_text_path, 'w') as f:
            f.write(f"Detections for {base_name}:\n")
            f.write(f"Total insects detected: {len(detections)}\n\n")
            
            for i, detection in enumerate(detections):
                f.write(f"Detection {i+1}:\n")
                f.write(f"  Class: {detection['class']}\n")
                f.write(f"  Confidence: {detection['confidence']:.4f}\n")
                f.write(f"  Bounding Box: [x1={detection['bbox'][0]}, y1={detection['bbox'][1]}, "
                       f"x2={detection['bbox'][2]}, y2={detection['bbox'][3]}]\n\n")
    
    print(f'Processing complete. Results saved to {output_dir}')

if __name__ == '__main__':
    main()