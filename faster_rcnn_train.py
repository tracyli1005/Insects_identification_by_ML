import os
import torch
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torch.utils.data import Dataset, DataLoader
import yaml
from PIL import Image
import numpy as np
from torchvision import transforms as T
from torchvision.transforms import functional as F

class InsectDataset(Dataset):
    def __init__(self, root_dir, split='train', transforms=None):
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
        
        # Convert PIL image to tensor
        img = self.transforms(img)
            
        return img, target

def get_model(num_classes):
    # Load a pre-trained model with weights parameter instead of pretrained
    model = fasterrcnn_resnet50_fpn(weights='DEFAULT')
    
    # Get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    
    # Replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    
    return model

def train_model(model, train_loader, device, num_epochs=10):
    # Move model to device
    model.to(device)
    
    # Optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
    
    # Learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)
    
    # Training loop
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        
        print(f"Starting epoch {epoch+1}/{num_epochs}")
        
        for i, (images, targets) in enumerate(train_loader):
            # Move images to device
            images = list(image.to(device) for image in images)
            
            # Move targets to device
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            
            # Forward pass
            loss_dict = model(images, targets)
            
            # Calculate total loss
            losses = sum(loss for loss in loss_dict.values())
            
            # Backward pass and optimize
            optimizer.zero_grad()
            losses.backward()
            optimizer.step()
            
            total_loss += losses.item()
            
            # Print progress every 50 batches
            if (i + 1) % 50 == 0:
                print(f"  Batch {i+1}/{len(train_loader)}, Loss: {losses.item():.4f}")
            
        # Update learning rate
        lr_scheduler.step()
        
        # Print epoch summary
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{num_epochs} completed, Avg Loss: {avg_loss:.4f}")
        
    return model



def main():
    # Configuration
    dataset_path = 'insects/yolo11_dataset'
    num_epochs = 1  # Set to 10 for full training
    batch_size = 2   # Increase based on your GPU memory
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f'Using device: {device}')
    
    # Create datasets and dataloaders
    train_dataset = InsectDataset(dataset_path, split='train')
    val_dataset = InsectDataset(dataset_path, split='val')
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                             collate_fn=lambda x: tuple(zip(*x)))
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False,
                           collate_fn=lambda x: tuple(zip(*x)))
    
    # Get number of classes from data.yaml
    yaml_file = os.path.join(dataset_path, 'data.yaml')
    print(f'Loading class definitions from: {yaml_file}')
    
    with open(yaml_file, 'r') as f:
        num_classes = len(yaml.safe_load(f)['names']) + 1  # +1 for background
    
    print(f'Number of classes: {num_classes}')
    
    # Get model
    model = get_model(num_classes)
    
    # Train model
    model = train_model(model, train_loader, device, num_epochs)
    
    # Save model
    os.makedirs('model', exist_ok=True)
    model_path = 'model/faster_rcnn_insect.pth'
    torch.save(model.state_dict(), model_path)
    print(f'Model saved to {model_path}')

if __name__ == '__main__':
    main()