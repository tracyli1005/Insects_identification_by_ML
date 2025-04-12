import os
import torch
import yaml
from pathlib import Path
from ultralytics import YOLO

def train_yolo():
    print("Starting YOLO11 model training")

    # Check if GPU is available
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Ensure model directory exists
    os.makedirs('model', exist_ok=True)
    
    # Load YOLO11 model
    model = YOLO('model/yolo11n.pt')
    
    # Set training parameters
    script_dir = Path(__file__).absolute().parent  # Get script directory
    dataset_path = os.path.join(script_dir, 'insects/yolo11_dataset')  # Combine paths
    data_yaml = os.path.join(dataset_path, 'data.yaml')
    print(f"Using dataset config file: {data_yaml}")

    # Start training
    results = model.train(
        data=data_yaml,       # Dataset config file
        epochs=1,           # Number of epochs
        imgsz=640,            # Image size
        batch=16,             # Batch size
        device=device,        # Training device
        workers=4,            # Number of workers
        patience=20,          # Early stopping patience
        save=True,            # Save model
        project='model',      # Save directory
        name='yolo11_insects',# Experiment name
        exist_ok=True,        # Overwrite existing experiment
        pretrained=True,      # Use pretrained weights
        optimizer='Adam',     # Optimizer
        lr0=0.001,            # Initial learning rate
        weight_decay=0.0005,  # Weight decay
        verbose=True          # Show detailed info
    )
    
    # Save final model
    model.export(format='onnx')  # Export ONNX format for deployment
    final_model_path = os.path.join(script_dir, 'model/yolo11_insects_final.pt')
    model.save(final_model_path)
    
    print(f"Training completed. Final model saved at: {final_model_path}")
    print(f"Training results summary: mAP={results.box.map:.4f}")
    
    return final_model_path

if __name__ == "__main__":
    trained_model_path = train_yolo()