import os
import shutil
import yaml
from pathlib import Path
import glob

def convert_and_merge_yolo_datasets(
    source_dir,         # Source dataset directory (first format)
    target_dir,         # Target dataset directory (second format)
    merge_classes=True  # Whether to merge classes
):
    # Create target directory structure (if not exists)
    os.makedirs(os.path.join(target_dir, 'images', 'train'), exist_ok=True)
    os.makedirs(os.path.join(target_dir, 'images', 'val'), exist_ok=True)
    os.makedirs(os.path.join(target_dir, 'labels', 'train'), exist_ok=True)
    os.makedirs(os.path.join(target_dir, 'labels', 'val'), exist_ok=True)
    
    # Read source dataset's data.yaml
    with open(os.path.join(source_dir, 'data.yaml'), 'r') as f:
        source_yaml = yaml.safe_load(f)
    
    # Read classes from source dataset
    source_classes = source_yaml['names']
    print(f"Source dataset contains {len(source_classes)} classes")
    
    # Read target dataset's data.yaml (if exists)
    target_yaml_path = os.path.join(target_dir, 'data.yaml')
    target_classes = []
    if os.path.exists(target_yaml_path):
        with open(target_yaml_path, 'r') as f:
            target_yaml = yaml.safe_load(f)
            target_classes = target_yaml['names']
        print(f"Target dataset contains {len(target_classes)} classes")
    
    # Build class mapping (source dataset class index -> target dataset class index)
    class_mapping = {}
    if merge_classes:
        # Merge class lists
        merged_classes = target_classes.copy()
        for cls in source_classes:
            if cls not in merged_classes:
                merged_classes.append(cls)
        
        # Create class mapping
        for i, cls in enumerate(source_classes):
            class_mapping[i] = merged_classes.index(cls)
        
        # Update target classes
        target_classes = merged_classes
    else:
        # Use source dataset classes directly
        target_classes = source_classes
        for i in range(len(source_classes)):
            class_mapping[i] = i
    
    print(f"Merged dataset will contain {len(target_classes)} classes")
    
    # Read train and validation file lists
    train_files = []
    with open(os.path.join(source_dir, source_yaml['train']), 'r') as f:
        train_files = [line.strip() for line in f.readlines()]
    
    val_files = []
    with open(os.path.join(source_dir, source_yaml['val']), 'r') as f:
        val_files = [line.strip() for line in f.readlines()]
    
    # Process training set
    print(f"Processing {len(train_files)} training images...")
    process_file_list(source_dir, target_dir, train_files, 'train', class_mapping)
    
    # Process validation set
    print(f"Processing {len(val_files)} validation images...")
    process_file_list(source_dir, target_dir, val_files, 'val', class_mapping)
    
    # Update target data.yaml
    target_yaml = {
        'path': os.path.abspath(target_dir),
        'train': 'images/train',
        'val': 'images/val',
        'nc': len(target_classes),
        'names': target_classes
    }
    
    with open(target_yaml_path, 'w') as f:
        yaml.dump(target_yaml, f, default_flow_style=False, sort_keys=False)
    
    print(f"Conversion completed! Dataset saved to {target_dir}")
    print(f"Contains {len(target_classes)} classes: {target_classes}")

def process_file_list(source_dir, target_dir, file_list, split, class_mapping):
    """Process file list and copy to target directory"""
    for file_path in file_list:
        # Get filename (without path)
        if file_path.startswith('images/'):
            file_path = file_path[7:]  # Remove 'images/' prefix
        file_name = os.path.basename(file_path)
        base_name = os.path.splitext(file_name)[0]
        
        # Source file paths
        source_img_path = os.path.join(source_dir, 'images', file_name)
        source_label_path = os.path.join(source_dir, 'labels', f"{base_name}.txt")
        
        # Target file paths
        target_img_path = os.path.join(target_dir, 'images', split, file_name)
        target_label_path = os.path.join(target_dir, 'labels', split, f"{base_name}.txt")
        
        # Copy image file
        if os.path.exists(source_img_path):
            shutil.copy2(source_img_path, target_img_path)
        else:
            print(f"Warning: Image file not found {source_img_path}")
            continue
        
        # Process label file (convert class indices)
        if os.path.exists(source_label_path):
            with open(source_label_path, 'r') as f_in:
                lines = f_in.readlines()
            
            with open(target_label_path, 'w') as f_out:
                for line in lines:
                    parts = line.strip().split()
                    if len(parts) >= 5:  # Ensure enough elements
                        old_class_id = int(parts[0])
                        new_class_id = class_mapping.get(old_class_id, old_class_id)
                        parts[0] = str(new_class_id)
                        f_out.write(' '.join(parts) + '\n')
        else:
            print(f"Warning: Label file not found {source_label_path}")

# Usage example
if __name__ == "__main__":
    script_dir = Path(__file__).absolute().parent  # Get script directory

    source_directory = os.path.join(script_dir, 'insects/yolo_data01')
    target_directory = os.path.join(script_dir, 'insects/yolo11_dataset')  
    
    convert_and_merge_yolo_datasets(source_directory, target_directory)