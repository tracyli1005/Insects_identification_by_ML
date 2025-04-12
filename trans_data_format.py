import os
import xml.etree.ElementTree as ET
import shutil
from pathlib import Path
import glob

def create_folder(folder_path):
    """Create folder if it doesn't exist"""
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print(f"Created folder: {folder_path}")

def get_classes_from_xml_files(xml_dirs):
    """Extract unique class names from all XML files"""
    classes = set()
    for xml_dir in xml_dirs:
        for xml_file in glob.glob(os.path.join(xml_dir, "*.xml")):
            tree = ET.parse(xml_file)
            root = tree.getroot()
            for obj in root.findall("object"):
                class_name = obj.find("name").text
                classes.add(class_name)
    return sorted(list(classes))

def convert_xml_to_yolo(xml_path, output_path, class_dict, image_width, image_height):
    """Convert a single XML file to YOLO format txt file"""
    tree = ET.parse(xml_path)
    root = tree.getroot()
    
    with open(output_path, 'w') as f:
        for obj in root.findall('object'):
            class_name = obj.find('name').text
            if class_name not in class_dict:
                print(f"Warning: Class {class_name} not in class list, skipping")
                continue
                
            class_id = class_dict[class_name]
            
            # Get bounding box coordinates
            bbox = obj.find('bndbox')
            xmin = float(bbox.find('xmin').text)
            ymin = float(bbox.find('ymin').text)
            xmax = float(bbox.find('xmax').text)
            ymax = float(bbox.find('ymax').text)
            
            # Convert to YOLO format (x_center, y_center, width, height), normalized to 0-1
            x_center = ((xmin + xmax) / 2) / image_width
            y_center = ((ymin + ymax) / 2) / image_height
            width = (xmax - xmin) / image_width
            height = (ymax - ymin) / image_height
            
            # Write to file
            f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")

def process_dataset(xml_dir, images_dir, output_images_dir, output_labels_dir, class_dict):
    """Process dataset (train or validation)"""
    # Create directories
    create_folder(output_images_dir)
    create_folder(output_labels_dir)
    
    # Process all XML files
    xml_files = glob.glob(os.path.join(xml_dir, "*.xml"))
    processed_count = 0
    
    for xml_file in xml_files:
        # Extract filename (without extension)
        base_name = os.path.basename(xml_file)
        file_name_without_ext = os.path.splitext(base_name)[0]
        
        # Parse XML to get image size and filename
        tree = ET.parse(xml_file)
        root = tree.getroot()
        
        # Get image filename
        image_filename = root.find("filename").text
        
        # Find possible image extensions
        image_path = ""
        for ext in ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']:
            possible_path = os.path.join(images_dir, file_name_without_ext + ext)
            if os.path.exists(possible_path):
                image_path = possible_path
                break
                
            # If not found by filename, try using filename field from XML
            possible_path = os.path.join(images_dir, image_filename)
            if os.path.exists(possible_path):
                image_path = possible_path
                break
        
        if not image_path:
            print(f"Warning: Could not find image file for {xml_file}, skipping")
            continue
        
        # Get image dimensions
        size_elem = root.find("size")
        image_width = int(size_elem.find("width").text)
        image_height = int(size_elem.find("height").text)
        
        # Create output label file path
        output_label_path = os.path.join(output_labels_dir, file_name_without_ext + ".txt")
        
        # Convert XML to YOLO format
        convert_xml_to_yolo(xml_file, output_label_path, class_dict, image_width, image_height)
        
        # Copy image file to output directory
        image_ext = os.path.splitext(image_path)[1]
        output_image_path = os.path.join(output_images_dir, file_name_without_ext + image_ext)
        shutil.copy2(image_path, output_image_path)
        
        processed_count += 1
    
    return processed_count

def main():
    # Define paths
    script_dir = Path(__file__).absolute().parent  # Get script directory
    train_xml_dir = os.path.join(script_dir, 'insects/train/annotations/xmls')
    val_xml_dir = os.path.join(script_dir, 'insects/val/annotations/xmls')
    train_images_dir = os.path.join(script_dir, 'insects/train/images')
    val_images_dir = os.path.join(script_dir, 'insects/val/images')
    
    # Create output directory
    output_dir = os.path.join(script_dir, "insects/yolo11_dataset")
    create_folder(output_dir)
    
    # Create required directory structure
    output_images_dir = os.path.join(output_dir, "images")
    output_labels_dir = os.path.join(output_dir, "labels")
    
    output_train_images_dir = os.path.join(output_images_dir, "train")
    output_val_images_dir = os.path.join(output_images_dir, "val")
    output_train_labels_dir = os.path.join(output_labels_dir, "train")
    output_val_labels_dir = os.path.join(output_labels_dir, "val")
    
    # Get all classes
    xml_dirs = [train_xml_dir, val_xml_dir]
    classes = get_classes_from_xml_files(xml_dirs)
    print(f"Found classes: {classes}")
    
    # Create class to ID mapping
    class_dict = {class_name: i for i, class_name in enumerate(classes)}
    
    # Process training and validation sets
    train_count = process_dataset(
        train_xml_dir, 
        train_images_dir, 
        output_train_images_dir, 
        output_train_labels_dir, 
        class_dict
    )
    
    val_count = process_dataset(
        val_xml_dir, 
        val_images_dir, 
        output_val_images_dir, 
        output_val_labels_dir, 
        class_dict
    )
    
    # Create data.yaml file (YOLOv11 format)
    with open(os.path.join(output_dir, "data.yaml"), 'w') as f:
        f.write(f"path: {output_dir}\n")
        f.write(f"train: images/train\n")
        f.write(f"val: images/val\n")
        f.write(f"nc: {len(classes)}\n")
        f.write(f"names: {str(classes)}\n")
    
    print(f"\nConversion complete! Output directory: {output_dir}")
    print(f"Training images: {train_count}")
    print(f"Validation images: {val_count}")
    print(f"Number of classes: {len(classes)}")
    print(f"\nTo use with YOLOv11:")
    print(f"yolo train model=yolov11n.pt data={os.path.join(output_dir, 'data.yaml')}")

if __name__ == "__main__":
    main()