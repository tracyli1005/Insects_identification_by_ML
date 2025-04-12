import os
import torch
from ultralytics import YOLO
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import yaml
from tqdm import tqdm
import time
import json
from pathlib import Path

script_dir = Path(__file__).absolute().parent  # Get script directory
# Global configuration
DEFAULT_MODEL_PATH = os.path.join(script_dir, 'model/yolo11_insects_final.pt')
DEFAULT_CONF_THRESHOLD = 0.25
DEFAULT_OUTPUT_DIR = os.path.join(script_dir, 'apply/output')

class Yolo11Detector:
    """YOLO11 insect detector class"""
    
    def __init__(self, model_path=None, conf_threshold=DEFAULT_CONF_THRESHOLD):
        """Initialize detector"""
        # Use default path if model_path is not specified
        if model_path is None:
            model_path = DEFAULT_MODEL_PATH
            # If default model doesn't exist, try using best model saved during training
            if not os.path.exists(model_path):
                alt_model_path = 'model/yolo11_insects/weights/best.pt'
                if os.path.exists(alt_model_path):
                    print(f"Default model {model_path} not found, using alternative: {alt_model_path}")
                    model_path = alt_model_path
                else:
                    raise FileNotFoundError("Trained model not found, please run training script first or specify correct model path")
        
        # Load model
        self.model = self._load_model(model_path)
        self.conf_threshold = conf_threshold
        
        # Load class names
        try:
            with open(os.path.join(script_dir, 'insects/yolo11_dataset/data.yaml'), 'r') as f:
                data_config = yaml.safe_load(f)
                self.class_names = data_config['names']
        except Exception as e:
            print(f"Error loading class names: {e}")
            print("Using default class names")
            self.class_names = ['Boerner', 'Leconte', 'Linnaeus', 'acuminatus', 'armandi', 'coleoptera', 'linnaeus']
    
    def _load_model(self, model_path):
        """Load trained YOLO11 model"""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Error: Model file {model_path} not found")
        
        try:
            model = YOLO(model_path)
            print(f"Successfully loaded model: {model_path}")
            return model
        except Exception as e:
            raise RuntimeError(f"Error loading model: {e}")
    
    def process_image(self, image_path, output_dir=DEFAULT_OUTPUT_DIR):
        """Process single image and save results"""
        # Check if image exists
        if not os.path.exists(image_path):
            print(f"Image file {image_path} not found")
            return None
        
        try:
            # Make predictions using model
            results = self.model.predict(image_path, conf=self.conf_threshold)
            result = results[0]  # Get first result (since we're processing single image)
            
            # Get original image
            img = Image.open(image_path)
            
            # Create drawing object
            draw = ImageDraw.Draw(img)
            
            # Try to load font, fallback to default if failed
            try:
                # Try to load common font
                font = ImageFont.truetype("arial.ttf", 20)
            except:
                # Fallback to default font
                font = ImageFont.load_default()
            
            # Detection information
            detection_info = []
            
            # Process bounding boxes
            if result.boxes is not None and len(result.boxes) > 0:
                boxes = result.boxes
                for i, box in enumerate(boxes):
                    # Get bounding box coordinates
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    
                    # Get class and confidence
                    cls_id = int(box.cls.item())
                    conf = float(box.conf.item())
                    cls_name = self.class_names[cls_id]
                    
                    # Select different color for each class
                    color = tuple(np.random.randint(0, 255, 3).tolist())
                    color_str = "#{:02x}{:02x}{:02x}".format(*color)
                    
                    # Draw bounding box
                    draw.rectangle([(x1, y1), (x2, y2)], outline=color_str, width=3)
                    
                    # Draw class and confidence label
                    label = f"{cls_name}: {conf:.2f}"
                    text_size = draw.textbbox((0, 0), label, font=font)
                    text_width = text_size[2] - text_size[0]
                    text_height = text_size[3] - text_size[1]
                    
                    # Draw label background
                    draw.rectangle(
                        [(x1, y1 - text_height - 4), (x1 + text_width + 4, y1)],
                        fill=color_str
                    )
                    
                    # Draw label text
                    draw.text((x1 + 2, y1 - text_height - 2), label, fill="white", font=font)
                    
                    # Save detection information
                    detection_info.append({
                        'class_id': cls_id,
                        'class_name': cls_name,
                        'confidence': conf,
                        'bbox': [x1, y1, x2, y2]
                    })
            
            # Create output directory
            os.makedirs(output_dir, exist_ok=True)
            
            # Generate output filenames
            base_filename = os.path.splitext(os.path.basename(image_path))[0]
            output_image_path = os.path.join(output_dir, f"{base_filename}_detected.jpg")
            output_json_path = os.path.join(output_dir, f"{base_filename}_result.json")
            
            # Save annotated image
            img.save(output_image_path)
            
            # Save detection results as JSON
            result_data = {
                'image_path': image_path,
                'processed_time': time.strftime("%Y-%m-%d %H:%M:%S"),
                'detections': detection_info
            }
            
            with open(output_json_path, 'w') as f:
                json.dump(result_data, f, indent=4)
            
            print(f"Processed image: {image_path}")
            print(f"Detected {len(detection_info)} insect objects")
            print(f"Results saved to: {output_image_path} and {output_json_path}")
            
            return {
                'output_image': output_image_path,
                'output_json': output_json_path,
                'detections': detection_info
            }
        
        except Exception as e:
            print(f"Error processing image {image_path}: {e}")
            return None
    
    def process_directory(self, input_dir, output_dir=DEFAULT_OUTPUT_DIR):
        """Process all images in directory"""
        # Check if directory exists
        if not os.path.exists(input_dir):
            print(f"Directory {input_dir} not found")
            return
        
        # Supported image formats
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff']
        
        # Get all files in directory
        files = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))]
        
        # Filter image files
        image_files = [f for f in files if os.path.splitext(f)[1].lower() in image_extensions]
        
        print(f"Found {len(image_files)} image files in directory {input_dir}")
        
        results = []
        
        # Process image files
        if image_files:
            print("Processing image files...")
            for image_file in tqdm(image_files, desc="Processing images"):
                image_path = os.path.join(input_dir, image_file)
                result = self.process_image(image_path, output_dir)
                if result:
                    results.append(result)
        
        print(f"All images processed, results saved to {output_dir}")
        return results

def detect_insects(input_path, model_path=None, conf_threshold=DEFAULT_CONF_THRESHOLD, output_dir=DEFAULT_OUTPUT_DIR):
    """
    Main function for detecting insects using YOLO11
    
    Args:
        input_path (str): Input image file path or directory path
        model_path (str, optional): YOLO11 model path, uses default if not specified
        conf_threshold (float, optional): Confidence threshold, default 0.25
        output_dir (str, optional): Output directory, default 'result'
        
    Returns:
        dict: Detection results information
    """
    try:
        # Initialize detector
        detector = Yolo11Detector(model_path, conf_threshold)
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Process based on input type
        if os.path.isdir(input_path):
            # Process all images in directory
            return detector.process_directory(input_path, output_dir)
        elif os.path.isfile(input_path):
            # Check if file is image
            file_ext = os.path.splitext(input_path)[1].lower()
            if file_ext in ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff']:
                # Process single image
                return detector.process_image(input_path, output_dir)
            else:
                print(f"Unsupported file format: {file_ext}")
                return None
        else:
            print(f"Input path {input_path} does not exist or is invalid")
            return None
    
    except Exception as e:
        print(f"Error during detection: {e}")
        return None

# Example usage
def main():
    # You can set parameters directly here instead of command line
    input_path = os.path.join(script_dir, 'apply/input')
    output_dir = os.path.join(script_dir, 'apply/output')
    
    # Call detection function
    results = detect_insects(
        input_path=input_path,
        model_path='model/yolo11_insects_final.pt',  # Optional, uses default if not specified
        conf_threshold=0.3,  # Optional, uses default 0.25 if not specified
        output_dir=output_dir  # Optional, uses default 'result' if not specified
    )
    
    # Print results summary
    if results:
        if isinstance(results, dict) and 'detections' in results:
            # Single image results
            print(f"Detection complete! Found {len(results['detections'])} insect objects")
            
            # Print information for each detected insect
            for i, detection in enumerate(results['detections']):
                print(f"Insect {i+1}: {detection['class_name']} (confidence: {detection['confidence']:.2f})")
                
        elif isinstance(results, list):
            # Directory processing results
            total_images = len(results)
            print(f"Batch processing complete! Processed {total_images} images")
            
            # Count total detected insects
            total_insects = sum(len(img_result['detections']) for img_result in results if 'detections' in img_result)
            print(f"Total detected insects: {total_insects}")
    else:
        print("Detection incomplete or error occurred")

if __name__ == "__main__":
    main()