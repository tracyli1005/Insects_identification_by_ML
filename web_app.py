import os
import sys
import glob
import shutil
import uuid
from flask import Flask, request, render_template, jsonify, send_from_directory, url_for, redirect
import faster_rcnn_apply
import yolo11_apply

app = Flask(__name__)

# Ensure the upload directories exist
script_dir = os.path.dirname(os.path.abspath(__file__))
INPUT_DIR = os.path.join(script_dir, 'apply/input')
OUTPUT_DIR = os.path.join(script_dir, 'apply/output')
os.makedirs(INPUT_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Maximum allowed upload size (5MB)
app.config['MAX_CONTENT_LENGTH'] = 5 * 1024 * 1024
# Allowed file extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def clear_directory(directory):
    """Clear all files in a directory"""
    for file in os.listdir(directory):
        file_path = os.path.join(directory, file)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
        except Exception as e:
            print(f"Error deleting {file_path}: {e}")

@app.route('/')
def index():
    """Render the main page"""
    return render_template('index.html')

@app.route('/detect', methods=['POST'])
def detect():
    """Handle the detection request"""
    # Check if a file was uploaded
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if not allowed_file(file.filename):
        return jsonify({'error': f'File type not allowed. Allowed types: {", ".join(ALLOWED_EXTENSIONS)}'}), 400
    
    # Get the selected model
    model = request.form.get('model', 'faster_rcnn')
    if model not in ['faster_rcnn', 'yolov11']:
        return jsonify({'error': 'Invalid model selection'}), 400
    
    # Clear input and output directories
    clear_directory(INPUT_DIR)
    clear_directory(OUTPUT_DIR)
    
    # Generate unique filename to avoid collisions
    unique_filename = f"{uuid.uuid4().hex}_{file.filename}"
    file_path = os.path.join(INPUT_DIR, unique_filename)
    file.save(file_path)
    
    # Run the selected detection model
    try:
        if model == 'faster_rcnn':
            # Run Faster R-CNN detection
            faster_rcnn_apply.main()
            
            # Get the detection results
            detection_files = glob.glob(os.path.join(OUTPUT_DIR, '*_detections.txt'))
            detected_images = glob.glob(os.path.join(OUTPUT_DIR, 'detected_*.jpg')) + \
                              glob.glob(os.path.join(OUTPUT_DIR, 'detected_*.jpeg')) + \
                              glob.glob(os.path.join(OUTPUT_DIR, 'detected_*.png')) + \
                              glob.glob(os.path.join(OUTPUT_DIR, 'detected_*.bmp'))
            
            if not detection_files or not detected_images:
                return jsonify({'error': 'Detection failed or no results generated'}), 500
            
            # Read detection results
            with open(detection_files[0], 'r') as f:
                detection_text = f.read()
            
            # Get the path to the output image relative to the OUTPUT_DIR
            image_filename = os.path.basename(detected_images[0])
            image_url = url_for('serve_result', filename=image_filename)
            
            return jsonify({
                'success': True,
                'model': 'Faster R-CNN',
                'detection_text': detection_text,
                'image_url': image_url,
                'original_filename': file.filename
            })
            
        else:  # yolov11
            # Run YOLOv11 detection
            results = yolo11_apply.main()
            
            # Get the detection results
            detected_images = glob.glob(os.path.join(OUTPUT_DIR, '*.jpg')) + \
                              glob.glob(os.path.join(OUTPUT_DIR, '*.jpeg')) + \
                              glob.glob(os.path.join(OUTPUT_DIR, '*.png')) + \
                              glob.glob(os.path.join(OUTPUT_DIR, '*.bmp'))
            
            if not detected_images:
                return jsonify({'error': 'Detection failed or no results generated'}), 500
            
            # Format detection text from results
            detection_text = "YOLOv11 Detection Results:\n"
            
            if isinstance(results, dict) and 'detections' in results:
                # Single image results
                detection_text += f"Total insects detected: {len(results['detections'])}\n\n"
                for i, detection in enumerate(results['detections']):
                    detection_text += f"Detection {i+1}:\n"
                    detection_text += f"  Class: {detection['class_name']}\n"
                    detection_text += f"  Confidence: {detection['confidence']:.4f}\n"
                    detection_text += f"  Bounding Box: {detection['bbox']}\n\n"
            elif isinstance(results, list):
                # Multiple image results
                total_insects = sum(len(img_result['detections']) for img_result in results if 'detections' in img_result)
                detection_text += f"Total images processed: {len(results)}\n"
                detection_text += f"Total insects detected: {total_insects}\n\n"
                
                for img_idx, img_result in enumerate(results):
                    if 'detections' in img_result:
                        detection_text += f"Image {img_idx+1}:\n"
                        detection_text += f"  File: {img_result.get('filename', 'unknown')}\n"
                        detection_text += f"  Insects detected: {len(img_result['detections'])}\n"
                        
                        for i, detection in enumerate(img_result['detections']):
                            detection_text += f"  Detection {i+1}:\n"
                            detection_text += f"    Class: {detection['class_name']}\n"
                            detection_text += f"    Confidence: {detection['confidence']:.4f}\n"
                            detection_text += f"    Bounding Box: {detection['bbox']}\n"
                        detection_text += "\n"
            
            # Get the path to the output image relative to the OUTPUT_DIR
            image_filename = os.path.basename(detected_images[0])
            image_url = url_for('serve_result', filename=image_filename)
            
            return jsonify({
                'success': True,
                'model': 'YOLOv11',
                'detection_text': detection_text,
                'image_url': image_url,
                'original_filename': file.filename
            })
            
    except Exception as e:
        return jsonify({'error': f'Error during detection: {str(e)}'}), 500

@app.route('/results/<filename>')
def serve_result(filename):
    """Serve the result images"""
    return send_from_directory(OUTPUT_DIR, filename)

@app.route('/api/detect', methods=['POST'])
def api_detect():
    """API endpoint for programmatic access"""
    # Check if a file was uploaded
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if not allowed_file(file.filename):
        return jsonify({'error': f'File type not allowed. Allowed types: {", ".join(ALLOWED_EXTENSIONS)}'}), 400
    
    # Get the selected model
    model = request.form.get('model', 'faster_rcnn')
    if model not in ['faster_rcnn', 'yolov11']:
        return jsonify({'error': 'Invalid model selection'}), 400
    
    # Clear input and output directories
    clear_directory(INPUT_DIR)
    clear_directory(OUTPUT_DIR)
    
    # Generate unique filename to avoid collisions
    unique_filename = f"{uuid.uuid4().hex}_{file.filename}"
    file_path = os.path.join(INPUT_DIR, unique_filename)
    file.save(file_path)
    
    # Run the selected detection model
    try:
        results = {}
        
        if model == 'faster_rcnn':
            # Run Faster R-CNN detection
            faster_rcnn_apply.main()
            
            # Get the detection results
            detection_files = glob.glob(os.path.join(OUTPUT_DIR, '*_detections.txt'))
            detected_images = glob.glob(os.path.join(OUTPUT_DIR, 'detected_*.jpg')) + \
                              glob.glob(os.path.join(OUTPUT_DIR, 'detected_*.jpeg')) + \
                              glob.glob(os.path.join(OUTPUT_DIR, 'detected_*.png')) + \
                              glob.glob(os.path.join(OUTPUT_DIR, 'detected_*.bmp'))
            
            if not detection_files or not detected_images:
                return jsonify({'error': 'Detection failed or no results generated'}), 500
            
            # Read detection results
            with open(detection_files[0], 'r') as f:
                detection_text = f.read()
            
            # Parse detections from text
            detections = []
            lines = detection_text.split('\n')
            total_insects = 0
            
            for i, line in enumerate(lines):
                if line.startswith("Total insects detected:"):
                    total_insects = int(line.split(":")[1].strip())
                
                if line.startswith("Detection "):
                    insect = {}
                    # Extract class
                    if i+1 < len(lines) and lines[i+1].strip().startswith("Class:"):
                        insect['class'] = lines[i+1].split(":")[1].strip()
                    
                    # Extract confidence
                    if i+2 < len(lines) and lines[i+2].strip().startswith("Confidence:"):
                        insect['confidence'] = float(lines[i+2].split(":")[1].strip())
                    
                    # Extract bounding box
                    if i+3 < len(lines) and lines[i+3].strip().startswith("Bounding Box:"):
                        bbox_str = lines[i+3].split("[")[1].split("]")[0]
                        bbox_parts = bbox_str.split(",")
                        bbox = []
                        for part in bbox_parts:
                            key_value = part.strip().split("=")
                            if len(key_value) == 2:
                                bbox.append(float(key_value[1]))
                        insect['bbox'] = bbox
                    
                    if insect:
                        detections.append(insect)
            
            # Prepare the result
            results = {
                'success': True,
                'model': 'Faster R-CNN',
                'total_insects': total_insects,
                'detections': detections,
                'image_url': request.host_url.rstrip('/') + url_for('serve_result', filename=os.path.basename(detected_images[0])),
                'original_filename': file.filename
            }
            
        else:  # yolov11
            # Run YOLOv11 detection
            yolo_results = yolo11_apply.main()
            
            # Get the detection results
            detected_images = glob.glob(os.path.join(OUTPUT_DIR, '*.jpg')) + \
                              glob.glob(os.path.join(OUTPUT_DIR, '*.jpeg')) + \
                              glob.glob(os.path.join(OUTPUT_DIR, '*.png')) + \
                              glob.glob(os.path.join(OUTPUT_DIR, '*.bmp'))
            
            if not detected_images:
                return jsonify({'error': 'Detection failed or no results generated'}), 500
            
            # Format the API response based on the YOLO results
            if isinstance(yolo_results, dict) and 'detections' in yolo_results:
                # Single image results
                results = {
                    'success': True,
                    'model': 'YOLOv11',
                    'total_insects': len(yolo_results['detections']),
                    'detections': yolo_results['detections'],
                    'image_url': request.host_url.rstrip('/') + url_for('serve_result', filename=os.path.basename(detected_images[0])),
                    'original_filename': file.filename
                }
            elif isinstance(yolo_results, list):
                # Multiple image results (we'll simplify to the first image for API)
                if yolo_results and 'detections' in yolo_results[0]:
                    results = {
                        'success': True,
                        'model': 'YOLOv11',
                        'total_insects': len(yolo_results[0]['detections']),
                        'detections': yolo_results[0]['detections'],
                        'image_url': request.host_url.rstrip('/') + url_for('serve_result', filename=os.path.basename(detected_images[0])),
                        'original_filename': file.filename
                    }
                else:
                    results = {
                        'success': True,
                        'model': 'YOLOv11',
                        'total_insects': 0,
                        'detections': [],
                        'image_url': request.host_url.rstrip('/') + url_for('serve_result', filename=os.path.basename(detected_images[0])),
                        'original_filename': file.filename
                    }
        
        return jsonify(results)
            
    except Exception as e:
        return jsonify({'error': f'Error during detection: {str(e)}'}), 500

# Create templates directory and HTML template
def create_templates():
    templates_dir = os.path.join(script_dir, 'templates')
    os.makedirs(templates_dir, exist_ok=True)
    
    # Create index.html
    # Create index.html
    index_html = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Insect Detection</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
        }
        h1 {
            color: #2c3e50;
            text-align: center;
        }
        .container {
            display: flex;
            flex-direction: column;
            gap: 20px;
        }
        .upload-section {
            background-color: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .results-section {
            display: none;
            background-color: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .flex-container {
            display: flex;
            gap: 20px;
        }
        .image-container {
            flex: 1;
        }
        .text-container {
            flex: 1;
        }
        .result-image {
            max-width: 100%;
            border: 1px solid #ddd;
            border-radius: 4px;
        }
        .detection-text {
            font-family: monospace;
            white-space: pre-wrap;
            background-color: #f8f9fa;
            padding: 15px;
            border-radius: 4px;
            overflow-y: auto;
            max-height: 500px;
        }
        .form-group {
            margin-bottom: 15px;
        }
        label {
            display: block;
            margin-bottom: 5px;
            font-weight: bold;
        }
        .buttons {
            display: flex;
            gap: 10px;
        }
        button, .file-input-label {
            background-color: #3498db;
            color: white;
            border: none;
            padding: 10px 15px;
            border-radius: 4px;
            cursor: pointer;
            font-size: 14px;
        }
        button:hover, .file-input-label:hover {
            background-color: #2980b9;
        }
        .file-input {
            display: none;
        }
        .selected-file {
            margin-top: 10px;
            font-style: italic;
        }
        .loading {
            display: none;
            text-align: center;
            margin: 20px 0;
        }
        .spinner {
            border: 4px solid #f3f3f3;
            border-top: 4px solid #3498db;
            border-radius: 50%;
            width: 30px;
            height: 30px;
            animation: spin 2s linear infinite;
            margin: 0 auto;
        }
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        .error-message {
            color: #e74c3c;
            padding: 10px;
            background-color: #fadbd8;
            border-radius: 4px;
            margin-top: 10px;
            display: none;
        }
        .try-again-btn {
            background-color: #2ecc71;
        }
        .try-again-btn:hover {
            background-color: #27ae60;
        }
    </style>
</head>
<body>
    <h1>Insect Detection System</h1>
    
    <div class="container">
        <div class="upload-section" id="upload-section">
            <h2>Upload an Image</h2>
            <div class="form-group">
                <label for="model-select">Select Detection Model:</label>
                <select id="model-select">
                    <option value="faster_rcnn">Faster R-CNN</option>
                    <option value="yolov11">YOLOv11</option>
                </select>
            </div>
            
            <div class="form-group">
                <label class="file-input-label" for="file-input">Choose Image</label>
                <input type="file" id="file-input" class="file-input" accept=".jpg, .jpeg, .png, .bmp">
                <div class="selected-file" id="selected-file">No file selected</div>
            </div>
            
            <div class="buttons">
                <button id="detect-btn" disabled>Detect Insects</button>
            </div>
            
            <div class="error-message" id="error-message"></div>
        </div>
        
        <div class="loading" id="loading">
            <div class="spinner"></div>
            <p>Processing image... Please wait.</p>
        </div>
        
        <div class="results-section" id="results-section">
            <h2>Detection Results</h2>
            <p id="model-used"></p>
            
            <div class="flex-container">
                <div class="image-container">
                    <h3>Detected Image</h3>
                    <img id="result-image" class="result-image" src="" alt="Detection Result">
                </div>
                
                <div class="text-container">
                    <h3>Detection Details</h3>
                    <div class="detection-text" id="detection-text"></div>
                </div>
            </div>
            
            <div class="buttons" style="margin-top: 20px;">
                <button class="try-again-btn" id="try-again-btn">Try Another Image</button>
            </div>
        </div>
    </div>
    
    <script>
        // DOM elements
        const fileInput = document.getElementById('file-input');
        const selectedFileText = document.getElementById('selected-file');
        const detectBtn = document.getElementById('detect-btn');
        const modelSelect = document.getElementById('model-select');
        const uploadSection = document.getElementById('upload-section');
        const loadingSection = document.getElementById('loading');
        const resultsSection = document.getElementById('results-section');
        const resultImage = document.getElementById('result-image');
        const detectionText = document.getElementById('detection-text');
        const modelUsed = document.getElementById('model-used');
        const tryAgainBtn = document.getElementById('try-again-btn');
        const errorMessage = document.getElementById('error-message');
        
        // Event listeners
        fileInput.addEventListener('change', function() {
            if (this.files.length > 0) {
                selectedFileText.textContent = `Selected file: ${this.files[0].name}`;
                detectBtn.disabled = false;
            } else {
                selectedFileText.textContent = 'No file selected';
                detectBtn.disabled = true;
            }
            errorMessage.style.display = 'none';
        });
        
        detectBtn.addEventListener('click', detectInsects);
        tryAgainBtn.addEventListener('click', resetForm);
        
        // Functions
        function detectInsects() {
            const file = fileInput.files[0];
            if (!file) {
                showError('Please select a file first.');
                return;
            }
            
            // Validate file type
            const validTypes = ['image/jpeg', 'image/jpg', 'image/png', 'image/bmp'];
            if (!validTypes.includes(file.type)) {
                showError('Invalid file type. Please select a JPG, PNG, or BMP image.');
                return;
            }
            
            // Validate file size (5MB max)
            if (file.size > 5 * 1024 * 1024) {
                showError('File too large. Maximum size is 5MB.');
                return;
            }
            
            // Show loading state
            uploadSection.style.display = 'none';
            loadingSection.style.display = 'block';
            resultsSection.style.display = 'none';
            
            // Create form data
            const formData = new FormData();
            formData.append('file', file);
            formData.append('model', modelSelect.value);
            
            // Send request to server
            fetch('/detect', {
                method: 'POST',
                body: formData
            })
            .then(response => response.json())
            .then(data => {
                // Hide loading
                loadingSection.style.display = 'none';
                
                if (data.error) {
                    uploadSection.style.display = 'block';
                    showError(data.error);
                    return;
                }
                
                // Show results
                resultsSection.style.display = 'block';
                resultImage.src = data.image_url;
                detectionText.textContent = data.detection_text;
                modelUsed.textContent = `Model used: ${data.model}`;
            })
            .catch(error => {
                loadingSection.style.display = 'none';
                uploadSection.style.display = 'block';
                showError('An error occurred during processing. Please try again.');
                console.error('Error:', error);
            });
        }
        
        function resetForm() {
            // Reset form and show upload section
            fileInput.value = '';
            selectedFileText.textContent = 'No file selected';
            detectBtn.disabled = true;
            errorMessage.style.display = 'none';
            
            uploadSection.style.display = 'block';
            resultsSection.style.display = 'none';
        }
        
        function showError(message) {
            errorMessage.textContent = message;
            errorMessage.style.display = 'block';
        }
    </script>
</body>
</html>
'''
    
    with open(os.path.join(templates_dir, 'index.html'), 'w') as f:
        f.write(index_html)

if __name__ == '__main__':
    create_templates()
    app.run(host='0.0.0.0', port=5001, debug=True)