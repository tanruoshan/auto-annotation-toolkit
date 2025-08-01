"""
Quick Auto Annotation Script
Simple script to generate LabelMe annotations from trained models
"""

import cv2
import numpy as np
import json
import os
from pathlib import Path

def create_sample_annotation(image_path: str, detections: list = None) -> dict:
    """
    Create a LabelMe format annotation
    
    Args:
        image_path: Path to the image
        detections: List of detections [{'bbox': [x1,y1,x2,y2], 'label': 'class_name', 'confidence': 0.9}]
    """
    # Load image to get dimensions
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not load image: {image_path}")
    
    height, width = image.shape[:2]
    image_filename = os.path.basename(image_path)
    
    # Create shapes from detections
    shapes = []
    if detections:
        for detection in detections:
            x1, y1, x2, y2 = detection['bbox']
            label = detection.get('label', 'DieDefect')
            confidence = detection.get('confidence', 1.0)
            
            shape = {
                "label": label,
                "points": [
                    [float(x1), float(y1)],  # Top-left corner
                    [float(x2), float(y2)]   # Bottom-right corner
                ],
                "group_id": None,
                "description": f"confidence: {confidence:.3f}",
                "shape_type": "rectangle",
                "flags": {},
                "mask": None
            }
            shapes.append(shape)
    
    # Create LabelMe annotation structure
    annotation = {
        "version": "5.8.1",
        "flags": {},
        "shapes": shapes,
        "imagePath": image_filename,
        "imageData": None,
        "imageHeight": height,
        "imageWidth": width
    }
    
    return annotation

def process_with_yolo_model(model_path: str, image_folder: str, output_folder: str):
    """Process images with YOLO model (.pt format)"""
    try:
        from ultralytics import YOLO
    except ImportError:
        print("Please install ultralytics: pip install ultralytics")
        return
    
    # Load model
    model = YOLO(model_path)
    print(f"Loaded YOLO model: {model_path}")
    
    # Create output folder
    os.makedirs(output_folder, exist_ok=True)
    
    # Process images
    image_files = []
    for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
        image_files.extend(Path(image_folder).glob(ext))
        image_files.extend(Path(image_folder).glob(ext.upper()))
    
    print(f"Found {len(image_files)} images to process")
    
    for image_file in image_files:
        try:
            # Run inference
            results = model(str(image_file), conf=0.5)
            
            # Extract detections
            detections = []
            for result in results:
                if result.boxes is not None:
                    boxes = result.boxes.xyxy.cpu().numpy()
                    confidences = result.boxes.conf.cpu().numpy()
                    class_ids = result.boxes.cls.cpu().numpy()
                    
                    # Map class IDs to names (customize based on your model)
                    class_names = {0: 'Good', 1: 'DieDefect', 2: 'Scratch'}
                    
                    for box, conf, cls_id in zip(boxes, confidences, class_ids):
                        label = class_names.get(int(cls_id), f'class_{int(cls_id)}')
                        detections.append({
                            'bbox': box.tolist(),
                            'label': label,
                            'confidence': float(conf)
                        })
            
            # Create annotation
            annotation = create_sample_annotation(str(image_file), detections)
            
            # Save annotation
            json_filename = image_file.stem + '.json'
            json_path = Path(output_folder) / json_filename
            
            with open(json_path, 'w') as f:
                json.dump(annotation, f, indent=2)
            
            print(f"Processed {image_file.name} -> {len(detections)} detections")
            
        except Exception as e:
            print(f"Error processing {image_file}: {e}")

def process_with_onnx_model(model_path: str, image_folder: str, output_folder: str):
    """Process images with ONNX model"""
    try:
        import onnxruntime as ort
    except ImportError:
        print("Please install onnxruntime: pip install onnxruntime")
        return
    
    # Load ONNX model
    session = ort.InferenceSession(model_path)
    input_name = session.get_inputs()[0].name
    input_shape = session.get_inputs()[0].shape
    print(f"Loaded ONNX model: {model_path}")
    print(f"Input shape: {input_shape}")
    
    # Create output folder
    os.makedirs(output_folder, exist_ok=True)
    
    # Process images
    image_files = []
    for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
        image_files.extend(Path(image_folder).glob(ext))
        image_files.extend(Path(image_folder).glob(ext.upper()))
    
    print(f"Found {len(image_files)} images to process")
    
    for image_file in image_files:
        try:
            # Load and preprocess image
            image = cv2.imread(str(image_file))
            if image is None:
                continue
            
            orig_h, orig_w = image.shape[:2]
            
            # Resize to model input size
            input_h, input_w = input_shape[2], input_shape[3]
            resized = cv2.resize(image, (input_w, input_h))
            
            # Normalize and format for model
            normalized = resized.astype(np.float32) / 255.0
            input_data = np.transpose(normalized, (2, 0, 1))
            input_data = np.expand_dims(input_data, axis=0)
            
            # Run inference
            outputs = session.run(None, {input_name: input_data})
            
            # Simple postprocessing (customize based on your model output)
            detections = []
            # This is a placeholder - you'll need to adapt based on your model's output format
            # For classification models, you might create a single detection covering the whole image
            
            # Example for classification result:
            if len(outputs[0][0]) > 1:  # Multi-class output
                predicted_class = np.argmax(outputs[0][0])
                confidence = float(np.max(outputs[0][0]))
                
                if confidence > 0.5:  # Threshold
                    class_names = {0: 'Good', 1: 'DieDefect', 2: 'Scratch'}
                    label = class_names.get(predicted_class, f'class_{predicted_class}')
                    
                    # Create detection covering whole image (for classification)
                    detections.append({
                        'bbox': [0, 0, orig_w, orig_h],
                        'label': label,
                        'confidence': confidence
                    })
            
            # Create annotation
            annotation = create_sample_annotation(str(image_file), detections)
            
            # Save annotation
            json_filename = image_file.stem + '.json'
            json_path = Path(output_folder) / json_filename
            
            with open(json_path, 'w') as f:
                json.dump(annotation, f, indent=2)
            
            print(f"Processed {image_file.name} -> {len(detections)} detections")
            
        except Exception as e:
            print(f"Error processing {image_file}: {e}")

def create_manual_annotations(image_folder: str, output_folder: str):
    """Create template annotations manually (for testing)"""
    os.makedirs(output_folder, exist_ok=True)
    
    image_files = []
    for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
        image_files.extend(Path(image_folder).glob(ext))
    
    print(f"Creating template annotations for {len(image_files)} images")
    
    for image_file in image_files:
        try:
            # Create empty annotation (no detections)
            annotation = create_sample_annotation(str(image_file), detections=[])
            
            # Save annotation
            json_filename = image_file.stem + '.json'
            json_path = Path(output_folder) / json_filename
            
            with open(json_path, 'w') as f:
                json.dump(annotation, f, indent=2)
            
            print(f"Created template for {image_file.name}")
            
        except Exception as e:
            print(f"Error processing {image_file}: {e}")

if __name__ == "__main__":
    # Configuration
    MODEL_PATH = "model/gn24_deployment/best.pt"  # Change to your model path
    IMAGE_FOLDER = "images_GN24/CLAHE"           # Change to your image folder
    OUTPUT_FOLDER = "auto_annotations"            # Output folder for JSON files
    
    print("Auto Annotation Generator")
    print("=" * 40)
    
    # Detect model type and process
    if MODEL_PATH.endswith('.pt'):
        print("Processing with PyTorch/YOLO model...")
        process_with_yolo_model(MODEL_PATH, IMAGE_FOLDER, OUTPUT_FOLDER)
    elif MODEL_PATH.endswith('.onnx'):
        print("Processing with ONNX model...")
        process_with_onnx_model(MODEL_PATH, IMAGE_FOLDER, OUTPUT_FOLDER)
    else:
        print("Creating template annotations (no model inference)...")
        create_manual_annotations(IMAGE_FOLDER, OUTPUT_FOLDER)
    
    print(f"\nCompleted! Check '{OUTPUT_FOLDER}' folder for JSON annotation files.")
