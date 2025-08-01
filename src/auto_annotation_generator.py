"""
Auto Annotation Generator
Uses trained models (.pt or .onnx) to generate LabelMe format JSON annotations
"""

import cv2
import numpy as np
import json
import os
from pathlib import Path
import logging
from typing import List, Dict, Tuple, Optional
import base64

# Model inference imports
try:
    import torch
    from ultralytics import YOLO
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("PyTorch/Ultralytics not available. Only ONNX models supported.")

try:
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False
    print("ONNX Runtime not available. Only PyTorch models supported.")

class AutoAnnotationGenerator:
    """
    Generate LabelMe format annotations using trained models
    Supports both .pt (YOLO/PyTorch) and .onnx models
    """
    
    def __init__(self, model_path: str, confidence_threshold: float = 0.5):
        """
        Initialize the auto annotation generator
        
        Args:
            model_path: Path to the trained model (.pt or .onnx)
            confidence_threshold: Minimum confidence for detections
        """
        # Set up logging first
        if not logging.getLogger().handlers:
            logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.model_type = self._detect_model_type()
        self.model = self._load_model()
        
        # Default class names (customize based on your model)
        self.class_names = {
            0: "Good",
            1: "DieDefect", 
            2: "Scratch",
            3: "Crack",
            4: "Contamination"
        }
        
        # Auto-detect class names from model if possible
        self._auto_detect_classes()
    
    def _detect_model_type(self) -> str:
        """Detect model type from file extension"""
        if self.model_path.endswith('.pt'):
            if not TORCH_AVAILABLE:
                raise ImportError("PyTorch/Ultralytics required for .pt models")
            return 'pytorch'
        elif self.model_path.endswith('.onnx'):
            if not ONNX_AVAILABLE:
                raise ImportError("ONNX Runtime required for .onnx models")
            return 'onnx'
        else:
            raise ValueError(f"Unsupported model format: {self.model_path}")
    
    def _load_model(self):
        """Load the trained model"""
        try:
            if self.model_type == 'pytorch':
                # Load YOLO model
                model = YOLO(self.model_path)
                self.logger.info(f"Loaded PyTorch model: {self.model_path}")
                return model
            
            elif self.model_type == 'onnx':
                # Load ONNX model
                session = ort.InferenceSession(self.model_path)
                self.input_name = session.get_inputs()[0].name
                self.input_shape = session.get_inputs()[0].shape
                self.logger.info(f"Loaded ONNX model: {self.model_path}")
                self.logger.info(f"Input shape: {self.input_shape}")
                return session
                
        except Exception as e:
            self.logger.error(f"Failed to load model: {e}")
            raise
    
    def _preprocess_onnx(self, image: np.ndarray) -> np.ndarray:
        """Preprocess image for ONNX model"""
        # Resize to model input size
        input_h, input_w = self.input_shape[2], self.input_shape[3]
        resized = cv2.resize(image, (input_w, input_h))
        
        # Normalize
        normalized = resized.astype(np.float32) / 255.0
        
        # Convert HWC to CHW and add batch dimension
        preprocessed = np.transpose(normalized, (2, 0, 1))
        preprocessed = np.expand_dims(preprocessed, axis=0)
        
        return preprocessed
    
    def _postprocess_onnx(self, outputs: np.ndarray, original_shape: Tuple[int, int]) -> List[Dict]:
        """Postprocess ONNX model outputs to detection format"""
        detections = []
        
        # This is a simplified postprocessing - adjust based on your model output format
        # Assuming output format: [batch, num_detections, [x1, y1, x2, y2, confidence, class_id]]
        
        for detection in outputs[0]:  # Remove batch dimension
            if len(detection) >= 6:
                x1, y1, x2, y2, confidence, class_id = detection[:6]
                
                if confidence >= self.confidence_threshold:
                    # Scale coordinates back to original image size
                    orig_h, orig_w = original_shape
                    input_h, input_w = self.input_shape[2], self.input_shape[3]
                    
                    x1 = x1 * orig_w / input_w
                    y1 = y1 * orig_h / input_h
                    x2 = x2 * orig_w / input_w
                    y2 = y2 * orig_h / input_h
                    
                    detections.append({
                        'bbox': [float(x1), float(y1), float(x2), float(y2)],
                        'confidence': float(confidence),
                        'class_id': int(class_id)
                    })
        
        return detections
    
    def predict(self, image: np.ndarray) -> List[Dict]:
        """
        Run inference on image and return detections
        
        Args:
            image: Input image as numpy array (BGR format)
            
        Returns:
            List of detections with bbox, confidence, class_id
        """
        if self.model_type == 'pytorch':
            # YOLO inference
            results = self.model(image, conf=self.confidence_threshold)
            
            detections = []
            for result in results:
                if result.boxes is not None:
                    boxes = result.boxes.xyxy.cpu().numpy()  # [x1, y1, x2, y2]
                    confidences = result.boxes.conf.cpu().numpy()
                    class_ids = result.boxes.cls.cpu().numpy()
                    
                    for box, conf, cls_id in zip(boxes, confidences, class_ids):
                        detections.append({
                            'bbox': box.tolist(),
                            'confidence': float(conf),
                            'class_id': int(cls_id)
                        })
            
            return detections
        
        elif self.model_type == 'onnx':
            # ONNX inference
            preprocessed = self._preprocess_onnx(image)
            outputs = self.model.run(None, {self.input_name: preprocessed})
            
            return self._postprocess_onnx(outputs, image.shape[:2])
    
    def create_labelme_annotation(self, image_path: str, detections: List[Dict]) -> Dict:
        """
        Create LabelMe format annotation from detections
        
        Args:
            image_path: Path to the image file
            detections: List of detection results
            
        Returns:
            LabelMe format annotation dictionary
        """
        # Load image to get dimensions
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        height, width = image.shape[:2]
        image_filename = os.path.basename(image_path)
        
        # Create shapes from detections
        shapes = []
        for detection in detections:
            x1, y1, x2, y2 = detection['bbox']
            class_id = detection['class_id']
            confidence = detection['confidence']
            
            # Get class label
            label = self.class_names.get(class_id, f"class_{class_id}")
            
            shape = {
                "label": label,
                "points": [
                    [float(x1), float(y1)],  # Top-left
                    [float(x2), float(y2)]   # Bottom-right
                ],
                "group_id": None,
                "description": f"confidence: {confidence:.3f}",
                "shape_type": "rectangle",
                "flags": {},
                "mask": None
            }
            shapes.append(shape)
        
        # Create LabelMe annotation
        annotation = {
            "version": "5.8.1",
            "flags": {},
            "shapes": shapes,
            "imagePath": image_filename,
            "imageData": None,  # Set to None to avoid large files
            "imageHeight": height,
            "imageWidth": width
        }
        
        return annotation
    
    def create_visual_report(self, image_path: str, detections: List[Dict], 
                           output_path: str, show_confidence: bool = True):
        """
        Create a visual report image with bounding boxes and confidence scores
        
        Args:
            image_path: Path to the original image
            detections: List of detection results
            output_path: Path to save the report image
            show_confidence: Whether to display confidence scores
        """
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        # Make a copy for drawing
        report_image = image.copy()
        
        # Define colors for different classes (BGR format)
        colors = [
            (0, 255, 0),    # Green
            (255, 0, 0),    # Blue  
            (0, 0, 255),    # Red
            (255, 255, 0),  # Cyan
            (255, 0, 255),  # Magenta
            (0, 255, 255),  # Yellow
            (128, 0, 128),  # Purple
            (255, 165, 0),  # Orange
            (0, 128, 255),  # Orange-Red
            (128, 128, 128) # Gray
        ]
        
        # Draw detections
        for detection in detections:
            x1, y1, x2, y2 = [int(coord) for coord in detection['bbox']]
            class_id = detection['class_id']
            confidence = detection['confidence']
            
            # Get class name and color
            class_name = self.class_names.get(class_id, f"class_{class_id}")
            color = colors[class_id % len(colors)]
            
            # Draw bounding box
            cv2.rectangle(report_image, (x1, y1), (x2, y2), color, 2)
            
            # Prepare label text
            if show_confidence:
                label = f"{class_name}: {confidence:.2f}"
            else:
                label = class_name
            
            # Calculate text size and position
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.6
            thickness = 2
            (text_width, text_height), baseline = cv2.getTextSize(label, font, font_scale, thickness)
            
            # Draw label background
            label_y = max(y1 - 10, text_height + 10)
            cv2.rectangle(report_image, 
                         (x1, label_y - text_height - 5), 
                         (x1 + text_width + 5, label_y + baseline), 
                         color, -1)
            
            # Draw label text
            cv2.putText(report_image, label, (x1 + 2, label_y - 5), 
                       font, font_scale, (255, 255, 255), thickness)
        
        # Add summary information
        height, width = report_image.shape[:2]
        summary_text = f"Detections: {len(detections)}"
        cv2.putText(report_image, summary_text, (10, height - 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Save the report image
        cv2.imwrite(output_path, report_image)

    def process_images(self, input_folder: str, output_folder: str, 
                      image_extensions: List[str] = ['.jpg', '.jpeg', '.png', '.bmp'],
                      report_folder: str = None, generate_reports: bool = False):
        """
        Process all images in a folder and generate annotations
        
        Args:
            input_folder: Folder containing images to annotate
            output_folder: Folder to save JSON annotations
            image_extensions: List of supported image file extensions
            report_folder: Folder to save visual report images (optional)
            generate_reports: Whether to generate visual reports
        """
        input_path = Path(input_folder)
        output_path = Path(output_folder)
        output_path.mkdir(exist_ok=True)
        
        # Create report folder if needed
        report_path = None
        if generate_reports and report_folder:
            report_path = Path(report_folder)
            report_path.mkdir(exist_ok=True)
            self.logger.info(f"Report images will be saved to: {report_path}")
        
        # Get all image files
        image_files = []
        for ext in image_extensions:
            image_files.extend(input_path.glob(f"*{ext}"))
            image_files.extend(input_path.glob(f"*{ext.upper()}"))
        
        self.logger.info(f"Found {len(image_files)} images to process")
        
        processed_count = 0
        error_count = 0
        
        for image_file in image_files:
            try:
                # Load image
                image = cv2.imread(str(image_file))
                if image is None:
                    self.logger.warning(f"Could not load image: {image_file}")
                    error_count += 1
                    continue
                
                # Run inference
                detections = self.predict(image)
                
                # Create annotation
                annotation = self.create_labelme_annotation(str(image_file), detections)
                
                # Save annotation
                json_filename = image_file.stem + '.json'
                json_path = output_path / json_filename
                
                with open(json_path, 'w') as f:
                    json.dump(annotation, f, indent=2)
                
                # Generate visual report if requested
                if generate_reports and report_path:
                    report_filename = image_file.stem + '_report.jpg'
                    report_file_path = report_path / report_filename
                    self.create_visual_report(str(image_file), detections, str(report_file_path))
                
                processed_count += 1
                self.logger.info(f"Processed {image_file.name} -> {len(detections)} detections")
                
                if processed_count % 50 == 0:
                    self.logger.info(f"Progress: {processed_count}/{len(image_files)} images processed")
                    
            except Exception as e:
                self.logger.error(f"Error processing {image_file}: {e}")
                error_count += 1
        
        self.logger.info(f"Completed: {processed_count} processed, {error_count} errors")
        if generate_reports and report_path:
            self.logger.info(f"Generated {processed_count} report images in: {report_path}")
    
    def set_class_names(self, class_names: Dict[int, str]):
        """Update class names mapping"""
        self.class_names = class_names
        self.logger.info(f"Updated class names: {class_names}")
    
    def _auto_detect_classes(self):
        """Auto-detect class names from the model if available"""
        try:
            if self.model_type == 'pytorch' and hasattr(self.model, 'names'):
                # YOLO model with class names
                model_names = self.model.names
                if isinstance(model_names, dict):
                    self.class_names = {int(k): v for k, v in model_names.items()}
                    self.logger.info(f"Auto-detected {len(self.class_names)} classes from model")
                elif isinstance(model_names, list):
                    self.class_names = {i: name for i, name in enumerate(model_names)}
                    self.logger.info(f"Auto-detected {len(self.class_names)} classes from model")
        except Exception as e:
            self.logger.warning(f"Could not auto-detect classes: {e}")
    
    def verify_model_configuration(self) -> Dict:
        """
        Verify and return model configuration information
        
        Returns:
            Dictionary containing model information including classes, shapes, etc.
        """
        info = {
            'model_path': self.model_path,
            'model_type': self.model_type,
            'confidence_threshold': self.confidence_threshold,
            'classes': self.class_names.copy(),
            'num_classes': len(self.class_names),
            'file_exists': os.path.exists(self.model_path),
            'file_size': self._get_file_size() if os.path.exists(self.model_path) else None,
        }
        
        try:
            if self.model_type == 'pytorch':
                info.update(self._get_pytorch_model_info())
            elif self.model_type == 'onnx':
                info.update(self._get_onnx_model_info())
        except Exception as e:
            info['verification_error'] = str(e)
        
        return info
    
    def _get_pytorch_model_info(self) -> Dict:
        """Get PyTorch model specific information"""
        info = {}
        
        try:
            # Try to get model metadata
            if hasattr(self.model, 'model'):
                if hasattr(self.model.model, 'nc'):
                    info['model_num_classes'] = self.model.model.nc
                
                # Check if class count matches
                if 'model_num_classes' in info:
                    if info['model_num_classes'] != len(self.class_names):
                        info['class_count_mismatch'] = True
                        info['warning'] = f"Model has {info['model_num_classes']} classes, but {len(self.class_names)} class names configured"
            
            # Get model names if available
            if hasattr(self.model, 'names'):
                model_names = self.model.names
                if isinstance(model_names, (dict, list)):
                    info['model_class_names'] = model_names
                    
                    # Compare with configured names
                    if isinstance(model_names, dict):
                        model_classes = {int(k): v for k, v in model_names.items()}
                    else:
                        model_classes = {i: name for i, name in enumerate(model_names)}
                    
                    if model_classes != self.class_names:
                        info['class_names_mismatch'] = True
                        info['model_classes'] = model_classes
            
            # Get input size (patch size) from YOLO model
            input_size = None
            
            # Method 1: From model args
            if hasattr(self.model.model, 'args') and hasattr(self.model.model.args, 'imgsz'):
                input_size = self.model.model.args.imgsz
            
            # Method 2: From model yaml
            elif hasattr(self.model.model, 'yaml'):
                yaml_info = self.model.model.yaml
                if 'imgsz' in yaml_info:
                    input_size = yaml_info['imgsz']
            
            # Process and store input size
            if input_size:
                if isinstance(input_size, (list, tuple)):
                    info['input_size'] = input_size[0] if len(input_size) > 0 else 640
                else:
                    info['input_size'] = input_size
            else:
                info['input_size'] = 640  # Default YOLO size
                info['input_size_source'] = 'default'
            
            info['input_channels'] = 3
            info['input_format'] = 'RGB'
        
        except Exception as e:
            info['pytorch_info_error'] = str(e)
        
        return info
    
    def _get_onnx_model_info(self) -> Dict:
        """Get ONNX model specific information"""
        info = {}
        
        try:
            if hasattr(self.model, 'get_inputs'):
                inputs = self.model.get_inputs()
                outputs = self.model.get_outputs()
                
                if inputs:
                    info['input_shape'] = inputs[0].shape
                    info['input_name'] = inputs[0].name
                    
                    # Extract input size from shape
                    shape = inputs[0].shape
                    if len(shape) >= 3:
                        if len(shape) == 4:  # [B, C, H, W] or [B, H, W, C]
                            if shape[1] in [1, 3, 4]:  # Likely channels
                                info['input_size'] = shape[2]
                                info['input_channels'] = shape[1]
                            else:  # Likely [B, H, W, C] format
                                info['input_size'] = shape[1]
                                info['input_channels'] = shape[3]
                        elif len(shape) == 3:  # [C, H, W] or [H, W, C]
                            if shape[0] in [1, 3, 4]:  # Likely channels first
                                info['input_size'] = shape[1]
                                info['input_channels'] = shape[0]
                            else:  # Likely channels last
                                info['input_size'] = shape[0]
                                info['input_channels'] = shape[2]
                    
                    info['input_format'] = 'RGB (typically)'
                
                if outputs:
                    info['output_shape'] = outputs[0].shape
                    info['output_name'] = outputs[0].name
                    
                    # Try to infer number of classes from output shape
                    if len(outputs[0].shape) >= 2:
                        last_dim = outputs[0].shape[-1]
                        if last_dim > 5:  # Assuming YOLO format: x,y,w,h,conf + classes
                            inferred_classes = last_dim - 5
                            info['inferred_num_classes'] = inferred_classes
                            
                            if inferred_classes != len(self.class_names):
                                info['class_count_mismatch'] = True
                                info['warning'] = f"Output shape suggests {inferred_classes} classes, but {len(self.class_names)} class names configured"
        
        except Exception as e:
            info['onnx_info_error'] = str(e)
        
        return info
    
    def _get_file_size(self) -> str:
        """Get human-readable file size"""
        try:
            size = os.path.getsize(self.model_path)
            for unit in ['B', 'KB', 'MB', 'GB']:
                if size < 1024.0:
                    return f"{size:.1f} {unit}"
                size /= 1024.0
            return f"{size:.1f} TB"
        except:
            return "Unknown"
    
    def print_model_verification(self):
        """Print a formatted model verification report"""
        info = self.verify_model_configuration()
        
        print("=" * 60)
        print("MODEL CONFIGURATION VERIFICATION")
        print("=" * 60)
        print(f"Model Path: {info['model_path']}")
        print(f"Model Type: {info['model_type']}")
        print(f"File Exists: {info['file_exists']}")
        if info['file_size']:
            print(f"File Size: {info['file_size']}")
        print(f"Confidence Threshold: {info['confidence_threshold']}")
        print()
        
        # Input Size Information
        print("INPUT SIZE REQUIREMENTS:")
        if info.get('input_size'):
            input_size = info['input_size']
            print(f"📐 Input Size (Patch Size): {input_size}x{input_size} pixels")
        else:
            print("📐 Input Size: Not detected")
        
        if info.get('input_channels'):
            print(f"🎨 Input Channels: {info['input_channels']}")
        
        if info.get('input_format'):
            print(f"🖼️  Input Format: {info['input_format']}")
            
        if info.get('input_size_source') == 'default':
            print(f"ℹ️  Note: Using default size (actual may vary)")
        print()
        
        # Class Configuration
        print("CLASS CONFIGURATION:")
        print(f"Configured Classes: {info['num_classes']}")
        for class_id, class_name in sorted(info['classes'].items()):
            print(f"  {class_id}: {class_name}")
        print()
        
        # Model-specific information
        if self.model_type == 'pytorch':
            if 'model_num_classes' in info:
                print(f"Model Reports: {info['model_num_classes']} classes")
            if 'model_class_names' in info:
                print("Model Class Names:")
                model_names = info['model_class_names']
                if isinstance(model_names, dict):
                    for k, v in sorted(model_names.items()):
                        print(f"  {k}: {v}")
                elif isinstance(model_names, list):
                    for i, name in enumerate(model_names):
                        print(f"  {i}: {name}")
        
        elif self.model_type == 'onnx':
            if 'input_shape' in info:
                print(f"Input Shape: {info['input_shape']}")
            if 'output_shape' in info:
                print(f"Output Shape: {info['output_shape']}")
            if 'inferred_num_classes' in info:
                print(f"Inferred Classes: {info['inferred_num_classes']}")
        
        # Warnings
        if 'warning' in info:
            print()
            print("⚠️  WARNING:")
            print(f"   {info['warning']}")
        
        if 'class_count_mismatch' in info:
            print()
            print("⚠️  CLASS COUNT MISMATCH DETECTED!")
            if 'model_classes' in info:
                print("   Model classes vs configured classes differ")
        
        if 'class_names_mismatch' in info:
            print("⚠️  CLASS NAMES MISMATCH DETECTED!")
            print("   Model class names vs configured names differ")
        
        # Errors
        error_keys = [k for k in info.keys() if 'error' in k]
        if error_keys:
            print()
            print("❌ ERRORS:")
            for key in error_keys:
                print(f"   {key}: {info[key]}")
        
        print("=" * 60)


def main():
    """Example usage"""
    # Configuration
    model_path = "model/gn24_deployment/best.pt"  # or "model.onnx"
    input_folder = "images_GN24/CLAHE"
    output_folder = "auto_annotations"
    confidence_threshold = 0.5
    
    # Custom class names (adjust based on your model)
    class_names = {
        0: "Good",
        1: "DieDefect",
        2: "Scratch", 
        3: "Crack"
    }
    
    try:
        # Initialize generator
        generator = AutoAnnotationGenerator(model_path, confidence_threshold)
        generator.set_class_names(class_names)
        
        # Process images
        generator.process_images(input_folder, output_folder)
        
        print(f"Auto-annotation completed! Check '{output_folder}' for results.")
        
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
