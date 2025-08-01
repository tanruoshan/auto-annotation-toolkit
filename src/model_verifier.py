"""
Model Verification Tool
Analyzes model configuration including class information, input/output shapes, and metadata
"""

import os
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging

# Model inference imports
try:
    import torch
    from ultralytics import YOLO
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import onnxruntime as ort
    import onnx
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False

class ModelVerifier:
    """
    Verify and analyze model configuration including classes, shapes, and metadata
    """
    
    def __init__(self, model_path: str):
        """
        Initialize model verifier
        
        Args:
            model_path: Path to the model file (.pt or .onnx)
        """
        self.model_path = model_path
        self.model_type = self._detect_model_type()
        
        # Set up logging only if not already configured
        if not logging.getLogger().handlers:
            logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
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
    
    def verify_model(self) -> Dict:
        """
        Verify model and extract configuration information
        
        Returns:
            Dictionary containing model information
        """
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model file not found: {self.model_path}")
        
        if self.model_type == 'pytorch':
            return self._verify_pytorch_model()
        elif self.model_type == 'onnx':
            return self._verify_onnx_model()
    
    def _verify_pytorch_model(self) -> Dict:
        """Verify PyTorch (.pt) model"""
        try:
            # Load model metadata without full initialization
            # Handle PyTorch 2.6+ security changes by using weights_only=False for YOLO models
            try:
                # For YOLO models, we need to allow loading the full model
                model_data = torch.load(self.model_path, map_location='cpu', weights_only=False)
                self.logger.info("Loaded model metadata successfully")
            except Exception as e:
                self.logger.warning(f"Could not load model directly: {e}")
                model_data = None
            
            # Try loading as YOLO model for additional info
            yolo_model = None
            try:
                yolo_model = YOLO(self.model_path)
            except Exception as e:
                self.logger.warning(f"Could not load as YOLO model: {e}")
            
            # Extract information
            info = {
                'model_type': 'PyTorch (.pt)',
                'file_size': self._get_file_size(),
                'classes': {},
                'num_classes': None,
                'input_shape': None,
                'output_shape': None,
                'metadata': {}
            }
            
            # Extract class information
            if isinstance(model_data, dict):
                # Standard PyTorch checkpoint format
                if 'model' in model_data:
                    # Get number of classes from model structure
                    model_state = model_data['model']
                    if hasattr(model_state, 'nc'):
                        info['num_classes'] = model_state.nc
                    elif hasattr(model_state, 'model') and hasattr(model_state.model[-1], 'nc'):
                        info['num_classes'] = model_state.model[-1].nc
                
                # Get class names if available
                if 'names' in model_data:
                    names = model_data['names']
                    if isinstance(names, dict):
                        info['classes'] = {int(k): v for k, v in names.items()}
                    elif isinstance(names, list):
                        info['classes'] = {i: name for i, name in enumerate(names)}
                
                # Get number of classes
                if 'nc' in model_data:
                    info['num_classes'] = model_data['nc']
                
                # Get other metadata
                for key in ['epoch', 'best_fitness', 'optimizer', 'date']:
                    if key in model_data:
                        info['metadata'][key] = str(model_data[key])
            
            # Get additional info from YOLO model if loaded
            if yolo_model:
                try:
                    if hasattr(yolo_model.model, 'names'):
                        yolo_names = yolo_model.model.names
                        if isinstance(yolo_names, dict):
                            info['classes'].update({int(k): v for k, v in yolo_names.items()})
                        elif isinstance(yolo_names, list):
                            info['classes'].update({i: name for i, name in enumerate(yolo_names)})
                    
                    if hasattr(yolo_model.model, 'nc'):
                        info['num_classes'] = yolo_model.model.nc
                        
                    # Get input size from model (patch size)
                    input_size = None
                    
                    # Method 1: From model args
                    if hasattr(yolo_model.model, 'args') and hasattr(yolo_model.model.args, 'imgsz'):
                        input_size = yolo_model.model.args.imgsz
                    
                    # Method 2: From model yaml
                    elif hasattr(yolo_model.model, 'yaml'):
                        yaml_info = yolo_model.model.yaml
                        if 'imgsz' in yaml_info:
                            input_size = yaml_info['imgsz']
                    
                    # Method 3: From model metadata
                    if not input_size and isinstance(model_data, dict):
                        if 'imgsz' in model_data:
                            input_size = model_data['imgsz']
                        elif 'model' in model_data and hasattr(model_data['model'], 'args'):
                            args = model_data['model'].args
                            if hasattr(args, 'imgsz'):
                                input_size = args.imgsz
                    
                    # Process and store input size
                    if input_size:
                        # Handle different input size formats
                        if isinstance(input_size, (list, tuple)):
                            if len(input_size) == 1:
                                info['input_size'] = input_size[0]
                            elif len(input_size) == 2:
                                info['input_size'] = input_size[0]  # Usually square
                                info['input_height'] = input_size[0]
                                info['input_width'] = input_size[1]
                            else:
                                info['input_size'] = input_size[0]
                        else:
                            info['input_size'] = input_size
                    else:
                        # Default YOLO input size if not found
                        info['input_size'] = 640
                        info['input_size_source'] = 'default'
                    
                    # Add input format information
                    info['input_channels'] = 3  # RGB
                    info['input_format'] = 'RGB'
                    info['normalization'] = '0-1 (divide by 255)'
                
                except Exception as e:
                    self.logger.warning(f"Error extracting YOLO info: {e}")
            
            # If no classes found, check if it's a single-class model
            if not info['classes'] and info['num_classes'] == 1:
                info['classes'] = {0: 'detection'}
                info['notes'] = ['Single-class model detected']
            elif not info['classes'] and info['num_classes']:
                info['classes'] = {i: f'class_{i}' for i in range(info['num_classes'])}
                info['notes'] = ['Class names not found, using generic names']
            
            return info
            
        except Exception as e:
            raise RuntimeError(f"Failed to verify PyTorch model: {e}")
    
    def _verify_onnx_model(self) -> Dict:
        """Verify ONNX (.onnx) model"""
        try:
            # Load ONNX model
            session = ort.InferenceSession(self.model_path)
            
            # Load ONNX model for metadata
            onnx_model = onnx.load(self.model_path) if ONNX_AVAILABLE else None
            
            info = {
                'model_type': 'ONNX (.onnx)',
                'file_size': self._get_file_size(),
                'classes': {},
                'num_classes': None,
                'input_shape': None,
                'output_shape': None,
                'metadata': {}
            }
            
            # Get input/output information
            inputs = session.get_inputs()
            outputs = session.get_outputs()
            
            if inputs:
                info['input_shape'] = inputs[0].shape
                info['input_name'] = inputs[0].name
                info['input_type'] = str(inputs[0].type)
                
                # Extract input size (patch size) from shape
                # Common formats: [batch, channels, height, width] or [batch, height, width, channels]
                shape = inputs[0].shape
                if len(shape) >= 3:
                    if len(shape) == 4:  # [B, C, H, W] or [B, H, W, C]
                        # Assume [B, C, H, W] format (most common)
                        if shape[1] in [1, 3, 4]:  # Likely channels
                            info['input_size'] = shape[2]  # Height
                            info['input_height'] = shape[2]
                            info['input_width'] = shape[3]
                            info['input_channels'] = shape[1]
                        else:  # Likely [B, H, W, C] format
                            info['input_size'] = shape[1]  # Height
                            info['input_height'] = shape[1]
                            info['input_width'] = shape[2]
                            info['input_channels'] = shape[3]
                    elif len(shape) == 3:  # [C, H, W] or [H, W, C]
                        if shape[0] in [1, 3, 4]:  # Likely channels first
                            info['input_size'] = shape[1]
                            info['input_height'] = shape[1]
                            info['input_width'] = shape[2]
                            info['input_channels'] = shape[0]
                        else:  # Likely channels last
                            info['input_size'] = shape[0]
                            info['input_height'] = shape[0]
                            info['input_width'] = shape[1]
                            info['input_channels'] = shape[2]
                
                # Add input format information for ONNX
                info['input_format'] = 'RGB (typically)'
                info['normalization'] = 'Model-dependent (check training preprocessing)'
            
            if outputs:
                info['output_shape'] = outputs[0].shape
                info['output_name'] = outputs[0].name
                info['output_type'] = str(outputs[0].type)
                
                # Try to infer number of classes from output shape
                if len(outputs[0].shape) >= 2:
                    # Common YOLO format: [batch, detections, (x,y,w,h,conf,class_probs...)]
                    last_dim = outputs[0].shape[-1]
                    if last_dim > 5:  # x,y,w,h,conf + classes
                        info['num_classes'] = last_dim - 5
                        info['classes'] = {i: f'class_{i}' for i in range(info['num_classes'])}
                        info['notes'] = ['Classes inferred from output shape - verify accuracy']
            
            # Get metadata from ONNX model
            if onnx_model:
                try:
                    # Get model metadata
                    for prop in onnx_model.metadata_props:
                        info['metadata'][prop.key] = prop.value
                    
                    # Look for class information in metadata
                    for prop in onnx_model.metadata_props:
                        if 'class' in prop.key.lower() or 'names' in prop.key.lower():
                            try:
                                # Try to parse as JSON
                                class_data = json.loads(prop.value)
                                if isinstance(class_data, dict):
                                    info['classes'].update({int(k): v for k, v in class_data.items()})
                                elif isinstance(class_data, list):
                                    info['classes'].update({i: name for i, name in enumerate(class_data)})
                            except:
                                # Store as raw metadata
                                info['metadata'][f'class_info_{prop.key}'] = prop.value
                
                except Exception as e:
                    self.logger.warning(f"Error reading ONNX metadata: {e}")
            
            return info
            
        except Exception as e:
            raise RuntimeError(f"Failed to verify ONNX model: {e}")
    
    def _get_file_size(self) -> str:
        """Get human-readable file size"""
        size = os.path.getsize(self.model_path)
        for unit in ['B', 'KB', 'MB', 'GB']:
            if size < 1024.0:
                return f"{size:.1f} {unit}"
            size /= 1024.0
        return f"{size:.1f} TB"
    
    def print_verification_report(self, info: Dict = None):
        """Print a formatted verification report"""
        if info is None:
            info = self.verify_model()
        
        print("=" * 60)
        print("MODEL VERIFICATION REPORT")
        print("=" * 60)
        print(f"Model Path: {self.model_path}")
        print(f"Model Type: {info['model_type']}")
        print(f"File Size: {info['file_size']}")
        print()
        
        # Input Size Information (Patch Size)
        print("INPUT SIZE REQUIREMENTS:")
        if info.get('input_size'):
            input_size = info['input_size']
            print(f"📐 Input Size (Patch Size): {input_size}x{input_size} pixels")
            
            if info.get('input_height') and info.get('input_width'):
                if info['input_height'] != info['input_width']:
                    print(f"📐 Exact Dimensions: {info['input_height']}x{info['input_width']} pixels")
        else:
            print("📐 Input Size: Not detected")
        
        if info.get('input_channels'):
            print(f"🎨 Input Channels: {info['input_channels']}")
        
        if info.get('input_format'):
            print(f"🖼️  Input Format: {info['input_format']}")
            
        if info.get('normalization'):
            print(f"⚡ Normalization: {info['normalization']}")
            
        if info.get('input_size_source') == 'default':
            print(f"ℹ️  Note: Using default size (actual size may vary)")
        print()
        
        # Input/Output Shape Details
        if info.get('input_shape'):
            print(f"📊 Input Shape: {info['input_shape']}")
        if info.get('output_shape'):
            print(f"📊 Output Shape: {info['output_shape']}")
        if info.get('input_shape') or info.get('output_shape'):
            print()
        
        # Class Information
        print("CLASS CONFIGURATION:")
        if info['classes']:
            print(f"Number of Classes: {len(info['classes'])}")
            for class_id, class_name in sorted(info['classes'].items()):
                print(f"  {class_id}: {class_name}")
        else:
            print("  No class information found")
            if info['num_classes']:
                print(f"  Detected {info['num_classes']} classes but no names available")
        print()
        
        # Metadata
        if info['metadata']:
            print("METADATA:")
            for key, value in info['metadata'].items():
                print(f"  {key}: {value}")
            print()
        
        # Notes/Warnings
        if 'notes' in info:
            print("NOTES:")
            for note in info['notes']:
                print(f"  • {note}")
            print()
        
        print("=" * 60)
    
    def save_verification_report(self, output_path: str, info: Dict = None):
        """Save verification report to JSON file"""
        if info is None:
            info = self.verify_model()
        
        info['model_path'] = self.model_path
        
        with open(output_path, 'w') as f:
            json.dump(info, f, indent=2)
        
        print(f"Verification report saved to: {output_path}")


def verify_model_cli(model_path: str, output_file: str = None):
    """Command line interface for model verification"""
    try:
        verifier = ModelVerifier(model_path)
        info = verifier.verify_model()
        verifier.print_verification_report(info)
        
        if output_file:
            verifier.save_verification_report(output_file, info)
            
    except Exception as e:
        print(f"Error verifying model: {e}")
        return False
    
    return True


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Verify model configuration")
    parser.add_argument("model", help="Path to model file (.pt or .onnx)")
    parser.add_argument("--output", "-o", help="Save report to JSON file")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose logging")
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)
    
    verify_model_cli(args.model, args.output)
