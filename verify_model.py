#!/usr/bin/env python3
"""
Model Verification Script
Standalone tool to verify model configuration and class information

Usage:
    python verify_model.py model.pt
    python verify_model.py model.onnx --output report.json
    python verify_model.py --config config/annotation_config.ini
"""

import argparse
import configparser
import os
import sys
from pathlib import Path

# Add src directory to path to import our modules
src_path = os.path.join(os.path.dirname(__file__), 'src')
sys.path.append(src_path)

try:
    from model_verifier import ModelVerifier
except ImportError:
    print("Error: Could not import ModelVerifier")
    print("Make sure model_verifier.py is in the src/ directory")
    sys.exit(1)

try:
    from auto_annotation_generator import AutoAnnotationGenerator
except ImportError:
    print("Error: Could not import AutoAnnotationGenerator")
    print("Make sure auto_annotation_generator.py is in the src/ directory")
    sys.exit(1)

def load_model_from_config(config_file: str) -> str:
    """Load model path from configuration file"""
    config = configparser.ConfigParser()
    
    if not os.path.exists(config_file):
        raise FileNotFoundError(f"Config file not found: {config_file}")
    
    config.read(config_file)
    
    if 'MODEL' not in config or 'model_path' not in config['MODEL']:
        raise ValueError("No model_path found in [MODEL] section of config file")
    
    return config['MODEL']['model_path']

def verify_with_annotation_generator(model_path: str):
    """Verify using the AutoAnnotationGenerator approach"""
    try:
        print("="*60)
        print("VERIFICATION USING ANNOTATION GENERATOR")
        print("="*60)
        
        generator = AutoAnnotationGenerator(model_path, confidence_threshold=0.1)
        generator.print_model_verification()
        return True
        
    except Exception as e:
        print(f"Error with AutoAnnotationGenerator verification: {e}")
        return False

def verify_with_model_verifier(model_path: str, output_file: str = None):
    """Verify using the standalone ModelVerifier"""
    try:
        print("="*60)
        print("DETAILED MODEL VERIFICATION")
        print("="*60)
        
        verifier = ModelVerifier(model_path)
        info = verifier.verify_model()
        verifier.print_verification_report(info)
        
        if output_file:
            verifier.save_verification_report(output_file, info)
        
        return True
        
    except Exception as e:
        print(f"Error with ModelVerifier: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description='Verify model configuration and class information')
    
    # Model input options
    model_group = parser.add_mutually_exclusive_group(required=True)
    model_group.add_argument('model', nargs='?', help='Path to model file (.pt or .onnx)')
    model_group.add_argument('--config', '-c', help='Path to configuration file')
    
    # Output options
    parser.add_argument('--output', '-o', help='Save detailed report to JSON file')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    parser.add_argument('--generator-only', action='store_true', 
                       help='Only use AutoAnnotationGenerator verification')
    parser.add_argument('--verifier-only', action='store_true',
                       help='Only use standalone ModelVerifier verification')
    
    args = parser.parse_args()
    
    # Determine model path
    if args.model:
        model_path = args.model
    else:
        try:
            model_path = load_model_from_config(args.config)
            print(f"Loaded model path from config: {model_path}")
        except Exception as e:
            print(f"Error loading from config: {e}")
            sys.exit(1)
    
    # Check if model file exists
    if not os.path.exists(model_path):
        print(f"Error: Model file not found: {model_path}")
        sys.exit(1)
    
    print(f"Verifying model: {model_path}")
    print()
    
    success = True
    
    # Run verification methods
    if not args.verifier_only:
        print("Running annotation generator verification...")
        if not verify_with_annotation_generator(model_path):
            success = False
    
    if not args.generator_only:
        print("Running detailed model verification...")
        if not verify_with_model_verifier(model_path, args.output):
            success = False
    
    # Summary
    print("="*60)
    if success:
        print("✅ MODEL VERIFICATION COMPLETED SUCCESSFULLY")
        print("Recommendations:")
        print("• Check for any warnings about class mismatches")
        print("• Update your annotation_config.ini if needed")
        print("• Test with a sample image to verify detection works")
    else:
        print("❌ MODEL VERIFICATION COMPLETED WITH ERRORS")
        print("Troubleshooting:")
        print("• Check if the model file is corrupted")
        print("• Verify you have the correct model type (.pt vs .onnx)")
        print("• Install required dependencies (torch/ultralytics for .pt, onnxruntime for .onnx)")
    
    print("="*60)

if __name__ == "__main__":
    main()
