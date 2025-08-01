#!/usr/bin/env python3
"""
Command Line Auto Annotation Tool
Generates LabelMe format annotations using trained models

Usage:
    python annotate_dataset.py
    python annotate_dataset.py --config custom_config.ini
    python annotate_dataset.py --model model.pt --input images/ --output annotations/
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
    from auto_annotation_generator import AutoAnnotationGenerator
except ImportError:
    print("Error: Could not import AutoAnnotationGenerator")
    print("Make sure auto_annotation_generator.py is in the src/ directory")
    sys.exit(1)

def load_config(config_file: str) -> dict:
    """Load configuration from INI file"""
    config = configparser.ConfigParser()
    
    if not os.path.exists(config_file):
        print(f"Config file not found: {config_file}")
        sys.exit(1)
    
    config.read(config_file)
    
    # Parse class names
    class_names = {}
    if 'CLASSES' in config:
        for key, value in config['CLASSES'].items():
            try:
                class_names[int(key)] = value
            except ValueError:
                print(f"Warning: Invalid class ID '{key}' in config")
    
    # Parse image extensions
    extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    if 'PATHS' in config and 'image_extensions' in config['PATHS']:
        extensions = [ext.strip() for ext in config['PATHS']['image_extensions'].split(',')]
    
    return {
        'model_path': config.get('MODEL', 'model_path', fallback='model.pt'),
        'confidence_threshold': config.getfloat('MODEL', 'confidence_threshold', fallback=0.5),
        'input_size': config.get('MODEL', 'input_size', fallback='auto'),
        'input_folder': config.get('PATHS', 'input_folder', fallback='images'),
        'output_folder': config.get('PATHS', 'output_folder', fallback='annotations'),
        'report_folder': config.get('PATHS', 'report_folder', fallback=''),
        'image_extensions': extensions,
        'class_names': class_names,
        'skip_existing': config.getboolean('PROCESSING', 'skip_existing', fallback=False),
        'show_progress': config.getboolean('PROCESSING', 'show_progress', fallback=True),
        'include_confidence': config.getboolean('ANNOTATION', 'include_confidence', fallback=True),
        'generate_reports': config.getboolean('REPORTS', 'generate_reports', fallback=False),
        'show_confidence_in_reports': config.getboolean('REPORTS', 'show_confidence_in_reports', fallback=True),
        'report_image_format': config.get('REPORTS', 'report_image_format', fallback='jpg')
    }

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Auto Annotation Tool for Dataset Generation')
    parser.add_argument('--config', '-c', default='config/annotation_config.ini',
                       help='Path to configuration file (default: config/annotation_config.ini)')
    parser.add_argument('--model', '-m', 
                       help='Path to model file (.pt or .onnx) - overrides config')
    parser.add_argument('--input', '-i',
                       help='Input folder with images - overrides config')
    parser.add_argument('--output', '-o',
                       help='Output folder for annotations - overrides config')
    parser.add_argument('--confidence', '-conf', type=float,
                       help='Confidence threshold (0.0-1.0) - overrides config')
    parser.add_argument('--report-folder', '-r',
                       help='Output folder for visual report images - overrides config')
    parser.add_argument('--no-reports', action='store_true',
                       help='Disable report generation even if enabled in config')
    parser.add_argument('--dry-run', action='store_true',
                       help='Show what would be processed without actually processing')
    parser.add_argument('--verify-model', action='store_true',
                       help='Verify model configuration and class information')
    parser.add_argument('--verify-only', action='store_true',
                       help='Only verify model, do not process images')
    
    args = parser.parse_args()
    
    # Load configuration
    try:
        config = load_config(args.config)
    except Exception as e:
        print(f"Error loading config: {e}")
        sys.exit(1)
    
    # Override config with command line arguments
    if args.model:
        config['model_path'] = args.model
    if args.input:
        config['input_folder'] = args.input
    if args.output:
        config['output_folder'] = args.output
    if args.confidence:
        config['confidence_threshold'] = args.confidence
    if args.report_folder:
        config['report_folder'] = args.report_folder
        config['generate_reports'] = True  # Enable reports if folder is specified
    if args.no_reports:
        config['generate_reports'] = False
    
    # Validate inputs
    if not os.path.exists(config['model_path']):
        print(f"Error: Model file not found: {config['model_path']}")
        sys.exit(1)
    
    if not os.path.exists(config['input_folder']):
        print(f"Error: Input folder not found: {config['input_folder']}")
        sys.exit(1)
    
    # Count images to process
    input_path = Path(config['input_folder'])
    image_files = []
    for ext in config['image_extensions']:
        image_files.extend(input_path.glob(f"*{ext}"))
        image_files.extend(input_path.glob(f"*{ext.upper()}"))
    
    # Initialize generator for model verification
    try:
        generator = AutoAnnotationGenerator(config['model_path'], config['confidence_threshold'])
        generator.set_class_names(config['class_names'])
    except Exception as e:
        print(f"Error initializing model: {e}")
        sys.exit(1)
    
    # Model verification
    if args.verify_model or args.verify_only:
        print("\n" + "="*60)
        print("PERFORMING MODEL VERIFICATION")
        print("="*60)
        generator.print_model_verification()
        
        if args.verify_only:
            sys.exit(0)
        
        print("\nPress Enter to continue with annotation or Ctrl+C to cancel...")
        try:
            input()
        except KeyboardInterrupt:
            print("\nCancelled by user")
            sys.exit(0)
    
    print(f"Configuration:")
    print(f"  Model: {config['model_path']}")
    print(f"  Input: {config['input_folder']}")
    print(f"  Output: {config['output_folder']}")
    print(f"  Confidence: {config['confidence_threshold']}")
    print(f"  Images found: {len(image_files)}")
    print(f"  Classes: {len(config['class_names'])}")
    if config['generate_reports'] and config['report_folder']:
        print(f"  Reports: {config['report_folder']}")
    else:
        print(f"  Reports: Disabled")
    
    if args.dry_run:
        print("Dry-run mode - showing first 10 files that would be processed:")
        for i, img_file in enumerate(image_files[:10]):
            print(f"  {i+1}. {img_file.name}")
        if len(image_files) > 10:
            print(f"  ... and {len(image_files) - 10} more files")
        sys.exit(0)
    
    if len(image_files) == 0:
        print("No images found to process!")
        sys.exit(1)
    
    # Confirm processing
    response = input(f"\nProcess {len(image_files)} images? (y/N): ")
    if response.lower() not in ['y', 'yes']:
        print("Cancelled.")
        sys.exit(0)
    
    try:
        # Model is already initialized earlier for verification
        print(f"\nUsing model: {config['model_path']}")
        
        # Set class names (already done during initialization)
        if config['class_names']:
            print(f"Using custom class names: {list(config['class_names'].values())}")
        
        # Create output directory
        os.makedirs(config['output_folder'], exist_ok=True)
        
        # Create report directory if needed
        if config['generate_reports'] and config['report_folder']:
            os.makedirs(config['report_folder'], exist_ok=True)
            print(f"Report images will be saved to: {config['report_folder']}")
        
        # Process images
        print(f"\nProcessing images...")
        generator.process_images(
            config['input_folder'],
            config['output_folder'],
            config['image_extensions'],
            report_folder=config['report_folder'] if config['generate_reports'] else None,
            generate_reports=config['generate_reports']
        )
        
        print(f"\n✅ Auto-annotation completed!")
        print(f"📁 Results saved to: {config['output_folder']}")
        
        # Count generated annotations
        output_path = Path(config['output_folder'])
        json_files = list(output_path.glob("*.json"))
        print(f"📄 Generated {len(json_files)} annotation files")
        
        # Count generated reports if enabled
        if config['generate_reports'] and config['report_folder']:
            report_path = Path(config['report_folder'])
            report_files = list(report_path.glob("*_report.jpg"))
            print(f"🖼️  Generated {len(report_files)} report images in: {config['report_folder']}")
        
    except KeyboardInterrupt:
        print("\n\nProcess interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error during processing: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
