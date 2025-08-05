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
    
    # Parse sliding window configuration
    sliding_window_config = {}
    if 'SLIDING_WINDOW' in config:
        sliding_window_config = {
            'enable_sliding_window': config.getboolean('SLIDING_WINDOW', 'enable_sliding_window', fallback=False),
            'slice_height': config.getint('SLIDING_WINDOW', 'slice_height', fallback=640),
            'slice_width': config.getint('SLIDING_WINDOW', 'slice_width', fallback=640),
            'overlap_height_ratio': config.getfloat('SLIDING_WINDOW', 'overlap_height_ratio', fallback=0.2),
            'overlap_width_ratio': config.getfloat('SLIDING_WINDOW', 'overlap_width_ratio', fallback=0.2),
            'min_image_size_for_slicing': config.getint('SLIDING_WINDOW', 'min_image_size_for_slicing', fallback=1024),
            'enable_padding_for_small_images': config.getboolean('SLIDING_WINDOW', 'enable_padding_for_small_images', fallback=False),
            'padding_color': config.get('SLIDING_WINDOW', 'padding_color', fallback='114,114,114'),
            'postprocess_match_threshold': config.getfloat('SLIDING_WINDOW', 'postprocess_match_threshold', fallback=0.5),
            'postprocess_match_metric': config.get('SLIDING_WINDOW', 'postprocess_match_metric', fallback='IOS'),
            'postprocess_class_agnostic': config.getboolean('SLIDING_WINDOW', 'postprocess_class_agnostic', fallback=False)
        }
    
    return {
        'model_path': config.get('MODEL', 'model_path', fallback='model.pt'),
        'confidence_threshold': config.getfloat('MODEL', 'confidence_threshold', fallback=0.5),
        'input_size': config.get('MODEL', 'input_size', fallback='auto'),
        'annotation_folder': config.get('PATHS', 'annotation_folder', fallback='images'),
        'report_folder': config.get('PATHS', 'report_folder', fallback=''),
        'image_extensions': extensions,
        'class_names': class_names,
        'skip_existing': config.getboolean('PROCESSING', 'skip_existing', fallback=False),
        'show_progress': config.getboolean('PROCESSING', 'show_progress', fallback=True),
        'include_confidence': config.getboolean('ANNOTATION', 'include_confidence', fallback=True),
        'generate_reports': config.getboolean('REPORTS', 'generate_reports', fallback=False),
        'show_confidence_in_reports': config.getboolean('REPORTS', 'show_confidence_in_reports', fallback=True),
        'report_image_format': config.get('REPORTS', 'report_image_format', fallback='jpg'),
        'sliding_window_config': sliding_window_config
    }

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Auto Annotation Tool for Dataset Generation')
    parser.add_argument('--config', '-c', default='config/annotation_config.ini',
                       help='Path to configuration file (default: config/annotation_config.ini)')
    parser.add_argument('--model', '-m', 
                       help='Path to model file (.pt or .onnx) - overrides config')
    parser.add_argument('--folder', '-f',
                       help='Folder with images to annotate (annotations generated in same folder) - overrides config')
    parser.add_argument('--confidence', '-conf', type=float,
                       help='Confidence threshold (0.0-1.0) - overrides config')
    parser.add_argument('--report-folder', '-r',
                       help='Output folder for visual report images - overrides config')
    parser.add_argument('--no-reports', action='store_true',
                       help='Disable report generation even if enabled in config')
    parser.add_argument('--enable-sliding-window', action='store_true',
                       help='Enable sliding window inference for large images')
    parser.add_argument('--disable-sliding-window', action='store_true',
                       help='Disable sliding window inference even if enabled in config')
    parser.add_argument('--enable-padding', action='store_true',
                       help='Enable padding for small images to use sliding window on any size')
    parser.add_argument('--slice-size', type=int, default=640,
                       help='Slice size for sliding window (default: 640)')
    parser.add_argument('--padding-color', type=str, default='114,114,114',
                       help='Padding color in R,G,B format (default: 114,114,114)')
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
    if args.folder:
        config['annotation_folder'] = args.folder
    if args.confidence:
        config['confidence_threshold'] = args.confidence
    if args.report_folder:
        config['report_folder'] = args.report_folder
        config['generate_reports'] = True  # Enable reports if folder is specified
    if args.no_reports:
        config['generate_reports'] = False
    if args.enable_sliding_window:
        config['sliding_window_config']['enable_sliding_window'] = True
        config['sliding_window_config']['slice_height'] = args.slice_size
        config['sliding_window_config']['slice_width'] = args.slice_size
    if args.disable_sliding_window:
        config['sliding_window_config']['enable_sliding_window'] = False
    if args.enable_padding:
        config['sliding_window_config']['enable_padding_for_small_images'] = True
        config['sliding_window_config']['enable_sliding_window'] = True  # Enable sliding window if padding is enabled
    if args.padding_color:
        config['sliding_window_config']['padding_color'] = args.padding_color
    
    # Validate inputs
    if not os.path.exists(config['model_path']):
        print(f"Error: Model file not found: {config['model_path']}")
        sys.exit(1)
    
    if not os.path.exists(config['annotation_folder']):
        print(f"Error: Annotation folder not found: {config['annotation_folder']}")
        sys.exit(1)
    
    # Count images to process
    annotation_path = Path(config['annotation_folder'])
    image_files = []
    for ext in config['image_extensions']:
        image_files.extend(annotation_path.glob(f"*{ext}"))
        image_files.extend(annotation_path.glob(f"*{ext.upper()}"))
    
    # Initialize generator for model verification
    try:
        generator = AutoAnnotationGenerator(
            config['model_path'], 
            config['confidence_threshold'],
            sliding_window_config=config['sliding_window_config']
        )
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
    print(f"  Annotation Folder: {config['annotation_folder']}")
    print(f"  Confidence: {config['confidence_threshold']}")
    print(f"  Images found: {len(image_files)}")
    print(f"  Classes: {len(config['class_names'])}")
    if config['generate_reports'] and config['report_folder']:
        print(f"  Reports: {config['report_folder']}")
    else:
        print(f"  Reports: Disabled")
    if config['sliding_window_config'].get('enable_sliding_window', False):
        print(f"  Sliding Window: Enabled ({config['sliding_window_config']['slice_width']}x{config['sliding_window_config']['slice_height']})")
    else:
        print(f"  Sliding Window: Disabled")
    
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
        
        # Create annotation directory (usually already exists)
        os.makedirs(config['annotation_folder'], exist_ok=True)
        
        # Create report directory if needed
        if config['generate_reports'] and config['report_folder']:
            os.makedirs(config['report_folder'], exist_ok=True)
            print(f"Report images will be saved to: {config['report_folder']}")
        
        # Process images
        print(f"\nProcessing images...")
        generator.process_images(
            config['annotation_folder'],
            config['annotation_folder'],  # Same folder for input and output
            config['image_extensions'],
            report_folder=config['report_folder'] if config['generate_reports'] else None,
            generate_reports=config['generate_reports']
        )
        
        print(f"\n✅ Auto-annotation completed!")
        print(f"📁 Results saved to: {config['annotation_folder']}")
        
        # Count generated annotations
        annotation_path = Path(config['annotation_folder'])
        json_files = list(annotation_path.glob("*.json"))
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
