#!/usr/bin/env python3
"""
Example: Sliding Window Inference with SAHI
Demonstrates how to use sliding window inference for large images or small object detection
"""

import os
import sys
from pathlib import Path

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from auto_annotation_generator import AutoAnnotationGenerator

def sliding_window_example():
    """Example of using sliding window inference"""
    
    print("🪟 Sliding Window Inference Example")
    print("=" * 60)
    
    # Configuration for sliding window
    config = {
        "model_path": "path/to/your/model.pt",  # Update this path
        "input_folder": "path/to/large/images",  # Update this path
        "output_folder": "sliding_window_annotations",
        "report_folder": "sliding_window_reports",
        "confidence_threshold": 0.5,
        "class_names": {
            0: "DieDefect",
            1: "DieDMark",
            2: "Scratch",
            3: "SmallDefect"
        },
        "sliding_window_config": {
            "enable_sliding_window": True,
            "slice_height": 640,              # Size of each window
            "slice_width": 640,
            "overlap_height_ratio": 0.2,     # 20% overlap between windows
            "overlap_width_ratio": 0.2,
            "min_image_size_for_slicing": 1024,  # Only use sliding window for images >= 1024px
            "postprocess_match_threshold": 0.5,  # NMS threshold for merging detections
            "postprocess_match_metric": "IOS",   # IOS or IOU
            "postprocess_class_agnostic": False  # Class-specific NMS
        }
    }
    
    # Validate paths
    if not os.path.exists(config["model_path"]):
        print(f"❌ Model not found: {config['model_path']}")
        print("Please update the model_path in this script")
        return
    
    if not os.path.exists(config["input_folder"]):
        print(f"❌ Input folder not found: {config['input_folder']}")
        print("Please update the input_folder in this script")
        return
    
    try:
        # Initialize generator with sliding window support
        print(f"🔄 Initializing model with sliding window support...")
        generator = AutoAnnotationGenerator(
            config["model_path"], 
            config["confidence_threshold"],
            sliding_window_config=config["sliding_window_config"]
        )
        
        # Set custom class names
        generator.set_class_names(config["class_names"])
        
        # Check if SAHI is available
        if hasattr(generator, 'sahi_model') and generator.sahi_model is not None:
            print("✅ SAHI model initialized successfully")
        else:
            print("⚠️  SAHI not available or failed to initialize")
            print("   Falling back to regular inference")
        
        # Display configuration
        print(f"\n📋 Configuration:")
        print(f"   Model: {config['model_path']}")
        print(f"   Input: {config['input_folder']}")
        print(f"   Slice size: {config['sliding_window_config']['slice_width']}x{config['sliding_window_config']['slice_height']}")
        print(f"   Overlap: {config['sliding_window_config']['overlap_width_ratio']*100}%")
        print(f"   Min size for slicing: {config['sliding_window_config']['min_image_size_for_slicing']}px")
        
        # Process images
        print(f"\n🔄 Processing images with sliding window inference...")
        generator.process_images(
            config["input_folder"],
            config["output_folder"],
            image_extensions=['.jpg', '.jpeg', '.png', '.bmp', '.tiff'],
            report_folder=config["report_folder"],
            generate_reports=True
        )
        
        print(f"\n✅ Sliding window processing completed!")
        print(f"📁 Annotations: {config['output_folder']}")
        print(f"🖼️  Reports: {config['report_folder']}")
        
        # Count results
        output_path = Path(config["output_folder"])
        if output_path.exists():
            json_files = list(output_path.glob("*.json"))
            print(f"📄 Generated {len(json_files)} annotation files")
        
        report_path = Path(config["report_folder"])
        if report_path.exists():
            report_files = list(report_path.glob("*_report.jpg"))
            print(f"🖼️  Generated {len(report_files)} visual reports")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

def compare_inference_modes():
    """Compare regular vs sliding window inference"""
    
    print("\n🔬 Inference Mode Comparison")
    print("=" * 60)
    
    comparison = [
        ("Feature", "Regular Inference", "Sliding Window"),
        ("---", "---", "---"),
        ("Memory Usage", "High for large images", "Low (processes patches)"),
        ("Speed", "Fast for small images", "Slower (multiple passes)"),
        ("Small Objects", "May miss small objects", "Better small object detection"),
        ("Large Images", "May run out of memory", "Handles any size"),
        ("Overlap Handling", "N/A", "NMS merges overlapping detections"),
        ("GPU Memory", "Scales with image size", "Constant (patch size)"),
        ("Best For", "< 1024px images", "> 1024px images, small objects")
    ]
    
    for row in comparison:
        print(f"{row[0]:<20} | {row[1]:<25} | {row[2]:<30}")

def main():
    """Run sliding window examples"""
    
    print("🪟 SAHI Sliding Window Integration")
    print("=" * 60)
    print("This example demonstrates sliding window inference for:")
    print("• Large images (> 1024px)")
    print("• Small object detection")
    print("• Memory-efficient processing")
    print("")
    
    # Run the example
    sliding_window_example()
    
    # Show comparison
    compare_inference_modes()
    
    print("\n💡 Tips for Sliding Window:")
    print("• Use for images larger than 1024x1024 pixels")
    print("• Increase overlap for better small object detection")
    print("• Adjust slice size based on your objects' typical size")
    print("• Monitor GPU memory usage - should be constant")
    print("• Use IOS metric for overlapping objects, IOU for separate objects")

if __name__ == "__main__":
    main()
