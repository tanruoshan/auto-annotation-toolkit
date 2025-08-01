#!/usr/bin/env python3
"""
Example: Generate Visual Reports
Demonstrates how to use the report generation functionality
"""

import os
import sys
from pathlib import Path

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from auto_annotation_generator import AutoAnnotationGenerator

def main():
    """Example of generating visual reports with annotations"""
    
    # Configuration
    model_path = "path/to/your/model.pt"  # Update this path
    input_folder = "path/to/your/images"   # Update this path
    output_folder = "annotations"
    report_folder = "reports"
    confidence_threshold = 0.5
    
    print("Visual Report Generation Example")
    print("=" * 50)
    
    # Verify paths exist
    if not os.path.exists(model_path):
        print(f"❌ Model not found: {model_path}")
        print("Please update the model_path in this script")
        return
    
    if not os.path.exists(input_folder):
        print(f"❌ Input folder not found: {input_folder}")
        print("Please update the input_folder in this script")
        return
    
    try:
        # Initialize generator
        print(f"🔄 Loading model: {model_path}")
        generator = AutoAnnotationGenerator(model_path, confidence_threshold)
        
        # Set custom class names (optional)
        class_names = {
            0: "DieDefect",
            1: "DieDMark",
            2: "Scratch",
            3: "Crack"
        }
        generator.set_class_names(class_names)
        
        # Process images with report generation
        print(f"🔄 Processing images...")
        print(f"   Input: {input_folder}")
        print(f"   Annotations: {output_folder}")
        print(f"   Reports: {report_folder}")
        
        generator.process_images(
            input_folder=input_folder,
            output_folder=output_folder,
            image_extensions=['.jpg', '.jpeg', '.png', '.bmp'],
            report_folder=report_folder,
            generate_reports=True
        )
        
        print(f"\n✅ Processing completed!")
        print(f"📁 Check the '{report_folder}' folder for visual reports")
        print(f"📄 Check the '{output_folder}' folder for JSON annotations")
        
        # List generated files
        report_path = Path(report_folder)
        if report_path.exists():
            report_files = list(report_path.glob("*_report.jpg"))
            print(f"\n📊 Generated {len(report_files)} report images:")
            for i, report_file in enumerate(report_files[:5]):  # Show first 5
                print(f"   {i+1}. {report_file.name}")
            if len(report_files) > 5:
                print(f"   ... and {len(report_files) - 5} more")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return

if __name__ == "__main__":
    main()
