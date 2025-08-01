"""
Example Usage: Auto Annotation Generator
"""

import sys
import os

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from auto_annotation_generator import AutoAnnotationGenerator

def example_usage():
    """Example of how to use the auto annotation generator"""
    
    # Configuration
    config = {
        "model_path": "../model/gn24_deployment/best.pt",  # or "model.onnx"
        "input_folder": "../images_GN24/CLAHE",
        "output_folder": "../auto_annotations",
        "report_folder": "../auto_annotations_reports",  # New: visual reports
        "confidence_threshold": 0.5,
        "class_names": {
            0: "Good",
            1: "DieDefect", 
            2: "Scratch",
            3: "Crack",
            4: "Contamination"
        }
    }
    
    try:
        # Initialize generator
        generator = AutoAnnotationGenerator(
            config["model_path"], 
            config["confidence_threshold"]
        )
        
        # Set custom class names
        generator.set_class_names(config["class_names"])
        
        # Process all images in folder with report generation
        generator.process_images(
            config["input_folder"], 
            config["output_folder"],
            report_folder=config["report_folder"],
            generate_reports=True
        )
        
        print(f"✅ Auto-annotation completed!")
        print(f"📁 Annotations saved to: {config['output_folder']}")
        print(f"🖼️  Report images saved to: {config['report_folder']}")
        
    except Exception as e:
        print(f"❌ Error: {e}")

def process_single_image_example():
    """Example of processing a single image"""
    import cv2
    import json
    
    # Configuration
    model_path = "../model/gn24_deployment/best.pt"
    image_path = "../images_GN24/CLAHE/test_image.jpg"
    output_path = "test_annotation.json"
    
    try:
        # Initialize generator
        generator = AutoAnnotationGenerator(model_path, confidence_threshold=0.5)
        
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        # Run inference
        detections = generator.predict(image)
        print(f"Found {len(detections)} detections")
        
        # Create annotation
        annotation = generator.create_labelme_annotation(image_path, detections)
        
        # Save annotation
        with open(output_path, 'w') as f:
            json.dump(annotation, f, indent=2)
        
        print(f"✅ Annotation saved to: {output_path}")
        
        # Print detection details
        for i, det in enumerate(detections):
            class_name = generator.class_names.get(det['class_id'], f"class_{det['class_id']}")
            print(f"Detection {i+1}: {class_name} (confidence: {det['confidence']:.3f})")
        
    except Exception as e:
        print(f"❌ Error: {e}")

def batch_process_example():
    """Example of batch processing with custom settings"""
    
    # Multiple input folders
    input_folders = [
        "../images_GN24/CLAHE",
        "../images_GN24/original", 
        "../images_GN24/CLAHE-cropped"
    ]
    
    model_path = "../model/gn24_deployment/best.pt"
    base_output_folder = "batch_annotations"
    
    try:
        generator = AutoAnnotationGenerator(model_path, confidence_threshold=0.6)
        
        # Custom class names for your specific use case
        custom_classes = {
            0: "Good",
            1: "DieDefect",
            2: "DieCrack", 
            3: "DieChip",
            4: "Contamination",
            5: "Scratch"
        }
        generator.set_class_names(custom_classes)
        
        # Process each folder
        for folder in input_folders:
            folder_name = folder.split('/')[-1]  # Get folder name
            output_folder = f"{base_output_folder}/{folder_name}_annotations"
            
            print(f"Processing folder: {folder}")
            generator.process_images(folder, output_folder)
            print(f"Completed: {folder} -> {output_folder}")
        
        print("✅ Batch processing completed!")
        
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    print("Auto Annotation Examples")
    print("=" * 40)
    
    # Choose which example to run
    example_choice = input("Choose example (1: Basic, 2: Single Image, 3: Batch): ")
    
    if example_choice == "1":
        example_usage()
    elif example_choice == "2":
        process_single_image_example()
    elif example_choice == "3":
        batch_process_example()
    else:
        print("Running basic example...")
        example_usage()
