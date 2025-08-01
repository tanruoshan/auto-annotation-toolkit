"""
Model Verification Examples
Demonstrates how to use the model verification tools
"""

import sys
import os

# Add src directory to path
src_path = os.path.join(os.path.dirname(__file__), '..', 'src')
sys.path.append(src_path)

from auto_annotation_generator import AutoAnnotationGenerator
from model_verifier import ModelVerifier

def example_basic_verification():
    """Example: Basic model verification with AutoAnnotationGenerator"""
    print("Example 1: Basic Model Verification")
    print("-" * 40)
    
    # Replace with your model path
    model_path = "path/to/your/model.pt"
    
    try:
        # Initialize generator
        generator = AutoAnnotationGenerator(model_path, confidence_threshold=0.5)
        
        # Print verification report
        generator.print_model_verification()
        
        # Get verification info as dictionary
        info = generator.verify_model_configuration()
        
        # Check for specific issues
        if 'class_count_mismatch' in info:
            print("⚠️ Warning: Class count mismatch detected!")
            print("Consider updating your configuration file.")
        
        print("✅ Basic verification completed")
        
    except Exception as e:
        print(f"❌ Error: {e}")

def example_detailed_verification():
    """Example: Detailed model verification with ModelVerifier"""
    print("\\nExample 2: Detailed Model Verification")
    print("-" * 40)
    
    # Replace with your model path
    model_path = "path/to/your/model.pt"
    
    try:
        # Initialize verifier
        verifier = ModelVerifier(model_path)
        
        # Get detailed info
        info = verifier.verify_model()
        
        # Print report
        verifier.print_verification_report(info)
        
        # Save report to file
        verifier.save_verification_report("model_report.json", info)
        
        print("✅ Detailed verification completed")
        print("📄 Report saved to model_report.json")
        
    except Exception as e:
        print(f"❌ Error: {e}")

def example_class_configuration_check():
    """Example: Check and update class configuration"""
    print("\\nExample 3: Class Configuration Check")
    print("-" * 40)
    
    model_path = "path/to/your/model.pt"
    
    try:
        generator = AutoAnnotationGenerator(model_path)
        
        # Get model info
        info = generator.verify_model_configuration()
        
        print("Current configuration:")
        print(f"  Configured classes: {len(info['classes'])}")
        for class_id, class_name in sorted(info['classes'].items()):
            print(f"    {class_id}: {class_name}")
        
        # Check if model has different classes
        if 'model_class_names' in info:
            print("\\nModel's built-in classes:")
            model_names = info['model_class_names']
            if isinstance(model_names, dict):
                for k, v in sorted(model_names.items()):
                    print(f"    {k}: {v}")
            elif isinstance(model_names, list):
                for i, name in enumerate(model_names):
                    print(f"    {i}: {name}")
        
        # Suggest updates
        if 'class_names_mismatch' in info:
            print("\\n💡 Suggestion: Update your annotation_config.ini [CLASSES] section:")
            if 'model_classes' in info:
                print("[CLASSES]")
                for k, v in sorted(info['model_classes'].items()):
                    print(f"{k} = {v}")
        
    except Exception as e:
        print(f"❌ Error: {e}")

def example_onnx_model_verification():
    """Example: ONNX model verification"""
    print("\\nExample 4: ONNX Model Verification")
    print("-" * 40)
    
    model_path = "path/to/your/model.onnx"
    
    try:
        verifier = ModelVerifier(model_path)
        info = verifier.verify_model()
        
        print("ONNX Model Information:")
        print(f"  Input shape: {info.get('input_shape', 'Unknown')}")
        print(f"  Output shape: {info.get('output_shape', 'Unknown')}")
        
        if 'inferred_num_classes' in info:
            print(f"  Inferred classes: {info['inferred_num_classes']}")
            print("\\n💡 Note: Class names need to be configured manually for ONNX models")
            print("Update your [CLASSES] section in annotation_config.ini")
        
    except Exception as e:
        print(f"❌ Error: {e}")

def example_batch_model_verification():
    """Example: Verify multiple models"""
    print("\\nExample 5: Batch Model Verification")
    print("-" * 40)
    
    model_paths = [
        "path/to/model1.pt",
        "path/to/model2.pt", 
        "path/to/model3.onnx"
    ]
    
    for i, model_path in enumerate(model_paths, 1):
        print(f"\\nVerifying Model {i}: {model_path}")
        print("-" * 30)
        
        try:
            if not os.path.exists(model_path):
                print(f"⚠️ Model file not found: {model_path}")
                continue
                
            verifier = ModelVerifier(model_path)
            info = verifier.verify_model()
            
            # Quick summary
            print(f"✅ Type: {info['model_type']}")
            print(f"✅ Size: {info['file_size']}")
            print(f"✅ Classes: {len(info.get('classes', {}))}")
            
            if 'warning' in info:
                print(f"⚠️ Warning: {info['warning']}")
                
        except Exception as e:
            print(f"❌ Error: {e}")

if __name__ == "__main__":
    print("Model Verification Examples")
    print("=" * 50)
    
    # Update these paths to your actual model files
    print("📝 Note: Update the model paths in this script before running")
    print()
    
    # Run examples
    example_basic_verification()
    example_detailed_verification()
    example_class_configuration_check()
    example_onnx_model_verification()
    example_batch_model_verification()
    
    print("\\n" + "=" * 50)
    print("🎯 Examples completed!")
    print("\\nTo use with your models:")
    print("1. Update the model_path variables in each example")
    print("2. Run individual examples or the entire script")
    print("3. Check the generated reports and warnings")
