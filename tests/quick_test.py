"""
Quick test for model verification
"""
import sys
import os

# Add src to path (now we're in tests/ subfolder)
sys.path.append('../src')

def test_basic_import():
    try:
        from auto_annotation_generator import AutoAnnotationGenerator
        print("✅ AutoAnnotationGenerator imported successfully")
        return True
    except Exception as e:
        print(f"❌ Import failed: {e}")
        return False

def test_model_loading():
    model_path = r"C:\Users\ruoshant\Downloads\initial.pt"
    
    if not os.path.exists(model_path):
        print(f"❌ Model file not found: {model_path}")
        return False
    
    try:
        from auto_annotation_generator import AutoAnnotationGenerator
        print(f"📁 Loading model: {model_path}")
        
        generator = AutoAnnotationGenerator(model_path, confidence_threshold=0.1)
        print("✅ Model loaded successfully!")
        
        # Get basic info
        info = generator.verify_model_configuration()
        print(f"📊 Model type: {info.get('model_type')}")
        print(f"📊 File exists: {info.get('file_exists')}")
        print(f"📊 Classes configured: {len(info.get('classes', {}))}")
        
        # Show configured classes
        classes = info.get('classes', {})
        if classes:
            print("📋 Configured classes:")
            for class_id, class_name in sorted(classes.items()):
                print(f"   {class_id}: {class_name}")
        
        # Check for warnings
        if 'warning' in info:
            print(f"⚠️ Warning: {info['warning']}")
        
        if 'class_count_mismatch' in info:
            print("⚠️ Class count mismatch detected")
        
        if 'model_classes' in info:
            print("📋 Model's actual classes:")
            model_classes = info['model_classes']
            if isinstance(model_classes, dict):
                for k, v in sorted(model_classes.items()):
                    print(f"   {k}: {v}")
        
        return True
        
    except Exception as e:
        print(f"❌ Model loading failed: {e}")
        return False

if __name__ == "__main__":
    print("🔍 Quick Model Verification Test")
    print("=" * 40)
    
    if test_basic_import():
        test_model_loading()
    
    print("=" * 40)
    print("Test completed!")
