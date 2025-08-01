"""
Test Input Size Detection
"""
import sys
import os

sys.path.append('../src')

def test_input_size_detection():
    model_path = r"C:\Users\ruoshant\Downloads\initial.pt"
    
    print("🔍 Testing Input Size Detection")
    print("=" * 40)
    
    if not os.path.exists(model_path):
        print(f"❌ Model file not found: {model_path}")
        return
    
    try:
        # Test with AutoAnnotationGenerator
        from auto_annotation_generator import AutoAnnotationGenerator
        
        print("Testing with AutoAnnotationGenerator...")
        generator = AutoAnnotationGenerator(model_path, confidence_threshold=0.1)
        info = generator.verify_model_configuration()
        
        print(f"✅ Model loaded successfully")
        print(f"📐 Detected input size: {info.get('input_size', 'Not found')}")
        print(f"🎨 Input channels: {info.get('input_channels', 'Not found')}")
        print(f"🖼️  Input format: {info.get('input_format', 'Not found')}")
        
        if 'input_size_source' in info:
            print(f"ℹ️  Source: {info['input_size_source']}")
        
        print("\n" + "="*40)
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

if __name__ == "__main__":
    test_input_size_detection()
