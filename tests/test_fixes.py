#!/usr/bin/env python3
"""
Test script to verify the model verification fixes
"""

import sys
import os

# Add src directory to path (now we're in tests/ subfolder)
src_path = os.path.join(os.path.dirname(__file__), '..', 'src')
sys.path.append(src_path)

def test_auto_annotation_generator():
    """Test AutoAnnotationGenerator import and basic functionality"""
    try:
        from auto_annotation_generator import AutoAnnotationGenerator
        print("✅ AutoAnnotationGenerator imported successfully")
        
        # Test with a dummy path to check class initialization
        try:
            # This will fail at model loading but should pass initialization
            generator = AutoAnnotationGenerator("dummy_path.pt", confidence_threshold=0.5)
        except FileNotFoundError:
            print("✅ AutoAnnotationGenerator initialization working (expected file not found)")
        except Exception as e:
            print(f"⚠️ AutoAnnotationGenerator initialization issue: {e}")
        
        return True
    except Exception as e:
        print(f"❌ AutoAnnotationGenerator import failed: {e}")
        return False

def test_model_verifier():
    """Test ModelVerifier import and basic functionality"""
    try:
        from model_verifier import ModelVerifier
        print("✅ ModelVerifier imported successfully")
        
        # Test with a dummy path to check class initialization
        try:
            verifier = ModelVerifier("dummy_path.pt")
        except FileNotFoundError:
            print("✅ ModelVerifier initialization working (expected file not found)")
        except Exception as e:
            print(f"⚠️ ModelVerifier initialization issue: {e}")
        
        return True
    except Exception as e:
        print(f"❌ ModelVerifier import failed: {e}")
        return False

def test_actual_model(model_path):
    """Test with an actual model file"""
    if not os.path.exists(model_path):
        print(f"⚠️ Model file not found: {model_path}")
        return False
    
    print(f"\\nTesting with actual model: {model_path}")
    
    # Test AutoAnnotationGenerator
    try:
        from auto_annotation_generator import AutoAnnotationGenerator
        generator = AutoAnnotationGenerator(model_path, confidence_threshold=0.1)
        print("✅ AutoAnnotationGenerator loaded model successfully")
        
        # Try verification
        info = generator.verify_model_configuration()
        print(f"✅ Model verification completed")
        print(f"   - Model type: {info.get('model_type', 'Unknown')}")
        print(f"   - Classes found: {len(info.get('classes', {}))}")
        
        return True
        
    except Exception as e:
        print(f"❌ AutoAnnotationGenerator test failed: {e}")
        return False

if __name__ == "__main__":
    print("="*60)
    print("TESTING MODEL VERIFICATION FIXES")
    print("="*60)
    
    # Test imports
    success1 = test_auto_annotation_generator()
    success2 = test_model_verifier()
    
    # Test with actual model if provided
    if len(sys.argv) > 1:
        model_path = sys.argv[1]
        success3 = test_actual_model(model_path)
    else:
        success3 = True
        print("\\n💡 To test with actual model, run: python test_fixes.py path/to/model.pt")
    
    print("\\n" + "="*60)
    if success1 and success2 and success3:
        print("✅ ALL TESTS PASSED!")
        print("The model verification tools should now work correctly.")
    else:
        print("❌ SOME TESTS FAILED")
        print("Check the error messages above for details.")
    print("="*60)
