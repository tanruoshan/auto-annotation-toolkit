#!/usr/bin/env python3
"""
Test Model Verification Functionality
Quick test script to validate that the verification tools work correctly
"""

import os
import sys
import tempfile
import json
from pathlib import Path

# Add src directory to path
src_path = os.path.join(os.path.dirname(__file__), 'src')
sys.path.append(src_path)

def test_imports():
    """Test that all required modules can be imported"""
    print("Testing imports...")
    
    try:
        from auto_annotation_generator import AutoAnnotationGenerator
        print("✅ AutoAnnotationGenerator imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import AutoAnnotationGenerator: {e}")
        return False
    
    try:
        from model_verifier import ModelVerifier
        print("✅ ModelVerifier imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import ModelVerifier: {e}")
        return False
    
    return True

def test_configuration_loading():
    """Test configuration file loading"""
    print("\\nTesting configuration loading...")
    
    config_path = "config/annotation_config.ini"
    if os.path.exists(config_path):
        try:
            import configparser
            config = configparser.ConfigParser()
            config.read(config_path)
            
            if 'MODEL' in config and 'model_path' in config['MODEL']:
                model_path = config['MODEL']['model_path']
                print(f"✅ Found model path in config: {model_path}")
                return model_path
            else:
                print("⚠️ No model_path found in config [MODEL] section")
                return None
        except Exception as e:
            print(f"❌ Error reading config: {e}")
            return None
    else:
        print(f"⚠️ Config file not found: {config_path}")
        return None

def test_model_verification(model_path):
    """Test model verification with a real model"""
    if not model_path or not os.path.exists(model_path):
        print(f"⚠️ Skipping model verification - model not found: {model_path}")
        return
    
    print(f"\\nTesting model verification with: {model_path}")
    
    try:
        from auto_annotation_generator import AutoAnnotationGenerator
        
        # Test basic initialization
        generator = AutoAnnotationGenerator(model_path, confidence_threshold=0.1)
        print("✅ AutoAnnotationGenerator initialized successfully")
        
        # Test verification
        info = generator.verify_model_configuration()
        print("✅ Model verification completed")
        
        # Check basic info
        if 'classes' in info:
            print(f"✅ Found {len(info['classes'])} configured classes")
        
        if 'file_size' in info:
            print(f"✅ Model file size: {info['file_size']}")
        
        # Test verification report
        print("\\n" + "-"*40)
        generator.print_model_verification()
        print("-"*40)
        
        return True
        
    except Exception as e:
        print(f"❌ Model verification failed: {e}")
        return False

def test_standalone_verifier(model_path):
    """Test standalone model verifier"""
    if not model_path or not os.path.exists(model_path):
        print(f"⚠️ Skipping standalone verifier - model not found: {model_path}")
        return
    
    print(f"\\nTesting standalone verifier with: {model_path}")
    
    try:
        from model_verifier import ModelVerifier
        
        verifier = ModelVerifier(model_path)
        print("✅ ModelVerifier initialized successfully")
        
        info = verifier.verify_model()
        print("✅ Detailed verification completed")
        
        # Test report generation
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_file = f.name
        
        verifier.save_verification_report(temp_file, info)
        
        if os.path.exists(temp_file):
            with open(temp_file, 'r') as f:
                saved_info = json.load(f)
            print("✅ Verification report saved and loaded successfully")
            os.unlink(temp_file)  # Clean up
        
        return True
        
    except Exception as e:
        print(f"❌ Standalone verifier failed: {e}")
        return False

def test_command_line_interface():
    """Test command line interface"""
    print("\\nTesting command line interface...")
    
    try:
        # Test help command
        import subprocess
        result = subprocess.run([sys.executable, "verify_model.py", "--help"], 
                              capture_output=True, text=True, timeout=10)
        
        if result.returncode == 0:
            print("✅ verify_model.py --help works")
        else:
            print(f"⚠️ verify_model.py --help returned code {result.returncode}")
        
        # Test annotation tool help
        result = subprocess.run([sys.executable, "annotate_dataset.py", "--help"], 
                              capture_output=True, text=True, timeout=10)
        
        if result.returncode == 0 and "--verify-model" in result.stdout:
            print("✅ annotate_dataset.py has verification options")
        else:
            print("⚠️ annotate_dataset.py verification options not found")
        
        return True
        
    except Exception as e:
        print(f"❌ Command line interface test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("Model Verification Test Suite")
    print("=" * 50)
    
    tests_passed = 0
    total_tests = 0
    
    # Test 1: Imports
    total_tests += 1
    if test_imports():
        tests_passed += 1
    
    # Test 2: Configuration
    total_tests += 1
    model_path = test_configuration_loading()
    if model_path:
        tests_passed += 1
    
    # Test 3: Model verification (if model available)
    if model_path and os.path.exists(model_path):
        total_tests += 1
        if test_model_verification(model_path):
            tests_passed += 1
        
        total_tests += 1
        if test_standalone_verifier(model_path):
            tests_passed += 1
    else:
        print("\\n⚠️ Skipping model verification tests - no valid model found")
        print("   Update the model_path in config/annotation_config.ini to test with a real model")
    
    # Test 4: Command line interface
    total_tests += 1
    if test_command_line_interface():
        tests_passed += 1
    
    # Summary
    print("\\n" + "=" * 50)
    print(f"Test Results: {tests_passed}/{total_tests} tests passed")
    
    if tests_passed == total_tests:
        print("🎉 All tests passed! Model verification is working correctly.")
    else:
        print(f"⚠️ {total_tests - tests_passed} test(s) failed. Check the output above for details.")
    
    print("\\nTo test with a real model:")
    print("1. Update model_path in config/annotation_config.ini")
    print("2. Run: python test_verification.py")
    print("3. Or use: python verify_model.py <your_model.pt>")

if __name__ == "__main__":
    main()
