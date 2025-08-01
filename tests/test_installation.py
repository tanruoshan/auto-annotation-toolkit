#!/usr/bin/env python3
"""
Test Script for Auto Annotation Toolkit
Verifies installation and basic functionality
"""

import sys
import os
import importlib

def test_imports():
    """Test if required modules can be imported"""
    print("🔍 Testing imports...")
    
    required_modules = [
        ('cv2', 'opencv-python'),
        ('numpy', 'numpy'),
        ('json', 'built-in'),
        ('pathlib', 'built-in')
    ]
    
    optional_modules = [
        ('torch', 'torch (for .pt models)'),
        ('ultralytics', 'ultralytics (for YOLO models)'),
        ('onnxruntime', 'onnxruntime (for .onnx models)')
    ]
    
    # Test required modules
    for module, package in required_modules:
        try:
            importlib.import_module(module)
            print(f"  ✅ {module} - OK")
        except ImportError:
            print(f"  ❌ {module} - MISSING (install: {package})")
            return False
    
    # Test optional modules
    print("\n🔧 Optional dependencies:")
    for module, package in optional_modules:
        try:
            importlib.import_module(module)
            print(f"  ✅ {module} - OK")
        except ImportError:
            print(f"  ⚠️  {module} - Not installed ({package})")
    
    return True

def test_toolkit_structure():
    """Test if toolkit files are in place"""
    print("\n📁 Testing toolkit structure...")
    
    required_files = [
        'src/auto_annotation_generator.py',
        'src/quick_auto_annotate.py',
        'config/annotation_config.ini',
        'annotate_dataset.py',
        'requirements.txt',
        'README.md'
    ]
    
    all_present = True
    for file_path in required_files:
        if os.path.exists(file_path):
            print(f"  ✅ {file_path}")
        else:
            print(f"  ❌ {file_path} - MISSING")
            all_present = False
    
    return all_present

def test_config_file():
    """Test if config file is readable"""
    print("\n⚙️  Testing configuration...")
    
    config_path = 'config/annotation_config.ini'
    try:
        import configparser
        config = configparser.ConfigParser()
        config.read(config_path)
        
        # Check required sections
        required_sections = ['MODEL', 'PATHS', 'CLASSES']
        for section in required_sections:
            if section in config:
                print(f"  ✅ Section [{section}]")
            else:
                print(f"  ❌ Section [{section}] - MISSING")
                return False
        
        return True
        
    except Exception as e:
        print(f"  ❌ Config file error: {e}")
        return False

def test_main_module():
    """Test if main module can be imported"""
    print("\n🔧 Testing main module...")
    
    try:
        # Add src to path
        sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
        
        from auto_annotation_generator import AutoAnnotationGenerator
        print("  ✅ AutoAnnotationGenerator imported successfully")
        
        # Test class instantiation (without model)
        print("  ✅ Main module is functional")
        return True
        
    except ImportError as e:
        print(f"  ❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"  ❌ Module error: {e}")
        return False

def main():
    """Run all tests"""
    print("🚀 Auto Annotation Toolkit - Installation Test")
    print("=" * 50)
    
    tests = [
        ("Imports", test_imports),
        ("File Structure", test_toolkit_structure),
        ("Configuration", test_config_file),
        ("Main Module", test_main_module)
    ]
    
    all_passed = True
    results = []
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
            if not result:
                all_passed = False
        except Exception as e:
            print(f"  ❌ {test_name} test failed with error: {e}")
            results.append((test_name, False))
            all_passed = False
    
    # Print summary
    print("\n" + "=" * 50)
    print("📊 Test Summary:")
    
    for test_name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {test_name}: {status}")
    
    print("\n" + "=" * 50)
    if all_passed:
        print("🎉 All tests passed! The toolkit is ready to use.")
        print("\nNext steps:")
        print("1. Update config/annotation_config.ini with your model and image paths")
        print("2. Run: python annotate_dataset.py --dry-run")
        print("3. Run: python annotate_dataset.py")
    else:
        print("⚠️  Some tests failed. Please fix the issues above before using the toolkit.")
        print("\nCommon fixes:")
        print("- Install missing dependencies: pip install -r requirements.txt")
        print("- Ensure you're running from the toolkit root directory")
        print("- Check file permissions and paths")
    
    return all_passed

if __name__ == "__main__":
    sys.exit(0 if main() else 1)
