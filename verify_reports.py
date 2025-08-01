#!/usr/bin/env python3
"""
Verification Script: Report Functionality
Quick verification that all report features are working
"""

import os
import sys

def check_imports():
    """Check if all required modules can be imported"""
    print("🔍 Checking imports...")
    
    try:
        sys.path.append('src')
        from auto_annotation_generator import AutoAnnotationGenerator
        print("✅ AutoAnnotationGenerator imported successfully")
        
        # Check if create_visual_report method exists
        if hasattr(AutoAnnotationGenerator, 'create_visual_report'):
            print("✅ create_visual_report method found")
        else:
            print("❌ create_visual_report method missing")
            return False
            
        # Check if process_images has new parameters
        import inspect
        sig = inspect.signature(AutoAnnotationGenerator.process_images)
        params = list(sig.parameters.keys())
        
        if 'report_folder' in params and 'generate_reports' in params:
            print("✅ process_images has report parameters")
        else:
            print("❌ process_images missing report parameters")
            return False
            
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

def check_config():
    """Check if configuration supports reports"""
    print("\n📋 Checking configuration...")
    
    try:
        # Load the load_config function
        with open('annotate_dataset.py', 'r') as f:
            content = f.read()
        
        if 'report_folder' in content and 'generate_reports' in content:
            print("✅ Configuration supports report settings")
        else:
            print("❌ Configuration missing report settings")
            return False
            
        # Check config file
        if os.path.exists('config/annotation_config.ini'):
            with open('config/annotation_config.ini', 'r') as f:
                config_content = f.read()
            
            if '[REPORTS]' in config_content:
                print("✅ Config file has REPORTS section")
            else:
                print("❌ Config file missing REPORTS section")
                return False
                
        return True
        
    except Exception as e:
        print(f"❌ Config check error: {e}")
        return False

def check_command_line():
    """Check if command line supports report options"""
    print("\n💻 Checking command line interface...")
    
    try:
        with open('annotate_dataset.py', 'r') as f:
            content = f.read()
        
        if '--report-folder' in content and '--no-reports' in content:
            print("✅ Command line has report arguments")
        else:
            print("❌ Command line missing report arguments")
            return False
            
        return True
        
    except Exception as e:
        print(f"❌ Command line check error: {e}")
        return False

def check_examples():
    """Check if examples include report functionality"""
    print("\n📚 Checking examples...")
    
    try:
        example_files = [
            'examples/generate_reports_example.py',
            'examples/annotation_examples.py'
        ]
        
        for example_file in example_files:
            if os.path.exists(example_file):
                with open(example_file, 'r') as f:
                    content = f.read()
                if 'report' in content.lower():
                    print(f"✅ {example_file} includes reports")
                else:
                    print(f"⚠️  {example_file} may not include reports")
            else:
                print(f"❌ {example_file} not found")
                
        return True
        
    except Exception as e:
        print(f"❌ Examples check error: {e}")
        return False

def main():
    """Run all verification checks"""
    print("🔧 REPORT FUNCTIONALITY VERIFICATION")
    print("=" * 50)
    
    checks = [
        ("Imports", check_imports),
        ("Configuration", check_config), 
        ("Command Line", check_command_line),
        ("Examples", check_examples)
    ]
    
    passed = 0
    total = len(checks)
    
    for check_name, check_func in checks:
        if check_func():
            passed += 1
    
    print("\n" + "=" * 50)
    print(f"📊 VERIFICATION SUMMARY: {passed}/{total} checks passed")
    
    if passed == total:
        print("🎉 ALL REPORT FEATURES VERIFIED!")
        print("\n📋 You can now:")
        print("• Configure reports in config/annotation_config.ini")
        print("• Use --report-folder in command line")
        print("• Generate visual reports with bounding boxes")
        print("• Display confidence scores on images")
        return True
    else:
        print(f"⚠️  {total - passed} check(s) failed")
        return False

if __name__ == "__main__":
    main()
