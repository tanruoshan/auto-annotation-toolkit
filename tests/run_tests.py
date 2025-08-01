#!/usr/bin/env python3
"""
Test Runner for Auto Annotation Toolkit
Runs all tests to verify functionality

Usage:
    python tests/run_tests.py
    python tests/run_tests.py --model path/to/model.pt
"""

import sys
import os
import argparse

# Add parent src directory to path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
src_path = os.path.join(parent_dir, 'src')
sys.path.append(src_path)

def run_all_tests(model_path: str = None):
    """Run all available tests"""
    
    print("🧪 AUTO ANNOTATION TOOLKIT TEST SUITE")
    print("=" * 60)
    
    tests_passed = 0
    tests_failed = 0
    
    # Test 1: Basic imports and functionality
    print("\n1️⃣ Testing Basic Imports...")
    try:
        from quick_test import test_basic_import
        if test_basic_import():
            tests_passed += 1
            print("✅ Basic imports test passed")
        else:
            tests_failed += 1
            print("❌ Basic imports test failed")
    except Exception as e:
        tests_failed += 1
        print(f"❌ Basic imports test error: {e}")
    
    # Test 2: Model verification fixes
    print("\n2️⃣ Testing Model Verification Fixes...")
    try:
        from test_fixes import test_auto_annotation_generator, test_model_verifier
        success1 = test_auto_annotation_generator()
        success2 = test_model_verifier()
        
        if success1 and success2:
            tests_passed += 1
            print("✅ Model verification fixes test passed")
        else:
            tests_failed += 1
            print("❌ Model verification fixes test failed")
    except Exception as e:
        tests_failed += 1
        print(f"❌ Model verification fixes test error: {e}")
    
    # Test 3: Input size detection (if model provided)
    if model_path and os.path.exists(model_path):
        print("\n3️⃣ Testing Input Size Detection...")
        try:
            from test_input_size import test_input_size_detection
            # Temporarily update the model path in the test
            import test_input_size
            original_path = test_input_size.test_input_size_detection.__code__.co_consts
            
            # Run the test
            if test_input_size_detection():
                tests_passed += 1
                print("✅ Input size detection test passed")
            else:
                tests_failed += 1
                print("❌ Input size detection test failed")
        except Exception as e:
            tests_failed += 1
            print(f"❌ Input size detection test error: {e}")
    else:
        print("\n3️⃣ Skipping Input Size Detection (no model provided)")
    
    # Test 4: Visual report generation
    print("\n4️⃣ Testing Visual Report Generation...")
    try:
        from test_reports import test_report_generation
        if test_report_generation():
            tests_passed += 1
            print("✅ Visual report generation test passed")
        else:
            tests_failed += 1
            print("❌ Visual report generation test failed")
    except Exception as e:
        tests_failed += 1
        print(f"❌ Visual report generation test error: {e}")
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    print(f"✅ Tests Passed: {tests_passed}")
    print(f"❌ Tests Failed: {tests_failed}")
    print(f"📊 Total Tests: {tests_passed + tests_failed}")
    
    if tests_failed == 0:
        print("\n🎉 ALL TESTS PASSED! The toolkit is ready to use.")
        return True
    else:
        print(f"\n⚠️ {tests_failed} test(s) failed. Check the errors above.")
        return False

def main():
    parser = argparse.ArgumentParser(description='Run Auto Annotation Toolkit tests')
    parser.add_argument('--model', '-m', help='Path to model file for testing')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    
    args = parser.parse_args()
    
    if args.verbose:
        print("🔧 Running in verbose mode")
    
    success = run_all_tests(args.model)
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
