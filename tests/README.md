# Tests

This folder contains all test files for the Auto Annotation Toolkit.

## Test Files

- **`quick_test.py`** - Quick import and basic functionality test
- **`test_fixes.py`** - Tests for model verification fixes and logger issues
- **`test_input_size.py`** - Tests input size detection functionality
- **`test_installation.py`** - Tests package installation and dependencies
- **`test_verification.py`** - Comprehensive model verification tests
- **`run_tests.py`** - Test runner script to execute all tests

## Running Tests

### Run All Tests
```bash
python tests/run_tests.py
```

### Run All Tests with a Model
```bash
python tests/run_tests.py --model "C:\path\to\your\model.pt"
```

### Run Individual Tests
```bash
# Quick functionality test
python tests/quick_test.py

# Input size detection test
python tests/test_input_size.py

# Model verification fixes test
python tests/test_fixes.py
```

## Test from Root Directory

If you're in the main project directory, you can run:
```bash
# Run all tests
python tests/run_tests.py

# Run with your model
python tests/run_tests.py --model "C:\Users\ruoshant\Downloads\initial.pt"
```

## What Tests Check

1. **Import Functionality** - Ensures all modules can be imported correctly
2. **Model Loading** - Tests model loading with different formats (.pt, .onnx)
3. **Class Detection** - Verifies class information extraction from models
4. **Input Size Detection** - Tests patch size detection from model metadata
5. **Error Handling** - Ensures proper error handling and logging
6. **Configuration** - Tests config file parsing and validation

## Expected Output

When all tests pass, you should see:
```
🎉 ALL TESTS PASSED! The toolkit is ready to use.
```

If tests fail, check the error messages for troubleshooting guidance.
