# Auto Annotation Toolkit

A powerful toolkit for generating LabelMe format JSON annotations using trained models (.pt or .onnx). Perfect for creating training datasets from your existing computer vision models.

## 🚀 Features

- **Multiple Model Support**: Works with PyTorch (.pt) and ONNX (.onnx) models
- **LabelMe Format**: Generates standard LabelMe JSON annotations
- **Visual Reports**: Generate annotated images with confidence scores and bounding boxes
- **Model Verification**: Verify model class configuration and detect mismatches
- **Batch Processing**: Process entire folders of images automatically
- **Command Line Interface**: Easy-to-use CLI with configuration file support
- **Customizable Classes**: Map model outputs to custom class names
- **Confidence Filtering**: Set minimum confidence thresholds
- **Progress Tracking**: Real-time processing feedback

## 📁 Project Structure

```
auto-annotation-toolkit/
├── src/
│   ├── auto_annotation_generator.py  # Main annotation generator class
│   ├── model_verifier.py            # Model verification utilities
│   └── quick_auto_annotate.py       # Simple standalone script
├── examples/
│   ├── annotation_examples.py       # Usage examples
│   ├── model_verification_examples.py # Model verification examples
│   └── generate_reports_example.py  # Visual report generation example
├── tests/
│   ├── quick_test.py                # Quick functionality test
│   ├── test_fixes.py                # Model verification fixes test
│   ├── test_input_size.py           # Input size detection test
│   ├── test_installation.py         # Installation test
│   ├── test_verification.py         # Comprehensive verification test
│   ├── run_tests.py                 # Test runner script
│   └── README.md                    # Test documentation
├── config/
│   └── annotation_config.ini        # Configuration file
├── annotate_dataset.py              # Command line interface
├── verify_model.py                  # Standalone model verification tool
├── requirements.txt                 # Dependencies
└── README.md                        # This file
```

## 🛠️ Installation

1. **Clone or copy this toolkit to your project**

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

3. **For PyTorch models (.pt files):**
```bash
pip install torch ultralytics
```

4. **For ONNX models (.onnx files):**
```bash
pip install onnxruntime
```

## 💻 Usage

### Option 1: Command Line Interface (Recommended)

1. **Configure settings** in `config/annotation_config.ini`:
```ini
[MODEL]
model_path = ../model/gn24_deployment/best.pt
confidence_threshold = 0.5

[PATHS]
input_folder = ../images_GN24/CLAHE
output_folder = ./output_annotations
report_folder = ./output_reports

[CLASSES]
0 = Good
1 = DieDefect
2 = Scratch

[REPORTS]
generate_reports = true
show_confidence_in_reports = true
```

2. **Run the annotation tool:**
```bash
# Basic usage (generates both annotations and reports if configured)
python annotate_dataset.py

# With custom settings
python annotate_dataset.py --model ../model/best.pt --input ../images/ --output ./annotations/ --report-folder ./reports/

# Disable reports even if configured
python annotate_dataset.py --no-reports

# Verify model configuration before processing
python annotate_dataset.py --verify-model

# Only verify model without processing
python annotate_dataset.py --verify-only

# Dry run to preview
python annotate_dataset.py --dry-run
```

### Option 2: Python Script

```python
from src.auto_annotation_generator import AutoAnnotationGenerator

# Initialize
generator = AutoAnnotationGenerator("model/best.pt", confidence_threshold=0.5)

# Set custom class names
generator.set_class_names({0: "Good", 1: "DieDefect", 2: "Scratch"})

# Process images
generator.process_images("input_folder/", "output_annotations/")
```

### Option 3: Quick Test Script

```bash
# Edit paths in src/quick_auto_annotate.py, then run:
python src/quick_auto_annotate.py
```

### Option 4: Model Verification

Before processing images, verify your model's class configuration:

```bash
# Standalone model verification
python verify_model.py model/best.pt

# Verify using config file
python verify_model.py --config config/annotation_config.ini

# Save detailed report
python verify_model.py model/best.pt --output model_report.json

# Quick verification during annotation
python annotate_dataset.py --verify-model
```

### Option 5: Python Model Verification

```python
from src.auto_annotation_generator import AutoAnnotationGenerator
from src.model_verifier import ModelVerifier

# Quick verification
generator = AutoAnnotationGenerator("model/best.pt")
generator.print_model_verification()

# Detailed verification  
verifier = ModelVerifier("model/best.pt")
info = verifier.verify_model()
verifier.print_verification_report(info)
```

### Option 6: Visual Reports

Generate annotated images with bounding boxes and confidence scores for visual verification:

```python
from src.auto_annotation_generator import AutoAnnotationGenerator

# Initialize generator
generator = AutoAnnotationGenerator("model/best.pt", confidence_threshold=0.5)

# Process with visual reports
generator.process_images(
    input_folder="images/",
    output_folder="annotations/",
    report_folder="reports/",         # New: visual reports folder
    generate_reports=True             # New: enable report generation
)
```

**Report Features:**
- 🎯 Bounding boxes with class-specific colors
- 📊 Confidence scores displayed on each detection
- 📈 Detection count summary on each image
- 🖼️ High-quality images for presentation and verification

**Command Line Usage:**
```bash
# Generate reports with annotations
python annotate_dataset.py --report-folder ./reports/

# Disable reports (even if enabled in config)
python annotate_dataset.py --no-reports
```

## 📄 Output Format

The toolkit generates LabelMe-compatible JSON files:

```json
{
  "version": "5.8.1",
  "flags": {},
  "shapes": [
    {
      "label": "DieDefect",
      "points": [[x1, y1], [x2, y2]],
      "group_id": null,
      "description": "confidence: 0.85",
      "shape_type": "rectangle",
      "flags": {},
      "mask": null
    }
  ],
  "imagePath": "image.jpg",
  "imageData": null,
  "imageHeight": 512,
  "imageWidth": 512
}
```

## ⚙️ Configuration

### Model Support
- **YOLO models** (.pt): Automatic detection and bounding boxes
- **ONNX models** (.onnx): Custom inference with configurable post-processing
- **Classification models**: Creates image-level annotations

### Customization
- **Class mapping**: Map model class IDs to meaningful names
- **Confidence thresholds**: Filter low-confidence detections
- **Image formats**: Support for .jpg, .png, .bmp, .tiff
- **Batch processing**: Handle thousands of images efficiently

## 🔧 Examples

Check the `examples/` folder for:
- Basic usage examples
- Single image processing
- Batch processing multiple folders
- Custom class configuration
- Model verification examples

## 🔍 Model Verification

The toolkit includes comprehensive model verification to help you identify and fix configuration issues:

### What Gets Verified:
- **File existence and format** (.pt vs .onnx)
- **Model class information** (number of classes, class names)
- **Input/output shapes** (for ONNX models)
- **Class configuration mismatches** between model and config
- **Model metadata** (training info, version, etc.)

### Common Issues Detected:
- ✅ **Class count mismatch**: Model has different number of classes than configured
- ✅ **Missing class names**: Model has no class name information
- ✅ **Class name mismatch**: Model class names differ from configuration
- ✅ **Single-class models**: Models with only one detection class
- ✅ **ONNX shape inference**: Automatic class count detection from output shape

### Verification Commands:
```bash
# Verify before annotation
python annotate_dataset.py --verify-model

# Standalone verification
python verify_model.py model.pt

# Save verification report
python verify_model.py model.pt --output report.json
```

## 📋 Requirements

- Python 3.7+
- OpenCV
- NumPy
- For PyTorch models: torch, ultralytics
- For ONNX models: onnxruntime

## 🚨 Common Issues

1. **Import errors**: Make sure you're running from the correct directory
2. **Model loading failures**: Check model file paths and formats
3. **Class configuration mismatches**: Use `--verify-model` to check class setup
4. **Memory issues**: Process images in smaller batches for large datasets
5. **Permission errors**: Ensure write access to output directories
6. **Missing dependencies**: Install torch/ultralytics for .pt or onnxruntime for .onnx

## 🛠️ Customization

### For Custom Models

Modify the post-processing in `src/auto_annotation_generator.py`:

```python
def _postprocess_onnx(self, outputs, original_shape):
    # Add your custom post-processing logic here
    detections = []
    # ... your code ...
    return detections
```

### For Different Annotation Formats

Extend the `create_labelme_annotation` method to support other formats like COCO, YOLO, etc.

## 🧪 Testing

The toolkit includes comprehensive tests to ensure reliability:

### Run All Tests:
```bash
python tests/run_tests.py
```

### Test with Your Model:
```bash
python tests/run_tests.py --model "C:\path\to\your\model.pt"
```

### Individual Tests:
```bash
# Quick functionality test
python tests/quick_test.py

# Input size detection test  
python tests/test_input_size.py

# Model verification test
python tests/test_fixes.py
```

The tests verify:
- ✅ Model loading and class detection
- ✅ Input size (patch size) detection
- ✅ Configuration file parsing
- ✅ Error handling and logging
- ✅ Import functionality

## 📄 License

This toolkit is part of the Intel Manufacturing AI project. Please follow your organization's licensing guidelines.

## 🤝 Contributing

To contribute to this toolkit:
1. **Run tests** before submitting changes: `python tests/run_tests.py`
2. Test your changes with different model types
3. Update documentation for new features
4. Add examples for new functionality
5. Ensure compatibility with existing configurations

---

**Happy Annotating! 🎯** 