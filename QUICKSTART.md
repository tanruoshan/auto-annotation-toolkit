# Quick Start Guide

## 🚀 Get Started in 3 Steps

### Step 1: Install Dependencies
```bash
cd auto-annotation-toolkit
pip install -r requirements.txt
```

### Step 2: Configure Your Settings
Edit `config/annotation_config.ini`:

```ini
[MODEL]
# Update with your model path
model_path = ../model/gn24_deployment/best.pt

[PATHS]
# Update with your image folder
input_folder = ../images_GN24/CLAHE
output_folder = ./my_annotations
report_folder = ./my_reports

[CLASSES]
# Update with your class names
0 = Good
1 = DieDefect
2 = Scratch

[REPORTS]
# Enable visual reports with bounding boxes
generate_reports = true
show_confidence_in_reports = true
```

### Step 3: Run the Tool
```bash
# Preview what will be processed
python annotate_dataset.py --dry-run

# Process your images
python annotate_dataset.py
```

## 🎯 Common Use Cases

### Case 1: YOLO Model with Custom Classes and Reports
```bash
python annotate_dataset.py \
    --model ../model/best.pt \
    --input ../my_images/ \
    --output ./annotations/ \
    --report-folder ./reports/ \
    --confidence 0.6
```

### Case 2: ONNX Model for Classification
```bash
python annotate_dataset.py \
    --model ../model/classifier.onnx \
    --input ../test_images/ \
    --output ./class_annotations/
```

### Case 3: Multiple Folders
Run the tool multiple times with different input folders, or use the batch example in `examples/annotation_examples.py`.

## 📊 Check Your Results

After processing, you'll find:
- **JSON files**: One per image in your output folder (LabelMe annotations)
- **Report images**: Visual reports with bounding boxes (if enabled)
- **Summary**: Printed to console showing processed count
- **Logs**: Real-time progress during processing

### 🖼️ Visual Reports

Enable visual reports to get annotated images alongside JSON files:

```bash
# Generate reports with annotations
python annotate_dataset.py --report-folder ./reports/

# Disable reports even if configured
python annotate_dataset.py --no-reports
```

Visual reports include:
- 🎯 Bounding boxes with class colors
- 📊 Confidence scores on each detection  
- 📈 Summary count of detections per image
- 🎨 Professional presentation format

## 🔧 Troubleshooting

### "Model not found"
- Check the model path in your config file
- Ensure the model file exists and has the correct extension (.pt or .onnx)

### "No images found"
- Verify the input folder path
- Check that images have supported extensions (.jpg, .png, .bmp)

### "Import errors"
- Run from the toolkit root directory
- Ensure all dependencies are installed

## 📝 Example Output

Your annotations will look like this:
```json
{
  "version": "5.8.1",
  "shapes": [
    {
      "label": "DieDefect",
      "points": [[100, 100], [200, 200]],
      "shape_type": "rectangle"
    }
  ],
  "imagePath": "test_image.jpg",
  "imageHeight": 512,
  "imageWidth": 512
}
```

## 🎉 Next Steps

- **Verify annotations**: Open JSON files in LabelMe to check results
- **Train new models**: Use generated annotations for model training
- **Iterate**: Adjust confidence thresholds and re-run if needed
- **Scale up**: Process larger datasets with the same configuration

Need help? Check the full README.md for detailed documentation!
