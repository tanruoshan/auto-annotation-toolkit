"""
Auto Annotation Toolkit
Generates LabelMe format annotations using trained models

Main Components:
- AutoAnnotationGenerator: Core annotation generation class
- Command line interface: annotate_dataset.py
- Quick scripts: src/quick_auto_annotate.py
"""

__version__ = "1.0.0"
__author__ = "Intel Manufacturing AI Team"
__description__ = "Auto annotation toolkit for generating LabelMe JSON annotations"

# Make the main class easily importable
try:
    from src.auto_annotation_generator import AutoAnnotationGenerator
    __all__ = ["AutoAnnotationGenerator"]
except ImportError:
    # Fallback if src module not found
    __all__ = []
