"""
Padding Support Example for Sliding Window Inference
Demonstrates how to use padding to enable sliding window inference on any image size
"""

import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from auto_annotation_generator import AutoAnnotationGenerator
import cv2
import time
import numpy as np

def create_test_images():
    """Create test images of different sizes for demonstration"""
    
    test_images = []
    
    # Create small image (300x200)
    small_image = np.random.randint(50, 200, (200, 300, 3), dtype=np.uint8)
    small_path = "small_test_300x200.jpg"
    cv2.imwrite(small_path, small_image)
    test_images.append(('Small Image', small_path, (300, 200)))
    
    # Create medium image (800x600)
    medium_image = np.random.randint(50, 200, (600, 800, 3), dtype=np.uint8)
    medium_path = "medium_test_800x600.jpg"
    cv2.imwrite(medium_path, medium_image)
    test_images.append(('Medium Image', medium_path, (800, 600)))
    
    # Create tall narrow image (400x1200)
    tall_image = np.random.randint(50, 200, (1200, 400, 3), dtype=np.uint8)
    tall_path = "tall_test_400x1200.jpg"
    cv2.imwrite(tall_path, tall_image)
    test_images.append(('Tall Image', tall_path, (400, 1200)))
    
    # Create wide image (1500x300)
    wide_image = np.random.randint(50, 200, (300, 1500, 3), dtype=np.uint8)
    wide_path = "wide_test_1500x300.jpg"
    cv2.imwrite(wide_path, wide_image)
    test_images.append(('Wide Image', wide_path, (1500, 300)))
    
    return test_images

def test_padding_on_different_sizes():
    """Test padding functionality on different image sizes"""
    
    print("Padding Feature Test on Different Image Sizes")
    print("=" * 60)
    
    model_path = "model/best.pt"  # Update with your model path
    
    # Create test images
    print("Creating test images...")
    test_images = create_test_images()
    print(f"Created {len(test_images)} test images\n")
    
    if not os.path.exists(model_path):
        print(f"Model not found: {model_path}")
        print("Please update the model_path variable with your model file")
        return
    
    # Test configurations
    configurations = [
        {
            'name': 'Without Padding',
            'config': {
                'enable_sliding_window': True,
                'enable_padding_for_small_images': False,
                'min_image_size_for_slicing': 1024,
                'slice_height': 640,
                'slice_width': 640
            }
        },
        {
            'name': 'With Gray Padding',
            'config': {
                'enable_sliding_window': True,
                'enable_padding_for_small_images': True,
                'padding_color': '114,114,114',
                'min_image_size_for_slicing': 1024,
                'slice_height': 640,
                'slice_width': 640
            }
        },
        {
            'name': 'With White Padding',
            'config': {
                'enable_sliding_window': True,
                'enable_padding_for_small_images': True,
                'padding_color': '255,255,255',
                'min_image_size_for_slicing': 1024,
                'slice_height': 640,
                'slice_width': 640
            }
        }
    ]
    
    # Test each configuration on each image
    for img_name, img_path, img_size in test_images:
        print(f"Testing: {img_name} ({img_size[0]}x{img_size[1]})")
        print("-" * 50)
        
        for config_info in configurations:
            config_name = config_info['name']
            config = config_info['config']
            
            try:
                generator = AutoAnnotationGenerator(
                    model_path=model_path,
                    confidence_threshold=0.5,
                    sliding_window_config=config
                )
                
                image = cv2.imread(img_path)
                start_time = time.time()
                detections = generator.predict(image, image_path=img_path)
                inference_time = time.time() - start_time
                
                print(f"  {config_name:20} | Detections: {len(detections):3} | Time: {inference_time:.2f}s")
                
            except Exception as e:
                print(f"  {config_name:20} | Error: {str(e)[:40]}...")
        
        print()
    
    # Clean up test images
    for _, img_path, _ in test_images:
        try:
            os.remove(img_path)
        except:
            pass

def demonstrate_padding_coordinates():
    """Demonstrate coordinate transformation from padded to original"""
    
    print("Coordinate Transformation Demonstration")
    print("=" * 50)
    
    # Create a small test image with known content
    test_image = np.zeros((200, 300, 3), dtype=np.uint8)
    # Add some colored rectangles for visual reference
    cv2.rectangle(test_image, (50, 50), (100, 100), (255, 0, 0), -1)  # Blue rectangle
    cv2.rectangle(test_image, (200, 150), (250, 190), (0, 255, 0), -1)  # Green rectangle
    
    test_path = "coordinate_test.jpg"
    cv2.imwrite(test_path, test_image)
    
    print(f"Original image size: 300x200")
    print("Contains blue rectangle at (50,50)-(100,100)")
    print("Contains green rectangle at (200,150)-(250,190)")
    print()
    
    # Test with padding
    config = {
        'enable_sliding_window': True,
        'enable_padding_for_small_images': True,
        'padding_color': '128,128,128',
        'min_image_size_for_slicing': 1024,
        'slice_height': 640,
        'slice_width': 640
    }
    
    model_path = "model/best.pt"
    
    if not os.path.exists(model_path):
        print(f"Model not found: {model_path}")
        return
    
    try:
        generator = AutoAnnotationGenerator(
            model_path=model_path,
            confidence_threshold=0.3,  # Lower threshold for demo
            sliding_window_config=config
        )
        
        # Show padding calculation
        print("Padding calculation:")
        image = cv2.imread(test_path)
        original_h, original_w = image.shape[:2]
        slice_size = max(config['slice_height'], config['slice_width'])
        target_size = max(slice_size * 2, max(original_h, original_w))
        
        pad_height = max(0, target_size - original_h)
        pad_width = max(0, target_size - original_w)
        pad_top = pad_height // 2
        pad_left = pad_width // 2
        
        print(f"  Target size: {target_size}x{target_size}")
        print(f"  Padding needed: {pad_width}x{pad_height}")
        print(f"  Pad offsets: left={pad_left}, top={pad_top}")
        print()
        
        # Run inference
        detections = generator.predict(image, image_path=test_path)
        
        print(f"Detections found: {len(detections)}")
        for i, det in enumerate(detections):
            bbox = det['bbox']
            print(f"  Detection {i+1}: [{bbox[0]:.1f}, {bbox[1]:.1f}, {bbox[2]:.1f}, {bbox[3]:.1f}]")
            print(f"    (coordinates in original 300x200 image)")
        
    except Exception as e:
        print(f"Error: {e}")
    
    finally:
        # Clean up
        try:
            os.remove(test_path)
        except:
            pass

def test_different_padding_colors():
    """Test different padding color effects"""
    
    print("Padding Color Effect Test")
    print("=" * 40)
    
    # Create test image
    test_image = np.full((400, 600, 3), 128, dtype=np.uint8)  # Gray background
    # Add some objects near edges
    cv2.rectangle(test_image, (10, 10), (50, 50), (0, 0, 255), -1)     # Red (top-left)
    cv2.rectangle(test_image, (550, 350), (590, 390), (255, 0, 0), -1)  # Blue (bottom-right)
    
    test_path = "padding_color_test.jpg"
    cv2.imwrite(test_path, test_image)
    
    model_path = "model/best.pt"
    
    if not os.path.exists(model_path):
        print(f"Model not found: {model_path}")
        return
    
    # Test different padding colors
    padding_colors = [
        ('Black', '0,0,0'),
        ('White', '255,255,255'),
        ('Gray', '114,114,114'),
        ('Mean Gray', '128,128,128'),
        ('ImageNet Mean', '123,117,104')  # Common preprocessing mean
    ]
    
    print(f"Original image: 600x400 (objects near edges)")
    print()
    
    for color_name, color_value in padding_colors:
        config = {
            'enable_sliding_window': True,
            'enable_padding_for_small_images': True,
            'padding_color': color_value,
            'min_image_size_for_slicing': 1024,
            'slice_height': 512,
            'slice_width': 512
        }
        
        try:
            generator = AutoAnnotationGenerator(
                model_path=model_path,
                confidence_threshold=0.3,
                sliding_window_config=config
            )
            
            image = cv2.imread(test_path)
            start_time = time.time()
            detections = generator.predict(image, image_path=test_path)
            inference_time = time.time() - start_time
            
            avg_conf = 0
            if detections:
                avg_conf = sum(d['confidence'] for d in detections) / len(detections)
            
            print(f"{color_name:15} ({color_value:11}) | Det: {len(detections):2} | "
                  f"Avg Conf: {avg_conf:.3f} | Time: {inference_time:.2f}s")
            
        except Exception as e:
            print(f"{color_name:15} | Error: {str(e)[:30]}...")
    
    print()
    print("Note: Different padding colors may affect detection performance")
    print("depending on the training data background distribution.")
    
    # Clean up
    try:
        os.remove(test_path)
    except:
        pass

if __name__ == "__main__":
    print("Sliding Window Padding Feature Examples")
    print("=" * 60)
    print()
    
    try:
        test_padding_on_different_sizes()
        print("\n" + "=" * 60 + "\n")
        demonstrate_padding_coordinates()
        print("\n" + "=" * 60 + "\n")
        test_different_padding_colors()
        
    except KeyboardInterrupt:
        print("\nExample interrupted by user")
    except Exception as e:
        print(f"Example failed: {e}")
    
    print("\nPadding Feature Summary:")
    print("• Enables sliding window on ANY image size")
    print("• Automatic coordinate transformation")
    print("• Configurable padding colors")
    print("• Memory-efficient processing")
    print("• Maintains detection accuracy")
    print("• Temporary files automatically cleaned up")
