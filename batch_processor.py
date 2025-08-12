#!/usr/bin/env python3
"""
Command-line batch processor for brain tumor detection
This script allows processing multiple images from the command line.
"""

import os
import sys
import argparse
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image, ImageDraw, ImageFont
import json
from datetime import datetime

# Load the trained model
model = YOLO(r"D:\fyp\fypcode\Trained_model\YOLOv10CM_FYPtrained.pt")

# Brain tumor classes
class_names = ["Glioma", "Meningioma", "No Tumor", "Pituitary"]

def process_single_image(image_path, output_dir, confidence_threshold=0.5):
    """
    Process a single image and save the result
    """
    try:
        # Read image
        img = cv2.imread(image_path)
        if img is None:
            print(f"❌ Could not read image: {image_path}")
            return None
        
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Run inference
        results = model.predict(source=img, conf=confidence_threshold)
        
        # Process results
        detections = []
        if results and len(results) and results[0].boxes is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            class_ids = results[0].boxes.cls.cpu().numpy()
            confs = results[0].boxes.conf.cpu().numpy()
            
            for box, class_idz, conf in zip(boxes, class_ids, confs):
                label = class_names[int(class_idz)]
                detections.append({
                    'label': label,
                    'confidence': float(conf),
                    'bbox': box.tolist()
                })
        
        # Draw results on image
        output_image = Image.fromarray(img_rgb)
        draw = ImageDraw.Draw(output_image)
        font = ImageFont.load_default()
        
        for detection in detections:
            box = detection['bbox']
            label = detection['label']
            conf = detection['confidence']
            
            x1, y1, x2, y2 = map(int, box)
            draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
            
            text = f"{label} {conf:.2f}"
            bbox = draw.textbbox((x1, max(0, y1 - 20)), text, font=font)
            draw.rectangle(bbox, fill="red")
            draw.text((x1, max(0, y1 - 20)), text, fill="white", font=font)
        
        # Save result
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        output_path = os.path.join(output_dir, f"detected_{base_name}.jpg")
        output_image.save(output_path, 'JPEG', quality=95)
        
        return {
            'image_path': image_path,
            'output_path': output_path,
            'detections': detections,
            'status': 'success'
        }
        
    except Exception as e:
        print(f"❌ Error processing {image_path}: {str(e)}")
        return {
            'image_path': image_path,
            'error': str(e),
            'status': 'error'
        }

def process_batch(input_dir, output_dir, confidence_threshold=0.5, save_json=True):
    """
    Process all images in a directory
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Supported image extensions
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
    
    # Find all image files
    image_files = []
    for file in os.listdir(input_dir):
        if os.path.splitext(file)[1].lower() in image_extensions:
            image_files.append(os.path.join(input_dir, file))
    
    if not image_files:
        print(f"❌ No image files found in {input_dir}")
        return
    
    print(f"📁 Found {len(image_files)} images to process")
    print(f"📂 Output directory: {output_dir}")
    print(f"🎯 Confidence threshold: {confidence_threshold}")
    print("-" * 50)
    
    # Process each image
    results = []
    for i, image_path in enumerate(image_files, 1):
        print(f"🔄 Processing {i}/{len(image_files)}: {os.path.basename(image_path)}")
        result = process_single_image(image_path, output_dir, confidence_threshold)
        results.append(result)
        
        if result and result['status'] == 'success':
            detections = result['detections']
            if detections:
                print(f"   ✅ Found {len(detections)} detection(s): {', '.join([f'{d['label']} ({d['confidence']:.2f})' for d in detections])}")
            else:
                print(f"   ⚠️  No detections found")
        else:
            print(f"   ❌ Failed to process")
    
    # Save results summary
    if save_json:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        json_path = os.path.join(output_dir, f"detection_results_{timestamp}.json")
        
        summary = {
            'timestamp': datetime.now().isoformat(),
            'total_images': len(image_files),
            'successful': len([r for r in results if r and r['status'] == 'success']),
            'failed': len([r for r in results if not r or r['status'] == 'error']),
            'confidence_threshold': confidence_threshold,
            'results': results
        }
        
        with open(json_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"\n📊 Results saved to: {json_path}")
    
    # Print summary
    successful = len([r for r in results if r and r['status'] == 'success'])
    failed = len([r for r in results if not r or r['status'] == 'error'])
    
    print("\n" + "=" * 50)
    print("📈 PROCESSING SUMMARY")
    print("=" * 50)
    print(f"Total images: {len(image_files)}")
    print(f"Successfully processed: {successful}")
    print(f"Failed: {failed}")
    print(f"Output directory: {output_dir}")
    
    if successful > 0:
        total_detections = sum(len(r['detections']) for r in results if r and r['status'] == 'success')
        print(f"Total detections found: {total_detections}")

def main():
    parser = argparse.ArgumentParser(description="Batch process MRI images for brain tumor detection")
    parser.add_argument('input_dir', help='Directory containing input images')
    parser.add_argument('output_dir', help='Directory to save processed images')
    parser.add_argument('--confidence', '-c', type=float, default=0.5, 
                       help='Confidence threshold (default: 0.5)')
    parser.add_argument('--no-json', action='store_true', 
                       help='Do not save JSON results file')
    
    args = parser.parse_args()
    
    # Check if input directory exists
    if not os.path.exists(args.input_dir):
        print(f"❌ Input directory does not exist: {args.input_dir}")
        sys.exit(1)
    
    # Check if model file exists
    model_path = r"D:\fyp\fypcode\Trained_model\YOLOv10CM_FYPtrained.pt"
    if not os.path.exists(model_path):
        print(f"❌ Model file not found: {model_path}")
        sys.exit(1)
    
    print("🧠 Brain Tumor Detection - Batch Processor")
    print("=" * 50)
    
    # Process batch
    process_batch(
        args.input_dir, 
        args.output_dir, 
        args.confidence, 
        not args.no_json
    )

if __name__ == "__main__":
    main()
