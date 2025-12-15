"""
Shared inference module for brain tumor detection.
This module is used by both the FastAPI backend and CLI batch processor.
"""
import sys
from pathlib import Path

# Add training directory to path for custom modules (CBAM, BiFPN)
# This is needed for loading models trained with custom modules
_training_dir = Path(__file__).parent.parent.parent / "training"
if str(_training_dir) not in sys.path:
    sys.path.insert(0, str(_training_dir))

import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO

from .config import get_model_path, CLASS_NAMES, DEFAULT_CONFIDENCE

# Global model instance (loaded once)
_model: Optional[YOLO] = None


def load_model(model_path: Optional[Path] = None) -> YOLO:
    """
    Load the YOLO model. Uses a singleton pattern to load only once.
    
    Args:
        model_path: Path to the model file. If None, uses default from config.
    
    Returns:
        Loaded YOLO model instance
    """
    global _model
    
    if _model is None:
        if model_path is None:
            model_path = get_model_path()
        
        print(f"Loading model from: {model_path}")
        _model = YOLO(str(model_path))
        print("Model loaded successfully!")
    
    return _model


def predict_image(
    image: np.ndarray,
    confidence: float = DEFAULT_CONFIDENCE,
    model: Optional[YOLO] = None
) -> List[Dict]:
    """
    Run inference on an image and return structured detection results.
    
    Args:
        image: Input image as numpy array (BGR format from cv2 or RGB)
        confidence: Confidence threshold for detections
        model: YOLO model instance. If None, uses the global model.
    
    Returns:
        List of detections, each containing:
            - label: class name
            - confidence: detection confidence score
            - bbox: [x1, y1, x2, y2] bounding box coordinates
    """
    if model is None:
        model = load_model()
    
    # Run inference
    results = model.predict(source=image, conf=confidence, verbose=False)
    
    # Parse results
    detections = []
    if results and len(results) > 0:
        result = results[0]
        
        if result.boxes is not None and len(result.boxes) > 0:
            boxes = result.boxes.xyxy.cpu().numpy()  # [x1, y1, x2, y2]
            class_ids = result.boxes.cls.cpu().numpy()
            confidences = result.boxes.conf.cpu().numpy()
            
            for box, class_id, conf in zip(boxes, class_ids, confidences):
                label = CLASS_NAMES[int(class_id)]
                detections.append({
                    'label': label,
                    'confidence': float(conf),
                    'bbox': box.tolist()
                })
    
    return detections


def draw_detections(
    image_rgb: np.ndarray,
    detections: List[Dict],
    box_color: str = "red",
    text_color: str = "white",
    box_width: int = 3
) -> Image.Image:
    """
    Draw bounding boxes and labels on an image.
    
    Args:
        image_rgb: Input image as numpy array (RGB format)
        detections: List of detection dictionaries from predict_image()
        box_color: Color for bounding boxes
        text_color: Color for text labels
        box_width: Width of bounding box lines
    
    Returns:
        PIL Image with drawn detections
    """
    # Convert to PIL Image
    output_image = Image.fromarray(image_rgb)
    draw = ImageDraw.Draw(output_image)
    
    try:
        # Try to load a better font if available
        font = ImageFont.truetype("arial.ttf", 16)
    except:
        font = ImageFont.load_default()
    
    # Draw each detection
    for detection in detections:
        bbox = detection['bbox']
        label = detection['label']
        conf = detection['confidence']
        
        x1, y1, x2, y2 = map(int, bbox)
        
        # Draw bounding box
        draw.rectangle([x1, y1, x2, y2], outline=box_color, width=box_width)
        
        # Draw label with background
        text = f"{label} {conf:.2f}"
        
        # Get text bounding box
        text_bbox = draw.textbbox((x1, max(0, y1 - 20)), text, font=font)
        
        # Draw text background
        draw.rectangle(text_bbox, fill=box_color)
        
        # Draw text
        draw.text((x1, max(0, y1 - 20)), text, fill=text_color, font=font)
    
    return output_image


def process_image_full(
    image_bgr: np.ndarray,
    confidence: float = DEFAULT_CONFIDENCE,
    model: Optional[YOLO] = None
) -> Tuple[Image.Image, List[Dict]]:
    """
    Complete processing pipeline: inference + annotation.
    
    Args:
        image_bgr: Input image in BGR format (OpenCV default)
        confidence: Confidence threshold
        model: YOLO model instance
    
    Returns:
        Tuple of (annotated PIL Image, list of detections)
    """
    # Convert BGR to RGB
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    
    # Run inference
    detections = predict_image(image_bgr, confidence, model)
    
    # Draw detections
    annotated_image = draw_detections(image_rgb, detections)
    
    return annotated_image, detections


def process_image_from_file(
    image_path: str,
    confidence: float = DEFAULT_CONFIDENCE,
    model: Optional[YOLO] = None
) -> Tuple[Image.Image, List[Dict]]:
    """
    Process an image file: load, infer, annotate.
    
    Args:
        image_path: Path to image file
        confidence: Confidence threshold
        model: YOLO model instance
    
    Returns:
        Tuple of (annotated PIL Image, list of detections)
    """
    # Read image
    image_bgr = cv2.imread(str(image_path))
    if image_bgr is None:
        raise ValueError(f"Could not read image: {image_path}")
    
    return process_image_full(image_bgr, confidence, model)

