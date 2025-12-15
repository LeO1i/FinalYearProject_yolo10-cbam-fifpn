#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test script for the Brain Tumor Detection Backend
This script tests basic functionality without launching the full web application.
"""

import os
import sys
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent / "backend"))

# Add training directory to path for custom modules (CBAM, BiFPN)
# This is needed for loading models trained with custom modules
training_dir = Path(__file__).parent / "training"
if str(training_dir) not in sys.path:
    sys.path.insert(0, str(training_dir))

def test_imports():
    """Test if required modules can be imported"""
    print("[TEST] Testing imports...")
    
    try:
        import cv2
        print("[OK] OpenCV imported successfully")
    except ImportError:
        print("[FAIL] OpenCV not available")
        return False
    
    try:
        import numpy as np
        print("[OK] NumPy imported successfully")
    except ImportError:
        print("[FAIL] NumPy not available")
        return False
    
    try:
        from PIL import Image, ImageDraw, ImageFont
        print("[OK] PIL imported successfully")
    except ImportError:
        print("[FAIL] PIL not available")
        return False
    
    try:
        from ultralytics import YOLO
        print("[OK] Ultralytics imported successfully")
    except ImportError:
        print("[FAIL] Ultralytics not available")
        return False
    
    try:
        import fastapi
        print("[OK] FastAPI imported successfully")
    except ImportError:
        print("[FAIL] FastAPI not available - Backend will not work")
        return False
    
    try:
        import uvicorn
        print("[OK] Uvicorn imported successfully")
    except ImportError:
        print("[FAIL] Uvicorn not available - Server will not start")
        return False
    
    try:
        from backend.app import config, inference, schemas
        print("[OK] Backend modules imported successfully")
    except ImportError as e:
        print(f"[FAIL] Backend modules not available: {e}")
        return False
    
    return True

def test_model_loading():
    """Test if the model can be loaded"""
    print("\n[TEST] Testing model loading...")
    
    try:
        from backend.app.config import get_model_path
        from backend.app.inference import load_model
        
        model_path = get_model_path()
        print(f"Model path: {model_path}")
        
        if not model_path.exists():
            print(f"[FAIL] Model file not found: {model_path}")
            return False
        
        model = load_model()
        print("[OK] Model loaded successfully")
        return True
    except Exception as e:
        print(f"[FAIL] Error loading model: {e}")
        return False

def test_basic_functionality():
    """Test basic detection functionality"""
    print("\n[TEST] Testing basic functionality...")
    
    try:
        import cv2
        import numpy as np
        from backend.app.inference import load_model, predict_image, draw_detections
        from backend.app.config import CLASS_NAMES
        
        # Load model
        model = load_model()
        
        # Create a dummy image (black image)
        dummy_image = np.zeros((640, 640, 3), dtype=np.uint8)
        
        # Test prediction
        detections = predict_image(dummy_image, confidence=0.5, model=model)
        print(f"[OK] Model prediction test passed (found {len(detections)} detections)")
        
        # Test drawing
        annotated = draw_detections(dummy_image, detections)
        print("[OK] Image annotation test passed")
        
        # Test class names
        print(f"[OK] Class names loaded: {CLASS_NAMES}")
        
        return True
        
    except Exception as e:
        print(f"[FAIL] Error in basic functionality test: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main test function"""
    print("Brain Tumor Detection System - Test Suite")
    print("=" * 50)
    
    # Test imports
    imports_ok = test_imports()
    
    # Test model loading
    model_ok = test_model_loading()
    
    # Test basic functionality
    func_ok = test_basic_functionality()
    
    print("\n" + "=" * 50)
    print("TEST RESULTS")
    print("=" * 50)
    print(f"Imports: {'PASS' if imports_ok else 'FAIL'}")
    print(f"Model Loading: {'PASS' if model_ok else 'FAIL'}")
    print(f"Basic Functionality: {'PASS' if func_ok else 'FAIL'}")
    
    if imports_ok and model_ok and func_ok:
        print("\n[SUCCESS] All tests passed! The system should work correctly.")
        print("\nTo launch the web application:")
        print("\nOption 1 - Quick start (both servers):")
        print("  python start_dev.py")
        print("\nOption 2 - Manual start (two terminals):")
        print("  Terminal 1: python run_ui.py")
        print("  Terminal 2: cd frontend && npm run dev")
        print("\nThen open: http://localhost:3000")
    else:
        print("\n[WARNING] Some tests failed. Please check the errors above.")
        
        if not imports_ok:
            print("\nTo install Python dependencies:")
            print("  pip install -r requirements.txt")
        
        if not model_ok:
            print("\nTo fix model issues:")
            print("  1. Ensure the model file exists in Trained_model/")
            print("  2. Check file permissions")
        
        if not func_ok:
            print("\nTo fix functionality issues:")
            print("  1. Check CUDA/GPU compatibility")
            print("  2. Verify PyTorch installation")
            print("  3. Ensure backend modules are correctly structured")

if __name__ == "__main__":
    main()
