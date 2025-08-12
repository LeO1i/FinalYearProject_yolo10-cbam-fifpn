#!/usr/bin/env python3
"""
Simple test script for the Brain Tumor Detection UI
This script tests basic functionality without launching the full web interface.
"""

import os
import sys

def test_imports():
    """Test if required modules can be imported"""
    print("🧪 Testing imports...")
    
    try:
        import cv2
        print("✅ OpenCV imported successfully")
    except ImportError:
        print("❌ OpenCV not available")
        return False
    
    try:
        import numpy as np
        print("✅ NumPy imported successfully")
    except ImportError:
        print("❌ NumPy not available")
        return False
    
    try:
        from PIL import Image, ImageDraw, ImageFont
        print("✅ PIL imported successfully")
    except ImportError:
        print("❌ PIL not available")
        return False
    
    try:
        from ultralytics import YOLO
        print("✅ Ultralytics imported successfully")
    except ImportError:
        print("❌ Ultralytics not available")
        return False
    
    try:
        import gradio as gr
        print("✅ Gradio imported successfully")
    except ImportError:
        print("❌ Gradio not available - Web UI will not work")
        return False
    
    return True

def test_model_loading():
    """Test if the model can be loaded"""
    print("\n🧪 Testing model loading...")
    
    model_path = r"D:\fyp\fypcode\Trained_model\YOLOv10CM_FYPtrained.pt"
    
    if not os.path.exists(model_path):
        print(f"❌ Model file not found: {model_path}")
        return False
    
    try:
        from ultralytics import YOLO
        model = YOLO(model_path)
        print("✅ Model loaded successfully")
        return True
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return False

def test_basic_functionality():
    """Test basic detection functionality"""
    print("\n🧪 Testing basic functionality...")
    
    try:
        from ultralytics import YOLO
        import cv2
        import numpy as np
        from PIL import Image, ImageDraw, ImageFont
        
        # Load model
        model_path = r"D:\fyp\fypcode\Trained_model\YOLOv10CM_FYPtrained.pt"
        model = YOLO(model_path)
        
        # Create a dummy image (black image)
        dummy_image = np.zeros((640, 640, 3), dtype=np.uint8)
        
        # Test prediction
        results = model.predict(source=dummy_image, conf=0.5)
        print("✅ Model prediction test passed")
        
        # Test image processing
        class_names = ["Glioma", "Meningioma", "No Tumor", "Pituitary"]
        print(f"✅ Class names loaded: {class_names}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error in basic functionality test: {e}")
        return False

def main():
    """Main test function"""
    print("🧠 Brain Tumor Detection System - Test Suite")
    print("=" * 50)
    
    # Test imports
    imports_ok = test_imports()
    
    # Test model loading
    model_ok = test_model_loading()
    
    # Test basic functionality
    func_ok = test_basic_functionality()
    
    print("\n" + "=" * 50)
    print("📊 TEST RESULTS")
    print("=" * 50)
    print(f"Imports: {'✅ PASS' if imports_ok else '❌ FAIL'}")
    print(f"Model Loading: {'✅ PASS' if model_ok else '❌ FAIL'}")
    print(f"Basic Functionality: {'✅ PASS' if func_ok else '❌ FAIL'}")
    
    if imports_ok and model_ok and func_ok:
        print("\n🎉 All tests passed! The system should work correctly.")
        print("\nTo launch the web interface:")
        print("1. Install missing dependencies: pip install -r requirements.txt")
        print("2. Run: python FYPUI.py")
        print("3. Or use the launcher: python run_ui.py")
    else:
        print("\n⚠️  Some tests failed. Please check the errors above.")
        
        if not imports_ok:
            print("\nTo install dependencies:")
            print("pip install -r requirements.txt")
        
        if not model_ok:
            print("\nTo fix model issues:")
            print("1. Ensure the model file exists in Trained_model/")
            print("2. Check file permissions")
        
        if not func_ok:
            print("\nTo fix functionality issues:")
            print("1. Check CUDA/GPU compatibility")
            print("2. Verify PyTorch installation")

if __name__ == "__main__":
    main()
