#!/usr/bin/env python3
"""
Brain Tumor Detection UI Launcher
This script launches the Gradio web interface for brain tumor detection.
"""

import os
import sys
import subprocess

def check_dependencies():
    """Check if required dependencies are installed"""
    required_packages = [
        'gradio',
        'ultralytics', 
        'opencv-python',
        'PIL',
        'numpy'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"Missing required packages: {', '.join(missing_packages)}")
        print("Please install them using: pip install -r requirements.txt")
        return False
    return True

def check_model_file():
    """Check if the trained model file exists"""
    model_path = r"D:\fyp\fypcode\Trained_model\YOLOv10CM_FYPtrained.pt"
    if not os.path.exists(model_path):
        print(f"Model file not found: {model_path}")
        print("Please ensure the trained model file exists in the specified location.")
        return False
    return True

def main():
    """Main function to launch the UI"""
    print("🧠 Brain Tumor Detection System")
    print("=" * 40)
    
    # Check dependencies
    if not check_dependencies():
        sys.exit(1)
    
    # Check model file
    if not check_model_file():
        sys.exit(1)
    
    print("✅ All checks passed!")
    print("🚀 Starting the web interface...")
    print("📱 The interface will be available at: http://localhost:7860")
    print("🌐 Public link will be provided once the server starts")
    print("\nPress Ctrl+C to stop the server")
    
    try:
        # Import and run the UI
        from FYPUI import demo
        demo.launch(
            share=True,
            server_name="0.0.0.0", 
            server_port=7860,
            show_error=True
        )
    except KeyboardInterrupt:
        print("\n🛑 Server stopped by user")
    except Exception as e:
        print(f"❌ Error starting the server: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
