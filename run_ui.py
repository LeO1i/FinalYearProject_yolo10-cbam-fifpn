#!/usr/bin/env python3
"""
Brain Tumor Detection System Launcher
This script launches the FastAPI backend server.
"""

import os
import sys
import subprocess
from pathlib import Path

# Add training directory to path for custom modules (CBAM, BiFPN)
# This is needed for loading models trained with custom modules
training_dir = Path(__file__).parent / "training"
if str(training_dir) not in sys.path:
    sys.path.insert(0, str(training_dir))

def check_dependencies():
    """Check if required dependencies are installed"""
    required_packages = [
        'fastapi',
        'uvicorn',
        'ultralytics', 
        'opencv-python',
        'PIL',
        'numpy'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package.replace('-', '_').replace('PIL', 'PIL'))
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"❌ Missing required packages: {', '.join(missing_packages)}")
        print("Please install them using: pip install -r requirements.txt")
        return False
    return True

def check_model_file():
    """Check if the trained model file exists"""
    sys.path.insert(0, str(Path(__file__).parent / "backend"))
    
    try:
        from backend.app.config import get_model_path
        model_path = get_model_path()
        print(f"✅ Model found: {model_path}")
        return True
    except FileNotFoundError as e:
        print(f"❌ {e}")
        return False

def main():
    """Main function to launch the backend server"""
    print("🧠 Brain Tumor Detection System")
    print("=" * 50)
    
    # Check dependencies
    print("Checking dependencies...")
    if not check_dependencies():
        sys.exit(1)
    
    # Check model file
    print("Checking model file...")
    if not check_model_file():
        sys.exit(1)
    
    print("\n✅ All checks passed!")
    print("🚀 Starting the FastAPI backend server...")
    print("=" * 50)
    print("📱 Backend API: http://localhost:8000")
    print("📚 API Documentation: http://localhost:8000/docs")
    print("🌐 Frontend (if running): http://localhost:3000")
    print("\n💡 To start the frontend:")
    print("   cd frontend")
    print("   npm install")
    print("   npm run dev")
    print("=" * 50)
    print("\nPress Ctrl+C to stop the server\n")
    
    try:
        # Run uvicorn
        import uvicorn
        uvicorn.run(
            "backend.app.main:app",
            host="0.0.0.0",
            port=8000,
            reload=True,
            log_level="info"
        )
    except KeyboardInterrupt:
        print("\n🛑 Server stopped by user")
    except Exception as e:
        print(f"❌ Error starting the server: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
