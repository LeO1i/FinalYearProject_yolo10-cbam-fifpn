#!/usr/bin/env python3
"""
Development environment startup script
Starts both backend and frontend servers in separate processes
"""

import os
import sys
import subprocess
import platform
import time
from pathlib import Path

def is_windows():
    return platform.system() == "Windows"

def check_node_installed():
    """Check if Node.js is installed"""
    try:
        subprocess.run(
            ["node", "--version"],
            capture_output=True,
            check=True,
            shell=is_windows()
        )
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False

def check_npm_installed():
    """Check if npm is installed"""
    try:
        subprocess.run(
            ["npm", "--version"],
            capture_output=True,
            check=True,
            shell=is_windows()
        )
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False

def main():
    print("🧠 Brain Tumor Detection System - Development Startup")
    print("=" * 60)
    
    # Check Node.js and npm
    if not check_node_installed():
        print("❌ Node.js is not installed!")
        print("Please install Node.js from: https://nodejs.org/")
        sys.exit(1)
    
    if not check_npm_installed():
        print("❌ npm is not installed!")
        print("Please install Node.js (includes npm) from: https://nodejs.org/")
        sys.exit(1)
    
    print("✅ Node.js and npm are installed")
    
    # Check if frontend dependencies are installed
    frontend_path = Path(__file__).parent / "frontend"
    node_modules = frontend_path / "node_modules"
    
    if not node_modules.exists():
        print("\n📦 Installing frontend dependencies...")
        print("This may take a few minutes on first run...")
        try:
            subprocess.run(
                ["npm", "install"],
                cwd=frontend_path,
                check=True,
                shell=is_windows()
            )
            print("✅ Frontend dependencies installed")
        except subprocess.CalledProcessError:
            print("❌ Failed to install frontend dependencies")
            sys.exit(1)
    
    print("\n🚀 Starting development servers...")
    print("=" * 60)
    
    # Start backend
    print("Starting backend on http://localhost:8000")
    if is_windows():
        backend_cmd = f'start cmd /k "python run_ui.py"'
        os.system(backend_cmd)
    else:
        backend_process = subprocess.Popen(
            ["python", "run_ui.py"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
    
    # Give backend time to start
    time.sleep(2)
    
    # Start frontend
    print("Starting frontend on http://localhost:3000")
    if is_windows():
        frontend_cmd = f'start cmd /k "cd frontend && npm run dev"'
        os.system(frontend_cmd)
    else:
        frontend_process = subprocess.Popen(
            ["npm", "run", "dev"],
            cwd=frontend_path,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
    
    print("\n✅ Development servers starting!")
    print("=" * 60)
    print("🌐 Frontend: http://localhost:3000")
    print("🔧 Backend API: http://localhost:8000")
    print("📚 API Docs: http://localhost:8000/docs")
    print("=" * 60)
    print("\n💡 Tip: Check the terminal windows for detailed logs")
    print("⚠️  Press Ctrl+C in each terminal to stop the servers")
    
    if not is_windows():
        print("\nPress Ctrl+C here to stop both servers...")
        try:
            backend_process.wait()
            frontend_process.wait()
        except KeyboardInterrupt:
            print("\n🛑 Stopping servers...")
            backend_process.terminate()
            frontend_process.terminate()
            print("✅ Servers stopped")

if __name__ == "__main__":
    main()

