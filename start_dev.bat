@echo off
REM Brain Tumor Detection System - Development Startup Script for Windows
REM This script starts both backend and frontend servers

title Brain Tumor Detection System

echo ========================================
echo Brain Tumor Detection System
echo Development Environment Startup
echo ========================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python is not installed!
    echo Please install Python from: https://www.python.org/downloads/
    pause
    exit /b 1
)
echo [OK] Python is installed

REM Check if Node.js is installed
node --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Node.js is not installed!
    echo Please install Node.js from: https://nodejs.org/
    pause
    exit /b 1
)
echo [OK] Node.js is installed

REM Check if frontend dependencies are installed
if not exist "frontend\node_modules\" (
    echo.
    echo [INFO] Installing frontend dependencies...
    echo This may take a few minutes on first run...
    cd frontend
    call npm install
    if errorlevel 1 (
        echo [ERROR] Failed to install frontend dependencies
        pause
        exit /b 1
    )
    cd ..
    echo [OK] Frontend dependencies installed
)

echo.
echo ========================================
echo Starting Development Servers
echo ========================================
echo.

REM Start backend in new window
echo Starting Backend Server on http://localhost:8000
start "Backend - Brain Tumor Detection" cmd /k "python run_ui.py"

REM Wait a bit for backend to start
timeout /t 3 /nobreak >nul

REM Start frontend in new window
echo Starting Frontend Server on http://localhost:3000
start "Frontend - Brain Tumor Detection" cmd /k "cd frontend && npm run dev"

echo.
echo ========================================
echo Development Servers Started!
echo ========================================
echo.
echo Frontend:  http://localhost:3000
echo Backend:   http://localhost:8000
echo API Docs:  http://localhost:8000/docs
echo.
echo Check the new terminal windows for detailed logs
echo Press Ctrl+C in each window to stop the servers
echo.
echo Opening frontend in browser...
timeout /t 5 /nobreak >nul
start http://localhost:3000

echo.
echo Press any key to close this window...
pause >nul

