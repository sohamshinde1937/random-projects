@echo off
title Bhagyalaxmi Object Detector - Setup

echo ============================================
echo   Bhagyalaxmi Object Detector
echo   Setting up... please wait
echo ============================================
echo.

:: Check Python
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH.
    echo Please install Python from https://www.python.org/downloads/
    echo Make sure to check "Add Python to PATH" during install.
    pause
    exit /b 1
)

echo [1/3] Installing OpenCV...
pip install opencv-python --quiet

echo [2/3] Installing Ultralytics (YOLOv8)...
pip install ultralytics --quiet

echo [3/3] Done! Starting the detector...
echo.
echo NOTE: On first run, a ~6 MB model file will be downloaded.
echo       Your webcam window will open shortly.
echo       Press Q in the camera window to quit.
echo.

python detect.py

pause
