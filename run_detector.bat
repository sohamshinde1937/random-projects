@echo off
title Live Object Detector
echo Starting Live Object Detection...
echo Press 'q' in the video window to close the application.

:: Run the Python script
python live_detector.py

:: Pause the terminal so you can see any error messages if it crashes
pause