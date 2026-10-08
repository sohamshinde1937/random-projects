@echo off
title JARVIS Launcher
color 0A

echo.
echo  ============================================
echo    JARVIS - Offline Desktop Assistant
echo  ============================================
echo.

:: Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo  [ERROR] Python is not installed or not in PATH.
    echo  Download it from https://www.python.org/downloads/
    echo  Make sure to check "Add Python to PATH" during install.
    pause
    exit /b 1
)

echo  [OK] Python found.

:: Check if jarvis_assistant.py exists in same folder
if not exist "%~dp0jarvis_assistant.py" (
    echo  [ERROR] jarvis_assistant.py not found.
    echo  Make sure launch_jarvis.bat and jarvis_assistant.py are in the same folder.
    pause
    exit /b 1
)

echo  [OK] jarvis_assistant.py found.
echo.
echo  Installing / checking required packages...
echo.

pip install SpeechRecognition pyttsx3 pyaudio --quiet --exists-action i
if errorlevel 1 (
    echo.
    echo  [WARN] Some packages may have failed to install.
    echo  PyAudio sometimes needs a manual install. Try:
    echo    pip install pipwin
    echo    pipwin install pyaudio
    echo.
)

echo.
echo  Launching JARVIS...
echo.

:: Launch the Python app (pythonw hides the console window on Windows)
start "" pythonw "%~dp0jarvis_assistant.py"

:: If pythonw fails, fall back to python (shows console)
if errorlevel 1 (
    python "%~dp0jarvis_assistant.py"
)

exit