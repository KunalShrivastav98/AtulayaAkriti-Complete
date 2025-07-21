@echo off
title AtulayaAkriti Setup and Launch
echo.
echo =====================================================
echo   🎨 AtulayaAkriti Texture Rendering Tool
echo   Professional Interior Design Visualization
echo =====================================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ Python is not installed or not in PATH
    echo Please install Python 3.8 or higher from https://python.org
    pause
    exit /b 1
)

echo ✅ Python found
echo.

REM Create virtual environment if it doesn't exist
if not exist "venv" (
    echo 🔄 Creating virtual environment...
    python -m venv venv
)

REM Activate virtual environment
echo 🔄 Activating virtual environment...
call venv\Scripts\activate.bat

REM Upgrade pip
echo 🔄 Upgrading pip...
python -m pip install --upgrade pip

REM Install requirements
echo 🔄 Installing requirements...
if exist "requirements.txt" (
    pip install -r requirements.txt
) else (
    echo ❌ requirements.txt not found
    pause
    exit /b 1
)

REM Check if SAM model exists
echo 🔄 Checking for SAM models...
if not exist "models\sam_vit_b_01ec64.pth" (
    if not exist "models\sam_vit_l_0b3195.pth" (
        if not exist "models\sam_vit_h_4b8939.pth" (
            echo 📥 No SAM models found. Starting download...
            python download_sam_model.py
        )
    )
)

REM Test setup
echo 🔄 Testing setup...
python test_setup.py

REM Launch the application
echo.
echo 🚀 Launching AtulayaAkriti...
echo.
echo Your browser will open automatically.
echo Press Ctrl+C to stop the application.
echo.
streamlit run atulayakriti_app.py

pause
