@echo off
REM =============================================================================
REM setup.bat — WhatsApp Analyzer Setup for Windows
REM =============================================================================
REM Cross-platform setup script for Windows environments
REM Run this file by double-clicking or: setup.bat
REM
REM Requirements: Python 3.9+ installed and added to PATH
REM =============================================================================

setlocal enabledelayedexpansion
cls

echo.
echo ===============================================================================
echo   WhatsApp NLP Analyzer Setup for Windows
echo ===============================================================================
echo.

REM Check Python version
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python not found in PATH
    echo.
    echo Please install Python 3.9 or higher from https://www.python.org
    echo Make sure to check "Add Python to PATH" during installation
    echo.
    pause
    exit /b 1
)

for /f "tokens=2" %%i in ('python --version 2^>^&1') do set PYTHON_VERSION=%%i
echo ✓ Python %PYTHON_VERSION%

REM Create virtual environment
echo.
echo Step 1/5: Creating virtual environment...
if exist venv (
    echo Virtual environment already exists
) else (
    python -m venv venv
    if errorlevel 1 (
        echo ERROR: Failed to create virtual environment
        pause
        exit /b 1
    )
    echo ✓ Virtual environment created
)

REM Activate virtual environment
echo Step 2/5: Activating virtual environment...
call venv\Scripts\activate.bat
if errorlevel 1 (
    echo ERROR: Failed to activate virtual environment
    pause
    exit /b 1
)
echo ✓ Virtual environment activated

REM Upgrade pip
echo Step 3/5: Upgrading pip...
python -m pip install --upgrade pip >nul 2>&1
if errorlevel 1 (
    echo WARNING: pip upgrade failed (non-critical)
) else (
    echo ✓ pip upgraded
)

REM Install dependencies
echo Step 4/5: Installing dependencies...
pip install -r requirements.txt
if errorlevel 1 (
    echo ERROR: Failed to install dependencies
    pause
    exit /b 1
)
echo ✓ Dependencies installed

REM Download NLTK data
echo Step 5/5: Downloading NLP models...
python -m nltk.downloader punkt stopwords wordnet averaged_perceptron_tagger -d "%USERPROFILE%\nltk_data" >nul 2>&1
if errorlevel 1 (
    echo WARNING: NLTK download failed. Try manually:
    echo   python -m nltk.downloader punkt stopwords wordnet
) else (
    echo ✓ NLP models downloaded
)

REM Verify setup
echo.
echo Verifying setup...
python -c "import flask, pandas, sklearn, nltk, textblob; print('✓ All imports successful')" 2>nul
if errorlevel 1 (
    echo WARNING: Some imports failed
) else (
    echo ✓ All packages verified
)

REM Complete
echo.
echo ===============================================================================
echo   ✓ Setup Complete!
echo ===============================================================================
echo.
echo Next steps:
echo.
echo 1. Start the web application:
echo    python app.py
echo.
echo 2. Open your browser:
echo    http://localhost:5000
echo.
echo 3. Upload a WhatsApp chat and explore!
echo.
echo For command-line usage:
echo    python main.py ^<path-to-chat.txt^>
echo.
echo TIP: Keep this command prompt open while the app is running.
echo      Press Ctrl+C to stop the application.
echo.
pause
