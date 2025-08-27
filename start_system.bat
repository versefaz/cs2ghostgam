@echo off
REM CS2 Betting System - Quick Start Script
echo ===============================================================================
echo                    CS2 BETTING SYSTEM V2.0 - PRODUCTION READY                
echo ===============================================================================
echo.
echo Starting CS2 Betting System...
echo.
echo Optional: Start Redis server for caching (not required)
echo Redis can be downloaded from: https://redis.io/download
echo.
echo System will start in 3 seconds...
timeout /t 3 /nobreak > nul
echo.

REM Clear Python cache for clean start
if exist __pycache__ rmdir /s /q __pycache__
if exist "*.pyc" del /q *.pyc

REM Check Python installation
python --version > nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.8+ from https://python.org
    pause
    exit /b 1
)

REM Install dependencies if needed
echo Checking dependencies...
pip install -q aiohttp beautifulsoup4 backoff scikit-learn pandas numpy

REM Run the system
echo.
echo 🚀 Starting CS2 Betting System...
echo.
python main.py

echo.
echo System stopped. Press any key to exit...
pause > nul
