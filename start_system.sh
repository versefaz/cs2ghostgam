#!/bin/bash
# CS2 Betting System - Quick Start Script (Linux/Mac)

echo "==============================================================================="
echo "                    CS2 BETTING SYSTEM V2.0 - PRODUCTION READY                "
echo "==============================================================================="
echo
echo "Starting CS2 Betting System..."
echo
echo "Optional: Start Redis server for caching (not required)"
echo "Redis can be installed with: sudo apt install redis-server (Ubuntu/Debian)"
echo "                           or: brew install redis (macOS)"
echo
echo "System will start in 3 seconds..."
sleep 3
echo

# Clear Python cache for clean start
find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null
find . -name "*.pyc" -delete 2>/dev/null

# Check Python installation
if ! command -v python3 &> /dev/null; then
    echo "ERROR: Python 3 is not installed or not in PATH"
    echo "Please install Python 3.8+ from your package manager"
    exit 1
fi

# Install dependencies if needed
echo "Checking dependencies..."
pip3 install -q aiohttp beautifulsoup4 backoff scikit-learn pandas numpy

# Run the system
echo
echo "🚀 Starting CS2 Betting System..."
echo
python3 main.py

echo
echo "System stopped."
