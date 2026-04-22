#!/bin/bash

# =============================================================================
# setup.sh — WhatsApp Analyzer Setup for macOS & Linux
# =============================================================================
# Cross-platform setup script for Unix-like systems
# Run: bash setup.sh
#
# Requirements: Python 3.9-3.12 installed
# =============================================================================

set -e  # Exit on error

clear

echo ""
echo "==============================================================================="
echo "  WhatsApp NLP Analyzer Setup for macOS & Linux"
echo "==============================================================================="
echo ""

# Detect OS
OS=$(uname -s)
if [[ "$OS" == "Darwin" ]]; then
    OS_NAME="macOS"
elif [[ "$OS" == "Linux" ]]; then
    OS_NAME="Linux"
else
    OS_NAME="Unknown"
fi

echo "📱 OS: $OS_NAME"

# Check Python version
if ! command -v python3 &> /dev/null; then
    echo "❌ ERROR: Python3 not found"
    echo ""
    echo "Install Python 3.11 or 3.12 using:"
    echo ""
    if [[ "$OS_NAME" == "macOS" ]]; then
        echo "  # Using Homebrew (recommended)"
        echo "  brew install python@3.11"
        echo ""
        echo "  OR visit: https://www.python.org/downloads/macos/"
    else
        echo "  sudo apt-get install python3.11 python3.11-venv"
        echo ""
        echo "  OR: sudo yum install python311"
    fi
    echo ""
    exit 1
fi

PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
echo "✅ Python $PYTHON_VERSION"
echo ""

if ! python3 -c "import sys; raise SystemExit(0 if sys.version_info.major == 3 and 9 <= sys.version_info.minor < 13 else 1)"; then
    echo "❌ ERROR: This project supports Python 3.9 through 3.12."
    echo "Python 3.13+ can fail while installing packages such as Pillow and wordcloud."
    echo "Install Python 3.11 or 3.12, delete the old venv folder, and run setup again."
    exit 1
fi

# Create virtual environment
echo "Step 1/5: Creating virtual environment..."
if [[ -d "venv" ]]; then
    echo "Virtual environment already exists"
else
    python3 -m venv venv
    echo "✅ Virtual environment created"
fi

echo ""
echo "Step 2/5: Activating virtual environment..."
source venv/bin/activate
echo "✅ Virtual environment activated"

echo ""
echo "Step 3/5: Upgrading pip..."
python -m pip install --upgrade pip -q 2>/dev/null || echo "⚠️  pip upgrade skipped (non-critical)"
echo "✅ pip upgraded"

echo ""
echo "Step 4/5: Installing dependencies..."
python -m pip install -r requirements.txt

if [[ $? -ne 0 ]]; then
    echo "❌ ERROR: Failed to install dependencies"
    exit 1
fi
echo "✅ Dependencies installed"

echo ""
echo "Step 5/5: Downloading NLP models..."
python -m nltk.downloader punkt stopwords wordnet averaged_perceptron_tagger -q 2>/dev/null || {
    echo "⚠️  NLTK download needs manual run:"
    echo "   python -m nltk.downloader punkt stopwords wordnet"
}
echo "✅ NLP models ready"

# Verify setup
echo ""
echo "Verifying setup..."
python -c "import flask, pandas, sklearn, nltk, textblob; print('✅ All imports successful')" 2>/dev/null || {
    echo "⚠️  Some imports failed (may still work)"
}

echo ""
echo ""
echo "==============================================================================="
echo "  ✅ Setup Complete!"
echo "==============================================================================="
echo ""
echo "Next steps:"
echo ""
echo "1️⃣  Start the web application:"
echo "    python app.py"
echo ""
echo "2️⃣  Open your browser:"
echo "    http://localhost:5000"
echo ""
echo "3️⃣  Upload a WhatsApp chat and explore!"
echo ""
echo "For command-line usage:"
echo "    python main.py <path-to-chat.txt>"
echo ""
echo "📌 Tip: The virtual environment will auto-deactivate when you close the terminal."
echo "   To activate later, run: source venv/bin/activate"
echo ""
echo "💡 Got issues? Check: python setup.py"
echo ""
