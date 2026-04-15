#!/usr/bin/env python3
# =============================================================================
# setup.py — WhatsApp Analyzer Setup Script
# =============================================================================
# Cross-platform setup for all dependencies and NLP models
# Works on: Windows, macOS, Linux
#
# Usage:
#   python setup.py
#
# =============================================================================

import os
import sys
import subprocess
import platform
from pathlib import Path


def print_header(text):
    """Print a formatted header."""
    print("\n" + "=" * 70)
    print(f"  {text}")
    print("=" * 70 + "\n")


def print_step(step_num, total, text):
    """Print a numbered step."""
    print(f"\n📌 STEP {step_num}/{total}: {text}")
    print("-" * 70)


def print_success(text):
    """Print success message."""
    print(f"✅ {text}")


def print_error(text):
    """Print error message."""
    print(f"❌ ERROR: {text}")


def print_warning(text):
    """Print warning message."""
    print(f"⚠️  WARNING: {text}")


def check_python_version():
    """Verify Python >= 3.9."""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 9):
        print_error(f"Python 3.9+ required. You have {version.major}.{version.minor}")
        sys.exit(1)
    print_success(f"Python {version.major}.{version.minor}.{version.micro}")
    return True


def get_os_name():
    """Get operating system name."""
    system = platform.system()
    return "Windows" if system == "Windows" else "macOS" if system == "Darwin" else "Linux"


def upgrade_pip():
    """Upgrade pip to latest version."""
    print_step(1, 5, "Upgrading pip")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "pip"])
        print_success("pip upgraded")
        return True
    except subprocess.CalledProcessError:
        print_warning("pip upgrade failed (non-critical, continuing)")
        return False


def install_requirements():
    """Install dependencies from requirements.txt."""
    print_step(2, 5, "Installing dependencies from requirements.txt")
    
    req_file = Path(__file__).parent / "requirements.txt"
    if not req_file.exists():
        print_error(f"requirements.txt not found at {req_file}")
        sys.exit(1)
    
    try:
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", 
            "-r", str(req_file)
        ])
        print_success("All dependencies installed")
        return True
    except subprocess.CalledProcessError as e:
        print_error(f"Dependency installation failed: {e}")
        sys.exit(1)


def download_nltk_data():
    """Download required NLTK datasets."""
    print_step(3, 5, "Downloading NLTK language models")
    
    import nltk
    from nltk.downloader import Downloader
    
    datasets = ['punkt', 'stopwords', 'wordnet', 'averaged_perceptron_tagger']
    failed = []
    
    for dataset in datasets:
        try:
            nltk.download(dataset, quiet=True)
            print_success(f"Downloaded NLTK dataset: {dataset}")
        except Exception as e:
            print_warning(f"Failed to download {dataset}: {e}")
            failed.append(dataset)
    
    if failed:
        print_warning(f"Some datasets failed: {failed}. Try manually:")
        for ds in failed:
            print(f"  python -m nltk.downloader {ds}")
    else:
        print_success("All NLTK datasets downloaded")
    
    return len(failed) == 0


def verify_imports():
    """Verify critical packages can be imported."""
    print_step(4, 5, "Verifying package imports")
    
    critical_packages = {
        'flask': 'Flask',
        'pandas': 'pandas',
        'sklearn': 'scikit-learn',
        'nltk': 'NLTK',
        'textblob': 'TextBlob',
        'vaderSentiment': 'VADER Sentiment',
    }
    
    failed = []
    for import_name, display_name in critical_packages.items():
        try:
            __import__(import_name)
            print_success(f"✓ {display_name}")
        except ImportError:
            print_error(f"✗ {display_name} not found")
            failed.append(display_name)
    
    if failed:
        print_error(f"Missing packages: {', '.join(failed)}")
        print("Try: pip install -r requirements.txt")
        return False
    
    print_success("All critical packages verified")
    return True


def create_directories():
    """Create necessary project directories."""
    print_step(5, 5, "Creating project directories")
    
    dirs = [
        'models',
        'results/visualizations',
        'logs',
        'uploads',
    ]
    
    for dir_path in dirs:
        full_path = Path(__file__).parent / dir_path
        full_path.mkdir(parents=True, exist_ok=True)
        print_success(f"✓ {dir_path}")
    
    print_success("All directories ready")


def main():
    """Main setup orchestration."""
    os.system('clear' if os.name != 'nt' else 'cls')
    
    print_header("🚀 WhatsApp NLP Analyzer — Setup")
    print(f"📱 OS: {get_os_name()}")
    print(f"🐍 Python: {sys.version.split()[0]}")
    print()
    
    # Step 1: Check Python
    print_step(0, 5, "Checking Python version")
    if not check_python_version():
        sys.exit(1)
    
    # Step 2-5: Setup
    try:
        if not upgrade_pip():
            pass  # Non-critical
        
        if not install_requirements():
            sys.exit(1)
        
        if not download_nltk_data():
            pass  # Non-critical, allow continuation
        
        if not verify_imports():
            print_warning("Some imports failed, but setup may still work")
        
        create_directories()
        
    except KeyboardInterrupt:
        print_error("\nSetup cancelled by user")
        sys.exit(1)
    except Exception as e:
        print_error(f"Unexpected error: {e}")
        sys.exit(1)
    
    # Success
    print_header("✅ Setup Complete!")
    print("Next steps:")
    print()
    print("1. Start the web application:")
    print("   python app.py")
    print()
    print("2. Open your browser:")
    print("   http://localhost:5000")
    print()
    print("3. Export a WhatsApp chat and upload it!")
    print()
    print("For command-line usage:")
    print("   python main.py <path-to-chat.txt>")
    print()
    print("=" * 70)


if __name__ == "__main__":
    main()
