# 🚀 Installation Guide — WhatsApp NLP Analyzer

Complete cross-platform installation instructions for Windows, macOS, and Linux.

---

## 📋 Prerequisites

- **Python 3.9 or higher** — [Download](https://www.python.org/downloads/)
- **Git** (optional, for cloning) — [Download](https://git-scm.com/)
- **Terminal/Command Prompt** knowledge (basic)

### Check Your Python Version

**Windows (Command Prompt):**
```cmd
python --version
```

**macOS / Linux (Terminal):**
```bash
python3 --version
```

Expected output: `Python 3.9.x` or higher

If not installed, download from [python.org](https://www.python.org/downloads)

---

## ⚡ Quick Install (Recommended)

Pick your operating system:

### 🪟 Windows

1. **Download the project:**
   - Click "Code" → "Download ZIP"
   - Extract to your desired folder
   - OR use Git: `git clone <repo-url>`

2. **Run the setup:**
   - Open the folder in File Explorer
   - Double-click `setup.bat`
   - Wait for completion (5-10 minutes)

3. **Start the app:**
   - Open Command Prompt in the project folder
   - Type: `python app.py`
   - Open browser: http://localhost:5000

### 🍎 macOS

1. **Download the project:**
   ```bash
   git clone <repo-url>
   cd whatsapp-analyzer
   ```

2. **Run the setup:**
   ```bash
   bash setup.sh
   ```
   Leave the terminal open during setup (5-10 minutes)

3. **Start the app:**
   ```bash
   python app.py
   ```
   Open browser: http://localhost:5000

### 🐧 Linux

1. **Install system dependencies (Debian/Ubuntu):**
   ```bash
   sudo apt-get update
   sudo apt-get install python3.11 python3.11-venv python3-pip
   ```

2. **Download and navigate:**
   ```bash
   git clone <repo-url>
   cd whatsapp-analyzer
   ```

3. **Run the setup:**
   ```bash
   bash setup.sh
   ```

4. **Start the app:**
   ```bash
   python app.py
   ```

---

## 📖 Manual Installation (Step-by-Step)

Use this if the automatic scripts don't work.

### Step 1: Create Virtual Environment

**Windows:**
```cmd
python -m venv venv
venv\Scripts\activate
```

**macOS / Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

✅ You should see `(venv)` prefix in terminal when active.

### Step 2: Upgrade pip

**All platforms:**
```bash
python -m pip install --upgrade pip
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

This installs:
- Flask (web framework)
- pandas & numpy (data processing)
- scikit-learn (machine learning)
- nltk, textblob, vaderSentiment (NLP)
- matplotlib, seaborn, plotly (visualization)
- And more...

### Step 4: Download NLP Models

```bash
python -m nltk.downloader punkt stopwords wordnet averaging_perceptron_tagger
```

Or if that fails:
```bash
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet')"
```

### Step 5: Create Project Directories

```bash
mkdir models results results/visualizations logs uploads
```

### Step 6: Verify Setup

```bash
python setup.py
```

This checks all imports and verifies installation.

---

## 🚀 Running the Application

### Option 1: Web Interface (Recommended)

```bash
python app.py
```

Then open: **http://localhost:5000** in your browser

### Option 2: Command Line

```bash
python main.py path/to/chat.txt
```

---

## ❌ Troubleshooting

### ❌ "python: command not found"

**macOS/Linux Solution:**
```bash
# Use python3 instead
python3 --version
python3 app.py
```

**Windows Solution:**
- Reinstall Python with "Add Python to PATH" checked
- Restart Command Prompt
- Try `py --version`

---

### ❌ "ModuleNotFoundError: No module named 'flask'"

**Solution:**
1. Check if virtual environment is **active** (look for `(venv)` prefix)
2. Reactivate if needed:
   - Windows: `venv\Scripts\activate`
   - macOS/Linux: `source venv/bin/activate`
3. Reinstall: `pip install -r requirements.txt`

---

### ❌ "pip: command not found"

**Solution:**
- Virtual environment not active
- Always activate first (Step 1 of manual installation)
- Then: `python -m pip install -r requirements.txt`

---

### ❌ "NLTK data not found" during runtime

**Solution:**
```bash
python -c "import nltk; nltk.download('punkt', 'stopwords', 'wordnet')"
```

---

### ❌ "Port 5000 already in use"

**Solution 1: Use different port**
```bash
$ In app.py, change: app.run(port=5001)
```

**Solution 2: Kill the existing process**

**Windows:**
```cmd
netstat -ano | findstr :5000
taskkill /PID <PID> /F
```

**macOS/Linux:**
```bash
lsof -ti:5000 | xargs kill -9
```

---

### ❌ "Permission denied" on macOS/Linux

**Solution:**
```bash
chmod +x setup.sh
bash setup.sh
```

---

### ❌ Slow on first run (5-10 seconds)

**This is normal** — models loading and NLP initialization take time.
Subsequent runs are fast.

---

### ❌ "Out of memory" on large chats

- Your chat is likely > 50,000 messages
- Try on a machine with more RAM
- Or limit the chat size to 20,000 messages

---

## 🔄 Updating Dependencies

To update to latest versions:

```bash
pip install --upgrade -r requirements.txt
```

---

## 🧹 Clean Installation (Start Fresh)

If something broke, start over:

```bash
# Deactivate if active
deactivate

# Remove virtual environment
# Windows: rmdir /s venv
# macOS/Linux: rm -rf venv

# Restart setup
python setup.py  # OR bash setup.sh  # OR setup.bat
```

---

## 🔧 Advanced: Development Mode

For developers who want to contribute:

```bash
pip install -r requirements.txt
pip install pytest black pylint  # Dev tools

# Run tests
pytest tests/

# Format code
black src/

# Check code style
pylint src/
```

---

## 📙 What Gets Installed?

| Package | Purpose |
|---------|---------|
| Flask | Web framework |
| pandas | Data processing |
| numpy | Numerical computing |
| scikit-learn | Machine learning |
| nltk | Natural language processing |
| vaderSentiment | Sentiment baseline |
| textblob | Text processing |
| matplotlib, seaborn | Visualization |
| plotly | Interactive charts |
| wordcloud | Word cloud generation |
| Pillow | Image manipulation |

---

## ✅ Verify Installation

Run this to check everything is installed:

```bash
python -c "
import flask, pandas, numpy as np, sklearn
import nltk, textblob, vaderSentiment
import matplotlib, seaborn, plotly
print('✅ All dependencies installed successfully!')
"
```

---

## 🆘 Still Having Issues?

1. **Check the logs:**
   ```bash
   cat logs/app.log  # macOS/Linux
   type logs\app.log  # Windows
   ```

2. **Run diagnostic:**
   ```bash
   python setup.py
   ```

3. **Check Python path:**
   ```bash
   which python      # macOS/Linux
   where python      # Windows
   ```

4. **Check virtual environment:**
   ```bash
   pip list | grep Flask
   ```

---

## 📚 Additional Resources

- **Python Virtual Environments:** https://docs.python.org/3/tutorial/venv.html
- **Flask Documentation:** https://flask.palletsprojects.com/
- **scikit-learn:** https://scikit-learn.org/
- **NLTK:** https://www.nltk.org/

---

**Happy analyzing! 🎉**

Still stuck? Open an issue on GitHub with:
- Your OS and Python version (`python --version`)
- The error message (copy full traceback)
- What you've already tried
