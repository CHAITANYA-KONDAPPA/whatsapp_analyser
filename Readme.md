# WhatsApp Chat Analysis System

NLP-based sentiment analysis and communication pattern extraction from WhatsApp chats.

## 🎯 Features

- **Parse WhatsApp Exports**: Automatically parse exported .txt files from both Android & iPhone
- **Data Cleaning**: Remove sensitive info, normalize text, handle emojis & URLs
- **NLP Processing**: Tokenization, stemming, TF-IDF vectorization, n-grams
- **Sentiment Analysis**: Dual-model ensemble (Naive Bayes + Logistic Regression with VADER baseline)
- **Visualizations**: Interactive Plotly charts, word clouds, heatmaps, trends
- **Web Dashboard**: Beautiful Flask UI with real-time predictions
- **ML Training**: Train custom models on filtered high-confidence labels

---

## ⚙️ System Requirements

- **Python**: 3.9 or higher
- **OS**: Windows, macOS, Linux
- **RAM**: 2+ GB recommended
- **Disk**: 500 MB for dependencies + data

**Check your Python version:**
```bash
python --version  # Should show 3.9.x or higher
```

If not, download from [python.org](https://www.python.org/downloads)

---

## 🚀 Quick Start (All Platforms)

### Step 1: Clone & Navigate
```bash
git clone <repo-url>
cd whatsapp-analyzer
```

### Step 2: Create Virtual Environment

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

> ✅ You'll see `(venv)` prefix in your terminal when activated

### Step 3: Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### Step 4: Download NLP Models (First Time Only)
```bash
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet')"
```

### Step 5: Run the Application

**Web Interface (Recommended):**
```bash
python app.py
# Opens at http://localhost:5000
```

**Command Line:**
```bash
python main.py <path-to-chat-export.txt>
```

---

## 📁 Project Structure

```
whatsapp-analyzer/
├── app.py                 # Flask web application
├── main.py               # CLI entry point
├── config.py             # Configuration settings
├── requirements.txt      # Python dependencies
├── README.md            # This file
│
├── src/
│   ├── whatsapp_parser.py        # Parse .txt exports
│   ├── data_processor.py         # Clean & normalize data
│   ├── nlp_processor.py          # NLP feature extraction
│   ├── sentiment_analyzer.py     # VADER sentiment baseline
│   ├── sentiment_classifier.py   # ML model training & prediction
│   ├── pipeline.py               # Orchestrate full pipeline
│   └── visualizer.py             # Generate charts
│
├── templates/
│   ├── base.html         # Base layout
│   ├── index.html        # Upload page
│   ├── dashboard.html    # Main results dashboard
│   ├── results.html      # Detailed breakdowns
│   └── base.html         # Shared components
│
├── static/
│   ├── css/style.css     # Styling
│   ├── js/script.js      # UI interactions
│   └── images/           # Assets
│
├── models/               # Trained ML models (auto-created)
├── results/              # Generated charts & exports
└── logs/                 # Application logs
```

---

## 🔧 Troubleshooting

### ❌ "python: command not found"
- **macOS/Linux**: Use `python3` instead of `python`
- **Windows**: Add Python to PATH or use full path
- **Solution**: `python3 --version` should show 3.9+

### ❌ "ModuleNotFoundError: No module named 'flask'"
- Virtual environment not activated
- **Solution**: Run activation command again:
  - Windows: `venv\Scripts\activate`
  - macOS/Linux: `source venv/bin/activate`

### ❌ "pip: command not found"
- Virtual environment not active
- **Solution**: Always activate first (see Step 2)

### ❌ "NLTK data not found" error
- Download didn't complete
- **Solution**: Run manually:
  ```bash
  python -m nltk.downloader punkt stopwords wordnet
  ```

### ❌ "Port 5000 already in use"
- Another app using port 5000
- **Solution**: Use different port:
  ```bash
  python app.py --port 5001
  ```

### ❌ Slow on first run
- Models loading & NLP initialization
- **Expected**: First run takes 5-10s; subsequent runs are fast

### ❌ "Out of memory" on large chats
- Chat export > 50k messages
- **Solution**: Increase available RAM or enable virtual memory

---

## 🐍 Python & Virtual Environment Guide

### Why Virtual Environments?
- Isolate project dependencies
- Prevent conflicts with system Python
- Easy setup on different machines

### Check Virtual Environment Status
```bash
# Should show path to venv Python
which python          # macOS/Linux
where python          # Windows
```

### Deactivate Virtual Environment
```bash
deactivate
```

---

## 📊 Usage Examples

### Web Interface
1. Run `python app.py`
2. Open http://localhost:5000
3. Upload WhatsApp .txt export
4. View interactive dashboard
5. Test live sentiment predictions

### Command Line
```bash
python main.py data/sample_chats/chat.txt
```

---

## 🌍 Export WhatsApp Chat

### Android 📱
1. Open WhatsApp → Group/Chat
2. Tap ⋮ (Menu) → More → Export Chat
3. Choose **Without Media**
4. Save to computer

### iPhone 📱
1. Open WhatsApp → Group/Chat
2. Tap Contact Name → Export Chat
3. Choose **Without Media**
4. Save to computer

---

## 🤝 Contributing

Bugs? Suggestions? Open an issue or pull request.

---

## 📝 License

MIT License - Feel free to use and modify

---

## 💡 Tips for Best Results

- ✅ Use chats with 50+ messages for meaningful ML training
- ✅ Mix of Positive, Neutral, Negative sentiment works best
- ✅ Longer chats (500+ msgs) give more reliable patterns
- ✅ Check logs if models fail to train
- ✅ Sentiment prediction works best for English text

---

**Questions?** Check logs/ for detailed error messages

**First time?** Start with sample chat: `python main.py data/sample_chats/chat.txt`
- `data/` - Input data
- `results/` - Output analysis

## Technologies

- Python 3.8+
- Pandas, NumPy
- NLTK, Scikit-learn
- Flask
- Matplotlib, Seaborn

## License

MIT

## 🧪 Testing Suite (Industry Standard)

**Setup:**
```bash
pip install -r requirements.txt
pre-commit install
```

**Run Tests:**
```bash
pytest --cov  # 80%+ coverage
black src tests --check  # formatting
pylint src  # linting
bandit -r src  # security
```

**Full CI:**
```bash
bash tests/run_tests.sh
```

**Coverage Report:** `./htmlcov/index.html`
- Unit: src modules
- Integration: pipeline
- E2E: Flask routes
- Security: bandit scans
