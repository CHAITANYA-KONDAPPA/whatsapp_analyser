# WhatsApp Chat Analysis System

NLP-based sentiment analysis and communication pattern extraction from WhatsApp chats.

## Features

- **Parse WhatsApp Exports**: Automatically parse exported .txt files
- **Data Cleaning**: Remove sensitive info, normalize text
- **NLP Processing**: Tokenization, stemming, TF-IDF vectorization
- **Sentiment Analysis**: 3-model ensemble (Naive Bayes, Logistic Regression, Random Forest)
- **Visualizations**: Charts, heatmaps, word clouds
- **Web Interface**: Beautiful Flask UI for easy analysis

## Quick Start

### Installation
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download NLTK data
python -m nltk.downloader punkt stopwords wordnet

#Download TextBlob for Sentiment Analysing
python -m pip install textblob
```

### Usage

#### Command Line:
```bash
python main.py chat_export.txt
```

#### Web Interface:
```bash
python app.py
# Open http://localhost:5000
```

## Project Structure

- `src/` - Python modules
- `templates/` - Flask HTML templates
- `static/` - CSS, JavaScript
- `models/` - Trained ML models
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