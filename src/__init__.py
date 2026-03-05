# =============================================================================
# src/__init__.py
# -----------------------------------------------------------------------------
# PURPOSE : Makes the /src directory a proper Python package so all modules
#           can import from each other using:
#               from src.whatsapp_parser import parse_chat
#               from src.pipeline        import run_pipeline
#
#           Also declares the public API of the src package — only the
#           functions that app.py, main.py and pipeline.py actually need
#           are exposed here. Internal helpers (prefixed with _) stay hidden.
#
# WITHOUT THIS FILE:
#           Python treats /src as a plain folder.
#           Every import like "from src.pipeline import run_pipeline"
#           will throw: ModuleNotFoundError: No module named 'src'
#
# AUTHOR  : Team — Chaitanya / Anudeep / Ram Teja
# =============================================================================

# ── Package version ──────────────────────────────────────────────────────── #
__version__ = '1.0.0'
__author__  = 'Chaitanya Kondappa, D. Anudeep, K. Ram Teja'
__project__ = 'NLP-Based WhatsApp Chat Analysis System'

# ── whatsapp_parser ───────────────────────────────────────────────────────── #
# parse_chat()       → main entry point, reads .txt → returns DataFrame
# get_chat_summary() → basic stats dict (total messages, date range, senders)
from src.whatsapp_parser import (
    parse_chat,
    get_chat_summary,
)

# ── data_processor ───────────────────────────────────────────────────────── #
# process_dataframe()  → cleans messages, extracts emojis, detects URLs
# get_top_emojis()     → top N emoji Series for dashboard
# get_processing_stats()→ cleaning stats dict
from src.data_processor import (
    process_dataframe,
    get_top_emojis,
    get_processing_stats,
)

# ── nlp_processor ────────────────────────────────────────────────────────── #
# NLPResult      → dataclass returned by process_nlp()
# process_nlp()  → word freq, n-grams, TF-IDF, user stats, activity patterns
# get_top_words()→ top N words DataFrame
# get_top_bigrams() → top N bigrams DataFrame
# get_top_trigrams()→ top N trigrams DataFrame
# get_nlp_summary() → flat summary dict for dashboard
from src.nlp_processor import (
    NLPResult,
    process_nlp,
    get_top_words,
    get_top_bigrams,
    get_top_trigrams,
    get_nlp_summary,
)

# ── sentiment_analyzer ───────────────────────────────────────────────────── #
# analyze_sentiment()        → VADER + TextBlob scoring on full DataFrame
# get_sentiment_by_sender()  → per-sender breakdown DataFrame
# get_sentiment_over_time()  → weekly/monthly trend Series
# get_overall_mood()         → summary dict with mood description
# get_high_confidence_samples() → clean training data for ML model
from src.sentiment_analyzer import (
    analyze_sentiment,
    get_sentiment_by_sender,
    get_sentiment_over_time,
    get_overall_mood,
    get_high_confidence_samples,
)

# ── sentiment_classifier ─────────────────────────────────────────────────── #
# ClassifierResult   → dataclass returned by train_classifier()
# train_classifier() → trains Naive Bayes + Logistic Regression, saves best
# predict_sentiment()→ predict label for a single text string
# predict_bulk()     → add predicted_sentiment column to full DataFrame
# load_model_metadata() → load saved accuracy/F1 from JSON (no model reload)
# is_model_trained() → True if models/sentiment_model.pkl exists
from src.sentiment_classifier import (
    ClassifierResult,
    train_classifier,
    predict_sentiment,
    predict_bulk,
    load_model_metadata,
    is_model_trained,
)

# ── visualizer ───────────────────────────────────────────────────────────── #
# generate_all_charts() → runs all 10 chart generators, returns path dict
# get_chart_paths()     → returns existing chart paths without regenerating
from src.visualizer import (
    generate_all_charts,
    get_chart_paths,
)

# ── pipeline ─────────────────────────────────────────────────────────────── #
# PipelineOutput      → dataclass with all results bundled together
# run_pipeline()      → single entry point for the entire system
# get_pipeline_status()→ check what's been set up (model, charts)
from src.pipeline import (
    PipelineOutput,
    run_pipeline,
    get_pipeline_status,
)

# ── Public API — what's available when someone does "from src import *" ───── #
__all__ = [
    # Parser
    'parse_chat', 'get_chat_summary',
    # Processor
    'process_dataframe', 'get_top_emojis', 'get_processing_stats',
    # NLP
    'NLPResult', 'process_nlp', 'get_top_words',
    'get_top_bigrams', 'get_top_trigrams', 'get_nlp_summary',
    # Sentiment
    'analyze_sentiment', 'get_sentiment_by_sender',
    'get_sentiment_over_time', 'get_overall_mood',
    'get_high_confidence_samples',
    # Classifier
    'ClassifierResult', 'train_classifier', 'predict_sentiment',
    'predict_bulk', 'load_model_metadata', 'is_model_trained',
    # Visualizer
    'generate_all_charts', 'get_chart_paths',
    # Pipeline
    'PipelineOutput', 'run_pipeline', 'get_pipeline_status',
]