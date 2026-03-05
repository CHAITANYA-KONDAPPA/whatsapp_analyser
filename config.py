# =============================================================================
# config.py
# -----------------------------------------------------------------------------
# PURPOSE : Single source of truth for ALL project settings, paths, thresholds
#           and constants. Every module imports what it needs from here.
#
# WHY THIS FILE EXISTS:
#           Without config.py, constants like file paths, thresholds and
#           model parameters are scattered across 7 different files.
#           If you want to change the output folder, you'd have to edit
#           visualizer.py, pipeline.py, and app.py separately.
#           With config.py, you change it ONCE here and it applies everywhere.
#
# HOW TO USE:
#           from config import Paths, MLConfig, SentimentConfig, ChartConfig
#           model_path = Paths.MODEL
#           threshold  = SentimentConfig.POSITIVE_THRESHOLD
#
# AUTHOR  : Team — Chaitanya / Anudeep / Ram Teja
# =============================================================================

import os

# --------------------------------------------------------------------------- #
# PROJECT METADATA                                                             #
# --------------------------------------------------------------------------- #

PROJECT_NAME    = 'NLP-Based WhatsApp Chat Analysis System'
VERSION         = '1.0.0'
TEAM_MEMBERS    = ['Chaitanya Kondappa', 'D. Anudeep', 'K. Ram Teja']
COORDINATORS    = ['Dr. M. Sreenu', 'Dr. Afreen Fathima']


# --------------------------------------------------------------------------- #
# PATHS — All file and directory paths in one place                           #
# --------------------------------------------------------------------------- #
# Change BASE_DIR if you move the project to a different location.
# Everything else is relative to BASE_DIR automatically.

class Paths:
    """
    All file and directory paths used across the project.

    Usage:
        from config import Paths
        df = parse_chat(Paths.SAMPLE_CHAT)
        model = load(Paths.MODEL)
    """

    # ── Root directory ────────────────────────────────────────────────────── #
    # os.path.dirname(__file__) = folder where config.py lives = project root
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

    # ── Input ─────────────────────────────────────────────────────────────── #
    DATA_DIR        = os.path.join(BASE_DIR, 'data')
    SAMPLE_CHATS    = os.path.join(DATA_DIR, 'sample_chats')
    UPLOADS_DIR     = os.path.join(BASE_DIR, 'uploads')     # Flask upload destination

    # ── Output ────────────────────────────────────────────────────────────── #
    RESULTS_DIR     = os.path.join(BASE_DIR, 'results')
    CHARTS_DIR      = os.path.join(RESULTS_DIR, 'visualizations')
    LOGS_DIR        = os.path.join(BASE_DIR, 'logs')

    # ── Models ────────────────────────────────────────────────────────────── #
    MODELS_DIR      = os.path.join(BASE_DIR, 'models')
    MODEL           = os.path.join(MODELS_DIR, 'sentiment_model.pkl')
    VECTORIZER      = os.path.join(MODELS_DIR, 'tfidf_vectorizer.pkl')
    MODEL_METADATA  = os.path.join(MODELS_DIR, 'model_metadata.json')

    # ── Templates & Static (Flask) ────────────────────────────────────────── #
    TEMPLATES_DIR   = os.path.join(BASE_DIR, 'templates')
    STATIC_DIR      = os.path.join(BASE_DIR, 'static')
    CSS_DIR         = os.path.join(STATIC_DIR, 'css')
    JS_DIR          = os.path.join(STATIC_DIR, 'js')
    IMAGES_DIR      = os.path.join(STATIC_DIR, 'images')

    # ── Chart file names (fixed so app.py always knows where to find them) ── #
    CHART_FILES = {
        'wordcloud'             : '01_wordcloud.png',
        'top_words'             : '02_top_words.png',
        'hourly_activity'       : '03_hourly_activity.png',
        'daily_activity'        : '04_daily_activity.png',
        'monthly_activity'      : '05_monthly_activity.png',
        'user_participation'    : '06_user_participation.png',
        'top_bigrams'           : '07_top_bigrams.png',
        'sentiment_distribution': '08_sentiment_distribution.png',
        'sentiment_trend'       : '09_sentiment_trend.png',
        'sentiment_by_user'     : '10_sentiment_by_user.png',
    }

    @classmethod
    def chart_path(cls, key: str) -> str:
        """Return full path for a chart by key name."""
        filename = cls.CHART_FILES.get(key, '')
        return os.path.join(cls.CHARTS_DIR, filename) if filename else ''

    @classmethod
    def create_all_dirs(cls):
        """
        Create all required directories if they don't exist.
        Call this once at app startup in app.py.
        """
        dirs = [
            cls.DATA_DIR, cls.SAMPLE_CHATS, cls.UPLOADS_DIR,
            cls.RESULTS_DIR, cls.CHARTS_DIR, cls.LOGS_DIR,
            cls.MODELS_DIR, cls.TEMPLATES_DIR,
            cls.STATIC_DIR, cls.CSS_DIR, cls.JS_DIR, cls.IMAGES_DIR,
        ]
        for d in dirs:
            os.makedirs(d, exist_ok=True)


# --------------------------------------------------------------------------- #
# FLASK CONFIG                                                                 #
# --------------------------------------------------------------------------- #

class FlaskConfig:
    """
    Flask application settings.
    Import in app.py: app.config.from_object(FlaskConfig)
    """
    SECRET_KEY              = 'whatsapp-nlp-secret-2024'  # change in production
    DEBUG                   = True                          # set False for client delivery
    HOST                    = '0.0.0.0'
    PORT                    = 5000
    MAX_CONTENT_LENGTH      = 16 * 1024 * 1024             # 16MB max upload size
    UPLOAD_FOLDER           = Paths.UPLOADS_DIR
    ALLOWED_EXTENSIONS      = {'txt'}                       # only .txt chat exports
    TEMPLATES_AUTO_RELOAD   = True


# --------------------------------------------------------------------------- #
# PARSER CONFIG                                                                #
# --------------------------------------------------------------------------- #

class ParserConfig:
    """
    Settings for whatsapp_parser.py
    """
    # Messages that count as media placeholders (skip in text analysis)
    MEDIA_STRINGS = [
        '<media omitted>', 'image omitted', 'video omitted',
        'audio omitted', 'document omitted', 'sticker omitted',
        'gif omitted',
    ]

    # Messages that mean sender deleted their message
    DELETED_STRINGS = [
        'this message was deleted',
        'you deleted this message',
    ]

    # Encoding fallback order when reading .txt files
    ENCODINGS = ['utf-8', 'latin-1', 'utf-8-sig']

    # Minimum message length (characters) to keep
    MIN_MESSAGE_LENGTH = 2


# --------------------------------------------------------------------------- #
# NLP / TEXT PROCESSING CONFIG                                                #
# --------------------------------------------------------------------------- #

class NLPConfig:
    """
    Settings for data_processor.py and nlp_processor.py
    """
    # Extra chat-specific stopwords beyond NLTK defaults
    CUSTOM_STOP_WORDS = {
        'ok', 'okay', 'yeah', 'yep', 'yes', 'no', 'oh',
        'hmm', 'hm', 'ah', 'ha', 'lol', 'omg', 'lmao',
        'bro', 'guy', 'guys', 'hi', 'hey', 'hello', 'bye',
        'haha', 'hahaha', 'hehe', 'na', 'nah', 'yaar',
        'message', 'deleted', 'media', 'omitted', 'null',
    }

    # Minimum word length after cleaning (shorter = usually noise)
    MIN_WORD_LENGTH = 2

    # Top N words to return from get_top_words()
    TOP_WORDS_DEFAULT = 20

    # Top N bigrams / trigrams
    TOP_NGRAMS_DEFAULT = 10

    # Top N TF-IDF keywords per sender
    TOP_TFIDF_KEYWORDS = 10

    # Top N emojis to show in chart
    TOP_EMOJIS = 10

    # Response time cap in seconds (gaps > this = new conversation, not a reply)
    MAX_RESPONSE_GAP_SECONDS = 43200  # 12 hours


# --------------------------------------------------------------------------- #
# SENTIMENT CONFIG                                                             #
# --------------------------------------------------------------------------- #

class SentimentConfig:
    """
    Settings for sentiment_analyzer.py
    """
    # VADER compound score thresholds (industry standard)
    POSITIVE_THRESHOLD = 0.05
    NEGATIVE_THRESHOLD = -0.05

    # Confidence level thresholds based on absolute compound score
    HIGH_CONFIDENCE   = 0.5    # |score| >= 0.5
    MEDIUM_CONFIDENCE = 0.2    # |score| >= 0.2

    # Labels (consistent strings used everywhere)
    LABEL_POSITIVE = 'Positive'
    LABEL_NEGATIVE = 'Negative'
    LABEL_NEUTRAL  = 'Neutral'

    ALL_LABELS = [LABEL_POSITIVE, LABEL_NEUTRAL, LABEL_NEGATIVE]

    # Sentiment trend chart time frequency
    # 'D' = daily, 'W' = weekly, 'M' = monthly
    TREND_FREQUENCY = 'W'

    # Human-readable mood descriptions by score range
    MOOD_DESCRIPTIONS = {
        (0.30,  1.00) : 'Very Positive 😊',
        (0.05,  0.30) : 'Generally Positive 🙂',
        (-0.05, 0.05) : 'Mostly Neutral 😶',
        (-0.30,-0.05) : 'Generally Negative 😐',
        (-1.00,-0.30) : 'Very Negative 😔',
    }


# --------------------------------------------------------------------------- #
# ML MODEL CONFIG                                                              #
# --------------------------------------------------------------------------- #

class MLConfig:
    """
    Settings for sentiment_classifier.py
    """
    # Minimum text messages required to train the model
    MIN_SAMPLES_FOR_TRAINING = 30

    # Minimum total messages to attempt ML at all
    MIN_MESSAGES_FOR_ML = 50

    # How many high-confidence samples to pull per training run
    TRAINING_SAMPLES = 300

    # Train / test split ratio
    TEST_SIZE    = 0.20
    RANDOM_STATE = 42       # fixed seed = reproducible results

    # TF-IDF vectorizer settings
    TFIDF_MAX_FEATURES = 5000
    TFIDF_NGRAM_RANGE  = (1, 2)    # unigrams + bigrams
    TFIDF_MIN_DF       = 2         # word must appear in 2+ messages
    TFIDF_SUBLINEAR_TF = True      # log-scale term frequency

    # Logistic Regression settings
    LR_C            = 1.0
    LR_MAX_ITER     = 500
    LR_CLASS_WEIGHT = 'balanced'
    LR_SOLVER       = 'lbfgs'

    # Naive Bayes settings
    NB_ALPHA = 0.1      # Laplace smoothing

    # Cross validation folds
    CV_FOLDS = 5


# --------------------------------------------------------------------------- #
# CHART / VISUALISATION CONFIG                                                 #
# --------------------------------------------------------------------------- #

class ChartConfig:
    """
    Settings for visualizer.py — change these to restyle all charts at once.
    """
    # Resolution — 150 dpi is crisp on web without being huge
    DPI = 150

    # Standard figure sizes (width, height) in inches
    FIGSIZE_STANDARD = (10, 5)
    FIGSIZE_SQUARE   = (8, 8)
    FIGSIZE_WIDE     = (12, 5)
    FIGSIZE_TALL     = (10, 7)

    # Color palette (consistent across ALL charts)
    COLORS = {
        'primary'    : '#2E75B6',
        'secondary'  : '#1F4E79',
        'accent'     : '#70AD47',
        'positive'   : '#70AD47',
        'neutral'    : '#FFC000',
        'negative'   : '#FF4B4B',
        'background' : '#F8F9FA',
        'grid'       : '#E0E0E0',
        'text'       : '#1A1A2E',
    }

    # Per-sender colors (up to 8 senders supported)
    SENDER_PALETTE = [
        '#2E75B6', '#70AD47', '#FF4B4B', '#FFC000',
        '#9B59B6', '#E67E22', '#1ABC9C', '#E74C3C',
    ]

    # WordCloud settings
    WORDCLOUD_MAX_WORDS        = 150
    WORDCLOUD_BACKGROUND_COLOR = 'white'
    WORDCLOUD_COLORMAP         = 'Blues'
    WORDCLOUD_WIDTH            = 1200
    WORDCLOUD_HEIGHT           = 600

    # Default number of items in charts
    TOP_WORDS_CHART  = 20
    TOP_BIGRAMS_CHART = 10


# --------------------------------------------------------------------------- #
# LOGGING CONFIG                                                               #
# --------------------------------------------------------------------------- #

class LogConfig:
    """
    Logging settings used by all modules.
    """
    LEVEL   = 'INFO'
    FORMAT  = '%(levelname)s: %(message)s'
    DATE_FMT= '%Y-%m-%d %H:%M:%S'

    # Log file path (app.py writes to this)
    LOG_FILE = os.path.join(Paths.LOGS_DIR, 'app.log')

    # Max log file size before rotation (5MB)
    MAX_BYTES    = 5 * 1024 * 1024
    BACKUP_COUNT = 3


# --------------------------------------------------------------------------- #
# QUICK VALIDATION — python config.py                                         #
# --------------------------------------------------------------------------- #

if __name__ == '__main__':
    print(f"\n{'='*55}")
    print(f" {PROJECT_NAME}")
    print(f" Version : {VERSION}")
    print(f" Team    : {', '.join(TEAM_MEMBERS)}")
    print(f"{'='*55}")

    print("\n── Paths ─────────────────────────────────────────────")
    print(f"  Base Dir    : {Paths.BASE_DIR}")
    print(f"  Uploads     : {Paths.UPLOADS_DIR}")
    print(f"  Charts      : {Paths.CHARTS_DIR}")
    print(f"  Models      : {Paths.MODELS_DIR}")
    print(f"  Model File  : {Paths.MODEL}")
    print(f"  Logs        : {Paths.LOGS_DIR}")

    print("\n── Sentiment Thresholds ──────────────────────────────")
    print(f"  Positive    : score >= {SentimentConfig.POSITIVE_THRESHOLD}")
    print(f"  Negative    : score <= {SentimentConfig.NEGATIVE_THRESHOLD}")
    print(f"  High Conf   : |score| >= {SentimentConfig.HIGH_CONFIDENCE}")

    print("\n── ML Config ─────────────────────────────────────────")
    print(f"  Min Samples : {MLConfig.MIN_SAMPLES_FOR_TRAINING}")
    print(f"  Test Size   : {MLConfig.TEST_SIZE}")
    print(f"  TF-IDF Feats: {MLConfig.TFIDF_MAX_FEATURES}")
    print(f"  N-gram Range: {MLConfig.TFIDF_NGRAM_RANGE}")
    print(f"  CV Folds    : {MLConfig.CV_FOLDS}")

    print("\n── Flask Config ──────────────────────────────────────")
    print(f"  Host        : {FlaskConfig.HOST}:{FlaskConfig.PORT}")
    print(f"  Debug       : {FlaskConfig.DEBUG}")
    print(f"  Max Upload  : {FlaskConfig.MAX_CONTENT_LENGTH // (1024*1024)}MB")

    print("\n── Chart Config ──────────────────────────────────────")
    print(f"  DPI         : {ChartConfig.DPI}")
    print(f"  Colors      : {len(ChartConfig.COLORS)} defined")
    print(f"  WordCloud   : {ChartConfig.WORDCLOUD_MAX_WORDS} words max")

    print("\n── Creating directories... ───────────────────────────")
    Paths.create_all_dirs()
    print("  ✓ All directories created/verified")

    print(f"\n{'='*55}\n")