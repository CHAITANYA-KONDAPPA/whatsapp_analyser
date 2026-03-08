# =============================================================================
# app.py
# -----------------------------------------------------------------------------
# PURPOSE : Flask web server — the bridge between the Python analysis backend
#           and the HTML frontend that the user sees in their browser.
#
# ROUTES  :
#   GET  /              → index.html      (upload page)
#   POST /upload        → run pipeline    (analysis entry point)
#   GET  /dashboard     → dashboard.html  (charts + stats)
#   GET  /results       → results.html    (detailed per-user breakdown)
#   POST /predict       → JSON response   (live sentiment checker)
#   GET  /charts/<file> → serve chart images
#   GET  /status        → JSON system status
#   GET  /reset         → clear session, back to home
#
# STARTUP :
#   python app.py
#   Then open: http://localhost:5000
#
# AUTHOR  : Team — Chaitanya / Anudeep / Ram Teja
# =============================================================================

import os
import json
import logging
import traceback
from logging.handlers import RotatingFileHandler
from datetime         import datetime

from flask import (
    Flask,
    render_template,
    request,
    redirect,
    url_for,
    flash,
    jsonify,
    send_from_directory,
    session,
)

# ── Project imports ───────────────────────────────────────────────────────── #
from config import Paths, FlaskConfig, LogConfig
from src.pipeline             import run_pipeline, get_pipeline_status
from src.sentiment_classifier import predict_sentiment
from src.sentiment_analyzer   import get_sentiment_over_time


# --------------------------------------------------------------------------- #
# APP INITIALISATION                                                           #
# --------------------------------------------------------------------------- #

app = Flask(
    __name__,
    template_folder = Paths.TEMPLATES_DIR,
    static_folder   = Paths.STATIC_DIR,
)

# Load all Flask settings from FlaskConfig in config.py
app.config.from_object(FlaskConfig)

# Ensure all required directories exist on startup
Paths.create_all_dirs()


# --------------------------------------------------------------------------- #
# LOGGING SETUP                                                               #
# --------------------------------------------------------------------------- #

def _setup_logging():
    """
    Sets up two log handlers:
        1. Console  — INFO level, visible in terminal while running
        2. File     — INFO level, written to logs/app.log with rotation

    RotatingFileHandler limits log file to 5MB then starts a new one.
    Keeps last 3 log files (logs/app.log, app.log.1, app.log.2).
    """
    formatter = logging.Formatter(
        fmt     = '%(asctime)s | %(levelname)-8s | %(message)s',
        datefmt = LogConfig.DATE_FMT,
    )

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    console_handler.setLevel(logging.INFO)

    # File handler with rotation
    file_handler = RotatingFileHandler(
        LogConfig.LOG_FILE,
        maxBytes    = LogConfig.MAX_BYTES,
        backupCount = LogConfig.BACKUP_COUNT,
    )
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.INFO)

    # Apply to root logger so all modules log consistently
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)


_setup_logging()
logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# IN-MEMORY STORE                                                             #
# --------------------------------------------------------------------------- #
# Flask sessions store simple values (strings, ints) in a cookie.
# Our PipelineOutput is a large Python object — it can't go in a cookie.
# So we store it in this dict in server memory, keyed by session ID.
#
# NOTE: This resets when the server restarts.
# For a production deployment you would use Redis or a database instead,
# but for a local/lab demo this is perfectly fine.

_pipeline_store: dict = {}    # { session_id : PipelineOutput }


def _save_output(output) -> str:
    """Save PipelineOutput to memory store. Returns a unique session key."""
    key = datetime.now().strftime('%Y%m%d%H%M%S%f')
    _pipeline_store[key] = output
    session['analysis_key'] = key
    return key


def _load_output():
    """Load PipelineOutput from memory store using session key. Returns None if not found."""
    key = session.get('analysis_key')
    return _pipeline_store.get(key) if key else None


# --------------------------------------------------------------------------- #
# HELPERS                                                                     #
# --------------------------------------------------------------------------- #

def _allowed_file(filename: str) -> bool:
    """Check the uploaded file has a .txt extension."""
    return (
        '.' in filename and
        filename.rsplit('.', 1)[1].lower() in FlaskConfig.ALLOWED_EXTENSIONS
    )


def _safe_filename(filename: str) -> str:
    """
    Sanitise filename — remove path separators and dangerous characters.
    Prevents directory traversal attacks (e.g. '../../etc/passwd').
    """
    filename = os.path.basename(filename)
    keep     = set('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._- ')
    return ''.join(c for c in filename if c in keep).strip()


def _dataframe_to_records(df):
    """Convert a DataFrame to a list of dicts for Jinja2 template rendering."""
    if df is None or df.empty:
        return []
    return df.to_dict(orient='records')


# --------------------------------------------------------------------------- #
# ROUTE 1 — Home / Upload Page                                               #
# --------------------------------------------------------------------------- #

@app.route('/', methods=['GET'])
def index():
    """
    Renders the home page with the chat file upload form.

    Also passes system status so the page can show:
        "Model trained: Yes | Charts available: 10"
    before any analysis is run.
    """
    status = get_pipeline_status()
    return render_template('index.html', status=status)


# --------------------------------------------------------------------------- #
# ROUTE 2 — File Upload & Pipeline Trigger                                   #
# --------------------------------------------------------------------------- #

@app.route('/upload', methods=['POST'])
def upload():
    """
    Handles the chat file upload form submission.

    Steps:
        1. Validate — file present, .txt extension, not empty
        2. Save to uploads/
        3. Run run_pipeline() on the saved file
        4. Store PipelineOutput in memory
        5. Redirect to /dashboard on success
        6. Flash error and redirect to / on failure

    Flash messages are shown as banners in the HTML templates.
    """
    logger.info("──────────────────────────────────────")
    logger.info("Upload request received")

    # ── 1. Check file was submitted ───────────────────────────────────────── #
    if 'chat_file' not in request.files:
        flash('No file selected. Please choose a WhatsApp chat export (.txt).', 'error')
        return redirect(url_for('index'))

    file = request.files['chat_file']

    if file.filename == '':
        flash('No file selected. Please choose a WhatsApp chat export (.txt).', 'error')
        return redirect(url_for('index'))

    # ── 2. Validate extension ─────────────────────────────────────────────── #
    if not _allowed_file(file.filename):
        flash('Invalid file type. Please upload a .txt WhatsApp export file.', 'error')
        return redirect(url_for('index'))

    # ── 3. Save file to uploads/ ──────────────────────────────────────────── #
    safe_name  = _safe_filename(file.filename)
    timestamp  = datetime.now().strftime('%Y%m%d_%H%M%S')
    saved_name = f"{timestamp}_{safe_name}"
    file_path  = os.path.join(Paths.UPLOADS_DIR, saved_name)

    file.save(file_path)
    logger.info(f"File saved: {file_path}")

    # ── 4. Check if force retrain was requested ───────────────────────────── #
    force_retrain = request.form.get('force_retrain', 'false').lower() == 'true'

    # ── 5. Run pipeline ───────────────────────────────────────────────────── #
    logger.info("Starting pipeline...")
    output = run_pipeline(file_path, force_retrain=force_retrain)

    if not output.success:
        logger.error(f"Pipeline failed: {output.error_message}")
        flash(f"Analysis failed: {output.error_message}", 'error')
        return redirect(url_for('index'))

    # ── 6. Store output and redirect to dashboard ─────────────────────────── #
    _save_output(output)
    logger.info(f"Pipeline complete in {output.processing_time}s — redirecting to dashboard")
    flash(f"Analysis complete! Processed {output.chat_summary.get('total_messages', 0):,} messages "
          f"in {output.processing_time}s.", 'success')

    return redirect(url_for('dashboard'))


# --------------------------------------------------------------------------- #
# ROUTE 3 — Dashboard                                                        #
# --------------------------------------------------------------------------- #

@app.route('/dashboard', methods=['GET'])
def dashboard():
    """
    Renders the main analysis dashboard.

    Passes to dashboard.html:
        chart_paths     → dict of chart image URLs
        overall_mood    → mood description, score, percentages
        chat_summary    → total messages, senders, date range
        nlp_summary     → top word, peak hour, fastest responder
        model_metadata  → ML accuracy, F1 score
        top_emojis      → list of (emoji, count) tuples
        processing_time → seconds the analysis took
        mode_used       → FULL / FAST / NO_ML
    """
    output = _load_output()

    if output is None:
        flash('No analysis found. Please upload a chat file first.', 'info')
        return redirect(url_for('index'))

    # Convert chart absolute paths → URL paths for <img src=""> tags
    chart_urls = {}
    for key, path in output.chart_paths.items():
        if path and os.path.exists(path):
            chart_urls[key] = url_for('serve_chart', filename=os.path.basename(path))
        else:
            chart_urls[key] = ''

    # Convert top_emojis Series → list of (emoji, count) for Jinja2
    top_emojis_list = []
    if not output.top_emojis.empty:
        top_emojis_list = list(output.top_emojis.items())

    # ── Serialize raw NLPResult data for Plotly charts (JSON) ────────────── #
    nlp = output.nlp_result
    chart_data = {}

    if nlp:
        # Activity data
        chart_data['hourly']  = nlp.hour_activity.tolist() if not nlp.hour_activity.empty else []
        chart_data['daily']   = {
            'labels': nlp.day_activity.index.tolist(),
            'values': nlp.day_activity.tolist(),
        } if not nlp.day_activity.empty else {}
        chart_data['monthly'] = {
            'labels': [str(k) for k in nlp.month_activity.index.tolist()],
            'values': nlp.month_activity.tolist(),
        } if not nlp.month_activity.empty else {}

        # Word data
        top_words = nlp.word_frequencies.most_common(20)
        chart_data['top_words']  = {'words': [w for w,_ in top_words], 'counts': [c for _,c in top_words]}
        top_bigrams = nlp.bigrams.most_common(10)
        chart_data['bigrams']    = {'phrases': [' '.join(b) if isinstance(b,tuple) else str(b) for b,_ in top_bigrams], 'counts': [c for _,c in top_bigrams]}
        chart_data['tfidf']      = {s: [w for w,_ in kws] for s, kws in nlp.tfidf_keywords.items()}
        chart_data['response_times'] = output.nlp_result.response_times if output.nlp_result.response_times else {}

        # User participation
        if not nlp.user_stats.empty:
            chart_data['user_msgs'] = {
                'senders': nlp.user_stats['sender'].tolist(),
                'counts' : nlp.user_stats['total_messages'].tolist(),
            }
        else:
            chart_data['user_msgs'] = {}

    # Sentiment data for Plotly
    if not output.sentiment_by_sender.empty:
        sb = output.sentiment_by_sender
        chart_data['sentiment_by_sender'] = {
            'senders'    : sb['sender'].tolist(),
            'positive'   : sb['positive_pct'].tolist(),
            'neutral'    : sb['neutral_pct'].tolist(),
            'negative'   : sb['negative_pct'].tolist(),
            'avg_scores' : sb['avg_score'].tolist(),
        }

    # Sentiment over time
    try:
        sent_trend = get_sentiment_over_time(output.df, freq='W')
        if not sent_trend.empty:
            chart_data['sentiment_trend'] = {
                'dates' : [str(d) for d in sent_trend.index.tolist()],
                'scores': sent_trend.tolist(),
            }
    except Exception:
        chart_data['sentiment_trend'] = {}

    # Overall sentiment distribution
    chart_data['sentiment_dist'] = {
        'labels': ['Positive', 'Neutral', 'Negative'],
        'values': [
            round(output.overall_mood.get('positive_pct', 0), 1),
            round(output.overall_mood.get('neutral_pct',  0), 1),
            round(output.overall_mood.get('negative_pct', 0), 1),
        ],
    }

    # Emojis
    chart_data['emojis'] = {
        'emojis': [e for e, _ in top_emojis_list],
        'counts': [c for _, c in top_emojis_list],
    }

    # Model data
    meta = output.model_metadata
    if meta:
        chart_data['model'] = {
            'accuracy'        : meta.get('accuracy',         0),
            'f1_score'        : meta.get('f1_score',         0),
            'cv_mean'         : meta.get('cross_val_mean',   0),
            'cv_std'          : meta.get('cross_val_std',    0),
            'training_samples': meta.get('training_samples', 0),
            'test_samples'    : meta.get('test_samples',     0),
            'model_name'      : meta.get('model_name',       ''),
            'trained_at'      : meta.get('trained_at',       ''),
            'classes'         : meta.get('classes',          []),
        }

    return render_template(
        'dashboard.html',
        chart_data      = json.dumps(chart_data),
        overall_mood    = output.overall_mood,
        chat_summary    = output.chat_summary,
        nlp_summary     = output.nlp_summary,
        model_metadata  = output.model_metadata,
        top_emojis      = top_emojis_list,
        sentiment_rows  = _dataframe_to_records(output.sentiment_by_sender),
        user_stats_rows = _dataframe_to_records(nlp.user_stats) if nlp and not nlp.user_stats.empty else [],
        tfidf_keywords  = nlp.tfidf_keywords if nlp else {},
        response_times  = nlp.response_times if nlp else {},
        processing_time = output.processing_time,
        mode_used       = output.mode_used,
        ml_trained      = output.ml_trained,
        run_timestamp   = output.run_timestamp,
    )


# --------------------------------------------------------------------------- #
# ROUTE 4 — Results (Detailed Per-User Breakdown)                            #
# --------------------------------------------------------------------------- #

@app.route('/results', methods=['GET'])
def results():
    """
    Renders the detailed results page.

    Passes to results.html:
        sentiment_by_sender → per-user Positive/Neutral/Negative %
        user_stats          → messages, words, emojis per user
        processing_stats    → data cleaning summary
        model_metadata      → ML model performance details
        tfidf_keywords      → each user's signature keywords
    """
    output = _load_output()

    if output is None:
        flash('No analysis found. Please upload a chat file first.', 'info')
        return redirect(url_for('index'))

    # Convert DataFrames to lists of dicts for Jinja2
    sentiment_rows = _dataframe_to_records(output.sentiment_by_sender)

    # User stats from NLPResult
    user_stats_rows = []
    if output.nlp_result and not output.nlp_result.user_stats.empty:
        user_stats_rows = _dataframe_to_records(output.nlp_result.user_stats)

    # TF-IDF keywords per sender
    tfidf_keywords = {}
    if output.nlp_result:
        tfidf_keywords = output.nlp_result.tfidf_keywords

    # Response times
    response_times = {}
    if output.nlp_result:
        response_times = output.nlp_result.response_times

    return render_template(
        'results.html',
        sentiment_rows  = sentiment_rows,
        user_stats_rows = user_stats_rows,
        tfidf_keywords  = tfidf_keywords,
        response_times  = response_times,
        processing_stats= output.processing_stats,
        model_metadata  = output.model_metadata,
        chat_summary    = output.chat_summary,
    )


# --------------------------------------------------------------------------- #
# ROUTE 5 — Live Sentiment Predictor (AJAX)                                  #
# --------------------------------------------------------------------------- #

@app.route('/predict', methods=['POST'])
def predict():
    """
    Live sentiment prediction endpoint.
    Called by dashboard.html's JavaScript when user types a message
    and clicks "Check Sentiment" — returns JSON, no page reload.

    Request body (JSON):
        { "text": "I love this project!!" }

    Response (JSON):
        {
            "label"        : "Positive",
            "confidence"   : 0.91,
            "probabilities": { "Positive": 0.91, "Neutral": 0.07, "Negative": 0.02 },
            "success"      : true
        }
    """
    data = request.get_json(silent=True)

    if not data or 'text' not in data:
        return jsonify({'success': False, 'error': 'No text provided.'}), 400

    text = data['text'].strip()

    if not text:
        return jsonify({'success': False, 'error': 'Text is empty.'}), 400

    if len(text) > 500:
        return jsonify({'success': False, 'error': 'Text too long (max 500 characters).'}), 400

    try:
        result = predict_sentiment(text)
        return jsonify({
            'success'      : True,
            'label'        : result['label'],
            'confidence'   : result['confidence'],
            'probabilities': result['probabilities'],
        })
    except FileNotFoundError:
        return jsonify({
            'success': False,
            'error'  : 'Model not trained yet. Upload and analyse a chat file first.'
        }), 503
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return jsonify({'success': False, 'error': 'Prediction failed.'}), 500


# --------------------------------------------------------------------------- #
# ROUTE 6 — Serve Chart Images                                               #
# --------------------------------------------------------------------------- #

@app.route('/charts/<filename>')
def serve_chart(filename: str):
    """
    Serves chart .png files from results/visualizations/ to the browser.

    Called automatically by <img src="/charts/01_wordcloud.png"> in HTML.
    Flask handles the file reading and response headers.

    Security: send_from_directory prevents directory traversal —
    it will only serve files inside Paths.CHARTS_DIR.
    """
    return send_from_directory(Paths.CHARTS_DIR, filename)


# --------------------------------------------------------------------------- #
# ROUTE 7 — System Status (JSON)                                             #
# --------------------------------------------------------------------------- #

@app.route('/status', methods=['GET'])
def status():
    """
    Returns system status as JSON.
    Can be called from the browser or by automated tests to check
    if the system is running and what state it's in.

    Response:
    {
        "server"         : "running",
        "model_trained"  : true,
        "charts_available": 10,
        "has_analysis"   : true,
        "model_accuracy" : 0.847,
        "timestamp"      : "2024-01-23 21:45:00"
    }
    """
    pipeline_status = get_pipeline_status()
    output          = _load_output()

    return jsonify({
        'server'           : 'running',
        'model_trained'    : pipeline_status.get('model_trained', False),
        'charts_available' : pipeline_status.get('charts_available', 0),
        'has_analysis'     : output is not None,
        'model_accuracy'   : pipeline_status.get('model_metadata', {}).get('accuracy', None),
        'timestamp'        : datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    })


# --------------------------------------------------------------------------- #
# ROUTE 8 — Reset (Clear Session)                                            #
# --------------------------------------------------------------------------- #

@app.route('/reset', methods=['GET'])
def reset():
    """
    Clears the current analysis from memory and session.
    User can upload a new file after this.
    """
    key = session.get('analysis_key')
    if key and key in _pipeline_store:
        del _pipeline_store[key]
        logger.info(f"Session {key} cleared from memory store.")

    session.clear()
    flash('Analysis cleared. You can upload a new chat file.', 'info')
    return redirect(url_for('index'))


# --------------------------------------------------------------------------- #
# ERROR HANDLERS                                                              #
# --------------------------------------------------------------------------- #

@app.errorhandler(404)
def not_found(e):
    """Custom 404 page — shown when user navigates to a non-existent route."""
    return render_template('base.html',
        error_code    = 404,
        error_message = 'Page not found.',
        show_error    = True,
    ), 404


@app.errorhandler(413)
def file_too_large(e):
    """Shown when uploaded file exceeds MAX_CONTENT_LENGTH (16MB)."""
    flash('File too large. Maximum size is 16MB.', 'error')
    return redirect(url_for('index'))


@app.errorhandler(500)
def server_error(e):
    """Custom 500 page — shown on unexpected server errors."""
    logger.error(f"500 error: {e}\n{traceback.format_exc()}")
    return render_template('base.html',
        error_code    = 500,
        error_message = 'Something went wrong on our end. Please try again.',
        show_error    = True,
    ), 500


# --------------------------------------------------------------------------- #
# TEMPLATE CONTEXT PROCESSORS                                                 #
# --------------------------------------------------------------------------- #

@app.context_processor
def inject_globals():
    """
    Injects variables into EVERY template automatically.
    No need to pass these manually in every render_template() call.

    Available in all HTML templates as:
        {{ project_name }}
        {{ current_year }}
        {{ has_analysis }}
    """
    return {
        'project_name' : 'WhatsApp NLP Analyser',
        'current_year' : datetime.now().year,
        'has_analysis' : _load_output() is not None,
        'version'      : '1.0.0',
    }


# --------------------------------------------------------------------------- #
# TEMPLATE FILTERS                                                            #
# --------------------------------------------------------------------------- #

@app.template_filter('format_number')
def format_number(value):
    """Jinja2 filter: formats 1420 as '1,420' in templates."""
    try:
        return f"{int(value):,}"
    except (ValueError, TypeError):
        return value


@app.template_filter('format_pct')
def format_pct(value):
    """Jinja2 filter: formats 0.847 as '84.7%' in templates."""
    try:
        return f"{float(value) * 100:.1f}%"
    except (ValueError, TypeError):
        return value


@app.template_filter('sentiment_color')
def sentiment_color(label):
    """Jinja2 filter: maps sentiment label to a CSS color class."""
    colors = {
        'Positive': 'text-success',
        'Negative': 'text-danger',
        'Neutral' : 'text-warning',
    }
    return colors.get(label, 'text-secondary')


# --------------------------------------------------------------------------- #
# STARTUP                                                                     #
# --------------------------------------------------------------------------- #

if __name__ == '__main__':
    logger.info("=" * 55)
    logger.info(" WhatsApp NLP Analysis System — Starting up")
    logger.info("=" * 55)
    logger.info(f" URL      : http://localhost:{FlaskConfig.PORT}")
    logger.info(f" Debug    : {FlaskConfig.DEBUG}")
    logger.info(f" Uploads  : {Paths.UPLOADS_DIR}")
    logger.info(f" Charts   : {Paths.CHARTS_DIR}")
    logger.info(f" Models   : {Paths.MODELS_DIR}")
    logger.info(f" Logs     : {LogConfig.LOG_FILE}")
    logger.info("=" * 55)

    app.run(
        host  = FlaskConfig.HOST,
        port  = FlaskConfig.PORT,
        debug = FlaskConfig.DEBUG,
    )