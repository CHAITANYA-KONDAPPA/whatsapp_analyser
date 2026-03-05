# =============================================================================
# pipeline.py
# -----------------------------------------------------------------------------
# PURPOSE : The single entry point for the entire analysis system.
#           Chains all modules together in the correct order and returns one
#           clean PipelineOutput object containing everything app.py and
#           main.py need to render the dashboard and reports.
#
# MODES   :
#   FULL      → Parse → Clean → NLP → Sentiment → Train ML → Predict → Charts
#   FAST      → Parse → Clean → NLP → Sentiment → Load Model → Predict → Charts
#   NO_ML     → Parse → Clean → NLP → Sentiment → Charts (no ML)
#
# USAGE   :
#   from src.pipeline import run_pipeline
#   output = run_pipeline('data/sample_chats/chat.txt')
#
#   if output.success:
#       print(output.overall_mood)
#       print(output.chart_paths)
#   else:
#       print(output.error_message)
#
# AUTHOR  : Team — Chaitanya / Anudeep / Ram Teja
# =============================================================================

import os
import time
import logging
import traceback
from dataclasses import dataclass, field
from typing      import Dict, Optional
from datetime    import datetime

import pandas as pd

# ── All module imports ───────────────────────────────────────────────────── #
from src.whatsapp_parser     import parse_chat,               get_chat_summary
from src.data_processor      import process_dataframe,        get_processing_stats, get_top_emojis
from src.nlp_processor       import process_nlp,              get_nlp_summary,      NLPResult
from src.sentiment_analyzer  import (
    analyze_sentiment,
    get_overall_mood,
    get_sentiment_by_sender,
    get_high_confidence_samples,
)
from src.sentiment_classifier import (
    train_classifier,
    predict_bulk,
    load_model_metadata,
    is_model_trained,
    ClassifierResult,
)
from src.visualizer          import generate_all_charts,      get_chart_paths

# --------------------------------------------------------------------------- #
# Logging                                                                      #
# --------------------------------------------------------------------------- #
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# CONSTANTS                                                                    #
# --------------------------------------------------------------------------- #

# Minimum messages to attempt ML training
MIN_MESSAGES_FOR_ML = 50

# High-confidence samples to pull for ML training
ML_TRAINING_SAMPLES = 300


# --------------------------------------------------------------------------- #
# PIPELINE OUTPUT CONTAINER                                                    #
# --------------------------------------------------------------------------- #

@dataclass
class PipelineOutput:
    """
    Single object returned by run_pipeline().
    Contains everything app.py and main.py need — no further processing required.

    Usage:
        output = run_pipeline('chat.txt')

        # Check success first
        if not output.success:
            print(output.error_message)

        # Access data
        output.df                  → full DataFrame with ALL columns
        output.nlp_result          → NLPResult (word freq, activity, user stats)
        output.chart_paths         → {'wordcloud': 'results/.../01_wordcloud.png', ...}
        output.overall_mood        → {'mood_description': 'Generally Positive 🙂', ...}
        output.chat_summary        → {'total_messages': 1420, 'date_start': '2023-01-01', ...}
        output.nlp_summary         → {'top_word': 'meet', 'peak_hour': '21:00', ...}
        output.model_metadata      → {'accuracy': 0.847, 'f1_score': 0.831, ...}
        output.sentiment_by_sender → per-sender sentiment breakdown DataFrame
        output.top_emojis          → Series of top 10 emojis
        output.processing_stats    → data cleaning stats dict
        output.processing_time     → total seconds taken
        output.mode_used           → 'FULL' / 'FAST' / 'NO_ML'
        output.ml_trained          → True if ML model was trained this run
        output.success             → True if pipeline completed without fatal error
        output.error_message       → description of error if success=False
        output.run_timestamp       → datetime string of when pipeline ran
    """

    # ── Core data ────────────────────────────────────────────────────────── #
    df                  : pd.DataFrame          = field(default_factory=pd.DataFrame)
    nlp_result          : Optional[NLPResult]   = None

    # ── Chart paths ──────────────────────────────────────────────────────── #
    chart_paths         : Dict[str, str]        = field(default_factory=dict)

    # ── Summary dicts for dashboard ──────────────────────────────────────── #
    overall_mood        : Dict                  = field(default_factory=dict)
    chat_summary        : Dict                  = field(default_factory=dict)
    nlp_summary         : Dict                  = field(default_factory=dict)
    model_metadata      : Dict                  = field(default_factory=dict)
    processing_stats    : Dict                  = field(default_factory=dict)

    # ── DataFrame-based outputs ───────────────────────────────────────────── #
    sentiment_by_sender : pd.DataFrame          = field(default_factory=pd.DataFrame)
    top_emojis          : pd.Series             = field(default_factory=pd.Series)

    # ── Pipeline metadata ────────────────────────────────────────────────── #
    processing_time     : float                 = 0.0
    mode_used           : str                   = ''
    ml_trained          : bool                  = False
    success             : bool                  = False
    error_message       : str                   = ''
    run_timestamp       : str                   = ''


# --------------------------------------------------------------------------- #
# INTERNAL : Determine which mode to run                                      #
# --------------------------------------------------------------------------- #

def _determine_mode(df: pd.DataFrame, force_retrain: bool) -> str:
    """
    Decide which pipeline mode to use based on data size and model state.

    FULL  → model doesn't exist yet OR force_retrain=True AND enough data
    FAST  → model already trained AND enough data to predict on
    NO_ML → not enough messages to train a meaningful model
    """
    n_messages = len(df[~df['is_media'] & ~df['is_deleted']])

    if n_messages < MIN_MESSAGES_FOR_ML:
        logger.info(f"  Mode: NO_ML — only {n_messages} text messages "
                    f"(need {MIN_MESSAGES_FOR_ML} for ML)")
        return 'NO_ML'

    if force_retrain or not is_model_trained():
        logger.info("  Mode: FULL — training ML model from scratch")
        return 'FULL'

    logger.info("  Mode: FAST — loading existing trained model")
    return 'FAST'


# --------------------------------------------------------------------------- #
# INTERNAL : Step runners (each returns updated state or raises on failure)   #
# --------------------------------------------------------------------------- #

def _step_parse(file_path: str) -> pd.DataFrame:
    """Step 1 — Parse the WhatsApp .txt export into a DataFrame."""
    logger.info("─" * 50)
    logger.info("STEP 1/7 — Parsing chat file...")
    df = parse_chat(file_path)
    logger.info(f"  ✓ {len(df)} messages parsed from {df['sender'].nunique()} senders")
    return df


def _step_process(df: pd.DataFrame) -> pd.DataFrame:
    """Step 2 — Clean and enrich the DataFrame."""
    logger.info("─" * 50)
    logger.info("STEP 2/7 — Cleaning and processing...")
    df = process_dataframe(df)
    cleaned = (df['cleaned_message'] != '').sum()
    logger.info(f"  ✓ {cleaned} messages cleaned | "
                f"{int(df['emoji_count'].sum())} emojis found | "
                f"{int(df['has_url'].sum())} URLs found")
    return df


def _step_nlp(df: pd.DataFrame) -> NLPResult:
    """Step 3 — Run NLP feature extraction."""
    logger.info("─" * 50)
    logger.info("STEP 3/7 — NLP feature extraction...")
    result = process_nlp(df)
    logger.info(f"  ✓ {result.total_words:,} words | "
                f"{result.unique_words:,} unique | "
                f"Peak hour: {result.hour_activity.idxmax()}:00 | "
                f"Top word: '{list(result.word_frequencies.most_common(1))[0][0]}'")
    return result


def _step_sentiment(df: pd.DataFrame) -> pd.DataFrame:
    """Step 4 — Run VADER + TextBlob sentiment analysis."""
    logger.info("─" * 50)
    logger.info("STEP 4/7 — Sentiment analysis (VADER + TextBlob)...")
    df = analyze_sentiment(df)
    counts     = df['sentiment_label'].value_counts()
    total      = len(df)
    pos_pct    = counts.get('Positive', 0) / total * 100
    neg_pct    = counts.get('Negative', 0) / total * 100
    neu_pct    = counts.get('Neutral',  0) / total * 100
    avg_score  = df['sentiment_score'].mean()
    logger.info(f"  ✓ Positive: {pos_pct:.1f}% | "
                f"Neutral: {neu_pct:.1f}% | "
                f"Negative: {neg_pct:.1f}% | "
                f"Avg score: {avg_score:+.3f}")
    return df


def _step_train_ml(df: pd.DataFrame) -> bool:
    """
    Step 5a — Train ML classifier (FULL mode only).
    Returns True if training succeeded, False if it failed gracefully.
    """
    logger.info("─" * 50)
    logger.info("STEP 5/7 — Training ML classifier...")

    try:
        training_df = get_high_confidence_samples(df, n=ML_TRAINING_SAMPLES)

        if len(training_df) < 30:
            logger.warning(f"  Only {len(training_df)} high-confidence samples — "
                           f"skipping ML training.")
            return False

        result = train_classifier(training_df)
        logger.info(f"  ✓ Best model : {result.best_model_name}")
        logger.info(f"    Accuracy   : {result.accuracy:.4f} "
                    f"({result.accuracy * 100:.1f}%)")
        logger.info(f"    F1 Score   : {result.f1_score:.4f}")
        logger.info(f"    CV Mean    : {result.cross_val_mean:.4f} "
                    f"(+/- {result.cross_val_std:.4f})")
        return True

    except Exception as e:
        logger.warning(f"  ML training failed (non-fatal): {e}")
        logger.warning("  Continuing pipeline without ML predictions.")
        return False


def _step_predict(df: pd.DataFrame) -> pd.DataFrame:
    """
    Step 5b — Run bulk ML predictions on entire DataFrame.
    Called in both FULL and FAST modes (after model exists).
    """
    logger.info("─" * 50)
    logger.info("STEP 5/7 — Running ML predictions on full chat...")

    try:
        df = predict_bulk(df)
        agree = (
            (df['sentiment_label'] == df['predicted_sentiment'])
            & ~df['is_media'] & ~df['is_deleted']
        ).mean() * 100
        logger.info(f"  ✓ ML predictions added | "
                    f"VADER↔ML agreement: {agree:.1f}%")
    except Exception as e:
        logger.warning(f"  Bulk prediction failed (non-fatal): {e}")
        df['predicted_sentiment'] = df.get('sentiment_label', 'Neutral')

    return df


def _step_charts(df: pd.DataFrame, nlp_result: NLPResult) -> dict:
    """Step 6 — Generate all 10 charts."""
    logger.info("─" * 50)
    logger.info("STEP 6/7 — Generating charts...")
    chart_paths = generate_all_charts(df, nlp_result)
    success_count = sum(1 for p in chart_paths.values() if p)
    logger.info(f"  ✓ {success_count}/10 charts saved to results/visualizations/")
    return chart_paths


def _step_summaries(df: pd.DataFrame, nlp_result: NLPResult) -> dict:
    """Step 7 — Build all summary dicts for the dashboard."""
    logger.info("─" * 50)
    logger.info("STEP 7/7 — Building dashboard summaries...")

    summaries = {
        'chat_summary'        : get_chat_summary(df),
        'overall_mood'        : get_overall_mood(df),
        'nlp_summary'         : get_nlp_summary(nlp_result),
        'processing_stats'    : get_processing_stats(df),
        'model_metadata'      : load_model_metadata(),
        'sentiment_by_sender' : get_sentiment_by_sender(df),
        'top_emojis'          : get_top_emojis(df, top_n=10),
    }

    logger.info(f"  ✓ Chat mood   : {summaries['overall_mood'].get('mood_description', 'N/A')}")
    logger.info(f"  ✓ Top word    : {summaries['nlp_summary'].get('top_word', 'N/A')}")
    logger.info(f"  ✓ Peak hour   : {summaries['nlp_summary'].get('peak_hour', 'N/A')}")

    return summaries


# --------------------------------------------------------------------------- #
# MAIN PUBLIC FUNCTION : run_pipeline()                                       #
# --------------------------------------------------------------------------- #

def run_pipeline(
    file_path     : str,
    force_retrain : bool = False,
) -> PipelineOutput:
    """
    Master pipeline function — single entry point for the entire system.

    Call this from app.py (on file upload) and main.py (CLI mode).

    Parameters:
        file_path     : Path to the exported WhatsApp .txt file
        force_retrain : If True, retrain ML model even if one already exists.
                        Default False — reuses saved model for speed.

    Returns:
        PipelineOutput dataclass — contains everything for the dashboard.
        Always returns an object — check output.success before using data.

    Fail-safe design:
        - Parser failure     → output.success = False, stops immediately
        - NLP failure        → output.success = False, stops immediately
        - ML failure         → continues without ML (charts still generated)
        - Single chart fail  → other 9 charts still saved
    """

    output    = PipelineOutput()
    start_time = time.time()
    output.run_timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    logger.info("=" * 50)
    logger.info("WHATSAPP NLP ANALYSIS PIPELINE STARTED")
    logger.info(f"File      : {file_path}")
    logger.info(f"Timestamp : {output.run_timestamp}")
    logger.info("=" * 50)

    try:
        # ── STEP 1 : Parse ─────────────────────────────────────────────── #
        # CRITICAL — if this fails, nothing else can run
        df = _step_parse(file_path)

        # ── STEP 2 : Clean & Process ───────────────────────────────────── #
        # CRITICAL — cleaned_message needed by all downstream modules
        df = _step_process(df)

        # ── Determine mode ─────────────────────────────────────────────── #
        mode = _determine_mode(df, force_retrain)
        output.mode_used = mode

        # ── STEP 3 : NLP Feature Extraction ───────────────────────────── #
        # CRITICAL — NLPResult needed for all charts
        nlp_result = _step_nlp(df)

        # ── STEP 4 : Sentiment Analysis ────────────────────────────────── #
        # CRITICAL — sentiment_label needed for sentiment charts + ML
        df = _step_sentiment(df)

        # ── STEP 5 : ML Training / Prediction ─────────────────────────── #
        # NON-CRITICAL — pipeline continues even if ML fails

        if mode == 'FULL':
            trained = _step_train_ml(df)
            output.ml_trained = trained
            if trained:
                df = _step_predict(df)

        elif mode == 'FAST':
            output.ml_trained = False   # not trained THIS run
            df = _step_predict(df)

        else:  # NO_ML
            output.ml_trained = False
            df['predicted_sentiment'] = df.get('sentiment_label', 'Neutral')
            logger.info("STEP 5/7 — ML skipped (not enough data)")

        # ── STEP 6 : Charts ───────────────────────────────────────────── #
        # NON-CRITICAL — individual chart failures don't stop others
        chart_paths = _step_charts(df, nlp_result)

        # ── STEP 7 : Summaries ────────────────────────────────────────── #
        summaries = _step_summaries(df, nlp_result)

        # ── Populate output ───────────────────────────────────────────── #
        output.df                  = df
        output.nlp_result          = nlp_result
        output.chart_paths         = chart_paths
        output.overall_mood        = summaries['overall_mood']
        output.chat_summary        = summaries['chat_summary']
        output.nlp_summary         = summaries['nlp_summary']
        output.processing_stats    = summaries['processing_stats']
        output.model_metadata      = summaries['model_metadata']
        output.sentiment_by_sender = summaries['sentiment_by_sender']
        output.top_emojis          = summaries['top_emojis']
        output.success             = True

    except FileNotFoundError as e:
        output.success       = False
        output.error_message = f"File not found: {e}"
        logger.error(f"PIPELINE FAILED — {output.error_message}")

    except ValueError as e:
        output.success       = False
        output.error_message = f"Data error: {e}"
        logger.error(f"PIPELINE FAILED — {output.error_message}")

    except Exception as e:
        output.success       = False
        output.error_message = f"Unexpected error: {e}"
        logger.error(f"PIPELINE FAILED — {output.error_message}")
        logger.error(traceback.format_exc())

    finally:
        output.processing_time = round(time.time() - start_time, 2)

        logger.info("=" * 50)
        if output.success:
            logger.info("PIPELINE COMPLETED SUCCESSFULLY")
        else:
            logger.info("PIPELINE FAILED")
            logger.info(f"Error: {output.error_message}")
        logger.info(f"Mode           : {output.mode_used}")
        logger.info(f"ML Trained     : {output.ml_trained}")
        logger.info(f"Processing Time: {output.processing_time}s")
        logger.info("=" * 50)

    return output


# --------------------------------------------------------------------------- #
# PUBLIC UTILITY : get_pipeline_status()                                      #
# --------------------------------------------------------------------------- #

def get_pipeline_status() -> dict:
    """
    Returns a status dict showing what has been set up.
    Used by app.py to show a status panel before the first run.

    Returns:
    {
        'model_trained'     : True/False,
        'charts_available'  : 10,
        'model_metadata'    : { accuracy, f1_score, trained_at },
        'output_dir_exists' : True/False,
    }
    """
    chart_paths   = get_chart_paths()
    charts_ready  = sum(1 for p in chart_paths.values() if p)
    model_meta    = load_model_metadata()

    return {
        'model_trained'    : is_model_trained(),
        'charts_available' : charts_ready,
        'model_metadata'   : model_meta,
        'output_dir_exists': os.path.exists(os.path.join('results', 'visualizations')),
    }


# --------------------------------------------------------------------------- #
# QUICK TEST — python src/pipeline.py <chat.txt>                              #
# --------------------------------------------------------------------------- #

if __name__ == '__main__':
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    path = sys.argv[1] if len(sys.argv) > 1 else 'data/sample_chats/chat.txt'

    # Optional: pass --retrain to force retrain even if model exists
    force_retrain = '--retrain' in sys.argv

    print(f"\nRunning pipeline on: {path}")
    print(f"Force retrain      : {force_retrain}\n")

    output = run_pipeline(path, force_retrain=force_retrain)

    if not output.success:
        print(f"\n❌ Pipeline failed: {output.error_message}")
        sys.exit(1)

    print("\n" + "="*60)
    print("PIPELINE RESULTS SUMMARY")
    print("="*60)

    print(f"\n  Status           : ✓ SUCCESS")
    print(f"  Mode             : {output.mode_used}")
    print(f"  ML Trained       : {output.ml_trained}")
    print(f"  Processing Time  : {output.processing_time}s")
    print(f"  Run Timestamp    : {output.run_timestamp}")

    print("\n── Chat Summary ──────────────────────────────────────")
    for k, v in output.chat_summary.items():
        print(f"  {k:<25}: {v}")

    print("\n── Overall Mood ──────────────────────────────────────")
    for k, v in output.overall_mood.items():
        print(f"  {k:<25}: {v}")

    print("\n── NLP Summary ───────────────────────────────────────")
    for k, v in output.nlp_summary.items():
        print(f"  {k:<25}: {v}")

    if output.model_metadata:
        print("\n── ML Model ──────────────────────────────────────────")
        for k, v in output.model_metadata.items():
            print(f"  {k:<25}: {v}")

    print("\n── Charts Generated ──────────────────────────────────")
    for key, path in output.chart_paths.items():
        status = "✓" if path else "✗"
        print(f"  {status}  {key:<30}: {os.path.basename(path) if path else 'FAILED'}")

    print("\n── Sentiment by Sender ───────────────────────────────")
    if not output.sentiment_by_sender.empty:
        print(output.sentiment_by_sender[
            ['sender', 'total', 'positive_pct', 'neutral_pct',
             'negative_pct', 'avg_score']
        ].to_string(index=False))

    print("\n── Top Emojis ────────────────────────────────────────")
    if not output.top_emojis.empty:
        print(output.top_emojis.head(10).to_string())

    print(f"\n{'='*60}")
    print(f"All charts saved to: results/visualizations/")
    print(f"{'='*60}\n")