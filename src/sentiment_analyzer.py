# =============================================================================
# sentiment_analyzer.py
# -----------------------------------------------------------------------------
# PURPOSE : Score every message in the DataFrame with sentiment using two
#           complementary approaches:
#               1. VADER  — rule-based, designed for social/chat text
#               2. TextBlob — statistical, gives subjectivity score
#
#           VADER runs on the ORIGINAL message (preserves emojis + caps).
#           TextBlob runs on the CLEANED message (better for subjectivity).
#
# INPUT   : DataFrame from data_processor.process_dataframe()
#           Must have columns: message, cleaned_message, sender,
#                              date, is_media, is_deleted
#
# OUTPUT  : Same DataFrame + 4 new columns:
#               - sentiment_label      : 'Positive' / 'Negative' / 'Neutral'
#               - sentiment_score      : VADER compound score (-1.0 to +1.0)
#               - subjectivity         : TextBlob score (0.0 = fact, 1.0 = opinion)
#               - sentiment_confidence : 'High' / 'Medium' / 'Low'
#
#           + utility functions for visualizer.py and dashboard
#
# USAGE   : from src.sentiment_analyzer import analyze_sentiment
#           df = analyze_sentiment(df)
#
# AUTHOR  : Team — Chaitanya / Anudeep / Ram Teja
# =============================================================================

import logging
import pandas as pd
import numpy as np
import nltk

# VADER — rule-based sentiment analyser built for social media text
nltk.download('vader_lexicon', quiet=True)
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# TextBlob — statistical NLP library for polarity + subjectivity
from textblob import TextBlob

# --------------------------------------------------------------------------- #
# Logging                                                                      #
# --------------------------------------------------------------------------- #
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# CONSTANTS                                                                    #
# --------------------------------------------------------------------------- #

# VADER compound score thresholds (industry standard values)
# compound >= +0.05  →  Positive
# compound <= -0.05  →  Negative
# in between         →  Neutral
POSITIVE_THRESHOLD =  0.05
NEGATIVE_THRESHOLD = -0.05

# Confidence thresholds
# When VADER compound score is strong (far from 0), confidence is High
# When it's borderline, confidence is Low
HIGH_CONFIDENCE_THRESHOLD   = 0.5    # |score| >= 0.5  → High
MEDIUM_CONFIDENCE_THRESHOLD = 0.2    # |score| >= 0.2  → Medium
                                     # |score| <  0.2  → Low

# Initialise VADER once — expensive to create, so we reuse one instance
vader = SentimentIntensityAnalyzer()


# --------------------------------------------------------------------------- #
# STEP 1 : Score a single message with VADER                                  #
# --------------------------------------------------------------------------- #

def _vader_score(message: str) -> dict:
    """
    Run VADER on a single message and return all 4 scores.

    VADER returns:
        {
          'neg': 0.0,       # proportion of negative words
          'neu': 0.254,     # proportion of neutral words
          'pos': 0.746,     # proportion of positive words
          'compound': 0.82  # normalised overall score (-1 to +1)
        }

    We primarily use 'compound' for labeling.
    We keep all 4 in case visualizer.py wants detailed breakdowns.

    WHY ORIGINAL MESSAGE (not cleaned)?
        VADER understands:
          - "GREAT!!!"  is stronger than "great"
          - "😂🔥"      signals positivity
          - "NOT good"  is negative even though "good" alone is positive
        Cleaning removes all of this context.
    """
    if not message or not isinstance(message, str) or not message.strip():
        return {'neg': 0.0, 'neu': 1.0, 'pos': 0.0, 'compound': 0.0}

    return vader.polarity_scores(message)


# --------------------------------------------------------------------------- #
# STEP 2 : Score a single message with TextBlob                               #
# --------------------------------------------------------------------------- #

def _textblob_score(cleaned_message: str) -> tuple:
    """
    Run TextBlob on cleaned text and return (polarity, subjectivity).

    TextBlob polarity    : -1.0 (very negative) → +1.0 (very positive)
    TextBlob subjectivity:  0.0 (objective/factual) → 1.0 (subjective/opinion)

    WHY CLEANED MESSAGE (not original)?
        TextBlob is a statistical model trained on formal text.
        Emojis and caps confuse it. Cleaned text gives better subjectivity scores.

    WHY SUBJECTIVITY MATTERS?
        "The meeting is at 5pm"     → subjectivity ~0.0  (just a fact)
        "I absolutely love this!!"  → subjectivity ~0.9  (strong opinion)
        High subjectivity + positive = genuine emotional expression.
        Low subjectivity = informational, not really sentiment-bearing.
    """
    if not cleaned_message or not isinstance(cleaned_message, str) or not cleaned_message.strip():
        return 0.0, 0.0

    try:
        blob = TextBlob(cleaned_message)
        return round(blob.sentiment.polarity, 4), round(blob.sentiment.subjectivity, 4)
    except Exception:
        return 0.0, 0.0


# --------------------------------------------------------------------------- #
# STEP 3 : Assign sentiment label from VADER compound score                   #
# --------------------------------------------------------------------------- #

def _assign_label(compound_score: float) -> str:
    """
    Convert a VADER compound score to a human-readable label.

        compound >= +0.05  →  'Positive'
        compound <= -0.05  →  'Negative'
        in between         →  'Neutral'

    These thresholds are the standard recommended by the VADER authors.
    """
    if compound_score >= POSITIVE_THRESHOLD:
        return 'Positive'
    elif compound_score <= NEGATIVE_THRESHOLD:
        return 'Negative'
    else:
        return 'Neutral'


# --------------------------------------------------------------------------- #
# STEP 4 : Assign confidence level                                            #
# --------------------------------------------------------------------------- #

def _assign_confidence(compound_score: float, vader_scores: dict, textblob_polarity: float) -> str:
    """
    Determine how confident we are in the sentiment label.

    Confidence is HIGH when:
        - VADER compound score is strong (|score| >= 0.5)
        - VADER and TextBlob AGREE on direction (both positive or both negative)

    Confidence is MEDIUM when:
        - |score| is between 0.2 and 0.5
        - OR VADER and TextBlob slightly disagree

    Confidence is LOW when:
        - |score| < 0.2 (borderline neutral/sentiment)
        - OR VADER and TextBlob strongly disagree

    WHY TRACK CONFIDENCE?
        Low-confidence predictions can be flagged on the results page.
        The ML classifier (sentiment_classifier.py) can focus on
        high-confidence examples for training — cleaner labels = better model.
    """
    abs_score = abs(compound_score)

    # Check if VADER and TextBlob agree on direction
    vader_direction    = 1 if compound_score > 0 else (-1 if compound_score < 0 else 0)
    textblob_direction = 1 if textblob_polarity > 0.05 else (-1 if textblob_polarity < -0.05 else 0)
    models_agree = (vader_direction == textblob_direction)

    if abs_score >= HIGH_CONFIDENCE_THRESHOLD and models_agree:
        return 'High'
    elif abs_score >= MEDIUM_CONFIDENCE_THRESHOLD:
        return 'Medium'
    else:
        return 'Low'


# --------------------------------------------------------------------------- #
# CORE FUNCTION : Analyse a single row                                        #
# --------------------------------------------------------------------------- #

def _analyze_row(row: pd.Series) -> pd.Series:
    """
    Apply sentiment analysis to a single DataFrame row.
    Returns a Series with 4 new values to be added as columns.

    Called via df.apply(_analyze_row, axis=1)

    Skips media and deleted messages — returns neutral defaults for those.
    """
    # Skip media and deleted messages — they carry no sentiment
    if row.get('is_media', False) or row.get('is_deleted', False):
        return pd.Series({
            'sentiment_label'      : 'Neutral',
            'sentiment_score'      : 0.0,
            'subjectivity'         : 0.0,
            'sentiment_confidence' : 'Low',
            'vader_pos'            : 0.0,
            'vader_neg'            : 0.0,
            'vader_neu'            : 1.0,
        })

    # VADER on original message
    vader_scores   = _vader_score(row.get('message', ''))
    compound_score = vader_scores['compound']

    # TextBlob on cleaned message
    tb_polarity, tb_subjectivity = _textblob_score(row.get('cleaned_message', ''))

    # Label and confidence
    label      = _assign_label(compound_score)
    confidence = _assign_confidence(compound_score, vader_scores, tb_polarity)

    return pd.Series({
        'sentiment_label'      : label,
        'sentiment_score'      : round(compound_score, 4),
        'subjectivity'         : round(tb_subjectivity, 4),
        'sentiment_confidence' : confidence,
        'vader_pos'            : round(vader_scores['pos'], 4),
        'vader_neg'            : round(vader_scores['neg'], 4),
        'vader_neu'            : round(vader_scores['neu'], 4),
    })


# --------------------------------------------------------------------------- #
# MAIN PUBLIC FUNCTION : analyze_sentiment()                                  #
# --------------------------------------------------------------------------- #

def analyze_sentiment(df: pd.DataFrame) -> pd.DataFrame:
    """
    Master function — call this from pipeline.py.

    Applies VADER + TextBlob sentiment analysis to every message in the
    DataFrame and adds 7 new columns:
        - sentiment_label       : 'Positive' / 'Negative' / 'Neutral'
        - sentiment_score       : VADER compound score (-1.0 to +1.0)
        - subjectivity          : TextBlob subjectivity (0.0 to 1.0)
        - sentiment_confidence  : 'High' / 'Medium' / 'Low'
        - vader_pos             : VADER positive component
        - vader_neg             : VADER negative component
        - vader_neu             : VADER neutral component

    Parameters:
        df (pd.DataFrame): Output of process_dataframe() from data_processor.py
                           Must have: message, cleaned_message,
                                      sender, date, is_media, is_deleted

    Returns:
        pd.DataFrame: Same df with 7 new columns added
    """

    # ── Validate input columns ────────────────────────────────────────────── #
    required = ['message', 'cleaned_message', 'sender', 'date', 'is_media', 'is_deleted']
    missing  = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"DataFrame missing required columns: {missing}\n"
            f"Run whatsapp_parser → data_processor first."
        )

    logger.info("Starting sentiment analysis...")
    logger.info(f"Input: {len(df)} messages")

    # Work on a copy
    df = df.copy()

    # ── Apply sentiment analysis to every row ─────────────────────────────── #
    logger.info("Running VADER + TextBlob on all messages...")
    sentiment_cols = df.apply(_analyze_row, axis=1)
    df = pd.concat([df, sentiment_cols], axis=1)

    # ── Log summary ───────────────────────────────────────────────────────── #
    label_counts = df['sentiment_label'].value_counts()
    total        = len(df)

    logger.info("Sentiment analysis complete!")
    logger.info(f"  Positive  : {label_counts.get('Positive', 0):>5} "
                f"({label_counts.get('Positive', 0)/total*100:.1f}%)")
    logger.info(f"  Neutral   : {label_counts.get('Neutral',  0):>5} "
                f"({label_counts.get('Neutral',  0)/total*100:.1f}%)")
    logger.info(f"  Negative  : {label_counts.get('Negative', 0):>5} "
                f"({label_counts.get('Negative', 0)/total*100:.1f}%)")
    logger.info(f"  Avg score : {df['sentiment_score'].mean():.4f}")

    return df


# --------------------------------------------------------------------------- #
# UTILITY 1 : get_sentiment_by_sender()                                       #
# --------------------------------------------------------------------------- #

def get_sentiment_by_sender(df: pd.DataFrame) -> pd.DataFrame:
    """
    Returns a per-sender sentiment breakdown DataFrame.

    Columns: sender, Positive, Negative, Neutral,
             positive_pct, negative_pct, neutral_pct, avg_score

    Used by:
        - Stacked bar chart in visualizer.py
        - Per-user sentiment cards on results.html
    """
    # Only analyse text messages
    text_df = df[~df['is_media'] & ~df['is_deleted']].copy()

    rows = []
    for sender, group in text_df.groupby('sender'):
        counts    = group['sentiment_label'].value_counts()
        total     = len(group)
        pos       = counts.get('Positive', 0)
        neg       = counts.get('Negative', 0)
        neu       = counts.get('Neutral',  0)
        avg_score = round(group['sentiment_score'].mean(), 4)

        rows.append({
            'sender'        : sender,
            'Positive'      : pos,
            'Negative'      : neg,
            'Neutral'       : neu,
            'total'         : total,
            'positive_pct'  : round(pos / total * 100, 1) if total > 0 else 0,
            'negative_pct'  : round(neg / total * 100, 1) if total > 0 else 0,
            'neutral_pct'   : round(neu / total * 100, 1) if total > 0 else 0,
            'avg_score'     : avg_score,
        })

    result = pd.DataFrame(rows).sort_values('avg_score', ascending=False)
    return result


# --------------------------------------------------------------------------- #
# UTILITY 2 : get_sentiment_over_time()                                       #
# --------------------------------------------------------------------------- #

def get_sentiment_over_time(df: pd.DataFrame, freq: str = 'W') -> pd.DataFrame:
    """
    Returns average sentiment score grouped by time period.

    freq options:
        'D'  → daily averages
        'W'  → weekly averages (default — smooths out noise)
        'M'  → monthly averages

    Returns DataFrame with columns: date, avg_score, message_count

    Used by:
        - Sentiment trend line chart in visualizer.py
        "How did the chat mood change over time?"
    """
    text_df = df[~df['is_media'] & ~df['is_deleted']].copy()
    text_df['date'] = pd.to_datetime(text_df['date'])

    grouped = text_df.resample(freq, on='date').agg(
        avg_score     = ('sentiment_score', 'mean'),
        message_count = ('sentiment_score', 'count')
    ).reset_index()

    grouped['avg_score'] = grouped['avg_score'].round(4)
    grouped = grouped[grouped['message_count'] > 0]  # drop empty periods

    return grouped


# --------------------------------------------------------------------------- #
# UTILITY 3 : get_overall_mood()                                              #
# --------------------------------------------------------------------------- #

def get_overall_mood(df: pd.DataFrame) -> dict:
    """
    Returns a single summary dict of the chat's overall emotional tone.
    Used for the dashboard hero stats section.

    Example output:
    {
        'overall_label'     : 'Positive',
        'overall_score'     : 0.312,
        'positive_pct'      : 58.4,
        'negative_pct'      : 14.2,
        'neutral_pct'       : 27.4,
        'most_positive_user': 'Chaitanya',
        'most_negative_user': 'Anudeep',
        'happiest_day'      : 'Sunday',
        'mood_description'  : 'Generally Positive'
    }
    """
    text_df = df[~df['is_media'] & ~df['is_deleted']]

    if text_df.empty:
        return {'overall_label': 'Neutral', 'overall_score': 0.0}

    label_counts = text_df['sentiment_label'].value_counts()
    total        = len(text_df)
    avg_score    = round(float(text_df['sentiment_score'].mean()), 4)
    pos_pct      = round(label_counts.get('Positive', 0) / total * 100, 1)
    neg_pct      = round(label_counts.get('Negative', 0) / total * 100, 1)
    neu_pct      = round(label_counts.get('Neutral',  0) / total * 100, 1)

    # Overall label based on avg score
    overall_label = _assign_label(avg_score)

    # Most positive / most negative sender
    sender_avg = text_df.groupby('sender')['sentiment_score'].mean()
    most_positive_user = sender_avg.idxmax() if not sender_avg.empty else 'N/A'
    most_negative_user = sender_avg.idxmin() if not sender_avg.empty else 'N/A'

    # Happiest day of week
    day_avg = text_df.groupby('day_name')['sentiment_score'].mean()
    happiest_day = day_avg.idxmax() if not day_avg.empty else 'N/A'

    # Human-readable mood description
    if avg_score >= 0.3:
        mood_description = 'Very Positive 😊'
    elif avg_score >= 0.05:
        mood_description = 'Generally Positive 🙂'
    elif avg_score <= -0.3:
        mood_description = 'Very Negative 😔'
    elif avg_score <= -0.05:
        mood_description = 'Generally Negative 😐'
    else:
        mood_description = 'Mostly Neutral 😶'

    return {
        'overall_label'      : overall_label,
        'overall_score'      : avg_score,
        'positive_pct'       : pos_pct,
        'negative_pct'       : neg_pct,
        'neutral_pct'        : neu_pct,
        'most_positive_user' : most_positive_user,
        'most_negative_user' : most_negative_user,
        'happiest_day'       : happiest_day,
        'mood_description'   : mood_description,
        'total_analysed'     : total,
        'high_confidence'    : int((text_df['sentiment_confidence'] == 'High').sum()),
    }


# --------------------------------------------------------------------------- #
# UTILITY 4 : get_high_confidence_samples()                                   #
# --------------------------------------------------------------------------- #

def get_high_confidence_samples(df: pd.DataFrame, n: int = 100) -> pd.DataFrame:
    """
    Returns n high-confidence labeled messages per sentiment class.
    Used by sentiment_classifier.py as clean training data.

    WHY THIS MATTERS FOR THE ML MODEL:
        If we train the classifier on ALL messages including uncertain ones,
        the model learns noisy patterns.
        Training only on High-confidence VADER labels = cleaner training data
        = better ML model accuracy.

    Returns DataFrame with columns: cleaned_message, sentiment_label
    """
    text_df = df[
        (~df['is_media']) &
        (~df['is_deleted']) &
        (df['cleaned_message'].str.strip() != '') &
        (df['sentiment_confidence'] == 'High')
    ][['cleaned_message', 'sentiment_label']].copy()

    # Sample equally from each class to avoid class imbalance
    # (prevents model from just predicting "Positive" for everything)
    classes = text_df['sentiment_label'].unique()
    min_count = min(text_df['sentiment_label'].value_counts().min(), n)

    balanced_samples = []
    for label in classes:
        class_df = text_df[text_df['sentiment_label'] == label]
        sample   = class_df.sample(min(len(class_df), min_count), random_state=42)
        balanced_samples.append(sample)

    result = pd.concat(balanced_samples).sample(frac=1, random_state=42).reset_index(drop=True)
    logger.info(f"High-confidence samples: {len(result)} "
                f"({len(classes)} classes, ~{min_count} each)")
    return result


# --------------------------------------------------------------------------- #
# QUICK TEST — python src/sentiment_analyzer.py <chat.txt>                    #
# --------------------------------------------------------------------------- #

if __name__ == '__main__':
    import sys, os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    from src.whatsapp_parser import parse_chat
    from src.data_processor  import process_dataframe

    path = sys.argv[1] if len(sys.argv) > 1 else 'data/sample_chats/chat.txt'

    try:
        print("\nStep 1: Parsing...")
        df = parse_chat(path)

        print("Step 2: Processing...")
        df = process_dataframe(df)

        print("Step 3: Sentiment analysis...")
        df = analyze_sentiment(df)

        print("\n" + "="*60)
        print("SAMPLE — Message + Sentiment (first 8 text messages):")
        print("="*60)
        sample = df[
            ~df['is_media'] & ~df['is_deleted']
        ][['message', 'sentiment_label', 'sentiment_score',
           'subjectivity', 'sentiment_confidence']].head(8)
        for _, row in sample.iterrows():
            print(f"\n  Message    : {row['message'][:70]}")
            print(f"  Label      : {row['sentiment_label']:<10} "
                  f"Score: {row['sentiment_score']:>7.4f}  "
                  f"Subjectivity: {row['subjectivity']:.2f}  "
                  f"Confidence: {row['sentiment_confidence']}")

        print("\n" + "="*60)
        print("SENTIMENT BY SENDER:")
        print("="*60)
        print(get_sentiment_by_sender(df).to_string(index=False))

        print("\n" + "="*60)
        print("OVERALL MOOD:")
        print("="*60)
        for k, v in get_overall_mood(df).items():
            print(f"  {k:<25}: {v}")

        print("\n" + "="*60)
        print("SENTIMENT OVER TIME (weekly):")
        print("="*60)
        print(get_sentiment_over_time(df, freq='W').head(10).to_string(index=False))

        print("\n" + "="*60)
        print("HIGH-CONFIDENCE TRAINING SAMPLES (top 5):")
        print("="*60)
        samples = get_high_confidence_samples(df, n=50)
        print(samples.head(5).to_string(index=False))
        print(f"\nTotal high-confidence samples: {len(samples)}")

    except (FileNotFoundError, ValueError) as e:
        print(f"\nERROR: {e}")
        print("Usage: python src/sentiment_analyzer.py <path_to_chat.txt>")