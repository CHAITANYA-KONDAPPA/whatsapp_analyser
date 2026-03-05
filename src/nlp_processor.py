# =============================================================================
# nlp_processor.py
# -----------------------------------------------------------------------------
# PURPOSE : Extract deep NLP features from the cleaned DataFrame produced by
#           data_processor.py. This module does NOT clean text — it analyses
#           it and produces statistical and linguistic insights.
#
# INPUT   : Processed DataFrame from process_dataframe() — must already have
#           the 'cleaned_message' column present.
#
# OUTPUT  : A rich NLPResult object (and helper DataFrames) containing:
#               - word_frequencies    : overall word counts
#               - bigrams             : common 2-word phrases
#               - trigrams            : common 3-word phrases
#               - tfidf_keywords      : per-sender signature keywords
#               - user_stats          : per-sender statistics DataFrame
#               - hour_activity       : messages per hour (0-23)
#               - day_activity        : messages per day of week
#               - month_activity      : messages per month
#               - response_times      : average reply time per sender
#
# USAGE   : from src.nlp_processor import process_nlp
#           result = process_nlp(df)
#
# AUTHOR  : Team — Chaitanya / Anudeep / Ram Teja
# =============================================================================

import logging
import pandas as pd
import numpy as np
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import timedelta
from typing import Dict, List, Tuple

# Scikit-learn for TF-IDF
from sklearn.feature_extraction.text import TfidfVectorizer

# NLTK for n-grams
import nltk
nltk.download('punkt',    quiet=True)
nltk.download('punkt_tab', quiet=True)
from nltk.util import ngrams

# --------------------------------------------------------------------------- #
# Logging                                                                      #
# --------------------------------------------------------------------------- #
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# RESULT CONTAINER                                                             #
# --------------------------------------------------------------------------- #
# We use a dataclass to bundle all outputs together so pipeline.py and
# visualizer.py can access everything from one clean object:
#     result = process_nlp(df)
#     result.word_frequencies   → Counter of all words
#     result.user_stats         → DataFrame of per-user stats
#     result.hour_activity      → Series of messages per hour

@dataclass
class NLPResult:
    # ── Word-level features ──────────────────────────────────────────────── #
    word_frequencies : Counter        = field(default_factory=Counter)
    bigrams          : Counter        = field(default_factory=Counter)
    trigrams         : Counter        = field(default_factory=Counter)

    # ── TF-IDF keywords per sender ───────────────────────────────────────── #
    # { 'Chaitanya': [('pipeline', 0.82), ('model', 0.76), ...], ... }
    tfidf_keywords   : Dict[str, List[Tuple[str, float]]] = field(default_factory=dict)

    # ── Per-user statistics ──────────────────────────────────────────────── #
    user_stats       : pd.DataFrame   = field(default_factory=pd.DataFrame)

    # ── Temporal activity patterns ───────────────────────────────────────── #
    hour_activity    : pd.Series      = field(default_factory=pd.Series)
    day_activity     : pd.Series      = field(default_factory=pd.Series)
    month_activity   : pd.Series      = field(default_factory=pd.Series)
    date_activity    : pd.Series      = field(default_factory=pd.Series)

    # ── Response time analysis ───────────────────────────────────────────── #
    # { 'Chaitanya': avg_seconds, 'Anudeep': avg_seconds, ... }
    response_times   : Dict[str, float] = field(default_factory=dict)

    # ── Corpus-level stats ───────────────────────────────────────────────── #
    total_words      : int  = 0
    unique_words     : int  = 0
    vocabulary       : List = field(default_factory=list)


# --------------------------------------------------------------------------- #
# FEATURE 1 : Word Frequency                                                   #
# --------------------------------------------------------------------------- #

def _get_word_frequencies(df: pd.DataFrame) -> Counter:
    """
    Count how many times every word appears across ALL cleaned messages.

    Why Counter?
        Counter is like a dictionary but automatically handles counting.
        Counter({'meet': 47, 'project': 38, ...})
        counter.most_common(20) → top 20 words instantly.

    Used by:
        - WordCloud generator (visualizer.py)
        - Top words bar chart
        - NLP stats on dashboard
    """
    all_words = []

    for message in df['cleaned_message']:
        if isinstance(message, str) and message.strip():
            all_words.extend(message.split())

    freq = Counter(all_words)
    logger.info(f"  Word frequency: {len(freq)} unique words found.")
    return freq


# --------------------------------------------------------------------------- #
# FEATURE 2 : N-grams (Bigrams & Trigrams)                                    #
# --------------------------------------------------------------------------- #

def _get_ngrams(df: pd.DataFrame, n: int) -> Counter:
    """
    Extract the most common n-word phrases from all cleaned messages.

    n=2 → bigrams  : ('good', 'morning'), ('let', 'know')
    n=3 → trigrams : ('let', 'me', 'know'), ('on', 'the', 'way')

    WHY N-GRAMS MATTER:
        The word "not" alone looks neutral.
        "not good" as a bigram signals negativity.
        "not bad" as a bigram signals positivity.
        Single-word analysis misses this completely.

    Used by:
        - sentiment_classifier.py (as additional features)
        - Dashboard "common phrases" section
    """
    all_ngrams = []

    for message in df['cleaned_message']:
        if isinstance(message, str) and message.strip():
            tokens = message.split()
            if len(tokens) >= n:
                all_ngrams.extend(list(ngrams(tokens, n)))

    result = Counter(all_ngrams)
    logger.info(f"  {n}-grams: {len(result)} unique {n}-grams found.")
    return result


# --------------------------------------------------------------------------- #
# FEATURE 3 : TF-IDF Keywords per Sender                                      #
# --------------------------------------------------------------------------- #

def _get_tfidf_keywords(df: pd.DataFrame, top_n: int = 10) -> dict:
    """
    Find the most "signature" keywords for each sender using TF-IDF.

    HOW TF-IDF WORKS (simple explanation):
        - TF  (Term Frequency)         : how often does this word appear for THIS user?
        - IDF (Inverse Doc Frequency)  : how rare is this word across ALL users?
        - TF-IDF score = TF × IDF

        High TF-IDF = word this user uses a LOT but others use rarely.
        This reveals each person's unique communication style.

    Example:
        Chaitanya   → ['pipeline', 'model', 'accuracy', 'training']
        Anudeep     → ['data', 'dataset', 'parse', 'clean']
        Ram Teja    → ['chart', 'dashboard', 'color', 'visual']

    Returns:
        { 'sender_name': [('keyword', tfidf_score), ...], ... }

    Used by:
        - Per-user keyword display on results.html
        - sentiment_classifier.py feature matrix
    """
    # Group all cleaned messages per sender into one big document per sender
    sender_docs = df.groupby('sender')['cleaned_message'].apply(
        lambda msgs: ' '.join([m for m in msgs if isinstance(m, str) and m.strip()])
    ).reset_index()

    # Drop senders who have no text after cleaning
    sender_docs = sender_docs[sender_docs['cleaned_message'].str.strip() != '']

    if len(sender_docs) < 2:
        logger.warning("  TF-IDF needs at least 2 senders. Skipping TF-IDF.")
        return {}

    # Fit TF-IDF — each sender's combined messages = one "document"
    vectorizer = TfidfVectorizer(
        max_features=500,       # consider top 500 words in vocabulary
        min_df=1,               # word must appear in at least 1 sender's messages
        sublinear_tf=True       # apply log normalization — prevents very frequent
                                # words from dominating the score
    )

    try:
        tfidf_matrix = vectorizer.fit_transform(sender_docs['cleaned_message'])
    except ValueError as e:
        logger.warning(f"  TF-IDF failed: {e}")
        return {}

    feature_names = vectorizer.get_feature_names_out()
    keywords = {}

    for i, row in sender_docs.iterrows():
        sender = row['sender']
        # Get TF-IDF scores for this sender's row in the matrix
        scores = tfidf_matrix[sender_docs.index.get_loc(i)].toarray().flatten()
        # Sort by score descending, pick top N
        top_indices = scores.argsort()[::-1][:top_n]
        keywords[sender] = [
            (feature_names[idx], round(float(scores[idx]), 4))
            for idx in top_indices
            if scores[idx] > 0
        ]

    logger.info(f"  TF-IDF keywords extracted for {len(keywords)} senders.")
    return keywords


# --------------------------------------------------------------------------- #
# FEATURE 4 : Per-User Statistics                                             #
# --------------------------------------------------------------------------- #

def _get_user_stats(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build a comprehensive per-sender statistics DataFrame.

    Each row = one sender, columns = their stats.

    WHY THIS IS IMPORTANT:
        The dashboard needs to show cards like:
        "Chaitanya sent 342 messages, used 89 emojis, avg 8 words/msg"
        All of that comes from this function.

    Returns a DataFrame with columns:
        sender, total_messages, text_messages, media_messages,
        deleted_messages, total_words, avg_words, total_emojis,
        avg_emojis, urls_shared, message_share_pct
    """
    stats = []

    total_msgs = len(df)

    for sender, group in df.groupby('sender'):
        text_msgs    = group[~group['is_media'] & ~group['is_deleted']]
        media_msgs   = group[group['is_media']]
        deleted_msgs = group[group['is_deleted']]

        total_words  = int(group['word_count'].sum())
        avg_words    = round(text_msgs['word_count'].mean(), 2) if len(text_msgs) > 0 else 0
        total_emojis = int(group['emoji_count'].sum())
        avg_emojis   = round(group['emoji_count'].mean(), 2)
        urls_shared  = int(group['has_url'].sum())
        msg_share    = round((len(group) / total_msgs) * 100, 2)

        # Most active hour for this sender
        if len(group) > 0:
            most_active_hour = int(group['hour'].value_counts().idxmax())
        else:
            most_active_hour = 0

        stats.append({
            'sender'             : sender,
            'total_messages'     : len(group),
            'text_messages'      : len(text_msgs),
            'media_messages'     : len(media_msgs),
            'deleted_messages'   : len(deleted_msgs),
            'total_words'        : total_words,
            'avg_words_per_msg'  : avg_words,
            'total_emojis'       : total_emojis,
            'avg_emojis_per_msg' : avg_emojis,
            'urls_shared'        : urls_shared,
            'message_share_pct'  : msg_share,
            'most_active_hour'   : most_active_hour,
        })

    result_df = pd.DataFrame(stats).sort_values('total_messages', ascending=False)
    logger.info(f"  User stats computed for {len(result_df)} senders.")
    return result_df


# --------------------------------------------------------------------------- #
# FEATURE 5 : Temporal Activity Patterns                                      #
# --------------------------------------------------------------------------- #

def _get_activity_patterns(df: pd.DataFrame) -> tuple:
    """
    Analyse WHEN messages are sent across different time dimensions.

    Returns four Series:
        hour_activity   : messages per hour (0-23)         → heatmap x-axis
        day_activity    : messages per day name             → weekly bar chart
        month_activity  : messages per month name           → monthly trend
        date_activity   : messages per calendar date        → daily timeline

    WHY THIS MATTERS:
        "Your group is most active at 9PM on Sundays in December"
        This is the kind of insight that makes the dashboard impressive.
    """
    # ── Messages per hour ────────────────────────────────────────────────── #
    hour_activity = df.groupby('hour').size()
    # Fill missing hours with 0 so the chart has all 24 bars
    hour_activity = hour_activity.reindex(range(24), fill_value=0)

    # ── Messages per day of week ─────────────────────────────────────────── #
    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    day_activity = df.groupby('day_name').size()
    day_activity = day_activity.reindex(day_order, fill_value=0)

    # ── Messages per month ───────────────────────────────────────────────── #
    month_order = ['January', 'February', 'March', 'April', 'May', 'June',
                   'July', 'August', 'September', 'October', 'November', 'December']
    month_activity = df.groupby('month_name').size()
    month_activity = month_activity.reindex(month_order, fill_value=0)
    # Drop months with 0 messages (chat didn't exist then)
    month_activity = month_activity[month_activity > 0]

    # ── Messages per calendar date ───────────────────────────────────────── #
    date_activity = df.groupby('date').size()

    logger.info("  Activity patterns computed (hour / day / month / date).")
    return hour_activity, day_activity, month_activity, date_activity


# --------------------------------------------------------------------------- #
# FEATURE 6 : Response Time Analysis                                          #
# --------------------------------------------------------------------------- #

def _get_response_times(df: pd.DataFrame) -> dict:
    """
    Calculate the average response time (in minutes) for each sender.

    HOW IT WORKS:
        For each message, we look at the previous message.
        If the previous sender is DIFFERENT from the current sender,
        the time gap = this sender's response time.

        We cap gaps at 12 hours (43200 seconds) to exclude cases where
        someone replied the next day — that's not really a "response",
        it's a new conversation.

    Returns:
        { 'Chaitanya': 3.4, 'Anudeep': 7.1, 'Ram Teja': 2.8 }
        (values = average response time in minutes)

    Used by:
        - "Fastest responder" stat on dashboard
    """
    response_data = defaultdict(list)

    # Sort by datetime to ensure correct order
    df_sorted = df.sort_values('datetime').reset_index(drop=True)

    for i in range(1, len(df_sorted)):
        current  = df_sorted.iloc[i]
        previous = df_sorted.iloc[i - 1]

        # Only count if different senders (actual response, not self-reply)
        if current['sender'] != previous['sender']:
            gap = (current['datetime'] - previous['datetime']).total_seconds()

            # Cap at 12 hours — anything longer is a new conversation, not a reply
            if 0 < gap <= 43200:
                response_data[current['sender']].append(gap)

    # Average response time in minutes per sender
    avg_response = {}
    for sender, times in response_data.items():
        if times:
            avg_seconds = np.mean(times)
            avg_response[sender] = round(avg_seconds / 60, 2)  # convert to minutes

    logger.info(f"  Response times computed for {len(avg_response)} senders.")
    return avg_response


# --------------------------------------------------------------------------- #
# MAIN PUBLIC FUNCTION : process_nlp()                                        #
# --------------------------------------------------------------------------- #

def process_nlp(df: pd.DataFrame) -> NLPResult:
    """
    Master function — call this from pipeline.py.

    Runs all 6 NLP feature extraction steps and returns one NLPResult object
    containing everything visualizer.py and sentiment modules need.

    Parameters:
        df (pd.DataFrame): Output of process_dataframe() from data_processor.py
                           Must have columns: cleaned_message, sender, hour,
                           day_name, month_name, date, datetime,
                           is_media, is_deleted, emoji_count, has_url, word_count

    Returns:
        NLPResult dataclass with all features populated.
    """

    # Validate input
    required_cols = [
        'cleaned_message', 'sender', 'hour', 'day_name',
        'month_name', 'date', 'datetime', 'is_media',
        'is_deleted', 'emoji_count', 'has_url', 'word_count'
    ]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(
            f"DataFrame is missing required columns: {missing}\n"
            f"Make sure you have run whatsapp_parser.py → data_processor.py first."
        )

    logger.info("Starting NLP processing pipeline...")
    logger.info(f"Input: {len(df)} messages from {df['sender'].nunique()} senders")

    result = NLPResult()

    # ── Feature 1 : Word Frequency ───────────────────────────────────────── #
    logger.info("Step 1/6 — Computing word frequencies...")
    result.word_frequencies = _get_word_frequencies(df)
    result.total_words      = sum(result.word_frequencies.values())
    result.unique_words     = len(result.word_frequencies)
    result.vocabulary       = list(result.word_frequencies.keys())

    # ── Feature 2 : N-grams ──────────────────────────────────────────────── #
    logger.info("Step 2/6 — Extracting bigrams and trigrams...")
    result.bigrams  = _get_ngrams(df, n=2)
    result.trigrams = _get_ngrams(df, n=3)

    # ── Feature 3 : TF-IDF Keywords ──────────────────────────────────────── #
    logger.info("Step 3/6 — Computing TF-IDF keywords per sender...")
    result.tfidf_keywords = _get_tfidf_keywords(df)

    # ── Feature 4 : User Stats ───────────────────────────────────────────── #
    logger.info("Step 4/6 — Computing per-user statistics...")
    result.user_stats = _get_user_stats(df)

    # ── Feature 5 : Activity Patterns ───────────────────────────────────────#
    logger.info("Step 5/6 — Analysing activity patterns...")
    (
        result.hour_activity,
        result.day_activity,
        result.month_activity,
        result.date_activity
    ) = _get_activity_patterns(df)

    # ── Feature 6 : Response Times ───────────────────────────────────────── #
    logger.info("Step 6/6 — Computing response times...")
    result.response_times = _get_response_times(df)

    # ── Summary ──────────────────────────────────────────────────────────── #
    logger.info("NLP processing complete!")
    logger.info(f"  Total words     : {result.total_words:,}")
    logger.info(f"  Unique words    : {result.unique_words:,}")
    logger.info(f"  Bigrams found   : {len(result.bigrams):,}")
    logger.info(f"  Trigrams found  : {len(result.trigrams):,}")
    logger.info(f"  Senders         : {len(result.user_stats)}")
    logger.info(f"  Peak hour       : {result.hour_activity.idxmax()}:00")
    logger.info(f"  Most active day : {result.day_activity.idxmax()}")

    return result


# --------------------------------------------------------------------------- #
# UTILITY FUNCTIONS (used by visualizer.py and dashboard)                     #
# --------------------------------------------------------------------------- #

def get_top_words(result: NLPResult, top_n: int = 20) -> pd.DataFrame:
    """
    Returns top N words as a DataFrame for bar chart / WordCloud.

    Columns: word, count
    """
    top = result.word_frequencies.most_common(top_n)
    return pd.DataFrame(top, columns=['word', 'count'])


def get_top_bigrams(result: NLPResult, top_n: int = 10) -> pd.DataFrame:
    """
    Returns top N bigrams as a DataFrame.

    Columns: bigram (as string 'word1 word2'), count
    """
    top = result.bigrams.most_common(top_n)
    return pd.DataFrame(
        [(' '.join(bg), count) for bg, count in top],
        columns=['bigram', 'count']
    )


def get_top_trigrams(result: NLPResult, top_n: int = 10) -> pd.DataFrame:
    """
    Returns top N trigrams as a DataFrame.

    Columns: trigram (as string), count
    """
    top = result.trigrams.most_common(top_n)
    return pd.DataFrame(
        [(' '.join(tg), count) for tg, count in top],
        columns=['trigram', 'count']
    )


def get_nlp_summary(result: NLPResult) -> dict:
    """
    Returns a flat summary dict for display on the dashboard overview card.
    """
    fastest_responder = (
        min(result.response_times, key=result.response_times.get)
        if result.response_times else 'N/A'
    )

    return {
        'total_words'        : f"{result.total_words:,}",
        'unique_words'       : f"{result.unique_words:,}",
        'peak_hour'          : f"{result.hour_activity.idxmax()}:00",
        'most_active_day'    : result.day_activity.idxmax(),
        'top_word'           : result.word_frequencies.most_common(1)[0][0]
                               if result.word_frequencies else 'N/A',
        'top_bigram'         : ' '.join(result.bigrams.most_common(1)[0][0])
                               if result.bigrams else 'N/A',
        'fastest_responder'  : fastest_responder,
        'fastest_resp_mins'  : result.response_times.get(fastest_responder, 'N/A'),
    }


# --------------------------------------------------------------------------- #
# QUICK TEST — python src/nlp_processor.py <chat.txt>                         #
# --------------------------------------------------------------------------- #

if __name__ == '__main__':
    import sys, os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    from src.whatsapp_parser  import parse_chat
    from src.data_processor   import process_dataframe

    path = sys.argv[1] if len(sys.argv) > 1 else 'data/sample_chats/chat.txt'

    try:
        print("\nStep 1: Parsing...")
        df = parse_chat(path)

        print("Step 2: Processing...")
        df = process_dataframe(df)

        print("Step 3: NLP analysis...")
        result = process_nlp(df)

        print("\n" + "="*60)
        print("TOP 15 WORDS:")
        print("="*60)
        print(get_top_words(result, 15).to_string(index=False))

        print("\n" + "="*60)
        print("TOP 10 BIGRAMS:")
        print("="*60)
        print(get_top_bigrams(result, 10).to_string(index=False))

        print("\n" + "="*60)
        print("TOP 10 TRIGRAMS:")
        print("="*60)
        print(get_top_trigrams(result, 10).to_string(index=False))

        print("\n" + "="*60)
        print("USER STATS:")
        print("="*60)
        print(result.user_stats.to_string(index=False))

        print("\n" + "="*60)
        print("ACTIVITY — MESSAGES PER HOUR:")
        print("="*60)
        print(result.hour_activity.to_string())

        print("\n" + "="*60)
        print("ACTIVITY — MESSAGES PER DAY:")
        print("="*60)
        print(result.day_activity.to_string())

        print("\n" + "="*60)
        print("TF-IDF KEYWORDS PER SENDER:")
        print("="*60)
        for sender, kws in result.tfidf_keywords.items():
            print(f"\n  {sender}:")
            for word, score in kws:
                print(f"    {word:<20} {score}")

        print("\n" + "="*60)
        print("RESPONSE TIMES (minutes):")
        print("="*60)
        for sender, mins in sorted(result.response_times.items(), key=lambda x: x[1]):
            print(f"  {sender:<20} {mins} min avg")

        print("\n" + "="*60)
        print("NLP SUMMARY:")
        print("="*60)
        for k, v in get_nlp_summary(result).items():
            print(f"  {k:<25}: {v}")

    except (FileNotFoundError, ValueError) as e:
        print(f"\nERROR: {e}")
        print("Usage: python src/nlp_processor.py <path_to_chat.txt>")