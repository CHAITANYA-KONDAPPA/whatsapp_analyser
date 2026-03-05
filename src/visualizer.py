# =============================================================================
# visualizer.py
# -----------------------------------------------------------------------------
# PURPOSE : Generate all 10 analysis charts and save them as .png files to
#           results/visualizations/. The Flask app (app.py) serves these
#           images directly to the HTML dashboard.
#
# INPUT   : NLPResult object from nlp_processor.process_nlp()
#           DataFrame with sentiment columns from sentiment_analyzer.analyze_sentiment()
#
# OUTPUT  : 10 .png chart files saved to results/visualizations/:
#               01_wordcloud.png
#               02_top_words.png
#               03_hourly_activity.png
#               04_daily_activity.png
#               05_monthly_activity.png
#               06_user_participation.png
#               07_top_bigrams.png
#               08_sentiment_distribution.png
#               09_sentiment_trend.png
#               10_sentiment_by_user.png
#
# USAGE   : from src.visualizer import generate_all_charts
#           chart_paths = generate_all_charts(df, nlp_result)
#
# CRITICAL: Uses matplotlib.use('Agg') — non-interactive backend.
#           This is REQUIRED for Flask (server has no display screen).
#           Must be set BEFORE any other matplotlib import.
#
# AUTHOR  : Team — Chaitanya / Anudeep / Ram Teja
# =============================================================================

import os
import logging
import warnings
warnings.filterwarnings('ignore')

# CRITICAL — set non-interactive backend BEFORE importing pyplot
# Without this, Flask will crash with "cannot connect to display" error
import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import seaborn as sns
import pandas as pd
import numpy as np
from wordcloud import WordCloud

# Our modules
from src.nlp_processor      import NLPResult, get_top_words, get_top_bigrams
from src.sentiment_analyzer import (
    get_sentiment_by_sender,
    get_sentiment_over_time,
    get_overall_mood,
)

# --------------------------------------------------------------------------- #
# Logging                                                                      #
# --------------------------------------------------------------------------- #
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# CONSTANTS — Change these to restyle ALL charts at once                      #
# --------------------------------------------------------------------------- #

OUTPUT_DIR = os.path.join('results', 'visualizations')

# Color palette — consistent across all charts
COLORS = {
    'primary'    : '#2E75B6',   # main blue
    'secondary'  : '#1F4E79',   # dark blue
    'accent'     : '#70AD47',   # green
    'positive'   : '#70AD47',   # green  for positive sentiment
    'neutral'    : '#FFC000',   # amber  for neutral sentiment
    'negative'   : '#FF4B4B',   # red    for negative sentiment
    'background' : '#F8F9FA',   # light grey background
    'grid'       : '#E0E0E0',   # light grid lines
    'text'       : '#1A1A2E',   # dark text
}

# Multi-color palette for per-sender charts
SENDER_PALETTE = [
    '#2E75B6', '#70AD47', '#FF4B4B', '#FFC000',
    '#9B59B6', '#E67E22', '#1ABC9C', '#E74C3C',
]

# Chart output settings
DPI         = 150       # high enough for crisp web display
FIGSIZE_STD = (10, 5)   # standard chart size
FIGSIZE_SQ  = (8, 8)    # square (pie charts, wordcloud)
FIGSIZE_WIDE= (12, 5)   # wide (timeline charts)

# Chart file names — fixed names so app.py always knows where to find them
CHART_FILES = {
    'wordcloud'            : '01_wordcloud.png',
    'top_words'            : '02_top_words.png',
    'hourly_activity'      : '03_hourly_activity.png',
    'daily_activity'       : '04_daily_activity.png',
    'monthly_activity'     : '05_monthly_activity.png',
    'user_participation'   : '06_user_participation.png',
    'top_bigrams'          : '07_top_bigrams.png',
    'sentiment_distribution': '08_sentiment_distribution.png',
    'sentiment_trend'      : '09_sentiment_trend.png',
    'sentiment_by_user'    : '10_sentiment_by_user.png',
}


# --------------------------------------------------------------------------- #
# SETUP HELPERS                                                               #
# --------------------------------------------------------------------------- #

def _setup_output_dir():
    """Create output directory if it doesn't exist."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)


def _apply_style(ax, title: str, xlabel: str = '', ylabel: str = ''):
    """
    Apply consistent styling to any chart axis.
    Sets title, labels, grid, background, and font sizes in one call.
    """
    ax.set_title(title, fontsize=14, fontweight='bold',
                 color=COLORS['text'], pad=15)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=11, color=COLORS['text'])
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=11, color=COLORS['text'])

    ax.set_facecolor(COLORS['background'])
    ax.grid(axis='y', color=COLORS['grid'], linewidth=0.8, linestyle='--', alpha=0.7)
    ax.tick_params(colors=COLORS['text'], labelsize=9)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color(COLORS['grid'])
    ax.spines['bottom'].set_color(COLORS['grid'])


def _save_figure(fig, chart_key: str) -> str:
    """Save figure to output directory and close it. Returns saved path."""
    path = os.path.join(OUTPUT_DIR, CHART_FILES[chart_key])
    fig.savefig(path, dpi=DPI, bbox_inches='tight',
                facecolor=COLORS['background'])
    plt.close(fig)
    logger.info(f"  Saved: {CHART_FILES[chart_key]}")
    return path


# --------------------------------------------------------------------------- #
# CHART 1 — WordCloud                                                         #
# --------------------------------------------------------------------------- #

def chart_wordcloud(result: NLPResult) -> str:
    """
    Generate a WordCloud from word frequencies.
    Bigger word = more frequent across all messages.

    Uses the full word_frequencies Counter from NLPResult.
    """
    if not result.word_frequencies:
        logger.warning("  WordCloud skipped — no word frequencies.")
        return ''

    wc = WordCloud(
        width            = 1200,
        height           = 600,
        background_color = 'white',
        colormap         = 'Blues',
        max_words        = 150,
        min_font_size    = 10,
        max_font_size    = 120,
        collocations     = False,   # don't repeat bigrams already in the data
        prefer_horizontal= 0.85,
    ).generate_from_frequencies(result.word_frequencies)

    fig, ax = plt.subplots(figsize=(12, 6))
    fig.patch.set_facecolor('white')
    ax.imshow(wc, interpolation='bilinear')
    ax.axis('off')
    ax.set_title('Most Used Words', fontsize=16, fontweight='bold',
                 color=COLORS['text'], pad=15)

    return _save_figure(fig, 'wordcloud')


# --------------------------------------------------------------------------- #
# CHART 2 — Top 20 Words Bar Chart                                            #
# --------------------------------------------------------------------------- #

def chart_top_words(result: NLPResult, top_n: int = 20) -> str:
    """
    Horizontal bar chart of the top N most used words.
    More precise than WordCloud — shows exact counts.
    """
    df_words = get_top_words(result, top_n)
    if df_words.empty:
        logger.warning("  Top words chart skipped — no data.")
        return ''

    fig, ax = plt.subplots(figsize=(10, 7))
    fig.patch.set_facecolor(COLORS['background'])

    # Gradient colors — darker for more frequent
    colors = plt.cm.Blues(
        np.linspace(0.4, 0.9, len(df_words))
    )[::-1]

    bars = ax.barh(df_words['word'], df_words['count'],
                   color=colors, edgecolor='white', height=0.7)

    # Add count labels on the bars
    for bar, count in zip(bars, df_words['count']):
        ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height() / 2,
                str(count), va='center', ha='left',
                fontsize=9, color=COLORS['text'])

    _apply_style(ax, f'Top {top_n} Most Used Words',
                 xlabel='Frequency', ylabel='Word')
    ax.invert_yaxis()   # most frequent at top
    ax.grid(axis='x', color=COLORS['grid'], linewidth=0.8,
            linestyle='--', alpha=0.7)
    ax.grid(axis='y', visible=False)

    fig.tight_layout()
    return _save_figure(fig, 'top_words')


# --------------------------------------------------------------------------- #
# CHART 3 — Messages Per Hour (Activity Heatmap)                             #
# --------------------------------------------------------------------------- #

def chart_hourly_activity(result: NLPResult) -> str:
    """
    Bar chart showing how many messages were sent each hour (0-23).
    Reveals the chat's peak activity time.

    Peak hour bar is highlighted in accent color.
    All others in primary blue.
    """
    hour_data = result.hour_activity
    if hour_data.empty:
        logger.warning("  Hourly activity chart skipped — no data.")
        return ''

    fig, ax = plt.subplots(figsize=FIGSIZE_WIDE)
    fig.patch.set_facecolor(COLORS['background'])

    hours  = hour_data.index.tolist()
    counts = hour_data.values.tolist()
    peak   = int(hour_data.idxmax())

    bar_colors = [
        COLORS['accent'] if h == peak else COLORS['primary']
        for h in hours
    ]

    bars = ax.bar(hours, counts, color=bar_colors,
                  edgecolor='white', width=0.8)

    # Label the peak bar
    peak_count = int(hour_data[peak])
    ax.annotate(
        f'Peak\n{peak}:00\n({peak_count} msgs)',
        xy=(peak, peak_count),
        xytext=(peak + 1.5, peak_count * 0.85),
        fontsize=8, color=COLORS['secondary'], fontweight='bold',
        arrowprops=dict(arrowstyle='->', color=COLORS['secondary'], lw=1.2),
    )

    ax.set_xticks(range(24))
    ax.set_xticklabels([f'{h:02d}:00' for h in range(24)],
                       rotation=45, ha='right', fontsize=8)

    _apply_style(ax, 'Message Activity by Hour of Day',
                 xlabel='Hour', ylabel='Number of Messages')
    fig.tight_layout()
    return _save_figure(fig, 'hourly_activity')


# --------------------------------------------------------------------------- #
# CHART 4 — Messages Per Day of Week                                         #
# --------------------------------------------------------------------------- #

def chart_daily_activity(result: NLPResult) -> str:
    """
    Bar chart of messages per day of week (Mon → Sun).
    Most active day bar highlighted in accent color.
    """
    day_data = result.day_activity
    if day_data.empty:
        logger.warning("  Daily activity chart skipped — no data.")
        return ''

    fig, ax = plt.subplots(figsize=FIGSIZE_STD)
    fig.patch.set_facecolor(COLORS['background'])

    peak_day = day_data.idxmax()
    bar_colors = [
        COLORS['accent'] if d == peak_day else COLORS['primary']
        for d in day_data.index
    ]

    bars = ax.bar(day_data.index, day_data.values,
                  color=bar_colors, edgecolor='white', width=0.6)

    # Value labels on top of each bar
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.5,
                str(int(h)), ha='center', va='bottom',
                fontsize=9, color=COLORS['text'])

    _apply_style(ax, 'Message Activity by Day of Week',
                 xlabel='Day', ylabel='Number of Messages')
    ax.tick_params(axis='x', rotation=0)
    fig.tight_layout()
    return _save_figure(fig, 'daily_activity')


# --------------------------------------------------------------------------- #
# CHART 5 — Messages Per Month Timeline                                       #
# --------------------------------------------------------------------------- #

def chart_monthly_activity(result: NLPResult) -> str:
    """
    Line chart with area fill showing message volume per month.
    Shows how chat activity grew or shrank over the chat's lifetime.
    """
    month_data = result.month_activity
    if month_data.empty:
        logger.warning("  Monthly activity chart skipped — no data.")
        return ''

    fig, ax = plt.subplots(figsize=FIGSIZE_WIDE)
    fig.patch.set_facecolor(COLORS['background'])

    x = range(len(month_data))

    ax.plot(x, month_data.values, color=COLORS['primary'],
            linewidth=2.5, marker='o', markersize=6,
            markerfacecolor=COLORS['accent'], markeredgecolor='white',
            markeredgewidth=1.5, zorder=3)

    ax.fill_between(x, month_data.values, alpha=0.15,
                    color=COLORS['primary'])

    # Value labels above each point
    for i, val in enumerate(month_data.values):
        ax.text(i, val + max(month_data.values) * 0.02,
                str(int(val)), ha='center', va='bottom',
                fontsize=8, color=COLORS['text'])

    ax.set_xticks(list(x))
    ax.set_xticklabels(month_data.index, rotation=30, ha='right', fontsize=9)

    _apply_style(ax, 'Message Volume by Month',
                 xlabel='Month', ylabel='Number of Messages')
    fig.tight_layout()
    return _save_figure(fig, 'monthly_activity')


# --------------------------------------------------------------------------- #
# CHART 6 — User Participation Pie Chart                                      #
# --------------------------------------------------------------------------- #

def chart_user_participation(result: NLPResult) -> str:
    """
    Donut-style pie chart showing each sender's share of total messages.
    Uses SENDER_PALETTE for distinct colors per person.
    """
    user_data = result.user_stats
    if user_data.empty:
        logger.warning("  User participation chart skipped — no data.")
        return ''

    fig, ax = plt.subplots(figsize=FIGSIZE_SQ)
    fig.patch.set_facecolor(COLORS['background'])

    senders = user_data['sender'].tolist()
    counts  = user_data['total_messages'].tolist()
    colors  = SENDER_PALETTE[:len(senders)]

    wedges, texts, autotexts = ax.pie(
        counts,
        labels      = senders,
        autopct     = '%1.1f%%',
        colors      = colors,
        startangle  = 140,
        pctdistance = 0.82,
        wedgeprops  = dict(width=0.55, edgecolor='white', linewidth=2),  # donut hole
    )

    for text in texts:
        text.set_fontsize(11)
        text.set_color(COLORS['text'])
    for autotext in autotexts:
        autotext.set_fontsize(9)
        autotext.set_color('white')
        autotext.set_fontweight('bold')

    ax.set_title('Message Share by User', fontsize=14, fontweight='bold',
                 color=COLORS['text'], pad=20)

    # Center label showing total
    total = sum(counts)
    ax.text(0, 0, f'{total:,}\nMessages', ha='center', va='center',
            fontsize=12, fontweight='bold', color=COLORS['text'])

    fig.tight_layout()
    return _save_figure(fig, 'user_participation')


# --------------------------------------------------------------------------- #
# CHART 7 — Top Bigrams Bar Chart                                             #
# --------------------------------------------------------------------------- #

def chart_top_bigrams(result: NLPResult, top_n: int = 10) -> str:
    """
    Horizontal bar chart of the most common 2-word phrases.
    Shows conversational patterns ("let know", "good morning", "on way").
    """
    df_bigrams = get_top_bigrams(result, top_n)
    if df_bigrams.empty:
        logger.warning("  Top bigrams chart skipped — no data.")
        return ''

    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor(COLORS['background'])

    colors = plt.cm.Greens(
        np.linspace(0.4, 0.85, len(df_bigrams))
    )[::-1]

    bars = ax.barh(df_bigrams['bigram'], df_bigrams['count'],
                   color=colors, edgecolor='white', height=0.65)

    for bar, count in zip(bars, df_bigrams['count']):
        ax.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height() / 2,
                str(count), va='center', ha='left',
                fontsize=9, color=COLORS['text'])

    _apply_style(ax, f'Top {top_n} Most Common Phrases (Bigrams)',
                 xlabel='Frequency', ylabel='Phrase')
    ax.invert_yaxis()
    ax.grid(axis='x', color=COLORS['grid'], linewidth=0.8,
            linestyle='--', alpha=0.7)
    ax.grid(axis='y', visible=False)

    fig.tight_layout()
    return _save_figure(fig, 'top_bigrams')


# --------------------------------------------------------------------------- #
# CHART 8 — Sentiment Distribution Pie Chart                                  #
# --------------------------------------------------------------------------- #

def chart_sentiment_distribution(df: pd.DataFrame) -> str:
    """
    Donut pie chart showing the overall split of
    Positive / Neutral / Negative messages across the entire chat.
    """
    text_df = df[~df['is_media'] & ~df['is_deleted']]
    if text_df.empty or 'sentiment_label' not in df.columns:
        logger.warning("  Sentiment distribution chart skipped — no data.")
        return ''

    counts = text_df['sentiment_label'].value_counts()
    labels = counts.index.tolist()
    values = counts.values.tolist()

    # Map consistent colors to labels
    color_map = {
        'Positive': COLORS['positive'],
        'Neutral' : COLORS['neutral'],
        'Negative': COLORS['negative'],
    }
    colors = [color_map.get(l, COLORS['primary']) for l in labels]

    fig, ax = plt.subplots(figsize=(8, 7))
    fig.patch.set_facecolor(COLORS['background'])

    wedges, texts, autotexts = ax.pie(
        values,
        labels      = labels,
        autopct     = '%1.1f%%',
        colors      = colors,
        startangle  = 90,
        pctdistance = 0.82,
        wedgeprops  = dict(width=0.55, edgecolor='white', linewidth=2),
    )

    for text in texts:
        text.set_fontsize(12)
        text.set_fontweight('bold')
        text.set_color(COLORS['text'])
    for autotext in autotexts:
        autotext.set_fontsize(10)
        autotext.set_color('white')
        autotext.set_fontweight('bold')

    # Overall mood in center
    mood = get_overall_mood(df)
    ax.text(0, 0.1, mood.get('mood_description', ''),
            ha='center', va='center', fontsize=10,
            color=COLORS['text'], fontweight='bold')
    ax.text(0, -0.2, f"avg: {mood.get('overall_score', 0):+.3f}",
            ha='center', va='center', fontsize=9, color='#666666')

    ax.set_title('Overall Sentiment Distribution', fontsize=14,
                 fontweight='bold', color=COLORS['text'], pad=20)

    fig.tight_layout()
    return _save_figure(fig, 'sentiment_distribution')


# --------------------------------------------------------------------------- #
# CHART 9 — Sentiment Trend Over Time                                         #
# --------------------------------------------------------------------------- #

def chart_sentiment_trend(df: pd.DataFrame) -> str:
    """
    Line chart of average weekly sentiment score over time.

    Shows if the chat's mood improved or worsened over its lifetime.
    A horizontal dashed line at y=0 marks the Positive/Negative boundary.
    Areas above = positive periods, below = negative periods.
    """
    if 'sentiment_score' not in df.columns:
        logger.warning("  Sentiment trend chart skipped — no sentiment_score column.")
        return ''

    trend_df = get_sentiment_over_time(df, freq='W')
    if trend_df.empty or len(trend_df) < 2:
        logger.warning("  Sentiment trend chart skipped — not enough data points.")
        return ''

    fig, ax = plt.subplots(figsize=FIGSIZE_WIDE)
    fig.patch.set_facecolor(COLORS['background'])

    dates  = pd.to_datetime(trend_df['date'])
    scores = trend_df['avg_score'].values

    # Positive / negative area fills
    ax.fill_between(dates, scores, 0,
                    where=(scores >= 0),
                    alpha=0.25, color=COLORS['positive'],
                    label='Positive period')
    ax.fill_between(dates, scores, 0,
                    where=(scores < 0),
                    alpha=0.25, color=COLORS['negative'],
                    label='Negative period')

    ax.plot(dates, scores, color=COLORS['primary'],
            linewidth=2, marker='o', markersize=4,
            markerfacecolor=COLORS['primary'], zorder=3)

    # Zero line
    ax.axhline(y=0, color='#888888', linewidth=1,
               linestyle='--', alpha=0.8, label='Neutral baseline')

    ax.set_ylim(-1, 1)
    ax.set_ylabel('Avg Sentiment Score', fontsize=11, color=COLORS['text'])
    ax.set_xlabel('Date', fontsize=11, color=COLORS['text'])

    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%+.2f'))
    ax.tick_params(axis='x', rotation=30, labelsize=8)
    ax.tick_params(axis='y', labelsize=9)

    ax.legend(fontsize=9, loc='upper left',
              framealpha=0.85, edgecolor=COLORS['grid'])
    ax.set_facecolor(COLORS['background'])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_title('Sentiment Trend Over Time (Weekly Avg)',
                 fontsize=14, fontweight='bold',
                 color=COLORS['text'], pad=15)

    fig.tight_layout()
    return _save_figure(fig, 'sentiment_trend')


# --------------------------------------------------------------------------- #
# CHART 10 — Per-User Sentiment Stacked Bar Chart                            #
# --------------------------------------------------------------------------- #

def chart_sentiment_by_user(df: pd.DataFrame) -> str:
    """
    Stacked horizontal bar chart showing each user's
    Positive / Neutral / Negative message breakdown.

    Makes it immediately clear who is the most positive or negative
    communicator in the chat.
    """
    if 'sentiment_label' not in df.columns:
        logger.warning("  Sentiment by user chart skipped — no sentiment_label column.")
        return ''

    sender_df = get_sentiment_by_sender(df)
    if sender_df.empty:
        return ''

    fig, ax = plt.subplots(figsize=(10, max(5, len(sender_df) * 1.2)))
    fig.patch.set_facecolor(COLORS['background'])

    senders  = sender_df['sender'].tolist()
    pos_pct  = sender_df['positive_pct'].tolist()
    neu_pct  = sender_df['neutral_pct'].tolist()
    neg_pct  = sender_df['negative_pct'].tolist()
    y_pos    = range(len(senders))

    # Stack: Positive | Neutral | Negative
    bars_pos = ax.barh(y_pos, pos_pct, color=COLORS['positive'],
                       edgecolor='white', height=0.55, label='Positive')
    bars_neu = ax.barh(y_pos, neu_pct, left=pos_pct,
                       color=COLORS['neutral'],
                       edgecolor='white', height=0.55, label='Neutral')
    bars_neg = ax.barh(y_pos, neg_pct,
                       left=[p + n for p, n in zip(pos_pct, neu_pct)],
                       color=COLORS['negative'],
                       edgecolor='white', height=0.55, label='Negative')

    # Percentage labels inside bars (only if segment wide enough)
    for i, (p, n, ng) in enumerate(zip(pos_pct, neu_pct, neg_pct)):
        if p > 8:
            ax.text(p / 2, i, f'{p:.0f}%', ha='center', va='center',
                    fontsize=8, color='white', fontweight='bold')
        if n > 8:
            ax.text(p + n / 2, i, f'{n:.0f}%', ha='center', va='center',
                    fontsize=8, color=COLORS['text'], fontweight='bold')
        if ng > 8:
            ax.text(p + n + ng / 2, i, f'{ng:.0f}%', ha='center', va='center',
                    fontsize=8, color='white', fontweight='bold')

    ax.set_yticks(list(y_pos))
    ax.set_yticklabels(senders, fontsize=10, color=COLORS['text'])
    ax.set_xlim(0, 100)
    ax.set_xlabel('Percentage of Messages (%)', fontsize=11, color=COLORS['text'])

    ax.legend(loc='lower right', fontsize=9,
              framealpha=0.9, edgecolor=COLORS['grid'])
    ax.set_facecolor(COLORS['background'])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_title('Sentiment Breakdown by User',
                 fontsize=14, fontweight='bold',
                 color=COLORS['text'], pad=15)

    fig.tight_layout()
    return _save_figure(fig, 'sentiment_by_user')


# --------------------------------------------------------------------------- #
# MAIN PUBLIC FUNCTION : generate_all_charts()                               #
# --------------------------------------------------------------------------- #

def generate_all_charts(df: pd.DataFrame, result: NLPResult) -> dict:
    """
    Master function — call this from pipeline.py.

    Runs all 10 chart generators in order and returns a dict of
    { chart_key: saved_file_path } for app.py to serve.

    Parameters:
        df      : Full DataFrame (must have sentiment columns from analyze_sentiment())
        result  : NLPResult from process_nlp()

    Returns:
        {
            'wordcloud'             : 'results/visualizations/01_wordcloud.png',
            'top_words'             : 'results/visualizations/02_top_words.png',
            ...
        }
    """
    _setup_output_dir()

    logger.info("Generating all charts...")
    logger.info(f"Output directory: {OUTPUT_DIR}")

    chart_paths = {}

    generators = [
        ('wordcloud',             lambda: chart_wordcloud(result)),
        ('top_words',             lambda: chart_top_words(result)),
        ('hourly_activity',       lambda: chart_hourly_activity(result)),
        ('daily_activity',        lambda: chart_daily_activity(result)),
        ('monthly_activity',      lambda: chart_monthly_activity(result)),
        ('user_participation',    lambda: chart_user_participation(result)),
        ('top_bigrams',           lambda: chart_top_bigrams(result)),
        ('sentiment_distribution',lambda: chart_sentiment_distribution(df)),
        ('sentiment_trend',       lambda: chart_sentiment_trend(df)),
        ('sentiment_by_user',     lambda: chart_sentiment_by_user(df)),
    ]

    for i, (key, generator) in enumerate(generators, 1):
        logger.info(f"Chart {i:02d}/10 — {key}...")
        try:
            path = generator()
            if path:
                chart_paths[key] = path
        except Exception as e:
            logger.error(f"  Chart '{key}' failed: {e}")
            chart_paths[key] = ''

    success = sum(1 for p in chart_paths.values() if p)
    logger.info(f"Chart generation complete: {success}/10 charts saved.")

    return chart_paths


# --------------------------------------------------------------------------- #
# UTILITY : get_chart_paths()                                                 #
# --------------------------------------------------------------------------- #

def get_chart_paths() -> dict:
    """
    Returns the expected file paths for all charts.
    Used by app.py to check which charts exist before rendering dashboard.
    Returns empty string for charts that haven't been generated yet.
    """
    return {
        key: os.path.join(OUTPUT_DIR, filename)
        if os.path.exists(os.path.join(OUTPUT_DIR, filename))
        else ''
        for key, filename in CHART_FILES.items()
    }


# --------------------------------------------------------------------------- #
# QUICK TEST — python src/visualizer.py <chat.txt>                            #
# --------------------------------------------------------------------------- #

if __name__ == '__main__':
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    from src.whatsapp_parser    import parse_chat
    from src.data_processor     import process_dataframe
    from src.nlp_processor      import process_nlp
    from src.sentiment_analyzer import analyze_sentiment

    path = sys.argv[1] if len(sys.argv) > 1 else 'data/sample_chats/chat.txt'

    try:
        print("\nStep 1: Parsing...")
        df = parse_chat(path)

        print("Step 2: Processing...")
        df = process_dataframe(df)

        print("Step 3: NLP analysis...")
        nlp_result = process_nlp(df)

        print("Step 4: Sentiment analysis...")
        df = analyze_sentiment(df)

        print("Step 5: Generating all charts...")
        chart_paths = generate_all_charts(df, nlp_result)

        print("\n" + "="*60)
        print("CHARTS GENERATED:")
        print("="*60)
        for key, path in chart_paths.items():
            status = "✓" if path else "✗ FAILED"
            print(f"  {status}  {key:<30} → {path}")

        print(f"\nAll charts saved to: {OUTPUT_DIR}/")

    except (FileNotFoundError, ValueError) as e:
        print(f"\nERROR: {e}")
        print("Usage: python src/visualizer.py <path_to_chat.txt>")
        