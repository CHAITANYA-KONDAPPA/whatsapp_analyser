# =============================================================================
# data_processor.py
# -----------------------------------------------------------------------------
# PURPOSE : Take the raw DataFrame produced by whatsapp_parser.py and clean
#           every message so it is ready for NLP and ML analysis.
#
# INPUT   : Raw DataFrame from parse_chat()
# OUTPUT  : Same DataFrame with 4 new columns added:
#               - cleaned_message  : fully cleaned text for NLP/ML
#               - emoji_list       : list of emojis found in original message
#               - emoji_count      : number of emojis in message
#               - has_url          : True if original message had a URL
#
# USAGE   : from src.data_processor import process_dataframe
#           df = process_dataframe(df)
#
# AUTHOR  : Chaitanya 
# =============================================================================

import re
import logging
import pandas as pd
import nltk

# --------------------------------------------------------------------------- #
# Download required NLTK data (only downloads if not already present)         #
# --------------------------------------------------------------------------- #
# stopwords : common words like "the", "is", "a" that carry no meaning
# wordnet   : database used by the lemmatizer to find root words
# averaged_perceptron_tagger : used internally by some NLTK functions

nltk.download('stopwords',quiet=True)
nltk.download('wordnet',quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)
nltk.download('omw-1.4',quiet=True)

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# --------------------------------------------------------------------------- #
# Logging                                                                      #
# --------------------------------------------------------------------------- #
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# CONSTANTS                                                                    #
# --------------------------------------------------------------------------- #

# Standard English stopwords from NLTK
# e.g. {"i", "me", "the", "a", "is", "was", "are", ...}
STOP_WORDS = set(stopwords.words('english'))

# Extra words that are common in WhatsApp chats but carry no meaning
# Add more as you discover them in your own chat data
CUSTOM_STOP_WORDS = {
    'ok', 'okay', 'yeah', 'yep', 'yes', 'no', 'oh',
    'hmm', 'hm', 'ah', 'ha', 'lol', 'omg', 'lmao',
    'bro', 'guy', 'guys', 'hi', 'hey', 'hello', 'bye',
    'haha', 'hahaha', 'hehe', 'na', 'nah', 'yaar',
    'message', 'deleted', 'media', 'omitted', 'null'
}

ALL_STOP_WORDS = STOP_WORDS | CUSTOM_STOP_WORDS

# Lemmatizer instance — reuse one instance, don't create inside a loop
lemmatizer = WordNetLemmatizer()

# Regex to detect URLs (http, https, www)
URL_PATTERN = re.compile(
    r'(https?://\S+|www\.\S+)',
    re.IGNORECASE
)

# Regex to keep ONLY lowercase English letters and spaces
# Everything else (numbers, symbols, punctuation) gets removed
KEEP_ONLY_LETTERS = re.compile(r'[^a-z\s]')

# Emoji unicode ranges — covers virtually all modern emojis
# We extract these BEFORE stripping non-letter characters
EMOJI_PATTERN = re.compile(
    "["
    "\U0001F600-\U0001F64F"   # emoticons (😀 → 🙏)
    "\U0001F300-\U0001F5FF"   # symbols & pictographs (🌀 → 🗿)
    "\U0001F680-\U0001F6FF"   # transport & map (🚀 → 🛿)
    "\U0001F1E0-\U0001F1FF"   # flags
    "\U00002702-\U000027B0"   # dingbats
    "\U000024C2-\U0001F251"   # enclosed characters
    "\U0001f926-\U0001f937"   # supplemental symbols
    "\U00010000-\U0010ffff"   # other emoji blocks
    "\u2640-\u2642"
    "\u2600-\u2B55"
    "\u200d"
    "\u23cf"
    "\u23e9"
    "\u231a"
    "\ufe0f"
    "\u3030"
    "]+",
    flags=re.UNICODE
)


# --------------------------------------------------------------------------- #
# STEP 1 : Check if a message should be skipped entirely                      #
# --------------------------------------------------------------------------- #

def _is_skippable(message: str) -> bool:
    """
    Returns True if a message carries no useful text for analysis.

    We skip:
    - Media placeholders  : "<Media omitted>"
    - Deleted messages    : "This message was deleted"
    - System messages     : "Messages are end-to-end encrypted"
    - Very short messages : single characters, just punctuation
    - Empty strings
    """
    if not message or not isinstance(message, str):
        return True

    msg = message.strip().lower()

    skippable_phrases = [
        '<media omitted>',
        'media omitted',
        'image omitted',
        'video omitted',
        'audio omitted',
        'document omitted',
        'sticker omitted',
        'this message was deleted',
        'you deleted this message',
        'messages are end-to-end encrypted',
        'null',
        '',
    ]

    if msg in skippable_phrases:
        return True

    # Skip if the message is only 1-2 characters after stripping
    if len(msg.replace(' ', '')) <= 2:
        return True

    return False


# --------------------------------------------------------------------------- #
# STEP 2 : Check and extract whether message contains a URL                   #
# --------------------------------------------------------------------------- #

def _extract_url_flag(message: str) -> bool:
    """
    Returns True if the message contains a URL (http/https/www).
    We record this BEFORE removing URLs so we don't lose the information.

    Why useful? → You can later show "who shares the most links" in the dashboard.
    """
    return bool(URL_PATTERN.search(message))


# --------------------------------------------------------------------------- #
# STEP 3 : Extract emojis from message                                        #
# --------------------------------------------------------------------------- #

def _extract_emojis(message: str) -> list:
    """
    Finds all emojis in the message and returns them as a list.
    We do this BEFORE any cleaning because cleaning removes emojis.

    Example:
        "Great work!! 🔥😂"  →  ['🔥', '😂']

    Why useful?
        Emojis strongly signal emotion. 😂 = positive, 😡 = negative.
        The visualizer uses this list to build an emoji frequency chart.
    """
    return EMOJI_PATTERN.findall(message)


# --------------------------------------------------------------------------- #
# STEP 4 : Core text cleaning function                                        #
# --------------------------------------------------------------------------- #

def _clean_text(message: str) -> str:
    """
    Applies the full cleaning pipeline to a single message string.

    Pipeline:
        1. Lowercase
        2. Remove URLs
        3. Remove emojis and special characters (keep only a-z and spaces)
        4. Remove stopwords
        5. Lemmatize each word (run/running/ran → run)
        6. Strip extra whitespace

    Returns a cleaned string. Returns empty string if nothing useful remains.
    """

    # ── 1. Lowercase ─────────────────────────────────────────────────────── #
    text = message.lower()

    # ── 2. Remove URLs ────────────────────────────────────────────────────── #
    text = URL_PATTERN.sub('', text)

    # ── 3. Remove emojis and keep only a-z letters + spaces ──────────────── #
    # First remove emojis explicitly
    text = EMOJI_PATTERN.sub('', text)
    # Then remove anything that isn't a letter or space
    text = KEEP_ONLY_LETTERS.sub(' ', text)

    # ── 4. Tokenize (split into words) ────────────────────────────────────── #
    words = text.split()

    # ── 5. Remove stopwords ───────────────────────────────────────────────── #
    words = [w for w in words if w not in ALL_STOP_WORDS]

    # ── 6. Lemmatize each word ────────────────────────────────────────────── #
    # lemmatize(word, pos='v') treats word as a verb → better for chat text
    # e.g. "running" → "run", "went" → "go", "better" → "good"
    words = [lemmatizer.lemmatize(w, pos='v') for w in words]

    # ── 7. Remove any words that became too short after lemmatization ─────── #
    words = [w for w in words if len(w) > 2]

    # ── 8. Rejoin into a clean string ─────────────────────────────────────── #
    return ' '.join(words).strip()


# --------------------------------------------------------------------------- #
# MAIN PUBLIC FUNCTION : process_dataframe()                                  #
# --------------------------------------------------------------------------- #

def process_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Master function — call this from pipeline.py.

    Takes the raw DataFrame from whatsapp_parser.parse_chat() and adds
    4 new columns to it:
        - cleaned_message  : clean text ready for NLP/ML
        - emoji_list       : list of emojis from original message
        - emoji_count      : integer count of emojis
        - has_url          : True/False

    Parameters:
        df (pd.DataFrame): Output of parse_chat()

    Returns:
        pd.DataFrame: Same DataFrame with 4 new columns added
    """

    logger.info("Starting data processing pipeline...")
    logger.info(f"Input: {len(df)} messages")

    # Work on a copy so we never modify the original DataFrame
    df = df.copy()

    # ── Extract emoji list for every message ──────────────────────────────── #
    logger.info("Step 1/3 — Extracting emojis...")
    df['emoji_list']  = df['message'].apply(
        lambda m: _extract_emojis(m) if not _is_skippable(m) else []
    )
    df['emoji_count'] = df['emoji_list'].apply(len)

    # ── Check for URLs ────────────────────────────────────────────────────── #
    logger.info("Step 2/3 — Detecting URLs...")
    df['has_url'] = df['message'].apply(
        lambda m: _extract_url_flag(m) if not _is_skippable(m) else False
    )

    # ── Clean message text ────────────────────────────────────────────────── #
    logger.info("Step 3/3 — Cleaning message text...")
    df['cleaned_message'] = df['message'].apply(
        lambda m: _clean_text(m) if not _is_skippable(m) else ''
    )

    # ── Final stats ───────────────────────────────────────────────────────── #
    total           = len(df)
    with_text       = (df['cleaned_message'] != '').sum()
    total_emojis    = df['emoji_count'].sum()
    msgs_with_url   = df['has_url'].sum()
    msgs_with_emoji = (df['emoji_count'] > 0).sum()

    logger.info("Data processing complete!")
    logger.info(f"  Total messages         : {total}")
    logger.info(f"  Messages with text     : {with_text}")
    logger.info(f"  Messages with emojis   : {msgs_with_emoji} ({total_emojis} total emojis)")
    logger.info(f"  Messages with URLs     : {msgs_with_url}")

    return df


# --------------------------------------------------------------------------- #
# UTILITY : get_top_emojis()                                                  #
# --------------------------------------------------------------------------- #

def get_top_emojis(df: pd.DataFrame, top_n: int = 10) -> pd.Series:
    """
    Returns the top N most used emojis across all messages.
    Pass this to visualizer.py for the emoji frequency chart.

    Example output:
        😂    142
        🔥     89
        ❤️     74
        ...
    """
    # Flatten the list of emoji lists into one big list
    all_emojis = [emoji for emoji_list in df['emoji_list'] for emoji in emoji_list]

    if not all_emojis:
        logger.warning("No emojis found in this chat.")
        return pd.Series(dtype=int)

    return pd.Series(all_emojis).value_counts().head(top_n)


# --------------------------------------------------------------------------- #
# UTILITY : get_processing_stats()                                            #
# --------------------------------------------------------------------------- #

def get_processing_stats(df: pd.DataFrame) -> dict:
    """
    Returns a summary of what was found during processing.
    Useful for displaying on the dashboard.
    """
    text_df = df[df['cleaned_message'] != '']

    return {
        'total_messages'        : len(df),
        'processable_messages'  : len(text_df),
        'messages_with_emoji'   : int((df['emoji_count'] > 0).sum()),
        'total_emojis'          : int(df['emoji_count'].sum()),
        'messages_with_url'     : int(df['has_url'].sum()),
        'unique_words'          : len(set(
            ' '.join(text_df['cleaned_message']).split()
        )),
        'avg_words_after_clean' : round(
            text_df['cleaned_message'].apply(lambda x: len(x.split())).mean(), 2
        ),
    }


# --------------------------------------------------------------------------- #
# QUICK TEST — python src/data_processor.py                                   #
# --------------------------------------------------------------------------- #

if __name__ == '__main__':
    import sys
    import os

    # Add parent directory to path so we can import whatsapp_parser
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from src.whatsapp_parser import parse_chat

    if len(sys.argv) > 1:
        path = sys.argv[1]
    else:
        path = 'data/sample_chats/chat.txt'

    try:
        # Step 1: Parse
        print("\nStep 1: Parsing chat file...")
        df = parse_chat(path)
        print(f"  Parser output: {len(df)} messages, columns: {list(df.columns)}")

        # Step 2: Process
        print("\nStep 2: Processing / cleaning...")
        df = process_dataframe(df)

        # Step 3: Show results
        print("\n" + "="*60)
        print("SAMPLE — Original vs Cleaned (first 5 text messages):")
        print("="*60)
        sample = df[df['cleaned_message'] != ''][['message', 'cleaned_message', 'emoji_list', 'has_url']].head(5)
        for _, row in sample.iterrows():
            print(f"\n  Original : {row['message']}")
            print(f"  Cleaned  : {row['cleaned_message']}")
            print(f"  Emojis   : {row['emoji_list']}")
            print(f"  Has URL  : {row['has_url']}")

        print("\n" + "="*60)
        print("TOP 10 EMOJIS:")
        print("="*60)
        print(get_top_emojis(df))

        print("\n" + "="*60)
        print("PROCESSING STATS:")
        print("="*60)
        for k, v in get_processing_stats(df).items():
            print(f"  {k:<30}: {v}")

    except (FileNotFoundError, ValueError) as e:
        print(f"\nERROR: {e}")
        print("Usage: python src/data_processor.py <path_to_chat.txt>")