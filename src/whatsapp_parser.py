# =============================================================================
# whatsapp_parser.py
# -----------------------------------------------------------------------------
# PURPOSE : Read a WhatsApp exported .txt chat file and convert it into a
#           clean pandas DataFrame with columns:
#               date | time | sender | message | is_media | is_deleted
#
# WHY     : Every other module (sentiment, visualizer, etc.) depends on this
#           DataFrame. If the parser is wrong, everything downstream is wrong.
#
# AUTHOR  : Team — Chaitanya / Anudeep / Ram Teja
# =============================================================================

import re
import pandas as pd
from datetime import datetime
import logging
import os

# --------------------------------------------------------------------------- #
# Logging setup — prints info/warnings to console while running               #
# --------------------------------------------------------------------------- #
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# CONSTANTS                                                                    #
# --------------------------------------------------------------------------- #

# WhatsApp exports dates in different formats depending on phone OS / region.
# We support both DD/MM/YYYY and MM/DD/YYYY, 12-hr and 24-hr clocks.
# Each tuple is (regex_pattern, datetime_format_string)

DATE_TIME_PATTERNS = [
    # Android 12-hr  →  23/01/2024, 10:45 am
    (
        r'(\d{1,2}/\d{1,2}/\d{4}),\s(\d{1,2}:\d{2}\s?[aApP][mM])\s-\s',
        ['%d/%m/%Y', '%I:%M\u202f%p', '%I:%M %p']
    ),
    # Android 24-hr  →  23/01/2024, 22:45 -
    (
        r'(\d{1,2}/\d{1,2}/\d{4}),\s(\d{1,2}:\d{2})\s-\s',
        ['%d/%m/%Y', '%H:%M']
    ),
    # iOS 12-hr      →  [23/01/2024, 10:45:30 AM]
    (
        r'\[(\d{1,2}/\d{1,2}/\d{4}),\s(\d{1,2}:\d{2}:\d{2}\s?[aApP][mM])\]',
        ['%d/%m/%Y', '%I:%M:%S\u202f%p', '%I:%M:%S %p']
    ),
    # US format      →  1/23/2024, 10:45 AM
    (
        r'(\d{1,2}/\d{1,2}/\d{4}),\s(\d{1,2}:\d{2}\s?[aApP][mM])\s-\s',
        ['%m/%d/%Y', '%I:%M\u202f%p', '%I:%M %p']
    ),
]

# Messages that mean no actual text was sent
MEDIA_STRINGS = [
    '<media omitted>',
    'image omitted',
    'video omitted',
    'audio omitted',
    'document omitted',
    'sticker omitted',
    'gif omitted',
]

# Messages that mean the sender deleted it
DELETED_STRINGS = [
    'this message was deleted',
    'you deleted this message',
]


# --------------------------------------------------------------------------- #
# HELPER : try to parse a datetime string using multiple format strings        #
# --------------------------------------------------------------------------- #

def _parse_datetime(date_str: str, time_str: str, date_fmts: list) -> datetime | None:
    """
    Try to build a datetime object from date_str + time_str.
    We try multiple time formats because of the narrow no-break space (\\u202f)
    that some systems insert between the time and AM/PM.

    Returns a datetime object on success, or None if all formats fail.
    """
    # Normalise: replace narrow no-break space with regular space
    time_str = time_str.replace('\u202f', ' ').strip()
    date_str = date_str.strip()

    date_fmt = date_fmts[0]
    time_fmts = date_fmts[1:]  # one or more possible time formats

    for time_fmt in time_fmts:
        try:
            return datetime.strptime(f"{date_str} {time_str}", f"{date_fmt} {time_fmt}")
        except ValueError:
            continue
    return None


# --------------------------------------------------------------------------- #
# CORE FUNCTION 1 : Detect which regex pattern matches this chat file          #
# --------------------------------------------------------------------------- #

def _detect_pattern(raw_text: str) -> tuple | None:
    """
    WhatsApp formats dates differently on Android vs iOS and by region.
    We test the first 20 lines to find which pattern produces the most matches.

    Returns the (pattern, formats) tuple that matched best, or None.
    """
    sample = '\n'.join(raw_text.splitlines()[:30])

    best_pattern = None
    best_count = 0

    for pattern, fmts in DATE_TIME_PATTERNS:
        count = len(re.findall(pattern, sample))
        if count > best_count:
            best_count = count
            best_pattern = (pattern, fmts)

    if best_count == 0:
        logger.warning("Could not detect a known WhatsApp date format in this file.")
        return None

    logger.info(f"Detected format pattern with {best_count} matches in sample.")
    return best_pattern


# --------------------------------------------------------------------------- #
# CORE FUNCTION 2 : Split raw text into a list of raw message dicts            #
# --------------------------------------------------------------------------- #

def _split_messages(raw_text: str, pattern: str) -> list[dict]:
    """
    Split the entire chat file into individual messages.

    WHY THIS IS TRICKY:
        WhatsApp messages can span multiple lines. The next message only starts
        when a new timestamp appears. So we find all timestamp positions first,
        then extract everything between consecutive timestamps as one message.

    Returns a list of dicts: {'date': str, 'time': str, 'raw_body': str}
    """
    # Find every timestamp position in the file
    matches = list(re.finditer(pattern, raw_text))

    if not matches:
        logger.error("No messages found. Check that the file is a valid WhatsApp export.")
        return []

    messages = []
    for i, match in enumerate(matches):
        date_str = match.group(1)
        time_str = match.group(2)

        # Everything after this timestamp until the next timestamp = message body
        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(raw_text)
        raw_body = raw_text[start:end].strip()

        messages.append({
            'date': date_str,
            'time': time_str,
            'raw_body': raw_body
        })

    logger.info(f"Found {len(messages)} raw message blocks.")
    return messages


# --------------------------------------------------------------------------- #
# CORE FUNCTION 3 : Parse sender and message text from raw_body                #
# --------------------------------------------------------------------------- #

def _parse_body(raw_body: str) -> tuple[str, str]:
    """
    Each message body looks like:
        'Chaitanya: Hello everyone!'
    or for system messages (someone joined, left, etc.):
        'Chaitanya joined using this group's invite link'

    Returns (sender, message_text).
    If no colon is found, it's a system message → sender = 'System'.
    """
    if ': ' in raw_body:
        # Split only on the FIRST colon (sender names can't have colons,
        # but message text can — e.g. "sure: let's do it")
        sender, _, message = raw_body.partition(': ')
        return sender.strip(), message.strip()
    else:
        return 'System', raw_body.strip()


# --------------------------------------------------------------------------- #
# CORE FUNCTION 4 : Classify message type                                      #
# --------------------------------------------------------------------------- #

def _classify_message(message: str) -> tuple[bool, bool]:
    """
    Check if a message is a media placeholder or a deleted message.

    Returns (is_media: bool, is_deleted: bool)
    """
    msg_lower = message.lower().strip()
    is_media = any(s in msg_lower for s in MEDIA_STRINGS)
    is_deleted = any(s in msg_lower for s in DELETED_STRINGS)
    return is_media, is_deleted


# --------------------------------------------------------------------------- #
# MAIN PUBLIC FUNCTION : parse_chat()                                          #
# --------------------------------------------------------------------------- #

def parse_chat(file_path: str) -> pd.DataFrame:
    """
    Master function — call this from pipeline.py or main.py.

    Steps:
        1. Read the .txt file
        2. Auto-detect the date/time format
        3. Split into individual messages (handling multi-line messages)
        4. Parse sender and message text
        5. Classify media and deleted messages
        6. Build and return a clean DataFrame

    Parameters:
        file_path (str): Path to the exported WhatsApp .txt file

    Returns:
        pd.DataFrame with columns:
            - datetime   : Python datetime object
            - date       : date only (2024-01-23)
            - time       : time only (10:45:00)
            - year       : int
            - month      : int  (1-12)
            - month_name : str  ('January')
            - day        : int  (1-31)
            - day_name   : str  ('Monday')
            - hour       : int  (0-23)
            - sender     : str  ('Chaitanya')
            - message    : str  (cleaned message text)
            - is_media   : bool
            - is_deleted : bool
            - word_count : int
            - char_count : int
    """

    # ── 1. Read file ──────────────────────────────────────────────────────── #
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Chat file not found: {file_path}")

    logger.info(f"Reading file: {file_path}")

    # Try UTF-8 first (standard), fall back to latin-1 for older exports
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            raw_text = f.read()
    except UnicodeDecodeError:
        logger.warning("UTF-8 failed, retrying with latin-1 encoding.")
        with open(file_path, 'r', encoding='latin-1') as f:
            raw_text = f.read()

    # ── 2. Detect format ──────────────────────────────────────────────────── #
    detected = _detect_pattern(raw_text)
    if detected is None:
        raise ValueError("Unsupported WhatsApp export format. Please export as .txt without media.")

    pattern, date_fmts = detected

    # ── 3. Split into messages ────────────────────────────────────────────── #
    raw_messages = _split_messages(raw_text, pattern)
    if not raw_messages:
        raise ValueError("No messages could be extracted from this file.")

    # ── 4 & 5. Parse each message ─────────────────────────────────────────── #
    records = []
    skipped = 0

    for msg in raw_messages:
        # Parse datetime
        dt = _parse_datetime(msg['date'], msg['time'], date_fmts)
        if dt is None:
            skipped += 1
            continue  # skip lines we can't parse a timestamp for

        # Parse sender and text
        sender, message_text = _parse_body(msg['raw_body'])

        # Skip pure system messages (group name changes, join/leave notifications)
        if sender == 'System':
            continue

        # Classify
        is_media, is_deleted = _classify_message(message_text)

        records.append({
            'datetime'  : dt,
            'date'      : dt.date(),
            'time'      : dt.time(),
            'year'      : dt.year,
            'month'     : dt.month,
            'month_name': dt.strftime('%B'),      # 'January', 'February', ...
            'day'       : dt.day,
            'day_name'  : dt.strftime('%A'),      # 'Monday', 'Tuesday', ...
            'hour'      : dt.hour,
            'sender'    : sender,
            'message'   : message_text,
            'is_media'  : is_media,
            'is_deleted': is_deleted,
            'word_count': len(message_text.split()) if not is_media else 0,
            'char_count': len(message_text) if not is_media else 0,
        })

    if skipped > 0:
        logger.warning(f"Skipped {skipped} lines due to unrecognised timestamp format.")

    # ── 6. Build DataFrame ────────────────────────────────────────────────── #
    df = pd.DataFrame(records)

    if df.empty:
        raise ValueError("DataFrame is empty after parsing. Check your chat file.")

    # Sort chronologically
    df = df.sort_values('datetime').reset_index(drop=True)

    logger.info(f"Successfully parsed {len(df)} messages from {df['sender'].nunique()} senders.")
    logger.info(f"Date range: {df['date'].min()} → {df['date'].max()}")

    return df


# --------------------------------------------------------------------------- #
# UTILITY : get_chat_summary()                                                 #
# --------------------------------------------------------------------------- #

def get_chat_summary(df: pd.DataFrame) -> dict:
    """
    Returns a quick summary dictionary about the parsed chat.
    Useful for displaying stats on the dashboard.

    Call this after parse_chat().
    """
    text_only = df[~df['is_media'] & ~df['is_deleted']]

    summary = {
        'total_messages'   : len(df),
        'total_text_msgs'  : len(text_only),
        'total_media_msgs' : df['is_media'].sum(),
        'total_deleted'    : df['is_deleted'].sum(),
        'total_senders'    : df['sender'].nunique(),
        'senders'          : df['sender'].unique().tolist(),
        'date_start'       : str(df['date'].min()),
        'date_end'         : str(df['date'].max()),
        'total_days'       : (df['date'].max() - df['date'].min()).days,
        'total_words'      : int(df['word_count'].sum()),
        'avg_words_per_msg': round(text_only['word_count'].mean(), 2),
        'most_active_user' : df['sender'].value_counts().idxmax(),
    }
    return summary


# --------------------------------------------------------------------------- #
# QUICK TEST — run this file directly to test: python whatsapp_parser.py       #
# --------------------------------------------------------------------------- #

if __name__ == '__main__':
    import sys

    # If you pass a file path as argument: python whatsapp_parser.py chat.txt
    # Otherwise it looks for a default sample file
    if len(sys.argv) > 1:
        path = sys.argv[1]
    else:
        path = 'data/sample_chats/chat.txt'

    try:
        df = parse_chat(path)

        print("\n" + "="*60)
        print("PARSED DATAFRAME — First 5 rows:")
        print("="*60)
        print(df.head())

        print("\n" + "="*60)
        print("DATAFRAME INFO:")
        print("="*60)
        print(df.dtypes)

        print("\n" + "="*60)
        print("CHAT SUMMARY:")
        print("="*60)
        summary = get_chat_summary(df)
        for key, value in summary.items():
            print(f"  {key:<22}: {value}")

    except (FileNotFoundError, ValueError) as e:
        print(f"\nERROR: {e}")
        print("Usage: python whatsapp_parser.py <path_to_chat.txt>")
        