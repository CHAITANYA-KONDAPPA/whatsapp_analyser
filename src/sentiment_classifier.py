# =============================================================================
# sentiment_classifier.py
# -----------------------------------------------------------------------------
# PURPOSE : Train a Machine Learning model to predict sentiment on WhatsApp
#           messages. Uses high-confidence VADER labels from
#           sentiment_analyzer.py as training data.
#
#           Two models are trained and compared:
#               1. Multinomial Naive Bayes   — fast, strong baseline for text
#               2. Logistic Regression       — more accurate, handles nuance
#
#           The best model + its vectorizer are saved to /models/ as .pkl files
#           so pipeline.py and app.py can load and reuse them without
#           retraining every time.
#
# INPUT   : DataFrame with 'cleaned_message' and 'sentiment_label' columns
#           (output of sentiment_analyzer.analyze_sentiment())
#
# OUTPUT  :
#   Saved files:
#       models/sentiment_model.pkl      — best trained classifier
#       models/tfidf_vectorizer.pkl     — fitted TF-IDF vectorizer
#       models/model_metadata.json      — accuracy, F1, model name, date
#
#   Return values:
#       ClassifierResult dataclass      — evaluation metrics + predictions
#       predict_sentiment()             — predict on new text strings
#
# USAGE   :
#   from src.sentiment_classifier import train_classifier, predict_sentiment
#   result = train_classifier(df)
#   label  = predict_sentiment("I love this project!")
#
# AUTHOR  : Team — Chaitanya / Anudeep / Ram Teja
# =============================================================================

import os
import json
import logging
import pickle
import warnings
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Tuple

warnings.filterwarnings('ignore')

# Scikit-learn — all ML tools come from here
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes             import MultinomialNB
from sklearn.linear_model            import LogisticRegression
from sklearn.model_selection         import train_test_split, cross_val_score
from sklearn.metrics                 import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score
)
from sklearn.pipeline                import Pipeline
from sklearn.preprocessing           import LabelEncoder

# --------------------------------------------------------------------------- #
# Logging                                                                      #
# --------------------------------------------------------------------------- #
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# CONSTANTS                                                                    #
# --------------------------------------------------------------------------- #

# Where to save the trained model files
# Pipeline creates this folder if it doesn't exist
MODELS_DIR = 'models'

MODEL_PATH      = os.path.join(MODELS_DIR, 'sentiment_model.pkl')
VECTORIZER_PATH = os.path.join(MODELS_DIR, 'tfidf_vectorizer.pkl')
METADATA_PATH   = os.path.join(MODELS_DIR, 'model_metadata.json')

# Minimum messages needed to train a meaningful model
# Below this we can't reliably split 80/20 and get useful results
MIN_SAMPLES_REQUIRED = 30

# Train/test split — 80% train, 20% test
TEST_SIZE    = 0.20
RANDOM_STATE = 42   # fixed seed → reproducible results every time you run


# --------------------------------------------------------------------------- #
# RESULT CONTAINER                                                             #
# --------------------------------------------------------------------------- #

@dataclass
class ClassifierResult:
    """
    Bundles all outputs from train_classifier() into one clean object.

    Access like:
        result.best_model_name    → 'Logistic Regression'
        result.accuracy           → 0.847
        result.f1_score           → 0.831
        result.report             → full sklearn classification report string
        result.confusion_matrix   → 2D numpy array
        result.cross_val_scores   → [0.82, 0.85, 0.83, 0.86, 0.84]
    """
    best_model_name   : str             = ''
    accuracy          : float           = 0.0
    f1_score          : float           = 0.0
    report            : str             = ''
    confusion_matrix  : np.ndarray      = field(default_factory=lambda: np.array([]))
    cross_val_scores  : List[float]     = field(default_factory=list)
    cross_val_mean    : float           = 0.0
    cross_val_std     : float           = 0.0
    label_distribution: Dict[str, int]  = field(default_factory=dict)
    training_samples  : int             = 0
    test_samples      : int             = 0
    classes           : List[str]       = field(default_factory=list)
    model_saved_path  : str             = ''
    trained_at        : str             = ''


# --------------------------------------------------------------------------- #
# STEP 1 : Prepare training data                                              #
# --------------------------------------------------------------------------- #

def _prepare_data(df: pd.DataFrame) -> Tuple[List[str], List[str]]:
    """
    Extract and validate the training data from the DataFrame.

    Filters out:
        - Empty cleaned messages
        - Media / deleted messages (no real text)
        - Rows with missing labels

    Returns:
        texts  : list of cleaned message strings
        labels : list of sentiment label strings ('Positive'/'Negative'/'Neutral')

    WHY BALANCED DATA MATTERS:
        If 80% of messages are Positive, a dumb model can get 80% accuracy
        by always predicting "Positive" — without learning anything.
        get_high_confidence_samples() in sentiment_analyzer already
        returns a balanced set, so we just validate it here.
    """
    required = ['cleaned_message', 'sentiment_label']
    missing  = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing columns: {missing}\n"
            f"Run analyze_sentiment() first to generate sentiment_label column."
        )

    # Filter to usable rows
    valid_df = df[
        df['cleaned_message'].notna() &
        (df['cleaned_message'].str.strip() != '') &
        df['sentiment_label'].notna() &
        (df['sentiment_label'].isin(['Positive', 'Negative', 'Neutral']))
    ].copy()

    if len(valid_df) < MIN_SAMPLES_REQUIRED:
        raise ValueError(
            f"Only {len(valid_df)} valid training samples found. "
            f"Need at least {MIN_SAMPLES_REQUIRED}. "
            f"Your chat may be too short, or VADER found very few high-confidence labels."
        )

    texts  = valid_df['cleaned_message'].tolist()
    labels = valid_df['sentiment_label'].tolist()

    # Log class distribution
    label_counts = pd.Series(labels).value_counts()
    logger.info(f"  Training data prepared: {len(texts)} samples")
    for label, count in label_counts.items():
        logger.info(f"    {label:<12}: {count} ({count/len(texts)*100:.1f}%)")

    return texts, labels


# --------------------------------------------------------------------------- #
# STEP 2 : Build model pipelines                                              #
# --------------------------------------------------------------------------- #

def _build_pipelines() -> dict:
    """
    Creates two complete sklearn Pipeline objects.

    A Pipeline chains steps together so that:
        raw text → TF-IDF → classifier
    happens in one .fit() and .predict() call.

    WHY PIPELINE INSTEAD OF SEPARATE STEPS?
        When you save a Pipeline, you save the vectorizer AND classifier
        together in one object. This prevents the common mistake of saving
        the model but forgetting to save the vectorizer.

    TF-IDF settings explained:
        max_features=5000  : consider only top 5000 words (keeps it fast)
        ngram_range=(1,2)  : use single words AND 2-word phrases
                             "not good" as a bigram → better than just "good"
        min_df=2           : ignore words that appear in fewer than 2 messages
                             (probably typos or noise)
        sublinear_tf=True  : log-scale term frequency → prevents very
                             common words from dominating the score
    """
    tfidf_params = {
        'max_features' : 5000,
        'ngram_range'  : (1, 2),   # unigrams + bigrams
        'min_df'       : 2,
        'sublinear_tf' : True,
    }

    pipelines = {
        # ── Model 1: Multinomial Naive Bayes ─────────────────────────────── #
        # WHY: Naive Bayes is the classic text classification algorithm.
        # It assumes each word contributes independently to sentiment.
        # Fast to train, works surprisingly well even with small datasets.
        # alpha=0.1 → Laplace smoothing (handles words not seen in training)
        'Naive Bayes': Pipeline([
            ('tfidf',       TfidfVectorizer(**tfidf_params)),
            ('classifier',  MultinomialNB(alpha=0.1)),
        ]),

        # ── Model 2: Logistic Regression ─────────────────────────────────── #
        # WHY: Logistic Regression finds a decision boundary between classes.
        # It considers word combinations and is generally more accurate
        # than Naive Bayes on longer text.
        # C=1.0       → regularisation strength (prevents overfitting)
        # max_iter=500 → enough iterations to converge
        # class_weight='balanced' → handles slight class imbalance
        'Logistic Regression': Pipeline([
            ('tfidf',       TfidfVectorizer(**tfidf_params)),
            ('classifier',  LogisticRegression(
                C=1.0,
                max_iter=500,
                random_state=RANDOM_STATE,
                class_weight='balanced',
                solver='lbfgs',
                multi_class='multinomial',
            )),
        ]),
    }

    return pipelines


# --------------------------------------------------------------------------- #
# STEP 3 : Train and evaluate both models                                     #
# --------------------------------------------------------------------------- #

def _train_and_evaluate(
    pipelines : dict,
    X_train   : List[str],
    X_test    : List[str],
    y_train   : List[str],
    y_test    : List[str],
    X_all     : List[str],
    y_all     : List[str],
) -> Tuple[str, object, ClassifierResult]:
    """
    Trains both pipelines, evaluates them, and returns the best one.

    Evaluation metrics:
        Accuracy  : overall % correct
        F1-Score  : weighted average of precision + recall
                    (best single metric for multi-class classification)
        Cross-Val : 5-fold cross validation — trains/tests on 5 different
                    splits to get a more reliable performance estimate

    The model with the higher weighted F1-Score wins.

    Returns: (best_model_name, best_pipeline, ClassifierResult)
    """
    best_name     = None
    best_pipeline = None
    best_f1       = -1
    results       = {}

    for name, pipeline in pipelines.items():
        logger.info(f"  Training {name}...")

        # Train
        pipeline.fit(X_train, y_train)

        # Predict on test set
        y_pred   = pipeline.predict(X_test)

        # Metrics
        accuracy = round(accuracy_score(y_test, y_pred), 4)
        f1       = round(f1_score(y_test, y_pred, average='weighted'), 4)
        report   = classification_report(y_test, y_pred,
                                         target_names=['Negative', 'Neutral', 'Positive'])
        cm       = confusion_matrix(y_test, y_pred,
                                    labels=['Negative', 'Neutral', 'Positive'])

        # 5-Fold Cross Validation on full dataset
        # This gives a more honest estimate than a single train/test split
        cv_scores = cross_val_score(pipeline, X_all, y_all,
                                    cv=5, scoring='f1_weighted')

        logger.info(f"    Accuracy : {accuracy:.4f}")
        logger.info(f"    F1 Score : {f1:.4f}")
        logger.info(f"    CV Mean  : {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

        results[name] = {
            'accuracy'    : accuracy,
            'f1'          : f1,
            'report'      : report,
            'cm'          : cm,
            'cv_scores'   : cv_scores,
        }

        # Keep track of winner
        if f1 > best_f1:
            best_f1       = f1
            best_name     = name
            best_pipeline = pipeline

    logger.info(f"  Best model: {best_name} (F1: {best_f1:.4f})")

    # Build result object from best model's stats
    r = results[best_name]
    result = ClassifierResult(
        best_model_name  = best_name,
        accuracy         = r['accuracy'],
        f1_score         = r['f1'],
        report           = r['report'],
        confusion_matrix = r['cm'],
        cross_val_scores = r['cv_scores'].tolist(),
        cross_val_mean   = round(float(r['cv_scores'].mean()), 4),
        cross_val_std    = round(float(r['cv_scores'].std()), 4),
        training_samples = len(X_train),
        test_samples     = len(X_test),
        classes          = ['Negative', 'Neutral', 'Positive'],
        trained_at       = datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    )

    return best_name, best_pipeline, result


# --------------------------------------------------------------------------- #
# STEP 4 : Save model to disk                                                 #
# --------------------------------------------------------------------------- #

def _save_model(pipeline: object, result: ClassifierResult) -> str:
    """
    Saves the trained Pipeline (vectorizer + classifier together) to disk.

    WHY SAVE AS A PIPELINE?
        The TF-IDF vectorizer learns the vocabulary during .fit().
        If you save only the classifier and load a fresh vectorizer later,
        the word indices will be different → completely wrong predictions.
        Saving the full Pipeline saves both, locked together.

    Also saves model_metadata.json with performance stats so app.py
    can display "Model accuracy: 84.7%" on the dashboard without
    having to reload the entire model.

    Returns: path where model was saved
    """
    os.makedirs(MODELS_DIR, exist_ok=True)

    # Save full pipeline (vectorizer + classifier)
    with open(MODEL_PATH, 'wb') as f:
        pickle.dump(pipeline, f)

    # Save metadata as JSON (human-readable)
    metadata = {
        'model_name'      : result.best_model_name,
        'accuracy'        : result.accuracy,
        'f1_score'        : result.f1_score,
        'cross_val_mean'  : result.cross_val_mean,
        'cross_val_std'   : result.cross_val_std,
        'training_samples': result.training_samples,
        'test_samples'    : result.test_samples,
        'classes'         : result.classes,
        'trained_at'      : result.trained_at,
    }
    with open(METADATA_PATH, 'w') as f:
        json.dump(metadata, f, indent=2)

    logger.info(f"  Model saved   → {MODEL_PATH}")
    logger.info(f"  Metadata saved→ {METADATA_PATH}")

    return MODEL_PATH


# --------------------------------------------------------------------------- #
# MAIN PUBLIC FUNCTION 1 : train_classifier()                                 #
# --------------------------------------------------------------------------- #

def train_classifier(df: pd.DataFrame) -> ClassifierResult:
    """
    Master training function — call this from pipeline.py.

    Full flow:
        1. Extract and validate training data from DataFrame
        2. Split into 80% train / 20% test
        3. Build Naive Bayes and Logistic Regression pipelines
        4. Train both, evaluate, pick the best
        5. Save best model + vectorizer + metadata to /models/
        6. Return ClassifierResult with all metrics

    Parameters:
        df (pd.DataFrame): Must have 'cleaned_message' and 'sentiment_label'.
                           Best to pass the output of
                           get_high_confidence_samples() from sentiment_analyzer
                           for cleanest training labels.

    Returns:
        ClassifierResult dataclass with all performance metrics
    """
    logger.info("="*50)
    logger.info("Starting ML classifier training...")
    logger.info("="*50)

    # ── Step 1: Prepare data ─────────────────────────────────────────────── #
    logger.info("Step 1/4 — Preparing training data...")
    texts, labels = _prepare_data(df)

    # ── Step 2: Train / test split ───────────────────────────────────────── #
    logger.info("Step 2/4 — Splitting 80% train / 20% test...")
    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels,
        test_size    = TEST_SIZE,
        random_state = RANDOM_STATE,
        stratify     = labels,  # ensures same class ratio in both splits
    )
    logger.info(f"  Train: {len(X_train)} samples | Test: {len(X_test)} samples")

    # ── Step 3: Build pipelines ──────────────────────────────────────────── #
    logger.info("Step 3/4 — Building model pipelines...")
    pipelines = _build_pipelines()

    # ── Step 4: Train, evaluate, pick best ───────────────────────────────── #
    logger.info("Step 4/4 — Training and evaluating models...")
    best_name, best_pipeline, result = _train_and_evaluate(
        pipelines, X_train, X_test, y_train, y_test, texts, labels
    )

    # Label distribution
    result.label_distribution = pd.Series(labels).value_counts().to_dict()

    # ── Save best model ───────────────────────────────────────────────────── #
    saved_path = _save_model(best_pipeline, result)
    result.model_saved_path = saved_path

    logger.info("="*50)
    logger.info("Training complete!")
    logger.info(f"  Best model : {result.best_model_name}")
    logger.info(f"  Accuracy   : {result.accuracy:.4f}")
    logger.info(f"  F1 Score   : {result.f1_score:.4f}")
    logger.info(f"  CV Mean    : {result.cross_val_mean:.4f} "
                f"(+/- {result.cross_val_std:.4f})")
    logger.info("="*50)

    return result


# --------------------------------------------------------------------------- #
# MAIN PUBLIC FUNCTION 2 : predict_sentiment()                                #
# --------------------------------------------------------------------------- #

def predict_sentiment(text: str) -> dict:
    """
    Predict the sentiment of a single new text string.
    Loads the saved model from disk and returns prediction + confidence.

    This is what app.py calls when a user types a message and asks
    "what is the sentiment of this?"

    Parameters:
        text (str): Any text string (will be preprocessed internally)

    Returns:
        {
            'label'       : 'Positive',
            'confidence'  : 0.91,
            'probabilities': {
                'Positive': 0.91,
                'Neutral' : 0.07,
                'Negative': 0.02
            }
        }

    Raises:
        FileNotFoundError if model hasn't been trained yet
    """
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"No trained model found at {MODEL_PATH}.\n"
            f"Run train_classifier(df) first to train and save the model."
        )

    # Load saved pipeline
    with open(MODEL_PATH, 'rb') as f:
        pipeline = pickle.load(f)

    if not text or not text.strip():
        return {
            'label'         : 'Neutral',
            'confidence'    : 1.0,
            'probabilities' : {'Positive': 0.0, 'Neutral': 1.0, 'Negative': 0.0}
        }

    # Predict label
    label = pipeline.predict([text])[0]

    # Get probability for each class
    proba = pipeline.predict_proba([text])[0]
    classes = pipeline.classes_

    probabilities = {
        cls: round(float(prob), 4)
        for cls, prob in zip(classes, proba)
    }

    confidence = round(float(max(proba)), 4)

    return {
        'label'         : label,
        'confidence'    : confidence,
        'probabilities' : probabilities,
    }


# --------------------------------------------------------------------------- #
# PUBLIC FUNCTION 3 : predict_bulk()                                          #
# --------------------------------------------------------------------------- #

def predict_bulk(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds ML-predicted sentiment to the full DataFrame.
    Used by pipeline.py after model is trained to add a second
    opinion alongside VADER's sentiment_label.

    Adds column:
        predicted_sentiment : ML model's label ('Positive'/'Negative'/'Neutral')

    This lets visualizer.py compare VADER vs ML predictions — a great
    insight to show on the dashboard.

    Parameters:
        df (pd.DataFrame): Must have 'cleaned_message' column

    Returns:
        DataFrame with 'predicted_sentiment' column added
    """
    if not os.path.exists(MODEL_PATH):
        logger.warning("No saved model found. Skipping bulk prediction.")
        df['predicted_sentiment'] = df.get('sentiment_label', 'Neutral')
        return df

    with open(MODEL_PATH, 'rb') as f:
        pipeline = pickle.load(f)

    df = df.copy()

    # Only predict on real text messages
    mask = (
        (~df['is_media']) &
        (~df['is_deleted']) &
        (df['cleaned_message'].str.strip() != '')
    )

    predictions = pd.Series('Neutral', index=df.index)
    if mask.sum() > 0:
        preds = pipeline.predict(df.loc[mask, 'cleaned_message'].tolist())
        predictions[mask] = preds

    df['predicted_sentiment'] = predictions

    # Agreement rate between VADER and ML model
    if 'sentiment_label' in df.columns:
        agree_mask  = df.loc[mask, 'sentiment_label'] == df.loc[mask, 'predicted_sentiment']
        agree_rate  = round(agree_mask.mean() * 100, 1)
        logger.info(f"VADER vs ML agreement rate: {agree_rate}%")

    return df


# --------------------------------------------------------------------------- #
# PUBLIC FUNCTION 4 : load_model_metadata()                                   #
# --------------------------------------------------------------------------- #

def load_model_metadata() -> dict:
    """
    Load saved model metadata without loading the full model.
    Used by app.py to display model stats on the dashboard quickly.

    Returns dict from model_metadata.json, or empty dict if not trained yet.
    """
    if not os.path.exists(METADATA_PATH):
        return {}

    with open(METADATA_PATH, 'r') as f:
        return json.load(f)


# --------------------------------------------------------------------------- #
# PUBLIC FUNCTION 5 : is_model_trained()                                      #
# --------------------------------------------------------------------------- #

def is_model_trained() -> bool:
    """
    Check if a saved model already exists.
    Used by pipeline.py to decide whether to retrain or load existing model.
    """
    return os.path.exists(MODEL_PATH)


# --------------------------------------------------------------------------- #
# QUICK TEST — python src/sentiment_classifier.py <chat.txt>                  #
# --------------------------------------------------------------------------- #

if __name__ == '__main__':
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    from src.whatsapp_parser    import parse_chat
    from src.data_processor     import process_dataframe
    from src.sentiment_analyzer import analyze_sentiment, get_high_confidence_samples

    path = sys.argv[1] if len(sys.argv) > 1 else 'data/sample_chats/chat.txt'

    try:
        print("\nStep 1: Parsing...")
        df = parse_chat(path)

        print("Step 2: Processing...")
        df = process_dataframe(df)

        print("Step 3: Sentiment analysis (VADER)...")
        df = analyze_sentiment(df)

        print("Step 4: Getting high-confidence training samples...")
        training_df = get_high_confidence_samples(df, n=200)
        print(f"  Training samples: {len(training_df)}")
        print(f"  Distribution:\n{training_df['sentiment_label'].value_counts()}")

        print("\nStep 5: Training ML classifier...")
        result = train_classifier(training_df)

        print("\n" + "="*60)
        print("TRAINING RESULTS:")
        print("="*60)
        print(f"  Best Model       : {result.best_model_name}")
        print(f"  Accuracy         : {result.accuracy:.4f} "
              f"({result.accuracy*100:.1f}%)")
        print(f"  F1 Score         : {result.f1_score:.4f}")
        print(f"  Cross-Val Mean   : {result.cross_val_mean:.4f} "
              f"(+/- {result.cross_val_std:.4f})")
        print(f"  Training Samples : {result.training_samples}")
        print(f"  Test Samples     : {result.test_samples}")
        print(f"  Trained At       : {result.trained_at}")
        print(f"  Model Saved To   : {result.model_saved_path}")

        print("\n" + "="*60)
        print("CLASSIFICATION REPORT:")
        print("="*60)
        print(result.report)

        print("\n" + "="*60)
        print("CONFUSION MATRIX (Negative | Neutral | Positive):")
        print("="*60)
        print(result.confusion_matrix)

        print("\n" + "="*60)
        print("LIVE PREDICTION TEST:")
        print("="*60)
        test_messages = [
            "I love this project so much!!",
            "This is terrible, I hate it",
            "Meeting at 5pm today",
            "Not bad, could be better",
            "AMAZING work everyone!! 🔥🔥",
            "why is nobody responding??",
        ]
        for msg in test_messages:
            pred = predict_sentiment(msg)
            print(f"\n  Input       : {msg}")
            print(f"  Label       : {pred['label']}")
            print(f"  Confidence  : {pred['confidence']:.2%}")
            print(f"  Probs       : {pred['probabilities']}")

        print("\n" + "="*60)
        print("BULK PREDICTION ON FULL CHAT:")
        print("="*60)
        df = predict_bulk(df)
        print(df[['message', 'sentiment_label', 'predicted_sentiment']].head(8).to_string(index=False))

    except (FileNotFoundError, ValueError) as e:
        print(f"\nERROR: {e}")
        print("Usage: python src/sentiment_classifier.py <path_to_chat.txt>")