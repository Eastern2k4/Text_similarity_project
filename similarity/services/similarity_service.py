from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.pipeline import FeatureUnion
import pickle
from pathlib import Path

from .text_preprocessing import clean_text

# Path to save/load trained model
MODEL_PATH = Path(__file__).resolve().parent.parent.parent / "ml" / "artifacts" / "tfidf_model.pkl"

# Global variable to cache the vectorizer
_vectorizer = None


def build_tfidf_vectorizer(min_df=2, max_df=0.95):
    """
    Build a stronger TF-IDF model by combining word and character features.

    Word n-grams help with phrase-level overlap. Character n-grams help with
    typos, spelling variants, and short lexical overlap.
    """
    return FeatureUnion([
        ("word", TfidfVectorizer(
            analyzer="word",
            ngram_range=(1, 2),
            min_df=min_df,
            max_df=max_df,
            sublinear_tf=True,
            lowercase=False,
            stop_words="english",
            max_features=20000,
        )),
        ("char_wb", TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(3, 5),
            min_df=min_df,
            max_df=max_df,
            sublinear_tf=True,
            lowercase=False,
            max_features=30000,
        )),
    ])


def is_upgraded_vectorizer(vectorizer) -> bool:
    return (
        isinstance(vectorizer, FeatureUnion)
        and [name for name, _ in vectorizer.transformer_list] == ["word", "char_wb"]
    )


def get_vectorizer():
    """Load trained vectorizer or create new one"""
    global _vectorizer
    
    if _vectorizer is not None:
        return _vectorizer
    
    # Try to load pre-trained model
    if MODEL_PATH.exists():
        try:
            with open(MODEL_PATH, 'rb') as f:
                loaded_vectorizer = pickle.load(f)
                if is_upgraded_vectorizer(loaded_vectorizer):
                    _vectorizer = loaded_vectorizer
                    print("Loaded upgraded TF-IDF model from file")
                    return _vectorizer

                print("Existing TF-IDF model is old. Please retrain it.")
        except Exception as e:
            print(f"Failed to load model: {e}")
    
    # Create new vectorizer if no trained model found
    print("No upgraded trained model found. Creating fallback vectorizer...")
    _vectorizer = build_tfidf_vectorizer(min_df=1, max_df=1.0)
    
    return _vectorizer


def get_similarity_label(score: float) -> str:
    if score >= 70:
        return "Rất giống nhau"
    elif score >= 40:
        return "Tương đối giống nhau"
    else:
        return "Khác nhau"


def calculate_similarity(text1: str, text2: str) -> dict:
    """
    Calculate similarity using pre-trained TF-IDF model.
    
    Uses trained vectorizer to transform texts instead of fitting,
    which avoids parameter conflicts with small document sets.
    """
    # Clean texts
    text1_clean = clean_text(text1)
    text2_clean = clean_text(text2)
    
    # Get vectorizer
    vectorizer = get_vectorizer()
    
    # Transform texts using pre-trained vectorizer
    texts = [text1_clean, text2_clean]
    
    try:
        # Use transform (works with trained model)
        tfidf_matrix = vectorizer.transform(texts)
    except ValueError:
        # If transform fails (model not fitted), fit_transform instead
        print("⚠ Vectorizer not fitted, fitting now...")
        tfidf_matrix = vectorizer.fit_transform(texts)
    
    # Calculate cosine similarity
    similarity_matrix = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])
    score = similarity_matrix[0][0]
    
    # Convert to percentage
    score_percent = round(score * 100, 2)
    
    return {
        "score": score_percent,
        "label": get_similarity_label(score_percent),
        "text1": text1,
        "text2": text2,
    }
