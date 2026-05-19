import pickle
from pathlib import Path

from .tfidf_similarity_model import (
    TfidfSimilarityModel,
    build_tfidf_vectorizer,
    is_upgraded_vectorizer,
)


MODEL_PATH = Path(__file__).resolve().parent.parent.parent / "ml" / "artifacts" / "tfidf_model.pkl"

_similarity_model = None


def get_similarity_model():
    """Load and cache the self-trained similarity model."""
    global _similarity_model

    if _similarity_model is not None:
        return _similarity_model

    if MODEL_PATH.exists():
        try:
            with open(MODEL_PATH, "rb") as file:
                loaded_model = pickle.load(file)

            if isinstance(loaded_model, TfidfSimilarityModel):
                _similarity_model = loaded_model
                print("Loaded self-trained TF-IDF similarity model from file")
                return _similarity_model

            if is_upgraded_vectorizer(loaded_model):
                _similarity_model = TfidfSimilarityModel(
                    vectorizer=loaded_model,
                    weights={"combined": 1.0},
                )
                print("Loaded legacy TF-IDF vectorizer. Retrain to enable calibrated scoring.")
                return _similarity_model

            print("Existing TF-IDF artifact is old. Please retrain it.")
        except Exception as exc:
            print(f"Failed to load model: {exc}")

    raise RuntimeError(
        f"Trained model not found at {MODEL_PATH}. "
        "Run: python ml\\training\\train_model.py"
    )


def calculate_similarity(text1: str, text2: str) -> dict:
    """
    Calculate similarity using the self-trained TF-IDF model.

    The model is trained from the local dataset. It does not fit on user input
    during prediction because that makes scores unstable between requests.
    """
    model = get_similarity_model()
    return model.predict(text1, text2)
