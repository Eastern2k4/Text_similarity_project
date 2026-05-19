from __future__ import annotations

from dataclasses import dataclass, field
from difflib import SequenceMatcher
from typing import Dict, Iterable

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.pipeline import FeatureUnion

from .text_preprocessing import clean_text


DEFAULT_WEIGHTS = {
    "word": 0.55,
    "char_wb": 0.30,
    "token_containment": 0.10,
    "sequence": 0.05,
}

DEFAULT_THRESHOLDS = {
    "related": 42.0,
    "very_similar": 72.0,
}


def build_tfidf_vectorizer(
    min_df=2,
    max_df=0.95,
    word_max_features=30000,
    char_max_features=50000,
):
    """
    Build the self-trained TF-IDF feature extractor.

    This is not a pretrained semantic model. It learns vocabulary and inverse
    document frequencies from the local questions dataset during training.
    """
    return FeatureUnion([
        ("word", TfidfVectorizer(
            analyzer="word",
            ngram_range=(1, 3),
            min_df=min_df,
            max_df=max_df,
            sublinear_tf=True,
            lowercase=False,
            stop_words="english",
            max_features=word_max_features,
        )),
        ("char_wb", TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(3, 6),
            min_df=min_df,
            max_df=max_df,
            sublinear_tf=True,
            lowercase=False,
            max_features=char_max_features,
        )),
    ])


def is_upgraded_vectorizer(vectorizer) -> bool:
    return (
        isinstance(vectorizer, FeatureUnion)
        and [name for name, _ in vectorizer.transformer_list] == ["word", "char_wb"]
    )


def normalize_weights(weights: Dict[str, float]) -> Dict[str, float]:
    total = sum(value for value in weights.values() if value > 0)
    if total <= 0:
        return DEFAULT_WEIGHTS.copy()
    return {name: value / total for name, value in weights.items() if value > 0}


def clamp_score(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def token_features(clean_text1: str, clean_text2: str) -> Dict[str, float]:
    tokens1 = set(clean_text1.split())
    tokens2 = set(clean_text2.split())

    if not tokens1 or not tokens2:
        return {
            "token_jaccard": 0.0,
            "token_containment": 0.0,
            "length_similarity": 0.0,
            "sequence": 0.0,
        }

    intersection = len(tokens1 & tokens2)
    union = len(tokens1 | tokens2)
    shorter = min(len(tokens1), len(tokens2))
    longer_text_length = max(len(clean_text1), len(clean_text2), 1)
    length_similarity = 1 - (abs(len(clean_text1) - len(clean_text2)) / longer_text_length)
    sequence = SequenceMatcher(None, clean_text1[:500], clean_text2[:500]).ratio()

    return {
        "token_jaccard": intersection / union if union else 0.0,
        "token_containment": intersection / shorter if shorter else 0.0,
        "length_similarity": clamp_score(length_similarity),
        "sequence": clamp_score(sequence),
    }


def rowwise_cosine(matrix1, matrix2) -> np.ndarray:
    dot = np.asarray(matrix1.multiply(matrix2).sum(axis=1)).ravel()
    norm1 = np.sqrt(np.asarray(matrix1.multiply(matrix1).sum(axis=1)).ravel())
    norm2 = np.sqrt(np.asarray(matrix2.multiply(matrix2).sum(axis=1)).ravel())
    denom = norm1 * norm2
    return np.divide(dot, denom, out=np.zeros_like(dot, dtype=float), where=denom != 0)


@dataclass
class TfidfSimilarityModel:
    vectorizer: FeatureUnion
    weights: Dict[str, float] = field(default_factory=lambda: DEFAULT_WEIGHTS.copy())
    thresholds: Dict[str, float] = field(default_factory=lambda: DEFAULT_THRESHOLDS.copy())
    metrics: Dict[str, float] = field(default_factory=dict)

    def __post_init__(self):
        self.weights = normalize_weights(self.weights)

    def feature_scores(self, text1: str, text2: str) -> Dict[str, float]:
        clean_text1 = clean_text(text1)
        clean_text2 = clean_text(text2)

        scores = token_features(clean_text1, clean_text2)
        texts = [clean_text1, clean_text2]

        for name, transformer in self.vectorizer.transformer_list:
            matrix = transformer.transform(texts)
            scores[name] = float(cosine_similarity(matrix[0:1], matrix[1:2])[0][0])

        combined = self.vectorizer.transform(texts)
        scores["combined"] = float(cosine_similarity(combined[0:1], combined[1:2])[0][0])
        return scores

    def score(self, text1: str, text2: str) -> float:
        features = self.feature_scores(text1, text2)
        weighted_score = sum(
            features.get(name, 0.0) * weight
            for name, weight in self.weights.items()
        )
        return clamp_score(weighted_score)

    def score_percent(self, text1: str, text2: str) -> float:
        return round(self.score(text1, text2) * 100, 2)

    def label_for_score(self, score_percent: float) -> str:
        if score_percent >= self.thresholds.get("very_similar", DEFAULT_THRESHOLDS["very_similar"]):
            return "Rất giống nhau"
        if score_percent >= self.thresholds.get("related", DEFAULT_THRESHOLDS["related"]):
            return "Có liên quan"
        return "Khác nhau"

    def severity_for_score(self, score_percent: float) -> str:
        if score_percent >= self.thresholds.get("very_similar", DEFAULT_THRESHOLDS["very_similar"]):
            return "high"
        if score_percent >= self.thresholds.get("related", DEFAULT_THRESHOLDS["related"]):
            return "medium"
        return "low"

    def predict(self, text1: str, text2: str) -> dict:
        score_percent = self.score_percent(text1, text2)
        return {
            "score": score_percent,
            "label": self.label_for_score(score_percent),
            "severity": self.severity_for_score(score_percent),
            "text1": text1,
            "text2": text2,
        }


def build_scores_from_features(features: Dict[str, np.ndarray], weights: Dict[str, float]) -> np.ndarray:
    normalized = normalize_weights(weights)
    score = np.zeros_like(next(iter(features.values())), dtype=float)
    for name, weight in normalized.items():
        if name in features:
            score += features[name] * weight
    return np.clip(score, 0.0, 1.0)


def evaluate_threshold(y_true: np.ndarray, scores: np.ndarray, threshold: float) -> Dict[str, float]:
    y_pred = scores >= threshold
    y_bool = y_true.astype(bool)

    tp = int(np.sum(y_pred & y_bool))
    fp = int(np.sum(y_pred & ~y_bool))
    fn = int(np.sum(~y_pred & y_bool))
    tn = int(np.sum(~y_pred & ~y_bool))

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    accuracy = (tp + tn) / len(y_true) if len(y_true) else 0.0

    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "true_positive": tp,
        "false_positive": fp,
        "false_negative": fn,
        "true_negative": tn,
    }


def tune_weights_and_threshold(
    features: Dict[str, np.ndarray],
    y_true: np.ndarray,
    candidate_weights: Iterable[Dict[str, float]] | None = None,
) -> tuple[Dict[str, float], float, Dict[str, float]]:
    if candidate_weights is None:
        candidate_weights = [
            {"combined": 1.0},
            {"word": 0.65, "char_wb": 0.25, "token_containment": 0.07, "sequence": 0.03},
            {"word": 0.55, "char_wb": 0.30, "token_containment": 0.10, "sequence": 0.05},
            {"word": 0.50, "char_wb": 0.35, "token_jaccard": 0.10, "sequence": 0.05},
            {"word": 0.45, "char_wb": 0.40, "token_containment": 0.10, "sequence": 0.05},
            {"word": 0.40, "char_wb": 0.45, "token_containment": 0.10, "length_similarity": 0.05},
        ]

    best_weights = DEFAULT_WEIGHTS.copy()
    best_threshold = DEFAULT_THRESHOLDS["related"] / 100
    best_metrics = {"f1": -1.0, "accuracy": -1.0, "precision": 0.0, "recall": 0.0}

    thresholds = np.linspace(0.05, 0.95, 91)
    for weights in candidate_weights:
        scores = build_scores_from_features(features, weights)
        for threshold in thresholds:
            metrics = evaluate_threshold(y_true, scores, threshold)
            rank = (
                metrics["f1"],
                metrics["accuracy"],
                metrics["precision"],
                metrics["recall"],
            )
            best_rank = (
                best_metrics["f1"],
                best_metrics["accuracy"],
                best_metrics["precision"],
                best_metrics["recall"],
            )
            if rank > best_rank:
                best_weights = normalize_weights(weights)
                best_threshold = float(threshold)
                best_metrics = metrics

    return best_weights, best_threshold, best_metrics


def choose_very_similar_threshold(y_true: np.ndarray, scores: np.ndarray, related_threshold: float) -> float:
    min_examples = max(20, int(len(y_true) * 0.005))
    best_threshold = max(related_threshold + 0.25, 0.70)

    for threshold in np.linspace(max(related_threshold, 0.50), 0.95, 46):
        metrics = evaluate_threshold(y_true, scores, float(threshold))
        predicted_positive = metrics["true_positive"] + metrics["false_positive"]
        if metrics["precision"] >= 0.85 and predicted_positive >= min_examples:
            return float(threshold)

    return float(min(best_threshold, 0.95))
