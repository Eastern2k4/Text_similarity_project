"""
Train the project-owned TF-IDF similarity model.

The model is not pretrained. It learns TF-IDF vocabulary from
ml/datasets/questions.csv, then uses the dataset's is_duplicate labels to
calibrate feature weights and decision thresholds.
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from similarity.services.text_preprocessing import clean_text
from similarity.services.tfidf_similarity_model import (
    TfidfSimilarityModel,
    build_scores_from_features,
    build_tfidf_vectorizer,
    choose_very_similar_threshold,
    rowwise_cosine,
    token_features,
    tune_weights_and_threshold,
)


REQUIRED_COLUMNS = {"question1", "question2"}
LABEL_COLUMN = "is_duplicate"
RANDOM_STATE = 42


def default_dataset_path() -> Path:
    return Path(__file__).resolve().parent.parent / "datasets" / "questions.csv"


def default_model_path() -> Path:
    return Path(__file__).resolve().parent.parent / "artifacts" / "tfidf_model.pkl"


def clean_question_series(series: pd.Series) -> pd.Series:
    return series.fillna("").astype(str).map(clean_text)


def load_and_prepare_dataset(dataset_path: Path, max_rows: int | None = None) -> pd.DataFrame:
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found at {dataset_path}")

    read_kwargs = {}
    if max_rows:
        read_kwargs["nrows"] = max_rows

    df = pd.read_csv(dataset_path, **read_kwargs)
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing)}")

    df["question1_clean"] = clean_question_series(df["question1"])
    df["question2_clean"] = clean_question_series(df["question2"])

    before = len(df)
    df = df[(df["question1_clean"] != "") & (df["question2_clean"] != "")].copy()
    removed = before - len(df)
    if removed:
        print(f"Removed {removed:,} rows with empty text after cleaning")

    if df.empty:
        raise ValueError("No valid question pairs remain after cleaning")

    if LABEL_COLUMN in df.columns:
        df[LABEL_COLUMN] = pd.to_numeric(df[LABEL_COLUMN], errors="coerce")

    return df


def training_corpus(df: pd.DataFrame) -> pd.Series:
    corpus = pd.concat([df["question1_clean"], df["question2_clean"]], ignore_index=True)
    corpus = corpus.drop_duplicates()
    corpus = corpus[corpus != ""]
    return corpus


def build_pair_features(model: TfidfSimilarityModel, df: pd.DataFrame) -> dict[str, np.ndarray]:
    texts1 = df["question1_clean"].tolist()
    texts2 = df["question2_clean"].tolist()
    features: dict[str, np.ndarray] = {}

    for name, transformer in model.vectorizer.transformer_list:
        matrix1 = transformer.transform(texts1)
        matrix2 = transformer.transform(texts2)
        features[name] = rowwise_cosine(matrix1, matrix2)

    combined1 = model.vectorizer.transform(texts1)
    combined2 = model.vectorizer.transform(texts2)
    features["combined"] = rowwise_cosine(combined1, combined2)

    lexical_rows = [
        token_features(text1, text2)
        for text1, text2 in zip(texts1, texts2)
    ]
    for name in ["token_jaccard", "token_containment", "length_similarity", "sequence"]:
        features[name] = np.array([row[name] for row in lexical_rows], dtype=float)

    return features


def split_labeled_sample(df: pd.DataFrame, eval_sample_size: int) -> tuple[pd.DataFrame, pd.DataFrame] | None:
    if LABEL_COLUMN not in df.columns:
        print("No is_duplicate column found. Skipping threshold calibration.")
        return None

    labeled = df.dropna(subset=[LABEL_COLUMN]).copy()
    labeled = labeled[labeled[LABEL_COLUMN].isin([0, 1])]
    if labeled.empty:
        print("No usable is_duplicate labels found. Skipping threshold calibration.")
        return None

    sample_size = min(eval_sample_size, len(labeled))
    sample = labeled.sample(n=sample_size, random_state=RANDOM_STATE)
    calibration = sample.sample(frac=0.5, random_state=RANDOM_STATE)
    holdout = sample.drop(calibration.index)

    if holdout.empty:
        holdout = calibration

    print(f"Calibration pairs: {len(calibration):,}")
    print(f"Holdout pairs: {len(holdout):,}")
    return calibration, holdout


def calibrate_model(
    model: TfidfSimilarityModel,
    calibration_df: pd.DataFrame,
    holdout_df: pd.DataFrame,
) -> dict:
    print("\nCalibrating weights and thresholds from is_duplicate labels...")
    calibration_features = build_pair_features(model, calibration_df)
    calibration_labels = calibration_df[LABEL_COLUMN].astype(int).to_numpy()

    weights, related_threshold, calibration_metrics = tune_weights_and_threshold(
        calibration_features,
        calibration_labels,
    )

    holdout_features = build_pair_features(model, holdout_df)
    holdout_labels = holdout_df[LABEL_COLUMN].astype(int).to_numpy()
    holdout_scores = build_scores_from_features(holdout_features, weights)
    very_similar_threshold = choose_very_similar_threshold(
        holdout_labels,
        holdout_scores,
        related_threshold,
    )

    from similarity.services.tfidf_similarity_model import evaluate_threshold

    holdout_metrics = evaluate_threshold(holdout_labels, holdout_scores, related_threshold)

    duplicate_threshold_percent = round(related_threshold * 100, 2)
    related_label_threshold = max(30.0, duplicate_threshold_percent - 5.0)

    model.weights = weights
    model.thresholds = {
        "related": round(related_label_threshold, 2),
        "very_similar": round(very_similar_threshold * 100, 2),
    }
    model.metrics = {
        "duplicate_threshold": duplicate_threshold_percent,
        "calibration_f1": round(calibration_metrics["f1"], 4),
        "calibration_accuracy": round(calibration_metrics["accuracy"], 4),
        "holdout_f1": round(holdout_metrics["f1"], 4),
        "holdout_accuracy": round(holdout_metrics["accuracy"], 4),
        "holdout_precision": round(holdout_metrics["precision"], 4),
        "holdout_recall": round(holdout_metrics["recall"], 4),
    }

    return {
        "weights": model.weights,
        "thresholds": model.thresholds,
        "calibration_metrics": calibration_metrics,
        "holdout_metrics": holdout_metrics,
    }


def save_metadata(
    metadata_path: Path,
    dataset_path: Path,
    model_path: Path,
    df: pd.DataFrame,
    corpus_size: int,
    model: TfidfSimilarityModel,
    calibration_report: dict | None,
):
    metadata = {
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "dataset_path": str(dataset_path),
        "model_path": str(model_path),
        "dataset_rows_after_cleaning": int(len(df)),
        "training_corpus_size": int(corpus_size),
        "sklearn_version": sklearn.__version__,
        "weights": model.weights,
        "thresholds": model.thresholds,
        "metrics": model.metrics,
        "calibration_report": calibration_report,
        "notes": "Self-trained TF-IDF model. No pretrained semantic model is used.",
    }

    with open(metadata_path, "w", encoding="utf-8") as file:
        json.dump(metadata, file, ensure_ascii=False, indent=2)


def train_tfidf_model(
    dataset_path: Path | None = None,
    model_path: Path | None = None,
    eval_sample_size: int = 40000,
    max_rows: int | None = None,
) -> bool:
    dataset_path = dataset_path or default_dataset_path()
    model_path = model_path or default_model_path()
    metadata_path = model_path.with_suffix(".metadata.json")

    print(f"Loading dataset from: {dataset_path}")
    df = load_and_prepare_dataset(dataset_path, max_rows=max_rows)
    print(f"Loaded {len(df):,} valid question pairs")

    corpus = training_corpus(df)
    print(f"Training corpus: {len(corpus):,} unique cleaned questions")

    print("\nTraining self-owned TF-IDF feature extractor...")
    print("  - word n-grams: 1-3")
    print("  - char_wb n-grams: 3-6")
    print("  - max features: 30,000 word + 50,000 char")

    vectorizer = build_tfidf_vectorizer(min_df=2, max_df=0.95)
    vectorizer.fit(corpus.tolist())

    model = TfidfSimilarityModel(vectorizer=vectorizer)

    for name, transformer in vectorizer.transformer_list:
        print(f"  - {name} vocabulary size: {len(transformer.vocabulary_):,}")

    calibration_report = None
    labeled_split = split_labeled_sample(df, eval_sample_size=eval_sample_size)
    if labeled_split:
        calibration_report = calibrate_model(model, *labeled_split)
        print("\nSelected scoring weights:")
        for name, weight in model.weights.items():
            print(f"  - {name}: {weight:.2f}")
        print("Selected thresholds:")
        print(f"  - duplicate decision: {model.metrics['duplicate_threshold']}%")
        print(f"  - related: {model.thresholds['related']}%")
        print(f"  - very_similar: {model.thresholds['very_similar']}%")
        print("Holdout metrics:")
        for name, value in model.metrics.items():
            if name.startswith("holdout_"):
                print(f"  - {name}: {value}")

    model_path.parent.mkdir(parents=True, exist_ok=True)
    with open(model_path, "wb") as file:
        pickle.dump(model, file, protocol=pickle.HIGHEST_PROTOCOL)

    save_metadata(
        metadata_path=metadata_path,
        dataset_path=dataset_path,
        model_path=model_path,
        df=df,
        corpus_size=len(corpus),
        model=model,
        calibration_report=calibration_report,
    )

    print(f"\nModel saved to: {model_path}")
    print(f"Metadata saved to: {metadata_path}")
    return True


def parse_args():
    parser = argparse.ArgumentParser(description="Train the self-owned TF-IDF similarity model.")
    parser.add_argument("--dataset", type=Path, default=default_dataset_path())
    parser.add_argument("--model", type=Path, default=default_model_path())
    parser.add_argument("--eval-sample-size", type=int, default=40000)
    parser.add_argument("--max-rows", type=int, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    print("=" * 60)
    print("Self-Trained TF-IDF Similarity Model")
    print("=" * 60)

    try:
        train_tfidf_model(
            dataset_path=args.dataset,
            model_path=args.model,
            eval_sample_size=args.eval_sample_size,
            max_rows=args.max_rows,
        )
    except Exception as exc:
        print(f"\nTraining failed: {exc}")
        sys.exit(1)

    print("\nTraining completed successfully.")
    print("The app will load this model automatically when comparing texts.")
