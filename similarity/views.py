import pandas as pd
import logging
from pathlib import Path
from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.http import require_http_methods
from django.conf import settings

from .forms import TextSimilarityForm
from .services.similarity_service import calculate_similarity

logger = logging.getLogger(__name__)
_questions_cache = None


def get_dataset_path():
    """Get the path to the questions dataset"""
    return Path(settings.BASE_DIR) / "ml" / "datasets" / "questions.csv"


def get_questions_dataframe():
    """Load and cache valid question pairs for random examples."""
    global _questions_cache

    if _questions_cache is not None:
        return _questions_cache

    dataset_path = get_dataset_path()
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset file not found at {dataset_path}")

    df = pd.read_csv(dataset_path, usecols=["question1", "question2"])
    df = df.dropna(subset=["question1", "question2"])
    df["question1"] = df["question1"].astype(str).str.strip()
    df["question2"] = df["question2"].astype(str).str.strip()
    df = df[(df["question1"] != "") & (df["question2"] != "")]

    if df.empty:
        raise ValueError("Dataset does not contain valid question pairs")

    _questions_cache = df
    return _questions_cache


def load_random_questions():
    """Load random questions from dataset"""
    try:
        df = get_questions_dataframe()
        random_row = df.sample(n=1).iloc[0]

        return random_row["question1"], random_row["question2"]
    except Exception:
        logger.exception("Error loading random questions")
        return None, None


def index(request):
    result = None

    if request.method == "POST":
        form = TextSimilarityForm(request.POST)

        if form.is_valid():
            text1 = form.cleaned_data["text1"]
            text2 = form.cleaned_data["text2"]

            try:
                result = calculate_similarity(text1, text2)
            except RuntimeError:
                logger.exception("Similarity model is not available")
                form.add_error(
                    None,
                    "Không thể tải model đã huấn luyện. Hãy chạy python ml\\training\\train_model.py trước.",
                )
            except Exception:
                logger.exception("Unexpected similarity calculation error")
                form.add_error(None, "Không thể tính độ tương đồng. Vui lòng thử lại.")

    else:
        form = TextSimilarityForm()

    return render(request, "similarity/index.html", {
        "form": form,
        "result": result,
    })


@require_http_methods(["GET"])
def generate_random_texts(request):
    """API endpoint to generate random texts from dataset"""
    question1, question2 = load_random_questions()
    
    if question1 and question2:
        return JsonResponse({
            "success": True,
            "text1": question1,
            "text2": question2
        })
    else:
        return JsonResponse({
            "success": False,
            "error": "Không thể tải dữ liệu từ dataset"
        }, status=400)
