import json
import pandas as pd
import random
from pathlib import Path
from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.http import require_http_methods

from .forms import TextSimilarityForm
from .services.similarity_service import calculate_similarity


def get_dataset_path():
    """Get the path to the questions dataset"""
    return Path(__file__).resolve().parent.parent.parent / "ml" / "datasets" / "questions.csv"


def load_random_questions():
    """Load random questions from dataset"""
    try:
        dataset_path = get_dataset_path()
        if not dataset_path.exists():
            return None, None
        
        # Load CSV file
        df = pd.read_csv(dataset_path)
        
        # Get random row
        random_row = df.sample(n=1).iloc[0]
        
        question1 = str(random_row['question1']).strip()
        question2 = str(random_row['question2']).strip()
        
        return question1, question2
    except Exception as e:
        print(f"Error loading dataset: {str(e)}")
        return None, None


def index(request):
    result = None

    if request.method == "POST":
        form = TextSimilarityForm(request.POST)

        if form.is_valid():
            text1 = form.cleaned_data["text1"]
            text2 = form.cleaned_data["text2"]

            result = calculate_similarity(text1, text2)

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