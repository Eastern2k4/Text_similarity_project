from django.shortcuts import render

from .forms import TextSimilarityForm
from .services.similarity_service import calculate_similarity


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