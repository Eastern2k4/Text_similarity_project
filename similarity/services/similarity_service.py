from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from .text_preprocessing import clean_text


model = SentenceTransformer("all-MiniLM-L6-v2")


def get_similarity_label(score: float) -> str:
    if score >= 70:
        return "Rất giống nhau"
    elif score >= 40:
        return "Tương đối giống nhau"
    else:
        return "Khác nhau"


def calculate_similarity(text1: str, text2: str) -> dict:
    text1_clean = clean_text(text1)
    text2_clean = clean_text(text2)

    embeddings = model.encode([text1_clean, text2_clean])

    score = cosine_similarity(
        [embeddings[0]],
        [embeddings[1]]
    )[0][0]

    score_percent = round(score * 100, 2)

    return {
        "score": score_percent,
        "label": get_similarity_label(score_percent),
        "text1": text1,
        "text2": text2,
    }