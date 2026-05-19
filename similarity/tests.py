from django.test import SimpleTestCase

from similarity.services.text_preprocessing import clean_text
from similarity.services.tfidf_similarity_model import (
    TfidfSimilarityModel,
    build_tfidf_vectorizer,
)


class TextPreprocessingTests(SimpleTestCase):
    def test_expands_common_contractions(self):
        self.assertEqual(
            clean_text("I don't know what's wrong."),
            "i do not know what is wrong",
        )


class TfidfSimilarityModelTests(SimpleTestCase):
    def build_small_model(self):
        corpus = [
            "how can i learn python programming",
            "what is the best way to learn python",
            "how do i cook rice",
            "best rice cooking method",
        ]
        vectorizer = build_tfidf_vectorizer(
            min_df=1,
            max_df=1.0,
            word_max_features=1000,
            char_max_features=1000,
        )
        vectorizer.fit([clean_text(text) for text in corpus])
        return TfidfSimilarityModel(vectorizer=vectorizer)

    def test_related_text_scores_higher_than_unrelated_text(self):
        model = self.build_small_model()

        related = model.score_percent(
            "How can I learn Python?",
            "What is the best way to learn Python programming?",
        )
        unrelated = model.score_percent(
            "How can I learn Python?",
            "How do I cook rice?",
        )

        self.assertGreater(related, unrelated)

    def test_predict_uses_model_thresholds(self):
        model = self.build_small_model()
        model.thresholds = {
            "related": 10.0,
            "very_similar": 95.0,
        }

        result = model.predict(
            "How can I learn Python?",
            "What is the best way to learn Python programming?",
        )

        self.assertEqual(result["label"], "Có liên quan")
        self.assertEqual(result["severity"], "medium")
