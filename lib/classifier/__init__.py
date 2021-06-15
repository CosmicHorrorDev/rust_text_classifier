from __future__ import annotations

from numpy import float64
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, List, Dict
import pickle

from lib.classifier.datasets import Category, Posts, PostsLoader


@dataclass
class ScoreData:
    num_correct: int
    num_incorrect: int
    num_ignored: int

    def __init__(self) -> None:
        self.num_correct = 0
        self.num_incorrect = 0
        self.num_ignored = 0

    def inc_correct(self) -> None:
        self.num_correct += 1

    def inc_incorrect(self) -> None:
        self.num_incorrect += 1

    def inc_ignored(self) -> None:
        self.num_ignored += 1


# TODO: use random state normally
def score_classifier(
    *,
    corpus_path: Path,
    training_percentage: float = 0.8,
    prediction_threshold: float = 0.5,
) -> Dict[Category, ScoreData]:
    if training_percentage > 1.0 or training_percentage < 0.0:
        raise ValueError("Percentage is represented as a float between 0.0 and 1.0")

    loader = PostsLoader(corpus_path)

    num_training_vals = int(loader.num_entries() * training_percentage)
    training_set = loader.take(num_training_vals)
    test_set = loader.take()

    # TODO: switch this over to a log message once that's setup
    print(f"Training set size: {len(training_set)}")
    print(f"Test set size: {len(test_set)}")

    classifier = TextClassifier.from_training(training_set)
    predictions = classifier.predict_set(test_set.as_data())

    score_data = {}
    for category in Category:
        score_data[category] = ScoreData()

    # Gather the results for the predictions
    for ((pred_category, pred_prob), (real_category, _)) in zip(
        predictions, test_set.category_post_pairs
    ):
        correct_prediction = pred_category == real_category

        if pred_prob < prediction_threshold:
            score_data[real_category].inc_ignored()
        else:
            if correct_prediction:
                score_data[real_category].inc_correct()
            else:
                score_data[real_category].inc_incorrect()

    return score_data


class TextClassifier:
    categories: List[Category]
    classifier: GridSearchCV

    def __init__(self, categories: List[Category], classifier: GridSearchCV) -> None:
        self.categories = list(categories)
        self.classifier = classifier

    @classmethod
    def from_cache_file_else_train(
        cls, *, cache_path: Path, corpus_path: Path
    ) -> TextClassifier:
        try:
            classifier = TextClassifier.from_cache_file(cache_path)
        except FileNotFoundError:
            loader = PostsLoader(corpus_path)
            training_set = loader.take()
            classifier = TextClassifier.from_training(training_set)

            with cache_path.open("wb") as to_pickle:
                pickle.dump(classifier.classifier, to_pickle)

        return classifier

    @classmethod
    def from_cache_file(cls, cache_path: Path) -> TextClassifier:
        with cache_path.open("rb") as pickled:
            grid_search_classifier = pickle.load(pickled)

        return cls(categories=list(Category), classifier=grid_search_classifier)

    @classmethod
    def from_training(cls, training_set: Posts) -> TextClassifier:
        # Setup a pipeline for the classifier
        # - Generates feature vectors using a count vectorizer
        # - Determines term frequency inverse document frequency
        # - Classifies using a linear SVM
        classifier_pipeline = Pipeline(
            [
                ("frequency_vectorizer", TfidfVectorizer()),
                (
                    "classifier",
                    SGDClassifier(
                        penalty="l2",
                        tol=None,
                    ),
                ),
            ]
        )

        # Select optimal pipeline parameters using grid search
        parameters = {
            "frequency_vectorizer__ngram_range": [(1, 1), (1, 2)],
            "frequency_vectorizer__use_idf": (True, False),
            "classifier__alpha": (1e-2, 1e-3),
            # These are the loss fuctions that support `predict_proba`
            # https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDClassifier.html#sklearn.linear_model.SGDClassifier.predict_proba
            "classifier__loss": ("log", "modified_huber"),
        }

        classifier = GridSearchCV(classifier_pipeline, parameters, cv=5, n_jobs=-1)
        classifier = classifier.fit(**training_set.as_data_target_kwargs())

        return cls(categories=list(Category), classifier=classifier)

    def predict(self, text: str) -> Tuple[Category, float64]:
        category = self.categories[self.classifier.predict([text])[0]]
        probabilities = self.classifier.predict_proba([text])[0]

        return category, max(probabilities)

    def predict_set(self, texts: List[str]) -> List[Tuple[Category, float64]]:
        results = []
        for text in texts:
            results.append(self.predict(text))

        return results

    def score(self, test_set: Posts) -> float64:
        return self.classifier.score(**test_set.as_data_target_kwargs())
