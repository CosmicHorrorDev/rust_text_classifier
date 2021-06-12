from __future__ import annotations

from numpy import float64, ndarray
from numpy.random import RandomState
from sklearn.datasets import load_files
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

from pathlib import Path
from typing import Tuple, List
import pickle


# TODO: store this in a config?
MODEL_PATH = Path("text_model.pkl")


# TODO: dedup logic around setting up the pipeline. This can be done when the classifier
# gets multiple constructors setup
# TODO: set this up to return more info if possible. Would be nice to see:
# - Per category information
# - Number of incorrect matches
# - Ability to set a threshhold along with reporting how many are ignored by it
# TODO: use random state normally
# TODO: setup tests for things
def score_classifier(*, training_percentage: float = 0.8) -> float64:
    assert (
        0.0 < training_percentage < 1.0
    ), "Percentage is represented as a float between 0.0 and 1.0"

    # Load the training data
    training_set = load_files(
        "posts_corpus", shuffle=True, encoding="utf-8", random_state=RandomState()
    )

    split_index = int(len(training_set.data) * training_percentage)
    classifier = TextClassifier.from_training(
        training_set.target_names,
        training_set.data[:split_index],
        training_set.target[:split_index],
    )

    # TODO: switch this over to a log message once that's setup
    print(f"Training set size: {split_index}")
    print(f"Test set size: {len(training_set.data) - split_index}")

    return classifier.classifier.score(
        training_set.data[split_index:], training_set.target[split_index:]
    )


class TextClassifier:
    categories: List[str]
    classifier: GridSearchCV

    def __init__(self, categories: List[str], classifier: GridSearchCV) -> None:
        self.categories = categories
        self.classifier = classifier

    @classmethod
    def from_training(
        cls, target_names: List[str], data: List[str], target: ndarray
    ) -> TextClassifier:
        return cls(
            categories=target_names,
            classifier=TextClassifier._from_training(data, target),
        )

    @classmethod
    def from_cached(cls, *, retrain: bool = False) -> TextClassifier:
        # TODO: switch this out for a custom function to load from multiple
        # sources and to slim down reading just the category types
        # - Pull from different sources
        # - read equal number of rust game and rust lang posts
        # - setup formatting the original json file on the fly
        # TODO: pickle target categories as well?
        # Load the training data
        training_set = load_files("posts_corpus", shuffle=True, encoding="utf-8")

        # Model is pickled when possible, so that we don't have to always retrain it
        if not MODEL_PATH.is_file() or retrain:
            grid_search_classifier = TextClassifier._from_training(
                training_set.data, training_set.target
            )
            TextClassifier._store_model(grid_search_classifier)
        else:
            grid_search_classifier = TextClassifier._load_model()

        return cls(
            categories=training_set.target_names, classifier=grid_search_classifier
        )

    @staticmethod
    def _load_model() -> GridSearchCV:
        with MODEL_PATH.open("rb") as pickled:
            return pickle.load(pickled)

    @staticmethod
    def _store_model(grid_search_classifier: GridSearchCV) -> None:
        with MODEL_PATH.open("wb") as to_pickle:
            pickle.dump(grid_search_classifier, to_pickle)

    @staticmethod
    def _from_training(data: List[str], target: ndarray) -> GridSearchCV:
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

        grid_search_classifier = GridSearchCV(
            classifier_pipeline, parameters, cv=5, n_jobs=-1
        )
        return grid_search_classifier.fit(data, target)

    def predict(self, text: str) -> Tuple[str, float64]:
        category = self.categories[self.classifier.predict([text])[0]]
        probabilities = self.classifier.predict_proba([text])[0]

        return category, max(probabilities)
