from __future__ import annotations

from numpy import float64
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.utils import Bunch

from pathlib import Path
from typing import Tuple, List
import pickle

from lib.datasets import PostsLoader


# TODO: store this in a config?
MODEL_PATH = Path("text_model.pkl")


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

    loader = PostsLoader(Path("posts_corpus"))

    num_training_vals = int(loader.num_entries() * training_percentage)
    training_set = loader.take(num_training_vals)
    test_set = loader.take()

    # TODO: switch this over to a log message once that's setup
    print(f"Training set size: {len(training_set.data)}")
    print(f"Test set size: {len(test_set.data)}")

    classifier = TextClassifier.from_training(training_set)
    return classifier.score(test_set)


class TextClassifier:
    categories: List[str]
    classifier: GridSearchCV

    def __init__(self, categories: List[str], classifier: GridSearchCV) -> None:
        self.categories = categories
        self.classifier = classifier

    @classmethod
    def from_training(cls, training_set: Bunch) -> TextClassifier:
        return cls(
            categories=training_set.target_names,
            classifier=TextClassifier._from_training(training_set),
        )

    # TODO: pickle target categories as well?
    @classmethod
    def from_cached(cls, *, retrain: bool = False) -> TextClassifier:
        loader = PostsLoader(Path("posts_corpus"))

        # Model is pickled when possible, so that we don't have to always retrain it.
        # This method also avoids loading the files unless training is needed
        if not MODEL_PATH.is_file() or retrain:
            training_set = loader.take()
            grid_search_classifier = TextClassifier._from_training(training_set)
            TextClassifier._store_model(grid_search_classifier)
        else:
            grid_search_classifier = TextClassifier._load_model()

        return cls(categories=loader.categories(), classifier=grid_search_classifier)

    @staticmethod
    def _load_model() -> GridSearchCV:
        with MODEL_PATH.open("rb") as pickled:
            return pickle.load(pickled)

    @staticmethod
    def _store_model(grid_search_classifier: GridSearchCV) -> None:
        with MODEL_PATH.open("wb") as to_pickle:
            pickle.dump(grid_search_classifier, to_pickle)

    @staticmethod
    def _from_training(training_set: Bunch) -> GridSearchCV:
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
        return grid_search_classifier.fit(training_set.data, training_set.target)

    def predict(self, text: str) -> Tuple[str, float64]:
        category = self.categories[self.classifier.predict([text])[0]]
        probabilities = self.classifier.predict_proba([text])[0]

        return category, max(probabilities)

    def score(self, test_set: Bunch) -> float64:
        return self.classifier.score(test_set.data, test_set.target)
