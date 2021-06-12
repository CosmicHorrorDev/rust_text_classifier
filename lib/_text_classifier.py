from sklearn.datasets import load_files
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
import numpy

from pathlib import Path
import pickle


# TODO: store this in a config?
MODEL_PATH = Path("text_model.pkl")


# TODO: score can be used to rank things off new data
class TextClassifier:
    # TODO: clean up initialization with multiple constructors
    def __init__(self, retrain=False):
        # TODO: switch this out for a custom function to load from multiple
        # sources and to slim down reading just the category types
        # TODO: pickle target categories as well?
        # Load the training data
        training_set = load_files("posts_corpus", shuffle=True, encoding="utf-8")

        # Model is pickled when possible, so that we don't have to always retrain it
        if not MODEL_PATH.is_file() or retrain:
            grid_search_classifier = self._from_training()
            self._store(grid_search_classifier)
        else:
            grid_search_classifier = self._load()

        self.categories = training_set.target_names
        self.classifier = grid_search_classifier

    def _load(self):
        with MODEL_PATH.open("rb") as pickled:
            return pickle.load(pickled)

    def _store(self, grid_search_classifier):
        with MODEL_PATH.open("wb") as to_pickle:
            pickle.dump(grid_search_classifier, to_pickle)

    def _from_training(self):
        # Load the training data
        training_set = load_files("posts_corpus", shuffle=True, encoding="utf-8")

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
        return grid_search_classifier.fit(
            training_set.data, training_set.target
        )

    def predict(self, text) -> (str, numpy.float64):
        category = self.categories[self.classifier.predict([text])[0]]
        probabilities = self.classifier.predict_proba([text])[0]

        return category, max(probabilities)
