from flaky import flaky

from lib import score_classifier, TextClassifier
from lib.datasets import Category, PostsLoader
from tests.constants import SAMPLE_CORPUS_DIR


# I wanted to keep the test time short and reaching a 90% score is still common even
# with such a small corpus, so failing twice in a row should be pretty rare
@flaky
def test_decent_scoring_accuracy():
    category_scores = score_classifier(corpus_path=SAMPLE_CORPUS_DIR)

    print(category_scores)
    assert False

    total_correct = 0
    total_incorrect = 0
    for score in category_scores.values():
        total_correct += score.num_correct
        total_incorrect += score.num_incorrect

    overall_score = total_correct / (total_correct + total_incorrect)
    assert overall_score >= 0.9, (
        "Scores are generally around ~95% accurate with this sample set, so an average"
        " of 90% should be reasonable"
    )


# Test _could_ fail since the probabilites aren't always reached, so the flaky plugin
# will rerun the test once if there was a failure
@flaky
def test_basic_snippets():
    EXPECTED_MINIMUM_PROBABILITY = 0.65

    loader = PostsLoader(SAMPLE_CORPUS_DIR)
    classifier = TextClassifier.from_training(loader.take())

    category, probability = classifier.predict(
        "2k hours, looking for a squad to raid with"
    )
    assert category == Category.GAME
    assert probability > EXPECTED_MINIMUM_PROBABILITY

    category, probability = classifier.predict(
        "What is the memory usage of String::new?"
    )
    assert category == Category.LANG
    assert probability > EXPECTED_MINIMUM_PROBABILITY
