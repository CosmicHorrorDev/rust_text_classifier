from lib.classifier.datasets import PostsLoader
from tests.constants import SAMPLE_CORPUS_DIR


def test_take_amount():
    loader = PostsLoader(SAMPLE_CORPUS_DIR)

    for i in range(6):
        posts = loader.take(i)
        assert len(posts) == i
