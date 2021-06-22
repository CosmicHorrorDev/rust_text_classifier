from lib.config import BASE_DIR, Config, PrawConfig
from tests.constants import REPO_DIR


def test_config() -> None:
    config = Config(REPO_DIR / "sample_config.json")

    ideal = Config.__new__(Config)
    ideal._praw_config = PrawConfig(
        "REDDIT_CLIENT_ID",
        "REDDIT_CLIENT_SECRET",
        (
            "Rust Text Classifier Bot by /u/KhorneLordOfChaos v0.2.1"
            " https://github.com/LovecraftianHorror/rust_text_classifier"
        ),
        "REDDIT_USERNAME",
        "REDDIT_PASSWORD",
    )

    ideal._posts_db = BASE_DIR / "posts.db"
    ideal._cached_classifier_path = BASE_DIR / "text_classifier.pkl"
    ideal._posts_corpus = BASE_DIR / "posts_corpus"

    ideal._cutoff_threshold = 0.5
    ideal._daily_comment_limit = 5

    assert ideal == config
