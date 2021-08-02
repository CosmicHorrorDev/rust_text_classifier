from lib.bot.db import Db, PostEntry
from lib.classifier.datasets import Category


def test_posts_db() -> None:
    # Create a test db in memory
    posts_db = Db("sqlite:///:memory:")

    # Insert some dummy values
    dummy_entry1 = PostEntry(
        id="test_id1",
        prediction=Category.LANG,
        probability=71.23,
        title="Lang title",
        body="Lang body",
    )
    dummy_entry2 = PostEntry(
        id="test_id2",
        prediction=Category.GAME,
        probability=98.76,
        title="Game title",
        body="Game body",
    )
    posts_db.insert(dummy_entry1)
    posts_db.insert(dummy_entry2)

    # Test out reading behavior
    assert posts_db.find("test_id1") == dummy_entry1
    assert posts_db.find("test_id2") == dummy_entry2
    assert posts_db.find("does not exist") is None

    posts_db.close()
