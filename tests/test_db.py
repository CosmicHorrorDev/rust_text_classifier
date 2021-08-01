from lib.bot.db import PostsDb, PostsEntry
from lib.classifier.datasets import Category


def test_posts_db() -> None:
    # Create a test db in memory
    posts_db = PostsDb.create(":memory:")

    # Insert some dummy values
    dummy_entry1 = PostsEntry(
        "test_id1", Category.LANG, 71.23, "Lang title", "Lang body"
    )
    dummy_entry2 = PostsEntry(
        "test_id2", Category.GAME, 98.76, "Game title", "Game body"
    )
    posts_db.insert(dummy_entry1)
    posts_db.insert(dummy_entry2)

    # Test out reading behavior
    assert posts_db.find("test_id1") == dummy_entry1
    assert posts_db.find("test_id2") == dummy_entry2
    assert posts_db.find("does not exist") is None

    posts_db.close()
