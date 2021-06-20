import traceback
from time import sleep

from praw import Reddit
from praw.models import Submission

from lib.bot.db import PostsDb, PostsEntry
from lib.classifier import TextClassifier
from lib.classifier.datasets import Category
from lib.cli import Args
from lib.config import Config


# TODO: switch all the `print` things over to logging
def run(config: Config, args: Args) -> None:
    print("Starting the reddit bot")

    print("Setting up the r/rust submission stream")
    reddit = Reddit(**config.as_praw_auth_kwargs())
    subreddit = reddit.subreddit("rust")
    submission_stream = subreddit.stream.submissions()

    print("Connecting to the old posts database")
    posts_db = PostsDb.from_file_else_create(config.posts_db())

    print("Setting up the text classifier")
    classifier = TextClassifier.from_cache_file_else_train(
        cache_path=config.cached_classifier_path(), corpus_path=config.posts_corpus()
    )
    threshold = config.cutoff_threshold()

    print("Beginning the event loop")
    # Event loop goes as follows:
    # 1. Read all text posts from r/rust
    # 2. For each text post
    #   a. Check if the post is already in the database. If so then skip
    #   b. Run the classifier on the title and body
    #   c. Add the entry to the posts database
    #   d. Comment on the post if it seems like it's about the game
    # 3. Delay then return to step 1.
    while True:
        sleep(30)

        try:
            for submission in submission_stream:
                handle_submission(
                    submission,
                    classifier,
                    posts_db,
                    config.cutoff_threshold(),
                    args.dont_comment,
                )
        except Exception:
            exception_traceback = traceback.format_exc()
            print(exception_traceback)


def handle_submission(
    submission: Submission,
    classifier: TextClassifier,
    posts_db: PostsDb,
    threshold: float,
    dont_comment: bool,
) -> None:
    id = submission.id
    title = submission.title
    truncated_title = title[:40]
    body = submission.selftext

    if not submission.is_self:
        # We only care about self (aka text) posts
        print(f"Ignored - Non-text post - {truncated_title}")
        return

    if posts_db.find(submission.id) is not None:
        # Ignore posts that we've already seen
        print(f"Ignored - Already in db - {truncated_title}")
        return

    # New post so time to classify it
    category, probability = classifier.predict(f"{title}\n{body}")
    print(f"Classified - {category} ({100 * probability:.2f}%) - {truncated_title}")

    # Add the new entry to the database
    posts_db.insert(PostsEntry(id, category, float(probability)))

    # Add a comment if it looks like the post is about the game
    if not dont_comment and category == Category.GAME and probability >= threshold:
        print(f"Replying - {truncated_title}")
        submission.reply(
            "Hello! It looks like this post is _likely_ about the Rust videogame"
            f" ({100 * probability:.2f}%) while this sub is for the"
            " [Rust Programming Language](https://www.rust-lang.org) instead."
            "\n\nIf this post is about the Rust video game then feel free to use one of"
            " the related subreddits (be sure to read the subreddit rules first)"
            "\n\n* r/playrust"
            "\n* r/rustconsole"
            "\n* r/rustlfg"
            "\n* r/playrustservers"
            "\n\nIf this post is _actually_ about the Rust Programming Language then"
            " congratz on being part of the lucky ~2% of posts that get incorrectly"
            " classified! 🎉"
            "\n\n[source](https://github.com/LovecraftianHorror/rust_text_classifier)"
            " | [author](https://www.reddit.com/message/compose/?to=KhorneLordOfChaos)"
            " | [sponsor](https://github.com/sponsors/LovecraftianHorror)"
        )
