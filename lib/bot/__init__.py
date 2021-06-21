import traceback
from datetime import datetime, timedelta
from time import sleep

from praw import Reddit

from lib.bot.db import PostsDb, PostsEntry
from lib.classifier import TextClassifier
from lib.classifier.datasets import Category
from lib.cli import Args
from lib.config import Config

COMMENT_TEMPLATE = (
    "Hello! It looks like this post is _likely_ about the Rust video game"
    " ({percent_probability:.2f}%) while this sub is for the"
    " [Rust Programming Language](https://www.rust-lang.org) instead."
    "\n\nIf this post is about the Rust video game then feel free to use one of the"
    " related subreddits (be sure to read the subreddit rules first)"
    "\n\n* r/playrust"
    "\n* r/rustconsole"
    "\n* r/rustlfg"
    "\n* r/playrustservers"
    "\n\nIf this post _is_ actually about the Rust Programming Language then congratz"
    " on being part of the lucky ~2% of posts that get incorrectly classified! 🎉"
    "\n\n[source](https://github.com/LovecraftianHorror/rust_text_classifier)"
    " | [author](https://www.reddit.com/message/compose/?to=KhorneLordOfChaos)"
    " | [sponsor](https://github.com/sponsors/LovecraftianHorror)"
)


def run(config: Config, args: Args) -> None:
    ONE_DAY = timedelta(days=1)

    daily_comment_total = 0
    daily_marker = datetime.now()

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

        # Reset daily comment limit if needed
        if (datetime.now() - daily_marker) >= ONE_DAY:
            daily_comment_total = 0
            daily_marker = datetime.now()

        try:
            for submission in submission_stream:
                id = submission.id
                title = submission.title
                truncated_title = title[:40]
                body = submission.selftext

                if not submission.is_self:
                    # We only care about self (aka text) posts
                    print(f"Ignored - Non-text post - {truncated_title}")
                    continue

                if posts_db.find(submission.id) is not None:
                    # Ignore posts that we've already seen
                    print(f"Ignored - Already in db - {truncated_title}")
                    continue

                # New post so time to classify it
                category, probability = classifier.predict(f"{title}\n{body}")
                percent_probability = 100 * probability
                print(
                    f"Classified - {category} ({percent_probability:.2f}%) -"
                    f" {truncated_title}"
                )

                # Add the new entry to the database
                posts_db.insert(PostsEntry(id, category, float(probability)))

                if daily_comment_total >= config.daily_comment_limit():
                    print(f"Ignored - Comment limit hit till {daily_marker + ONE_DAY}")
                    continue

                if args.dont_comment:
                    print("Ignored - Passive mode")
                    continue

                if category != Category.GAME or probability < config.cutoff_threshold():
                    print("Ignored - Post below threshold")
                    continue

                # Add a comment if it looks like the post is about the game
                print(f"Replying - {truncated_title}")
                daily_comment_total += 1
                submission.reply(
                    COMMENT_TEMPLATE.format(percent_probability=percent_probability)
                )
        except Exception:
            exception_traceback = traceback.format_exc()
            print(exception_traceback)
