import logging
import traceback
from datetime import datetime, timedelta
from time import sleep
from typing import List

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
    "\n* r/playrustlfg"
    "\n* r/playrustservers"
    "\n\nIf this post _is_ actually about the Rust Programming Language then congratz"
    " on being part of the lucky few posts that get incorrectly classified! 🎉"
    "\n\n[source](https://github.com/LovecraftianHorror/rust_text_classifier)"
    " | [author](https://www.reddit.com/message/compose/?to=KhorneLordOfChaos)"
    " | [sponsor](https://github.com/sponsors/LovecraftianHorror)"
)


# Just a simple cache to store IDs for this session which sits in front of the check on
# the database to help relieve strain
class IdCache:
    ids: List[str]
    ID_LIMIT: int = 40

    def __init__(self) -> None:
        self.ids = []

    def contains(self, needle: str) -> bool:
        return needle in self.ids

    def push(self, new_id: str) -> None:
        self.ids = [new_id] + self.ids
        while len(self.ids) > self.ID_LIMIT:
            self.ids.pop()


def run(config: Config, args: Args) -> None:
    SUBMISSION_FETCH_LIMIT = 20
    ONE_DAY = timedelta(days=1)

    daily_comment_total = 0
    daily_marker = datetime.now()
    id_cache = IdCache()

    logging.info("Starting the reddit bot")
    logging.info("Setting up the r/rust submission stream")
    reddit = Reddit(**config.as_praw_auth_kwargs())
    subreddit = reddit.subreddit("rust")

    logging.info("Connecting to the posts database")
    posts_db = PostsDb.from_file_else_create(config.posts_db())

    logging.info("Setting up the text classifier")
    classifier = TextClassifier.from_cache_file_else_train(
        cache_path=config.cached_classifier_path(), corpus_path=config.posts_corpus()
    )

    logging.info("Beginning the event loop")
    # Event loop goes as follows:
    # 1. Perform any possible housekeeping tasks
    # 2. Read the <SUBMISSION_FETCH_LIMIT> latest posts from /r/rust
    # 3. For each text post
    #   a. Check if the post is in the session cache. If so then skip
    #   b. Check if the post is already in the database. If so then skip
    #   c. Run the classifier on the title and body
    #   d. Add the entry to the posts database
    #   e. Comment on the post if it seems like it's about the game
    # 4. Delay then return to step 1.
    while True:
        sleep(30)

        # Reset daily comment limit if needed
        if (datetime.now() - daily_marker) >= ONE_DAY:
            logging.info("Resetting the daily comment limit")
            daily_comment_total = 0
            daily_marker = datetime.now()

        try:
            # /r/rust is a pretty low traffic subreddit so a limit of just 20 should be
            # more than enough
            for submission in subreddit.new(limit=SUBMISSION_FETCH_LIMIT):
                id = submission.id

                if id_cache.contains(id):
                    continue

                # I _think_ that submissions are fetched lazily so we avoid actually
                # fetching them by only reading this data after we're already past the
                # session ID cache
                title = submission.title
                truncated_title = title[:40]
                body = submission.selftext

                if posts_db.find(submission.id) is not None:
                    # Ignore posts that we've already seen
                    logging.debug(f"Ignored - Already in db - {truncated_title}")
                    continue

                if not submission.is_self:
                    # We only care about self (aka text) posts
                    logging.debug(f"Ignored - Non-text post - {truncated_title}")
                    continue

                # New post so time to classify it
                category, probability = classifier.predict(f"{title}\n{body}")
                percent_probability = 100 * probability
                logging.info(
                    f"Classified - {category} ({percent_probability:.2f}%) -"
                    f" {truncated_title}"
                )

                # Add the new entry to the database
                posts_db.insert(PostsEntry(id, category, float(probability)))

                if daily_comment_total >= config.daily_comment_limit():
                    logging.info(
                        f"Ignored - Comment limit hit till {daily_marker + ONE_DAY}"
                    )
                    continue

                if args.dont_comment:
                    logging.debug("Ignored - Passive mode")
                    continue

                if category != Category.GAME or probability < config.cutoff_threshold():
                    logging.debug("Ignored - Post below threshold")
                    continue

                # Add a comment if it looks like the post is about the game
                logging.info(f"Replying - {truncated_title}")
                daily_comment_total += 1
                submission.reply(
                    COMMENT_TEMPLATE.format(percent_probability=percent_probability)
                )
        except Exception:
            exception_traceback = traceback.format_exc()
            logging.warning(exception_traceback)
