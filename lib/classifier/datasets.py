from __future__ import annotations

import json
import random
from enum import Enum
from io import TextIOWrapper
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


class Category(Enum):
    LANG = 0
    GAME = 1

    def as_target(self) -> int:
        return self.value

    @classmethod
    def from_target(cls, target: int) -> Category:
        return list(Category)[target]

    @classmethod
    def default(cls) -> Category:
        return Category.LANG


class Post:
    title: str
    body: str
    id: str

    def __init__(self, title: str, body: str, id: str) -> None:
        self.title = title
        self.body = body
        self.id = id

    @classmethod
    def from_file_handle(cls, file_handle: TextIOWrapper) -> Post:
        contents = json.load(file_handle)

        return cls(
            title=contents["title"],
            body=contents["selftext"],
            id=contents["id"],
        )

    def __str__(self) -> str:
        return f"{self.title}\n{self.body}"


# `load_files` normally returns a `Bunch` which is weakly typed enough to make my skin
# crawl, so instead we make our own class to represent a set of training data instead
class Posts:
    category_post_pairs: List[Tuple[Category, Post]]

    def __init__(self, category_post_pairs: List[Tuple[Category, Post]]) -> None:
        self.category_post_pairs = category_post_pairs

    def __len__(self) -> int:
        return len(self.category_post_pairs)

    def as_data(self) -> List[str]:
        data_set = []
        for (_, data) in self.category_post_pairs:
            data_set.append(str(data))

        return data_set

    def as_target(self) -> np.ndarray:
        target_set = []
        for (category, _) in self.category_post_pairs:
            target_set.append(category.as_target())
        target_set = np.array(target_set)

        return target_set

    def as_data_target_kwargs(self) -> Dict[str, Any]:
        return {"X": self.as_data(), "y": self.as_target()}


# Lazily loads posts from their respective categories.
#
# Allows for taking specific batch sizes of data (useful for training then testing)
# while ensuring an equal number of rust game and rust lang posts are returned.
#
# For rust game posts it will prefer misposts in r/rust first, then after that it evenly
# samples from the other sources weighted to the number of posts in each source
#
# Number of entries is limited by the lesser of the number of rust game vs rust lang
# posts
class PostsLoader:
    corpus_dir: Path
    _posts_map: Optional[Dict[Category, List[Post]]]
    # Taking an odd amount means popping an extra value off some category so to keep
    # things fair this will be cycled between the two categories
    _odd_take_lang: bool

    def __init__(self, corpus_dir: Path) -> None:
        self.corpus_dir = corpus_dir
        self._posts_map = None
        self._odd_take_lang = False

    def num_entries(self) -> int:
        self._maybe_populate_posts_map()
        return sum([len(posts) for posts in self._posts_map.values()])

    def take(self, amount: Optional[int] = None) -> Posts:
        self._maybe_populate_posts_map()
        if amount is None or amount > self.num_entries():
            amount = self.num_entries()

        if amount < 0:
            raise ValueError("Can only take non-negative amounts")

        category_to_data_pairs = []
        for category in self._posts_map:
            # Trim off the desired number of entries
            posts = self._posts_map[category]
            split_off = posts[: (amount // 2)]
            self._posts_map[category] = posts[(amount // 2) :]

            for post in split_off:
                category_to_data_pairs.append((category, post))

        # Take an extra post from the correct category
        if amount % 2 == 1:
            if self._odd_take_lang:
                category = Category.LANG
            else:
                category = Category.GAME

            # Swap the one to take next
            self._odd_take_lang = not self._odd_take_lang

            extra_post = self._posts_map[category].pop()
            category_to_data_pairs.append((category, extra_post))

        random.shuffle(category_to_data_pairs)
        assert len(category_to_data_pairs) == amount
        return Posts(category_to_data_pairs)

    def _maybe_populate_posts_map(self) -> None:
        if self._posts_map is None:
            lang_posts = self._load_lang_posts()
            game_posts = self._load_game_posts(limit=len(lang_posts))
            # Limit `lang_posts` in case `game_posts` is longer
            lang_posts = lang_posts[: len(game_posts)]

            assert len(lang_posts) == len(
                game_posts
            ), "Categories should have the same number of entries"

            self._posts_map = {
                Category.LANG: lang_posts,
                Category.GAME: game_posts,
            }

    def _load_lang_posts(self, limit: Optional[int] = None) -> List[Post]:
        # Nothing complicated here, we just load from the lang post directory
        lang_dir = self.corpus_dir / "r_rust_correct"

        files = [entry for entry in lang_dir.iterdir() if entry.is_file()]

        return self._load_post_files(files, limit)

    def _load_game_posts(self, limit: Optional[int] = None) -> List[Post]:
        # Slightly more compilicated since there are multiple sources. For this we
        # prefer reading from incorrect r/rust posts first, then after sample from all
        # the `equal_dirs`
        preferred_dir = self.corpus_dir / "r_rust_incorrect"
        preferred_files = [
            entry for entry in preferred_dir.iterdir() if entry.is_file()
        ]

        equal_dirs = [
            self.corpus_dir / "r_playrust",
            self.corpus_dir / "r_rustconsole",
            self.corpus_dir / "r_rustlfg",
        ]

        # TODO: switch to sampling from sources evenly
        equal_files = []
        for equal_dir in equal_dirs:
            equal_files += [f for f in equal_dir.iterdir() if f.is_file()]
        random.shuffle(equal_files)

        return self._load_post_files(preferred_files + equal_files, limit)

    # TODO: pass shuffle into here so that it can be shuffled after being limited?
    @staticmethod
    def _load_post_files(
        post_files: List[Path], limit: Optional[int] = None
    ) -> List[Post]:
        if limit is None or limit > len(post_files):
            limit = len(post_files)

        posts = []
        for post_file in post_files[:limit]:
            with post_file.open() as file_handle:
                post = Post.from_file_handle(file_handle)
                posts.append(post)

        random.shuffle(posts)
        return posts
