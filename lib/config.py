from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict

BASE_DIR = Path("/") / "opt" / "rust_text_classifier"


@dataclass
class Config:
    # Reddit stuff
    _praw_config: PrawConfig
    _posts_db: Path
    # Classifier stuff
    _posts_corpus: Path
    _cached_classifier_path: Path
    _cutoff_threshold: float
    # Bot stuff
    _daily_comment_limit: int

    def __init__(self, config_path: Path = BASE_DIR / "config.json") -> None:
        with config_path.open() as file_handle:
            contents = json.load(file_handle)

        self._praw_config = PrawConfig(
            client_id=contents["reddit"]["client_id"],
            client_secret=contents["reddit"]["client_secret"],
            user_agent=contents["reddit"]["user_agent"],
            username=contents["reddit"]["username"],
            password=contents["reddit"]["password"],
        )

        self._posts_db = BASE_DIR / "posts.db"
        self._cached_classifier_path = BASE_DIR / "text_classifier.pkl"
        self._posts_corpus = BASE_DIR / "posts_corpus"

        self._cutoff_threshold = contents["cutoff_threshold"]
        self._daily_comment_limit = contents["daily_comment_limit"]

    def as_praw_auth_kwargs(self) -> Dict[str, str]:
        return self._praw_config.as_auth_kwargs()

    def posts_db(self) -> Path:
        return self._posts_db

    def posts_corpus(self) -> Path:
        return self._posts_corpus

    def cached_classifier_path(self) -> Path:
        return self._cached_classifier_path

    def cutoff_threshold(self) -> float:
        return self._cutoff_threshold

    def daily_comment_limit(self) -> int:
        return self._daily_comment_limit


@dataclass
class PrawConfig:
    client_id: str
    client_secret: str
    user_agent: str
    username: str
    password: str

    def as_auth_kwargs(self) -> Dict[str, str]:
        return {
            "client_id": self.client_id,
            "client_secret": self.client_secret,
            "user_agent": self.user_agent,
            "username": self.username,
            "password": self.password,
        }
