from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict

from xdg import xdg_cache_home, xdg_config_home, xdg_data_home

DIR_NAME = "rust_text_classifier"


def config_dir() -> Path:
    return xdg_config_home() / DIR_NAME


def data_dir() -> Path:
    return xdg_data_home() / DIR_NAME


def cache_dir() -> Path:
    return xdg_cache_home() / DIR_NAME


class Config:
    # Reddit stuff
    _praw_config: PrawConfig
    _posts_db: Path
    # Classifier stuff
    _posts_corpus: Path
    _cached_classifier_path: Path
    _cutoff_threshold: float

    def __init__(self) -> None:
        config_path = config_dir() / "config.json"

        with config_path.open() as file_handle:
            contents = json.load(file_handle)

        self._praw_config = PrawConfig(
            client_id=contents["praw"]["client_id"],
            client_secret=contents["praw"]["client_secret"],
            user_agent=contents["praw"]["user_agent"],
            username=contents["praw"]["username"],
            password=contents["praw"]["password"],
        )
        self._posts_db = data_dir() / "posts.db"

        self._classification_threshold = contents["classifier"]["cutoff_threshold"]
        self._cached_classifier_path = cache_dir() / "text_classifier.pkl"
        self._posts_corpus = data_dir() / "posts_corpus"

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
