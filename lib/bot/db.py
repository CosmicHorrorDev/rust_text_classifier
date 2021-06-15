from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sqlite3


@dataclass
class PostsEntry:
    id: str
    prediction: str
    probability: str


class PostsDb:
    _cursor: sqlite3.Cursor

    def __init__(self, cursor) -> None:
        self._cursor = cursor

    @classmethod
    def from_file_else_create(cls, db_path: Path) -> PostsDb:
        try:
            db = cls.create(db_path)
        except FileNotFoundError:
            db = cls.create(db_path)

        return db

    @classmethod
    def from_file(cls, db_path: Path) -> PostsDb:
        if not db_path.exists():
            raise FileNotFoundError()

        connection = sqlite3.connect(db_path)
        return cls(connection.cursor())

    @classmethod
    def create(cls, db_path: Path) -> PostsDb:
        connection = sqlite3.connect(db_path)
        cursor = connection.cursor()
        cursor.executescript(
            """
            CREATE TABLE posts(
                id TEXT NOT NULL PRIMARY KEY,
                prediction TEXT NOT NULL,
                probability REAL NOT NULL
            )
            """
        )

        return cls(cursor)
