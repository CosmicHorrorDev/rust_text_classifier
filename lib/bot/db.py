from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple, Optional, Union
from pathlib import Path
import sqlite3

from lib.classifier.datasets import Category


@dataclass
class PostsEntry:
    id: str
    prediction: Category
    probability: float

    @classmethod
    def from_db_entry(cls, entry: Tuple[str, int, float]) -> PostsEntry:
        id, prediction_target, probability = entry
        return cls(
            id=id,
            prediction=Category.from_target(prediction_target),
            probability=probability,
        )

    def as_db_entry(self) -> Tuple[str, int, float]:
        return (self.id, self.prediction.as_target(), self.probability)


class PostsDb:
    _connection: sqlite3.Connection
    _cursor: sqlite3.Cursor

    def __init__(self, connection: sqlite3.Connection, cursor: sqlite3.Cursor) -> None:
        self._connection = connection
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
        cursor = connection.cursor()
        return cls(connection, cursor)

    @classmethod
    def create(cls, db_path: Union[str, Path]) -> PostsDb:
        connection = sqlite3.connect(db_path)
        cursor = connection.cursor()
        cursor.executescript(
            """
            CREATE TABLE posts(
                id TEXT NOT NULL PRIMARY KEY,
                prediction INTEGER NOT NULL,
                probability REAL NOT NULL
            )
            """
        )

        return cls(connection, cursor)

    def insert(self, entry: PostsEntry) -> None:
        self._cursor.execute("INSERT INTO posts VALUES (?, ?, ?)", entry.as_db_entry())

        # This shouldn't be very performance sensitive so just commit on every insertion
        self._connection.commit()

    def find(self, desired_id: str) -> Optional[PostsEntry]:
        self._cursor.execute("SELECT * FROM posts WHERE id=?", (desired_id,))

        entries = self._cursor.fetchall()
        if len(entries) == 0:
            return None

        return PostsEntry.from_db_entry(entries[0])

    def close(self) -> None:
        self._cursor.close()
