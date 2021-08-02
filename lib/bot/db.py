from __future__ import annotations

from typing import Optional, Text, cast

from sqlalchemy import Column, Enum, Float, String, create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import Session
from sqlalchemy.orm.session import sessionmaker

from lib.classifier.datasets import Category

Base = declarative_base()


class PostEntry(Base):
    __tablename__ = "post"

    id: Column[Text] = Column(String, primary_key=True)
    prediction = Column(Enum(Category))
    probability: Column[Float] = Column(Float)
    title: Column[String] = Column(String)
    body: Column[String] = Column(String)


class Db:
    _session: Session

    def __init__(self, conn_string: str) -> None:
        engine = create_engine(conn_string)
        Session = sessionmaker(bind=engine)
        self._session = Session()
        Base.metadata.create_all(engine)

    def insert(self, entry: PostEntry) -> None:
        # We don't really care about performance for this amount of traffic
        self._session.add(entry)
        self._session.commit()

    def find(self, desired_id: str) -> Optional[PostEntry]:
        result = cast(
            Optional[PostEntry],
            self._session.query(PostEntry).filter_by(id=desired_id).first(),
        )

        return result

    def close(self) -> None:
        self._session.close()
