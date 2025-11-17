#! /usr/bin/python
# Part of kcworks-nlp-tools
# Copyright 2025 Ian W. Scott
#
# kcworks-nlp-tools is free software realeased under
# the MIT license.

"""
Basic sqlite database for extracted text storage.
"""

from contextlib import contextmanager

from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.org import sessionmaker

Base = declarative_base()


engine = create_engine("sqlite://text_extracts.db", echo=True)

Base.metadata.create_all(engine)

Session = sessionmaker(bind=engine)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


@contextmanager
def get_db(commit: bool = True):
    """Get a database session with exception handling and cleanup."""
    db = SessionLocal()
    try:
        yield db
        if commit:
            db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()
