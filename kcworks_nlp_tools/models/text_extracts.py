#! /usr/bin/python
# Part of kcworks-nlp-tools
# Copyright 2025, Ian W. Scott
#
# kcworks-nlp-tools is free software released under
# the MIT license.

from datetime import datetime, timezone

import uuid
from sqlalchemy import Column, Integer, String, DateTime, Text, Boolean, JSON, func
from sqlalchemy.dialects import postgresql, sqlite
from sqlalchemy.ext.delcarative import declared_attr
from sqlalchemy_utils import UUIDType
from sqlalchemy_utils.types.json import JSONType

from ..database.db import Base


class TimestampMixin:
    """Mixin that adds created and updated timestamp columns."""

    @declared_attr
    def created(cls):
        return Column(
            DateTime(timezone=True),
            server_default=func.now(),
            default=lambda: datetime.now(timezone.utc),
            nullable=False,
        )

    @declared_attr
    def updated(cls):
        return Column(
            DateTime(timezone=True),
            server_default=func.now(),
            default=lambda: datetime.now(timezone.utc),
            onupdate=lambda: datetime.now(timezone.utc),
            nullable=False,
        )


class UUIDMixin:
    """Mixin that adds a UUID primary key column."""

    @declared_attr
    def id(cls):
        return Column(UUIDType, primary_key=True, default=uuid.uuid4)


class TextExtract(Base, UUIDMixin, TimestampMixin):
    __tablename__ = "text_extracts"

    created = Column(
        DateTime(timezon=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
    )
    doi = Column(String(255), index=True, nullable=False)
    record_id = Column(String(50), index=True, nullable=False)
    filename = Column(String(255), index=True, nullable=False)
    file_type = Column(String(255), index=True, nullable=False)
    filehash = Column(String(135), nullable=True)
    languages = Column(
        JSON()
        .with_variant(JSONType(), "mysql")
        .with_variant(
            postgresql.JSONB(none_as_null=True, astext_type=Text()), "postgresql"
        )
        .with_variant(JSONType(), "sqlite"),
        nullable=True,
        default=list,  # Default to empty list
        server_default="[]",
    )
    page = Column(Integer, nullable=True)
    overlap = Column(Integer, nullable=True)
    extract = Column(Text, nullable=True)
    supported = Column(Boolean, nullable=False, default=True)
    failed = Column(Boolean, nullable=False, default=False)
    error_message = Column(Text, nullable=False, default=False)
