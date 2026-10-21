from __future__ import annotations

from functools import lru_cache

from sqlalchemy import create_engine
from sqlalchemy.engine import Engine

from . import config


@lru_cache(maxsize=1)
def motor() -> Engine:
    return create_engine(config.DATABASE_URL, pool_size=5, max_overflow=5, pool_pre_ping=True)
