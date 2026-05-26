from __future__ import annotations

from contextlib import contextmanager

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from backend.core.config import DB_ECHO, DB_MAX_OVERFLOW, DB_POOL_SIZE, DATABASE_URL
from backend.persistence.migration_service import MigrationService


class DatabaseService:
    def __init__(self, database_url: str = DATABASE_URL):
        self.database_url = database_url
        engine_kwargs = {"future": True, "echo": DB_ECHO, "pool_pre_ping": True}
        if database_url.startswith("sqlite"):
            engine_kwargs["connect_args"] = {"check_same_thread": False}
        else:
            engine_kwargs["pool_size"] = DB_POOL_SIZE
            engine_kwargs["max_overflow"] = DB_MAX_OVERFLOW

        self.engine = create_engine(database_url, **engine_kwargs)
        self.session_factory = sessionmaker(bind=self.engine, autocommit=False, autoflush=False, expire_on_commit=False)
        self.migrations = MigrationService(self.engine)
        self.migrations.apply_pending()

    @contextmanager
    def session_scope(self):
        session: Session = self.session_factory()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()
