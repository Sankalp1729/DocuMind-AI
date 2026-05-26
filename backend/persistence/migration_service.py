from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import text

from backend.persistence.migrations.versions import MIGRATIONS
from backend.persistence.models import SchemaMigration


class MigrationService:
    def __init__(self, engine):
        self.engine = engine

    def _ensure_migration_table(self) -> None:
        SchemaMigration.__table__.create(bind=self.engine, checkfirst=True)

    def _applied_versions(self) -> set[str]:
        with self.engine.connect() as connection:
            rows = connection.execute(text("SELECT version FROM schema_migrations")).fetchall()
        return {row[0] for row in rows}

    def apply_pending(self) -> list[str]:
        self._ensure_migration_table()
        applied_versions = self._applied_versions()
        applied_now: list[str] = []

        for version, migration in MIGRATIONS:
            if version in applied_versions:
                continue
            migration(self.engine)
            with self.engine.begin() as connection:
                connection.execute(
                    text("INSERT INTO schema_migrations (version, applied_at) VALUES (:version, :applied_at)"),
                    {"version": version, "applied_at": datetime.now(timezone.utc)},
                )
            applied_now.append(version)

        return applied_now
