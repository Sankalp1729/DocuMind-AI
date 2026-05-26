from __future__ import annotations

from backend.persistence.models import Base


def apply_initial_schema(engine) -> None:
    Base.metadata.create_all(bind=engine)


MIGRATIONS = [
    ("0001_initial", apply_initial_schema),
]
