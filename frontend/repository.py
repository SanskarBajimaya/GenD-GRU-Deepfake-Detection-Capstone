"""Persistence boundary for a future SQL implementation; disabled by default."""

from typing import Any, Protocol


class AnalysisRepository(Protocol):
    def save(self, record: dict[str, Any]) -> None: ...


class NullAnalysisRepository:
    """Privacy-first default: uploaded-video analyses are not persisted."""
    def save(self, record: dict[str, Any]) -> None:
        return None
