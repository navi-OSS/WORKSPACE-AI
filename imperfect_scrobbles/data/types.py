"""Shared type helpers for data generation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence


@dataclass(frozen=True)
class Variation:
    """Represents a single textual variation and its provenance."""

    text: str
    categories: Sequence[str] = ()
    metadata: Mapping[str, str] | None = None

    def type_label(self) -> str:
        if not self.categories:
            return "canonical"
        return "+".join(dict.fromkeys(self.categories))
