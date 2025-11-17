"""Hard-negative sampling utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

from .entities import Entity, EntityType


@dataclass(frozen=True)
class HardNegativePair:
    text1: str
    text2: str
    entity_type: EntityType
    rationale: str


AMBIGUOUS_NAMES: dict[EntityType, Sequence[str]] = {
    EntityType.TRACK: ("Intro", "Home", "Stay", "Love", "Forever"),
    EntityType.ARTIST: ("Phoenix", "Muse", "Justice", "Freedom"),
    EntityType.ALBUM: ("Greatest Hits", "Live", "Untitled", "Abbey Road"),
}


class HardNegativeGenerator:
    """Produces similar-looking negative pairs."""

    def generate(self, entities: Iterable[Entity]) -> list[HardNegativePair]:
        pool = list(entities)
        pairs: list[HardNegativePair] = []
        for ent in pool:
            ambiguous_strings = AMBIGUOUS_NAMES.get(ent.entity_type, ())
            if ent.name not in ambiguous_strings:
                continue
            for other in pool:
                if other is ent:
                    continue
                if other.entity_type != ent.entity_type:
                    continue
                if other.name == ent.name:
                    pairs.append(
                        HardNegativePair(
                            text1=ent.name,
                            text2=other.name,
                            entity_type=ent.entity_type,
                            rationale="identical string, different canonical entity",
                        )
                    )
        return pairs
