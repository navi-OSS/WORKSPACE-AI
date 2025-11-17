"""Core entity data structures and seed sampling utilities."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional


class EntityType(str, Enum):
    """Supported canonical entity categories."""

    TRACK = "track"
    ALBUM = "album"
    ARTIST = "artist"


@dataclass(frozen=True)
class Entity:
    """Represents a canonical entity plus optional metadata."""

    name: str
    entity_type: EntityType
    language: Optional[str] = None
    canonical_id: Optional[str] = None  # e.g., MusicBrainz ID


# Placeholder seed pools; to be filled programmatically or from external dumps.
DEFAULT_TRACKS = [
    "Bohemian Rhapsody",
    "Intro",
    "Untitled",
    "Home",
    "Stay",
    "Smooth Criminal",
    "Nandemonaiya",
    "宇宙よりも遠い場所",
    "紅豆",
    "삐딱하게",
    "Love Story",
    "Another One Bites the Dust",
    "We Found Love",
    "Bad Guy",
    "Lose Yourself",
    "Shape of You",
    "Faded",
    "Sugar Song and Bitter Step",
    "夜に駆ける",
    "君の知らない物語",
]

DEFAULT_ARTISTS = [
    "Hikaru Utada",
    "宇多田ヒカル",
    "Utada",
    "Phoenix",
    "BTS",
    "宇多田光",
    "Ayumi Hamasaki",
    "浜崎あゆみ",
    "Eason Chan",
    "陳奕迅",
    "Jay Chou",
    "周杰倫",
    "BLACKPINK",
    "방탄소년단",
    "TWICE",
    "Taylor Swift",
    "Queen",
    "Muse",
    "Sakanaction",
    "フレデリック",
]

DEFAULT_ALBUMS = [
    "Abbey Road",
    "Bohemian Rhapsody (Original Soundtrack)",
    "Boogiepop Phantom OST",
    "The Dark Side of the Moon",
    "Greatest Hits",
    "Thriller",
    "Divide",
    "Folklore",
    "Love Yourself 承 'Her'",
    "紅豆 EP",
    "宇宙よりも遠い場所 Original Soundtrack",
    "Made in Japan",
    "Nandemonaiya (Single)",
    "Live at Wembley Stadium",
    "Special Edition: Phoenix",
]


def iter_seed_entities() -> list[Entity]:
    """Return a baseline pool of entities for synthetic generation."""

    seeds: list[Entity] = []
    seeds.extend(Entity(name=t, entity_type=EntityType.TRACK) for t in DEFAULT_TRACKS)
    seeds.extend(Entity(name=a, entity_type=EntityType.ARTIST) for a in DEFAULT_ARTISTS)
    seeds.extend(Entity(name=a, entity_type=EntityType.ALBUM) for a in DEFAULT_ALBUMS)
    return seeds
