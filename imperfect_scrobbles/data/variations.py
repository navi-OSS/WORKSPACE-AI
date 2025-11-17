"""Synthetic variation helpers for tracks, albums, and general formatting."""

from __future__ import annotations

from dataclasses import dataclass
import random
from typing import Iterable, Sequence

from .entities import Entity, EntityType
from .types import Variation
from . import multilingual


@dataclass
class VariationConfig:
    """Configuration knobs for the variation generator."""

    max_suffixes: int = 2
    include_multilingual: bool = True
    max_variants: int = 8
    rng: random.Random = random.Random()


TRACK_SUFFIXES: Sequence[str] = (
    "Music Video",
    "Official Video",
    "Official Audio",
    "HD",
    "4K",
    "HQ",
    "Remastered",
    "Remastered 2019",
    "Live",
    "Live at Wembley Stadium",
    "Acoustic",
    "Acoustic Version",
    "Extended",
    "Extended Mix",
    "Radio Edit",
    "Album Version",
    "Single Version",
    "feat. Hikaru Utada",
    "ft. Phoenix",
    "featuring BTS",
    "with Beyoncé",
    "Instrumental",
    "Remix",
    "Demo",
    "Demo Version",
)


ALBUM_SUFFIXES: Sequence[str] = (
    "Deluxe Edition",
    "Deluxe",
    "Special Edition",
    "Expanded Edition",
    "Collector's Edition",
    "5th Anniversary Edition",
    "10th Anniversary Edition",
    "20th Anniversary Edition",
    "Remastered",
    "2019 Remaster",
    "Remastered 2019",
    "Japanese Edition",
    "Japan Bonus Tracks",
    "International Edition",
    "Explicit",
    "Explicit Version",
    "Clean Version",
)


SEPARATORS: Sequence[str] = (
    " - ",
    " – ",
    " — ",
    " : ",
    " (",
    " [",
    "",
)

PAREN_ENDINGS = {
    " (": ")",
    " [": "]",
}



class VariationGenerator:
    """Generates textual variations for entity names."""

    def __init__(self, config: VariationConfig | None = None):
        self.config = config or VariationConfig()
        self.rng = self.config.rng

    # ------------------------------------------------------------------
    def generate(self, entity: Entity, k: int | None = None) -> list[Variation]:
        """Generate up to k variations for a given entity."""

        limit = k or self.config.max_variants
        variants: dict[str, Variation] = {}

        if entity.entity_type == EntityType.TRACK:
            generator = self.track_variations
        elif entity.entity_type == EntityType.ALBUM:
            generator = self.album_variations
        else:
            generator = self.artist_variations

        for variant in generator(entity.name, limit * 2):
            if variant.text == entity.name:
                continue
            variants.setdefault(variant.text, variant)
            if len(variants) >= limit:
                break

        if self.config.include_multilingual:
            for variant in multilingual.generate_variants(entity.name):
                if variant.text == entity.name:
                    continue
                variants.setdefault(variant.text, variant)
                if len(variants) >= limit:
                    break

        return list(variants.values())[:limit]

    # ------------------------------------------------------------------
    def track_variations(self, base: str, budget: int) -> list[Variation]:
        return self._suffix_variations(base, TRACK_SUFFIXES, "track_suffix", budget)

    def album_variations(self, base: str, budget: int) -> list[Variation]:
        return self._suffix_variations(base, ALBUM_SUFFIXES, "album_suffix", budget)

    def artist_variations(self, base: str, budget: int) -> list[Variation]:
        variants: list[Variation] = []
        upper = base.upper()
        if upper != base:
            variants.append(Variation(text=upper, categories=["artist_case", "upper"]))
        lower = base.lower()
        if lower != base:
            variants.append(Variation(text=lower, categories=["artist_case", "lower"]))
        if " " in base:
            parts = base.split(" ")
            swapped = " ".join(reversed(parts))
            if swapped != base:
                variants.append(Variation(text=swapped, categories=["artist_reorder"]))
        return variants[:budget]

    # ------------------------------------------------------------------
    def _suffix_variations(
        self,
        base: str,
        pool: Sequence[str],
        category: str,
        budget: int,
    ) -> list[Variation]:
        variants: list[Variation] = []
        attempts = 0
        while len(variants) < budget and attempts < budget * 5:
            attempts += 1
            text, cats = self._apply_suffixes(base, pool)
            text, fmt_cats = self._apply_formatting_noise(text)
            cats = [category] + cats + fmt_cats
            if text == base:
                continue
            variants.append(Variation(text=text, categories=cats))
        return variants

    def _apply_suffixes(self, base: str, pool: Sequence[str]) -> tuple[str, list[str]]:
        variant = base
        categories: list[str] = []
        suffix_count = self.rng.randint(1, self.config.max_suffixes)
        for _ in range(suffix_count):
            suffix = self.rng.choice(pool)
            variant = self._join_with_separator(variant, suffix)
            categories.append(f"suffix:{self._slug(suffix)}")
        return variant, categories

    def _join_with_separator(self, left: str, right: str) -> str:
        sep = self.rng.choice(SEPARATORS)
        if sep in (" (", " ["):
            return f"{left}{sep}{right}{PAREN_ENDINGS[sep]}"
        if sep == "":
            return f"{left}{right}"
        return f"{left}{sep}{right}"

    def _apply_formatting_noise(self, text: str) -> tuple[str, list[str]]:
        categories: list[str] = []
        text, applied = self._random_spacing(text)
        if applied:
            categories.append("format:spacing")
        text, applied = self._ampersand_vs_and(text)
        if applied:
            categories.append("format:ampersand")
        text, applied = self._numbers_vs_words(text)
        if applied:
            categories.append("format:numeric")
        return text, categories

    # ------------------------------------------------------------------
    def _random_spacing(self, text: str) -> tuple[str, bool]:
        if self.rng.random() < 0.2:
            return text.replace(" ", "  "), True
        if self.rng.random() < 0.2:
            return text.strip(), True
        return text, False

    def _ampersand_vs_and(self, text: str) -> tuple[str, bool]:
        if " & " in text and self.rng.random() < 0.5:
            return text.replace(" & ", " and "), True
        if " and " in text and self.rng.random() < 0.5:
            return text.replace(" and ", " & "), True
        return text, False

    def _numbers_vs_words(self, text: str) -> tuple[str, bool]:
        replacements = {
            " 2 ": " Two ",
            " 3 ": " Three ",
            " II": " 2",
        }
        for needle, repl in replacements.items():
            if needle in text and self.rng.random() < 0.3:
                return text.replace(needle, repl), True
        return text, False

    def _slug(self, text: str) -> str:
        return text.lower().replace(" ", "_")


def cartesian_product(strings: Iterable[str]) -> list[tuple[str, str]]:
    """Utility for pairing all combinations of strings."""

    strings = list(strings)
    pairs: list[tuple[str, str]] = []
    for i, s1 in enumerate(strings):
        for j, s2 in enumerate(strings):
            if i < j:
                pairs.append((s1, s2))
    return pairs
