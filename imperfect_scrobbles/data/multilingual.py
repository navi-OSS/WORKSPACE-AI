"""Multilingual variation helpers for artist/track/album names."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Callable, Iterable

from unidecode import unidecode

from .types import Variation

try:
    import pykakasi
except ImportError:  # pragma: no cover - optional dependency
    pykakasi = None

try:
    import cutlet
except ImportError:  # pragma: no cover
    cutlet = None

try:
    import pypinyin
except ImportError:  # pragma: no cover
    pypinyin = None

try:
    from korean_romanizer.romanizer import Romanizer
except ImportError:  # pragma: no cover
    Romanizer = None


@dataclass(frozen=True)
class Detector:
    name: str
    predicate: Callable[[str], bool]


_JP_RANGES = ((0x3040, 0x30FF), (0x4E00, 0x9FFF))
_CN_RANGES = ((0x4E00, 0x9FFF),)
_KR_RANGES = ((0xAC00, 0xD7A3),)


def _contains_range(text: str, ranges: Iterable[tuple[int, int]]) -> bool:
    for char in text:
        code = ord(char)
        for start, end in ranges:
            if start <= code <= end:
                return True
    return False


def contains_japanese(text: str) -> bool:
    return _contains_range(text, _JP_RANGES)


def contains_chinese(text: str) -> bool:
    return _contains_range(text, _CN_RANGES)


def contains_korean(text: str) -> bool:
    return _contains_range(text, _KR_RANGES)


def generate_variants(text: str) -> list[Variation]:
    """Generate language-aware transliterations and formatting variants."""

    variants: dict[str, Variation] = {}
    if contains_japanese(text):
        for variant in _japanese_variants(text):
            variants.setdefault(variant.text, variant)
    if contains_chinese(text):
        for variant in _chinese_variants(text):
            variants.setdefault(variant.text, variant)
    if contains_korean(text):
        for variant in _korean_variants(text):
            variants.setdefault(variant.text, variant)
    for variant in _generic_variants(text):
        variants.setdefault(variant.text, variant)
    return list(variants.values())


def _japanese_variants(text: str) -> list[Variation]:
    results: list[Variation] = []
    if pykakasi:
        kk = pykakasi.kakasi()
        kk.setMode("J", "a")
        conv = kk.getConverter()
        roman = conv.do(text)
        results.append(Variation(text=roman.title(), categories=["multilingual_jp", "hepburn"]))
        results.append(Variation(text=roman.upper(), categories=["multilingual_jp", "upper"]))
    katsu = _safe_cutlet()
    if katsu is not None:
        try:
            romaji = katsu.romaji(text)
        except RuntimeError:
            romaji = None
        if romaji:
            results.append(Variation(text=romaji, categories=["multilingual_jp", "cutlet"]))
    return results


def _chinese_variants(text: str) -> list[Variation]:
    results: list[Variation] = []
    if not pypinyin:
        return results
    with_tones = " ".join(pypinyin.lazy_pinyin(text, style=pypinyin.Style.TONE3))
    without = " ".join(pypinyin.lazy_pinyin(text, style=pypinyin.Style.NORMAL))
    results.append(Variation(text=with_tones, categories=["multilingual_cn", "tone"]))
    results.append(Variation(text=without, categories=["multilingual_cn", "no_tone"]))
    results.append(Variation(text=without.title(), categories=["multilingual_cn", "title"]))
    return results


def _korean_variants(text: str) -> list[Variation]:
    results: list[Variation] = []
    if Romanizer is None:
        return results
    roman = Romanizer(text).romanize()
    results.append(Variation(text=roman, categories=["multilingual_kr", "romanized"]))
    results.append(Variation(text=roman.upper(), categories=["multilingual_kr", "upper"]))
    return results


def _generic_variants(text: str) -> list[Variation]:
    variants = [
        Variation(text=unidecode(text), categories=["format", "unidecode"]),
        Variation(text=text.lower(), categories=["format", "lowercase"]),
        Variation(text=text.upper(), categories=["format", "uppercase"]),
        Variation(text=text.title(), categories=["format", "titlecase"]),
        Variation(text=text.replace(" ", ""), categories=["format", "no_space"]),
    ]
    return [v for v in variants if v.text and v.text != text]


@lru_cache(maxsize=1)
def _safe_cutlet():
    if cutlet is None:
        return None
    try:
        return cutlet.Cutlet()
    except RuntimeError:
        return None
