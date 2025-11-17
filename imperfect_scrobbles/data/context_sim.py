"""User profile and scrobble simulation for contextual signals."""

from __future__ import annotations

import dataclasses
import datetime as dt
import random
from typing import Iterable, Literal, Sequence

from .entities import Entity, EntityType
from .variations import VariationGenerator, VariationConfig
from . import multilingual

Platform = Literal["spotify", "youtube_music", "apple_music", "soundcloud"]


@dataclasses.dataclass
class SimulationConfig:
    scrobbles_per_user: tuple[int, int] = (500, 2000)
    imperfect_ratio: float = 0.65
    multilingual_mix: float = 0.7


@dataclasses.dataclass
class Scrobble:
    user_id: str
    session_id: str
    timestamp: dt.datetime
    track_name: str
    artist_name: str
    album_name: str | None
    platform: Platform


@dataclasses.dataclass
class UserProfile:
    user_id: str
    archetype: Literal[
        "album_listener",
        "playlist_shuffler",
        "artist_superfan",
        "multi_platform",
        "multilingual",
    ]
    dominant_script: str | None = None


class ContextSimulator:
    """Generates synthetic scrobble histories with realistic noise."""

    def __init__(
        self,
        rng: random.Random | None = None,
        config: SimulationConfig | None = None,
    ):
        self.rng = rng or random.Random()
        self.config = config or SimulationConfig()
        self.variation = VariationGenerator(VariationConfig(rng=self.rng))
        self._variant_cache: dict[str, list[str]] = {}

    def simulate_users(
        self,
        entities: Iterable[Entity],
        count: int = 1000,
        scrobbles_per_user: tuple[int, int] | None = None,
    ) -> tuple[list[UserProfile], list[Scrobble]]:
        profiles: list[UserProfile] = []
        scrobbles: list[Scrobble] = []
        pools = self._split_pools(list(entities))
        archetypes = (
            "album_listener",
            "playlist_shuffler",
            "artist_superfan",
            "multi_platform",
            "multilingual",
        )
        low, high = scrobbles_per_user or self.config.scrobbles_per_user
        for idx in range(count):
            archetype = self.rng.choice(archetypes)
            profile = UserProfile(
                user_id=f"user_{idx:04d}",
                archetype=archetype,
                dominant_script="jp" if archetype == "multilingual" else None,
            )
            profiles.append(profile)
            target = self.rng.randint(low, high)
            scrobbles.extend(self._simulate_scrobbles(profile, pools, target))
        return profiles, scrobbles

    # ------------------------------------------------------------------
    def _simulate_scrobbles(
        self,
        profile: UserProfile,
        pools: dict[EntityType, list[Entity]],
        target: int,
    ) -> list[Scrobble]:
        scrobbles: list[Scrobble] = []
        session_idx = 0
        while len(scrobbles) < target:
            session_length = self.rng.randint(8, 40)
            scrobbles.extend(
                self._simulate_session(
                    profile,
                    pools,
                    session_idx=session_idx,
                    session_length=session_length,
                )
            )
            session_idx += 1
        return scrobbles[:target]

    def _simulate_session(
        self,
        profile: UserProfile,
        pools: dict[EntityType, list[Entity]],
        session_idx: int,
        session_length: int,
    ) -> list[Scrobble]:
        session: list[Scrobble] = []
        track_pool = pools.get(EntityType.TRACK, [])
        artist_pool = pools.get(EntityType.ARTIST, [])
        album_pool = pools.get(EntityType.ALBUM, [])
        start = dt.datetime.utcnow() - dt.timedelta(
            days=self.rng.randint(0, 60),
            hours=self.rng.randint(0, 23),
            minutes=self.rng.randint(0, 59),
        )
        duration = dt.timedelta(minutes=self.rng.randint(20, 120))
        artist_bias = self._choose_artist_bias(profile, artist_pool)
        album_bias = self._choose_album_bias(profile, album_pool)

        for i in range(session_length):
            platform = self._pick_platform(profile)
            track = self._pick_entity(track_pool)
            artist = self._pick_artist(artist_pool, artist_bias, profile)
            album = self._pick_album(album_pool, album_bias, profile)
            scrobble = Scrobble(
                user_id=profile.user_id,
                session_id=f"{profile.user_id}_sess_{session_idx:03d}",
                timestamp=start + i * (duration / max(session_length, 1)),
                track_name=self._render_track_name(track, platform, profile),
                artist_name=self._render_artist_name(artist, profile),
                album_name=self._render_album_name(album, profile),
                platform=platform,
            )
            session.append(scrobble)
        return session

    # ------------------------------------------------------------------
    def _split_pools(self, entities: list[Entity]) -> dict[EntityType, list[Entity]]:
        pools: dict[EntityType, list[Entity]] = {etype: [] for etype in EntityType}
        for entity in entities:
            pools[entity.entity_type].append(entity)
        return pools

    def _pick_entity(self, pool: Sequence[Entity]) -> Entity | None:
        if not pool:
            return None
        return self.rng.choice(pool)

    def _pick_artist(
        self,
        pool: Sequence[Entity],
        bias: Entity | None,
        profile: UserProfile,
    ) -> Entity | None:
        if bias and self.rng.random() < 0.8:
            return bias
        if profile.archetype == "playlist_shuffler" and self.rng.random() < 0.3:
            return None
        return self._pick_entity(pool)

    def _pick_album(
        self,
        pool: Sequence[Entity],
        bias: Entity | None,
        profile: UserProfile,
    ) -> Entity | None:
        if profile.archetype == "album_listener" and bias:
            return bias
        if bias and self.rng.random() < 0.6:
            return bias
        return self._pick_entity(pool)

    def _choose_artist_bias(
        self, profile: UserProfile, pool: Sequence[Entity]
    ) -> Entity | None:
        if not pool:
            return None
        if profile.archetype in {"artist_superfan", "multilingual"}:
            return self.rng.choice(pool)
        if self.rng.random() < 0.2:
            return self.rng.choice(pool)
        return None

    def _choose_album_bias(
        self, profile: UserProfile, pool: Sequence[Entity]
    ) -> Entity | None:
        if not pool:
            return None
        if profile.archetype == "album_listener":
            return self.rng.choice(pool)
        if self.rng.random() < 0.2:
            return self.rng.choice(pool)
        return None

    def _render_track_name(
        self, entity: Entity | None, platform: Platform, profile: UserProfile
    ) -> str:
        if entity is None:
            return "Unknown Track"
        name = entity.name
        if self.rng.random() < self.config.imperfect_ratio:
            name = self._sample_variant(entity)
        return self._apply_platform_noise(name, platform)

    def _render_artist_name(self, entity: Entity | None, profile: UserProfile) -> str:
        if entity is None:
            return "Various Artists"
        name = entity.name
        if profile.archetype == "multilingual" and self.rng.random() < self.config.multilingual_mix:
            variants = multilingual.generate_variants(name)
            if variants:
                name = self.rng.choice(variants).text
        elif self.rng.random() < 0.3:
            name = name.upper() if self.rng.random() < 0.5 else name.lower()
        return name

    def _render_album_name(self, entity: Entity | None, profile: UserProfile) -> str | None:
        if entity is None:
            return None
        name = entity.name
        if profile.archetype in {"album_listener", "multi_platform"} and self.rng.random() < 0.6:
            name = self._sample_variant(entity)
        return name

    def _apply_platform_noise(self, text: str, platform: Platform) -> str:
        if platform == "youtube_music" and "-" not in text and self.rng.random() < 0.4:
            return f"{text} - Music Video"
        if platform == "soundcloud" and self.rng.random() < 0.3:
            return f"{text} (Demo)"
        if platform == "apple_music" and self.rng.random() < 0.2:
            return f"{text} [Clean]"
        return text

    def _sample_variant(self, entity: Entity) -> str:
        cache = self._variant_cache.get(entity.name)
        if cache is None:
            generated = self.variation.generate(entity, k=6)
            cache = [v.text for v in generated] or [entity.name]
            self._variant_cache[entity.name] = cache
        return self.rng.choice(cache)

    def _pick_platform(self, profile: UserProfile) -> Platform:
        platforms: Sequence[Platform] = (
            "spotify",
            "youtube_music",
            "apple_music",
            "soundcloud",
        )
        if profile.archetype == "multi_platform":
            return self.rng.choice(platforms)
        if profile.archetype == "playlist_shuffler" and self.rng.random() < 0.2:
            return self.rng.choice(platforms)
        return "spotify"
