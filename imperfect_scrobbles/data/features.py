"""Context feature extraction utilities."""

from __future__ import annotations

import datetime as dt
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Iterable

from .context_sim import Scrobble


@dataclass
class ContextFeatures:
    freq_text1: int
    freq_text2: int
    total_freq: int
    freq_ratio: float
    same_session: bool
    session_overlap: int
    session_overlap_ratio: float
    time_window_minutes: float
    co_tracks: list[str]
    co_track_count: int
    platform_hint: str | None
    platform_confidence: float
    recency_hours_text1: float
    recency_hours_text2: float
    user_overlap: int

    def as_dict(self) -> dict:
        return {
            "freq_text1": self.freq_text1,
            "freq_text2": self.freq_text2,
            "total_freq": self.total_freq,
            "freq_ratio": self.freq_ratio,
            "same_session": self.same_session,
            "session_overlap": self.session_overlap,
            "session_overlap_ratio": self.session_overlap_ratio,
            "time_window_minutes": self.time_window_minutes,
            "co_tracks": self.co_tracks,
            "co_track_count": self.co_track_count,
            "platform_hint": self.platform_hint,
            "platform_confidence": self.platform_confidence,
            "recency_hours_text1": self.recency_hours_text1,
            "recency_hours_text2": self.recency_hours_text2,
            "user_overlap": self.user_overlap,
        }


@dataclass
class _TrackAggregate:
    freq: int = 0
    last_seen: dt.datetime | None = None
    platforms: Counter = field(default_factory=Counter)
    sessions: set[str] = field(default_factory=set)
    users: set[str] = field(default_factory=set)


def compute_context_features(
    scrobbles: Iterable[Scrobble], text1: str, text2: str
) -> ContextFeatures:
    """Derive lightweight context statistics for a pair of strings."""

    track_stats = defaultdict(_TrackAggregate)
    sessions = defaultdict(list)
    reference_time: dt.datetime | None = None

    for scrobble in scrobbles:
        session_key = scrobble.session_id or f"{scrobble.user_id}_{scrobble.timestamp.date()}"
        sessions[session_key].append(scrobble)

        stats = track_stats[scrobble.track_name]
        stats.freq += 1
        stats.platforms[scrobble.platform] += 1
        stats.sessions.add(session_key)
        stats.users.add(scrobble.user_id)
        if stats.last_seen is None or scrobble.timestamp > stats.last_seen:
            stats.last_seen = scrobble.timestamp

        if reference_time is None or scrobble.timestamp > reference_time:
            reference_time = scrobble.timestamp

    stats1 = track_stats[text1]
    stats2 = track_stats[text2]
    freq1 = stats1.freq
    freq2 = stats2.freq
    total_freq = freq1 + freq2
    freq_ratio = (freq1 + 1) / (freq2 + 1)

    same_session = False
    session_overlap = 0
    min_delta = None
    co_tracks: set[str] = set()
    platform_votes = Counter()

    for items in sessions.values():
        names = [s.track_name for s in items]
        if text1 in names and text2 in names:
            same_session = True
            session_overlap += 1
            times1 = [s.timestamp for s in items if s.track_name == text1]
            times2 = [s.timestamp for s in items if s.track_name == text2]
            for t1 in times1:
                for t2 in times2:
                    delta = abs((t1 - t2).total_seconds() / 60)
                    if min_delta is None or delta < min_delta:
                        min_delta = delta
            for s in items:
                if s.track_name in (text1, text2):
                    platform_votes[s.platform] += 1
                else:
                    co_tracks.add(s.track_name)

    platform_hint = None
    platform_confidence = 0.0
    if platform_votes:
        platform_hint, votes = platform_votes.most_common(1)[0]
        platform_confidence = votes / sum(platform_votes.values())

    session_overlap_ratio = 0.0
    min_sessions = min(len(stats1.sessions) or 1, len(stats2.sessions) or 1)
    if min_sessions > 0:
        session_overlap_ratio = session_overlap / min_sessions

    reference_time = reference_time or dt.datetime.utcnow()

    def recency_hours(stats: _TrackAggregate) -> float:
        if stats.last_seen is None:
            return 9999.0
        return max(0.0, (reference_time - stats.last_seen).total_seconds() / 3600)

    user_overlap = len(stats1.users & stats2.users)

    return ContextFeatures(
        freq_text1=freq1,
        freq_text2=freq2,
        total_freq=total_freq,
        freq_ratio=freq_ratio,
        same_session=same_session,
        session_overlap=session_overlap,
        session_overlap_ratio=session_overlap_ratio,
        time_window_minutes=min_delta or 999.0,
        co_tracks=sorted(co_tracks),
        co_track_count=len(co_tracks),
        platform_hint=platform_hint,
        platform_confidence=platform_confidence,
        recency_hours_text1=recency_hours(stats1),
        recency_hours_text2=recency_hours(stats2),
        user_overlap=user_overlap,
    )
