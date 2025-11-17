"""Runtime matcher combining semantic similarity and optional context boosts."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from sentence_transformers import SentenceTransformer


@dataclass
class RuleBoostConfig:
    freq_ratio_boost: float = 0.10
    same_session_boost: float = 0.05
    co_tracks_boost: float = 0.08
    platform_boost: float = 0.05
    conflict_penalty: float = 0.05
    platform_hint_sources: tuple[str, ...] = ("youtube_music", "soundcloud")


class ContextMLP(torch.nn.Module):
    """Simple feed-forward head to fuse context features."""

    def __init__(self, input_dim: int, hidden_dim: int = 32):
        super().__init__()
        self.network = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, 1),
            torch.nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class ImperfectScrobbleMatcher:
    """High-level API that exposes score(text1, text2, context?)."""

    def __init__(
        self,
        semantic_model_path: str | Path,
        rule_config: RuleBoostConfig | None = None,
        context_mlp_path: str | Path | None = None,
    ) -> None:
        self.semantic_model = SentenceTransformer(str(semantic_model_path))
        self.rule_config = rule_config or RuleBoostConfig()
        self.context_head: ContextMLP | None = None
        if context_mlp_path:
            state = torch.load(context_mlp_path, map_location="cpu")
            input_dim = state["input_dim"]
            model_state = state["state_dict"]
            self.context_head = ContextMLP(input_dim=input_dim)
            self.context_head.load_state_dict(model_state)
            self.context_head.eval()

    # ------------------------------------------------------------------
    def score(self, text1: str, text2: str, context: dict | None = None) -> float:
        base_score = self._semantic_similarity(text1, text2)
        if context is None:
            return base_score
        boosted = self._rule_boost(base_score, context)
        if self.context_head:
            mlp_score = self._mlp_adjust(base_score, context)
            return float((boosted + mlp_score) / 2)
        return boosted

    def batch_score(
        self, pairs: Iterable[tuple[str, str]], contexts: Iterable[dict | None] | None = None
    ) -> list[float]:
        texts_a, texts_b = zip(*pairs)
        embeddings_a = self.semantic_model.encode(list(texts_a), convert_to_tensor=True)
        embeddings_b = self.semantic_model.encode(list(texts_b), convert_to_tensor=True)
        sims = torch.nn.functional.cosine_similarity(embeddings_a, embeddings_b)
        scores: list[float] = sims.detach().cpu().tolist()
        if contexts is None:
            return scores
        adjusted = []
        for score, ctx in zip(scores, contexts):
            adjusted.append(self.score_from_base(score, ctx))
        return adjusted

    def score_from_base(self, base_score: float, context: dict | None) -> float:
        if context is None:
            return base_score
        boosted = self._rule_boost(base_score, context)
        if self.context_head:
            mlp_score = self._mlp_adjust(base_score, context)
            return float((boosted + mlp_score) / 2)
        return boosted

    # ------------------------------------------------------------------
    def _semantic_similarity(self, text1: str, text2: str) -> float:
        emb = self.semantic_model.encode([text1, text2], convert_to_tensor=True)
        score = torch.nn.functional.cosine_similarity(emb[0], emb[1], dim=0).item()
        return max(0.0, min(1.0, (score + 1) / 2))  # map [-1,1] -> [0,1]

    def _rule_boost(self, base_score: float, context: dict) -> float:
        cfg = self.rule_config
        boost = 0.0
        penalty = 0.0

        f1 = context.get("freq_text1", 0)
        f2 = context.get("freq_text2", 0)
        if f1 and f2:
            ratio = max(f1, f2) / max(1, min(f1, f2))
            if ratio > 10:
                boost += cfg.freq_ratio_boost
        if context.get("same_session"):
            boost += cfg.same_session_boost
        if len(context.get("co_tracks", []) or []) > 3:
            boost += cfg.co_tracks_boost
        if context.get("platform_hint") in cfg.platform_hint_sources:
            boost += cfg.platform_boost
        if f1 > 20 and f2 > 20:
            penalty += cfg.conflict_penalty
        return float(np.clip(base_score + boost - penalty, 0.0, 1.0))

    def _mlp_adjust(self, base_score: float, context: dict) -> float:
        assert self.context_head is not None
        features = self._context_vector(base_score, context)
        with torch.no_grad():
            tensor = torch.from_numpy(features).unsqueeze(0).float()
            return float(self.context_head(tensor).item())

    def _context_vector(self, base_score: float, context: dict) -> np.ndarray:
        freq1 = context.get("freq_text1", 0)
        freq2 = context.get("freq_text2", 0)
        log_ratio = np.log((freq1 + 1) / (freq2 + 1))
        same_session = 1.0 if context.get("same_session") else 0.0
        co_tracks = len(context.get("co_tracks", []) or [])
        time_window = context.get("time_window_minutes", 999.0)
        platform = context.get("platform_hint") or "unknown"
        platforms = ["spotify", "youtube_music", "apple_music", "soundcloud", "unknown"]
        platform_vec = [1.0 if platform == p else 0.0 for p in platforms]
        vec = [
            base_score,
            freq1,
            freq2,
            log_ratio,
            same_session,
            co_tracks,
            time_window,
        ] + platform_vec
        return np.asarray(vec, dtype=np.float32)
