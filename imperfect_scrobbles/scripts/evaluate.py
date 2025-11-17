"""Evaluation utilities for semantic vs context-aware scoring."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import numpy as np
from sklearn.metrics import classification_report, precision_recall_curve

from inference.matcher import ImperfectScrobbleMatcher


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/dataset.jsonl"))
    parser.add_argument("--semantic-checkpoint", type=Path, required=True)
    parser.add_argument("--context-head", type=Path)
    parser.add_argument("--output", type=Path, default=Path("reports/eval.json"))
    return parser.parse_args()


def load_samples(path: Path, split: str = "test") -> list[dict]:
    samples: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            if row.get("split", "train") == split:
                samples.append(row)
    return samples


def evaluate_split(samples: Iterable[dict], matcher: ImperfectScrobbleMatcher) -> dict:
    y_true: list[int] = []
    scores_semantic: list[float] = []
    scores_context: list[float] = []

    for row in samples:
        text1 = row["text1"]
        text2 = row["text2"]
        label = int(row.get("label", 0))
        context = row.get("context")
        base = matcher.score_from_base(matcher._semantic_similarity(text1, text2), None)
        y_true.append(label)
        scores_semantic.append(base)
        if context:
            scores_context.append(matcher.score_from_base(base, context))
        else:
            scores_context.append(base)

    report_semantic = classification_report(y_true, threshold(scores_semantic), output_dict=True)
    report_context = classification_report(y_true, threshold(scores_context), output_dict=True)

    pr_semantic = precision_recall_curve(y_true, scores_semantic)
    pr_context = precision_recall_curve(y_true, scores_context)

    return {
        "semantic": report_semantic,
        "context": report_context,
        "pr_semantic": _curve_to_list(pr_semantic),
        "pr_context": _curve_to_list(pr_context),
    }


def threshold(scores: list[float], t: float = 0.5) -> list[int]:
    return [int(s >= t) for s in scores]


def _curve_to_list(curve):
    precision, recall, thresholds = curve
    return {
        "precision": precision.tolist(),
        "recall": recall.tolist(),
        "thresholds": thresholds.tolist(),
    }


def main() -> None:
    args = parse_args()
    matcher = ImperfectScrobbleMatcher(
        semantic_model_path=args.semantic_checkpoint,
        context_mlp_path=args.context_head,
    )
    samples = load_samples(args.data, split="test")
    metrics = evaluate_split(samples, matcher)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved evaluation metrics to {args.output}")


if __name__ == "__main__":
    main()
