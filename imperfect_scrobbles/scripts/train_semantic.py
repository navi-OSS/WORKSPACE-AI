"""Fine-tune the semantic sentence-transformer model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import numpy as np
from sentence_transformers import InputExample, SentenceTransformer, losses
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
from torch.utils.data import DataLoader


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/dataset.jsonl"))
    parser.add_argument("--model", type=str, default="sentence-transformers/paraphrase-multilingual-mpnet-base-v2")
    parser.add_argument("--output-dir", type=Path, default=Path("models/semantic"))
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--warmup-steps", type=int, default=500)
    return parser.parse_args()


def load_examples(path: Path, split: str) -> list[InputExample]:
    examples: list[InputExample] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            if row.get("split", "train") != split:
                continue
            text1 = row["text1"]
            text2 = row["text2"]
            label = float(row.get("label", 0))
            examples.append(InputExample(texts=[text1, text2], label=label))
    return examples


def main() -> None:
    args = parse_args()

    train_examples = load_examples(args.data, split="train")
    val_examples = load_examples(args.data, split="val")

    model = SentenceTransformer(args.model)

    train_loader = DataLoader(train_examples, shuffle=True, batch_size=args.batch_size)
    train_loss = losses.CosineSimilarityLoss(model)

    evaluator = EmbeddingSimilarityEvaluator.from_input_examples(val_examples)

    warmup_steps = args.warmup_steps or int(len(train_loader) * 0.1)
    model.fit(
        train_objectives=[(train_loader, train_loss)],
        evaluator=evaluator,
        epochs=args.epochs,
        warmup_steps=warmup_steps,
        optimizer_params={"lr": args.learning_rate},
        output_path=str(args.output_dir),
        show_progress_bar=True,
    )


if __name__ == "__main__":
    main()
