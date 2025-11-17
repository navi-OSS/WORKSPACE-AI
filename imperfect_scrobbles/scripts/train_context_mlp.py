"""Train a lightweight MLP to fuse semantic scores with context features."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from inference.matcher import ContextMLP


class ContextDataset(Dataset):
    def __init__(self, path: Path, split: str = "train"):
        self.samples: list[tuple[np.ndarray, float]] = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                if row.get("split", "train") != split:
                    continue
                context = row.get("context")
                if not context:
                    continue
                base_score = float(row.get("similarity", 0.5))
                features = self._context_vector(base_score, context)
                label = float(row.get("label", 0))
                self.samples.append((features, label))

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

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        features, label = self.samples[idx]
        return torch.from_numpy(features), torch.tensor(label, dtype=torch.float32)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/dataset.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("models/context/mlp.pt"))
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train_ds = ContextDataset(args.data, split="train")
    val_ds = ContextDataset(args.data, split="val")

    if not train_ds:
        raise RuntimeError("Dataset does not contain contextual samples")

    input_dim = train_ds[0][0].shape[0]
    model = ContextMLP(input_dim=input_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.BCELoss()

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size)

    best_val = float("inf")
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        for batch_x, batch_y in train_loader:
            preds = model(batch_x).squeeze(1)
            loss = loss_fn(preds, batch_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * batch_x.size(0)
        avg_train = total_loss / len(train_loader.dataset)

        model.eval()
        with torch.no_grad():
            val_loss = 0.0
            for batch_x, batch_y in val_loader:
                preds = model(batch_x).squeeze(1)
                loss = loss_fn(preds, batch_y)
                val_loss += loss.item() * batch_x.size(0)
        avg_val = val_loss / max(1, len(val_loader.dataset))
        print(f"Epoch {epoch}: train_loss={avg_train:.4f} val_loss={avg_val:.4f}")
        if avg_val < best_val:
            best_val = avg_val
            args.output.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"input_dim": input_dim, "state_dict": model.state_dict()}, args.output)


if __name__ == "__main__":
    main()
