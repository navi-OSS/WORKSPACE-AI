"""CLI for building the synthetic imperfect-scrobbles dataset."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
import random
from typing import Iterable

import yaml

from data.entities import Entity, EntityType, iter_seed_entities
from data.variations import VariationGenerator, VariationConfig
from data.hard_negatives import HardNegativeGenerator
from data.context_sim import ContextSimulator, SimulationConfig
from data.features import compute_context_features


DEFAULT_TOTAL = 5000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("data_pairs.jsonl"))
    parser.add_argument("--with-context", action="store_true")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--config", type=Path, help="Optional YAML config overriding CLI flags")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    seed = int(cfg.get("seed", args.seed))
    rng = random.Random(seed)

    output_path = Path(cfg.get("output_path", args.output))
    with_context = bool(cfg.get("with_context", args.with_context))
    total_pairs = int(cfg.get("total_pairs", DEFAULT_TOTAL))
    ratios = {
        "positive": float(cfg.get("positive_ratio", 0.7)),
        "negative": float(cfg.get("negative_ratio", 0.2)),
        "hard_negative": float(cfg.get("hard_negative_ratio", 0.1)),
    }
    normalize_ratios(ratios)

    entities = iter_seed_entities()
    variation = VariationGenerator(VariationConfig(rng=rng))

    positive_candidates = build_positive_candidates(entities, variation)
    random_negatives = build_random_negatives(entities, variation, rng, samples=max(total_pairs, 2000))
    hard_negatives = build_hard_negative_dicts(entities)

    counts = compute_counts(total_pairs, ratios)
    dataset: list[dict] = []
    dataset += sample_pool(positive_candidates, counts["positive"], rng)
    dataset += sample_pool(random_negatives, counts["negative"], rng)
    dataset += sample_pool(hard_negatives, counts["hard_negative"], rng)

    split_cfg = cfg.get("train_val_test_split")
    dataset = assign_splits(dataset, split_cfg)

    if with_context:
        sim_params = cfg.get("simulation", {})
        sim_cfg = SimulationConfig(
            scrobbles_per_user=tuple(sim_params.get("scrobbles_per_user", (500, 2000))),
            imperfect_ratio=float(sim_params.get("imperfect_ratio", 0.65)),
            multilingual_mix=float(sim_params.get("multilingual_mix", 0.7)),
        )
        num_users = int(sim_params.get("num_users", cfg.get("num_users", 500)))
        simulator = ContextSimulator(rng=rng, config=sim_cfg)
        _, scrobbles = simulator.simulate_users(
            entities,
            count=num_users,
            scrobbles_per_user=sim_cfg.scrobbles_per_user,
        )
        for row in dataset:
            ctx = compute_context_features(scrobbles, row["text1"], row["text2"])
            row["context"] = ctx.as_dict()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for row in dataset:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"Wrote {len(dataset)} pairs to {output_path}")


def load_config(path: Path | None) -> dict:
    if not path:
        return {}
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError("Config file must define a YAML mapping")
    return data


def build_positive_candidates(entities: Iterable[Entity], variation: VariationGenerator) -> list[dict]:
    candidates: list[dict] = []
    for entity in entities:
        for variant in variation.generate(entity, k=8):
            candidates.append(
                {
                    "text1": entity.name,
                    "text2": variant.text,
                    "label": 1,
                    "entity_type": entity.entity_type.value,
                    "type": variant.type_label(),
                }
            )
    return candidates


def build_random_negatives(
    entities: Iterable[Entity],
    variation: VariationGenerator,
    rng: random.Random,
    samples: int,
) -> list[dict]:
    pools: dict[EntityType, list[Entity]] = defaultdict(list)
    for entity in entities:
        pools[entity.entity_type].append(entity)
    rows: list[dict] = []
    entity_types = [etype for etype, pool in pools.items() if len(pool) >= 2]
    if not entity_types:
        return rows
    for _ in range(samples):
        etype = rng.choice(entity_types)
        a, b = rng.sample(pools[etype], 2)
        rows.append(
            {
                "text1": pick_surface(a, variation, rng),
                "text2": pick_surface(b, variation, rng),
                "label": 0,
                "entity_type": etype.value,
                "type": "random_negative",
                "rationale": "different canonical entities",
            }
        )
    return rows


def build_hard_negative_dicts(entities: Iterable[Entity]) -> list[dict]:
    rows: list[dict] = []
    for hn in HardNegativeGenerator().generate(entities):
        rows.append(
            {
                "text1": hn.text1,
                "text2": hn.text2,
                "label": 0,
                "entity_type": hn.entity_type.value,
                "type": "hard_negative",
                "rationale": hn.rationale,
            }
        )
    return rows


def pick_surface(entity: Entity, variation: VariationGenerator, rng: random.Random) -> str:
    if rng.random() < 0.5:
        variants = variation.generate(entity, k=3)
        if variants:
            return rng.choice(variants).text
    return entity.name


def normalize_ratios(ratios: dict[str, float]) -> None:
    total = sum(ratios.values())
    if total == 0:
        raise ValueError("At least one ratio must be positive")
    for key in ratios:
        ratios[key] = ratios[key] / total


def compute_counts(total_pairs: int, ratios: dict[str, float]) -> dict[str, int]:
    counts = {key: int(total_pairs * value) for key, value in ratios.items()}
    remainder = total_pairs - sum(counts.values())
    if remainder > 0:
        counts["positive"] += remainder
    return counts


def sample_pool(pool: list[dict], count: int, rng: random.Random) -> list[dict]:
    if not pool or count <= 0:
        return []
    return [dict(rng.choice(pool)) for _ in range(count)]


def assign_splits(rows: list[dict], ratios: dict[str, float] | None = None) -> list[dict]:
    ratios = ratios or {"train": 0.7, "val": 0.15, "test": 0.15}
    rows = list(rows)
    rng = random.Random(42)
    rng.shuffle(rows)
    total = len(rows)
    train_end = int(total * ratios.get("train", 0.7))
    val_end = train_end + int(total * ratios.get("val", 0.15))
    for idx, row in enumerate(rows):
        if idx < train_end:
            row["split"] = "train"
        elif idx < val_end:
            row["split"] = "val"
        else:
            row["split"] = "test"
    return rows


if __name__ == "__main__":
    main()
