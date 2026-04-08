from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from metaheuristique import ACOParams, KnapsackInstance, KnapsackItem, TrainingRecord


@pytest.fixture()
def micro_instance() -> KnapsackInstance:
    return KnapsackInstance(
        capacity=5,
        items=(
            KnapsackItem(value=12, weight=4, name="A"),
            KnapsackItem(value=10, weight=3, name="B"),
            KnapsackItem(value=8, weight=2, name="C"),
            KnapsackItem(value=1, weight=10, name="D"),
        ),
    )


@pytest.fixture()
def larger_instance() -> KnapsackInstance:
    return KnapsackInstance(
        capacity=20,
        items=(
            KnapsackItem(value=24, weight=12, name="A"),
            KnapsackItem(value=13, weight=7, name="B"),
            KnapsackItem(value=23, weight=11, name="C"),
            KnapsackItem(value=15, weight=8, name="D"),
            KnapsackItem(value=16, weight=9, name="E"),
            KnapsackItem(value=9, weight=4, name="F"),
            KnapsackItem(value=11, weight=5, name="G"),
            KnapsackItem(value=14, weight=6, name="H"),
        ),
    )


@pytest.fixture()
def baseline_params() -> ACOParams:
    return ACOParams(alpha=1.1, beta=2.7, evaporation=0.3)


@pytest.fixture()
def benchmark_seeds() -> tuple[int, ...]:
    return (7, 17, 23, 31)


@pytest.fixture()
def training_records() -> list[TrainingRecord]:
    return [
        TrainingRecord(
            remaining_capacity_ratio=0.10,
            diversity_ratio=0.25,
            normalized_best_value=0.75,
            stagnation_ratio=0.90,
            mean_value_ratio=0.62,
            target_params=ACOParams(alpha=0.9, beta=3.1, evaporation=0.45),
        ),
        TrainingRecord(
            remaining_capacity_ratio=0.35,
            diversity_ratio=0.50,
            normalized_best_value=0.88,
            stagnation_ratio=0.40,
            mean_value_ratio=0.70,
            target_params=ACOParams(alpha=1.3, beta=2.4, evaporation=0.25),
        ),
        TrainingRecord(
            remaining_capacity_ratio=0.55,
            diversity_ratio=0.80,
            normalized_best_value=0.55,
            stagnation_ratio=0.15,
            mean_value_ratio=0.45,
            target_params=ACOParams(alpha=1.9, beta=1.8, evaporation=0.18),
        ),
    ]
