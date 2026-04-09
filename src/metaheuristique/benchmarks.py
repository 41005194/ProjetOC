from __future__ import annotations

from .ml import TrainingRecord
from .types import ACOParams, KnapsackInstance, KnapsackItem


def build_demo_instance() -> KnapsackInstance:
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


def build_report_instances() -> dict[str, KnapsackInstance]:
    return {
        "balanced_small": build_demo_instance(),
        "dense_medium": KnapsackInstance(
            capacity=35,
            items=(
                KnapsackItem(value=18, weight=4, name="A"),
                KnapsackItem(value=22, weight=6, name="B"),
                KnapsackItem(value=29, weight=7, name="C"),
                KnapsackItem(value=33, weight=9, name="D"),
                KnapsackItem(value=37, weight=10, name="E"),
                KnapsackItem(value=41, weight=12, name="F"),
                KnapsackItem(value=11, weight=3, name="G"),
                KnapsackItem(value=14, weight=5, name="H"),
                KnapsackItem(value=19, weight=6, name="I"),
                KnapsackItem(value=26, weight=8, name="J"),
            ),
        ),
        "tight_capacity": KnapsackInstance(
            capacity=18,
            items=(
                KnapsackItem(value=20, weight=11, name="A"),
                KnapsackItem(value=18, weight=10, name="B"),
                KnapsackItem(value=14, weight=7, name="C"),
                KnapsackItem(value=17, weight=9, name="D"),
                KnapsackItem(value=9, weight=4, name="E"),
                KnapsackItem(value=8, weight=3, name="F"),
                KnapsackItem(value=11, weight=5, name="G"),
                KnapsackItem(value=13, weight=6, name="H"),
                KnapsackItem(value=6, weight=2, name="I"),
            ),
        ),
        "wide_large": KnapsackInstance(
            capacity=60,
            items=(
                KnapsackItem(value=12, weight=3, name="A"),
                KnapsackItem(value=18, weight=5, name="B"),
                KnapsackItem(value=21, weight=6, name="C"),
                KnapsackItem(value=24, weight=7, name="D"),
                KnapsackItem(value=30, weight=9, name="E"),
                KnapsackItem(value=34, weight=10, name="F"),
                KnapsackItem(value=37, weight=11, name="G"),
                KnapsackItem(value=40, weight=13, name="H"),
                KnapsackItem(value=16, weight=4, name="I"),
                KnapsackItem(value=19, weight=5, name="J"),
                KnapsackItem(value=23, weight=8, name="K"),
                KnapsackItem(value=27, weight=9, name="L"),
                KnapsackItem(value=31, weight=10, name="M"),
                KnapsackItem(value=38, weight=12, name="N"),
            ),
        ),
    }


def build_demo_training_records() -> list[TrainingRecord]:
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
        TrainingRecord(
            remaining_capacity_ratio=0.22,
            diversity_ratio=0.42,
            normalized_best_value=0.82,
            stagnation_ratio=0.65,
            mean_value_ratio=0.66,
            target_params=ACOParams(alpha=1.0, beta=2.9, evaporation=0.38),
        ),
        TrainingRecord(
            remaining_capacity_ratio=0.48,
            diversity_ratio=0.72,
            normalized_best_value=0.60,
            stagnation_ratio=0.18,
            mean_value_ratio=0.50,
            target_params=ACOParams(alpha=1.7, beta=2.0, evaporation=0.20),
        ),
    ]


def default_report_seeds() -> tuple[int, ...]:
    return tuple(range(7, 17))
