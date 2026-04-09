from __future__ import annotations

import argparse
from typing import Iterable

from metaheuristique import (
    ACOParams,
    KnapsackACOSolver,
    KnapsackInstance,
    KnapsackItem,
    MLTunedKnapsackACOSolver,
    TrainingRecord,
    compare_result_batches,
    run_solver_batch,
)


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
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run baseline ACO and ML-tuned ACO on a demo knapsack instance."
    )
    parser.add_argument("--iterations", type=int, default=20, help="Number of iterations per run.")
    parser.add_argument("--colony-size", type=int, default=10, help="Number of ants per iteration.")
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[7, 17, 23, 31],
        help="Seeds used for both solvers.",
    )
    parser.add_argument(
        "--tuning-interval",
        type=int,
        default=2,
        help="How often the ML solver retunes alpha, beta and evaporation.",
    )
    parser.add_argument("--alpha", type=float, default=1.1, help="Baseline alpha parameter.")
    parser.add_argument("--beta", type=float, default=2.7, help="Baseline beta parameter.")
    parser.add_argument("--evaporation", type=float, default=0.3, help="Baseline evaporation parameter.")
    return parser.parse_args()


def format_metrics_block(name: str, metrics: object) -> str:
    return "\n".join(
        [
            f"{name}:",
            f"  best_value={metrics.best_value:.2f}",
            f"  mean_value={metrics.mean_value:.2f}",
            f"  stdev_value={metrics.stdev_value:.2f}",
            f"  invalid_solutions={metrics.invalid_solutions}",
            f"  total_runtime_seconds={metrics.total_runtime_seconds:.6f}",
        ]
    )


def format_paired_runs(paired_runs: Iterable[object]) -> str:
    lines = [
        "seed | baseline_best | ml_best | delta_best | baseline_time | ml_time | delta_time",
        "-----|---------------|---------|------------|---------------|---------|-----------",
    ]
    for pair in paired_runs:
        lines.append(
            " | ".join(
                [
                    f"{pair.seed}",
                    f"{pair.baseline_result.best_value:.2f}",
                    f"{pair.ml_result.best_value:.2f}",
                    f"{pair.best_value_delta:+.2f}",
                    f"{pair.baseline_result.runtime_seconds:.6f}",
                    f"{pair.ml_result.runtime_seconds:.6f}",
                    f"{pair.runtime_delta_seconds:+.6f}",
                ]
            )
        )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    params = ACOParams(alpha=args.alpha, beta=args.beta, evaporation=args.evaporation)
    instance = build_demo_instance()
    training_records = build_demo_training_records()

    baseline_results = run_solver_batch(
        KnapsackACOSolver(),
        instance,
        seeds=args.seeds,
        iterations=args.iterations,
        colony_size=args.colony_size,
        params=params,
    )
    ml_results = run_solver_batch(
        MLTunedKnapsackACOSolver(tuning_interval=args.tuning_interval).fit(training_records),
        instance,
        seeds=args.seeds,
        iterations=args.iterations,
        colony_size=args.colony_size,
        params=params,
    )
    summary = compare_result_batches(baseline_results, ml_results)

    print("Knapsack ACO comparison")
    print(f"capacity={instance.capacity} items={instance.item_count}")
    print(
        f"iterations={summary.iterations} colony_size={summary.colony_size} "
        f"seeds={','.join(str(seed) for seed in summary.seeds)}"
    )
    print()
    print(format_metrics_block("baseline", summary.baseline_metrics))
    print()
    print(format_metrics_block("ml", summary.ml_metrics))
    print()
    print(format_paired_runs(summary.paired_runs))


if __name__ == "__main__":
    main()
