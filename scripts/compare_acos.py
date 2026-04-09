from __future__ import annotations

import argparse
import csv
from pathlib import Path

from metaheuristique import (
    ACOParams,
    KnapsackACOSolver,
    MLTunedKnapsackACOSolver,
    build_demo_instance,
    build_demo_training_records,
    build_report_instances,
    compare_result_batches,
    default_report_seeds,
    run_solver_batch,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EXPORT_DIR = PROJECT_ROOT / "results"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run baseline ACO and ML-tuned ACO on one or several knapsack instances."
    )
    parser.add_argument("--iterations", type=int, default=30, help="Number of iterations per run.")
    parser.add_argument("--colony-size", type=int, default=12, help="Number of ants per iteration.")
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=list(default_report_seeds()),
        help="Seeds used for both solvers.",
    )
    parser.add_argument(
        "--instance-set",
        choices=("demo", "report"),
        default="report",
        help="Use a single demo instance or a small report-oriented benchmark suite.",
    )
    parser.add_argument(
        "--export-dir",
        type=Path,
        default=DEFAULT_EXPORT_DIR,
        help="Directory where CSV and Markdown summaries will be written.",
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


def format_paired_runs(paired_runs: tuple[object, ...]) -> str:
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


def select_instances(instance_set: str) -> dict[str, object]:
    if instance_set == "demo":
        return {"demo": build_demo_instance()}
    return build_report_instances()


def export_summary_rows(export_dir: Path, rows: list[dict[str, object]]) -> Path:
    export_dir.mkdir(parents=True, exist_ok=True)
    csv_path = export_dir / "comparison_summary.csv"
    fieldnames = [
        "instance",
        "capacity",
        "item_count",
        "iterations",
        "colony_size",
        "seed_count",
        "baseline_best",
        "baseline_mean",
        "baseline_stdev",
        "baseline_invalid_solutions",
        "baseline_total_runtime_seconds",
        "ml_best",
        "ml_mean",
        "ml_stdev",
        "ml_invalid_solutions",
        "ml_total_runtime_seconds",
        "mean_best_value_delta",
        "mean_runtime_delta_seconds",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return csv_path


def export_run_rows(export_dir: Path, rows: list[dict[str, object]]) -> Path:
    export_dir.mkdir(parents=True, exist_ok=True)
    csv_path = export_dir / "comparison_runs.csv"
    fieldnames = [
        "instance",
        "seed",
        "baseline_best",
        "ml_best",
        "best_value_delta",
        "baseline_runtime_seconds",
        "ml_runtime_seconds",
        "runtime_delta_seconds",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return csv_path


def build_markdown_report(summary_rows: list[dict[str, object]]) -> str:
    lines = [
        "# Benchmark Summary",
        "",
        "| Instance | Capacity | Items | Baseline Mean | ML Mean | Delta Mean | Baseline Time | ML Time |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary_rows:
        lines.append(
            "| {instance} | {capacity} | {item_count} | {baseline_mean:.2f} | {ml_mean:.2f} | {mean_best_value_delta:+.2f} | {baseline_total_runtime_seconds:.6f} | {ml_total_runtime_seconds:.6f} |".format(
                **row
            )
        )
    return "\n".join(lines) + "\n"


def export_markdown_report(export_dir: Path, summary_rows: list[dict[str, object]]) -> Path:
    export_dir.mkdir(parents=True, exist_ok=True)
    markdown_path = export_dir / "comparison_summary.md"
    markdown_path.write_text(build_markdown_report(summary_rows), encoding="utf-8")
    return markdown_path


def main() -> None:
    args = parse_args()
    params = ACOParams(alpha=args.alpha, beta=args.beta, evaporation=args.evaporation)
    training_records = build_demo_training_records()
    instances = select_instances(args.instance_set)
    summary_rows: list[dict[str, object]] = []
    run_rows: list[dict[str, object]] = []

    print("Knapsack ACO comparison")
    print(
        f"instances={len(instances)} iterations={args.iterations} "
        f"colony_size={args.colony_size} seeds={','.join(str(seed) for seed in args.seeds)}"
    )
    print()

    for instance_name, instance in instances.items():
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
        mean_best_delta = sum(pair.best_value_delta for pair in summary.paired_runs) / max(len(summary.paired_runs), 1)
        mean_runtime_delta = sum(pair.runtime_delta_seconds for pair in summary.paired_runs) / max(
            len(summary.paired_runs), 1
        )
        summary_rows.append(
            {
                "instance": instance_name,
                "capacity": instance.capacity,
                "item_count": instance.item_count,
                "iterations": summary.iterations,
                "colony_size": summary.colony_size,
                "seed_count": len(summary.seeds),
                "baseline_best": summary.baseline_metrics.best_value,
                "baseline_mean": summary.baseline_metrics.mean_value,
                "baseline_stdev": summary.baseline_metrics.stdev_value,
                "baseline_invalid_solutions": summary.baseline_metrics.invalid_solutions,
                "baseline_total_runtime_seconds": summary.baseline_metrics.total_runtime_seconds,
                "ml_best": summary.ml_metrics.best_value,
                "ml_mean": summary.ml_metrics.mean_value,
                "ml_stdev": summary.ml_metrics.stdev_value,
                "ml_invalid_solutions": summary.ml_metrics.invalid_solutions,
                "ml_total_runtime_seconds": summary.ml_metrics.total_runtime_seconds,
                "mean_best_value_delta": mean_best_delta,
                "mean_runtime_delta_seconds": mean_runtime_delta,
            }
        )
        for pair in summary.paired_runs:
            run_rows.append(
                {
                    "instance": instance_name,
                    "seed": pair.seed,
                    "baseline_best": pair.baseline_result.best_value,
                    "ml_best": pair.ml_result.best_value,
                    "best_value_delta": pair.best_value_delta,
                    "baseline_runtime_seconds": pair.baseline_result.runtime_seconds,
                    "ml_runtime_seconds": pair.ml_result.runtime_seconds,
                    "runtime_delta_seconds": pair.runtime_delta_seconds,
                }
            )

        print(
            f"[{instance_name}] capacity={instance.capacity} items={instance.item_count} "
            f"best_baseline={summary.baseline_metrics.best_value:.2f} "
            f"best_ml={summary.ml_metrics.best_value:.2f}"
        )
        print(format_metrics_block("baseline", summary.baseline_metrics))
        print()
        print(format_metrics_block("ml", summary.ml_metrics))
        print()
        print(format_paired_runs(summary.paired_runs))
        print()

    summary_csv = export_summary_rows(args.export_dir, summary_rows)
    runs_csv = export_run_rows(args.export_dir, run_rows)
    markdown_summary = export_markdown_report(args.export_dir, summary_rows)
    print(f"Exported summary CSV to {summary_csv}")
    print(f"Exported run CSV to {runs_csv}")
    print(f"Exported Markdown summary to {markdown_summary}")


if __name__ == "__main__":
    main()
