from __future__ import annotations

from statistics import mean, stdev
from typing import Iterable, Protocol

from .knapsack_aco import KnapsackACOSolver
from .knapsack_aco_ml import MLTunedKnapsackACOSolver
from .types import (
    ACOParams,
    ComparisonSummary,
    KnapsackInstance,
    PairedRunComparison,
    RunResult,
    SolverMetrics,
)


class SupportsSolve(Protocol):
    def solve(
        self,
        instance: KnapsackInstance,
        *,
        seed: int,
        iterations: int,
        colony_size: int,
        params: ACOParams | None = None,
    ) -> RunResult: ...


def run_solver_batch(
    solver: SupportsSolve,
    instance: KnapsackInstance,
    *,
    seeds: Iterable[int],
    iterations: int,
    colony_size: int,
    params: ACOParams,
) -> tuple[RunResult, ...]:
    normalized_seeds = tuple(seeds)
    if not normalized_seeds:
        raise ValueError("At least one seed is required to run a batch.")

    return tuple(
        solver.solve(
            instance,
            seed=seed,
            iterations=iterations,
            colony_size=colony_size,
            params=params,
        )
        for seed in normalized_seeds
    )


def compare_result_batches(
    baseline_results: Iterable[RunResult],
    ml_results: Iterable[RunResult],
) -> ComparisonSummary:
    normalized_baseline = tuple(baseline_results)
    normalized_ml = tuple(ml_results)
    if not normalized_baseline or not normalized_ml:
        raise ValueError("Both baseline and ML result batches must be non-empty.")
    if len(normalized_baseline) != len(normalized_ml):
        raise ValueError("Baseline and ML result batches must have the same length.")

    first_baseline = normalized_baseline[0]
    seeds: list[int] = []
    paired_runs: list[PairedRunComparison] = []

    for baseline_result, ml_result in zip(normalized_baseline, normalized_ml, strict=True):
        if baseline_result.seed != ml_result.seed:
            raise ValueError("Result batches must be aligned on the same seeds in the same order.")
        if baseline_result.iterations != ml_result.iterations:
            raise ValueError("Result batches must use the same iteration budget.")
        if baseline_result.colony_size != ml_result.colony_size:
            raise ValueError("Result batches must use the same colony size.")

        seeds.append(baseline_result.seed)
        paired_runs.append(
            PairedRunComparison(
                seed=baseline_result.seed,
                baseline_result=baseline_result,
                ml_result=ml_result,
                best_value_delta=ml_result.best_value - baseline_result.best_value,
                runtime_delta_seconds=ml_result.runtime_seconds - baseline_result.runtime_seconds,
            )
        )

    return ComparisonSummary(
        baseline_metrics=_aggregate_results(normalized_baseline),
        ml_metrics=_aggregate_results(normalized_ml),
        baseline_results=normalized_baseline,
        ml_results=normalized_ml,
        paired_runs=tuple(paired_runs),
        seeds=tuple(seeds),
        iterations=first_baseline.iterations,
        colony_size=first_baseline.colony_size,
    )


def compare_solvers(
    instance: KnapsackInstance,
    *,
    seeds: Iterable[int],
    iterations: int,
    colony_size: int,
    baseline_params: ACOParams,
    ml_solver: MLTunedKnapsackACOSolver,
) -> ComparisonSummary:
    baseline_solver = KnapsackACOSolver()
    baseline_results = run_solver_batch(
        baseline_solver,
        instance,
        seeds=seeds,
        iterations=iterations,
        colony_size=colony_size,
        params=baseline_params,
    )
    ml_results = run_solver_batch(
        ml_solver,
        instance,
        seeds=seeds,
        iterations=iterations,
        colony_size=colony_size,
        params=baseline_params,
    )
    return compare_result_batches(baseline_results, ml_results)


def _aggregate_results(results: tuple[RunResult, ...]) -> SolverMetrics:
    values = [result.best_value for result in results]
    runtimes = [result.runtime_seconds for result in results]
    invalid_total = sum(result.invalid_solutions for result in results)
    return SolverMetrics(
        best_value=max(values, default=0.0),
        mean_value=mean(values) if values else 0.0,
        stdev_value=stdev(values) if len(values) > 1 else 0.0,
        invalid_solutions=invalid_total,
        total_runtime_seconds=sum(runtimes),
    )
