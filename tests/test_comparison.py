from __future__ import annotations

from metaheuristique import (
    KnapsackACOSolver,
    MLTunedKnapsackACOSolver,
    compare_result_batches,
    compare_solvers,
    run_solver_batch,
)


def test_run_solver_batch_returns_raw_results_for_comparison(
    larger_instance,
    baseline_params,
    benchmark_seeds,
    training_records,
) -> None:
    baseline_solver = KnapsackACOSolver()
    ml_solver = MLTunedKnapsackACOSolver(tuning_interval=2).fit(training_records)

    baseline_results = run_solver_batch(
        baseline_solver,
        larger_instance,
        seeds=benchmark_seeds,
        iterations=20,
        colony_size=10,
        params=baseline_params,
    )
    ml_results = run_solver_batch(
        ml_solver,
        larger_instance,
        seeds=benchmark_seeds,
        iterations=20,
        colony_size=10,
        params=baseline_params,
    )
    summary = compare_result_batches(baseline_results, ml_results)

    assert summary.seeds == benchmark_seeds
    assert summary.iterations == 20
    assert summary.colony_size == 10
    assert len(summary.baseline_results) == len(benchmark_seeds)
    assert len(summary.ml_results) == len(benchmark_seeds)
    assert len(summary.paired_runs) == len(benchmark_seeds)
    assert [result.seed for result in summary.baseline_results] == list(benchmark_seeds)
    assert [result.seed for result in summary.ml_results] == list(benchmark_seeds)
    assert all(result.best_solution.is_feasible for result in summary.baseline_results)
    assert all(result.best_solution.is_feasible for result in summary.ml_results)
    assert summary.baseline_metrics.invalid_solutions == 0
    assert summary.ml_metrics.invalid_solutions == 0
    assert all(pair.seed in benchmark_seeds for pair in summary.paired_runs)
    assert all(
        pair.best_value_delta == pair.ml_result.best_value - pair.baseline_result.best_value
        for pair in summary.paired_runs
    )


def test_compare_solvers_keeps_convenience_api(
    larger_instance,
    baseline_params,
    benchmark_seeds,
    training_records,
) -> None:
    ml_solver = MLTunedKnapsackACOSolver(tuning_interval=2).fit(training_records)

    summary = compare_solvers(
        larger_instance,
        seeds=benchmark_seeds,
        iterations=20,
        colony_size=10,
        baseline_params=baseline_params,
        ml_solver=ml_solver,
    )

    assert summary.baseline_metrics.best_value >= 0
    assert summary.ml_metrics.best_value >= 0
    assert summary.baseline_metrics.total_runtime_seconds >= 0
    assert summary.ml_metrics.total_runtime_seconds >= 0
