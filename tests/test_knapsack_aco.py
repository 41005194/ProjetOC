from __future__ import annotations

from metaheuristique import KnapsackACOSolver, KnapsackInstance, KnapsackItem


def test_baseline_solver_finds_known_micro_optimum(micro_instance, baseline_params) -> None:
    solver = KnapsackACOSolver()

    result = solver.solve(
        micro_instance,
        seed=11,
        iterations=40,
        colony_size=10,
        params=baseline_params,
    )

    assert result.best_solution.is_feasible
    assert result.best_solution.total_weight <= micro_instance.capacity
    assert result.best_value == 18
    assert set(result.best_solution.selected_indices) == {1, 2}


def test_baseline_solver_is_reproducible_for_same_seed(larger_instance, baseline_params) -> None:
    solver = KnapsackACOSolver()

    first = solver.solve(
        larger_instance,
        seed=19,
        iterations=25,
        colony_size=12,
        params=baseline_params,
    )
    second = solver.solve(
        larger_instance,
        seed=19,
        iterations=25,
        colony_size=12,
        params=baseline_params,
    )

    assert first.best_value == second.best_value
    assert first.best_solution == second.best_solution
    assert first.best_value_history == second.best_value_history
    assert first.parameter_history == second.parameter_history


def test_baseline_solver_handles_empty_or_zero_capacity_cases(baseline_params) -> None:
    solver = KnapsackACOSolver()
    zero_capacity = KnapsackInstance(
        capacity=0,
        items=(KnapsackItem(value=8, weight=2, name="heavy"),),
    )
    empty_items = KnapsackInstance(capacity=10, items=())

    zero_capacity_result = solver.solve(
        zero_capacity,
        seed=3,
        iterations=5,
        colony_size=4,
        params=baseline_params,
    )
    empty_items_result = solver.solve(
        empty_items,
        seed=3,
        iterations=5,
        colony_size=4,
        params=baseline_params,
    )

    assert zero_capacity_result.best_value == 0
    assert zero_capacity_result.best_solution.selected_indices == ()
    assert empty_items_result.best_value == 0
    assert empty_items_result.best_solution.selected_indices == ()
