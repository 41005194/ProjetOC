from __future__ import annotations

import random
import time
from typing import Sequence

from .types import ACOParams, KnapsackInstance, KnapsackItem, RunResult, Solution


class KnapsackACOSolver:
    """Baseline Ant Colony Optimization solver for the knapsack problem."""

    def solve(
        self,
        instance: KnapsackInstance,
        *,
        seed: int,
        iterations: int,
        colony_size: int,
        params: ACOParams | None = None,
    ) -> RunResult:
        if iterations <= 0:
            raise ValueError("iterations must be strictly positive.")
        if colony_size <= 0:
            raise ValueError("colony_size must be strictly positive.")

        active_params = (params or ACOParams()).bounded()
        rng = random.Random(seed)
        pheromones = [1.0 for _ in range(instance.item_count)]
        best_solution = Solution.empty()
        best_value_history: list[float] = []
        parameter_history: list[ACOParams] = []
        invalid_solutions = 0

        start = time.perf_counter()

        for _ in range(iterations):
            colony = [
                self._construct_solution(instance, pheromones, active_params, rng)
                for _ in range(colony_size)
            ]
            invalid_solutions += sum(not solution.is_feasible for solution in colony)
            iteration_best = max(colony, key=lambda solution: solution.total_value, default=Solution.empty())
            if iteration_best.total_value > best_solution.total_value:
                best_solution = iteration_best

            pheromones = self._update_pheromones(
                pheromones=pheromones,
                instance=instance,
                solutions=colony,
                params=active_params,
                best_solution=iteration_best,
            )
            best_value_history.append(best_solution.total_value)
            parameter_history.append(active_params)

        runtime_seconds = time.perf_counter() - start
        return RunResult(
            best_solution=best_solution,
            best_value=best_solution.total_value,
            best_value_history=tuple(best_value_history),
            parameter_history=tuple(parameter_history),
            runtime_seconds=runtime_seconds,
            invalid_solutions=invalid_solutions,
            seed=seed,
            iterations=iterations,
            colony_size=colony_size,
            metadata={"pheromone_span": (min(pheromones, default=0.0), max(pheromones, default=0.0))},
        )

    def _construct_solution(
        self,
        instance: KnapsackInstance,
        pheromones: Sequence[float],
        params: ACOParams,
        rng: random.Random,
    ) -> Solution:
        remaining_capacity = instance.capacity
        selected_indices: list[int] = []
        chosen = set()

        while True:
            feasible_indices = [
                index
                for index, item in enumerate(instance.items)
                if index not in chosen and item.weight <= remaining_capacity
            ]
            if not feasible_indices:
                break

            weights = [
                self._selection_weight(item=instance.items[index], pheromone=pheromones[index], params=params)
                for index in feasible_indices
            ]
            selected_index = self._weighted_choice(feasible_indices, weights, rng)
            selected_item = instance.items[selected_index]

            chosen.add(selected_index)
            selected_indices.append(selected_index)
            remaining_capacity -= selected_item.weight

        selected_indices.sort()
        total_weight = sum(instance.items[index].weight for index in selected_indices)
        total_value = sum(instance.items[index].value for index in selected_indices)
        return Solution(
            selected_indices=tuple(selected_indices),
            total_value=total_value,
            total_weight=total_weight,
            is_feasible=total_weight <= instance.capacity,
        )

    def _selection_weight(self, *, item: KnapsackItem, pheromone: float, params: ACOParams) -> float:
        heuristic = self._heuristic_value(item)
        return max(pheromone, params.pheromone_min) ** params.alpha * heuristic**params.beta

    def _heuristic_value(self, item: KnapsackItem) -> float:
        if item.weight == 0:
            return max(item.value, 1.0) + 1.0
        return (item.value / item.weight) + 1e-9

    def _weighted_choice(
        self,
        candidates: Sequence[int],
        weights: Sequence[float],
        rng: random.Random,
    ) -> int:
        total_weight = sum(weights)
        if total_weight <= 0:
            return candidates[rng.randrange(len(candidates))]

        threshold = rng.random() * total_weight
        cumulative = 0.0
        for candidate, weight in zip(candidates, weights, strict=True):
            cumulative += weight
            if cumulative >= threshold:
                return candidate
        return candidates[-1]

    def _update_pheromones(
        self,
        *,
        pheromones: Sequence[float],
        instance: KnapsackInstance,
        solutions: Sequence[Solution],
        params: ACOParams,
        best_solution: Solution,
    ) -> list[float]:
        evaporated = [
            min(max(value * (1.0 - params.evaporation), params.pheromone_min), params.pheromone_max)
            for value in pheromones
        ]
        deposit_scale = params.deposit_weight / max(instance.capacity, 1.0)

        for solution in solutions:
            if not solution.is_feasible or solution.total_value <= 0:
                continue
            contribution = deposit_scale * solution.total_value
            for index in solution.selected_indices:
                evaporated[index] += contribution

        if best_solution.total_value > 0:
            elite_contribution = deposit_scale * best_solution.total_value * 0.5
            for index in best_solution.selected_indices:
                evaporated[index] += elite_contribution

        return [
            min(max(value, params.pheromone_min), params.pheromone_max)
            for value in evaporated
        ]
