from __future__ import annotations

import random
import time
from typing import Sequence

from .knapsack_aco import KnapsackACOSolver
from .ml import ParameterTuningModel, TrainingRecord, extract_state_features
from .types import ACOParams, KnapsackInstance, RunResult, Solution


class MLTunedKnapsackACOSolver(KnapsackACOSolver):
    """ACO solver with dynamic parameter tuning driven by a fitted ML model."""

    def __init__(
        self,
        *,
        tuning_model: ParameterTuningModel | None = None,
        tuning_interval: int = 3,
    ) -> None:
        if tuning_interval <= 0:
            raise ValueError("tuning_interval must be strictly positive.")
        self.tuning_model = tuning_model or ParameterTuningModel()
        self.tuning_interval = tuning_interval

    def fit(self, training_records: Sequence[TrainingRecord]) -> "MLTunedKnapsackACOSolver":
        self.tuning_model.fit(training_records)
        return self

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
        stagnation_iterations = 0
        last_colony: list[Solution] = []

        start = time.perf_counter()

        for iteration in range(iterations):
            if self.tuning_model.is_fitted and iteration > 0 and iteration % self.tuning_interval == 0:
                features = extract_state_features(
                    instance,
                    last_colony,
                    best_value=best_solution.total_value,
                    stagnation_iterations=stagnation_iterations,
                )
                active_params = self.tuning_model.predict(features)

            colony = [
                self._construct_solution(instance, pheromones, active_params, rng)
                for _ in range(colony_size)
            ]
            invalid_solutions += sum(not solution.is_feasible for solution in colony)
            iteration_best = max(colony, key=lambda solution: solution.total_value, default=Solution.empty())
            if iteration_best.total_value > best_solution.total_value:
                best_solution = iteration_best
                stagnation_iterations = 0
            else:
                stagnation_iterations += 1

            pheromones = self._update_pheromones(
                pheromones=pheromones,
                instance=instance,
                solutions=colony,
                params=active_params,
                best_solution=iteration_best,
            )

            last_colony = colony
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
            metadata={"ml_tuning_enabled": self.tuning_model.is_fitted},
        )
