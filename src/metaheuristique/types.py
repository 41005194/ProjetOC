from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class KnapsackItem:
    value: float
    weight: float
    name: str | None = None

    def __post_init__(self) -> None:
        if self.weight < 0:
            raise ValueError("Item weight must be non-negative.")
        if self.value < 0:
            raise ValueError("Item value must be non-negative.")


@dataclass(frozen=True)
class KnapsackInstance:
    capacity: float
    items: tuple[KnapsackItem, ...]

    def __post_init__(self) -> None:
        if self.capacity < 0:
            raise ValueError("Capacity must be non-negative.")

    @property
    def item_count(self) -> int:
        return len(self.items)

    @property
    def total_available_value(self) -> float:
        return sum(item.value for item in self.items)


@dataclass(frozen=True)
class Solution:
    selected_indices: tuple[int, ...]
    total_value: float
    total_weight: float
    is_feasible: bool

    @classmethod
    def empty(cls) -> "Solution":
        return cls(selected_indices=(), total_value=0.0, total_weight=0.0, is_feasible=True)


@dataclass(frozen=True)
class ACOParams:
    alpha: float = 1.2
    beta: float = 2.5
    evaporation: float = 0.25
    deposit_weight: float = 1.0
    pheromone_min: float = 0.05
    pheromone_max: float = 10.0

    def __post_init__(self) -> None:
        if self.alpha <= 0:
            raise ValueError("alpha must be strictly positive.")
        if self.beta <= 0:
            raise ValueError("beta must be strictly positive.")
        if not 0 < self.evaporation < 1:
            raise ValueError("evaporation must be between 0 and 1.")
        if self.deposit_weight <= 0:
            raise ValueError("deposit_weight must be strictly positive.")
        if self.pheromone_min <= 0:
            raise ValueError("pheromone_min must be strictly positive.")
        if self.pheromone_max < self.pheromone_min:
            raise ValueError("pheromone_max must be >= pheromone_min.")

    def bounded(self) -> "ACOParams":
        return ACOParams(
            alpha=min(max(self.alpha, 0.1), 6.0),
            beta=min(max(self.beta, 0.1), 8.0),
            evaporation=min(max(self.evaporation, 0.05), 0.95),
            deposit_weight=min(max(self.deposit_weight, 0.1), 10.0),
            pheromone_min=min(max(self.pheromone_min, 0.001), 5.0),
            pheromone_max=min(max(self.pheromone_max, self.pheromone_min), 50.0),
        )


@dataclass(frozen=True)
class RunResult:
    best_solution: Solution
    best_value: float
    best_value_history: tuple[float, ...]
    parameter_history: tuple[ACOParams, ...]
    runtime_seconds: float
    invalid_solutions: int
    seed: int
    iterations: int
    colony_size: int
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SolverMetrics:
    best_value: float
    mean_value: float
    stdev_value: float
    invalid_solutions: int
    total_runtime_seconds: float


@dataclass(frozen=True)
class PairedRunComparison:
    seed: int
    baseline_result: RunResult
    ml_result: RunResult
    best_value_delta: float
    runtime_delta_seconds: float


@dataclass(frozen=True)
class ComparisonSummary:
    baseline_metrics: SolverMetrics
    ml_metrics: SolverMetrics
    baseline_results: tuple[RunResult, ...]
    ml_results: tuple[RunResult, ...]
    paired_runs: tuple[PairedRunComparison, ...]
    seeds: tuple[int, ...]
    iterations: int
    colony_size: int
