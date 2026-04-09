"""Public package exports for the knapsack ACO project."""

from .benchmarks import (
    build_demo_instance,
    build_demo_training_records,
    build_report_instances,
    default_report_seeds,
)
from .comparison import compare_result_batches, compare_solvers, run_solver_batch
from .knapsack_aco import KnapsackACOSolver
from .knapsack_aco_ml import MLTunedKnapsackACOSolver
from .ml import TrainingRecord, extract_state_features
from .types import (
    ACOParams,
    ComparisonSummary,
    KnapsackInstance,
    KnapsackItem,
    PairedRunComparison,
    RunResult,
    SolverMetrics,
    Solution,
)

__all__ = [
    "ACOParams",
    "ComparisonSummary",
    "build_demo_instance",
    "build_demo_training_records",
    "build_report_instances",
    "compare_result_batches",
    "default_report_seeds",
    "KnapsackACOSolver",
    "KnapsackInstance",
    "KnapsackItem",
    "MLTunedKnapsackACOSolver",
    "PairedRunComparison",
    "RunResult",
    "run_solver_batch",
    "SolverMetrics",
    "Solution",
    "TrainingRecord",
    "compare_solvers",
    "extract_state_features",
]
