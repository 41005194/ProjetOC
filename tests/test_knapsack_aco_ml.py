from __future__ import annotations

from metaheuristique import MLTunedKnapsackACOSolver
from metaheuristique.ml import FEATURE_ORDER, ParameterTuningModel, extract_state_features
from metaheuristique.types import ACOParams
from metaheuristique.types import Solution


def test_ml_solver_fit_predicts_bounded_parameters(training_records) -> None:
    solver = MLTunedKnapsackACOSolver(tuning_interval=1)
    solver.fit(training_records)

    feature_map = {
        name: value
        for name, value in zip(FEATURE_ORDER, training_records[0].to_feature_vector(), strict=True)
    }
    prediction = solver.tuning_model.predict(feature_map)

    assert 0.1 <= prediction.alpha <= 6.0
    assert 0.1 <= prediction.beta <= 8.0
    assert 0.05 <= prediction.evaporation <= 0.95


def test_ml_solver_adjusts_parameters_and_keeps_solution_feasible(
    larger_instance,
    baseline_params,
    training_records,
) -> None:
    solver = MLTunedKnapsackACOSolver(tuning_interval=1).fit(training_records)

    result = solver.solve(
        larger_instance,
        seed=29,
        iterations=12,
        colony_size=8,
        params=baseline_params,
    )

    assert result.best_solution.is_feasible
    assert result.best_solution.total_weight <= larger_instance.capacity
    assert len(result.parameter_history) == 12
    assert any(parameters != baseline_params for parameters in result.parameter_history[1:])


def test_extract_state_features_reports_diversity_and_progress(micro_instance) -> None:
    solutions = [
        Solution(selected_indices=(1, 2), total_value=18, total_weight=5, is_feasible=True),
        Solution(selected_indices=(0,), total_value=12, total_weight=4, is_feasible=True),
        Solution(selected_indices=(2,), total_value=8, total_weight=2, is_feasible=True),
    ]

    features = extract_state_features(
        micro_instance,
        solutions,
        best_value=18,
        stagnation_iterations=2,
    )

    assert set(features) == set(FEATURE_ORDER)
    assert features["diversity_ratio"] == 1.0
    assert 0 < features["normalized_best_value"] <= 1.0


def test_parameter_tuning_model_clamps_out_of_range_predictions(training_records) -> None:
    model = ParameterTuningModel()
    model.fit(training_records)

    class StubRegressor:
        def predict(self, _: list[list[float]]) -> list[list[float]]:
            return [[-3.0, 20.0, 1.7]]

    model._model = StubRegressor()
    prediction = model.predict({name: 0.5 for name in FEATURE_ORDER})

    assert prediction == ACOParams(alpha=0.1, beta=8.0, evaporation=0.95)
