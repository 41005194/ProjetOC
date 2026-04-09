from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from statistics import fmean
from typing import Mapping, Sequence

from .types import ACOParams, KnapsackInstance, Solution

try:
    from sklearn.linear_model import LinearRegression
    from sklearn.multioutput import MultiOutputRegressor
except ImportError:  # pragma: no cover
    LinearRegression = None
    MultiOutputRegressor = None


FEATURE_ORDER = (
    "remaining_capacity_ratio",
    "diversity_ratio",
    "normalized_best_value",
    "stagnation_ratio",
    "mean_value_ratio",
)


@dataclass(frozen=True)
class TrainingRecord:
    remaining_capacity_ratio: float
    diversity_ratio: float
    normalized_best_value: float
    stagnation_ratio: float
    mean_value_ratio: float
    target_params: ACOParams

    def to_feature_vector(self) -> list[float]:
        return [float(getattr(self, feature_name)) for feature_name in FEATURE_ORDER]


def extract_state_features(
    instance: KnapsackInstance,
    solutions: Sequence[Solution],
    *,
    best_value: float,
    stagnation_iterations: int,
) -> dict[str, float]:
    if not solutions:
        return {
            "remaining_capacity_ratio": 1.0 if instance.capacity > 0 else 0.0,
            "diversity_ratio": 0.0,
            "normalized_best_value": 0.0,
            "stagnation_ratio": 0.0,
            "mean_value_ratio": 0.0,
        }

    remaining_ratios = []
    signatures = set()
    values = []
    for solution in solutions:
        signatures.add(solution.selected_indices)
        values.append(solution.total_value)
        if instance.capacity > 0:
            remaining_ratios.append(max(instance.capacity - solution.total_weight, 0.0) / instance.capacity)
        else:
            remaining_ratios.append(0.0)

    denominator = max(instance.total_available_value, 1.0)
    return {
        "remaining_capacity_ratio": fmean(remaining_ratios),
        "diversity_ratio": len(signatures) / max(len(solutions), 1),
        "normalized_best_value": best_value / denominator,
        "stagnation_ratio": stagnation_iterations / max(len(solutions), 1),
        "mean_value_ratio": fmean(values) / denominator,
    }


class ParameterTuningModel:
    """Predict ACO parameters from search-state features."""

    def __init__(self) -> None:
        self._model = None
        self._records: list[TrainingRecord] = []

    @property
    def is_fitted(self) -> bool:
        return self._model is not None or bool(self._records)

    def fit(self, training_records: Sequence[TrainingRecord]) -> None:
        self._records = list(training_records)
        if not self._records:
            raise ValueError("At least one training record is required.")

        if LinearRegression is None or MultiOutputRegressor is None:
            self._model = "fallback"
            return

        feature_matrix = [record.to_feature_vector() for record in self._records]
        targets = [
            [
                record.target_params.alpha,
                record.target_params.beta,
                record.target_params.evaporation,
            ]
            for record in self._records
        ]
        model = MultiOutputRegressor(LinearRegression())
        model.fit(feature_matrix, targets)
        self._model = model

    def predict(self, features: Mapping[str, float]) -> ACOParams:
        if not self.is_fitted:
            raise RuntimeError("The tuning model must be fitted before prediction.")

        vector = [float(features[name]) for name in FEATURE_ORDER]
        if self._model == "fallback":
            return self._predict_with_fallback(vector)

        prediction = self._model.predict([vector])[0]
        return self._build_bounded_params(
            alpha=float(prediction[0]),
            beta=float(prediction[1]),
            evaporation=float(prediction[2]),
        )

    def _predict_with_fallback(self, vector: Sequence[float]) -> ACOParams:
        distances: list[float] = []
        for record in self._records:
            distance = sqrt(
                sum(
                    (feature_value - target_value) ** 2
                    for feature_value, target_value in zip(vector, record.to_feature_vector(), strict=True)
                )
            )
            if distance == 0:
                return record.target_params.bounded()
            distances.append(distance)

        weights = [1.0 / distance for distance in distances]
        total_weight = sum(weights)
        alpha = 0.0
        beta = 0.0
        evaporation = 0.0
        for weight, record in zip(weights, self._records, strict=True):
            alpha += weight * record.target_params.alpha
            beta += weight * record.target_params.beta
            evaporation += weight * record.target_params.evaporation
        return self._build_bounded_params(
            alpha=alpha / total_weight,
            beta=beta / total_weight,
            evaporation=evaporation / total_weight,
        )

    def _build_bounded_params(self, *, alpha: float, beta: float, evaporation: float) -> ACOParams:
        return ACOParams(
            alpha=min(max(alpha, 0.1), 6.0),
            beta=min(max(beta, 0.1), 8.0),
            evaporation=min(max(evaporation, 0.05), 0.95),
        )


def fit_training_records(training_records: Sequence[TrainingRecord]) -> ParameterTuningModel:
    model = ParameterTuningModel()
    model.fit(training_records)
    return model
