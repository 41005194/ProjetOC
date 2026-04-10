from __future__ import annotations

from metaheuristique import (
    build_demo_instance,
    build_demo_training_records,
    build_report_instances,
    default_report_seeds,
)


def test_report_instances_provide_multiple_benchmark_shapes() -> None:
    instances = build_report_instances()

    assert len(instances) >= 4
    assert "balanced_small" in instances
    assert "tight_capacity" in instances
    assert all(instance.capacity > 0 for instance in instances.values())
    assert all(instance.item_count >= 8 for instance in instances.values())


def test_demo_helpers_keep_existing_demo_available() -> None:
    instance = build_demo_instance()
    records = build_demo_training_records()

    assert instance.capacity == 20
    assert instance.item_count == 8
    assert len(records) >= 3
    assert all(record.target_params.alpha > 0 for record in records)


def test_default_report_seeds_are_large_enough_for_comparison() -> None:
    seeds = default_report_seeds()

    assert len(seeds) == 10
    assert len(set(seeds)) == len(seeds)
