# src/slam/map_mutation.py
from __future__ import annotations

from numbers import Integral
from typing import Any


MAP_MUTATION_COUNT_FIELDS = (
    "added_observations",
    "skipped_duplicate_observations",
    "added_landmarks",
    "skipped_landmark_candidates",
    "skipped_mapped_keyframe_features",
    "removed_landmarks",
    "updated_active_lookup_entries",
    "feature_assignment_conflicts",
    "missing_landmarks",
)

MAP_MUTATION_CHANGE_FIELDS = (
    "added_observations",
    "added_landmarks",
    "removed_landmarks",
    "updated_active_lookup_entries",
)


# Create a fresh map mutation report
def new_map_mutation_report(context="map_update") -> dict[str, Any]:
    report: dict[str, Any] = {"context": str(context)}
    for field in MAP_MUTATION_COUNT_FIELDS:
        report[field] = 0
    report["warnings"] = []
    return report


# Check a map mutation report is usable
def _check_map_mutation_report(report, *, name: str = "report") -> dict[str, Any]:
    if not isinstance(report, dict):
        raise ValueError(f"{name} must be a dict")

    if "context" not in report:
        raise ValueError(f"{name} is missing required key 'context'")

    for field in MAP_MUTATION_COUNT_FIELDS:
        if field not in report:
            raise ValueError(f"{name} is missing required key '{field}'")
        value = report[field]
        if isinstance(value, bool) or not isinstance(value, Integral):
            raise ValueError(f"{name}['{field}'] must be an integer")
        if int(value) < 0:
            raise ValueError(f"{name}['{field}'] must be >= 0; got {int(value)}")

    if "warnings" not in report:
        raise ValueError(f"{name} is missing required key 'warnings'")
    if not isinstance(report["warnings"], list):
        raise ValueError(f"{name}['warnings'] must be a list")

    return report


# Merge map mutation reports without mutating the inputs
def merge_map_mutation_reports(*reports, context="map_update") -> dict[str, Any]:
    merged = new_map_mutation_report(context=str(context))
    for i, report in enumerate(reports):
        report = _check_map_mutation_report(report, name=f"reports[{i}]")
        for field in MAP_MUTATION_COUNT_FIELDS:
            merged[field] += int(report[field])
        merged["warnings"].extend(str(message) for message in report["warnings"])

    return merged


# Add one warning to a map mutation report
def add_report_warning(report, message) -> dict[str, Any]:
    report = _check_map_mutation_report(report)
    report["warnings"].append(str(message))
    return report


# Count the actual map changes in a report
def count_report_changes(report) -> int:
    report = _check_map_mutation_report(report)
    return int(sum(int(report[field]) for field in MAP_MUTATION_CHANGE_FIELDS))
