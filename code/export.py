"""
export.py — End-to-end pipeline: parse -> aggregate -> merge labels -> CSV.

Produces predictor_table.csv with one row per trajectory:
  trajectory_id, environment, critical_failure_module, failure_type,
  critical_failure_step, [12 predictors], extraction_warnings

NOTE: as currently wired (aggregator.py imports the DIAGNOSTIC tokenizer),
this output is for Step 8 pipeline verification only. It is NOT the frozen
predictor table for Step 5 analysis. See tokenizer.py docstring.
"""

import csv
import json
from pathlib import Path

from parser import parse_all_trajectories
from aggregator import compute_predictors

PIPELINE_DIR = Path(__file__).parent
DATA_DIR = PIPELINE_DIR / "data"
OUTPUT_PATH = PIPELINE_DIR / "predictor_table.csv"

PREDICTOR_FIELDS = [
    "mean_memory_length", "memory_proportion", "memory_variability",
    "mean_reflection_length", "reflection_proportion", "reflection_variability",
    "mean_plan_length", "plan_proportion", "plan_variability",
]
# Confirmatory (Step 5A/5C): first 6 fields (Memory Expression + Reflection Expression)
# Exploratory-only (Step 5B): all 9 fields, including the last 3 (Planning Expression)


def load_labels() -> dict:
    """Load all three label files into one dict keyed by trajectory_id."""
    labels = {}
    for fname in ["alfworld_labels.json", "gaia_labels.json", "webshop_labels.json"]:
        path = DATA_DIR / fname
        with open(path, "r", encoding="utf-8") as f:
            entries = json.load(f)
        for entry in entries:
            labels[entry["trajectory_id"]] = entry
    return labels


def main():
    trajectories = parse_all_trajectories(DATA_DIR)
    labels = load_labels()

    rows = []
    unmatched = []
    for traj in trajectories:
        preds = compute_predictors(traj)
        label = labels.get(traj.trajectory_id)
        if label is None:
            unmatched.append(traj.trajectory_id)
            continue

        row = {
            "trajectory_id": traj.trajectory_id,
            "environment": traj.environment,
            "model": preds.model,
            "n_steps_total": preds.n_steps_total,
            "critical_failure_module": label.get("critical_failure_module"),
            "failure_type": label["step_annotations"][0][label.get("critical_failure_module")].get("failure_type")
                if label.get("step_annotations") else None,
            "critical_failure_step": label.get("critical_failure_step"),
        }
        for field_name in PREDICTOR_FIELDS:
            row[field_name] = getattr(preds, field_name)
        row["extraction_warnings"] = ";".join(preds.extraction_warnings)
        row["parse_errors"] = ";".join(traj.parse_errors)
        rows.append(row)

    fieldnames = (
        ["trajectory_id", "environment", "model", "n_steps_total",
         "critical_failure_module", "failure_type", "critical_failure_step"]
        + PREDICTOR_FIELDS
        + ["extraction_warnings", "parse_errors"]
    )

    with open(OUTPUT_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} rows to {OUTPUT_PATH}")
    print(f"Unmatched trajectories (no label found): {len(unmatched)}")
    if unmatched:
        for u in unmatched:
            print(f"  UNMATCHED: {u}")

    return rows, unmatched


if __name__ == "__main__":
    main()
