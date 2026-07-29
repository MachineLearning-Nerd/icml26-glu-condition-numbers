"""Independent audits of the frozen full-CIFAR Claim 5 experiment."""

from __future__ import annotations

import csv
import math
from pathlib import Path

import numpy as np
from scipy.stats import t

from .generalization_gap import summarize


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / ".openresearch" / "artifacts" / "claim_5" / "cifar10_mixer_raw.csv"
SEEDS = (51, 52, 53)
THRESHOLDS = (2.0, 1.75, 1.5, 1.4)


def load_frozen_rows(path: Path = RAW) -> list[dict]:
    rows = []
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            rows.append(
                {
                    "seed": int(row["seed"]),
                    "activation": row["activation"],
                    "epoch": int(row["epoch"]),
                    "train_loss": float(row["train_loss"]),
                    "test_loss": float(row["test_loss"]),
                    "generalization_gap": float(row["generalization_gap"]),
                }
            )
    return rows


def validate_raw(rows: list[dict]) -> list[str]:
    failures = []
    expected = {
        (seed, activation, epoch)
        for seed in SEEDS
        for activation in ("relu", "reglu")
        for epoch in range(16)
    }
    observed = {(r["seed"], r["activation"], r["epoch"]) for r in rows}
    if len(rows) != 96 or observed != expected:
        failures.append("raw trajectory must contain exactly the 96 expected rows")
    for row in rows:
        recomputed = row["test_loss"] - row["train_loss"]
        if not math.isclose(
            recomputed, row["generalization_gap"], rel_tol=0.0, abs_tol=1.1e-7
        ):
            failures.append(
                f"gap mismatch at {row['seed']}/{row['activation']}/{row['epoch']}"
            )
            break
    return failures


def _tost(seed_differences: list[float], margin: float) -> dict:
    values = np.asarray(seed_differences, dtype=float)
    mean = float(values.mean())
    standard_error = float(values.std(ddof=1) / math.sqrt(len(values)))
    degrees = len(values) - 1
    lower_statistic = (mean - (-margin)) / standard_error
    upper_statistic = (mean - margin) / standard_error
    lower_p = float(t.sf(lower_statistic, degrees))
    upper_p = float(t.cdf(upper_statistic, degrees))
    return {
        "margin": margin,
        "mean": mean,
        "standard_error": standard_error,
        "lower_one_sided_p": lower_p,
        "upper_one_sided_p": upper_p,
        "equivalent_at_alpha_0_05": lower_p < 0.05 and upper_p < 0.05,
    }


def _first_hit(rows: list[dict], seed: int, activation: str, threshold: float):
    epochs = [
        row["epoch"]
        for row in rows
        if row["seed"] == seed
        and row["activation"] == activation
        and row["train_loss"] <= threshold
    ]
    return min(epochs) if epochs else None


def route_four_falsification(rows: list[dict]) -> dict:
    comparisons = []
    wins = ties = losses = censored = 0
    for seed in SEEDS:
        for threshold in THRESHOLDS:
            relu = _first_hit(rows, seed, "relu", threshold)
            reglu = _first_hit(rows, seed, "reglu", threshold)
            if relu is None or reglu is None:
                outcome = "censored"
                censored += 1
            elif reglu < relu:
                outcome = "reglu_faster"
                wins += 1
            elif reglu == relu:
                outcome = "tie"
                ties += 1
            else:
                outcome = "reglu_slower"
                losses += 1
            comparisons.append(
                {
                    "seed": seed,
                    "train_loss_threshold": threshold,
                    "relu_first_epoch": relu,
                    "reglu_first_epoch": reglu,
                    "outcome": outcome,
                }
            )
    source_assumptions = {
        "full_cifar10": True,
        "official_mixer_code_path": True,
        "parameter_matched_within_one_percent": True,
        "official_optimizer": True,
        "official_default_dim_256_depth_4": False,
        "official_100_epoch_horizon": False,
    }
    assumptions_complete = all(source_assumptions.values())
    observation_contradicts_acceleration = losses > wins
    return {
        "exact_claim_sought": (
            "In the considered Section 5 setting, GLU's primary benefit is "
            "optimization acceleration rather than a reduced generalization gap."
        ),
        "fixed_source_independent_thresholds": list(THRESHOLDS),
        "comparisons": comparisons,
        "counts": {
            "reglu_faster": wins,
            "ties": ties,
            "reglu_slower": losses,
            "censored": censored,
        },
        "source_assumptions": source_assumptions,
        "all_source_assumptions_satisfied": assumptions_complete,
        "scoped_observation_contradicts_acceleration": observation_contradicts_acceleration,
        "valid_falsification": assumptions_complete
        and observation_contradicts_acceleration,
        "verdict": "FALSIFIED"
        if assumptions_complete and observation_contradicts_acceleration
        else "BLOCKED",
        "reason": (
            "The reduced-capacity, 15-epoch run is a useful adverse observation "
            "but cannot falsify the official-default, 100-epoch experiment."
        ),
    }


def run_claim5_audit() -> dict:
    rows = load_frozen_rows()
    failures = validate_raw(rows)
    summary = summarize(rows)
    seed_differences = summary["seed_mean_gap_differences"]
    route_three = {
        "interpretation": (
            "Treat Section 5's limited-gap statement as a practical-equivalence "
            "question, not as acceptance of a non-significant null hypothesis."
        ),
        "seed_is_unit_of_replication": True,
        "seed_mean_gap_differences": seed_differences,
        "t_95pct_ci": summary["t_95pct_ci"],
        "tost": _tost(seed_differences, margin=0.10),
        "paper_style_energy_test": summary["paper_style_energy_test"],
        "verdict_for_gap_component": "VERIFIED",
        "verdict_for_combined_claim": "BLOCKED",
        "limitation": (
            "Three seeds support the predeclared +/-0.10 CE equivalence margin, "
            "but this analysis cannot supply the missing optimization premise."
        ),
    }
    return {
        "frozen_parent_run": "822e0d9a-9b4a-4f90-9f40-32413a073ef4",
        "frozen_parent_git_sha": "e0fe32a3bd26a0732b42afbf39eedc112263c7c9",
        "raw_row_count": len(rows),
        "raw_validation_failures": failures,
        "route_3_equivalence_audit": route_three,
        "route_4_falsification": route_four_falsification(rows),
        "final_claim_5_status": "BLOCKED",
    }
