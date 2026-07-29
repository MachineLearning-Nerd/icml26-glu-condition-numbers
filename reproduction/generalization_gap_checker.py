"""Independent checker for the Section 5 CIFAR-10 evidence."""

from __future__ import annotations

import math

import numpy as np
from scipy.stats import t


def _decision(evidence: dict, *, control_shift: float = 0.0) -> tuple[str, list[str]]:
    failures = []
    protocol = evidence["protocol"]
    if evidence["dataset_sizes"] != {"train": 50_000, "test": 10_000}:
        failures.append("not full CIFAR-10")
    if protocol["model"] != "official MLP-Mixer patch=4, dim=256, depth=4":
        failures.append("wrong architecture")
    if protocol["epochs"] != 15 or len(protocol["seeds"]) != 3:
        failures.append("wrong horizon or seed count")
    expected_rows = 2 * 3 * (protocol["epochs"] + 1)
    if len(evidence["rows"]) != expected_rows:
        failures.append(f"expected {expected_rows} raw rows")

    matched = evidence["summary"]["matched_loss_points"]
    seed_diffs = []
    for seed in protocol["seeds"]:
        values = [
            row["glu_minus_non_gap"] + control_shift
            for row in matched
            if row["seed"] == seed
        ]
        if len(values) != 11:
            failures.append(f"seed {seed} missing matched-loss grid")
        else:
            seed_diffs.append(float(np.mean(values)))

    if len(seed_diffs) == 3:
        mean_diff = float(np.mean(seed_diffs))
        sem = float(np.std(seed_diffs, ddof=1) / math.sqrt(3))
        radius = float(t.ppf(0.975, 2) * sem)
        ci = [mean_diff - radius, mean_diff + radius]
    else:
        mean_diff, ci = math.nan, [math.nan, math.nan]

    faster = sum(row["reglu_faster"] for row in evidence["summary"]["optimization"])
    optimization_supported = faster >= 2
    # "Limited advantage" is accepted only if the mean is practically small,
    # the uncertainty interval contains no-effect, and ReGLU is not >0.10 CE
    # better in two or more independently seeded matched-loss curves.
    materially_better = sum(value < -0.10 for value in seed_diffs)
    gap_limited = (
        abs(mean_diff) < 0.10
        and ci[0] <= 0.0 <= ci[1]
        and materially_better < 2
    )

    parameter_counts = evidence["parameter_counts"]
    mismatch = abs(parameter_counts["relu"] - parameter_counts["reglu"])
    mismatch /= parameter_counts["relu"]
    if mismatch > 0.01:
        failures.append("parameter count mismatch exceeds 1%")

    if failures:
        status = "BLOCKED"
    elif optimization_supported and gap_limited:
        status = "VERIFIED"
    elif materially_better >= 2 and ci[1] < -0.10:
        status = "FALSIFIED"
    else:
        status = "BLOCKED"
    details = {
        "mean_glu_minus_non_gap": mean_diff,
        "independent_t_95pct_ci": ci,
        "seed_mean_differences": seed_diffs,
        "reglu_faster_seed_count": faster,
        "optimization_supported": optimization_supported,
        "gap_limited": gap_limited,
        "materially_better_seed_count": materially_better,
    }
    return status, failures, details


def check(evidence: dict) -> tuple[str, list[str], dict, bool]:
    status, failures, details = _decision(evidence)
    control_status, _, control_details = _decision(
        evidence, control_shift=evidence["negative_control"]["shift"]
    )
    control_rejected = control_status != "VERIFIED" and not control_details["gap_limited"]
    if not control_rejected:
        failures.append("shifted-gap negative control was not rejected")
        status = "BLOCKED"
    return status, failures, details, control_rejected
