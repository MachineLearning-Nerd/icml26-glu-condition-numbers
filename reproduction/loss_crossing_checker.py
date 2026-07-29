"""Independent contract checker for serialized loss-crossing evidence."""

from __future__ import annotations


def check(result: dict) -> tuple[bool, list[str]]:
    failures: list[str] = []
    in_domain = []
    for row in result["rows"]:
        assumptions = (
            row["gaussian_input"]
            and row["d_plus_1_less_than_n"]
            and row["n_at_least_300"]
            and row["d_at_least_5"]
            and row["quadratic_form_nonnegative"]
        )
        if assumptions:
            in_domain.append(row)
            if not row["early_negative"]:
                failures.append(f"seed {row['seed']}: early ReLU advantage absent")
            if not row["crossing"]:
                failures.append(f"seed {row['seed']}: no first-hit crossing")
            elif not (row["delta_before"] <= 0.0 < row["delta_after"]):
                failures.append(f"seed {row['seed']}: invalid crossing bracket")
    if len(in_domain) < 3:
        failures.append(f"only {len(in_domain)} independently generated targets satisfy all assumptions")

    control = result["negative_control"]
    if control["assumption_satisfied"]:
        failures.append("negative control unexpectedly satisfies the quadratic-form assumption")
    if control["quadratic_form"] >= 0.0:
        failures.append("negative control failed for the intended reason")
    return not failures, failures

