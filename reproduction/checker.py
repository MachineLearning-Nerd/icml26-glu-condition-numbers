"""Independent, fixture-only checker with no imports from the kernel implementation."""

from __future__ import annotations


def check_historical_fixture(data: dict) -> tuple[bool, list[str]]:
    failures: list[str] = []
    c1 = data["claim_1"]
    if not (-1.30 <= c1["slope_non_glu"] <= -0.65):
        failures.append("non-GLU slope does not corroborate d^-1")
    if not (-2.30 <= c1["slope_glu"] <= -1.45):
        failures.append("GLU slope does not corroborate d^-2")
    if not c1["ratio_grows"]:
        failures.append("conditioning ratio does not grow")

    if data["claim_2"]["hadamard_relative_error"] >= 0.15:
        failures.append("Hadamard approximation error exceeds contract")

    c3 = data["claim_3"]
    if c3["lambda_max_glu_less_each_d"] is not True:
        failures.append("GLU largest eigenvalue is not uniformly smaller")
    if not all(b > a for a, b in zip(c3["lambda_min_width_sweep"][:-1], c3["lambda_min_width_sweep"][1:])):
        failures.append("minimum eigenvalue does not grow across the width sweep")

    c4 = data["claim_4"]
    if c4["crossing"] is not False:
        failures.append("historical raw output no longer records crossing=False")
    if c4["n"] >= 300:
        failures.append("negative control failed: historical n unexpectedly satisfies n>=300")
    return not failures, failures


def negative_control(data: dict) -> bool:
    """Return True only when an intentionally corrupted fixture is rejected."""
    corrupted = {
        **data,
        "claim_2": {**data["claim_2"], "hadamard_relative_error": 0.50},
    }
    accepted, _ = check_historical_fixture(corrupted)
    return not accepted
