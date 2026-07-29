"""Independent checker for the Claim 5 route dossier."""

from __future__ import annotations

import copy

from .claim5_audit import load_frozen_rows, run_claim5_audit, validate_raw


def check(evidence: dict) -> tuple[bool, list[str], dict]:
    failures = list(evidence["raw_validation_failures"])
    route_three = evidence["route_3_equivalence_audit"]
    route_four = evidence["route_4_falsification"]
    if not route_three["tost"]["equivalent_at_alpha_0_05"]:
        failures.append("practical-equivalence TOST did not reject both nulls")
    if route_three["verdict_for_combined_claim"] != "BLOCKED":
        failures.append("gap-only route must not resolve combined Claim 5")
    if route_four["valid_falsification"]:
        failures.append("downscaled experiment was incorrectly accepted as falsification")
    if route_four["verdict"] != "BLOCKED":
        failures.append("falsification route must remain BLOCKED")
    if evidence["final_claim_5_status"] != "BLOCKED":
        failures.append("final Claim 5 status must be BLOCKED")

    # Negative control 1: one corrupted raw loss/gap relation must be caught.
    tampered_rows = copy.deepcopy(load_frozen_rows())
    tampered_rows[0]["generalization_gap"] += 0.01
    tamper_rejected = bool(validate_raw(tampered_rows))

    # Negative control 2: an assumption-violating apparent counterexample must
    # not be promoted to FALSIFIED.
    invalid_counterexample_rejected = (
        route_four["scoped_observation_contradicts_acceleration"]
        and not route_four["all_source_assumptions_satisfied"]
        and not route_four["valid_falsification"]
    )
    if not tamper_rejected:
        failures.append("tampered raw-data control was not rejected")
    if not invalid_counterexample_rejected:
        failures.append("invalid-counterexample control was not rejected")
    controls = {
        "tampered_raw_row_rejected": tamper_rejected,
        "assumption_violating_counterexample_rejected": invalid_counterexample_rejected,
    }
    return not failures, failures, controls
