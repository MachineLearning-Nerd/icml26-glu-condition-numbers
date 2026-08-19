#!/usr/bin/env python3
"""Verify the committed publication contract for this repository."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parent
EXPECTED_STATUS = (
    "PARTIAL_C1_C2_C3_C4_C6_VERIFIED_C5_BLOCKED_HISTORICAL_SCORE_8_OF_12_NO_CURRENT_SCORE"
)
EXPECTED_BRANCHES = {
    "audit/c4-loss-crossing",
    "audit/c5-cifar-mixer-gap",
    "audit/c5-independent-audit",
    "audit/c6-real-architecture-ntks",
    "historical/judged-baseline",
    "main",
    "release/evaluator-candidate",
    "release/provenance-correction",
}
EXPECTED_COMMITS = 19
CANONICAL_IDENTITY = "MachineLearning-Nerd <MachineLearning-Nerd@users.noreply.github.com>"
CLAIM_IDS = ["C1", "C2", "C3", "C4", "C5", "C6"]


def load(name: str):
    return json.loads((ROOT / name).read_text())


def git(*args: str) -> str:
    return subprocess.run(
        ["git", *args],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(f"verification failed: {message}")


def published_branches() -> set[str]:
    remote = {
        name.removeprefix("origin/")
        for name in git(
            "for-each-ref", "refs/remotes/origin", "--format=%(refname:short)"
        ).splitlines()
        if name.startswith("origin/") and name != "origin/HEAD"
    }
    if remote:
        return remote
    return set(git("for-each-ref", "refs/heads", "--format=%(refname:short)").splitlines())


def main() -> None:
    claims = load("claims.json")
    verdicts = load("reproduction_verdicts.json")
    manifest = load("EVIDENCE_MANIFEST.json")
    state = load("AUTONOMOUS_STATE.json")
    campaign = load("evidence/campaign_results.json")
    checker = load("evidence/checker_output.json")
    logbook = load("space/logbook.json")

    require(claims["overall_status"] == EXPECTED_STATUS, "claims overall status")
    require(state["overall_status"] == EXPECTED_STATUS, "autonomous state overall status")
    require(verdicts["overall_verdict"] == "PARTIAL_C1_C2_C3_C4_C6_VERIFIED_C5_BLOCKED_PROTOCOL", "overall verdict")
    expected_statuses = {
        "C1": "VERIFIED_SCOPED",
        "C2": "VERIFIED_SCOPED",
        "C3": "VERIFIED_SCOPED",
        "C4": "VERIFIED_SCOPED",
        "C5": "BLOCKED_PROTOCOL",
        "C6": "VERIFIED_SCOPED",
    }
    require([claim["id"] for claim in claims["claims"]] == CLAIM_IDS, "claim ordering")
    require({claim["id"]: claim["status"] for claim in claims["claims"]} == expected_statuses, "claim statuses")
    require(verdicts["claim_statuses"] == expected_statuses, "verdict statuses")
    require(all((ROOT / path).exists() for path in manifest["required_paths"]), "manifest paths")

    require(campaign["paper"]["source_sha256"] == claims["paper"]["source_sha256"], "source hash")
    campaign_statuses = {str(item["claim"]): item["status"] for item in campaign["claims"]}
    require(campaign_statuses == {"1": "VERIFIED", "2": "VERIFIED", "3": "VERIFIED", "4": "VERIFIED", "5": "BLOCKED", "6": "VERIFIED"}, "campaign statuses")
    require(checker["eval_status"] == "PASS", "cumulative checker")
    require(checker["checks"]["claim_5_blocked_classification_audit"]["passed"] is True, "Claim 5 blocked audit")
    require(checker["negative_controls"]["claim_5_assumption_violating_counterexample_rejected"] is True, "Claim 5 control")
    require(campaign["claims"][5]["observed"]["glu_wins"] == "24/24 paired architecture/activation/seed comparisons", "Claim 6 wins")
    require(logbook["space_id"] == "DineshAI/w0JhOFWPJl", "Space identity")

    require(verdicts["historical_external_result"]["live_judge_score"] == "8/12", "historical score")
    require(verdicts["historical_external_result"]["current_score_claim"] is False, "current score claim")
    require(verdicts["publication"]["publication_allowed"] is False, "publication state")
    require(verdicts["publication"]["author_endorsement_claimed"] is False, "author endorsement state")

    branches = published_branches()
    require(branches == EXPECTED_BRANCHES, "published branches")
    require(not any(branch.startswith("orx/") for branch in branches), "legacy orx branch")
    require(int(git("rev-list", "--all", "--count")) == EXPECTED_COMMITS, "reachable commit count")
    identities = git("log", "--all", "--format=%an <%ae>\n%cn <%ce>").splitlines()
    require(identities and all(identity == CANONICAL_IDENTITY for identity in identities), "canonical commit identity")

    print(
        "FINAL_AUDIT=VERIFIED "
        f"branches={len(branches)} commits={EXPECTED_COMMITS} "
        "claims=C1:C4_C6_verified_scoped,C5_blocked_protocol "
        "historical_score=8/12 current_score_claim=false publication_allowed=false"
    )


if __name__ == "__main__":
    main()
