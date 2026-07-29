"""Fixed cumulative command for every experiment node."""

from __future__ import annotations

import json
import os
import platform
import time
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_info, threadpool_limits

from .checker import check_historical_fixture, negative_control
from .kernels import gaussian_inputs, kernel_gd_loss, reglu_ntk, relu_ntk, spectrum
from .loss_crossing import run_exact_loss_crossing
from .loss_crossing_checker import check as check_loss_crossing
from .architecture_ntk import run_architecture_ntks
from .architecture_ntk_checker import check as check_architecture_ntks


ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / ".openresearch" / "artifacts" / "historical_regression" / "raw.json"


def cpu_allocation() -> dict:
    affinity = None
    if hasattr(os, "sched_getaffinity"):
        affinity = len(os.sched_getaffinity(0))
    quota = None
    cpu_max = Path("/sys/fs/cgroup/cpu.max")
    if cpu_max.exists():
        quota_text, period_text = cpu_max.read_text().strip().split()
        if quota_text != "max":
            quota = float(quota_text) / float(period_text)
    return {
        "os_cpu_count": os.cpu_count(),
        "affinity_cpu_count": affinity,
        "cgroup_cpu_quota": quota,
    }


def current_kernel_regression() -> dict:
    dimensions = [20, 40, 80]
    rows = []
    for d in dimensions:
        x = gaussian_inputs(40, d, seed=1)
        non = spectrum(relu_ntk(x, width=1000))
        glu = spectrum(reglu_ntk(x, width=1000))
        rows.append({"d": d, "non_glu": non, "glu": glu})
    slope_non = float(
        np.polyfit(np.log(dimensions), np.log([r["non_glu"]["condition_number"] for r in rows]), 1)[0]
    )
    slope_glu = float(
        np.polyfit(np.log(dimensions), np.log([r["glu"]["condition_number"] for r in rows]), 1)[0]
    )

    x = gaussian_inputs(40, 40, seed=2)
    non = relu_ntk(x, width=1000)
    glu = reglu_ntk(x, width=1000)
    hadamard = non * ((x @ x.T) / 40.0)
    rel_error = float(np.linalg.norm(glu - hadamard) / np.linalg.norm(glu))

    x_cross = gaussian_inputs(40, 40, seed=7)
    target = np.maximum(x_cross @ np.random.default_rng(8).standard_normal(40), 0.0)
    loss_non = kernel_gd_loss(relu_ntk(x_cross, 1500), target, 60)
    loss_glu = kernel_gd_loss(reglu_ntk(x_cross, 1500), target, 60)
    initial_sign = np.sign(loss_non[0] - loss_glu[0])
    crossing = any(np.sign(a - b) not in (0, initial_sign) for a, b in zip(loss_non, loss_glu))
    return {
        "dimensions": rows,
        "slopes": {"non_glu": slope_non, "glu": slope_glu},
        "hadamard_relative_error": rel_error,
        "historical_loss_crossing_protocol": {
            "n": 40,
            "d": 40,
            "crossing": bool(crossing),
            "assumption_n_at_least_300": False,
            "final_non_glu": loss_non[-1],
            "final_glu": loss_glu[-1],
        },
    }


def main() -> int:
    started = time.perf_counter()
    fixture = json.loads(FIXTURE.read_text())
    fixture_ok, fixture_failures = check_historical_fixture(fixture)
    control_ok = negative_control(fixture)

    with threadpool_limits(limits=1):
        current = current_kernel_regression()
        exact_crossing = run_exact_loss_crossing()
    crossing_ok, crossing_failures = check_loss_crossing(exact_crossing)
    architecture_ntks = run_architecture_ntks()
    architecture_ok, architecture_failures = check_architecture_ntks(architecture_ntks)

    current_ok = (
        -1.30 <= current["slopes"]["non_glu"] <= -0.65
        and -2.30 <= current["slopes"]["glu"] <= -1.35
        and current["hadamard_relative_error"] < 0.15
        and current["historical_loss_crossing_protocol"]["crossing"] is False
    )
    result = {
        "suite": "baseline-cumulative-regression",
        "git_sha": os.environ.get("ORX_GIT_SHA", "reported-by-orx-run"),
        "seed_set": [1, 2, 7, 8],
        "requested_core_estimate": 1,
        "selected_compute": "local only if <=5 minutes; otherwise hf cpu-upgrade",
        "cpu_allocation": cpu_allocation(),
        "platform": platform.platform(),
        "threadpool_before_limit": threadpool_info(),
        "enforced_thread_limit": 1,
        "fixture_checker": {"passed": fixture_ok, "failures": fixture_failures},
        "negative_control_rejected": control_ok,
        "current": current,
        "claim_4_exact": {
            "status": "VERIFIED" if crossing_ok else "BLOCKED",
            "checker_passed": crossing_ok,
            "checker_failures": crossing_failures,
            "evidence": exact_crossing,
        },
        "claim_6_real_architectures": {
            "status": "VERIFIED" if architecture_ok else "BLOCKED",
            "checker_passed": architecture_ok,
            "checker_failures": architecture_failures,
            "evidence": architecture_ntks,
        },
        "limitations": [
            "Claims 1-3 are finite numerical corroboration of asymptotic statements.",
            "The historical n=40 no-crossing result violates Corollary 4.2's n>=300 assumption and is rejected as a current falsification.",
            "Claims 5-6 are intentionally absent from the frozen baseline and must be added by child experiments.",
        ],
        "runtime_seconds": time.perf_counter() - started,
    }
    print("BEGIN_EVAL_JSON")
    print(json.dumps(result, indent=2, sort_keys=True))
    print("END_EVAL_JSON")
    passed = fixture_ok and control_ok and current_ok and crossing_ok and architecture_ok
    print(f"EVAL_STATUS={'PASS' if passed else 'FAIL'}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
