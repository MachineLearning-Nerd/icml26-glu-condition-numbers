"""Exact finite evaluation of Proposition 4.1 under Corollary 4.2 assumptions."""

from __future__ import annotations

import math

import numpy as np

from .kernels import gaussian_inputs, reglu_ntk, relu_ntk


def _spectral_terms(kernel: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    eigvals, eigvecs = np.linalg.eigh((kernel + kernel.T) / 2.0)
    beta2 = (eigvecs.T @ target) ** 2
    return eigvals, beta2


def _expected_loss(eigvals: np.ndarray, beta2: np.ndarray, eta: float, step: int) -> float:
    # Proposition 4.1: Tr[(I-eta K)^(2k) K] + Y^T(I-eta K)^(2k)Y.
    decay = np.exp(2.0 * step * np.log1p(-eta * eigvals))
    return float(np.sum(decay * (eigvals + beta2)))


def _first_crossing(
    non_terms: tuple[np.ndarray, np.ndarray],
    glu_terms: tuple[np.ndarray, np.ndarray],
    eta: float,
) -> dict:
    def delta(step: int) -> float:
        return _expected_loss(*non_terms, eta, step) - _expected_loss(*glu_terms, eta, step)

    # Dense early horizon plus predeclared logarithmic horizons avoids choosing a
    # query budget from a claimed crossing formula.
    horizons = sorted(
        set(range(0, 257))
        | {int(round(v)) for v in np.geomspace(257, 10_000_000, num=800)}
    )
    values = [(k, delta(k)) for k in horizons]
    early_negative = all(v < 0.0 for k, v in values if k in (0, 1, 2))
    bracket = None
    previous = values[0]
    for current in values[1:]:
        if previous[1] <= 0.0 < current[1]:
            bracket = (previous[0], current[0])
            break
        previous = current
    if bracket is None:
        return {
            "early_negative": early_negative,
            "crossing": False,
            "first_positive_step": None,
            "searched_through": horizons[-1],
            "delta_at_0": values[0][1],
            "delta_at_last": values[-1][1],
        }

    low, high = bracket
    while high - low > 1:
        middle = (low + high) // 2
        if delta(middle) > 0.0:
            high = middle
        else:
            low = middle
    return {
        "early_negative": early_negative,
        "crossing": True,
        "first_positive_step": high,
        "searched_through": horizons[-1],
        "delta_before": delta(high - 1),
        "delta_after": delta(high),
        "delta_at_0": values[0][1],
    }


def run_exact_loss_crossing() -> dict:
    n, d, width = 300, 20, 5000
    rows = []
    for seed in (101, 102, 103, 104, 105):
        x = gaussian_inputs(n, d, seed)
        teacher = np.random.default_rng(seed + 10_000).standard_normal(d)
        target = np.maximum(x @ teacher, 0.0)
        non = relu_ntk(x, width)
        glu = reglu_ntk(x, width)
        qform = float(target @ (non - glu) @ target)
        non_terms = _spectral_terms(non, target)
        glu_terms = _spectral_terms(glu, target)
        lambda_max = max(float(non_terms[0][-1]), float(glu_terms[0][-1]))
        eta = 0.10 / lambda_max
        crossing = _first_crossing(non_terms, glu_terms, eta)
        rows.append(
            {
                "seed": seed,
                "n": n,
                "d": d,
                "width": width,
                "gaussian_input": True,
                "d_plus_1_less_than_n": d + 1 < n,
                "n_at_least_300": n >= 300,
                "d_at_least_5": d >= 5,
                "quadratic_form": qform,
                "quadratic_form_nonnegative": qform >= 0.0,
                "common_eta": eta,
                **crossing,
            }
        )

    # Assumption-violating control: the most negative eigendirection of K-K~.
    x = gaussian_inputs(n, d, 991)
    non = relu_ntk(x, width)
    glu = reglu_ntk(x, width)
    diff_eig, diff_vec = np.linalg.eigh((non - glu + (non - glu).T) / 2.0)
    control_target = diff_vec[:, 0]
    control_qform = float(control_target @ (non - glu) @ control_target)
    return {
        "claim": "Corollary 4.2 loss-crossing under its stated assumptions",
        "paper_anchor": "#S4.Thmtheorem2",
        "protocol": {
            "target": "independently seeded ReLU teacher",
            "common_step_size": "0.10 / max(lambda_max(K),lambda_max(K_tilde))",
            "horizons": "dense 0:256 plus 800 log-spaced integers through 10,000,000",
            "first_hit": "integer binary search inside first sign-change bracket",
        },
        "rows": rows,
        "negative_control": {
            "seed": 991,
            "construction": "minimum-eigenvector of K-K_tilde",
            "quadratic_form": control_qform,
            "assumption_satisfied": control_qform >= 0.0,
            "intended_rejection": "Y^T(K-K_tilde)Y is negative",
        },
    }

