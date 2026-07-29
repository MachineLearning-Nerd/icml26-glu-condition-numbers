"""Independent NumPy reconstruction of the paper's expected ReLU/ReGLU NTKs."""

from __future__ import annotations

import numpy as np


def gaussian_inputs(n: int, d: int, seed: int) -> np.ndarray:
    return np.random.default_rng(seed).standard_normal((n, d))


def relu_ntk(x: np.ndarray, width: int) -> np.ndarray:
    """Equation (3), specialized to ReLU and LeCun initialization."""
    d = x.shape[1]
    gram = x @ x.T
    norms = np.linalg.norm(x, axis=1)
    denom = np.outer(norms, norms)
    rho = np.divide(gram, denom, out=np.zeros_like(gram), where=denom > 0)
    rho = np.clip(rho, -1.0, 1.0)
    angle = np.arccos(rho)
    relu_cov = (
        denom
        * (np.sqrt(np.maximum(0.0, 1.0 - rho**2)) + (np.pi - angle) * rho)
        / (2.0 * np.pi * d)
    )
    derivative_cov = (np.pi - angle) / (2.0 * np.pi)
    # m E[phi_i phi_j] + m sigma_v^2 E[phi'_i phi'_j] <x_i,x_j>,
    # with sigma_v^2 = 1/m.
    return width * relu_cov + derivative_cov * gram


def reglu_ntk(x: np.ndarray, width: int) -> np.ndarray:
    """Equation (4), specialized to ReLU and LeCun initialization."""
    d = x.shape[1]
    gram = x @ x.T
    norms = np.linalg.norm(x, axis=1)
    denom = np.outer(norms, norms)
    rho = np.divide(gram, denom, out=np.zeros_like(gram), where=denom > 0)
    rho = np.clip(rho, -1.0, 1.0)
    angle = np.arccos(rho)
    relu_cov = (
        denom
        * (np.sqrt(np.maximum(0.0, 1.0 - rho**2)) + (np.pi - angle) * rho)
        / (2.0 * np.pi * d)
    )
    derivative_cov = (np.pi - angle) / (2.0 * np.pi)
    sigma_v2 = 1.0 / width
    sigma_p2 = 1.0 / d
    return width * (
        (sigma_v2 + sigma_p2) * relu_cov * gram
        + sigma_v2 * sigma_p2 * derivative_cov * gram**2
    )


def spectrum(matrix: np.ndarray) -> dict[str, float]:
    eig = np.linalg.eigvalsh((matrix + matrix.T) / 2.0)
    return {
        "lambda_min": float(eig[0]),
        "lambda_max": float(eig[-1]),
        "condition_number": float(eig[-1] / eig[0]),
    }


def kernel_gd_loss(kernel: np.ndarray, target: np.ndarray, steps: int) -> list[float]:
    """Historical judged protocol: per-kernel eta=1/lambda_max."""
    eta = 1.0 / spectrum(kernel)["lambda_max"]
    residual = target.astype(np.float64, copy=True)
    transition = np.eye(kernel.shape[0]) - eta * kernel
    losses: list[float] = []
    for _ in range(steps):
        residual = transition @ residual
        losses.append(float(residual @ residual))
    return losses

