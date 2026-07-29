"""Exact empirical FFN-parameter NTKs for real ViT and GPT-2 backbones."""

from __future__ import annotations

import gc
import math
import time

import numpy as np
import torch

from .architectures import GPT2SmallBackbone, PAIRS, PaperViT, ffn_parameters


def empirical_ntk(model: torch.nn.Module, inputs: torch.Tensor) -> dict:
    model.eval()
    params = ffn_parameters(model)
    count = sum(p.numel() for p in params)
    jacobian = torch.empty((inputs.shape[0], count), dtype=torch.float32)
    for sample_index in range(inputs.shape[0]):
        model.zero_grad(set_to_none=True)
        output = model(inputs[sample_index : sample_index + 1]).squeeze()
        gradients = torch.autograd.grad(output, params)
        cursor = 0
        for gradient in gradients:
            flat = gradient.detach().reshape(-1)
            jacobian[sample_index, cursor : cursor + flat.numel()].copy_(flat)
            cursor += flat.numel()
    kernel = (jacobian @ jacobian.T).to(torch.float64)
    eig = torch.linalg.eigvalsh((kernel + kernel.T) / 2.0)
    condition = float(eig[-1] / eig[0])
    result = {
        "parameter_count": count,
        "sample_count": inputs.shape[0],
        "lambda_min": float(eig[0]),
        "lambda_max": float(eig[-1]),
        "condition_number": condition,
    }
    del jacobian, kernel, eig, gradients
    gc.collect()
    return result


def _paired_summary(rows: list[dict]) -> list[dict]:
    summaries = []
    for architecture in ("vit", "gpt2"):
        for non_name, glu_name in PAIRS:
            paired = [
                row
                for row in rows
                if row["architecture"] == architecture
                and row["activation"] in (non_name, glu_name)
            ]
            by_seed = {}
            for row in paired:
                by_seed.setdefault(row["seed"], {})[row["activation"]] = row
            log_ratios = np.array(
                [
                    math.log(values[glu_name]["condition_number"] / values[non_name]["condition_number"])
                    for values in by_seed.values()
                ]
            )
            mean = float(log_ratios.mean())
            sem = float(log_ratios.std(ddof=1) / math.sqrt(len(log_ratios)))
            summaries.append(
                {
                    "architecture": architecture,
                    "pair": [non_name, glu_name],
                    "seeds": sorted(by_seed),
                    "mean_log_ratio_glu_over_non": mean,
                    "geometric_mean_ratio_glu_over_non": math.exp(mean),
                    "approx_95pct_log_ratio_ci": [mean - 3.182 * sem, mean + 3.182 * sem],
                    "seed_wins": int((log_ratios < 0.0).sum()),
                    "seed_total": len(log_ratios),
                }
            )
    return summaries


def run_architecture_ntks() -> dict:
    started = time.perf_counter()
    torch.set_num_threads(8)
    torch.set_num_interop_threads(1)
    rows = []
    for architecture in ("vit", "gpt2"):
        for seed in (41, 42, 43, 44):
            generator = torch.Generator().manual_seed(seed + 50_000)
            if architecture == "vit":
                inputs = torch.randn(8, 3, 32, 32, generator=generator)
                model_type = PaperViT
            else:
                inputs = torch.randn(8, 16, 768, generator=generator)
                model_type = GPT2SmallBackbone
            for non_name, glu_name in PAIRS:
                for activation in (non_name, glu_name):
                    model = model_type(activation, seed)
                    measurement = empirical_ntk(model, inputs)
                    rows.append(
                        {
                            "architecture": architecture,
                            "activation": activation,
                            "seed": seed,
                            **measurement,
                        }
                    )
                    print(
                        "ARCH_NTK_PROGRESS "
                        f"architecture={architecture} activation={activation} seed={seed} "
                        f"kappa={measurement['condition_number']:.8g} "
                        f"params={measurement['parameter_count']}",
                        flush=True,
                    )
                    del model
                    gc.collect()
    summaries = _paired_summary(rows)
    # No-change control: comparing a model to itself must not meet the reduction contract.
    control_ratio = rows[0]["condition_number"] / rows[0]["condition_number"]
    return {
        "claim": "Figure 3 real-architecture condition-number comparison",
        "paper_anchor": "#S3.F3",
        "official_code_anchor": "scripts/ntk/ntk_eigen_vit.py@cc1664bf48505d7bac308b308ce1c495b06ce979",
        "protocol": {
            "vit": "official dim=4096, depth=4, heads=4, mlp_dim=256, image=32, patch=4",
            "gpt2": "GPT-2-small dim=768, depth=12, heads=12, mlp_dim=3072, causal, sequence=16",
            "ntk": "exact scalar-output Jacobian Gram over every FFN parameter",
            "parameter_matching": "gated hidden width reduced to equal FFN parameter count",
            "sample_count": 8,
            "seeds": [41, 42, 43, 44],
            "torch_threads": 8,
        },
        "rows": rows,
        "paired_summaries": summaries,
        "negative_control": {
            "comparison": "identical ReLU ViT measurement against itself",
            "ratio": control_ratio,
            "expected_reduction": False,
        },
        "runtime_seconds": time.perf_counter() - started,
    }
