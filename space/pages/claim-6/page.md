# Claim 6: real ViT and GPT-2 FFN NTKs

![GLU/non-GLU condition-number ratios](../../reports/reproduction/images/architecture_ratios.svg)

## Exact claim and protocol

Figure 3 claims that empirical condition numbers are lower for
parameter-matched GLU than non-GLU structures in real ViT and GPT-2
architectures.

The current experiment uses a ViT with dimension `4096`, depth `4`, and four
heads, plus GPT-2-small with dimension `768`, depth `12`, twelve heads, and a
causal sequence length `16`. It computes the exact scalar-output Jacobian Gram
over every FFN parameter for `n=8` inputs, activation pairs ReLU/ReGLU,
GELU/GEGLU, and SiLU/SwiGLU, and seeds `41–44`.

## Inline result

| Architecture | Pair | Geometric mean κ(GLU)/κ(non) | Paired wins |
|---|---|---:|---:|
| ViT | ReLU/ReGLU | `0.661` | `4/4` |
| ViT | GELU/GEGLU | `0.469` | `4/4` |
| ViT | SiLU/SwiGLU | `0.473` | `4/4` |
| GPT-2 | ReLU/ReGLU | `0.243` | `4/4` |
| GPT-2 | GELU/GEGLU | `0.473` | `4/4` |
| GPT-2 | SiLU/SwiGLU | `0.612` | `4/4` |

Status: `VERIFIED` scoped, confidence `MEDIUM`.
Aggregate result: GLU had the lower condition number in `24/24` paired
architecture/activation/seed comparisons.

The identical-ReLU measurement control has ratio `1.0` and is rejected as an
expected reduction. FFN parameter mismatches are at most `0.39%`.

## Code, raw data, and limitations

Raw 48-row CSV:
[`raw.csv`](../../.openresearch/artifacts/claim_6/raw.csv).
Code: [`architectures.py`](../../reproduction/architectures.py),
[`architecture_ntk.py`](../../reproduction/architecture_ntk.py), and
[`architecture_ntk_checker.py`](../../reproduction/architecture_ntk_checker.py).
Contract: [`claim_contract.json`](../../.openresearch/artifacts/claim_6/claim_contract.json).
Checker output:
[`checker_output.json`](../../.openresearch/artifacts/claim_6/checker_output.json).
Runtime: [`runtime.json`](../../.openresearch/artifacts/claim_6/runtime.json).

Material limitations: `n=8`, versus the official ViT script’s `n=64`; and the
GPT-2 path is independently reconstructed because the official code release
contains no GPT-2 NTK implementation or Figure 3 raw CSV. The model
width/depths themselves are not toy dimensions.
