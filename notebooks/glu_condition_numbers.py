import marimo

__generated_with = "0.23.15"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _(mo):
    mo.md(r"""
    # Why does a GLU gate improve conditioning?

    **Observed evidence first:** in exact FFN-parameter Jacobian Grams from
    full-width ViT and GPT-2-small backbones, GLU had a lower condition
    number in **24/24 paired comparisons**.
    """)
    return


@app.cell
def _(mo):
    ratios = {
        "ViT · ReLU → ReGLU": 0.6609539701,
        "ViT · GELU → GEGLU": 0.4692572303,
        "ViT · SiLU → SwiGLU": 0.4729704106,
        "GPT-2 · ReLU → ReGLU": 0.2429547933,
        "GPT-2 · GELU → GEGLU": 0.4731619271,
        "GPT-2 · SiLU → SwiGLU": 0.6115918297,
    }
    bars = "".join(
        f"""
        <div style="display:grid;grid-template-columns:190px 1fr 55px;gap:10px;
                    align-items:center;margin:12px 0">
          <span>{label}</span>
          <div style="height:24px;background:#172033;border-radius:5px">
            <div style="height:24px;width:{100 * ratio:.1f}%;
                        background:{'#38bdf8' if label.startswith('ViT') else '#a78bfa'};
                        border-radius:5px"></div>
          </div>
          <code>{ratio:.3f}</code>
        </div>
        """
        for label, ratio in ratios.items()
    )
    mo.Html(
        f"""
        <section style="padding:18px 22px;background:#0b1020;color:#e2e8f0;
                        border-radius:12px;font-family:system-ui">
          <h3 style="margin-top:0;color:#f8fafc">κ(GLU) / κ(non-GLU)</h3>
          <p style="color:#94a3b8">Geometric mean over four paired seeds.
          Every bar is below the no-change value 1.0; lower is better.</p>
          {bars}
          <small style="color:#94a3b8">n=8 inputs · seeds 41–44 · exact
          scalar-output Gram over all FFN parameters</small>
        </section>
        """
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## The idea in one equation

    The neural tangent kernel (NTK) describes first-order training dynamics.
    A large condition number

    \[
    \kappa(K)=\frac{\lambda_{\max}(K)}{\lambda_{\min}(K)}
    \]

    means that gradient descent learns some eigen-directions much faster
    than others. The paper approximates the gated kernel as

    \[
    \widetilde K \approx K \odot (XX^\top/d),
    \]

    predicting \(\kappa(\widetilde K)=O(n/d^2)\), versus
    \(\kappa(K)=O(n/d)\) without a gate.
    """)
    return


@app.cell
def _(mo):
    scaling = [
        {"d": 20, "non_glu": 192.2995, "glu": 43.0791},
        {"d": 40, "non_glu": 95.6695, "glu": 10.7134},
        {"d": 80, "non_glu": 50.7105, "glu": 4.1617},
    ]
    mo.vstack(
        [
            mo.md(
                """
                ## Finite scaling check

                The reproduced log–log slopes are **−0.96** for non-GLU and
                **−1.69** for GLU. The Hadamard approximation has relative
                Frobenius error **0.0372** at `d=40`.
                """
            ),
            mo.ui.table(scaling, selection=None),
            mo.callout(
                "This is finite numerical corroboration of an asymptotic "
                "theorem—not a proof of its universal big-O quantifiers.",
                kind="warn",
            ),
        ]
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## A repaired loss-crossing test

    The historical check used `n=40` and reported no crossing, but
    Corollary 4.2 assumes `n≥300`. With Gaussian inputs at `n=300, d=20`,
    all five independently generated targets crossed:

    | Seed | First crossing step |
    |---:|---:|
    | 101 | 1,834 |
    | 102 | 1,853 |
    | 103 | 1,943 |
    | 104 | 1,812 |
    | 105 | 1,939 |

    A minimum-eigenvector negative control made
    \(Y^T(K-\widetilde K)Y=-10{,}534\); the verifier rejected it because
    it violates the stated sign assumption.
    """)
    return


@app.cell
def _(mo):
    architecture = mo.ui.dropdown(
        options={
            "ViT": [0.6609539701, 0.4692572303, 0.4729704106],
            "GPT-2": [0.2429547933, 0.4731619271, 0.6115918297],
        },
        value="ViT",
        label="Architecture",
    )
    architecture
    return (architecture,)


@app.cell
def _(architecture, mo):
    labels = ["ReLU→ReGLU", "GELU→GEGLU", "SiLU→SwiGLU"]
    mo.vstack(
        [
            mo.md("### Explore the embedded real-architecture result"),
            mo.ui.table(
                [
                    {"activation pair": label, "κ ratio": round(value, 3)}
                    for label, value in zip(labels, architecture.value)
                ],
                selection=None,
            ),
        ]
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Where the evidence stops

    The full-CIFAR reduced Mixer gave a matched-loss gap difference of
    **−0.000777 CE**, with 95% interval **[−0.009303, +0.007749]**—well
    inside the predeclared ±0.10 equivalence margin. But ReGLU/ReLU
    training-loss AUC ratios were **1.0483, 1.0145, 1.0053**; lower than
    one would indicate acceleration.

    Because that run used dimension 64/depth 2 for 15 epochs instead of
    source-default dimension 256/depth 4 for 100 epochs, it cannot
    falsify the paper-level claim. The honest Claim 5 result is
    **BLOCKED**.

    ## Bottom line

    - Claims 1–4 and 6: **VERIFIED**, with stated finite/scoped limits.
    - Claim 5: **BLOCKED** after four different routes.
    - Previous live judge score: **8/12**.
    - Best-supported release forecast: **10/12**, not a judge result.

    This notebook embeds the evidence and does not require expensive
    experiments. To rerun the formal suite, use:

    ```bash
    uv sync --frozen --no-dev && uv run python -m reproduction.run_suite
    ```
    """)
    return


if __name__ == "__main__":
    app.run()
