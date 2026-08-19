# Source audit entry point

## Primary source

- Paper: *The Devil is in the Condition Numbers: Why is GLU Better than non-GLU Structure?*
- Authors: Xingyu Lyu, Qianqian Xu, Zhiyong Yang, Peisong Wen, and Qingming Huang
- arXiv: [2605.20749](https://arxiv.org/abs/2605.20749)
- HTML source: <https://ar5iv.labs.arxiv.org/html/2605.20749>
- Retrieved: `2026-07-29T07:48:03Z`
- Source SHA-256: `d391f59973c02be99395f55c770d31c8728c5bdc4373244b59c4597dad246e4f`
- ICML submission identifier: `w0JhOFWPJl`

## Claim mapping

The current pages map C1 to Theorem 3.1’s condition-number scaling, C2 to
Section 3 Equation 5’s Hadamard approximation, C3 to Theorem 3.1’s extreme
eigenvalues, C4 to Proposition 4.1/Corollary 4.2 loss crossing, C5 to Section
5/Figure 7 generalization and optimization, and C6 to Figure 3’s ViT/GPT-2
FFN condition numbers.

Corollary 4.2’s exact contract requires iid Gaussian inputs, `d>=5`, `n>=300`,
`d+1<n`, a nonnegative quadratic form, a small early step, and a sufficiently
large late integer step. The historical `n=40` result is not used as current
evidence.

## Fidelity boundary

Claims 1–3 are finite checks of asymptotic mechanisms. Claim 4 enforces the
paper’s assumptions. Claim 5’s CPU campaign uses the paper’s data family and
matched-loss test but changes model capacity and training horizon, so it remains
blocked. Claim 6 is a scoped real-architecture NTK audit with `n=8` inputs and
an independent GPT-2 reconstruction.
