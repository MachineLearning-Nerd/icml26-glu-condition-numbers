# Branch audit

The old `orx/*` names are retained here only as historical provenance. Each
branch is renamed to describe its claim or release role.

| Historical branch | Clean branch | Purpose |
| --- | --- | --- |
| `orx/locked-historical-regression-baseline` | `historical/judged-baseline` | Preserve the earlier judged Claims 1–3 result and the invalid `n=40` crossing baseline. |
| `orx/exact-corollary-4-2-loss-crossing` | `audit/c4-loss-crossing` | Verify the loss-crossing claim after enforcing every Corollary 4.2 assumption. |
| `orx/real-vit-and-gpt-2-ffn-ntks` | `audit/c6-real-architecture-ntks` | Measure parameter-matched FFN NTKs in ViT and GPT-2-small backbones. |
| `orx/full-cifar-10-mixer-generalization-gap` | `audit/c5-cifar-mixer-gap` | Compare matched-loss gaps and optimization on the CPU-feasible full-CIFAR route. |
| `orx/claim-5-independent-audit-and-falsification-rout` | `audit/c5-independent-audit` | Run the independent equivalence and assumption-preserving falsification dossier. |
| `orx/evaluator-visible-release-candidate` | `release/evaluator-candidate` | Preserve the evaluator-visible report, notebook, and release surface. |
| `orx/post-publication-provenance-correction` | `release/provenance-correction` | Preserve the post-publication provenance and source correction package. |

`main` is the publication surface. Superseded `orx/*` refs are deleted from
the live GitHub repository after the clean refs are published.
