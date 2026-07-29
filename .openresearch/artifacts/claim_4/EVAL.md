# Claim 4 evaluation

The formal run prints all rows, first-hit brackets, assumption audits, the
independent checker decision, seed list, CPU allocation, thread limit, and
runtime. No outcome is precomputed in this committed file.

Verdict rules:

- `VERIFIED`: the exact finite contract passes.
- `BLOCKED`: fewer than three predeclared targets are in-domain or any in-domain
  target lacks the stated two-stage behavior.
- The run exits nonzero when the evidence contract fails.

