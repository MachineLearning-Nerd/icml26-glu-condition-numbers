# Claim 5 low-confidence route record

## Route 1 — source-default capacity

Interpretation: reproduce the Figure 7(a) MLP-Mixer/CIFAR-10 source script at
its default dimension 256 and depth 4. The run used the fixed project command
on HF `cpu-upgrade` and was cancelled after 25m24s without an epoch-zero
metric. This is a resource-design result only; it neither verifies nor
falsifies the paper.

## Route 2 — direct real-data scoped reproduction

Interpretation: preserve the real architecture, complete dataset, official
optimizer and augmentations, but reduce capacity to dimension 64/depth 2 and
horizon to 15 epochs. The three-seed full-data run directly bounded the
matched-loss gap, while ReGLU had higher loss AUC in all three seeds. The
capacity and horizon deviations block an exact verdict.

## Route 3 — independent equivalence analysis

Interpretation: the phrase "limited generalization gap" is a practical
equivalence statement. A seed-level TOST with a predeclared +/-0.10 CE margin
is the primary analysis. It supports equivalence of the gap component. The
paper's flattened-coordinate energy test is reproduced only as secondary
evidence because epochs are dependent and the statistic mixes coordinates.
This route does not supply optimization acceleration, so the combined claim
remains BLOCKED.

## Route 4 — mandatory falsification search

Exact claim sought: in the considered Section 5 setting, GLU's primary benefit
is faster optimization rather than a smaller generalization gap. Fixed
source-independent loss thresholds (2.00, 1.75, 1.50, 1.40 CE) test
time-to-target. The scoped trajectories contain more ReGLU-slower outcomes
than ReGLU-faster outcomes. However, an independent assumption audit rejects
this as a valid counterexample because dimension 64/depth 2 and 15 epochs do
not match the source-default dimension 256/depth 4 and 100 epochs. This route
therefore ends BLOCKED, not FALSIFIED.

## Final status

`BLOCKED`. Unblocking requires a multi-seed, full-CIFAR run at the official
default capacity and horizon (or an assumption-complete counterexample) with
both optimization and matched-loss gap measurements.
