# Claim 5 method

Attempt 1 used parameter-matched ReLU and ReGLU versions of the official
default MLP-Mixer (patch 4, dimension 256, depth 4). It was cancelled after
24m48s without completing even the epoch-zero 50,000-example evaluation; the
projected protocol could not fit its six-hour HF CPU ceiling. No scientific
result is inferred from that resource-design failure.

Attempt 2 trains the same official Mixer code path at patch 4, dimension 64,
and depth 2 on all 50,000 CIFAR-10 training examples,
using the official augmentation and Appendix D optimizer. Evaluate empirical
risk on all unaugmented training examples and population-risk proxy on all
10,000 test examples after each epoch. Use seeds 51-53, interpolate each
seed's gap onto 11 points in the common training-loss support, and perform
inference over seeds rather than treating correlated epochs as independent.

Both reduced capacity (dimension 64/depth 2 versus 256/4 defaults) and the
15-epoch horizon (versus the README's 100) are predeclared CPU-budget
deviations. The paper plot helper's flattened-coordinate energy test is
reproduced for transparency but is not the primary checker. The negative
control artificially lowers every matched ReGLU gap by 0.50; the limited-gap
verdict must then be rejected.

The first launch of Attempt 2 was cancelled before any scientific metric
because torchvision's opaque downloader again emitted no progress. The repair
uses the same official archive URL with an explicit User-Agent, a bounded
socket timeout, progress output, safe extraction, and the authoritative
SHA-256 `6d958be074577803d12ecdefd02955f39262c83c16fe9348329d7fe0b5c001ce`.
No data or scientific setting changed.

Attempt 3 is an independent statistical route over the frozen raw trajectory.
It treats each seed—not each correlated epoch—as one replicate and performs a
two-one-sided t test against the predeclared +/-0.10 CE practical margin. This
route can verify only the limited-gap premise; it cannot manufacture the
optimization premise.

Attempt 4 is dedicated to falsification. Before inspecting first-hit outcomes,
it fixes loss thresholds at 2.00, 1.75, 1.50, and 1.40 CE. It compares the
first epoch reaching each threshold in every paired seed and explicitly audits
all source assumptions. A scoped adverse result is accepted as a valid
falsification only if the default dimension/depth and 100-epoch assumptions
are also satisfied. They are not, so the route must reject the tempting
counterexample and return BLOCKED.
