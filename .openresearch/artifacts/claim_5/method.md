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
