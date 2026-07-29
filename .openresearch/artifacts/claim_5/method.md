# Claim 5 method

Train parameter-matched ReLU and ReGLU versions of the official MLP-Mixer
(patch 4, dimension 256, depth 4) on all 50,000 CIFAR-10 training examples,
using the official augmentation and Appendix D optimizer. Evaluate empirical
risk on all unaugmented training examples and population-risk proxy on all
10,000 test examples after each epoch. Use seeds 51-53, interpolate each
seed's gap onto 11 points in the common training-loss support, and perform
inference over seeds rather than treating correlated epochs as independent.

The 15-epoch horizon is a predeclared CPU-budget deviation from the official
README's 100 epochs. The paper plot helper's flattened-coordinate energy test
is reproduced for transparency but is not the primary checker. The negative
control artificially lowers every matched ReGLU gap by 0.50; the limited-gap
verdict must then be rejected.
