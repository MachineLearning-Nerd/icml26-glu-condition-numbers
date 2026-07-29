# Claim 5 source audit

Source: ar5iv HTML for arXiv:2605.20749, retrieved 2026-07-29 with an
explicit browser User-Agent; SHA-256
`d391f59973c02be99395f55c770d31c8728c5bdc4373244b59c4597dad246e4f`.

Section 5 says that GLU and non-GLU models have highly overlapping joint
distributions of training loss and generalization gap, and that at a given
training loss the gaps are nearly indistinguishable. It reports a two-sample
energy-distance permutation result of p >= 0.05. Figure 7(a) is MLP-Mixer on
CIFAR-10. Appendix D specifies standard CIFAR-10 augmentation and SGD with
learning rate 0.005 and no additional regularization.

The prose does not define "limited", "nearly indistinguishable", an effect
margin, the number of random seeds, or a confidence interval. A failure to
reject at p >= 0.05 is not evidence of equivalence by itself. This reproduction
therefore predeclares a 0.10-cross-entropy practical margin and treats the
paper-style energy test only as secondary evidence.
