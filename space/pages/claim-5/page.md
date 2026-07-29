# Claim 5: BLOCKED after four routes

![Gap equivalence and optimization evidence](../../reports/reproduction/images/generalization_gap.svg)

## Exact combined claim

Section 5/Figure 7 attributes GLU’s primary benefit to faster optimization
rather than a materially smaller generalization gap in the considered
real-model experiments. Verification needs both premises. A falsification must
satisfy the source setting and contradict at least one premise.

## Direct full-CIFAR evidence

The answered experiment used all `50,000` CIFAR-10 training and `10,000` test
examples, official augmentation, SGD `lr=.005`, momentum `.9`, no weight
decay, parameter-matched ReLU (`78,986`) and ReGLU (`78,790`) Mixers, and
paired seeds `51–53`.

At matched training loss, ReGLU-minus-ReLU gap seed means are `−0.0000490`,
`−0.0045146`, and `+0.0022332` CE; mean `−0.0007768`; 95% interval
`[−0.0093029,+0.0077493]`. A TOST against the predeclared `±0.10` CE margin
passes (`p=.0001993/.0001932`). The gap component is practically equivalent.

ReGLU/ReLU loss-AUC ratios are `1.04828`, `1.01452`, `1.00526`; lower than
one would be faster. Acceleration is supported in `0/3` seeds.

## Four materially different routes

1. Source-default dimension `256`, depth `4`: resource-design attempt produced
   no scientific metric before cancellation.
2. Direct reduced-capacity, full-CIFAR training: gap limited, acceleration not
   observed.
3. Independent seed-level equivalence analysis: the gap premise is supported,
   but cannot supply acceleration.
4. Mandatory falsification route: fixed CE thresholds yield `1` ReGLU-faster,
   `4` ties, `7` ReGLU-slower. The assumption checker rejects this as valid
   falsification because dimension `64`/depth `2` and `15` epochs do not match
   source-default `256`/`4` and `100`.

Final status: `BLOCKED`, confidence `LOW`. The concrete unblocker is a
multi-seed, full-CIFAR run at the source-default capacity and 100-epoch
horizon, or another assumption-complete counterexample.

## Reproduce and inspect

Raw 96-row trajectory:
[`cifar10_mixer_raw.csv`](../../.openresearch/artifacts/claim_5/cifar10_mixer_raw.csv).
Routes: [`routes.md`](../../.openresearch/artifacts/claim_5/routes.md).
Contract: [`claim_contract.json`](../../.openresearch/artifacts/claim_5/claim_contract.json).
Checker output:
[`checker_output.json`](../../.openresearch/artifacts/claim_5/checker_output.json).
Code: [`generalization_gap.py`](../../reproduction/generalization_gap.py),
[`claim5_audit.py`](../../reproduction/claim5_audit.py), and
[`claim5_audit_checker.py`](../../reproduction/claim5_audit_checker.py).

Controls: a synthetic `−0.50` CE gap shift is rejected; a corrupted raw row is
rejected; the assumption-violating apparent counterexample is rejected.
The full run used HF `cpu-upgrade`, estimated/allocated eight CPUs, and took
`7,845.397 s`.
