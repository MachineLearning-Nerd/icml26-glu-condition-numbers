# Evaluator-blind pre-publication red team

The reviewer received only the freshly reconstructed candidate artifact and
the rubric. No repository-location hints were supplied. Traversal began at
`README.md`, then `logbook.json` and `pages/index.md`.

## Review 1

All links resolved, and all six claim rows exposed code, inline numbers, raw
data, checker, control, exact claim, and verdict. Two conclusions were less
discoverable than required:

- Claim 4’s canonical page showed five raw rows but did not say `5/5` in plain
  aggregate text.
- Claim 6’s canonical page showed six `4/4` rows but did not say `24/24` in
  plain aggregate text.

Both canonical pages were fixed; no scientific result changed.

## Review 2 after fixes

Result: `PASS`. Missing links: `0`. Required-content failures: `0`. The old
13-file path set is a subset of the candidate, and the historical overview
hash remains unchanged.

Every file opened during the second traversal:

1. `README.md`
2. `logbook.json`
3. `pages/index.md`
4. `pages/current/page.md`
5. `reports/reproduction/images/architecture_ratios.svg`
6. `reproduction/run_suite.py`
7. `reproduction/checker.py`
8. `reproduction/loss_crossing_checker.py`
9. `reproduction/claim5_audit_checker.py`
10. `reproduction/architecture_ntk_checker.py`
11. `pyproject.toml`
12. `uv.lock`
13. `evidence/campaign_results.json`
14. `evidence/checker_output.json`
15. `evidence/compute.json`
16. `evidence/source_audit.md`
17. `evidence/commands.md`
18. `reports/reproduction/report.md`
19. `pages/claims-1-3/page.md`
20. `reports/reproduction/images/scaling.svg`
21. `reproduction/kernels.py`
22. `.openresearch/artifacts/historical_regression/raw.json`
23. `pages/claim-4/page.md`
24. `reports/reproduction/images/loss_crossing.svg`
25. `reproduction/loss_crossing.py`
26. `.openresearch/artifacts/claim_4/raw.csv`
27. `.openresearch/artifacts/claim_4/claim_contract.json`
28. `.openresearch/artifacts/claim_4/checker_output.json`
29. `.openresearch/artifacts/claim_4/runtime.json`
30. `pages/claim-5/page.md`
31. `reports/reproduction/images/generalization_gap.svg`
32. `.openresearch/artifacts/claim_5/cifar10_mixer_raw.csv`
33. `.openresearch/artifacts/claim_5/routes.md`
34. `.openresearch/artifacts/claim_5/claim_contract.json`
35. `.openresearch/artifacts/claim_5/checker_output.json`
36. `reproduction/generalization_gap.py`
37. `reproduction/claim5_audit.py`
38. `pages/claim-6/page.md`
39. `.openresearch/artifacts/claim_6/raw.csv`
40. `reproduction/architectures.py`
41. `reproduction/architecture_ntk.py`
42. `.openresearch/artifacts/claim_6/claim_contract.json`
43. `.openresearch/artifacts/claim_6/checker_output.json`
44. `.openresearch/artifacts/claim_6/runtime.json`
45. `pages/release/page.md`
46. `evidence/experiment_tree.md`
47. `evidence/release_command_log.md`
48. `pages/overview/page.md`

Conclusions directly verified: Claims 1–4 and 6 have current executable
evidence and limitations; Claim 5 is explicitly `BLOCKED` after four routes;
the forecast is not presented as a judge result; the historical weak verifier
is reachable but not the default.
