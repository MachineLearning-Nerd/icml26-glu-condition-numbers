# Experiment tree and formal runs

All nodes inherited the fixed command:

`uv sync --frozen --no-dev && uv run python -m reproduction.run_suite`

| Node | Experiment ID | Git SHA | Answered run | Result | HF CPU runtime |
|---|---|---|---|---|---:|
| Locked historical regression baseline | `7db21752-3d0e-4282-9ff4-3c5ed84d872b` | `b97a8950f187f45171e4dd92507fd4a2fbc59d86` | `fe50301a-…` | Claims 1–3 retained; old Claim 4 assumption violation exposed | `26 s` |
| Exact Corollary 4.2 loss crossing | `3c9c98e8-11c0-4fac-a89b-dc1090de5648` | `386bf4d8…` | `23596694-…` | Claim 4 `VERIFIED`, `5/5` crossings | `37 s` |
| Real ViT and GPT-2 FFN NTKs | `d526abc6-b5eb-4c06-9975-cc5168e387b2` | `915a208608f311f562d215eb39be2bd142257b56` | `cea9fb5e-…` | Claim 6 `VERIFIED` scoped, `24/24` wins | `90.69 s` |
| Full CIFAR-10 Mixer gap | `41460cff-5e19-4b43-89ca-c9a424dc6d56` | `e0fe32a3bd26a0732b42afbf39eedc112263c7c9` | `822e0d9a-9b4a-4f90-9f40-32413a073ef4` | Gap component supported; combined Claim 5 `BLOCKED` | `7,845.40 s` |
| Claim 5 audit and falsification route | `b1162b82-8b5d-4921-986a-dd1ef4e2d865` | `9e8f80607b38ffe350cd0112b46518d2f365c668` | `16f0b1b5-a0e8-40a8-b15c-b6352de9f2da` | Four-route dossier complete; cumulative `PASS` with Claim 5 `BLOCKED` | `90.82 s` |
| Evaluator-visible release candidate | `39421238-fda0-4744-9ff0-9087f64a86d4` | `559aa787b0e9855d7f6f0608b27064755564c9e9` | `3f69d08f-088e-4192-9ea3-c7b3bc22b6b1` | Canonical pages, report, notebook, visibility audit; cumulative `PASS` | `78.72 s` |

The tree is stacked: exact Claim 4 builds on the frozen baseline; Claim 6
builds on Claim 4; Claim 5 builds on Claim 6; the independent audit builds on
the full-CIFAR result; the publication candidate builds on the completed
science.

No GPU was used. HF account-side billing did not expose a trustworthy currency
total through `orx`, so the release reports runtimes and allocations rather
than inventing cost.
