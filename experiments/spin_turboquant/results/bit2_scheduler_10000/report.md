# 2-bit 10,000-step scheduler comparison

Completed both requested conditions with frozen `d10aef7999a2b5ba950ab3974312feeedbfe0b77` activations: cosine/seed 35 and exponential/seed 35. No other seeds were run.

## Protocol

- Initial/final LR: 0.005 / 0.00025
- Exponential gamma: 0.99970047164
- Steps per condition: 10000
- Full calibration/validation interval: 100
- Minibatch/log interval: 256 tokens/head, every 10 steps
- Both schedulers used the same seed-35 Haar rotation and minibatch-index stream.

## Selection

| Scheduler | Absolute minimum step | Selected step | Selected WikiText validation MSE |
|---|---:|---:|---:|
| cosine | 10000 | 8100 | 0.000503605196 |
| exponential | 10000 | 7400 | 0.000505775462 |

Primary WikiText winner: **cosine**.

## TinyStories domain sanity check

| Scheduler | Normalized key MSE | Original-scale key MSE | Attention KL | Logit MSE | Output MSE |
|---|---:|---:|---:|---:|---:|
| cosine | 0.000508252514 | 0.317863115 | 0.0773976679 | 0.387582718 | 0.000790020977 |
| exponential | 0.000509742147 | 0.318814077 | 0.0799039891 | 0.38874793 | 0.00082621366 |

TinyStories is a domain-generalization sanity check only; it does not override the WikiText selection criterion.

## Runtime

- Smoke-estimated two-run training: 3.17 minutes
- Actual two-run training: 3.19 minutes

LongBench-E, perplexity, and retrieval were not run, as required by the specification.

Raw per-condition evidence is under `runs/`; deterministic selected rotations are under `final_rotation_artifacts/`.
