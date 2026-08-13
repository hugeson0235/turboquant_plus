# Channel-Clustered TurboQuant experiment

Run completed: 2026-08-13T17:00:11.781496+00:00

Calibration used WikiText-2 train tokens. Every reported evaluation uses disjoint WikiText-2 validation/test tokens. Quantization occurs at `k_proj`, before RoPE.

## Reconstruction and attention

| bits | K | effective b/ch | seq | key MSE | normalized MSE | logit MSE | prob KL | partition ARI |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 3 | 1 | 3.125 | 512 | 1.3954e-01 +/- 6.7e-04 | 2.6398e-04 | 2.2356e-01 | 3.2284e-02 | 1.000 |
| 3 | 1 | 3.125 | 2048 | 1.5188e-01 +/- 6.7e-04 | 2.6432e-04 | nan | nan | 1.000 |
| 3 | 2 | 3.250 | 512 | 1.1154e-01 +/- 4.9e-04 | 2.1346e-04 | 3.1471e-01 | 4.3204e-02 | 0.882 |
| 3 | 2 | 3.250 | 2048 | 1.2028e-01 +/- 3.0e-04 | 2.1189e-04 | nan | nan | 0.901 |
| 3 | 4 | 3.500 | 512 | 9.7621e-02 +/- 3.6e-04 | 1.8715e-04 | 2.7728e-01 | 3.8541e-02 | 0.636 |
| 3 | 4 | 3.500 | 2048 | 1.0500e-01 +/- 1.9e-04 | 1.8530e-04 | nan | nan | 0.694 |

Effective rate includes all fp16/fp32 group norms. Rotation, codebook, and partition metadata are reported separately in the CSV because they are static across tokens.

## Downstream baseline

Unquantized perplexity: 4.456847812543631

Unquantized retrieval: 3/3

## Quantized downstream results

| bits | K | effective b/ch | PPL | retrieval | seconds |
| --- | --- | --- | --- | --- | --- |
| 3 | 1 | 3.125 | 4.6991 | 1.000 | 5.75 |
| 3 | 2 | 3.250 | 4.7524 | 1.000 | 5.83 |
| 3 | 4 | 3.500 | 4.7292 | 1.000 | 6.00 |

## Result against K=1

| bits | K | rate delta | key MSE delta | logit MSE delta | PPL delta | retrieval |
| --- | --- | --- | --- | --- | --- | --- |
| 3 | 2 | +4.0% | -20.1% | +40.8% | +1.1% | 1.000 |
| 3 | 4 | +12.0% | -30.0% | +24.0% | +0.6% | 1.000 |

Channel grouping improves reconstruction but does not satisfy the main validation criterion in this run. Both grouped conditions use more dynamic storage than K=1 and have worse attention-logit MSE; mean downstream perplexity does not improve. The hypothesis is therefore not supported for this 3-bit Llama-3.1-8B setting. This is a scoped experimental conclusion, not a claim about other bit widths, calibration sets, or grouping objectives.

## Interpretation guardrail

K=1, K=2, and K=4 use the same scalar bit width and norm precision, but their effective rates differ because K group norms are stored per token. A same-bit improvement is therefore not by itself proof of an equal-storage improvement. Use the effective-rate column and, when multiple bit widths are run, the Pareto frontier before accepting the proposal's main validation criterion.

Raw per-seed values, worst layer/head errors, latency, and retrieval candidate scores are in `metrics.csv`, `downstream.csv`, and `results.json`.
