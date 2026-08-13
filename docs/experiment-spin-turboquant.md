# Head-wise learned rotations for TurboQuant

Date: 2026-08-13

Specification: `../../SpinTurboQuant.md`

Implementation: `experiments/spin_turboquant/`

## Result

The distortion hypothesis is supported in this initial Llama-3.1-8B study, but
end-to-end perplexity and retrieval evidence is mixed. Learned rotations improve
held-out normalized-key MSE and attention KL for every paired random
initialization, domain, and evaluated sequence length. The same consistency does
not hold for perplexity at 3- and 4-bit or for multi-key retrieval.

## Setup

- Model: `NousResearch/Meta-Llama-3.1-8B`, BF16, frozen
- Hardware: NVIDIA A100 80GB PCIe MIG 3g.40gb (40 GB visible)
- Geometry: 32 layers, 8 KV heads/layer, head dimension 128
- Calibration: 4,096 WikiText-2 train tokens
- Held out: 1,024 WikiText-2 validation tokens and 1,024 TinyStories tokens
- Quantizers: fixed local TurboQuant+ 2/3/4-bit Lloyd-Max codebooks
- Initializations: paired dense Haar rotations at seeds 17, 29, and 43
- Training: 80 Adam steps, 256 tokens/head/step, learning rate 0.005
- Attention lengths: 128, 256, 512, and 1,024 tokens
- End-to-end retrieval lengths: 512, 1,024, and 2,048 tokens
- TurboQuant norm correction was disabled to match the specification's direct
  rotated-space objective.

Only the Cayley parameters were optimized. Model parameters, codebooks, and
per-token rotations were not changed.

## Correctness gates

Forward rotation followed by inverse rotation before RoPE was checked on real
captured Llama keys without quantization:

| Metric | Result |
|---|---:|
| Maximum pre-RoPE key error | 1.14e-5 |
| Pre-RoPE key MSE | 3.08e-13 |
| Maximum attention-logit error | 2.67e-5 |
| Attention-logit MSE | 1.47e-12 |
| Maximum learned `abs(R^T R - I)` during training audit | 5.64e-8 |
| Maximum Gram error after saved FP32 tensors were reloaded | 7.15e-7 |

## Calibration and held-out reconstruction

All reductions below pair each learned rotation with its exact random start.

| Bits | Calibration MSE reduction | WikiText held-out | TinyStories held-out | Improved layer/head pairs |
|---:|---:|---:|---:|---:|
| 2 | 41.07% | 30.05% | 30.74% | 100.0% of 1,536 |
| 3 | 21.34% | 14.41% | 13.91% | 99.8% of 1,536 |
| 4 | 16.24% | 11.77% | 11.61% | 100.0% of 1,536 |

The smaller held-out gains show a measurable generalization gap, but the gains
are broadly distributed rather than concentrated in a few heads.

## Held-out attention preservation

Results pool 24 paired comparisons per bit width: 3 seeds, 2 domains, and 4
sequence lengths.

| Bits | Random attention KL | Learned attention KL | Reduction | Paired wins |
|---:|---:|---:|---:|---:|
| 2 | 0.143433 | 0.095792 | 32.89% | 24/24 |
| 3 | 0.035107 | 0.029934 | 14.84% | 24/24 |
| 4 | 0.009220 | 0.008120 | 12.00% | 24/24 |

The raw table also records attention-logit MSE/bias/variance, attention-output
MSE, and worst-layer KL.

## End-to-end quality

The BF16 baseline perplexity is 5.4174. Perplexity uses 1,022 predicted tokens
from two 512-token held-out windows. Retrieval uses six multi-key prompts in
which every candidate answer appears in the context.

| Bits | Random PPL | Learned PPL | Mean paired change | PPL wins | Random retrieval | Learned retrieval |
|---:|---:|---:|---:|---:|---:|---:|
| 2 | 6.8753 | 6.4805 | 5.74% better | 3/3 | 0.944 | 0.889 |
| 3 | 5.6797 | 5.6252 | 0.92% better | 2/3 | 0.944 | 1.000 |
| 4 | 5.4524 | 5.4837 | 0.58% worse | 1/3 | 1.000 | 1.000 |

Thus reconstruction MSE and offline attention improvements do not by themselves
prove an end-to-end improvement. In particular, 4-bit learned rotations are
slightly worse on mean perplexity despite better distortion metrics.

## Cost

A complete FP32 learned rotation set contains 4,194,304 values and occupies
16,777,216 bytes (16 MiB). On 65,536 captured vectors, one dense forward
rotation takes about 0.338 ms (0.0052 microseconds/vector); the unfused Python
prototype's complete rotate/quantize/inverse-rotate path takes 1.44-1.49 ms.
These timings characterize the prototype and are not an optimized inference
kernel claim.

## Evidence and reproduction

The local run directory is
`experiments/spin_turboquant/results/main/`. It contains the full report, run
configuration, captured activations, rotations, and these raw tables:

- `training.csv` (9 rows)
- `offline_metrics.csv` (210 rows)
- `head_metrics.csv` (10,752 rows)
- `perplexity.csv` (22 rows)
- `retrieval.csv` and `retrieval_details.csv` (22 and 132 rows)
- `latency.csv` (21 rows)

Reproduce from the repository root with:

```bash
conda run -n stq python -m experiments.spin_turboquant.run \
  --model /home/elicer/.cache/huggingface/hub/models--NousResearch--Meta-Llama-3.1-8B/snapshots/1f47e50cdbe801ad8a5174156ec3a0655108fb9f \
  --output-dir experiments/spin_turboquant/results/main
```

The focused experiment tests pass 6/6. The wider repository suite passes 560
tests with 6 skips when the existing macOS-only `nvidia-smi` absence assertion
is deselected on this NVIDIA host.

