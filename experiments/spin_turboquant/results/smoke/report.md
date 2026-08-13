# SpinTurboQuant experiment report

Generated: 2026-08-13T17:23:43.699716+00:00

## Verdict

**Supported in this initial experiment.** The strict automated criterion requires learned rotations to beat
their paired random initialization on held-out normalized-key MSE for every seed
in both domains, and on at least 75% of paired attention-KL comparisons pooled
over domains and sequence lengths. This is an initial controlled experiment, not
a claim of broad model-family generality.

## Fixed setup

- Model: `/home/elicer/.cache/huggingface/hub/models--NousResearch--Meta-Llama-3.1-8B/snapshots/1f47e50cdbe801ad8a5174156ec3a0655108fb9f` (32 layers,
  8 KV heads, head dimension 128)
- Calibration: WikiText-2 train, 64 tokens
- Held out in-domain: WikiText-2 validation, 64 tokens
- Held out cross-domain: TinyStories validation, 64 tokens
- Bit widths: 2
- Random seeds: 17, 29
- Optimization: 2 Adam steps, batch 32 tokens/head,
  learning rate 0.005
- Attention lengths: 32, 64
- Codec norm correction: False
- Existing LLM parameters and TurboQuant Lloyd-Max codebooks were frozen.

## Calibration objective

| bits | initial MSE | learned MSE | relative reduction | max abs(R^T R-I) |
|---|---|---|---|---|
| 2 | 0.00090592 | 0.000715062 | 21.07% | 0.000441 |

## Held-out paired results

Positive reductions mean learned is better than its exact paired random start.
Win rates are computed across seeds for reconstruction and across seeds, domains,
and sequence lengths for attention.

| bits | domain | identity MSE | random MSE | learned MSE | paired reduction | paired wins |
|---|---|---|---|---|---|---|
| 2 | wikitext | 0.00193162 | 0.00090456 | 0.000784628 | 13.26% | 100% |
| 2 | tinystories | 0.00197359 | 0.000905951 | 0.000809237 | 10.68% | 100% |

Attention probability KL divergence, pooled across both held-out domains and all
configured sequence lengths:

| bits | identity KL | random KL | learned KL | paired reduction | paired wins |
|---|---|---|---|---|---|
| 2 | 1.34962 | 0.0775994 | 0.0617802 | 20.57% | 100% |

## End-to-end results

The uncompressed model has perplexity 6.2227 and retrieval accuracy
1.000. Perplexity was measured on 64 held-out tokens;
retrieval used 1
prompts at lengths 128.

| bits | identity PPL | random PPL | learned PPL | paired PPL reduction | identity retrieval | random retrieval | learned retrieval |
|---|---|---|---|---|---|---|---|
| 2 | 2419.2735 | 11.3420 | 10.2029 | 10.06% | 1.000 | 1.000 | 1.000 |

## Cost

Each learned dense rotation set stores
16,777,216
bytes (16.0 MiB)
in FP32. Detailed measured dense-codec timings are in `latency.csv`; this prototype
does not claim an inference-speed improvement.

## Reproduce

```bash
conda run -n stq python -m experiments.spin_turboquant.run --model /home/elicer/.cache/huggingface/hub/models--NousResearch--Meta-Llama-3.1-8B/snapshots/1f47e50cdbe801ad8a5174156ec3a0655108fb9f --output-dir /home/elicer/SpinTurboQuant/turboquant_plus/experiments/spin_turboquant/results/smoke
```

Raw evidence: `training.csv`, `offline_metrics.csv`, `perplexity.csv`,
`retrieval.csv`, `retrieval_details.csv`, `latency.csv`, saved rotations, captured
activations, and `run_config.json` in this directory.
