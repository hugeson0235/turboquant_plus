# SpinTurboQuant experiment report

Generated: 2026-08-13T17:35:24.576650+00:00

## Verdict

**Distortion hypothesis supported; end-to-end perplexity evidence is mixed.** The strict automated criterion requires learned rotations to beat
their paired random initialization on held-out normalized-key MSE for every seed
in both domains and on every paired attention-KL comparison across seeds,
domains, and sequence lengths. This is an initial controlled experiment, not a
claim of broad model-family generality.

## Fixed setup

- Model: `/home/elicer/.cache/huggingface/hub/models--NousResearch--Meta-Llama-3.1-8B/snapshots/1f47e50cdbe801ad8a5174156ec3a0655108fb9f` (32 layers,
  8 KV heads, head dimension 128)
- Calibration: WikiText-2 train, 4096 tokens
- Held out in-domain: WikiText-2 validation, 1024 tokens
- Held out cross-domain: TinyStories validation, 1024 tokens
- Bit widths: 2, 3, 4
- Random seeds: 17, 29, 43
- Optimization: 80 Adam steps, batch 256 tokens/head,
  learning rate 0.005
- Attention lengths: 128, 256, 512, 1024
- Codec norm correction: False
- Existing LLM parameters and TurboQuant Lloyd-Max codebooks were frozen.

## No-quantization correctness gate

On real captured keys, forward rotation followed by inverse rotation before RoPE
had maximum key error 1.14e-05 and
maximum attention-logit error
2.67e-05
(MSE 1.47e-12). This verifies the
pre-RoPE restore-then-RoPE path independently of quantization.

## Calibration objective

| bits | initial MSE | learned MSE | relative reduction | max abs(R^T R-I) |
|---|---|---|---|---|
| 2 | 0.000906118 | 0.000533979 | 41.07% | 4.54e-08 |
| 3 | 0.000265851 | 0.000209125 | 21.34% | 4.81e-08 |
| 4 | 7.31713e-05 | 6.12856e-05 | 16.24% | 5.64e-08 |

The explicit calibration-to-held-out gap and the fraction of layer/head pairs
that improve are:

| bits | calibration reduction | WikiText reduction | WikiText gap | TinyStories reduction | TinyStories gap | head-pair wins |
|---|---|---|---|---|---|---|
| 2 | 41.07% | 30.05% | -11.02 pp | 30.74% | -10.33 pp | 100.0% |
| 3 | 21.34% | 14.41% | -6.92 pp | 13.91% | -7.43 pp | 99.8% |
| 4 | 16.24% | 11.77% | -4.47 pp | 11.61% | -4.63 pp | 100.0% |

## Held-out paired results

Positive reductions mean learned is better than its exact paired random start.
Win rates are computed across seeds for reconstruction and across seeds, domains,
and sequence lengths for attention.

| bits | domain | identity MSE | random MSE | learned MSE | paired reduction | paired wins |
|---|---|---|---|---|---|---|
| 2 | wikitext | 0.00190875 | 0.000905029 | 0.000633054 | 30.05% | 100% |
| 2 | tinystories | 0.00197552 | 0.000905742 | 0.000627305 | 30.74% | 100% |
| 3 | wikitext | 0.000951578 | 0.000265519 | 0.000227244 | 14.41% | 100% |
| 3 | tinystories | 0.000997777 | 0.000265218 | 0.000228331 | 13.91% | 100% |
| 4 | wikitext | 0.00052401 | 7.30492e-05 | 6.44516e-05 | 11.77% | 100% |
| 4 | tinystories | 0.000555679 | 7.28934e-05 | 6.44311e-05 | 11.61% | 100% |

Attention probability KL divergence, pooled across both held-out domains and all
configured sequence lengths:

| bits | identity KL | random KL | learned KL | paired reduction | paired wins |
|---|---|---|---|---|---|
| 2 | 1.76298 | 0.143433 | 0.0957922 | 32.89% | 100% |
| 3 | 1.09935 | 0.0351065 | 0.029934 | 14.84% | 100% |
| 4 | 0.696291 | 0.00921993 | 0.00811986 | 12.00% | 100% |

## End-to-end results

The uncompressed model has perplexity 5.4174 and retrieval accuracy
1.000. Perplexity was measured on 1024 held-out tokens;
retrieval used 6
multi-key prompts at lengths 512, 1024, 2048.
The target margin is the target-token logit minus the strongest distractor logit.

| bits | identity PPL | random PPL | learned PPL | paired PPL reduction | PPL wins | identity retrieval | random retrieval | learned retrieval | random margin | learned margin |
|---|---|---|---|---|---|---|---|---|---|---|
| 2 | 305.3314 | 6.8753 | 6.4805 | 5.74% | 100% | 0.667 | 0.944 | 0.889 | 5.052 | 4.056 |
| 3 | 69.9509 | 5.6797 | 5.6252 | 0.92% | 67% | 0.667 | 0.944 | 1.000 | 7.870 | 7.477 |
| 4 | 14.0548 | 5.4524 | 5.4837 | -0.58% | 33% | 1.000 | 1.000 | 1.000 | 7.665 | 7.972 |

The main reconstruction/attention criterion is
satisfied. End-to-end perplexity
improves for all seeds only when the PPL win column is 100%; retrieval accuracy
may saturate, so the candidate margin is reported as a secondary diagnostic.

## Cost

Each learned dense rotation set stores
16,777,216
bytes (16.0 MiB)
in FP32. Detailed measured dense-codec timings are in `latency.csv`; this prototype
does not claim an inference-speed improvement.

| bits | method | forward rotation ms | us/vector | full codec ms |
|---|---|---|---|---|
| 2 | identity | 0.337 | 0.0051 | 1.441 |
| 2 | random | 0.338 | 0.0052 | 1.444 |
| 2 | learned | 0.338 | 0.0052 | 1.445 |
| 3 | identity | 0.336 | 0.0051 | 1.460 |
| 3 | random | 0.338 | 0.0052 | 1.462 |
| 3 | learned | 0.338 | 0.0052 | 1.463 |
| 4 | identity | 0.336 | 0.0051 | 1.488 |
| 4 | random | 0.338 | 0.0052 | 1.486 |
| 4 | learned | 0.338 | 0.0052 | 1.486 |

## Reproduce

```bash
conda run -n stq python -m experiments.spin_turboquant.run --model /home/elicer/.cache/huggingface/hub/models--NousResearch--Meta-Llama-3.1-8B/snapshots/1f47e50cdbe801ad8a5174156ec3a0655108fb9f --output-dir /home/elicer/SpinTurboQuant/turboquant_plus/experiments/spin_turboquant/results/main
```

Raw evidence: `training.csv`, `offline_metrics.csv`, `head_metrics.csv`, `perplexity.csv`,
`retrieval.csv`, `retrieval_details.csv`, `latency.csv`, saved rotations, captured
activations, and `run_config.json` in this directory.
