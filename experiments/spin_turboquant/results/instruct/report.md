# SpinTurboQuant experiment report

Generated: 2026-08-14T03:04:29.521566+00:00

## Verdict

**Distortion hypothesis supported; end-to-end perplexity evidence is mixed.** The strict automated criterion requires learned rotations to beat
their paired random initialization on held-out normalized-key MSE for every seed
in both domains and on every paired attention-KL comparison across seeds,
domains, and sequence lengths. This is an initial controlled experiment, not a
claim of broad model-family generality.

## Fixed setup

- Model: `/home/elicer/.cache/huggingface/hub/models--NousResearch--Meta-Llama-3.1-8B-Instruct/snapshots/d10aef7999a2b5ba950ab3974312feeedbfe0b77` (32 layers,
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
had maximum key error 1.34e-05 and
maximum attention-logit error
2.29e-05
(MSE 1.35e-12). This verifies the
pre-RoPE restore-then-RoPE path independently of quantization.

## Calibration objective

| bits | initial MSE | learned MSE | relative reduction | max abs(R^T R-I) |
|---|---|---|---|---|
| 2 | 0.000905783 | 0.00053073 | 41.41% | 4.87e-08 |
| 3 | 0.000265818 | 0.000209473 | 21.20% | 4.89e-08 |
| 4 | 7.31786e-05 | 6.13512e-05 | 16.16% | 4.77e-08 |

The explicit calibration-to-held-out gap and the fraction of layer/head pairs
that improve are:

| bits | calibration reduction | WikiText reduction | WikiText gap | TinyStories reduction | TinyStories gap | head-pair wins |
|---|---|---|---|---|---|---|
| 2 | 41.41% | 30.52% | -10.88 pp | 30.67% | -10.73 pp | 100.0% |
| 3 | 21.20% | 14.32% | -6.87 pp | 13.88% | -7.31 pp | 100.0% |
| 4 | 16.16% | 11.83% | -4.33 pp | 11.68% | -4.48 pp | 100.0% |

## Held-out paired results

Positive reductions mean learned is better than its exact paired random start.
Win rates are computed across seeds for reconstruction and across seeds, domains,
and sequence lengths for attention.

| bits | domain | identity MSE | random MSE | learned MSE | paired reduction | paired wins |
|---|---|---|---|---|---|---|
| 2 | wikitext | 0.00191711 | 0.000904878 | 0.000628661 | 30.52% | 100% |
| 2 | tinystories | 0.0019883 | 0.000906027 | 0.00062811 | 30.67% | 100% |
| 3 | wikitext | 0.000955846 | 0.000265433 | 0.000227412 | 14.32% | 100% |
| 3 | tinystories | 0.00100652 | 0.000265478 | 0.000228619 | 13.88% | 100% |
| 4 | wikitext | 0.000527019 | 7.30685e-05 | 6.44226e-05 | 11.83% | 100% |
| 4 | tinystories | 0.000562511 | 7.29653e-05 | 6.44429e-05 | 11.68% | 100% |

Attention probability KL divergence, pooled across both held-out domains and all
configured sequence lengths:

| bits | identity KL | random KL | learned KL | paired reduction | paired wins |
|---|---|---|---|---|---|
| 2 | 1.63873 | 0.134621 | 0.0874978 | 34.73% | 100% |
| 3 | 1.01628 | 0.0325007 | 0.0274926 | 15.46% | 100% |
| 4 | 0.642527 | 0.0084344 | 0.00739281 | 12.45% | 100% |

## End-to-end results

The uncompressed model has perplexity 5.2089 and retrieval accuracy
1.000. Perplexity was measured on 1024 held-out tokens;
retrieval used 6
multi-key prompts at lengths 512, 1024, 2048.
The target margin is the target-token logit minus the strongest distractor logit.

| bits | identity PPL | random PPL | learned PPL | paired PPL reduction | PPL wins | identity retrieval | random retrieval | learned retrieval | random margin | learned margin |
|---|---|---|---|---|---|---|---|---|---|---|
| 2 | 330.6539 | 6.6007 | 5.9412 | 9.92% | 100% | 0.333 | 0.944 | 0.833 | 6.102 | 5.667 |
| 3 | 84.0535 | 5.4511 | 5.3953 | 1.02% | 100% | 1.000 | 0.944 | 1.000 | 10.090 | 10.142 |
| 4 | 14.6627 | 5.2650 | 5.2368 | 0.53% | 67% | 1.000 | 1.000 | 1.000 | 10.799 | 10.535 |

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
| 2 | identity | 0.337 | 0.0051 | 1.442 |
| 2 | random | 0.339 | 0.0052 | 1.444 |
| 2 | learned | 0.339 | 0.0052 | 1.445 |
| 3 | identity | 0.338 | 0.0052 | 1.459 |
| 3 | random | 0.338 | 0.0052 | 1.462 |
| 3 | learned | 0.339 | 0.0052 | 1.462 |
| 4 | identity | 0.336 | 0.0051 | 1.482 |
| 4 | random | 0.338 | 0.0052 | 1.486 |
| 4 | learned | 0.340 | 0.0052 | 1.487 |

## Reproduce

```bash
conda run -n stq python -m experiments.spin_turboquant.run --model /home/elicer/.cache/huggingface/hub/models--NousResearch--Meta-Llama-3.1-8B-Instruct/snapshots/d10aef7999a2b5ba950ab3974312feeedbfe0b77 --output-dir /home/elicer/SpinTurboQuant/turboquant_plus/experiments/spin_turboquant/results/instruct
```

Raw evidence: `training.csv`, `offline_metrics.csv`, `head_metrics.csv`, `perplexity.csv`,
`retrieval.csv`, `retrieval_details.csv`, `latency.csv`, saved rotations, captured
activations, and `run_config.json` in this directory.
