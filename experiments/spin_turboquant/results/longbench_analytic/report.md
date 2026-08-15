# Analytic head-wise PCA rotations: LongBench-E pilot

Generated: 2026-08-15T07:01:37.496882+00:00

## Overall results

| Method | LongBench macro | Normalized Key MSE | Logit MSE | Attention KL |
|---|---|---|---|---|
| Wk-PCA+H | 43.953 | 0.00081237 | 0.81464127 | 0.12704099 |
| Activation-K-PCA+H | 42.971 | 0.00076311 | 0.64568983 | 0.11509695 |
| Attention-aware-Q-PCA+H | 43.528 | 0.00088481 | 1.11576609 | 0.12032487 |

## Paired downstream comparisons

| First minus second | Macro difference | 95% paired CI | Positive tasks | Win/tie/loss examples |
|---|---|---|---|---|
| activation_k_pca_h_minus_wk_pca_h | -0.982 | [-5.694, 3.571] | 5 | 55/77/63 |
| attention_q_pca_h_minus_wk_pca_h | -0.425 | [-4.938, 4.112] | 7 | 60/82/53 |
| attention_q_pca_h_minus_activation_k_pca_h | 0.557 | [-4.351, 5.420] | 7 | 59/75/61 |

Confidence intervals use 10,000 paired, task-stratified resamples of the same 195 identities.

## Hypotheses

| Hypothesis | Outcome | Evidence |
|---|---|---|
| H1 | Supported | Activation normalized Key MSE 0.00076310773 vs Wk 0.00081237415 |
| H2 | Not supported | Attention-aware is not lowest on both held-out logit MSE and KL |
| H3 | Not supported | Attention-aware LongBench macro 43.528222; best macro 43.953297 |
| H4 | Supported | Both calibration-method vs Wk pilot confidence intervals include zero |

## Calibration stability

| Method | Covariance relative Frobenius | Top-16 | Top-32 | Top-64 | Held-out half2-half1 Key MSE |
|---|---|---|---|---|---|
| activation_k_pca_h | 0.1333 | 0.7309 | 0.7700 | 0.8635 | -0.00000022 |
| attention_q_pca_h | 0.1204 | 0.8454 | 0.8507 | 0.8751 | 0.00000160 |

No numeric instability threshold was specified, so the 4,096-token run is preserved and no unrequested 16,384-token robustness expansion was launched.

## Matched prior references

- Compatibility checks passed: `True`.
- These are reference lines only; no FP16, Identity, Random, or Learned condition was rerun.
- fp16_K16_V16: 57.085
- identity_K2_V16: 7.228
- random_K2_V16_three_seed_mean: 42.044
- learned_K2_V16_three_seed_mean: 48.013

## Protocol and guardrails

- Model revision: `d10aef7999a2b5ba950ab3974312feeedbfe0b77`.
- LongBench commit: `2e00731f8d0bff23dc4325161044d0ed8af94c1e`; dataset revision: `5e628be450b7e67fb7ae6e201bd6d8f7056f7672`.
- Calibration: WikiText-2 train, 8 x 512 tokens, seed 20020305; held-out: WikiText-2 validation, 8 x 512 tokens.
- Every method uses descending, sign-fixed float64 eigendecomposition and the same normalized Sylvester H_128 without random signs or permutations.
- Keys alone are quantized to K2 before RoPE; Values remain BF16 for every condition.
- This is quality emulation: reconstructed Keys are stored in BF16. Theoretical packed bytes are reported, but measured memory/latency is not a compressed-cache claim.
- The 195-example LongBench-E subset is a method-selection pilot, not a final benchmark result.
- Offline distortion and downstream association uses only three methods and is exploratory.
- TinyStories validation was optional and was not run.

See `heldout_diagnostics.csv`, `rotation_metadata.json`, `paired_comparison.csv`, and `offline_downstream_relationship.csv` for raw evidence.
