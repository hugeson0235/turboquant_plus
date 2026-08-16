# Learned rotation training-length sweep

Generated: 2026-08-16T14:13:12.469043+00:00

## Result

The primary selection used only the seed-mean terminal WikiText-2 validation normalized-key MSE. TinyStories and attention metrics were evaluated only after selection, and LongBench-E was gated by the predeclared 0.5% improvement rule.

## Selected horizons

| bits | argmin H | 0.1%-tolerant H* | adopted H | WikiText gain vs 80 | boundary | TinyStories change | domain warning |
|---|---|---|---|---|---|---|---|
| 2 | 1000 | 1000 | 1000 | 16.321% | yes | -15.876% | no |
| 3 | 1000 | 1000 | 1000 | 6.937% | yes | -6.717% | no |
| 4 | 1000 | 900 | 900 | 1.796% | yes | -1.629% | no |

A negative TinyStories change means lower MSE than the 80-step reference.

## Terminal sweep metrics

| bits | H | calibration MSE | validation MSE | gap | optimizer sec |
|---|---|---|---|---|---|
| 2 | 100 | 0.000511375642 | 0.000602647144 | 9.12715015e-05 | 0.867 |
| 2 | 200 | 0.000458111625 | 0.000562058872 | 0.000103947247 | 1.645 |
| 2 | 300 | 0.000435006328 | 0.000544152179 | 0.000109145851 | 2.470 |
| 2 | 400 | 0.000422232881 | 0.000534181533 | 0.000111948651 | 3.294 |
| 2 | 500 | 0.000413952131 | 0.00052813077 | 0.000114178639 | 4.118 |
| 2 | 600 | 0.00040848848 | 0.000524185807 | 0.000115697327 | 4.945 |
| 2 | 700 | 0.000404257746 | 0.00052134236 | 0.000117084614 | 5.763 |
| 2 | 800 | 0.000400921689 | 0.00051909968 | 0.000118177991 | 6.587 |
| 2 | 900 | 0.000398293294 | 0.000517400378 | 0.000119107085 | 7.410 |
| 2 | 1000 | 0.000395953555 | 0.000515792393 | 0.000119838838 | 8.236 |
| 3 | 100 | 0.000206094772 | 0.000225554582 | 1.94598105e-05 | 0.826 |
| 3 | 200 | 0.000194834027 | 0.000222072693 | 2.72386657e-05 | 1.650 |
| 3 | 300 | 0.000188063863 | 0.000219498875 | 3.1435012e-05 | 2.475 |
| 3 | 400 | 0.000183086913 | 0.000217356647 | 3.42697339e-05 | 3.300 |
| 3 | 500 | 0.000179469808 | 0.000215635956 | 3.6166148e-05 | 4.125 |
| 3 | 600 | 0.000176713415 | 0.000214375272 | 3.76618564e-05 | 4.954 |
| 3 | 700 | 0.000174551782 | 0.000213319091 | 3.87673093e-05 | 5.779 |
| 3 | 800 | 0.000172553123 | 0.000212155563 | 3.96024403e-05 | 6.610 |
| 3 | 900 | 0.000171102251 | 0.000211565712 | 4.04634609e-05 | 7.435 |
| 3 | 1000 | 0.000169635034 | 0.000210710788 | 4.10757545e-05 | 8.261 |
| 4 | 100 | 6.06914782e-05 | 6.40674248e-05 | 3.37594669e-06 | 0.830 |
| 4 | 200 | 5.84975939e-05 | 6.36673968e-05 | 5.16980294e-06 | 1.658 |
| 4 | 300 | 5.70756758e-05 | 6.34808126e-05 | 6.40513681e-06 | 2.488 |
| 4 | 400 | 5.60239644e-05 | 6.33568161e-05 | 7.33285167e-06 | 3.318 |
| 4 | 500 | 5.52105629e-05 | 6.32643731e-05 | 8.05381014e-06 | 4.142 |
| 4 | 600 | 5.45488419e-05 | 6.32084763e-05 | 8.65963439e-06 | 4.977 |
| 4 | 700 | 5.40074316e-05 | 6.31558011e-05 | 9.1483695e-06 | 5.800 |
| 4 | 800 | 5.35423583e-05 | 6.3104552e-05 | 9.56219369e-06 | 6.627 |
| 4 | 900 | 5.31711582e-05 | 6.30744591e-05 | 9.90330087e-06 | 7.458 |
| 4 | 1000 | 5.28207853e-05 | 6.3040024e-05 | 1.02192388e-05 | 8.289 |

## TinyStories post-selection sanity

| bits | condition | normalized MSE | original MSE | attention KL | logit MSE |
|---|---|---|---|---|---|
| 2 | random_initialization | 0.000905817616 | 0.565850891 | 0.175318378 | 1.41081402 |
| 2 | existing_80_step | 0.000618175455 | 0.386752273 | 0.112015743 | 0.573178054 |
| 2 | selected_horizon | 0.000520032906 | 0.325280355 | 0.0864168389 | 0.4061646 |
| 3 | random_initialization | 0.000265425468 | 0.165813727 | 0.0427810331 | 0.298193958 |
| 3 | existing_80_step | 0.000227695663 | 0.142735793 | 0.0362825275 | 0.217284695 |
| 3 | selected_horizon | 0.000212400233 | 0.133314943 | 0.0350910559 | 0.196535616 |
| 4 | random_initialization | 7.29849707e-05 | 0.0455803231 | 0.0111196606 | 0.0739858269 |
| 4 | existing_80_step | 6.42593943e-05 | 0.0402341233 | 0.00980946726 | 0.0595782697 |
| 4 | selected_horizon | 6.32124627e-05 | 0.0396656864 | 0.0095987674 | 0.0585449239 |

Attention metrics were not used to revise the selected horizons.

## LongBench-E decision

Status: skipped_by_explicit_flag. LongBench-E was skipped at the user's explicit request.
Partial artifacts were retained but were not used for a paired LongBench-E conclusion: smoke learned_K2_V16_s17=13/13; learned_K2_V16_s29=13/13, full learned_K2_V16_s17=195/195; learned_K2_V16_s29=60/195.

## Interpretation guardrails

- Every 100-1000 horizon is an independent zero-Cayley/Adam restart with the same bit/seed minibatch prefix and its own T_max=H cosine schedule.
- Primary comparisons are terminal checkpoints; equal intermediate step numbers across horizons do not share a learning rate.
- A minimum at 1000 steps is reported as a search-boundary result, not a confirmed global optimum.
- These offline reconstruction and attention measurements do not by themselves establish end-to-end perplexity or generation quality.

## Reproduce

```bash
conda activate stq
python -m experiments.spin_turboquant.training_length_sweep --stage orchestrate --model /home/elicer/.cache/huggingface/hub/models--NousResearch--Meta-Llama-3.1-8B-Instruct/snapshots/d10aef7999a2b5ba950ab3974312feeedbfe0b77 --reference-dir /home/elicer/SpinTurboQuant/turboquant_plus/experiments/spin_turboquant/results/instruct --output-dir /home/elicer/SpinTurboQuant/turboquant_plus/experiments/spin_turboquant/results/training_length_sweep --skip-longbench
```

Raw condition evidence is under `runs/`; aggregate CSVs, selected rotation artifacts, post-selection metrics, plots, and the completion audit are in this directory.
