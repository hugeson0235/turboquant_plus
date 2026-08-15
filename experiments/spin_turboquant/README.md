# SpinTurboQuant experiment

This directory implements the experiment in `../../../SpinTurboQuant.md`.
It uses the random Haar rotation and Lloyd-Max codebooks from the local
`turboquant_plus/turboquant` package, then learns only one Cayley-parameterized
orthogonal matrix per Llama layer/KV head. LLM parameters and codebooks remain
frozen.

The completed initial-run findings are recorded in
`../../docs/experiment-spin-turboquant.md`.

## Environment

The experiment was built in a dedicated Conda environment:

```bash
conda env create -f experiments/spin_turboquant/environment.yml
conda activate stq
```

If the `stq` environment already exists, install the pinned packages from the
`pip` section instead. Run commands from the `turboquant_plus` repository root
so Python imports this checkout's `turboquant` implementation.

## Full run

```bash
conda run -n stq python -m experiments.spin_turboquant.run \
  --model /home/elicer/.cache/huggingface/hub/models--NousResearch--Meta-Llama-3.1-8B/snapshots/1f47e50cdbe801ad8a5174156ec3a0655108fb9f \
  --output-dir experiments/spin_turboquant/results/main
```

The default controlled run uses all 32 layers and 8 KV heads, 2/3/4-bit fixed
TurboQuant codebooks, three paired random initializations, 4,096 calibration
tokens, two held-out domains, and four attention sequence lengths. It also
measures end-to-end perplexity, synthetic long-context retrieval, dense codec
latency, and FP32 rotation storage.

The stages are resumable:

```bash
conda run -n stq python -m experiments.spin_turboquant.run \
  --model /path/to/Meta-Llama-3.1-8B \
  --output-dir experiments/spin_turboquant/results/main \
  --stage train
```

Valid stages are `capture`, `train`, `offline`, `end-to-end`, and `report`.
Existing artifacts are reused unless `--force` is supplied.

## Outputs

- `training.csv`: calibration objective and orthogonality checks
- `offline_metrics.csv`: reconstruction and attention distortion measurements
- `perplexity.csv`: end-to-end held-out language-model loss
- `retrieval.csv` and `retrieval_details.csv`: long-context retrieval results
- `latency.csv`: dense codec timing and rotation storage
- `report.md` and `summary.json`: paired aggregation and strict verdict
- `activations/` and `rotations/`: raw resumable evidence

Run the focused tests with:

```bash
conda run -n stq python -m pytest tests/test_spin_turboquant.py -q
```

## LongBench-E subset downstream experiment

`../../../LongBenchSubset.md` is implemented by the resumable LongBench runner.
It pins the official LongBench code and dataset revisions, writes one immutable
`subset_manifest.json` using sampling seed `20020305`, selects 5 examples from
each of the 3 length intervals for every task (195 examples per condition),
performs one smoke example per task, and then runs all 22 conditions in the
specified paired order before building the paired bootstrap report.

The rotation directory must have been produced from the exact same model
snapshot. For the Instruct checkpoint used by LongBench, first create its
rotation artifacts with the full command above and a separate output directory,
then run:

```bash
conda activate stq
python -m experiments.spin_turboquant.longbench \
  --stage orchestrate \
  --model /path/to/Meta-Llama-3.1-8B-Instruct/snapshot \
  --rotation-dir experiments/spin_turboquant/results/instruct \
  --longbench-repo ../LongBench_official \
  --data-dir ../LongBench_data/5e628be450b7e67fb7ae6e201bd6d8f7056f7672/data \
  --output-dir experiments/spin_turboquant/results/longbench_subset
```

Each completed condition contains `run_config.json`, `predictions.jsonl`,
`scores.csv`, `task_summary.csv`, `category_summary.csv`,
`length_summary.csv`, `paired_comparison.csv`, `system_metrics.csv`, and
`report.md`. The study root contains combined copies of the required artifacts,
the final tables, plots, paired rows, bootstrap intervals, and verdicts.
Full runs use deterministic contiguous batches only for generation-heavy or
fixed-cap tasks; variable-length QA remains batch size 1. Batch membership is
fixed before resume filtering, and implementation, dataset, codebook, model,
and rotation hashes are recorded in the run configuration.

## Analytic PCA LongBench pilot

`../../../LongBenchAnalytic.md` is implemented separately so the completed
22-condition study remains immutable. The analytic runner constructs exactly
three K2/V16 row-vector rotations (`Wk-PCA+H`, `Activation-K-PCA+H`, and
`Attention-aware-Q-PCA+H`), using deterministic float64 eigendecomposition and
the same normalized Sylvester `H_128` for every method. Activation and
attention targets share an immutable WikiText-2 train manifest containing 8
unique 512-token document spans; held-out diagnostics use an independently
manifested 4,096 tokens from WikiText-2 validation.

```bash
conda activate stq
python -m experiments.spin_turboquant.longbench_analytic \
  --stage orchestrate \
  --model /path/to/Meta-Llama-3.1-8B-Instruct/snapshot \
  --longbench-repo ../LongBench_official \
  --data-dir ../LongBench_data/5e628be450b7e67fb7ae6e201bd6d8f7056f7672/data \
  --reference-subset-manifest experiments/spin_turboquant/results/longbench_subset/subset_manifest.json \
  --reference-study-dir experiments/spin_turboquant/results/longbench_subset \
  --output-dir experiments/spin_turboquant/results/longbench_analytic
```

Stages are `validate`, `rotations`, `diagnostics`, `smoke`, `full`, `report`,
and `orchestrate`. Moment construction and held-out diagnostics checkpoint after
every 512-token sequence; LongBench predictions checkpoint after every batch.
The root output includes the calibration/subset manifests, three rotation
artifacts, covariance/eigenvalue archive, per-head rotation metadata,
calibration stability, held-out diagnostics, 585 matched predictions, all
three pairwise 10,000-sample bootstrap comparisons, and `report.md`.

This path is a quality emulation: Keys are reconstructed before BF16 cache
storage and Values remain BF16. Reported packed-cache size is theoretical; GPU
memory and latency are not compressed-cache measurements.
