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
