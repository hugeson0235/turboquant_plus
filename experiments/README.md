# Channel-clustered TurboQuant experiment

`run_channel_clustered_llama.py` implements the experiment in the workspace's
`ClusterTurboQuant.md`. It learns per-layer/per-KV-head channel groups from
WikiText-2 train activations and evaluates disjoint validation/test tokens from
Llama-3.1-8B. The downstream hook replaces `k_proj` outputs before RoPE.

From this repository, with the requested environment:

```bash
conda activate ctq
python experiments/run_channel_clustered_llama.py
```

The defaults compare `K=1,2,4` at 3 bits with three fixed rotation seeds. The
output directory contains per-seed CSVs, mean/std summaries, learned
partitions, the full JSON record, environment metadata, and a Markdown report.

For a rate/quality Pareto sweep:

```bash
python experiments/run_channel_clustered_llama.py \
  --bit-widths 2 3 4 \
  --downstream-bit-widths 3
```

The effective rate is `bit_width + K * norm_bits / head_dim`; static partitions,
rotations, and codebooks are reported separately. This distinction is required
because equal scalar width does not mean equal total storage when `K` changes.
