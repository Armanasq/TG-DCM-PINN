# Reproduction

Install:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[all]"
```

Evaluate the released checkpoint on all test datasets plus DIODEM:

```bash
python scripts/evaluate.py \
  --split test \
  --datasets OxIOD EuRoC-MAV Caruso-Sassari Caruso-Sassari_orig RepoIMU Myon diodem \
  --data-dir data \
  --diodem-root diodem \
  --sequence-workers 8 \
  --out-dir results/full_evaluation
```

Run a smoke check:

```bash
python scripts/evaluate.py --datasets diodem --max-seqs 1 --no-plot
```

Train the TG-DCM configuration:

```bash
python scripts/train.py --data-dir data --out-dir runs --run-name tg_dcm
```

The released checkpoint is `checkpoints/tg_dcm.pt`. The evaluator uses EMA
weights by default. Pass `--raw` to evaluate raw checkpoint weights.

Regenerate figures from the saved table:

```bash
python scripts/plot_results.py results/full_evaluation/tables/per_sequence.csv
```

`--sequence-workers 8` was used on a 32-core CPU and 16 GB GPU. Set it to `1`
when generating `--sequence-plots all`, because Matplotlib rendering is not
thread-safe. Aggregate figures include every trial without requiring separate
per-sequence image files.

## Released artifacts

The repository includes the selected weights-only checkpoint, configuration,
training summary, per-sequence evaluation table, derived aggregate tables, and
figures. Raw datasets and the optimizer-state training checkpoint are not part
of the public release.

Verify the released checkpoint before evaluation:

```bash
(cd checkpoints && sha256sum -c SHA256SUMS)
```
