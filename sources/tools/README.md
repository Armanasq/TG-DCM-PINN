# Result sensitivity verification

From the repository root, with Python 3.10+ and no extra dependencies:

    python sources/tools/verify_result_sensitivity.py \
      --csv results/full_evaluation/tables/per_sequence.csv \
      --reps 20000 \
      --seed 7

**Outputs:** original 137-file mean/paired-bootstrap CI, 119-file set after omitting duplicated Caruso original variants, per-pair model/RIANN/VQF error differences, DIODEM 86-motion and 11-experiment grouped CIs, and the supplied CSV's SHA256.

**Interpretation:** cluster resampling of original independent motion/experiment identifiers is conditional on the frozen checkpoint and its preparation pipeline. Removing an alternate Caruso target file changes the *estimand*, not the trained network. The resulting bootstrap intervals are Monte Carlo approximations, not guaranteed byte-identical to the original NumPy RNG result. No hypothesis about why the paired Caruso scores differ can be established without comparing raw HDF5 arrays.

**Verification status:** the script was run with a synthetic fixture; the real 559-row CSV was independently analyzed in-memory through GitHub and showed 18 Caruso pairs, 422 DIODEM entries, 86 motion groups and 11 experiments. The external source/weights were not loaded or executed in this environment. [Audit narrative](../review/09_second_pass_verification.md).
