# Research source corpus: six-axis inertial attitude estimation

**Cutoff:** 2026-11-13. **Inclusion:** research first disseminated or published in 2024–2026. **Anchor:** TG-DCM-PINN tilt from accelerometer, gyroscope and timestep. **Scope:** direct tilt estimators, geometric filters, dynamic-acceleration compensation, transferable learned inertial odometry, equivariance, uncertainty, calibration, and self-supervision.

This folder contains **independent analytical notes** grounded in primary publications and inspected code, not copies of the authors' papers. The full-text status and exactly inspected implementation paths are recorded in the manifests. It is a curated technical research set, **not an exhaustive systematic search of all papers published in 2024–2026**.

## Detailed 2024–2026 review

The [eight-chapter research monograph plus a separate second-pass verification](review/README.md) systematically covers architectures, designs, mathematics, challenges, goals, limitations, future research, and a full critical review. The corpus currently indexes **32 studies** and **12 inspected codebases**. These are reviewed reference materials, **not** independently replicated experiments.

## Structure

- review/README.md: structured eight-chapter literature review and navigation.
- review/09_second_pass_verification.md: protocol and source-based second-pass checks, revised confidence intervals and remaining artifacts.
- papers/index.json: deduplicated bibliographic and analytical metadata with evidence status; papers/<id>.md: per-paper research notes.
- repositories/index.json: pinned repository snapshots, inspected implementation files, declared/checked license.
- synthesis/research-synthesis.md: analytical comparison and explicit observability/novelty boundaries.
- synthesis/experimental-design.md: reproducibility, adversarial tests, ablations and significance protocol.
- implementation/target-audit.md: TG-DCM code-to-method review and defects requiring tests.
- implementation/external-code-audit.md: actual code inspection evidence, not README summaries.
- methodology.md: search, grading, provenance, access/redistribution limitations.
- tools/acquire.py: acquisition tool for arXiv HTML/PDF/LaTeX, open primary HTML, pinned Git checkouts, checksums and run log.
- cache/: optional local-only paper bytes and cloned checkouts; deliberately ignored by Git.

## Acquisition

Use Python 3.10+ from the repository root; a working internet connection and Git are necessary:

    python sources/tools/acquire.py --papers
    python sources/tools/acquire.py --repos
    python sources/tools/acquire.py --all

The command **has not been run successfully in the investigation environment**, which did not resolve external Git hosts. Papers and external repositories were examined through accessible web and GitHub integrations instead; do not misrepresent the unpopulated cache as downloaded or benchmarked. Acquisition intentionally does not bypass access controls. For copyrighted papers, preserve metadata and original analysis in Git; store permissible copies only in the ignored cache and respect publisher-specific terms.

## Interpretation

Primary task is yaw-invariant tilt RMSE in degrees. Velocity, position, trajectory ATE, angular-bias calibration and full-orientation studies have different observability, metrics or additional sensor inputs and must not be ranked as equivalent benchmark competitors. Primary papers can be read in full, partially read, or publisher-abstract-only: the manifests label this honestly. Reported experimental values are authors' findings unless explicitly marked re-executed.

## Updating

Add a stable ID in papers/index.json (arXiv ID preferred, else DOI-derived slug), a corresponding note, and any inspected Git code with a pinned revision. Do not add the same work under both arXiv and journal DOI as two independent papers. Add reproducibility artifacts only after running and logging exact commands, dependency versions, dataset hashes, label frame conventions and model checkpoint SHA256.

The [second-pass verification program](tools/verify_result_sensitivity.py) recomputes results/CSV-level deduplication and DIODEM clustering independently of the target model. It does **not** verify the unreleased HDF5 input contents or regenerate neural predictions.
