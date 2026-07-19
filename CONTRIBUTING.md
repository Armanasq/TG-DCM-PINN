# Contributing

## Scope

Contributions must preserve the released sensor contract:

- acceleration, angular rate, and timestep input
- scalar-first `wxyz` body-to-world quaternion output
- tilt as the primary six-axis attitude metric

Changes to frame convention, quaternion order, data splits, gate definitions, or
aggregation rules require explicit documentation and regression tests.

## Development environment

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[all,dev]"
```

## Required checks

```bash
ruff check .
ruff format --check .
pytest -q
python -m build
twine check dist/*
```

Tests must cover numerical boundaries, invalid data, empty masks, frame
conventions, and serialization contracts affected by the change. Evaluation
claims require saved per-sequence results and the exact aggregation command.

## Pull requests

Keep changes focused. State the modified scientific assumption, implementation
surface, dataset scope, and validation evidence. Do not report performance from
pooled means alone when dataset composition is imbalanced.
