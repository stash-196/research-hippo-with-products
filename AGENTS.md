# AGENTS.md

## Cursor Cloud specific instructions

This is a Python 3.12 research package (`lag-op-hippo`) that reproduces the numerical
experiments/figures from the "Lag Operator SSMs" paper. It is a library + scripts, not a
server or web app; "running the app" means executing the demo/benchmark scripts, which
print numerical results and render matplotlib figures.

### Environment

- Dependencies are installed into a project virtualenv at `.venv` (gitignored) by the
  startup update script. The system Python is externally managed (PEP 668) and the
  editable `-e .` install fails against it, so always use the venv rather than the system
  interpreter.
- Activate with `. .venv/bin/activate`, or call tools directly via `.venv/bin/<tool>`
  (e.g. `.venv/bin/pytest`, `.venv/bin/python`). There is no `python` shim, only `python3`.
- Standard install/run commands are in `README.md`.

### Running scripts (headless)

- The VM has no display. The plotting helpers in `src/lag_op_hippo/visualizers.py` default
  to `plt.show()`, which is a harmless no-op under a non-interactive backend. Run scripts
  with `MPLBACKEND=Agg` to avoid an interactive-backend warning, e.g.
  `MPLBACKEND=Agg .venv/bin/python src/lag_op_hippo/demo.py`.
- To save figures instead of showing them, the plotting functions accept
  `save_path=..., show=False, save=True`.
- Entry points: `src/lag_op_hippo/demo.py` (matrix-equivalence tables + Lorenz63
  reconstruction) and `src/lag_op_hippo/benchmark_freq_models.py`. The frequency-preserving
  path is under active development and its benchmark may report `nan` — that is a known
  research-code state, not an environment problem.

### Tests

- `MPLBACKEND=Agg .venv/bin/python -m pytest` runs the suite in `tests/` (15 tests).

### Lint / type-check

- `flake8`, `black`, `isort`, and `mypy` are installed (see `dev-requirements.txt`) but the
  repo ships no lint config and no pre-commit hooks, so the tools run with strict defaults
  and report many pre-existing violations in existing code. Several modules are notebook
  style (`# %%` cells) and are not valid standalone modules, so `black`/`mypy` can raise
  parse/syntax errors on them (e.g. `freq_experiment_dev.py`, `prediction_task.py`). Do not
  treat these pre-existing findings as regressions.
