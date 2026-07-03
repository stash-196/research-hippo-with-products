# Lag Operator HiPPO

Supplementary code for the paper "Lag Operator SSMs: A Novel Framework for Structured State Space Modeling" (AISTATS 2026 submission). https://openreview.net/pdf?id=3woUBLisIm

This package reproduces numerical experiments and figures (e.g., matrix equivalences in Tables 2–4, signal reconstructions in Figure 4).

Code adapted from the S4 repository: https://github.com/state-spaces/s4 (Gu et al., 2021, MIT License). See hippo_by_original_authors.py for original HiPPO implementations.

## Installation
```
pip install -r requirements.txt
pip install -e .
```
## Usage
**To reproduce the paper's numerical experiments and figures, run the demo script:**
```
python src/lag_op_hippo/demo.py
```

This will:
- Compute matrix differences (e.g., A_infty vs. A_gen).
- Visualize basis shifts.
- Reconstruct Lorenz63 signals and compute MSE (e.g., ~5.76e-07).

## Development Workflow (Frequency-Preserving Experiments)

The frequency-preserving code path is now split for iterative development:

- Core definitions only (safe to import):
	- `src/lag_op_hippo/lag_hippo_freq.py`
- Diagnostics helpers (projection/reconstruction + orthonormality):
	- `src/lag_op_hippo/freq_diagnostics.py`
- Interactive VS Code cell workflow (`# %%`):
	- `src/lag_op_hippo/freq_experiment_dev.py`
- Reproducible benchmark runner:
	- `src/lag_op_hippo/benchmark_freq_models.py`

Run the benchmark script with:
```
python src/lag_op_hippo/benchmark_freq_models.py
```


## Dependencies
Python 3.12+. See requirements.txt for pinned versions.

## License
MIT (see LICENSE file).