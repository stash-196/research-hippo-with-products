"""Shared setup for the window checks. Output dir comes from WINDOW_CHECKS_OUT (set by run_window_checks)."""
import os, sys, json
OUT = os.environ.get("WINDOW_CHECKS_OUT")
if not OUT:
    raise RuntimeError("set WINDOW_CHECKS_OUT (use python -m lag_op_hippo.window_checks.run_window_checks)")
if os.path.basename(os.path.normpath(OUT)) == "latest" or "/runs/" in OUT:
    raise RuntimeError("refusing to write into outputs/freq_dev/latest or runs/")
os.makedirs(OUT, exist_ok=True)
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({"mathtext.fontset": "cm", "font.family": "serif", "font.size": 10,
                     "axes.grid": True, "grid.alpha": 0.3})
import contextlib, io

@contextlib.contextmanager
def quiet():
    with contextlib.redirect_stdout(io.StringIO()):
        yield

from lag_op_hippo.lag_hippo import compute_A_infty, compute_B_zoh, get_gauss_quadrature, vectorized_basis
from lag_op_hippo.lag_hippo_freq import make_frequency_preserving_basis, make_frequency_preserving_model
from lag_op_hippo.lag_hippo_translated import (make_fout_basis, make_fout_model, make_legt_model,
                                               make_legt_basis, fout_input_vector)
from lag_op_hippo.freq_A_discrepancy_checks import (compute_A_gen_for_basis, build_A_infty_map_catalog,
                                                    tilt_continuous_A, _safe_pinv)
from lag_op_hippo.hippo_by_original_authors import HiPPO, transition
from lag_op_hippo.window_checks.basis_utils import *

def signals(T, dt=DT):
    s = np.arange(0.0, T, dt)
    return s, {"cos": np.cos(2 * np.pi * 3.0 * s),            # repo 02.01/02.02 test input (L-periodic)
               "chirp": np.sin(2 * np.pi * (0.5 * s + 0.35 * s ** 2))}  # non-periodic, f(s)=0.5+0.7 s Hz

def save(fig, name):
    p = os.path.join(OUT, name)
    fig.savefig(p, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print("saved", p)

def dump(obj, name):
    def conv(o):
        if isinstance(o, (np.floating,)): return float(o)
        if isinstance(o, (np.integer,)): return int(o)
        if isinstance(o, np.ndarray): return o.tolist()
        if isinstance(o, complex): return [o.real, o.imag]
        return str(o)
    with open(os.path.join(OUT, name), "w") as f:
        json.dump(obj, f, indent=1, default=conv)
