"""Dump key library quantities to an .npz so pre-fix (tag pre-libfix-2026-11-19) and
post-fix library versions can be compared.  Uses only APIs present in both versions.
Usage: PYTHONPATH=<checkout>/src python libfix_dump.py OUT.npz"""
import contextlib, io, sys
import numpy as np
import torch
from scipy import signal
import lag_op_hippo
from lag_op_hippo.lag_hippo import (LagHiPPO, phi, phi_deriv, sigma, sigma_deriv, sigma_inv, sigma_inv_deriv,
                                    compute_A_infty, get_gauss_quadrature, vectorized_basis)
from lag_op_hippo.lag_hippo_freq import make_frequency_preserving_basis, make_frequency_preserving_model
from lag_op_hippo.lag_hippo_translated import make_legt_model, make_fout_model, make_legt_basis, make_fout_basis
from lag_op_hippo.hippo_by_original_authors import HiPPO, transition

N, DT, T = 16, 1e-3, 3.0
s = np.arange(0.0, T, DT); u = np.cos(2 * np.pi * 3.0 * s)
out = {"lib_file": np.array(lag_op_hippo.__file__)}

def q(f):
    with contextlib.redirect_stdout(io.StringIO()):
        return f()

def scan(dA, dB, u):
    c = np.zeros(dA.shape[0])
    with np.errstate(all="ignore"):
        for x in u:
            c = dA @ c + dB * x
    return c

def put(tag, m):
    out[tag + "_A"] = np.asarray(m.A, float); out[tag + "_B"] = np.asarray(m.B, float)
    out[tag + "_dA"] = m.dA.numpy().astype(float); out[tag + "_dB"] = m.dB.numpy().astype(float)
    E = np.asarray(m.eval_matrix, float); out[tag + "_E"] = E
    with torch.no_grad():
        ut = torch.tensor(u, dtype=torch.float)
        out[tag + "_rec32"] = m.reconstruct(m(ut)[-1]).numpy().astype(float)
    out[tag + "_rec64"] = np.flip(E @ scan(out[tag + "_dA"], out[tag + "_dB"], u))
    out[tag + "_lsq"] = np.flip(E @ np.linalg.lstsq(E, np.flip(u), rcond=None)[0])

legs = lambda **kw: LagHiPPO(N=N, phi_func=phi, phi_deriv_func=phi_deriv, sigma_func=sigma, sigma_deriv_func=sigma_deriv,
                             sigma_inv_func=sigma_inv, sigma_inv_deriv_func=sigma_inv_deriv, dt=DT, T=T, **kw)
put("legs_gen_delta", q(lambda: legs()))
put("legs_infty_zoh", q(lambda: legs(derive_A_from="infty", input_hold="zoh")))
put("legs_gen_zoh", q(lambda: legs(input_hold="zoh")))
put("legt", q(lambda: make_legt_model(N=N, dt=DT, T=T, window_length=1.0)))
put("fout", q(lambda: make_fout_model(N=N, dt=DT, T=T, window_length=1.0)))
put("foup_gen_zoh", q(lambda: make_frequency_preserving_model(N=N, dt=DT, T=T, alpha=0.2, window_length=1.0, eps=1e-8,
                                                              derive_A_from="gen", input_hold="zoh", quad_order=200 * N)))
put("foup_infty_zoh", q(lambda: make_frequency_preserving_model(N=N, dt=DT, T=T, alpha=0.2, window_length=1.0, eps=1e-8,
                                                                derive_A_from="infty", input_hold="zoh", quad_order=200 * N)))
for meth in ("legs", "legt", "fourier"):
    hp = q(lambda: HiPPO(method=meth, N=N, dt=DT, T=T))
    A, B = transition(meth, N)
    out[f"gu_{meth}_A"], out[f"gu_{meth}_B"] = A, B[:, 0]
    dA, dB, *_ = signal.cont2discrete((A, B, np.ones((1, N)), np.zeros((1,))), dt=DT, method="bilinear")
    out[f"gu_{meth}_dA"], out[f"gu_{meth}_dB"] = dA, dB[:, 0]
    out[f"gu_{meth}_E"] = np.asarray(hp.eval_matrix, float)
    out[f"gu_{meth}_rec64"] = np.flip(out[f"gu_{meth}_E"] @ scan(dA, dB[:, 0], u))
# warps (01.x geometry) and FouP Gram
fp = make_frequency_preserving_basis(alpha=0.2, window_length=1.0, eps=1e-8)
tt = np.linspace(0, 6, 3001); zz = np.linspace(1e-6, 1, 3001)
for nm, b in (("foup", fp), ("legt", make_legt_basis(1.0)), ("fout", make_fout_basis(1.0))):
    out[f"warp_{nm}_sigma"] = b.sigma(6.0, 6.0 - tt); out[f"warp_{nm}_sigma_deriv"] = b.sigma_deriv(6.0, 6.0 - tt)
    out[f"warp_{nm}_sigma_inv"] = b.sigma_inv(6.0, zz); out[f"warp_{nm}_sigma_inv_deriv"] = b.sigma_inv_deriv(6.0, zz)
out["warp_legs_sigma"] = sigma(6.0, 6.0 - tt); out["warp_legs_sigma_inv"] = sigma_inv(6.0, zz)
z, w = get_gauss_quadrature(20000, 0, 1)
P = vectorized_basis(fp.phi, np.arange(N), z)
out["foup_gram_full"] = np.einsum("i,ik,il->kl", w, P, P)
out["foup_A_infty"] = compute_A_infty(N, fp.phi, fp.sigma, fp.sigma_inv, dt=DT, quad_order=200 * N)
np.savez(sys.argv[1], **out)
print("saved", sys.argv[1], out["lib_file"])
