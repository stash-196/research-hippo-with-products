"""Fused LegS+FouP basis on Lorenz-63 (Murata-sensei's idea), two variants, vs single bases.

Usage (repo root):  python -m lag_op_hippo.window_checks.exp_e_lorenz_fused_basis
Writes ONLY into a NEW folder outputs/freq_dev/lorenz_fused_basis_<YYYYmmdd_HHMMSS>/.

Notation (lag tau >= 0, 0 = now; measure omega(tau) = dz/dtau; c_k(t) = int u(t-tau) Psi_k(tau) omega(tau) dtau):
  LegS: Psi=phi_k(e^{-tau}), omega_L = e^{-tau};   FouP: Psi=psi_k(-tau) (L-periodic), omega_F = h_n/L piecewise.
  (A) block-diagonal: states c_L (16), c_F (16) with the library's own exact recurrences.  Readout: the stored moments are
      <u,phi_L>_{omega_L}=c_L and <u,phi_F>_{omega_F}=c_F, so  u_hat = Phi a  with the moment matrix  M a = c,
      M=[[<phi_L,phi_L>_{wL}, <phi_L,phi_F>_{wL}],[<phi_F,phi_L>_{wF}, <phi_F,phi_F>_{wF}]], a = pinv(M) c.
  (B) orthogonalised union under a common measure omega_c:  G=<Phi,Phi>_{omega_c}, Psi = Phi V_r Lambda_r^{-1/2}
      (canonical orthogonalisation; = Lowdin up to a rotation inside the retained span), near-null eigenvalues dropped.
      Exact generator of "shift the reconstruction + feed the new input" (derived in the note):
        dc/dt = -D^T c + omega(0) Psi(0) (u(t) - Psi(0)^T c),   D_jk = <Psi_j', Psi_k>_omega
        i.e.  A = A_gen - kappa omega(0) Psi(0)Psi(0)^T,  B = kappa omega(0) Psi(0),  A_gen = -D^T,  kappa = 1.
      (library compute_A_gen gives the same A_gen = -<Psi, Psi'>^T.)  Non-closure (Psi' not in span Psi) appears as the
      gap between projection and recurrence and is measured by ||(I-P)Psi'||_omega.
"""
import contextlib, datetime, io, json, os, subprocess, sys, time
import numpy as np
from scipy import signal as sps
from scipy.integrate import solve_ivp
from scipy.special import roots_legendre
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker

from lag_op_hippo.lag_hippo import (LagHiPPO, phi, phi_deriv, sigma, sigma_deriv, sigma_inv, sigma_inv_deriv,
                                    vectorized_basis)
from lag_op_hippo.lag_hippo_freq import make_frequency_preserving_model, make_frequency_preserving_basis

plt.rcParams.update({"mathtext.fontset": "cm", "font.family": "serif", "font.size": 10,
                     "axes.grid": True, "grid.alpha": 0.3})
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
DT, T_MEM, ALPHA, WL, EPS = 1e-3, 3.0, 0.2, 1.0, 1e-8
N1 = 16
TF = [3.0, 4.5, 6.0]
K_TOT = int(round(max(TF) / DT))
TU = 3.0
NTRAJ = 8
NLAG = int(round(T_MEM / DT))
WIN = NLAG // 3
BIN = 50
COMP = {"x": 0, "z": 2}
TAU_G = 14.0                      # Gram / closure integrals run over tau in [0, 14]
RCOND = 1e-8                      # eigenvalue / singular-value cut for near-null directions

ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
OUT = os.path.join(REPO, "outputs", "freq_dev", f"lorenz_fused_basis_{ts}")
assert not os.path.exists(OUT)
os.makedirs(OUT)
T0 = time.time()


def log(*a):
    print(f"[{time.time() - T0:6.1f}s]", *a, flush=True)


def save(fig, name):
    fig.savefig(os.path.join(OUT, name), dpi=130, bbox_inches="tight")
    plt.close(fig)
    log("saved", name)


@contextlib.contextmanager
def quiet():
    with contextlib.redirect_stdout(io.StringIO()):
        yield


# ------------------------------------------------------------------ Lorenz (same as exp_d)
def lorenz(t, y, s_=10.0, r_=28.0, b_=8.0 / 3.0):
    return [s_ * (y[1] - y[0]), y[0] * (r_ - y[2]) - y[1], y[0] * y[1] - b_ * y[2]]


def make_trajs(span_tu, n, seed=0):
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        y0 = np.array([0.0, 1.0, 1.05]) + 0.5 * rng.standard_normal(3)
        y1 = solve_ivp(lorenz, [0, 50.0], y0, method="DOP853", rtol=1e-12, atol=1e-12).y[:, -1]
        out.append(solve_ivp(lorenz, [0, span_tu], y1, method="DOP853", rtol=1e-12, atol=1e-12, dense_output=True))
    return out


def attractor_stats():
    y0 = solve_ivp(lorenz, [0, 50.0], [1.0, 1.0, 1.05], method="DOP853", rtol=1e-12, atol=1e-12).y[:, -1]
    s = solve_ivp(lorenz, [0, 2000.0], y0, method="DOP853", rtol=1e-10, atol=1e-10, t_eval=np.arange(0, 2000, 0.01))
    return s.y.mean(1), s.y.std(1)


def tau_quadrature(t_f, h=0.01, order=16):
    edges = np.unique(np.round(np.concatenate([np.arange(0, t_f, h), [t_f], np.arange(0, t_f, 1.0)]), 12))
    edges = edges[edges <= t_f]
    x, w = roots_legendre(order)
    lo, hi = edges[:-1], edges[1:]
    tau = (lo[:, None] + (hi - lo)[:, None] * (x[None] + 1) / 2).ravel()
    wt = ((hi - lo)[:, None] * w[None] / 2).ravel()
    return tau, wt


def scan(dA, dB, U, idxs):
    C = np.zeros((dA.shape[0], U.shape[1]))
    out = {}
    with np.errstate(all="ignore"):
        for k in range(U.shape[0]):
            C = dA @ C + dB[:, None] * U[k][None, :]
            if k in idxs:
                out[k] = C.copy()
    return out


# ------------------------------------------------------------------ single bases (library) and tau-functions
FB = make_frequency_preserving_basis(alpha=ALPHA, window_length=WL, eps=EPS)


def build_single(N):
    with quiet():
        legs = LagHiPPO(N=N, phi_func=phi, phi_deriv_func=phi_deriv, sigma_func=sigma, sigma_deriv_func=sigma_deriv,
                        sigma_inv_func=sigma_inv, sigma_inv_deriv_func=sigma_inv_deriv, dt=DT, T=T_MEM)
        foup = make_frequency_preserving_model(N=N, dt=DT, T=T_MEM, alpha=ALPHA, window_length=WL, eps=EPS,
                                               input_hold="zoh", quad_order=200 * N, derive_A_from="gen")
    return {"LegS": legs, "FouP": foup}


def tau_funcs(tau, NL=N1, NF=N1):
    """Psi_L, Psi_F, d/dtau of each, omega_L, omega_F, d omega_reg / omega, evaluated at tau (arrays)."""
    tau = np.asarray(tau, float)
    z = np.exp(-tau)
    PL = vectorized_basis(phi, np.arange(NL), z)
    dPL = vectorized_basis(phi_deriv, np.arange(NL), z) * (-z)[:, None]
    PF = np.stack([FB.psi(k, -tau) for k in range(NF)], 1)
    dPF = np.stack([-FB.psi_deriv(k, -tau) for k in range(NF)], 1)
    wL = z
    wF = FB.sigma_deriv(0.0, -tau)
    return PL, dPL, PF, dPF, wL, wF


class Basis:
    """name, r states, Rd: (NLAG, r) state -> history on lag grid, Espan: (NLAG, d) span of the basis on the lag grid,
    ip(tau, wt, U): (r, ntraj) inner-product projection, dA/dB: float64 discrete recurrence."""
    def __init__(self, name, Rd, Espan, ip, dA, dB, Acont=None):
        self.name, self.Rd, self.Espan, self.ip, self.dA, self.dB, self.Acont = name, Rd, Espan, ip, dA, dB, Acont
        self.r = Rd.shape[1]
        self.rho = float(np.max(np.abs(np.linalg.eigvals(dA))))


def single_basis(name, m, N):
    dA, dB = m.dA.numpy().astype(np.float64), m.dB.numpy().astype(np.float64)
    lagv = np.arange(NLAG) * DT
    E = vectorized_basis(m.phi_func, np.arange(N), m.sigma_func(0, -lagv))

    def ip(tau, wt, U):
        z = m.sigma_func(0, -tau)
        P = vectorized_basis(m.phi_func, np.arange(N), z)
        return P.T @ ((wt * m.sigma_deriv_func(0, -tau))[:, None] * U)
    return Basis(name, E, E, ip, dA, dB, Acont=np.asarray(m.A, float))


# ------------------------------------------------------------------ the union quantities (common-measure Gram etc.)
tau_g, w_g = tau_quadrature(TAU_G)
PLg, dPLg, PFg, dPFg, wLg, wFg = tau_funcs(tau_g)
lagv = np.arange(NLAG) * DT
PLe, _, PFe, _, _, _ = tau_funcs(lagv)
PL0, _, PF0, _, wL0, wF0 = tau_funcs(np.array([0.0]))
Phi_g = np.hstack([PLg, PFg])                 # (nodes, 32)
dPhi_g = np.hstack([dPLg, dPFg])
Phi_e = np.hstack([PLe, PFe])
Phi_0 = np.hstack([PL0, PF0])[0]
MEASURES = {"mix": lambda wl, wf: 0.5 * (wl + wf), "F": lambda wl, wf: wf, "L": lambda wl, wf: wl}
# regular part of omega'/omega (needed only for the "weighted" closure measure): omega_L'=-omega_L, omega_F'_reg=0
def wprime_over_w(kind, wl, wf):
    if kind == "mix": return -0.5 * wl / (0.5 * (wl + wf))
    if kind == "F": return np.zeros_like(wl)
    return -np.ones_like(wl)


def make_B(kind, kappa=1.0):
    wc = MEASURES[kind](wLg, wFg)
    W = w_g * wc
    G = Phi_g.T @ (W[:, None] * Phi_g)
    lam, V = np.linalg.eigh(G)
    keep = lam > RCOND * lam.max()
    T = V[:, keep] / np.sqrt(lam[keep])[None, :]
    Psi_g, dPsi_g = Phi_g @ T, dPhi_g @ T
    D = dPsi_g.T @ (W[:, None] * Psi_g)       # D_jk = <Psi_j', Psi_k>
    A_gen = -D.T
    wc0 = float(MEASURES[kind](wL0, wF0)[0])
    psi0 = Phi_0 @ T
    A = A_gen - kappa * wc0 * np.outer(psi0, psi0)
    B = kappa * wc0 * psi0
    C = np.ones((1, A.shape[0]))
    dA, _, _, _, _ = sps.cont2discrete((A, B[:, None], C, np.zeros((1, 1))), dt=DT, method="bilinear")
    _, dB, _, _, _ = sps.cont2discrete((A, B[:, None], C, np.zeros((1, 1))), dt=DT, method="zoh")
    Rd = Phi_e @ T

    def ip(tau, wt, U):
        P = np.hstack(tau_funcs(tau)[0::2][:2]) @ T            # Phi(tau) T
        wcn = MEASURES[kind](*[tau_funcs(tau)[4], tau_funcs(tau)[5]])
        return P.T @ ((wt * wcn)[:, None] * U)
    info = dict(kind=kind, kappa=kappa, rank=int(keep.sum()), dropped=int((~keep).sum()),
                cond_G_full=float(lam.max() / max(lam.min(), 1e-300)), cond_G_kept=float(lam.max() / lam[keep].min()),
                smallest_eigs=[float(x) for x in lam[:4]], largest_eig=float(lam.max()), omega_c0=wc0)
    return Basis(f"B_{kind}" + ("" if kappa == 1.0 else f"_k{kappa:.3g}"), Rd, Rd, ip, dA[:, :], dB[:, 0], Acont=A), \
        dict(T=T, Psi_g=Psi_g, dPsi_g=dPsi_g, W=W, wc=wc, A=A, B=B, A_gen=A_gen, info=info, D=D)


def make_A(bL, bF):
    r = bL.r + bF.r
    dA = np.zeros((r, r)); dA[:bL.r, :bL.r] = bL.dA; dA[bL.r:, bL.r:] = bF.dA
    dB = np.concatenate([bL.dB, bF.dB])
    # moment matrix  M a = c   (c_L = <u,phi_L>_{wL}, c_F = <u,phi_F>_{wF})
    ML = PLg.T @ ((w_g * wLg)[:, None] * Phi_g)           # (16, 32) = [<phi_L,phi_L>_{wL} | <phi_L,phi_F>_{wL}]
    MF = PFg.T @ ((w_g * wFg)[:, None] * Phi_g)           # (16, 32)
    M = np.vstack([ML, MF])
    U_, s_, Vt_ = np.linalg.svd(M)
    keep = s_ > RCOND * s_.max()
    Mp = (Vt_[keep].T / s_[keep]) @ U_[:, keep].T
    Rd = Phi_e @ Mp
    Aco = np.zeros((r, r)); Aco[:bL.r, :bL.r] = bL.Acont; Aco[bL.r:, bL.r:] = bF.Acont

    def ip(tau, wt, U):
        zl = np.exp(-tau)
        PL_, _, PF_, _, wl_, wf_ = tau_funcs(tau)
        return np.vstack([PL_.T @ ((wt * wl_)[:, None] * U), PF_.T @ ((wt * wf_)[:, None] * U)])
    # span of the union on the lag grid, rank-reduced under omega_mix for the grid least squares
    info = dict(r_M=int(keep.sum()), dropped_M=int((~keep).sum()), cond_M_full=float(s_.max() / max(s_.min(), 1e-300)),
                cond_M_kept=float(s_.max() / s_[keep].min()), smallest_sv=[float(x) for x in s_[-4:]], largest_sv=float(s_.max()))
    return Basis("A_blockdiag", Rd, Phi_e, ip, dA, dB, Acont=Aco), info


def grid_lsq(Espan, H):
    return np.linalg.lstsq(Espan, H, rcond=RCOND)[0]


# ------------------------------------------------------------------ closure error  ||(I-P) d/dtau Psi||_omega
def closure(Psi, dPsi, W, wpw=None):
    G = Psi.T @ (W[:, None] * Psi)
    Gp = np.linalg.pinv(G, rcond=RCOND, hermitian=True)
    g = dPsi if wpw is None else dPsi + Psi * wpw[:, None]
    Q = Gp @ (Psi.T @ (W[:, None] * g))
    tot = np.einsum("ik,i,ik->k", g, W, g)
    res = np.maximum(tot - np.einsum("jk,jl,lk->k", Q, G, Q), 0.0)
    return np.sqrt(res), np.sqrt(tot)


# ------------------------------------------------------------------ run
log("repo", REPO, "-> out", OUT)
mu, sd = attractor_stats()
trajs = make_trajs(6.0 * TU + 1.0, NTRAJ, seed=0)
singles = {N: build_single(N) for N in (16, 32)}
bLegS16, bFouP16 = single_basis("LegS16", singles[16]["LegS"], 16), single_basis("FouP16", singles[16]["FouP"], 16)
bLegS32, bFouP32 = single_basis("LegS32", singles[32]["LegS"], 32), single_basis("FouP32", singles[32]["FouP"], 32)
bA, infoA = make_A(bLegS16, bFouP16)
Bs, Bd = {}, {}
for kind in ("mix", "F", "L"):
    Bs[f"B_{kind}"], Bd[f"B_{kind}"] = make_B(kind, 1.0)
for kap in (4.0 / 3.0, 2.0):
    b, d = make_B("mix", kap)
    Bs[b.name], Bd[b.name] = b, d
MODELS = [bLegS16, bFouP16, bLegS32, bFouP32, bA] + list(Bs.values())
MAIN = ["LegS16", "FouP16", "LegS32", "FouP32", "A_blockdiag", "B_mix"]
byname = {b.name: b for b in MODELS}
log("states:", {b.name: b.r for b in MODELS})
log("rho(dA):", {b.name: round(b.rho, 6) for b in MODELS})

# ---- sanity: my tau-functions vs library eval matrix; my A_gen vs library A_gen; exact generator vs library A
chk = {}
chk["Psi_L_vs_lib_eval"] = float(np.max(np.abs(PLe - byname["LegS16"].Rd)))
chk["Psi_F_vs_lib_eval"] = float(np.max(np.abs(PFe - byname["FouP16"].Rd)))
for nm, Pg, dPg, wg, N_ in (("LegS16", PLg, dPLg, wLg, 16), ("FouP16", PFg, dPFg, wFg, 16)):
    D = dPg.T @ ((w_g * wg)[:, None] * Pg)
    Ag = -D.T
    mlib = singles[16]["LegS" if "LegS" in nm else "FouP"]
    chk[f"A_gen_mine_vs_lib_{nm}"] = float(np.max(np.abs(Ag - mlib.A_gen)))
    chk[f"A_gen_scale_{nm}"] = float(np.max(np.abs(mlib.A_gen)))
    if nm == "LegS16":
        A_exact = Ag - 1.0 * np.outer(Pg[0] * 0 + PL0[0], PL0[0])   # kappa=1, omega_L(0)=1
        chk["A_exact_kappa1_vs_lib_legs_tilt_LegS16"] = float(np.max(np.abs(A_exact - mlib.A)))
# finite-difference test of  dc/dt = -D^T c  for u_hat = Psi c continued with constant input u_hat(0)  (union, mix, tau shift on a 0.005 grid)
kind = "mix"; d = Bd["B_mix"]
rng = np.random.default_rng(1)
cvec = rng.standard_normal(d["T"].shape[1])
def coef_after_shift(delta, hq=0.0025):
    tau, wt = tau_quadrature(TAU_G, h=hq)
    PL_, _, PF_, _, wl_, wf_ = tau_funcs(tau)
    Psi = np.hstack([PL_, PF_]) @ d["T"]
    wc_ = MEASURES[kind](wl_, wf_)
    # shifted history: u_{t+delta}(tau) = u_hat(tau-delta) for tau>delta, = u_hat(0) for tau<delta
    tl = np.maximum(tau - delta, 0.0)
    PLs, _, PFs, _, _, _ = tau_funcs(tl)
    uh = (np.hstack([PLs, PFs]) @ d["T"]) @ cvec
    return Psi.T @ (wt * wc_ * uh)
c0 = coef_after_shift(0.0)
fd = [(coef_after_shift(dl) - c0) / dl for dl in (0.005, 0.0025)]
rich = 2 * fd[1] - fd[0]
pred = -d["D"].T @ c0
chk["shift_FD_vs_minus_DT_c_rel_err_richardson"] = float(np.linalg.norm(rich - pred) / np.linalg.norm(pred))
chk["shift_FD_vs_minus_DT_c_rel_err_first_order"] = float(np.linalg.norm(fd[1] - pred) / np.linalg.norm(pred))
chk["shift_FD_check_coef_c0_vs_c_rel_err"] = float(np.linalg.norm(c0 - cvec) / np.linalg.norm(cvec))
log("checks:", json.dumps(chk, indent=1))

# ---- Gram / conditioning table
gram_info = {k: Bd[f"B_{k}"]["info"] for k in ("mix", "F", "L")}
gram_info["A_moment_matrix"] = infoA
log("Gram:", json.dumps(gram_info, indent=1))

# ---- closure errors
clo = {}
def clo_entry(name, Psi, dPsi, W, wpw):
    e, nrm = closure(Psi, dPsi, W)
    ew, nrmw = closure(Psi, dPsi, W, wpw)
    clo[name] = dict(per_vector_abs=e.tolist(), per_vector_rel=(e / np.maximum(nrm, 1e-300)).tolist(),
                     total_abs=float(np.sqrt((e ** 2).sum())), total_rel=float(np.sqrt((e ** 2).sum() / (nrm ** 2).sum())),
                     per_vector_abs_weighted=ew.tolist(), total_abs_weighted=float(np.sqrt((ew ** 2).sum())),
                     total_rel_weighted=float(np.sqrt((ew ** 2).sum() / (nrmw ** 2).sum())))
clo_entry("LegS16", PLg, dPLg, w_g * wLg, -np.ones_like(wLg))
clo_entry("FouP16", PFg, dPFg, w_g * wFg, np.zeros_like(wFg))
for N_ in (32,):
    PL_, dPL_, PF_, dPF_, _, _ = tau_funcs(tau_g, N_, N_)
    clo_entry("LegS32", PL_, dPL_, w_g * wLg, -np.ones_like(wLg))
    clo_entry("FouP32", PF_, dPF_, w_g * wFg, np.zeros_like(wFg))
for kind in ("mix", "F", "L"):
    d = Bd[f"B_{kind}"]
    clo_entry(f"union_{kind}", d["Psi_g"], d["dPsi_g"], d["W"], wprime_over_w(kind, wLg, wFg))
# closure of the ORIGINAL columns of Phi (LegS cols 0-15, FouP cols 16-31) w.r.t. the union span under each common measure
for kind in ("mix", "F", "L"):
    Wc = w_g * MEASURES[kind](wLg, wFg)
    e0, n0 = closure(Phi_g, dPhi_g, Wc)
    e1, n1 = closure(Phi_g, dPhi_g, Wc, wprime_over_w(kind, wLg, wFg))
    clo[f"union_cols_{kind}"] = dict(LegS_cols_abs=float(np.sqrt((e0[:16] ** 2).sum())), LegS_cols_abs_weighted=float(np.sqrt((e1[:16] ** 2).sum())),
                                     FouP_cols_excl_top_abs=float(np.sqrt((e0[16:31] ** 2).sum())), FouP_cols_excl_top_abs_weighted=float(np.sqrt((e1[16:31] ** 2).sum())),
                                     FouP_top_sin8_abs=float(e0[31]), per_col_abs=e0.tolist(), per_col_abs_weighted=e1.tolist(),
                                     LegS_cols_norm=float(np.sqrt((n1[:16] ** 2).sum())))
json.dump(clo, open(os.path.join(OUT, "closure_errors.json"), "w"), indent=1)
log("union columns closure:", {k: {a: round(b, 4) for a, b in v.items() if not a.startswith("per_")} for k, v in clo.items() if k.startswith("union_cols")})
log("closure totals (abs / rel / weighted-rel):", {k: [round(v["total_abs"], 4), round(v["total_rel"], 4), round(v["total_rel_weighted"], 4)] for k, v in clo.items() if "total_abs" in v})

# ---- runs
def run(comp, keep=False):
    s_all = np.arange(K_TOT) * DT
    Uall = np.stack([(t.sol(TU * s_all)[COMP[comp]] - mu[COMP[comp]]) / sd[COMP[comp]] for t in trajs], 1)
    idxs = {int(round(tf / DT)) - 1 for tf in TF}
    scans = {b.name: scan(b.dA, b.dB, Uall, idxs) for b in MODELS}
    res, ex = {}, {}
    for tf in TF:
        K = int(round(tf / DT)); t_f = (K - 1) * DT
        tau, wt = tau_quadrature(t_f)
        un = np.stack([(t.sol(TU * (t_f - tau))[COMP[comp]] - mu[COMP[comp]]) / sd[COMP[comp]] for t in trajs], 1)
        H = Uall[K - 1::-1][:NLAG]
        for b in MODELS:
            C = {"rec": scans[b.name][K - 1], "ip": b.ip(tau, wt, un), "lsq": None}
            for route in ("rec", "ip", "lsq"):
                with np.errstate(all="ignore"):
                    R = b.Espan @ grid_lsq(b.Espan, H) if route == "lsq" else b.Rd @ C[route]
                    se = (R - H) ** 2
                res[(b.name, route, tf)] = dict(mse=se.mean(0), win=np.stack([se[WIN * i:WIN * (i + 1)].mean(0) for i in range(3)]),
                                                bins=se.reshape(NLAG // BIN, BIN, -1).mean(1))
                if keep:
                    ex[(b.name, route, tf)] = R[:, 0].copy()
        if keep:
            ex[("H", tf)] = H[:, 0].copy()
    return res, ex

res_x, ex_x = run("x", keep=True)
log("x done")
res_z, _ = run("z")
log("z done")

# ------------------------------------------------------------------ tables
def summ(a):
    a = np.asarray(a, float); return [float(np.mean(a)), float(np.std(a))]
fmt = lambda a: f"{a[0]:.3g}±{a[1]:.2g}"
tab = {f"{c}|{n}|{r}|tf={tf}": dict(mse_mean_std=summ(v["mse"]), win_mean=[float(np.mean(v["win"][i])) for i in range(3)],
                                     n_nonfinite=int((~np.isfinite(v["mse"])).sum()))
       for c, res in (("x", res_x), ("z", res_z)) for (n, r, tf), v in res.items()}
json.dump(dict(table=tab, rho={b.name: b.rho for b in MODELS}, states={b.name: b.r for b in MODELS}, checks=chk, gram=gram_info),
          open(os.path.join(OUT, "results.json"), "w"), indent=1)
L_ = []
names_all = [b.name for b in MODELS]
for cname, res in (("x", res_x), ("z", res_z)):
    for tf in TF:
        L_.append(f"### {cname}(t), s={tf}: MSE over lag [0,3L) (mean±std, 8 traj; predict-0 = 1)")
        L_.append("| model | states | rho(dA) | rec | ip (projection) | lsq oracle |"); L_.append("|---|---|---|---|---|---|")
        for n in names_all:
            L_.append(f"| {n} | {byname[n].r} | {byname[n].rho:.5f} | " + " | ".join(fmt(summ(res[(n, r, tf)]['mse'])) for r in ("rec", "ip", "lsq")) + " |")
        L_.append("")
for tf in (3.0, 6.0):
    L_.append(f"### x(t), s={tf}: per-window MSE (windows lag [0,1),[1,2),[2,3) L; window 1 = most recent)")
    L_.append("| model | route | win1 | win2 | win3 |"); L_.append("|---|---|---|---|---|")
    for n in names_all:
        for r in ("rec", "ip"):
            v = res_x[(n, r, tf)]["win"]
            L_.append(f"| {n} | {r} | " + " | ".join(f"{np.mean(v[i]):.3g}" for i in range(3)) + " |")
    L_.append("")
L_.append("### closure error ||(I-P) d/dtau Psi||_omega (total over vectors; absolute, relative to ||Psi'||, relative incl. omega'/omega term)")
L_.append("| basis | abs | rel | rel (weighted) |"); L_.append("|---|---|---|---|")
for k, v in clo.items():
    if k.startswith("union_cols"): continue
    L_.append(f"| {k} | {v['total_abs']:.4g} | {v['total_rel']:.4g} | {v['total_rel_weighted']:.4g} |")
open(os.path.join(OUT, "results_tables.md"), "w").write("\n".join(L_))
log("tables written")

# ------------------------------------------------------------------ plots (past on the left: x = -lag)
CL = {"LegS16": "tab:blue", "FouP16": "tab:red", "LegS32": "deepskyblue", "FouP32": "salmon",
      "A_blockdiag": "tab:green", "B_mix": "tab:purple"}
LS = {"LegS32": "--", "FouP32": "--"}
lag = lagv
tt = np.linspace(0, T_MEM, 30001)
wLt, wFt = np.exp(-tt), FB.sigma_deriv(0.0, -tt)

def measure_panel(ax):
    ax.semilogy(-tt, wFt, color="tab:red", lw=1.6, label=r"FouP $\omega_F=h_n/L$")
    ax.semilogy(-tt, wLt, color="tab:blue", lw=1.6, label=r"LegS $\omega_L=e^{-\tau}$")
    ax.semilogy(-tt, 0.5 * (wLt + wFt), color="tab:purple", lw=1.4, ls="--", label=r"common (mixture) $\omega_c=(\omega_L+\omega_F)/2$")
    for e in (1, 2):
        ax.axvline(-e, color="gray", ls=":", lw=0.8)
    for xc, s_ in ((-0.5, "window 1\n(most recent)"), (-1.5, "window 2"), (-2.5, "window 3\n(oldest)")):
        ax.text(xc, 0.0095, s_, ha="center", va="bottom", fontsize=7, color="0.35")
    ax.set(xlim=(-T_MEM, 0), ylim=(0.008, 3.0), ylabel=r"$\omega(\tau)$")
    ax.legend(fontsize=6.5, loc="upper left")
    ax.set_xlabel(r"time relative to now $-\tau$ [$L$ = 3 Lorenz t.u.];  0 = now, $-3L$ = oldest", fontsize=9)

# Fig F1: history overlay
fig = plt.figure(figsize=(14, 11.5)); gs = fig.add_gridspec(4, 2, height_ratios=[3, 3, 3, 2.4], hspace=0.28)
axs = np.array([[fig.add_subplot(gs[i, j]) for j in range(2)] for i in range(3)])
groups = [["LegS16", "FouP16", "A_blockdiag"], ["B_mix", "LegS32", "FouP32"]]
titles = ["16 + 16 separate, and fused A (block-diagonal, 32 states)", "fused B (orthogonalised union) and 32-state single bases"]
fig.text(0.30, 0.935, titles[0], ha="center", fontsize=10); fig.text(0.72, 0.935, titles[1], ha="center", fontsize=10)
for i, tf in enumerate(TF):
    for j in range(2):
        ax = axs[i, j]
        ax.plot(-lag, ex_x[("H", tf)], "k", lw=1.0, label="true history")
        for n in groups[j]:
            ax.plot(-lag, ex_x[(n, "rec", tf)], color=CL[n], ls=LS.get(n, "-"), lw=1.0,
                    label=f"{n} ({byname[n].r} st.), MSE={res_x[(n, 'rec', tf)]['mse'][0]:.3g}")
        for e in (1, 2): ax.axvline(-e, color="gray", ls=":", lw=0.8)
        ax.set(title=fr"final time $s={tf}$", ylim=(-3.2, 5.0), xlim=(-T_MEM, 0))
        ax.tick_params(labelbottom=False); ax.legend(fontsize=6.5, loc="upper left", ncol=2)
        if j == 0: ax.set_ylabel("u")
for j in range(2):
    measure_panel(fig.add_subplot(gs[3, j], sharex=axs[2, j]))
fig.suptitle("Lorenz $x(t)$ history, recurrence, trajectory 0 (past on the left, now on the right)", y=0.965)
save(fig, "F1_history_fused_pastleft.png")

# Fig F2: error vs lag
bl = -(np.arange(NLAG // BIN) + 0.5) * BIN * DT
fig = plt.figure(figsize=(12, 9)); gs = fig.add_gridspec(2, 1, height_ratios=[3, 1.8], hspace=0.2)
ax = fig.add_subplot(gs[0])
for n in MAIN:
    v = np.mean([res_x[(n, "rec", tf)]["bins"] for tf in TF], axis=(0, 2))
    ax.semilogy(bl, v, color=CL[n], ls=LS.get(n, "-"), lw=1.5, label=f"{n} ({byname[n].r} states)")
for e in (1, 2): ax.axvline(-e, color="gray", ls=":", lw=0.8)
ax.set(xlim=(-T_MEM, 0), ylabel=r"squared error (bins of $0.05L$)", title=r"Error vs time-before-now, recurrence; mean over 8 trajectories $\times$ $s\in\{3,4.5,6\}$")
ax.legend(fontsize=8, ncol=2, loc="lower left"); ax.tick_params(labelbottom=False)
measure_panel(fig.add_subplot(gs[1], sharex=ax))
save(fig, "F2_error_vs_lag_fused_pastleft.png")

# Fig F3: MSE bars (all / windows)
fig, axs = plt.subplots(1, 4, figsize=(17, 4.4), sharey=True)
for ax, (lab, getter) in zip(axs, [("all windows [0,3L)", lambda v: v["mse"]), ("window 1 (most recent)", lambda v: v["win"][0]),
                                   ("window 2", lambda v: v["win"][1]), ("window 3 (oldest)", lambda v: v["win"][2])]):
    for q, n in enumerate(MAIN):
        v = getter(res_x[(n, "rec", 6.0)]); m_ = np.exp(np.log(np.maximum(v, 1e-300)).mean()); sdv = np.log(np.maximum(v, 1e-300)).std()
        ax.bar(q, m_, color=CL[n], yerr=[[m_ - m_ * np.exp(-sdv)], [m_ * np.exp(sdv) - m_]], capsize=2)
        ax.text(q, m_ * np.exp(sdv) * 1.12, f"{m_:.2g}", ha="center", fontsize=7)
    ax.axhline(1.0, color="k", lw=0.6, ls=":"); ax.set_yscale("log"); ax.set_ylim(5e-3, 8)
    ax.set_xticks(range(len(MAIN))); ax.set_xticklabels([f"{n}\n({byname[n].r})" for n in MAIN], rotation=60, fontsize=7)
    ax.set_title(lab)
axs[0].set_ylabel("MSE (geo. mean, bars ±1 s.d. of log)")
fig.suptitle(r"Lorenz $x(t)$, recurrence, final time $s=6$ (dotted: predict 0). Labels: model (number of states)", y=1.03)
save(fig, "F3_mse_bars_fused.png")

# Fig F4: closure errors
FLOOR = 1e-9
fig, axs = plt.subplots(1, 3, figsize=(19, 4.8))
ax = axs[0]
for nm, c_ in (("LegS16", "tab:blue"), ("FouP16", "tab:red"), ("union_mix", "tab:purple"), ("union_F", "tab:orange"), ("union_L", "tab:cyan")):
    v = np.maximum(np.array(clo[nm]["per_vector_rel"]), FLOOR); ax.plot(np.arange(len(v)), v, "o-", ms=3, color=c_, label=f"{nm} (total rel {clo[nm]['total_rel']:.3g})")
ax.set(xlabel="basis vector $k$ (union: orthonormal $\\Psi_k$)", ylabel=r"$\|(I-P)\partial_\tau\Psi_k\|_\omega\,/\,\|\partial_\tau\Psi_k\|_\omega$", yscale="log", ylim=(FLOOR / 3, 6),
       title=f"non-closure per basis vector (relative; 0 drawn at {FLOOR:g})")
ax.legend(fontsize=7, loc="center left")
ax = axs[1]
cc = clo["union_cols_mix"]; kk = np.arange(32)
ax.semilogy(kk, np.maximum(cc["per_col_abs"], FLOOR), "o-", ms=3, color="tab:purple", label=r"plain $\partial_\tau$")
ax.semilogy(kk, np.maximum(cc["per_col_abs_weighted"], FLOOR), "s--", ms=3, color="tab:green", label=r"with $\omega'/\omega$ term (mixture measure)")
ax.axvline(15.5, color="gray", ls=":"); ax.text(7.5, 3e-9, "LegS columns", ha="center", fontsize=8); ax.text(23.5, 3e-9, "FouP columns", ha="center", fontsize=8)
ax.set(xlabel="column $k$ of unorthogonalised $\\Phi$", ylabel=r"absolute $\|(I-P)\partial_\tau\Phi_k\|_{\omega_c}$", ylim=(FLOOR / 3, 300),
       title="union, original columns: only FouP sin8 fails (plain)")
ax.legend(fontsize=7, loc="center left")
ax = axs[2]
names_c = ["LegS16", "FouP16", "LegS32", "FouP32", "union_mix", "union_F", "union_L"]
x = np.arange(len(names_c))
for off, key, lab in ((-0.2, "total_rel", r"plain $\partial_\tau\Psi$"), (0.2, "total_rel_weighted", r"with $\omega'/\omega$ term")):
    vv = [clo[n][key] for n in names_c]
    ax.bar(x + off, np.maximum(vv, 1e-4), 0.4, label=lab)
    for xi, v_ in zip(x + off, vv): ax.text(xi, max(v_, 1e-4) * 1.2, f"{v_:.2g}", ha="center", fontsize=6, rotation=90)
ax.set_yscale("log"); ax.set_ylim(1e-4, 3); ax.set_xticks(x); ax.set_xticklabels(names_c, rotation=30, fontsize=8)
ax.set(ylabel="total relative closure error (floor 1e-4)", title="total non-closure"); ax.legend(fontsize=7, loc="lower right")
fig.suptitle(r"Closure error $\|(I-P)\partial_\tau\Psi\|_\omega$, $P$ = $\omega$-projection onto span $\Psi$", y=1.03)
save(fig, "F4_closure_error.png")

# Fig F5: eigenvalues
fig, axs = plt.subplots(1, 2, figsize=(14, 5))
for n, c_ in (("LegS16", "tab:blue"), ("FouP16", "tab:red"), ("B_mix", "tab:purple")):
    ev = np.linalg.eigvals(byname[n].Acont)
    axs[0].plot(ev.real, ev.imag, "o", ms=4, color=c_, alpha=0.8, label=f"{n}")
axs[0].axvline(0, color="k", lw=0.6)
axs[0].set(xlabel="Re", ylabel="Im", title="continuous $A$ (fused A = LegS16 $\\cup$ FouP16 blocks)"); axs[0].legend(fontsize=8)
th = np.linspace(0, 2 * np.pi, 400); axs[1].plot(np.cos(th), np.sin(th), "k:", lw=0.8)
for n in MAIN:
    ev = np.linalg.eigvals(byname[n].dA)
    axs[1].plot(ev.real, ev.imag, ".", ms=5, color=CL[n], label=f"{n}: $\\rho$={byname[n].rho:.5f}")
axs[1].set(xlabel="Re", ylabel="Im", xlim=(0.978, 1.0015), ylim=(-0.05, 0.05), title="discrete $\\mathrm{d}A$ (zoom near $z=1$; dotted = unit circle)")
axs[1].legend(fontsize=7, loc="lower left")
save(fig, "F5_eigenvalues.png")

# Fig F6: projection vs recurrence gap
fig, axs = plt.subplots(1, 2, figsize=(14, 4.4), sharey=True)
names6 = ["LegS16", "FouP16", "A_blockdiag", "B_mix", "B_F", "B_L"]
for ax, tf in zip(axs, (3.0, 6.0)):
    x = np.arange(len(names6))
    for q, (route, lab, a_) in enumerate((("rec", "recurrence", 1.0), ("ip", "projection (inner product)", 0.6), ("lsq", "grid-LS oracle", 0.3))):
        vals = [np.exp(np.log(np.maximum(res_x[(n, route, tf)]["mse"], 1e-300)).mean()) for n in names6]
        ax.bar(x + (q - 1) * 0.27, vals, 0.27, label=lab, alpha=a_, color="0.2")
    ax.set_ylim(0, 1.2); ax.axhline(1.0, color="k", lw=0.6, ls=":"); ax.set_xticks(x); ax.set_xticklabels(names6, rotation=30, fontsize=8)
    ax.set_title(fr"final time $s={tf}$"); ax.legend(fontsize=7, loc="upper right")
axs[0].set_ylabel("MSE over lag [0,3L) (geo. mean)")
fig.suptitle("Projection-only vs recurrence vs oracle: the gap = recurrence (non-closure) error", y=1.03)
save(fig, "F6_projection_vs_recurrence.png")

# Fig F7: kappa / measure sensitivity of B
fig, ax = plt.subplots(figsize=(9, 4))
nm7 = ["B_mix", "B_F", "B_L", "B_mix_k1.33", "B_mix_k2"]
for q, tf in enumerate(TF):
    vals = [np.exp(np.log(np.maximum(res_x[(n, "rec", tf)]["mse"], 1e-300)).mean()) for n in nm7]
    ax.bar(np.arange(len(nm7)) + (q - 1) * 0.27, vals, 0.27, label=f"s={tf}")
ax.set_ylim(0, 0.5); ax.set_xticks(range(len(nm7))); ax.set_xticklabels(nm7, rotation=20, fontsize=8)
ax.set(ylabel="recurrence MSE", title="fused B: choice of common measure and of $\\kappa$"); ax.legend(fontsize=8)
save(fig, "F7_B_variants.png")
log("DONE", OUT)
