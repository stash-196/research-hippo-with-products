"""Lorenz-63 history reconstruction: FouP vs LegS (LagOp library, fixed version >= 6574fb2).

Usage (repo root):  python -m lag_op_hippo.window_checks.exp_d_lorenz_foup_vs_legs
Writes ONLY into a NEW folder outputs/freq_dev/lorenz_foup_vs_legs_<YYYYmmdd_HHMMSS>/.

Conventions (same as check 02.02): model time s, dt = 1e-3, library models built with T = 3 (lag window
[0, 3) = 3 FouP windows of length L = 1), N = 16 (+ sweep), alpha = 0.2.  The recurrence is run in float64 with
the library's own (dA, dB) buffers; reconstruction of the history over lag tau in [0, 3) is  E c  with E the
library eval matrix (rows = lag).  Routes:
  rec : c_{k+1} = dA c_k + dB u_k scanned from c = 0 at s = 0
  ip  : inner-product projection c_k = int u(t_f - tau) phi_k(z(tau)) z'(tau) dtau over the whole history
        (zero before s = 0, i.e. the exact object the recurrence approximates)
  lsq : least squares of E c to the history on lag [0, 3)  (oracle = smallest MSE any N coefficients can give
        on the reported window)
"""
import contextlib, datetime, io, json, os, subprocess, sys, time
import numpy as np
from scipy.integrate import solve_ivp
from scipy.special import roots_legendre
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker

from lag_op_hippo.lag_hippo import (LagHiPPO, phi, phi_deriv, sigma, sigma_deriv, sigma_inv, sigma_inv_deriv,
                                    vectorized_basis)
from lag_op_hippo.lag_hippo_freq import make_frequency_preserving_model

plt.rcParams.update({"mathtext.fontset": "cm", "font.family": "serif", "font.size": 10,
                     "axes.grid": True, "grid.alpha": 0.3})

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
DT, T_MEM, ALPHA, WL, EPS = 1e-3, 3.0, 0.2, 1.0, 1e-8
NS = [8, 16, 32, 64]
N_MAIN = 16
TF = [3.0, 4.5, 6.0]                 # final times (model time s); 3.0 = check-02.02 sequence length
K_TOT = int(round(max(TF) / DT))     # 6000 samples
TU_MAIN = 3.0                        # Lorenz time units per FouP window L
TU_SWEEP = [1.5, 3.0, 6.0]
NTRAJ = 8
NLAG = int(round(T_MEM / DT))        # 3000
WIN = int(round(WL / DT))            # 1000 samples per window
BIN = 50                             # error-vs-lag bins of 0.05 L
COMP = {"x": 0, "z": 2}
MODELS = ["FouP", "LegS", "FouP_infty", "LegS_infty"]
ROUTES = ["rec", "ip", "lsq"]
COL = {"FouP": "tab:red", "LegS": "tab:blue"}

ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
OUT = os.path.join(REPO, "outputs", "freq_dev", f"lorenz_foup_vs_legs_{ts}")
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


# ------------------------------------------------------------------ Lorenz signal
def lorenz(t, y, s_=10.0, r_=28.0, b_=8.0 / 3.0):
    return [s_ * (y[1] - y[0]), y[0] * (r_ - y[2]) - y[1], y[0] * y[1] - b_ * y[2]]


def make_trajs(span_tu, n, seed=0):
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        y0 = np.array([0.0, 1.0, 1.05]) + 0.5 * rng.standard_normal(3)
        y1 = solve_ivp(lorenz, [0, 50.0], y0, method="DOP853", rtol=1e-12, atol=1e-12).y[:, -1]  # discard transient
        out.append(solve_ivp(lorenz, [0, span_tu], y1, method="DOP853", rtol=1e-12, atol=1e-12, dense_output=True))
    return out


def attractor_stats():
    y0 = solve_ivp(lorenz, [0, 50.0], [1.0, 1.0, 1.05], method="DOP853", rtol=1e-12, atol=1e-12).y[:, -1]
    s = solve_ivp(lorenz, [0, 2000.0], y0, method="DOP853", rtol=1e-10, atol=1e-10, t_eval=np.arange(0, 2000, 0.01))
    return s.y.mean(1), s.y.std(1)


# ------------------------------------------------------------------ models
def build_models(N):
    with quiet():
        legs = lambda **kw: LagHiPPO(N=N, phi_func=phi, phi_deriv_func=phi_deriv, sigma_func=sigma,
                                     sigma_deriv_func=sigma_deriv, sigma_inv_func=sigma_inv,
                                     sigma_inv_deriv_func=sigma_inv_deriv, dt=DT, T=T_MEM, **kw)
        foup = lambda **kw: make_frequency_preserving_model(N=N, dt=DT, T=T_MEM, alpha=ALPHA, window_length=WL,
                                                            eps=EPS, input_hold="zoh", quad_order=200 * N, **kw)
        return {"LegS": legs(), "LegS_infty": legs(derive_A_from="infty", input_hold="zoh"),
                "FouP": foup(derive_A_from="gen"), "FouP_infty": foup(derive_A_from="infty")}


class Mdl:
    def __init__(self, m, N):
        self.m, self.N = m, N
        self.dA = m.dA.numpy().astype(np.float64)
        self.dB = m.dB.numpy().astype(np.float64)
        vals = np.arange(NLAG) * DT
        self.E = vectorized_basis(m.phi_func, np.arange(N), m.sigma_func(0, -vals))   # (NLAG, N) float64
        assert self.E.shape == tuple(m.eval_matrix.shape)
        self.rho = float(np.max(np.abs(np.linalg.eigvals(self.dA))))

    def basis_at(self, tau):
        z = self.m.sigma_func(0, -tau)
        return vectorized_basis(self.m.phi_func, np.arange(self.N), z), self.m.sigma_deriv_func(0, -tau)


def scan(dA, dB, U, idxs):
    """U: (K, ntraj).  returns {idx: c (N, ntraj)} after processing u[0..idx]."""
    C = np.zeros((dA.shape[0], U.shape[1]))
    out = {}
    with np.errstate(all="ignore"):
        for k in range(U.shape[0]):
            C = dA @ C + dB[:, None] * U[k][None, :]
            if k in idxs:
                out[k] = C.copy()
    return out


def tau_quadrature(t_f, h=0.01, order=16):
    edges = np.unique(np.round(np.concatenate([np.arange(0, t_f, h), [t_f], np.arange(0, t_f, 1.0)]), 12))
    edges = edges[edges <= t_f]
    x, w = roots_legendre(order)
    lo, hi = edges[:-1], edges[1:]
    tau = (lo[:, None] + (hi - lo)[:, None] * (x[None] + 1) / 2).ravel()
    wt = ((hi - lo)[:, None] * w[None] / 2).ravel()
    return tau, wt


# ------------------------------------------------------------------ metrics
HW = np.hanning(WIN + 1)[:-1]            # periodic Hann: a rectangular FFT of a non-periodic window is dominated by edge leakage


def spec(X):
    """|FFT| of each window of a lag-ordered (NLAG, ntraj) array, Hann-windowed; axis 1 = m (cycles per L)."""
    return np.abs(np.fft.rfft(X.reshape(3, WIN, -1) * HW[None, :, None], axis=1)) / HW.sum()


def metrics(R, H, N, Sh):
    se = (R - H) ** 2
    mse = se.mean(0)
    win = np.stack([se[WIN * i:WIN * (i + 1)].mean(0) for i in range(3)])            # (3, ntraj)
    bins = se.reshape(NLAG // BIN, BIN, -1).mean(1)                                    # (60, ntraj)
    Sr = spec(R)                                                                       # (3, m, ntraj)
    M = N // 2 + 1                                                                     # modes m = 0..N/2
    emag = ((Sr[:, :M] - Sh[:, :M]) ** 2).sum(1) / (Sh[:, :M] ** 2).sum(1)             # (3, ntraj)
    pratio = (Sr[:, :M] ** 2).sum(1) / (Sh[:, :M] ** 2).sum(1)
    return dict(mse=mse, win=win, bins=bins, Sr=Sr, emag=emag, pratio=pratio)


def summ(a):
    a = np.asarray(a, float)
    return [float(np.mean(a)), float(np.std(a))]


def gstat(a):
    """geometric mean and exp(mean +- sd of log) -- for log-scale plots of positive errors."""
    L = np.log(np.maximum(np.asarray(a, float), 1e-300))
    m, sdv = L.mean(), L.std()
    return [float(np.exp(m)), float(np.exp(m - sdv)), float(np.exp(m + sdv))]


def run_cond(trajs, tu, comp, Ns, mu, sd, keep_example=False):
    """Returns res[(N, model, route, tf)] = metric dict (per-trajectory arrays)."""
    s_all = np.arange(K_TOT) * DT
    Uall = np.stack([(t.sol(tu * s_all)[COMP[comp]] - mu[COMP[comp]]) / sd[COMP[comp]] for t in trajs], 1)  # (K, ntraj)
    res, example = {}, {}
    quad = {}
    for tf in TF:
        K = int(round(tf / DT))
        t_f = (K - 1) * DT
        tau, wt = tau_quadrature(t_f)
        u_nodes = np.stack([(t.sol(tu * (t_f - tau))[COMP[comp]] - mu[COMP[comp]]) / sd[COMP[comp]] for t in trajs], 1)
        H = Uall[K - 1::-1][:NLAG] if K >= NLAG else None                              # lag-ordered history (NLAG, ntraj)
        Sh = spec(H)
        quad[tf] = (tau, wt, u_nodes, H, Sh)
    idxs = {int(round(tf / DT)) - 1 for tf in TF}
    for N in Ns:
        mods = {k: Mdl(v, N) for k, v in build_models(N).items()}
        log(f"  comp={comp} tu/L={tu} N={N}: built models; rho:",
            {k: round(v.rho, 5) for k, v in mods.items()})
        for name, md in mods.items():
            cs = scan(md.dA, md.dB, Uall, idxs)
            key = name.split("_")[0]
            for tf in TF:
                K = int(round(tf / DT))
                tau, wt, u_nodes, H, Sh = quad[tf]
                C = {"rec": cs[K - 1]}
                if True:
                    Phi, zp = md.basis_at(tau)
                    C["ip"] = Phi.T @ ((wt * zp)[:, None] * u_nodes)
                    C["lsq"] = np.linalg.lstsq(md.E, H, rcond=None)[0]
                for route in ROUTES:
                    with np.errstate(all="ignore"):
                        R = md.E @ C[route]
                        res[(N, name, route, tf)] = metrics(R, H, N, Sh)
                    if keep_example:
                        example[(N, name, route, tf)] = R[:, 0].copy()
                res[(N, name, "rho", tf)] = md.rho
                # per-window oracle: best N-coefficient fit to ONE window alone (what the basis could do for that window)
                wl = np.zeros((3, H.shape[1]))
                for n in range(3):
                    sl = slice(WIN * n, WIN * (n + 1))
                    cw = np.linalg.lstsq(md.E[sl], H[sl], rcond=None)[0]
                    wl[n] = ((md.E[sl] @ cw - H[sl]) ** 2).mean(0)
                res[(N, name, "lsqw", tf)] = dict(win=wl, mse=wl.mean(0))
        for tf in TF:
            example[("H", tf)] = quad[tf][3][:, 0].copy()
    return res, example, Uall, quad


# ------------------------------------------------------------------ main
log("repo", REPO, "-> out", OUT)
mu, sd = attractor_stats()
log("Lorenz-63 stats (sigma=10, rho=28, beta=8/3):  mean", mu.round(3), "std", sd.round(3))
span = 6.0 * max(TU_SWEEP) + 1.0
trajs = make_trajs(span, NTRAJ, seed=0)
log(f"{NTRAJ} trajectories, DOP853 rtol=atol=1e-12, transient 50 t.u. discarded, span {span} t.u.")

# ---- sanity: cos(2 pi 3 s) through this pipeline vs the library float32 forward (check 02.02 analogue)
sanity = {}
with quiet():
    mods = build_models(N_MAIN)
s_cos = np.arange(NLAG) * DT
ucos = np.cos(2 * np.pi * 3.0 * s_cos)
import torch
for name in ("FouP", "LegS"):
    md = Mdl(mods[name], N_MAIN)
    c = scan(md.dA, md.dB, ucos[:, None], {NLAG - 1})[NLAG - 1][:, 0]
    mse64 = float(np.mean((md.E @ c - ucos[::-1]) ** 2))
    with torch.no_grad():
        r32 = mods[name].reconstruct(mods[name](torch.tensor(ucos, dtype=torch.float))[-1]).numpy()   # time-ordered
    sanity[name] = dict(cos_recon_mse_float64=mse64, cos_recon_mse_torch32=float(np.mean((r32 - ucos) ** 2)))
log("sanity cos(2pi 3 s), N=16, T=3 (recurrence recon MSE vs input over the 3 windows):", sanity)

all_res = {}
# main: x and z at 3 Lorenz t.u. per window, full N sweep
res_x, ex_x, U_x, quad_x = run_cond(trajs, TU_MAIN, "x", NS, mu, sd, keep_example=True)
res_z, _, _, _ = run_cond(trajs, TU_MAIN, "z", NS, mu, sd)
# time-scale sweep (x, N = 16)
res_sweep = {}
for tu in TU_SWEEP:
    if tu == TU_MAIN:
        res_sweep[tu] = res_x
    else:
        res_sweep[tu], _, _, _ = run_cond(trajs, tu, "x", [N_MAIN], mu, sd)

# ------------------------------------------------------------------ tables / json
def row(res, N, name, route, tf):
    r = res[(N, name, route, tf)]
    return summ(r["mse"]), [summ(r["win"][i]) for i in range(3)], summ(r["emag"].mean(0)), summ(r["pratio"].mean(0))


tab = {}
for cname, res in (("x", res_x), ("z", res_z)):
    for (N, name, route, tf), v in res.items():
        if route in ("rho", "lsqw"):
            continue
        tab[f"{cname}|N={N}|{name}|{route}|tf={tf}"] = dict(
            mse_mean_std=summ(v["mse"]), win_mse_mean=[float(np.mean(v["win"][i])) for i in range(3)],
            spec_mag_err_mean_per_window=[float(np.mean(v["emag"][i])) for i in range(3)],
            band_power_ratio_mean_per_window=[float(np.mean(v["pratio"][i])) for i in range(3)],
            n_nonfinite=int((~np.isfinite(v["mse"])).sum()))
for tu, res in res_sweep.items():
    for name in MODELS:
        for route in ROUTES:
            for tf in TF:
                tab[f"x|sweep_tuPerL={tu}|N={N_MAIN}|{name}|{route}|tf={tf}"] = dict(
                    mse_mean_std=summ(res[(N_MAIN, name, route, tf)]["mse"]))
rho_tab = {f"N={N}|{name}": res_x[(N, name, "rho", TF[0])] for N in NS for name in MODELS}
json.dump(dict(table=tab, rho=rho_tab, sanity=sanity), open(os.path.join(OUT, "results.json"), "w"), indent=1)

md_lines = []
def mdline(s=""): md_lines.append(s)
fmt = lambda a: f"{a[0]:.3g}±{a[1]:.2g}"
for cname, res in (("x", res_x), ("z", res_z)):
    for tf in TF:
        mdline(f"### {cname}(t), final time s={tf}: MSE over lag [0,3L) (mean±std over {NTRAJ} trajectories; signal var = 1, predict-0 MSE = 1)")
        mdline("| N | FouP rec | FouP ip | FouP lsq(oracle) | LegS rec | LegS ip | LegS lsq(oracle) | FouP A_inf rec | LegS A_inf rec |")
        mdline("|---|---|---|---|---|---|---|---|---|")
        for N in NS:
            g = lambda n, r: fmt(summ(res[(N, n, r, tf)]["mse"]))
            mdline(f"| {N} | {g('FouP','rec')} | {g('FouP','ip')} | {g('FouP','lsq')} | {g('LegS','rec')} | {g('LegS','ip')} | {g('LegS','lsq')} | {g('FouP_infty','rec')} | {g('LegS_infty','rec')} |")
        mdline()
for tf in (3.0, 6.0):
    mdline(f"### x(t), final time s={tf}: per-window MSE (windows = lag [0,1),[1,2),[2,3) L); lsqw = oracle fit of that window alone")
    mdline("| N | model | route | win1 | win2 | win3 |"); mdline("|---|---|---|---|---|---|")
    for N in NS:
        for name in ("FouP", "LegS", "FouP_infty"):
            for route in (("rec", "ip", "lsqw") if name != "FouP_infty" else ("rec",)):
                v = res_x[(N, name, route, tf)]["win"]
                mdline(f"| {N} | {name} | {route} | " + " | ".join(f"{np.mean(v[i]):.3g}" for i in range(3)) + " |")
    mdline()
mdline("### x(t), N=16, tf=6: spectral magnitude error (sum_m(|S^|-|S|)^2/sum_m|S|^2 over m<=N/2, Hann-windowed per-window FFT) and band power ratio P^/P, per window, recurrence route")
mdline("| model | win1 err | win2 err | win3 err | win1 P^/P | win2 | win3 |"); mdline("|---|---|---|---|---|---|---|")
for name in MODELS:
    v = res_x[(16, name, "rec", 6.0)]
    mdline(f"| {name} | " + " | ".join(f"{np.mean(v['emag'][i]):.3g}" for i in range(3)) + " | " + " | ".join(f"{np.mean(v['pratio'][i]):.3g}" for i in range(3)) + " |")
mdline()
mdline("### time-scale sweep (x, N=16, rec): MSE for Lorenz t.u. per window L")
mdline("| t.u. per L | tf | FouP rec | FouP ip | FouP lsq | LegS rec | LegS ip | LegS lsq |"); mdline("|---|---|---|---|---|---|---|---|")
for tu in TU_SWEEP:
    for tf in (3.0, 6.0):
        g = lambda n, r: fmt(summ(res_sweep[tu][(N_MAIN, n, r, tf)]["mse"]))
        mdline(f"| {tu} | {tf} | {g('FouP','rec')} | {g('FouP','ip')} | {g('FouP','lsq')} | {g('LegS','rec')} | {g('LegS','ip')} | {g('LegS','lsq')} |")
mdline()
mdline("### spectral radius rho(dA)")
mdline("| N | " + " | ".join(MODELS) + " |"); mdline("|---|" + "---|" * len(MODELS))
for N in NS:
    mdline(f"| {N} | " + " | ".join(f"{res_x[(N, n, 'rho', 3.0)]:.6f}" for n in MODELS) + " |")
open(os.path.join(OUT, "results_tables.md"), "w").write("\n".join(md_lines))
log("tables written")

# ------------------------------------------------------------------ measure omega(tau) and orthonormality check
# The measure is taken from the library object, not from a hand formula:
#   LagHiPPO.measure (lag_hippo.py:430) -> lambda x: self.sigma_deriv_func(0, -x)  [stored as model.measure_fn, line 354;
#   model.measure is the same function evaluated on the lag grid, line 408/433]
#   FouP: sigma_deriv_func = FrequencyPreservingBasis.sigma_deriv (lag_hippo_freq.py:53) = h_n / L, h_n=(1-alpha)alpha^(n-1)
#   LegS: sigma_deriv_func = lag_hippo.sigma_deriv (lag_hippo.py:21) = exp(s-t) -> exp(-tau)
# omega(tau) = dz/dtau, so  int_0^inf omega dtau = int_0^1 dz = 1 for both models: already on a common scale.
MEAS = {k: mods[k].measure_fn for k in ("FouP", "LegS")}
lag_fine = np.linspace(0.0, T_MEM, 30001)
for k in MEAS:   # the stored array must equal the function on the library lag grid
    assert np.allclose(mods[k].measure, MEAS[k](np.arange(NLAG) * DT))
hand = {"FouP": lambda t: ((1 - ALPHA) * ALPHA ** (np.floor(t / WL))) / WL, "LegS": lambda t: np.exp(-t)}
meas_vs_hand = {k: float(np.max(np.abs(MEAS[k](lag_fine) - hand[k](lag_fine)))) for k in MEAS}


def gram_with_measure(model, N, tau_max, h=0.01, order=16):
    edges = np.unique(np.round(np.concatenate([np.arange(0, tau_max, h), [tau_max], np.arange(0, tau_max, 1.0)]), 12))
    edges = edges[edges <= tau_max]
    x, w = roots_legendre(order)
    lo, hi = edges[:-1], edges[1:]
    tau = (lo[:, None] + (hi - lo)[:, None] * (x[None] + 1) / 2).ravel()
    wt = ((hi - lo)[:, None] * w[None] / 2).ravel() * model.measure_fn(tau)
    P = vectorized_basis(model.phi_func, np.arange(N), model.sigma_func(0, -tau))
    return np.einsum("i,ik,il->kl", wt, P, P)


ortho = {}
mods64 = build_models(64)
for N, mm_ in ((16, mods), (64, mods64)):
    for k, tmax in (("FouP", 11.0), ("LegS", 40.0)):
        G = gram_with_measure(mm_[k], N, tmax)
        Gt = gram_with_measure(mm_[k], N, T_MEM)
        ortho[f"{k}|N={N}"] = dict(max_abs_G_minus_I_full_range=float(np.max(np.abs(G - np.eye(N)))), tau_max=tmax,
                                  max_abs_G_minus_I_truncated_0_3L=float(np.max(np.abs(Gt - np.eye(N)))))
mass = {}
for k in MEAS:
    xq, wq = roots_legendre(64)
    ed = np.arange(0, 4)
    mass[k] = [float(sum(np.sum(MEAS[k](a + (b - a) * (xq + 1) / 2) * wq * (b - a) / 2) for a, b in [(e, e + 1)])) for e in range(3)]
meas_info = dict(measure_vs_hand_formula_max_abs_diff=meas_vs_hand, window_masses=mass,
                 mass_in_0_3L={k: float(sum(v)) for k, v in mass.items()}, orthonormality=ortho,
                 source=dict(function="LagHiPPO.measure_fn = sigma_deriv_func(0,-tau) (lag_hippo.py:354,430)",
                             FouP="FrequencyPreservingBasis.sigma_deriv (lag_hippo_freq.py:53)", LegS="lag_hippo.sigma_deriv (lag_hippo.py:21)"))
json.dump(meas_info, open(os.path.join(OUT, "measure_and_orthonormality.json"), "w"), indent=1)
log("measure / orthonormality:", json.dumps(meas_info, indent=1))


def measure_panel(ax, legend=True):
    ax.semilogy(-lag_fine, MEAS["FouP"](lag_fine), color=COL["FouP"], lw=1.6,
                label=r"FouP $\omega=h_n/L$ (library `sigma_deriv`): window masses " + "/".join(f"{m_:.3g}" for m_ in mass["FouP"]))
    ax.semilogy(-lag_fine, MEAS["LegS"](lag_fine), color=COL["LegS"], lw=1.6,
                label=r"LegS $\omega=e^{-\tau}$ (library `sigma_deriv`): masses " + "/".join(f"{m_:.3g}" for m_ in mass["LegS"]))
    for e in (1, 2):
        ax.axvline(-e, color="gray", ls=":", lw=0.8)
    for xc, tx in ((-0.5, "window 1\n(most recent)"), (-1.5, "window 2"), (-2.5, "window 3\n(oldest)")):
        ax.text(xc, 0.0095, tx, ha="center", va="bottom", fontsize=7, color="0.35")
    ax.set_xlim(-T_MEM, 0)
    ax.set_ylim(0.008, 3.0)
    ax.set_ylabel(r"$\omega(\tau)=\mathrm{d}z/\mathrm{d}\tau$")
    ax.set_title(r"measure from the library (normalised: $\int_0^\infty\omega\,\mathrm{d}\tau=1$ for both)", fontsize=8)
    if legend: ax.legend(fontsize=6.5, loc="upper left")


# ------------------------------------------------------------------ plots
git = lambda *a: subprocess.run(["git", "-C", REPO, *a], capture_output=True, text=True).stdout.strip()
json.dump(dict(timestamp=ts, git_commit=git("rev-parse", "HEAD"), git_branch=git("rev-parse", "--abbrev-ref", "HEAD"),
               git_dirty=bool(git("status", "--porcelain", "--untracked-files=no")), python=sys.version,
               hyper=dict(dt=DT, T_mem=T_MEM, alpha=ALPHA, L=WL, eps=EPS, N_main=N_MAIN, N_sweep=NS, final_times=TF,
                          lorenz_time_units_per_L=TU_MAIN, tu_sweep=TU_SWEEP, ntraj=NTRAJ,
                          lorenz=dict(sigma=10, rho=28, beta=8 / 3, solver="DOP853 rtol=atol=1e-12",
                                      transient_discarded_tu=50, normalisation="mean/std of 2000 t.u. attractor run",
                                      attractor_mean=mu.tolist(), attractor_std=sd.tolist()),
                          foup=dict(derive_A_from="gen (rec) / infty (flagged)", input_hold="zoh", a_tilt="translate",
                                    edge_kappa="2(1-alpha)/(1+alpha) default", quad_order="200*N"),
                          legs=dict(derive_A_from="gen / infty", a_tilt="legs", input_hold="delta (gen) / zoh (infty)"))),
          open(os.path.join(OUT, "run_config.json"), "w"), indent=1)

lag = np.arange(NLAG) * DT
TUL = lambda ax: ax.secondary_xaxis("top", functions=(lambda x: x * TU_MAIN, lambda x: x / TU_MAIN)).set_xlabel("lag [Lorenz t.u.]")

# Fig 0: the signal
fig, ax = plt.subplots(1, 2, figsize=(13, 3.6))
s6 = np.arange(K_TOT) * DT
ax[0].plot(s6, U_x[:, 0], "k", lw=0.8)
for e in range(1, 7):
    ax[0].axvline(e, color="gray", ls=":", lw=0.8)
ax[0].set(xlabel=r"model time $s$  (1 window $L$ = 3 Lorenz t.u.)", ylabel=r"$u=(x-\bar x)/\sigma_x$",
          title=r"Lorenz-63 $x(t)$, trajectory 0 (dotted: window edges $L$)")
P = np.mean([np.abs(np.fft.rfft(U_x[WIN * i:WIN * (i + 1), j] * HW)) ** 2 / HW.sum() ** 2 for i in range(6) for j in range(NTRAJ)], 0)
m = np.arange(0, 41)
ax[1].semilogy(m, P[:41], "k.-")
for N_, c_ in ((16, "tab:orange"), (64, "tab:green")):
    ax[1].axvline(N_ // 2, color=c_, ls="--", label=fr"FouP top mode $m=N/2$, $N={N_}$")
ax[1].set(xlabel=r"$m$ = cycles per window $L$ (= cycles per 3 t.u.)", ylabel=r"$|\hat u_m|^2$",
          title=r"power spectrum of one window, Hann (mean over 8 traj $\times$ 6 windows)")
ax[1].legend(fontsize=8)
fig.tight_layout(); save(fig, "0_lorenz_signal_and_spectrum.png")


def ex_plot(route, fname, title_route):
    fig = plt.figure(figsize=(14, 11.5))
    gs = fig.add_gridspec(4, 2, height_ratios=[3, 3, 3, 2.4], hspace=0.28)
    axs = np.array([[fig.add_subplot(gs[i, j]) for j in range(2)] for i in range(3)])
    for i, tf in enumerate(TF):
        for j, N in enumerate((16, 64)):
            ax = axs[i, j]
            ax.plot(-lag, ex_x[("H", tf)], "k", lw=1.0, label="true history")
            for name, c_ in (("FouP", "tab:red"), ("LegS", "tab:blue")):
                mse_ = res_x[(N, name, route, tf)]["mse"][0]
                ax.plot(-lag, ex_x[(N, name, route, tf)], color=c_, lw=1.0, label=f"{name} ({title_route}), MSE={mse_:.3g}")
            for e in (1, 2):
                ax.axvline(-e, color="gray", ls=":", lw=0.8)
            ax.set(title=fr"$N={N}$, final time $s={tf}$", ylim=(-3.2, 3.2), xlim=(-T_MEM, 0))
            ax.tick_params(labelbottom=False)
            if j == 0: ax.set_ylabel("u")
            ax.legend(fontsize=7, loc="upper left")
    for j in range(2):
        am = fig.add_subplot(gs[3, j], sharex=axs[2, j])
        measure_panel(am, legend=True)
        am.set_xlabel(r"time relative to now $-\tau$ [$L$ = 3 Lorenz t.u.];  0 = now, $-3L$ = oldest", fontsize=9)
    fig.suptitle(f"History reconstruction of Lorenz $x(t)$, trajectory 0: FouP vs LegS, {title_route} (bottom: each model's measure)", y=0.93)
    save(fig, fname)

ex_plot("rec", "1a_history_recurrence_with_measure_pastleft.png", "recurrence")
ex_plot("ip", "1b_history_inner_product_projection_with_measure_pastleft.png", "inner-product projection")
ex_plot("lsq", "1c_history_lsq_oracle_with_measure_pastleft.png", "lag-window least squares (oracle)")

# Fig 2: error vs lag
bl = -(np.arange(NLAG // BIN) + 0.5) * BIN * DT   # time relative to now (past on the left)
fig = plt.figure(figsize=(13, 10.5))
gs = fig.add_gridspec(3, 2, height_ratios=[3, 3, 2.2], hspace=0.22)
axs = np.array([[fig.add_subplot(gs[i, j]) for j in range(2)] for i in range(2)])
for ax, N in zip(axs.ravel(), NS):
    for name in ("FouP", "LegS"):
        for route, ls_, lab in (("rec", "-", "recurrence"), ("ip", "--", "inner-prod. proj."), ("lsq", ":", "lsq oracle")):
            v = np.mean([res_x[(N, name, route, tf)]["bins"] for tf in TF], axis=(0, 2))
            ax.semilogy(bl, v, ls_, color=COL[name], lw=1.4, label=f"{name} {lab}")
    for e in (1, 2):
        ax.axvline(-e, color="gray", ls=":", lw=0.8)
    ax.set(title=fr"$N={N}$", ylabel=r"squared error (bins of $0.05L$)")
    ax.legend(fontsize=7, ncol=2, loc="lower left")
    ax.set_xlim(-T_MEM, 0)
    ax.tick_params(labelbottom=False)
am = fig.add_subplot(gs[2, :], sharex=axs[1, 0])
measure_panel(am)
am.set_xlim(-T_MEM, 0)
am.set_xlabel(r"time relative to now, $-\tau$ [$L$ = 3 Lorenz t.u.]  (−3L = oldest, 0 = now; dotted = FouP window edges)")
fig.suptitle(r"Error vs lag (past on left), Lorenz $x(t)$: mean over 8 trajectories $\times$ final times $s\in\{3,4.5,6\}$ (bottom: measure $\omega(\tau)$)", y=0.93)
save(fig, "2_error_vs_lag_with_measure_pastleft.png")

fig, ax = plt.subplots(1, 2, figsize=(12, 4))
for k, (N, tf) in enumerate(((16, 6.0), (64, 6.0))):
    w = 0.2
    for q, (name, route) in enumerate((("FouP", "rec"), ("FouP", "ip"), ("LegS", "rec"), ("LegS", "ip"))):
        v = res_x[(N, name, route, tf)]["win"]
        ax[k].bar(np.arange(3) + (q - 1.5) * w, v.mean(1), w, yerr=v.std(1), color=COL[name], alpha=1.0 if route == "rec" else 0.45,
                  label=f"{name} {'recurrence' if route == 'rec' else 'inner-prod.'}", capsize=2)
    ax[k].set_yscale("log"); ax[k].set_xticks(range(3)); ax[k].set_xticklabels(["window 1\nlag [0,1)L", "window 2\n[1,2)L", "window 3\n[2,3)L"])
    ax[k].set(title=fr"$N={N}$, $s=6$", ylabel="MSE per window (mean±std, 8 traj)")
    ax[k].legend(fontsize=7)
fig.suptitle(r"Per-window MSE, Lorenz $x(t)$", y=1.02)
fig.tight_layout(); save(fig, "2b_mse_per_window.png")

# Fig 3: MSE vs N
def mse_vs_N_plot(res, ax, tf, comp):
    for name in ("FouP", "LegS"):
        for route, ls_, mk, lab in (("rec", "-", "o", "recurrence"), ("ip", "--", "s", "inner-prod. proj."), ("lsq", ":", "^", "lsq oracle")):
            v = np.array([gstat(res[(N, name, route, tf)]["mse"]) for N in NS])
            ax.errorbar(NS, v[:, 0], [v[:, 0] - v[:, 1], v[:, 2] - v[:, 0]], color=COL[name], ls=ls_, marker=mk, ms=4, capsize=2, lw=1.3, label=f"{name} {lab}")
        v = np.array([summ(res[(N, name + "_infty", "rec", tf)]["mse"]) for N in NS])
        vmax = float(np.nanmax(v[:, 0]))
        ax.plot(NS, np.minimum(v[:, 0], 20.0), color="0.45" if name == "FouP" else "0.7", ls="-.", marker="x", lw=1.0,
                label=("FouP" if name == "FouP" else "LegS") + r" $A_\infty$ route, recurrence"
                + (fr" ($\rho>1$, diverges: MSE up to {vmax:.0e}, clipped at 20)" if vmax > 20 else ""))
    ax.axhline(1.0, color="k", lw=0.6, ls=":"); ax.text(NS[0], 1.05, "predict 0 (MSE=1)", fontsize=7)
    ax.set(xscale="log", yscale="log", xlabel="N", ylabel="MSE over lag [0, 3L)", xticks=NS, title=fr"{comp}(t), final time $s={tf}$")
    ax.set_xticklabels(NS); ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter()); ax.set_ylim(1e-3, 30)
for res_, comp_, fname_, ttl_ in ((res_x, "Lorenz x", "3_mse_vs_N_x.png", "Reconstruction MSE vs N, Lorenz x(t)"),
                                  (res_z, "Lorenz z", "3b_mse_vs_N_z.png", "Reconstruction MSE vs N, secondary signal z(t)")):
    fig, axs = plt.subplots(1, 3, figsize=(17, 5.2), sharey=True)
    for ax, tf in zip(axs, TF):
        mse_vs_N_plot(res_, ax, tf, comp_)
    h_, l_ = axs[0].get_legend_handles_labels()
    fig.legend(h_, l_, loc="lower center", ncol=3, fontsize=8, bbox_to_anchor=(0.5, -0.12))
    fig.suptitle(ttl_ + " (geometric mean over 8 trajectories, bars = ±1 s.d. of log MSE)", y=1.02)
    fig.tight_layout(); save(fig, fname_)

# Fig 3c: per-window MSE vs N
fig, axs = plt.subplots(2, 3, figsize=(16, 8), sharey=True)
for r_, tf in enumerate((3.0, 6.0)):
    for n in range(3):
        ax = axs[r_, n]
        for name in ("FouP", "LegS"):
            for route, ls_, mk, lab in (("rec", "-", "o", "recurrence"), ("ip", "--", "s", "inner-prod. proj."), ("lsqw", ":", "^", "oracle fit of this window alone")):
                v = np.array([gstat(res_x[(N, name, route, tf)]["win"][n]) for N in NS])
                ax.errorbar(NS, v[:, 0], [v[:, 0] - v[:, 1], v[:, 2] - v[:, 0]], color=COL[name], ls=ls_, marker=mk, ms=4, capsize=2, lw=1.3, label=f"{name} {lab}")
        ax.axhline(1.0, color="k", lw=0.6, ls=":")
        ax.set(xscale="log", yscale="log", xticks=NS, title=fr"window {n + 1} (lag [{n},{n + 1})$L$), final time $s={tf}$", xlabel="N", ylabel="MSE in this window")
        ax.set_xticklabels(NS); ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        if r_ == 0 and n == 0: ax.legend(fontsize=6.5, loc="lower left")
fig.suptitle(r"Per-window reconstruction MSE vs N, Lorenz $x(t)$ (geometric mean over 8 traj, bars ±1 s.d. of log; dotted black: predict 0)", y=1.0)
fig.tight_layout(); save(fig, "3c_mse_vs_N_per_window_x.png")

# Fig 4: spectra
mm = np.arange(0, 41)
fig, axs = plt.subplots(2, 3, figsize=(15, 7.5), sharey=True)
for i, N in enumerate((16, 64)):
    for n in range(3):
        ax = axs[i, n]
        Ptrue = np.mean([quad_x[tf][4][n][:41] ** 2 for tf in TF], axis=(0, 2))
        ax.semilogy(mm, Ptrue, "k.-", lw=1.2, label="true history")
        for name in ("FouP", "LegS"):
            Pr = np.mean([res_x[(N, name, "rec", tf)]["Sr"][n][:41] ** 2 for tf in TF], axis=(0, 2))
            ax.semilogy(mm, Pr, "o-", ms=3, color=COL[name], lw=1.2, label=f"{name} recurrence")
        Pi = np.mean([res_x[(N, "FouP", "ip", tf)]["Sr"][n][:41] ** 2 for tf in TF], axis=(0, 2))
        ax.semilogy(mm, Pi, "--", color="darkred", lw=0.9, label="FouP inner-prod. proj.")
        ax.axvline(N // 2, color="tab:orange", ls="--", lw=1.0, label=r"FouP top mode $m=N/2$")
        ax.set(title=fr"$N={N}$, window {n + 1} (lag [{n},{n + 1})$L$)", ylim=(1e-9, 3))
        if i == 1: ax.set_xlabel(r"$m$ = cycles per window $L$ (= per 3 Lorenz t.u.)")
        if n == 0: ax.set_ylabel(r"power $|\hat u_m|^2$ (Hann)")
        ax.legend(fontsize=7)
fig.suptitle(r"Spectrum of reconstructed vs true recent history, Lorenz $x(t)$ (mean over 8 traj $\times$ final times)", y=1.0)
fig.tight_layout(); save(fig, "4_spectrum_comparison.png")

# Fig 5: time-scale sweep
fig, ax = plt.subplots(1, 2, figsize=(12, 4.2), sharey=True)
for a, tf in zip(ax, (3.0, 6.0)):
    for name in ("FouP", "LegS"):
        for route, ls_, mk in (("rec", "-", "o"), ("ip", "--", "s"), ("lsq", ":", "^")):
            v = np.array([gstat(res_sweep[tu][(N_MAIN, name, route, tf)]["mse"]) for tu in TU_SWEEP])
            a.errorbar(TU_SWEEP, v[:, 0], [v[:, 0] - v[:, 1], v[:, 2] - v[:, 0]], color=COL[name], ls=ls_, marker=mk, capsize=2, ms=4, label=f"{name} {route}")
    a.set(xscale="log", yscale="log", xticks=TU_SWEEP, xlabel="Lorenz time units per window $L$ (more = more oscillations per $L$)",
          title=fr"$N=16$, final time $s={tf}$"); a.set_xticklabels(TU_SWEEP); a.legend(fontsize=7)
    a.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter()); a.set_ylim(0.1, 2.0)
    a.set_yticks([0.1, 0.2, 0.5, 1.0, 2.0]); a.set_yticklabels(["0.1", "0.2", "0.5", "1", "2"]); a.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
ax[0].set_ylabel("MSE over lag [0, 3L)")
fig.suptitle(r"Sensitivity to the Lorenz time scale, $x(t)$", y=1.02)
fig.tight_layout(); save(fig, "5_timescale_sweep.png")

log("DONE", OUT)
