"""Experiment C: eigen-structure of continuous A and discrete dA for every SSM in A/B,
plus LegS/LegT/FouT baselines.  All systems here are LTI (no time-varying A), so the
'two sides of the boundary' comparison is done across window blocks / windows."""
from lag_op_hippo.window_checks.common import *
from scipy.linalg import logm
from lag_op_hippo.lag_hippo import LagHiPPO, phi as legs_phi, phi_deriv as legs_dphi, sigma as legs_sigma, \
    sigma_deriv as legs_sd, sigma_inv as legs_si, sigma_inv_deriv as legs_sid
from lag_op_hippo.lag_hippo_edge_fix import make_translate_model_edge_fixed, make_frequency_preserving_basis_fixed, foup_edge_kappa
fb = make_fout_basis(window_length=L)
QOA = 200 * N

fp = make_frequency_preserving_basis(alpha=ALPHA, window_length=L, eps=1e-8)
gz = g0(); D = Dmat(); G1 = np.outer(gz, gz)
tau1 = np.arange(0, L, DT)

def ev_fun_single(V, M_extent=1):
    tau = np.arange(0, M_extent * L, DT)
    return tau, gmat(np.mod(tau / L, 1)) @ V

def ev_fun_block(V, M):
    tau = np.arange(0, M * L, DT); out = np.zeros((tau.size, V.shape[1]), complex)
    for n in range(1, M + 1):
        m = (tau >= (n - 1) * L) & (tau < n * L)
        out[m] = gmat(np.mod(tau[m] / L, 1)) @ V[(n - 1) * N:n * N] / np.sqrt(h(n))
    return tau, out

def entry(name, A, dA, evfun):
    lam = np.linalg.eigvals(A) if A is not None else np.log(np.linalg.eigvals(dA).astype(complex)) / DT
    mu, V = np.linalg.eig(dA)
    tau, Fv = evfun(V)
    nrm = np.sqrt(np.mean(np.abs(Fv) ** 2, 0))
    ok = np.where(nrm > 1e-6)[0]                      # drop null modes (e.g. FouT sin(0) slot)
    order = ok[np.argsort(-np.abs(mu[ok]))][:3]
    return dict(name=name, lam=lam, mu=mu, tau=tau, F=Fv[:, order] / nrm[order], mu_lead=mu[order],
                A_from_logm=A is None,
                stats=dict(max_re_lambda=float(np.max(lam.real)), rho_dA=float(np.max(np.abs(mu))),
                           n_unstable_dA=int(np.sum(np.abs(mu) > 1 + 1e-12)),
                           n_on_circle=int(np.sum(np.abs(np.abs(mu) - 1) <= 1e-12)),
                           leading_mu=[[float(m.real), float(m.imag)] for m in mu[order]]))

groups = {}
# ---- group 1: single windows n=1..4 -------------------------------------------------
with quiet():
    repo_fout = make_fout_model(N=N, dt=DT, T=L, window_length=L)
g1 = []
for n in (1, 2, 3, 4):
    A = (D - 2 * G1) / L
    g1.append(entry(f"window {n}: fixed Gu-FouT block $(D-2g_0g_0^\\top)/L$", A, bilinear(A, gz)[0], lambda V: ev_fun_single(V)))
    g1.append(entry(f"window {n}: repo LagOp-FouT path ($-A_{{gen}}$, LegS-biased)", np.array(repo_fout.A, float), repo_fout.dA.numpy().astype(float), lambda V: ev_fun_single(V)))
    phi, dphi = repo_psi_local(n, 1.0, fp)
    Ag = compute_A_gen_for_basis(N, phi, dphi, fb.sigma_inv_deriv, quad_order=QOA)
    p1 = np.array([phi(k, 1 - 1e-9) for k in range(N)]).ravel()
    A2 = (Ag - 2 * np.outer(p1, p1)) / L
    Erp = lambda V, phi=phi: (tau1, np.stack([phi(k, 1 - tau1 / L) for k in range(N)], 1) @ V)
    g1.append(entry(f"window {n}: repo FouP $\\psi$ (amp $1/\\sqrt{{h_1}}$) + rank-1", A2, bilinear(A2, p1)[0], Erp))
groups["C1_single_windows"] = g1
# ---- group 2: two-window, full FouP, 3-window cascade -------------------------------
g2 = []
for rn, (A, B, kind) in block_routes(2).items():
    if kind == "taps":
        continue
    g2.append(entry("2-window " + rn, A, bilinear(A, B)[0], lambda V: ev_fun_block(V, 2)))
Hh = h(1) + h(2)
A = (D - (h(1) / Hh) * G1) / L
g2.append(entry("2-window single $N$-state naive closure", A, bilinear(A, gz)[0], lambda V: ev_fun_single(V, 2)))
with quiet():
    m_gen = make_frequency_preserving_model(N=N, dt=DT, T=3.0, alpha=ALPHA, window_length=L, eps=1e-8, derive_A_from="gen", input_hold="zoh", quad_order=200 * N)
    m_inf = make_frequency_preserving_model(N=N, dt=DT, T=3.0, alpha=ALPHA, window_length=L, eps=1e-8, derive_A_from="infty", input_hold="zoh", quad_order=200 * N)
Erep = lambda V: (np.arange(0, 3.0, DT), np.array(m_gen.eval_matrix, float) @ V)
g2.append(entry("full FouP repo gen/legs-tilt (Jul31 path)", np.array(m_gen.A, float), m_gen.dA.numpy().astype(float), Erep))
g2.append(entry("full FouP repo infty $\\mathrm{inv}(A_\\infty^\\top)e^{-\\Delta t}$ (NaN route)", None, m_inf.dA.numpy().astype(float), Erep))
kap = foup_edge_kappa(ALPHA)
A = (D - kap * G1) / L
g2.append(entry(f"full FouP fixed, edge closure $\\kappa={kap:.3f}$", A, bilinear(A, gz)[0], lambda V: ev_fun_single(V, 3)))
for rn, (A, B, kind) in block_routes(3).items():
    if "chain" in rn:
        g2.append(entry("3-window " + rn, A, bilinear(A, B)[0], lambda V: ev_fun_block(V, 3)))
groups["C2_two_window_and_full_foup"] = g2
# ---- group 3: baselines -----------------------------------------------------------
g3 = []
for method, lbl in (("legs", "LegS"), ("legt", "LegT"), ("fourier", "FouT")):
    with quiet():
        hp = HiPPO(method=method, N=N, dt=DT, T=1.0)
    Eh = lambda V, hp=hp: (np.arange(0, 1.0, DT), np.array(hp.eval_matrix, float) @ V)
    A_g = np.array(hp.A, float); g3.append(entry(f"HiPPO-{lbl} (Gu)", A_g, bilinear(A_g, np.ones(N))[0], Eh))
    if method == "legs":
        with quiet():
            ml = LagHiPPO(N=N, phi_func=legs_phi, phi_deriv_func=legs_dphi, sigma_func=legs_sigma, sigma_deriv_func=legs_sd,
                          sigma_inv_func=legs_si, sigma_inv_deriv_func=legs_sid, dt=DT, T=1.0)
        g3.append(entry("LagOp-LegS (repo)", np.array(ml.A, float), ml.dA.numpy().astype(float), Eh))
    else:
        with quiet():
            mr = (make_legt_model if method == "legt" else make_fout_model)(N=N, dt=DT, T=1.0, window_length=L)
            mf = make_translate_model_edge_fixed(method, N=N, dt=DT, T=1.0, window_length=L)
        g3.append(entry(f"LagOp-{lbl} repo path", np.array(mr.A, float), mr.dA.numpy().astype(float), Eh))
        g3.append(entry(f"LagOp-{lbl} edge-fixed", mf.A, mf.dA64, Eh))
groups["C3_baselines"] = g3

# ---------------- plots + metrics -------------------------------------------------
metrics = {}
th = np.linspace(0, 2 * np.pi, 2000)
for gname, ents in groups.items():
    nr = len(ents)
    fig, axes = plt.subplots(nr, 3, figsize=(15, 2.7 * nr), squeeze=False)
    for i, e in enumerate(ents):
        ax = axes[i, 0]
        ax.scatter(e["lam"].real, e["lam"].imag, s=14, c="tab:red")
        ax.axvline(0, color="k", lw=0.8)
        ax.set_title(("$\\lambda=\\log\\mu/\\Delta t$ " if e["A_from_logm"] else "continuous $A$: ") + e["name"], fontsize=7.5)
        ax.set_xlabel(r"$\mathrm{Re}\,\lambda$", fontsize=8); ax.set_ylabel(r"$\mathrm{Im}\,\lambda$", fontsize=8)
        ax = axes[i, 1]
        mu = e["mu"]
        ax.plot(np.cos(th), np.sin(th), "k--", lw=0.8)
        ax.scatter(mu.real, mu.imag, s=14, c="tab:blue")
        x0, x1 = min(mu.real.min(), 0.99) - 0.002, max(mu.real.max(), 1.0) + 0.002
        y = max(np.abs(mu.imag).max(), 0.01) * 1.1
        ax.set_xlim(x0, x1); ax.set_ylim(-y, y)
        ax.set_title(rf"$\mathrm{{d}}A$ (bilinear), zoom near $\mu=1$; $\rho={e['stats']['rho_dA']:.6f}$", fontsize=7.5)
        ax.set_xlabel(r"$\mathrm{Re}\,\mu$", fontsize=8); ax.set_ylabel(r"$\mathrm{Im}\,\mu$", fontsize=8)
        ax = axes[i, 2]
        for j in range(e["F"].shape[1]):
            ax.plot(e["tau"], np.abs(e["F"][:, j]), lw=1, label=rf"$|v_{j}(\tau)|$, $|\mu|={abs(e['mu_lead'][j]):.6f}$")
            ax.plot(e["tau"], e["F"][:, j].real, lw=0.6, ls=":", color=ax.lines[-1].get_color())
        ax.set_xlabel(r"lookback $\tau$", fontsize=8); ax.set_title("leading eigenvectors of dA as functions (solid |v|, dotted Re v)", fontsize=7.5)
        ax.legend(fontsize=6)
        metrics[f"{gname} | {e['name']}"] = e["stats"]
    fig.tight_layout(); save(fig, f"{gname}_eigen.png")

# spectral shift across windows (group 1): eigenvalue sets relative to window 1
shift = {}
for tag in ("fixed Gu-FouT block", "repo LagOp-FouT path", "repo FouP $\\psi$"):
    es = [e for e in groups["C1_single_windows"] if tag in e["name"]]
    ref = np.sort_complex(es[0]["lam"])
    shift[tag] = [rel_fro(np.sort_complex(x["lam"]), ref) for x in es]
metrics["spectral_shift_vs_window1_(rel fro of sorted lambda)"] = shift
dump(metrics, "C_eigen_metrics.json")
print(json.dumps(shift, indent=1))
