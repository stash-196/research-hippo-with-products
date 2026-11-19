"""Step 0: make LagOp-LegT / LagOp-FouT match HiPPO (Gu) in our own copy.
Before = repo path (make_legt_model / make_fout_model defaults: A_gen with module-level
LegS sigma_inv', tilt 'translate' A=-A_gen, B_delta, bilinear).
After  = warp-correct A_gen (compute_A_gen_for_basis) + rank-one edge term:
         A = A_gen - kappa phi(1)phi(1)^T ,  B = kappa phi(1) ; kappa=1 (LegT), 2 (FouT)."""
from lag_op_hippo.window_checks.common import *
from scipy import signal as sps
import torch
from lag_op_hippo.lag_hippo import LagHiPPO, phi as legs_phi, phi_deriv as legs_dphi, sigma as legs_sigma, \
    sigma_deriv as legs_sd, sigma_inv as legs_si, sigma_inv_deriv as legs_sid, compute_A_gen

T = 3.0
s, sig = signals(T)
u = sig["cos"]
FAM = {"legt": (make_legt_basis, make_legt_model, 1.0), "fourier": (make_fout_basis, make_fout_model, 2.0)}
LBL = {"legt": "LegT", "fourier": "FouT"}

def gu(method, N):
    A, B = transition(method, N); B = B[:, 0]
    dA, dB = bilinear(A, B)
    return A, B, dA, dB[:, 0]

def lag_pair(method, N):
    mkb, mkm, kappa = FAM[method]
    b = mkb(window_length=1.0)
    with quiet():
        m = mkm(N=N, dt=DT, T=T, window_length=1.0)
    before = dict(A=np.array(m.A, float), B=np.array(m.B, float), dA=m.dA.numpy().astype(float), dB=m.dB.numpy().astype(float))
    Ag = compute_A_gen_for_basis(N, b.phi, b.phi_deriv, b.sigma_inv_deriv, quad_order=max(200 * N, 64))
    p1 = np.array([float(b.phi(k, np.array([1.0]))[0]) for k in range(N)])
    A = Ag - kappa * np.outer(p1, p1); B = kappa * p1
    dA, dB = bilinear(A, B)
    after = dict(A=A, B=B, dA=dA, dB=dB[:, 0], A_gen=Ag, phi1=p1)
    E = np.array(m.eval_matrix, float)  # LagOp eval matrix (lookback rows), float64 copy
    return b, m, before, after, E

res, store = {}, {}
for method in ("legt", "fourier"):
    N = 16
    A_g, B_g, dA_g, dB_g = gu(method, N)
    b, m, bef, aft, E = lag_pair(method, N)
    with quiet():
        hp = HiPPO(method=method, N=N, dt=DT, T=T)
    Eh = np.array(hp.eval_matrix, float)
    out = {"E_lag_vs_E_hippo": rel_fro(E, Eh)}
    ch = scan(dA_g, dB_g, u)[-1]; rh = np.flip(Eh @ ch)
    out["hippo_float64"] = dict(mse_vs_input=mse(rh, u))
    recs = {"HiPPO (Gu), float64": rh}
    for tag, P in (("before (repo path)", bef), ("after (fixed)", aft)):
        c = scan(P["dA"], P["dB"], u)[-1]; r = np.flip(E @ c)
        out[tag] = dict(rel_A_vs_Gu=rel_fro(P["A"], A_g), rel_B_vs_Gu=rel_fro(P["B"], B_g),
                        rel_dA_vs_HiPPO=rel_fro(P["dA"], dA_g), rel_dB_vs_HiPPO=rel_fro(P["dB"], dB_g),
                        rec_mse_vs_hippo=mse(r, rh), rec_rel_vs_hippo=float(np.linalg.norm(r - rh) / np.linalg.norm(rh)),
                        rec_mse_vs_input=mse(r, u), max_re_eig_A=float(np.max(np.linalg.eigvals(P["A"]).real)),
                        rho_dA=float(np.max(np.abs(np.linalg.eigvals(P["dA"])))))
        recs["LagOp " + tag] = r
    # repo float32 torch pipeline with fixed buffers (same code path as check 02.02)
    m.A = aft["A"]; m.dA = torch.tensor(aft["dA"], dtype=torch.float); m.dB = torch.tensor(aft["dB"], dtype=torch.float)
    ut = torch.tensor(u, dtype=torch.float)
    rl32 = m.reconstruct(m(ut)[-1]).numpy(); rh32 = hp.reconstruct(hp(ut)[-1]).numpy()
    out["after (fixed), repo float32 torch pipeline"] = dict(rec_mse_vs_hippo=mse(rl32, rh32), rec_mse_vs_input=mse(rl32, u),
                                                            hippo_mse_vs_input=mse(rh32, u))
    out["rank1_identity_check"] = dict(
        A_gen_fixed_minus_kappa_p1p1T_vs_Gu=rel_fro(aft["A"], A_g),
        repo_A_gen_LegSbiased_vs_fixed=rel_fro(-bef["A"], aft["A_gen"]))
    res[LBL[method]] = out
    store[method] = (A_g, bef, aft, recs)
    print(LBL[method], json.dumps(out, indent=1))

# LegS reference (repo path already exact)
with quiet():
    ml = LagHiPPO(N=16, phi_func=legs_phi, phi_deriv_func=legs_dphi, sigma_func=legs_sigma, sigma_deriv_func=legs_sd,
                  sigma_inv_func=legs_si, sigma_inv_deriv_func=legs_sid, dt=DT, T=T)
A_g, B_g, dA_g, dB_g = gu("legs", 16)
res["LegS (repo path, reference)"] = dict(rel_A_vs_Gu=rel_fro(np.array(ml.A), A_g), rel_dA_vs_HiPPO=rel_fro(ml.dA.numpy().astype(float), dA_g))

# ---------------- domain expansion (paper Fig.3 style), N=64, one-hot e_{N-2} -------------
N64 = 64; nidx = N64 - 2; ZOOM = 10
dex = {}
for method in ("legt", "fourier"):
    A_g, B_g, dA_g, dB_g = gu(method, N64)
    b, m, bef, aft, E = lag_pair(method, N64)
    e = np.zeros(N64); e[nidx] = 1.0
    tau = np.arange(0, T, DT)
    def recon(cvec):
        return E @ cvec  # lookback order, present first
    # projected GT under the shifted warp: phi_n(sigma_{t+dt}(s)) == mode shifted by one step in lookback
    gt_now = E[:, nidx]
    gt_fwd = np.concatenate([[np.nan], gt_now[:-1]])  # forward one step: lookback tau -> tau+dt
    W = {"HiPPO dA": dA_g, "before dA": bef["dA"], "after dA": aft["dA"]}
    wf = {k: recon(M @ e) for k, M in W.items()}
    wb = {k: recon(np.linalg.solve(M, e)) for k, M in W.items()}
    d = {}
    for tag in ("before", "after"):
        d[tag] = dict(rel_dA=rel_fro(W[tag + " dA"], dA_g),
                      rel_forward_waveform_vs_hippo=rel_fro(wf[tag + " dA"], wf["HiPPO dA"]),
                      rel_backward_waveform_vs_hippo=rel_fro(wb[tag + " dA"], wb["HiPPO dA"]))
    dex[LBL[method]] = d
    store["dex_" + method] = (tau, gt_now, gt_fwd, wf, wb)
res["domain_expansion_N64_mode62"] = dex
print(json.dumps(dex, indent=1))
# --- LegT residual: Gu eval matrix zeroes the tau=0 and tau=L rows (heaviside(x,0)); LagOp keeps them
A_g, B_g, dA_g, dB_g = gu("legt", 16)
b, m, bef, aft, E = lag_pair("legt", 16)
with quiet(): hp = HiPPO(method="legt", N=16, dt=DT, T=T)
Eh = np.array(hp.eval_matrix, float)
diff_rows = np.where(np.abs(E - Eh).max(1) > 1e-4)[0]
ch = scan(dA_g, dB_g, u)[-1]; cl = scan(aft["dA"], aft["dB"], u)[-1]
res["LegT_eval_endpoint_note"] = dict(tau_rows_where_eval_differs=(diff_rows * DT).tolist(),
    coef_rel_cT_after_vs_hippo=rel_fro(cl, ch), rec_mse_vs_hippo_common_E=mse(Eh @ cl, Eh @ ch),
    rec_mse_vs_hippo_excluding_endpoint_rows=mse(np.delete(E @ cl, diff_rows), np.delete(Eh @ ch, diff_rows)))
# --- library cross-check: lag_hippo_edge_fix.make_translate_model_edge_fixed == inline fix
from lag_op_hippo.lag_hippo_edge_fix import make_translate_model_edge_fixed
for method in ("legt", "fourier"):
    with quiet(): mf = make_translate_model_edge_fixed(method, N=16, dt=DT, T=T)
    A_g, B_g, dA_g, dB_g = gu(method, 16)
    res["library_" + LBL[method]] = dict(rel_A_vs_Gu=rel_fro(mf.A, A_g), rel_dA64_vs_HiPPO=rel_fro(mf.dA64, dA_g), rel_B_vs_Gu=rel_fro(mf.B, B_g))
print(json.dumps({k: res[k] for k in res if k.startswith("library") or k.startswith("LegT_eval")}, indent=1, default=str))
dump(res, "S0_legt_fout_fix_results.json")

# ---------------- plots ----------------
fig, axes = plt.subplots(2, 2, figsize=(13, 8))
for i, method in enumerate(("legt", "fourier")):
    A_g, bef, aft, recs = store[method]
    ax = axes[i, 0]
    ax.plot(s, u, color="0.6", lw=3, label=r"input $u(s)=\cos(6\pi s)$")
    ax.plot(s, recs["HiPPO (Gu), float64"], "b--", lw=1.5, label="HiPPO-" + LBL[method])
    ax.plot(s, np.clip(recs["LagOp before (repo path)"], -3, 3), "r-", lw=1, label="LagOp before (repo path)")
    ax.plot(s, recs["LagOp after (fixed)"], "g:", lw=2, label="LagOp after (fixed)")
    ax.set_ylim(-3, 3); ax.set_xlabel(r"$s$"); ax.set_title(rf"{LBL[method]}: $c_T$ from recurrence $\to\hat u$ ($N=16$, $\Delta t=10^{{-3}}$)")
    ax.legend(fontsize=7, loc="upper left")
    ax = axes[i, 1]
    ax.semilogy(s, np.abs(recs["LagOp before (repo path)"] - recs["HiPPO (Gu), float64"]) + 1e-17, "r", lw=0.8, label="before: |LagOp - HiPPO|")
    ax.semilogy(s, np.abs(recs["LagOp after (fixed)"] - recs["HiPPO (Gu), float64"]) + 1e-17, "g", lw=0.8, label="after: |LagOp - HiPPO|")
    ax.set_xlabel(r"$s$"); ax.set_title(r"pointwise $|\hat u_{\mathrm{LagOp}}-\hat u_{\mathrm{HiPPO}}|$"); ax.legend(fontsize=8)
fig.suptitle("Step 0: LagOp vs HiPPO recurrence (check 02.02 analogue), before vs after rank-one edge fix", y=1.0)
fig.tight_layout(); save(fig, "S0_1_recurrence_vs_hippo_before_after.png")

fig, axes = plt.subplots(2, 3, figsize=(14, 8))
for i, method in enumerate(("legt", "fourier")):
    A_g, bef, aft, recs = store[method]
    for j, (M, t) in enumerate(((A_g, "Gu $A$"), (bef["A"] - A_g, r"before: $A_{\mathrm{LagOp}}-A_{\mathrm{Gu}}$"), (aft["A"] - A_g, r"after: $A_{\mathrm{LagOp}}-A_{\mathrm{Gu}}$"))):
        ax = axes[i, j]; vm = np.max(np.abs(M)) + 1e-300
        im = ax.imshow(M, cmap="RdBu_r", vmin=-vm, vmax=vm); ax.grid(False)
        ax.set_title(f"{LBL[method]}: {t}\nmax abs = {vm:.2e}", fontsize=9); fig.colorbar(im, ax=ax, fraction=0.046)
fig.suptitle(r"Step 0: continuous $A$ ($N=16$). Fix: $A=A_{\mathrm{gen}}-\kappa\,\phi(1)\phi(1)^\top$, $\kappa_{\mathrm{LegT}}=1$, $\kappa_{\mathrm{FouT}}=2$", y=1.0)
fig.tight_layout(); save(fig, "S0_2_A_matrix_diff_before_after.png")

fig, axes = plt.subplots(2, 2, figsize=(13, 8))
for i, method in enumerate(("legt", "fourier")):
    tau, gt_now, gt_fwd, wf, wb = store["dex_" + method]
    for j, (w, nm) in enumerate(((wf, "forward  $\\mathrm{d}A\\,e_n$"), (wb, "backward $\\mathrm{d}A^{-1}e_n$"))):
        ax = axes[i, j]; sl = slice(0, 4 * ZOOM)
        ax.plot(tau[sl], gt_now[sl], color="0.7", lw=4, label=r"unshifted mode $\phi_n(\sigma_t)$")
        ax.plot(tau[sl], w["HiPPO dA"][sl], "b-", lw=2, label="HiPPO")
        ax.plot(tau[sl], w["before dA"][sl], "r--", lw=1.2, label="LagOp before")
        ax.plot(tau[sl], w["after dA"][sl], "g:", lw=2.5, label="LagOp after")
        ax.set_xlabel(r"lookback $\tau$ (first 40 samples)"); ax.set_title(f"{LBL[method]} {nm}, $N=64$, $n=62$")
        ax.legend(fontsize=7)
fig.suptitle("Step 0: domain expansion (02_domain_expansion analogue), one-hot mode, zoom near present", y=1.0)
fig.tight_layout(); save(fig, "S0_3_domain_expansion_before_after.png")
