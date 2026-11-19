"""Experiment B: two-window system (windows 1+2) and full FouP (single state + block cascade)."""
from lag_op_hippo.window_checks.common import *
import torch
from lag_op_hippo.lag_hippo import LagHiPPO

fp = make_frequency_preserving_basis(alpha=ALPHA, window_length=L, eps=1e-8)
gz = g0(); D = Dmat(); G1 = np.outer(gz, gz)
QO = 200 * N
T = 3.0
s, sig = signals(T); K = s.size
h1, h2, h3 = h(1), h(2), h(3)

def fixed_phi(k, z):
    tau = -fp.sigma_inv(0.0, z)
    return gmat(np.mod(tau / L, 1.0))[:, k] if np.ndim(z) else gmat(np.array([np.mod(tau / L, 1.0)]))[0, k]
def fixed_dphi(k, z):
    z = np.asarray(z, float)
    tau = -fp.sigma_inv(0.0, z)
    dtau_dz = -fp.sigma_inv_deriv(0.0, z)
    return gprime(np.atleast_1d(np.mod(tau / L, 1.0)))[:, k] / L * dtau_dz

class Trunc:
    """FouP warp truncated to the first M windows, rescaled to z' in [0,1] (dz' measure)."""
    def __init__(self, M, base_phi, base_dphi, scale=1.0):
        self.M, self.zlo, self.H = M, ALPHA ** M, 1 - ALPHA ** M
        self.bp, self.bd, self.sc = base_phi, base_dphi, scale
    def _z(self, zp): return self.zlo + self.H * np.clip(np.asarray(zp, float), 0, 1)
    def sigma(self, t, s):
        z = fp.sigma(t, s); return np.clip((z - self.zlo) / self.H, 0, 1)
    def sigma_deriv(self, t, s):
        tau = np.asarray(t, float) - np.asarray(s, float)
        out = fp.sigma_deriv(t, s) / self.H
        return np.where((tau >= 0) & (tau < self.M * L), out, 0.0)
    def sigma_inv(self, t, zp): return fp.sigma_inv(t, self._z(zp))
    def sigma_inv_deriv(self, t, zp): return fp.sigma_inv_deriv(t, self._z(zp)) * self.H
    def phi(self, k, zp): return self.sc * self.bp(k, self._z(np.clip(zp, 1e-12, 1 - 1e-12)))
    def phi_deriv(self, k, zp): return self.sc * self.H * self.bd(k, self._z(np.clip(zp, 1e-12, 1 - 1e-12)))

results = {"B_two_window": {}, "F_full_foup": {}}
curves = {}
# =========================== B: two-window system =================================
M = 2; Hh = h1 + h2
tr_fixed = Trunc(2, fixed_phi, fixed_dphi, scale=1.0)
A_gen2 = compute_A_gen_for_basis(N, tr_fixed.phi, tr_fixed.phi_deriv, tr_fixed.sigma_inv_deriv, quad_order=QO)
results["B_two_window"]["A_gen_fixed_2win_vs_D"] = rel_fro(A_gen2, D)
for sname, u in sig.items():
    tau_all = np.arange(K) * DT
    P = proj_block(u, M)
    Psingle = (h1 * F_window(u, 1) + h2 * F_window(u, 2)) / np.sqrt(Hh)
    out = {}
    for rn, (A, B, kind) in block_routes(M).items():
        dA, dB = bilinear(A, B)
        C = run(dA, dB, taps(u, M) if kind == "taps" else u)
        e_tot = np.linalg.norm(C - P, axis=1)
        e_w2 = np.linalg.norm(C[:, N:] - P[:, N:], axis=1)
        e_w1 = np.linalg.norm(C[:, :N] - P[:, :N], axis=1)
        rec = recon_block(C[-1], M, K=K); m2 = tau_all < M * L
        out[rn] = dict(rec_mse_over_2win=mse(np.flip(rec)[m2], np.flip(u)[m2]), rho_dA=float(np.max(np.abs(np.linalg.eigvals(dA)))),
                       abs_err_total=boundary_stats(e_tot), abs_err_window2_block=boundary_stats(e_w2), abs_err_window1_block=boundary_stats(e_w1))
        curves[("B", sname, rn)] = (e_tot, e_w1, e_w2, rec)
    # single-state (N) two-window routes
    sh = np.sqrt(Hh)
    singles = {}
    A = D / L; Bt = np.stack([h1 * gz, -(h1 - h2) * gz, -h2 * gz], 1) / (L * sh)
    singles["single exact delay taps (oracle)"] = (A, Bt, "taps")
    singles["single naive closure (D-(h1/H)phi phi^T)"] = ((D - (h1 / Hh) * G1) / L, h1 * gz / (L * sh), "u")
    singles["single LagOp fixed A_gen (2-win warp), translate -A_gen"] = (-A_gen2 / L, 2 * gz / L, "u")
    singles["single LagOp fixed A_gen (2-win warp), raw A_gen, no edge term"] = (A_gen2 / L, h1 * gz / (L * sh), "u")
    for rn, (A, B, kind) in singles.items():
        dA, dB = bilinear(A, B)
        C = run(dA, dB, taps(u, 2) if kind == "taps" else u)
        e = np.linalg.norm(C - Psingle, axis=1)
        rec = recon_single(C[-1] / sh, 2, K=K)  # basis g/sqrt(H)
        m2 = tau_all < 2 * L
        out[rn] = dict(rec_mse_over_2win=mse(np.flip(rec)[m2], np.flip(u)[m2]), rho_dA=float(np.max(np.abs(np.linalg.eigvals(dA)))),
                       abs_err_total=boundary_stats(e))
        curves[("B", sname, rn)] = (e, None, None, rec)
    # projections at T: block (2N) and single (N)
    recb = recon_block(P[-1], M, K=K); recs = recon_single(Psingle[-1] / sh, 2, K=K); m2 = tau_all < 2 * L
    out["projection block (2N, inner product)"] = dict(proj_mse_over_2win=mse(np.flip(recb)[m2], np.flip(u)[m2]))
    out["projection single (N, inner product)"] = dict(proj_mse_over_2win=mse(np.flip(recs)[m2], np.flip(u)[m2]))
    curves[("B", sname, "proj block")] = recb; curves[("B", sname, "proj single")] = recs
    results["B_two_window"][sname] = out
    print("B", sname, "done")

# =========================== F: full FouP (T=3) ====================================
F = results["F_full_foup"]
s3 = s
# repo path (Jul 31 config)
with quiet():
    m_gen = make_frequency_preserving_model(N=N, dt=DT, T=T, alpha=ALPHA, window_length=L, eps=1e-8,
                                            derive_A_from="gen", input_hold="zoh", quad_order=QO)
    m_inf = make_frequency_preserving_model(N=N, dt=DT, T=T, alpha=ALPHA, window_length=L, eps=1e-8,
                                            derive_A_from="infty", input_hold="zoh", quad_order=QO)
E_rep = np.array(m_gen.eval_matrix, float)
A_inf_rep = compute_A_infty(N, fp.phi, fp.sigma, fp.sigma_inv, dt=DT, quad_order=QO)
z, w = get_gauss_quadrature(4000, 0, 1)
Prep = np.stack([fp.phi(k, z) for k in range(N)], 1); Gram_rep = np.einsum("i,ik,il->kl", w, Prep, Prep)
F["repo_basis_gram_diag"] = np.diag(Gram_rep).round(4).tolist()
F["repo_infty_dA_spectral_radius"] = float(np.max(np.abs(np.linalg.eigvals(m_inf.dA.numpy().astype(float)))))
F["repo_gen_dA_spectral_radius"] = float(np.max(np.abs(np.linalg.eigvals(m_gen.dA.numpy().astype(float)))))
F["repo_gen_A_max_re_eig"] = float(np.max(np.linalg.eigvals(m_gen.A).real))
F["repo_B_zoh_vs_strip_ratio_const_mode"] = float(m_gen.dB.numpy()[0] / (fp.phi(0, np.array([1.0]))[0] * h1 * DT / L))
F["kappa_average_closure"] = 2 * (1 - ALPHA) / (1 + ALPHA)
A_genF = compute_A_gen_for_basis(N, fixed_phi, fixed_dphi, fp.sigma_inv_deriv, quad_order=QO)
F["A_gen_fixed_FouP_vs_D"] = rel_fro(A_genF, D)
A_infF = compute_A_infty(N, fixed_phi, fp.sigma, fp.sigma_inv, dt=DT, quad_order=QO)
zs, ws = get_gauss_quadrature(QO, fp.sigma(0.0, -DT), 1.0)
B_stripF = np.stack([fixed_phi(k, zs) for k in range(N)], 1).T @ ws
kap = F["kappa_average_closure"]
for sname, u in sig.items():
    out = {}
    ut = torch.tensor(u, dtype=torch.float)
    with torch.no_grad():
        r = m_gen.reconstruct(m_gen(ut)[-1]).numpy()
        ci = m_inf(ut)[-1]
    out["repo gen/zoh/legs-tilt (Jul31 02.02 path), torch f32"] = dict(rec_mse_vs_input=mse(r, u))
    out["repo infty/zoh (inv(A_inf^T)e^{-dt}), torch f32"] = dict(finite=bool(torch.all(torch.isfinite(ci))), max_abs_cT=float(torch.nan_to_num(ci.abs(), nan=np.inf).max()))
    # projections
    c_ls = np.linalg.lstsq(E_rep, np.flip(u), rcond=None)[0]
    out["projection repo basis, least squares (02.01)"] = dict(proj_mse=mse(np.flip(E_rep @ c_ls), u))
    zq, wq = get_gauss_quadrature(2 * N if False else 32, 0, 1)
    from lag_op_hippo.freq_diagnostics import run_projection_reconstruction, build_interpolated_signal
    pr = run_projection_reconstruction(build_interpolated_signal(s3, u), s3, fp.phi, fp.sigma, fp.sigma_inv, N, t=T, verbose=False)
    out["projection repo basis, inner product (02.01 gauss_projection, default quad)"] = dict(proj_mse=pr.mse)
    Pc = sum(h(n) * F_window(u, n) for n in range(1, 4))  # fixed basis projection coefficient (amp-1 g)
    rec_ip = recon_single(Pc[-1], K=K)
    out["projection fixed basis, inner product (exact Riemann)"] = dict(proj_mse=mse(rec_ip, u))
    Ef = gmat(np.mod(np.arange(K) * DT / L, 1))
    out["projection fixed basis, least squares"] = dict(proj_mse=mse(np.flip(Ef @ np.linalg.lstsq(Ef, np.flip(u), rcond=None)[0]), u))
    curves[("F", sname, "proj ip fixed")] = rec_ip
    routes = {
        "repo gen/zoh/legs-tilt, float64 scan": (m_gen.dA.numpy().astype(float), m_gen.dB.numpy().astype(float), "u", "repo"),
        "fixed exact delay taps (oracle)": (bilinear(D / L, np.stack([h1 * gz] + [-(h(n) - h(n + 1)) * gz for n in (1, 2, 3)], 1) / L), None, "taps", "fixed"),
        "fixed naive closure (D - h1 phi phi^T, h1 phi)": (bilinear((D - h1 * G1) / L, h1 * gz / L), None, "u", "fixed"),
        "fixed average closure kappa=2(1-a)/(1+a)": (bilinear((D - kap * G1) / L, kap * gz / L), None, "u", "fixed"),
        "fixed A_gen (FouP warp), legs tilt -(A_gen+I)^T": (bilinear(tilt_continuous_A(A_genF, "legs") / L, gz / L), None, "u", "fixed"),
        "fixed A_gen (FouP warp), translate -A_gen": (bilinear(-A_genF / L, gz / L), None, "u", "fixed"),
        "fixed A_gen - kappa phi(1)phi(1)^T (rank-1 edge)": (bilinear((A_genF - kap * G1) / L, kap * gz / L), None, "u", "fixed"),
        "fixed A_inf^T, strip B": ((A_infF.T, B_stripF), None, "u", "fixed"),
        "fixed pinv(A_inf^T)e^{-dt}, strip B": ((np.linalg.pinv(A_infF.T) * np.exp(-DT), B_stripF), None, "u", "fixed"),
        "repo-basis pinv(A_inf^T)e^{-dt}, repo B_zoh (=repo infty, f64)": ((np.linalg.pinv(A_inf_rep.T) * np.exp(-DT), m_inf.dB.numpy().astype(float)), None, "u", "repo"),
    }
    Pref = Pc
    for rn, spec in routes.items():
        if spec[3] == "repo" and rn.startswith("repo gen"):
            dA, dB = spec[0], spec[1]
        else:
            dA, dB = spec[0]
        C = run(dA, dB, taps(u, 3) if spec[2] == "taps" else u)
        fin = bool(np.all(np.isfinite(C[-1])))
        rec = np.flip(E_rep @ C[-1]) if spec[3] == "repo" else recon_single(C[-1], K=K)
        d = dict(rec_mse_vs_input=mse(rec, u) if fin else float("nan"), finite=fin,
                 rho_dA=float(np.max(np.abs(np.linalg.eigvals(dA)))))
        if spec[3] == "fixed":
            e = np.linalg.norm(C - Pref, axis=1) / (np.linalg.norm(Pref, axis=1) + 1e-12)
            d["rel_coef_err_T"] = float(e[-1]); d["boundary_t=L"] = boundary_stats(e); curves[("F", sname, rn)] = (e, rec)
        else:
            curves[("F", sname, rn)] = (None, rec)
        if not fin:
            first = int(np.argmax(~np.all(np.isfinite(C), 1)))
            d["first_nonfinite_step"] = first
        out[rn] = d
    # block cascade with 3 windows (true windowed memory)
    Pb = proj_block(u, 3)
    for rn, (A, B, kind) in block_routes(3).items():
        dA, dB = bilinear(A, B)
        C = run(dA, dB, taps(u, 3) if kind == "taps" else u)
        rec = recon_block(C[-1], 3, K=K)
        e = np.linalg.norm(C - Pb, axis=1) / (np.linalg.norm(Pb, axis=1) + 1e-12)
        out["3-window " + rn] = dict(rec_mse_vs_input=mse(rec, u), rho_dA=float(np.max(np.abs(np.linalg.eigvals(dA)))), rel_coef_err_T=float(e[-1]))
        curves[("F", sname, "3-window " + rn)] = (e, rec)
    F[sname] = out
    print("F", sname, "done")
dump(results, "B_F_results.json")
import pickle
pickle.dump({k: v for k, v in curves.items()}, open(os.path.join(OUT, "_curves_BF.pkl"), "wb"))

# ================================ plots ==========================================
tt = s
for sname in sig:
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))
    for rn, cc in zip(block_routes(2), ["k", "tab:orange", "tab:blue", "tab:green"]):
        e_tot, e1, e2, _ = curves[("B", sname, rn)]
        axes[0].semilogy(tt, np.clip(e1, 1e-8, 1e3), color=cc, lw=1, label=rn)
        axes[1].semilogy(tt, np.clip(e2, 1e-8, 1e3), color=cc, lw=1, label=rn)
    for rn, cc in zip(["single exact delay taps (oracle)", "single naive closure (D-(h1/H)phi phi^T)",
                       "single LagOp fixed A_gen (2-win warp), translate -A_gen", "single LagOp fixed A_gen (2-win warp), raw A_gen, no edge term"],
                      ["k", "tab:orange", "tab:red", "tab:purple"]):
        axes[2].semilogy(tt, np.clip(curves[("B", sname, rn)][0], 1e-8, 1e3), color=cc, lw=1, label=rn)
    for ax, ttl in zip(axes, [r"block state, window-1 part $\|c^{(1)}_{\rm rec}(t)-c^{(1)}_{\rm proj}(t)\|$",
                              r"block state, window-2 part $\|c^{(2)}_{\rm rec}(t)-c^{(2)}_{\rm proj}(t)\|$",
                              r"single $N$-state: $\|c_{\rm rec}(t)-c_{\rm proj}(t)\|$"]):
        ax.axvline(L, color="r", ls="--", lw=1); ax.axvline(2 * L, color="r", ls=":", lw=1)
        ax.set_xlabel(r"$t$ (red dashed: $t=L$, input reaches the window-1/2 boundary)"); ax.set_title(ttl, fontsize=9)
        ax.legend(fontsize=6.5)
    fig.suptitle(f"Exp. B: two-window system, coefficient error vs time ({sname})", y=1.02)
    fig.tight_layout(); save(fig, f"B1_two_window_coef_error_{sname}.png")

    fig, ax = plt.subplots(figsize=(12, 4.5))
    m2 = (T - s) <= 2 * L
    ax.plot(s[m2], sig[sname][m2], color="0.6", lw=3, label=r"input $u$")
    ax.plot(s[m2], curves[("B", sname, "proj block")][m2], "c--", label=r"projection, block $2N$")
    ax.plot(s[m2], curves[("B", sname, "proj single")][m2], "m--", label=r"projection, single $N$ (fold)")
    for rn, cc in zip(list(block_routes(2)) + ["single naive closure (D-(h1/H)phi phi^T)"], ["k", "tab:orange", "tab:blue", "tab:green", "tab:red"]):
        ax.plot(s[m2], np.clip(curves[("B", sname, rn)][3][m2], -3, 3), color=cc, lw=1, label="rec: " + rn)
    ax.axvline(T - L, color="r", ls="--", lw=1)
    ax.set_xlabel(r"absolute time $s$ (red: lookback $\tau=L$ boundary)"); ax.set_title(f"Exp. B: reconstructions at $T=3$ over the 2-window support ({sname})")
    ax.legend(fontsize=7, ncol=2); ax.set_ylim(-2.5, 2.5)
    fig.tight_layout(); save(fig, f"B2_two_window_recon_{sname}.png")

    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    ax = axes[0]
    ax.plot(s, sig[sname], color="0.6", lw=3, label=r"input $u$")
    for rn, cc in [("repo gen/zoh/legs-tilt, float64 scan", "tab:red"), ("fixed exact delay taps (oracle)", "k"),
                   ("fixed average closure kappa=2(1-a)/(1+a)", "tab:green"), ("fixed A_inf^T, strip B", "tab:purple"),
                   ("3-window block Gu-consistent chain (diag D-2phi phi^T)", "tab:blue")]:
        y = curves[("F", sname, rn)][1]
        if np.all(np.isfinite(y)): ax.plot(s, np.clip(y, -3, 3), color=cc, lw=1, label=rn)
    ax.plot(s, curves[("F", sname, "proj ip fixed")], "c--", lw=1, label="projection fixed basis (fold)")
    ax.set_ylim(-2.5, 2.5); ax.legend(fontsize=7); ax.set_xlabel(r"$s$"); ax.set_title(f"full FouP ($N=16$; block: $3N$), recon at $T=3$ ({sname})")
    ax = axes[1]
    for rn in F[sname]:
        k = ("F", sname, rn)
        if k in curves and curves[k][0] is not None:
            ax.semilogy(s, np.clip(curves[k][0], 1e-8, 1e3), lw=1, label=rn)
    for b in (1, 2): ax.axvline(b * L, color="r", ls="--", lw=0.8)
    ax.set_xlabel(r"$t$"); ax.set_title(r"rel. coefficient error vs projection"); ax.legend(fontsize=6)
    fig.tight_layout(); save(fig, f"F1_full_foup_{sname}.png")
