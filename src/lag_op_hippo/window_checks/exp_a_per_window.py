"""Experiment A: per-window FouT (window n of the FouP measure, alone)."""
from lag_op_hippo.window_checks.common import *
import torch

T = 4.0
s, sig = signals(T)
K = s.size
fp = make_frequency_preserving_basis(alpha=ALPHA, window_length=L, eps=1e-8)
fb = make_fout_basis(window_length=L)
gz = g0()
D = Dmat()
A_gu, B_gu = transition("fourier", N); B_gu = B_gu[:, 0]
sanity = {}
sanity["A_Gu_vs_D-2g0g0T"] = rel_fro(D - 2 * np.outer(gz, gz), A_gu)
sanity["B_Gu_vs_2g0"] = rel_fro(2 * gz, B_gu)
xx = np.linspace(0, 0.999, 50)
sanity["FouTBasis_phi_vs_g"] = rel_fro(np.stack([fb.phi(k, 1 - xx) for k in range(N)], 1), gmat(xx))

# ---- LagOp pieces on the FouT (= single window, local coords) basis ----------
QO = 200 * N
A_gen_fixed = compute_A_gen_for_basis(N, fb.phi, fb.phi_deriv, fb.sigma_inv_deriv, quad_order=QO)
with quiet():
    repo_fout = make_fout_model(N=N, dt=DT, T=L, window_length=L)  # repo default: gen(LegS-biased)/translate/delta
A_inf_fout = compute_A_infty(N, fb.phi, fb.sigma, fb.sigma_inv, dt=DT, quad_order=QO)
zN, wN = get_gauss_quadrature(QO, 1 - DT / L, 1.0)
B_strip_fout = vectorized_basis(fb.phi, np.arange(N), zN).T @ wN     # int over newest strip, dz
B_zoh_repo_fout = compute_B_zoh(N, fb.phi, DT, quad_order=QO)        # repo: [e^{-dt},1]
sanity["A_gen_fixed_vs_D"] = rel_fro(A_gen_fixed, D)
sanity["A_gen_repo_vs_D"] = rel_fro(repo_fout.A * -1, D)  # repo A = -A_gen(LegS-biased)

def routes_for_window(n):
    """return dict name -> (dA, dB, inputs_fn, E_fn) ; E_fn(tau_local)->eval matrix for c."""
    hn = h(n); sh = np.sqrt(hn)
    R = {}
    Eg = lambda x: gmat(x) / sh        # block-n orthonormal basis psi = g/sqrt(h_n)
    dly = lambda u: [delay(u, (n - 1) * L)]
    # Gu FouT block (the reference diagonal block)
    for nm, f in (("Gu-FouT block, bilinear", bilinear), ("Gu-FouT block, ZOH", zoh)):
        dA, dB = f(A_gu / L, sh * B_gu / L)
        R[nm] = (dA, dB, dly, Eg)
    # exact: delay taps u(t-(n-1)L), u(t-nL), no closure
    dA, dB = bilinear(D / L, np.stack([sh * gz / L, -sh * gz / L], 1))
    R["exact delay taps (oracle)"] = (dA, dB, lambda u: [delay(u, (n - 1) * L), delay(u, n * L)], Eg)
    # repo LagOp FouT path (make_fout_model default)
    R["repo LagOp-FouT (gen LegS-biased, translate)"] = (repo_fout.dA.numpy().astype(float),
                                                        sh * repo_fout.dB.numpy().astype(float), dly, Eg)
    for nm, A in (("fixed A_gen, translate tilt -A_gen", -A_gen_fixed),
                  ("fixed A_gen, rank-1: A_gen-2phi(1)phi(1)^T", A_gen_fixed - 2 * np.outer(gz, gz))):
        dA, dB = bilinear(A / L, sh * fout_input_vector(N) / L)
        R[nm] = (dA, dB, dly, Eg)
    R["A_inf repo map pinv(A_inf^T)e^{-dt}, B_zoh(repo)"] = (np.linalg.pinv(A_inf_fout.T) * np.exp(-DT),
                                                             sh * B_zoh_repo_fout, dly, Eg)
    R["A_inf^T, strip B"] = (A_inf_fout.T, sh * B_strip_fout, dly, Eg)
    # repo FouP psi restricted to window n (bug amplitude 1/sqrt(h1) vs intended 1/sqrt(h_n))
    for nm, scale in (("repo FouP psi (bug amp 1/sqrt(h1)), rank-1", 1.0),
                      ("repo FouP psi (intended amp 1/sqrt(h_n)), rank-1", np.sqrt(h(1) / hn))):
        phi, dphi = repo_psi_local(n, scale, fp)
        Ag = compute_A_gen_for_basis(N, phi, dphi, fb.sigma_inv_deriv, quad_order=QO)
        p1 = np.array([phi(k, 1.0 - 1e-9) for k in range(N)]).ravel()
        dA, dB = bilinear((Ag - 2 * np.outer(p1, p1)) / L, 2 * p1 / L)
        Ef = (lambda phi: (lambda x: np.stack([phi(k, 1 - x) for k in range(N)], 1)))(phi)
        R[nm] = (dA, dB, dly, Ef)
    return R

def gram_repo_psi(n, scale):
    phi, _ = repo_psi_local(n, scale, fp)
    z, w = get_gauss_quadrature(4000, 0, 1)
    P = np.stack([phi(k, z) for k in range(N)], 1)
    return np.einsum("i,ik,il->kl", w, P, P)

rows, traj = [], {}
recon_store = {}
for n in (1, 2, 3, 4):
    hn = h(n)
    jj = lookback_window_rows(K, n)
    tau_loc = (jj * DT - (n - 1) * L) / L
    for sname, u in sig.items():
        u_lb = u[K - 1 - jj]
        # projection 1: inner product with window measure (exact orthonormal basis)
        Fn = F_window(u, n)
        c_ip = np.sqrt(hn) * Fn[-1]
        Eg = gmat(tau_loc) / np.sqrt(hn)
        rec_ip = Eg @ c_ip
        c_ls = np.linalg.lstsq(Eg, u_lb, rcond=None)[0]
        rec_ls = Eg @ c_ls
        # repo-basis projections on window n (FouP z coords, Lebesgue dz)
        zrep = fp.sigma(0.0, -(jj * DT))
        Erep = np.stack([fp.phi(k, zrep) for k in range(N)], 1)
        wdz = hn / L * DT
        c_rep_ip = Erep.T @ (u_lb * wdz)
        rec_rep_ip = Erep @ c_rep_ip
        rec_rep_ls = Erep @ np.linalg.lstsq(Erep, u_lb, rcond=None)[0]
        base = dict(window=n, h_n=hn, signal=sname)
        rows.append(dict(base, route="projection: inner product (orthonormal g/sqrt(h_n))", proj_mse=mse(rec_ip, u_lb), rec_mse=None))
        rows.append(dict(base, route="projection: least squares on eval matrix (g/sqrt(h_n))", proj_mse=mse(rec_ls, u_lb), rec_mse=None))
        rows.append(dict(base, route="projection: inner product, repo FouP phi on window (dz)", proj_mse=mse(rec_rep_ip, u_lb), rec_mse=None))
        rows.append(dict(base, route="projection: least squares, repo FouP phi (02.01 style)", proj_mse=mse(rec_rep_ls, u_lb), rec_mse=None))
        recon_store[(n, sname, "u")] = u_lb; recon_store[(n, sname, "proj ip")] = rec_ip
        for rname, (dA, dB, inp, Ef) in routes_for_window(n).items():
            U = np.stack(inp(u), 1)
            C = scan(dA, dB, U)
            E = Ef(tau_loc)
            rec = E @ C[-1]
            finite = bool(np.all(np.isfinite(C[-1])))
            rho = float(np.max(np.abs(np.linalg.eigvals(dA))))
            # coefficient error vs projection (only meaningful for g/sqrt(h_n) basis routes)
            if "repo FouP psi" not in rname:
                err_t = np.linalg.norm(C - np.sqrt(hn) * Fn, axis=1) / (np.linalg.norm(np.sqrt(hn) * Fn, axis=1) + 1e-12)
                cerr = float(err_t[-1])
                traj[(n, sname, rname)] = (C, np.sqrt(hn) * Fn)
            else:
                cerr = None
            rows.append(dict(base, route=rname, proj_mse=None, rec_mse=mse(rec, u_lb) if finite else float("nan"),
                             rho_dA=rho, finite=finite, rel_coef_err_T=cerr))
            recon_store[(n, sname, rname)] = rec
        print("window", n, sname, "done")

# ---- sanity vs Jul 31 02.02 FouT (repo path, T=3, cos, torch float32 forward) -----
s3, sig3 = signals(3.0)
with quiet():
    m3 = make_fout_model(N=N, dt=DT, T=3.0, window_length=L)
    hp = HiPPO(method="fourier", N=N, dt=DT, T=3.0)
u3 = torch.tensor(sig3["cos"], dtype=torch.float)
rl = m3.reconstruct(m3(u3)[-1]).numpy(); rh = hp.reconstruct(hp(u3)[-1]).numpy()
sanity["repro_02.02_fout_lag_mse_vs_input(T=3, expect 0.9535)"] = mse(rl, sig3["cos"])
sanity["repro_02.02_fout_hippo_mse_vs_input(T=3, expect 0.3330)"] = mse(rh, sig3["cos"])
sanity["note"] = "02.02 MSE is over s in [0,3]; FouT only remembers 1 window, so 2/3 of the span is reconstructed as 0 -> floor 0.5*2/3=0.333"
# window-1 Gu block on T=3 over the whole [0,3] span for apples-to-apples with 02.02
dA, dB = bilinear(A_gu / L, B_gu / L)
c = scan(dA, dB, sig3["cos"])[-1]
Ew = gmat(np.arange(s3.size) * DT / L); Ew[np.arange(s3.size) * DT >= L] = 0
sanity["window1 Gu block (my float64 scan), mse over [0,3]"] = mse(np.flip(Ew @ c), sig3["cos"])
sanity["gram_diag_repo_psi_bug_per_window"] = {n: np.diag(gram_repo_psi(n, 1.0))[:4].round(4).tolist() for n in (1, 2, 3, 4)}
sanity["gram_diag_repo_psi_intended_per_window"] = {n: np.diag(gram_repo_psi(n, np.sqrt(h(1) / h(n))))[:4].round(4).tolist() for n in (1, 2, 3, 4)}
dump(dict(rows=rows, sanity=sanity, T=T, dt=DT, N=N, alpha=ALPHA, L=L), "A_per_window_results.json")
print(json.dumps(sanity, indent=1, default=str))

# ---------------- plots -------------------------------------------------------------
show = ["Gu-FouT block, bilinear", "exact delay taps (oracle)", "repo LagOp-FouT (gen LegS-biased, translate)",
        "fixed A_gen, translate tilt -A_gen", "A_inf^T, strip B", "repo FouP psi (bug amp 1/sqrt(h1)), rank-1"]
cols = ["tab:green", "k", "tab:red", "tab:orange", "tab:purple", "tab:brown"]
for sname in sig:
    fig, axes = plt.subplots(4, 1, figsize=(11, 12))
    for i, n in enumerate((1, 2, 3, 4)):
        ax = axes[i]; jj = lookback_window_rows(K, n); sabs = s[K - 1 - jj]
        ax.plot(sabs, recon_store[(n, sname, "u")], color="0.6", lw=3, label=r"input $u$")
        ax.plot(sabs, recon_store[(n, sname, "proj ip")], "c--", lw=1.5, label=r"projection $\hat u$ (inner product)")
        for r, cc in zip(show, cols):
            y = recon_store[(n, sname, r)]
            if np.all(np.isfinite(y)):
                ax.plot(sabs, np.clip(y, -3, 3), color=cc, lw=1, label="rec: " + r)
        ax.set_ylim(-2.5, 2.5)
        ax.set_title(rf"window $n={n}$ alone ($h_n={h(n):.4g}$), lookback $\tau\in[{n-1},{n})$, signal: {sname}")
        ax.set_xlabel(r"absolute time $s$"); ax.set_ylabel(r"$u(s)$, $\hat u(s)$")
    axes[0].legend(fontsize=7, ncol=2, loc="upper left")
    fig.suptitle(r"Exp. A: per-window FouT, $\hat u$ from $c_T$ at $T=4$ (y clipped to $\pm 2.5$)", y=1.0)
    fig.tight_layout(); save(fig, f"A1_per_window_recon_{sname}.png")

# coefficient trajectories (window 2, cos & chirp): modes 0, 2, 6 (cos m=0,1,3)
for sname in sig:
    fig, axes = plt.subplots(2, 3, figsize=(13, 6), sharex=True)
    for i, n in enumerate((1, 2)):
        for j, k in enumerate((0, 2, 6)):
            ax = axes[i, j]
            C, P = traj[(n, sname, "Gu-FouT block, bilinear")]
            ax.plot(s, P[:, k], color="0.6", lw=3, label=r"projection $c^{(n)}_k(t)$")
            for r, cc in zip(["Gu-FouT block, bilinear", "exact delay taps (oracle)", "repo LagOp-FouT (gen LegS-biased, translate)", "A_inf^T, strip B"],
                             ["tab:green", "k", "tab:red", "tab:purple"]):
                ax.plot(s, np.clip(traj[(n, sname, r)][0][:, k], -5, 5), color=cc, lw=1, label=r)
            ax.axvline((n - 1) * L, color="gray", ls=":"); ax.axvline(n * L, color="gray", ls=":")
            ax.set_title(rf"window {n}, mode $k={k}$ ({sname})"); ax.set_xlabel(r"$t$")
    axes[0, 0].legend(fontsize=7)
    fig.suptitle(r"Exp. A: coefficient trajectories $c^{(n)}_k(t)$, recurrence vs projection", y=1.0)
    fig.tight_layout(); save(fig, f"A2_coef_traj_{sname}.png")

# error vs n
fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
routes_all = [r for r in routes_for_window(1)]
for ax, sname in zip(axes, sig):
    for r in routes_all:
        ys = [next(x["rec_mse"] for x in rows if x["window"] == n and x["signal"] == sname and x["route"] == r) for n in (1, 2, 3, 4)]
        ys = [y if (y is not None and np.isfinite(y)) else np.nan for y in ys]
        ax.semilogy((1, 2, 3, 4), np.maximum(ys, 1e-16), "o-", label=r, ms=4)
    yp = [next(x["proj_mse"] for x in rows if x["window"] == n and x["signal"] == sname and x["route"].startswith("projection: inner product (")) for n in (1, 2, 3, 4)]
    ax.semilogy((1, 2, 3, 4), np.maximum(yp, 1e-16), "c--s", label="projection (inner product)")
    ax.set_xlabel(r"window index $n$"); ax.set_ylabel(r"MSE over window $n$"); ax.set_xticks([1, 2, 3, 4])
    ax.set_title(f"signal: {sname}")
axes[1].legend(fontsize=6.5, loc="center left", bbox_to_anchor=(1.0, 0.5))
fig.suptitle(r"Exp. A: reconstruction MSE vs window index $n$ (NaN/overflow omitted)")
fig.tight_layout(); save(fig, "A3_error_vs_window.png")
