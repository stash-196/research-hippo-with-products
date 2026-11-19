"""FouP per-window / two-window / eigen diagnostics (Grok Bot, 2026-11-19).
Reads and imports the repo read-only; nothing in the repo is modified."""
import numpy as np
from scipy import signal as sps
from scipy.linalg import expm

L = 1.0
ALPHA = 0.2
DT = 1e-3
N = 16


def h(n, alpha=ALPHA):
    return (1 - alpha) * alpha ** (n - 1)


# ---------------- Gu-ordered orthonormal Fourier basis on x in [0,1) -------------
# columns: [cos0=1, sin0=0 (null, kept for Gu parity), sqrt2 cos 2pi x, sqrt2 sin 2pi x, ...]
def gmat(x, N=N):
    x = np.asarray(x, dtype=np.float64)
    out = np.zeros((x.size, N))
    for m in range(N // 2):
        if m == 0:
            out[:, 0] = 1.0
            out[:, 1] = 0.0
        else:
            out[:, 2 * m] = np.sqrt(2) * np.cos(2 * np.pi * m * x)
            out[:, 2 * m + 1] = np.sqrt(2) * np.sin(2 * np.pi * m * x)
    return out


def gprime(x, N=N):
    x = np.asarray(x, dtype=np.float64)
    out = np.zeros((x.size, N))
    for m in range(1, N // 2):
        w = 2 * np.pi * m
        out[:, 2 * m] = -np.sqrt(2) * w * np.sin(w * x)
        out[:, 2 * m + 1] = np.sqrt(2) * w * np.cos(w * x)
    return out


def Dmat(N=N):
    """g_k' = sum_j D_kj g_j."""
    D = np.zeros((N, N))
    for m in range(1, N // 2):
        D[2 * m, 2 * m + 1] = -2 * np.pi * m
        D[2 * m + 1, 2 * m] = 2 * np.pi * m
    return D


def g0(N=N):
    return gmat(np.array([0.0]), N)[0]  # = g(1) by periodicity


# ---------------- discretisation & scan ------------------------------------------
def bilinear(A, B, dt=DT):
    B2 = B if B.ndim == 2 else B[:, None]
    C = np.zeros((1, A.shape[0])); D = np.zeros((1, B2.shape[1]))
    dA, dB, *_ = sps.cont2discrete((A, B2, C, D), dt=dt, method="bilinear")
    return dA, dB


def zoh(A, B, dt=DT):
    B2 = B if B.ndim == 2 else B[:, None]
    n, m = B2.shape
    M = np.zeros((n + m, n + m)); M[:n, :n] = A; M[:n, n:] = B2
    E = expm(M * dt)
    return E[:n, :n], E[:n, n:]


def scan(dA, dB, U):
    """c_k = dA c_{k-1} + dB U_k ; U: (K, m). Returns (K, n) float64."""
    U = np.asarray(U, dtype=np.float64)
    if U.ndim == 1:
        U = U[:, None]
    dB = dB if dB.ndim == 2 else dB[:, None]
    K = U.shape[0]
    c = np.zeros(dA.shape[0]); out = np.empty((K, dA.shape[0]))
    inp = U @ dB.T
    with np.errstate(over="ignore", invalid="ignore"):
        for k in range(K):
            c = dA @ c + inp[k]
            out[k] = c
    return out


def delay(u, tau, dt=DT):
    d = int(round(tau / dt))
    if d == 0:
        return u.copy()
    out = np.zeros_like(u)
    out[d:] = u[:-d]
    return out


# ---------------- projection reference (per-window FouT coefficients) ------------
def F_window(u, n, dt=DT, N=N):
    """F_n(t) = (1/L) int_{(n-1)L}^{nL} u(t-tau) g((tau-(n-1)L)/L) dtau, Riemann sum,
    evaluated on the whole grid. Block coefficient (orthonormal w.r.t. window measure
    (h_n/L) dtau with psi = g/sqrt(h_n)) is c^(n) = sqrt(h_n) F_n."""
    Lw = int(round(L / dt))
    x = np.arange(Lw) * dt / L
    G = gmat(x, N)  # (Lw, N)
    ud = delay(u, (n - 1) * L, dt)
    out = np.empty((u.size, N))
    for k in range(N):
        out[:, k] = sps.fftconvolve(ud, G[:, k] * dt / L)[: u.size]
    return out


def lookback_window_rows(K, n, dt=DT):
    Lw = int(round(L / dt))
    j = np.arange((n - 1) * Lw, n * Lw)
    j = j[j < K]
    return j


# ---------------- eigen helpers ---------------------------------------------------
def eig_sorted(M):
    w, V = np.linalg.eig(M)
    return w, V


def rel_fro(a, b):
    return float(np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-300))


def mse(a, b):
    return float(np.mean((np.asarray(a) - np.asarray(b)) ** 2))


G1 = np.outer(g0(), g0())
gz = g0()
D = Dmat()

def run(dA, dB, U):
    return scan(dA, dB, np.stack(U, 1) if isinstance(U, list) else U)

def recon_single(c, M=None, K=3000):
    tau = np.arange(K) * DT
    E = gmat(np.mod(tau / L, 1.0))
    if M is not None: E[tau >= M * L] = 0
    return np.flip(E @ c)

def recon_block(c, M, K=3000):
    tau = np.arange(K) * DT; out = np.zeros(K)
    for n in range(1, M + 1):
        m = (tau >= (n - 1) * L) & (tau < n * L)
        out[m] = gmat(np.mod(tau[m] / L, 1.0)) @ c[(n - 1) * N:n * N] / np.sqrt(h(n))
    return np.flip(out)

def block_routes(M):
    """block cascade state [c^(1);...;c^(M)], c^(n) orthonormal w.r.t. window-n measure (h_n/L)dtau."""
    sq = [np.sqrt(h(n)) for n in range(1, M + 1)]
    Z = np.zeros((M * N, M * N)); R = {}
    def blk(A, i, j, X): A[i * N:(i + 1) * N, j * N:(j + 1) * N] += X
    # exact: taps u(t), u(t-L), ..., u(t-ML)
    A = Z.copy(); Bt = np.zeros((M * N, M + 1))
    for n in range(M):
        blk(A, n, n, D / L)
        Bt[n * N:(n + 1) * N, n] += sq[n] * gz / L; Bt[n * N:(n + 1) * N, n + 1] -= sq[n] * gz / L
    R["block exact delay taps (oracle)"] = (A, Bt, "taps")
    # naive closure kappa=1: u(t-nL) ~ g0^T c^(n)/sqrt(h_n)
    A = Z.copy(); B = np.zeros(M * N)
    for n in range(M):
        blk(A, n, n, (D - G1) / L)
        if n > 0: blk(A, n, n - 1, sq[n] / sq[n - 1] * G1 / L)
    B[:N] = sq[0] * gz / L
    R["block naive (diag D-phi phi^T, sub sqrt(h_n/h_{n-1}))"] = (A, B, "u")
    # parent spec: Gu diag (D - 2 phi phi^T), subdiag sqrt(h_n/h_{n-1}) phi(0)phi(1)^T, B1 = 2 sqrt(h1) g0
    A = Z.copy(); B = np.zeros(M * N)
    for n in range(M):
        blk(A, n, n, (D - 2 * G1) / L)
        if n > 0: blk(A, n, n - 1, sq[n] / sq[n - 1] * G1 / L)
    B[:N] = 2 * sq[0] * gz / L
    R["block spec (Gu diag, sub x1)"] = (A, B, "u")
    # Gu-consistent chain: est_0=u, est_n = 2 r_n - est_{n-1}, r_n = g0^T c^(n)/sqrt(h_n)
    # dc^(n)/dt = (1/L)[D c^(n) + sqrt(h_n) g0 (est_{n-1} - est_n)] = (1/L)[D c^(n) + 2 sqrt(h_n) g0 (est_{n-1} - r_n)]
    # est_n as linear form: est_n = a_n . c + b_n u
    A = Z.copy(); B = np.zeros(M * N)
    a_prev, b_prev = np.zeros(M * N), 1.0
    for n in range(M):
        r = np.zeros(M * N); r[n * N:(n + 1) * N] = gz / sq[n]
        rowcoef = 2 * sq[n] * gz / L
        A[n * N:(n + 1) * N, :] += np.outer(rowcoef, a_prev - r)
        B[n * N:(n + 1) * N] += rowcoef * b_prev
        blk(A, n, n, D / L)
        a_prev, b_prev = 2 * r - a_prev, -b_prev
    R["block Gu-consistent chain (diag D-2phi phi^T)"] = (A, B, "u")
    return R

def taps(u, M):
    return [delay(u, n * L) for n in range(M + 1)]

def proj_block(u, M):
    return np.concatenate([np.sqrt(h(n)) * F_window(u, n) for n in range(1, M + 1)], 1)

def boundary_stats(err, t_b=L):
    K = err.size
    i = int(round(t_b / DT))
    w = int(round(0.1 / DT))
    return dict(mean_before=float(np.nanmean(err[i - w:i])), mean_after=float(np.nanmean(err[i:i + w])),
                max_0_to_b=float(np.nanmax(err[int(0.05 / DT):i])), max_b_to_2b=float(np.nanmax(err[i:min(2 * i, K)])),
                final=float(err[-1]))


def repo_psi_local(n, scale, fp):
    """window-n restriction of the repo FouP basis, in local z' in [0,1] with dz' measure.
    phi~_k(z') = sqrt(h_n)*scale*repo_phi_k(alpha^n + h_n z')  (sqrt(h_n): carries the window height)."""
    a_n, hn = ALPHA ** n, h(n)
    def phi(k, zp):
        return np.sqrt(hn) * scale * fp.phi(k, a_n + hn * np.clip(np.asarray(zp, float), 1e-12, 1 - 1e-12))
    def dphi(k, zp):
        return np.sqrt(hn) * scale * hn * fp.phi_deriv(k, a_n + hn * np.clip(np.asarray(zp, float), 1e-12, 1 - 1e-12))
    return phi, dphi

