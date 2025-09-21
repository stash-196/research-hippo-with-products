# %%
# import numpy as np
from scipy.special import legendre
from scipy.integrate import quad
import numpy as np
from tqdm import tqdm
import time


# Basis: Normalized Legendre on [0,1]
def phi(n, z):
    y = 2 * z - 1  # Map z from [0,1] to [-1,1]
    return np.sqrt(2 * n + 1) * legendre(n)(y) * ((-1) ** n)


# Warp: Exponential for HiPPO-LegS on [0,1]
def sigma(t, s):
    return np.exp(s - t)


# Inverse warp
def sigma_inv(t, z):
    return t + np.log(z)


# Lag operator
def lag(t, z):
    return sigma(t + 1, sigma_inv(t, z))


# Test orthonormality
def test_orthonormality(N=4, phi=phi):
    for n in range(N):
        for m in range(N):

            def integrand(z):
                return phi(n, z) * phi(m, z)

            result = quad(integrand, 0, 1)[0]
            if n == m:
                assert np.isclose(result, 1), f"Failed at n={n}, m={m}, got {result}"
            else:
                assert np.isclose(result, 0), f"Failed at n={n}, m={m}, got {result}"
    print("Orthonormality test passed!")


test_orthonormality()
# %%
N = 4
t = 10  # Arbitrary, constant for this warp
Z_min, Z_max = 0, 1

# Compute A_t
A_t = np.zeros((N, N))
start_time = time.time()

# Using tqdm to create progress bars for both loops
for n in tqdm(range(N), desc="Outer loop"):
    for m in tqdm(range(N), desc=f"Inner loop (n={n})", leave=False):

        def integrand(z):
            return phi(n, lag(t, z)) * phi(m, z)

        A_t[n, m] = quad(integrand, Z_min, Z_max)[0]

end_time = time.time()
print(f"\nTotal computation time: {end_time - start_time:.2f} seconds")
print("Computed A:\n", np.round(A_t, 3))

# %%
# Compute B_t
B_t = np.zeros(N)
start_time = time.time()
for n in tqdm(range(N), desc="Computing B"):

    def integrand(z):
        return phi(n, z)

    z_min = sigma(t + 1, t)
    z_max = sigma(t + 1, t + 1)
    B_t[n] = quad(integrand, z_min, z_max)[0]

end_time = time.time()
print(f"\nTotal computation time for B: {end_time - start_time:.2f} seconds")
print("Computed B:\n", np.round(B_t, 3))


# Compute A_gen and B_gen
A_gen = np.zeros((N, N))
start_time = time.time()
# Using tqdm to create progress bars for both loops
for n in tqdm(range(N), desc="Outer loop"):
    for m in tqdm(range(N), desc=f"Inner loop (n={n})", leave=False):

        def integrand(z):
            return phi(n, lag(t, z)) * z * phi(m, z)

        A_gen[n, m] = quad(integrand, Z_min, Z_max)[0]

end_time = time.time()

# %%
# Compute Analytical A and B for HiPPO-LegS
q = np.arange(N, dtype=np.float64)
col, row = np.meshgrid(q, q)
r = 2 * q + 1
M = -(np.where(row >= col, r, 0) - np.diag(q))
T = np.sqrt(np.diag(2 * q + 1))
A_org = T @ M @ np.linalg.inv(T)
B_org = np.diag(T)[:, None]
B_org = (
    B_org.copy()
)  # Otherwise "UserWarning: given NumPY array is not writeable..." after torch.as_tensor(B)

print("Analytical A:\n", np.round(A_org, 3))
print("Analytical B:\n", np.round(B_org.flatten(), 3))

# %%
