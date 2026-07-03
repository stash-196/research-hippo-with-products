# %%
import time
import numpy as np
import torch
from scipy.linalg import expm
from lag_op_hippo.hippo_by_original_authors import (
    HiPPO,
)
from lag_op_hippo.lag_hippo import (
    sigma,
    sigma_deriv,
    sigma_inv,
    sigma_inv_deriv,
    phi,
    phi_deriv,
    compute_A_infty,
    compute_A_gen,
    LagHiPPO,
    lorenzsignal,
)
import torch.optim as optim
import torch.nn as nn

from lag_op_hippo.visualizers import plot_reconstructed_signal

# Set seed for reproducibility
np.random.seed(0)

# Global settings
PAPER_READY_PLOTS = True
device = torch.device("cuda" if torch.cuda.is_available() else torch.device("cpu"))


def mackey_glass(length, beta=0.2, gamma=0.1, n=10, tau=17, initial=1.5):
    """Generate Mackey-Glass series (chaotic delay eq)."""
    x = torch.zeros(length)
    x[0] = initial
    for t in range(1, length):
        if t - tau >= 0:
            x[t] = x[t - 1] + gamma * (
                beta * x[t - tau] / (1 + x[t - tau] ** n) - x[t - 1]
            )
        else:
            x[t] = x[t - 1]
    return x  # Shape: (length,)


# Model: SSM + Linear Readout
class SSMForecaster(nn.Module):
    def __init__(self, ssm):
        super().__init__()
        self.ssm = ssm  # Your LagHiPPO or HiPPO
        self.readout = nn.Linear(N, K)  # From c_final to K preds

    def forward(self, x):
        x = x.squeeze(-1).transpose(0, 1)  # (batch, length) -> (length, batch)
        c = self.ssm(x)  # (length, batch, N)
        c = c[-1]  # (batch, N)
        return self.readout(c)  # (batch, K)


# %%
# Params
L = 1000  # Input window length
K = 15  # Forecast horizon
N_series = 10000  # Total series length
train_split = 8000
dt = 0.001  # Discrete step (as in your code)
T = N_series * dt  # Total time for SSM
N = 64  # SSM order

# Generate data
u = mackey_glass(N_series).unsqueeze(-1)  # (length, 1) for univariate
u = torch.asarray(
    lorenzsignal(T, dt, initial_state=[1.0, 1.0, 8.0])[:, 0], dtype=torch.float32
).unsqueeze(-1)

# Sliding windows: Inputs (N_series - L - K + 1, L, 1), Targets (..., K, 1)
inputs = torch.stack([u[i : i + L] for i in range(N_series - L - K + 1)])
targets = torch.stack([u[i + L : i + L + K] for i in range(N_series - L - K + 1)])

train_in, test_in = inputs[:train_split], inputs[train_split:]
train_tgt, test_tgt = targets[:train_split], targets[train_split:]

# Init models
hippo = HiPPO(N=N, method="legs", dt=dt, T=L * dt)  # Adjust T if needed
lagop = LagHiPPO(
    N=N,
    phi_func=phi,
    phi_deriv_func=phi_deriv,
    sigma_func=sigma,
    sigma_deriv_func=sigma_deriv,
    sigma_inv_func=sigma_inv,
    sigma_inv_deriv_func=sigma_inv_deriv,
    dt=dt,
    T=T,
    derive_A_from="infty",
    input_hold="zoh",
)

model_hippo = SSMForecaster(hippo).to(device)
model_lagop = SSMForecaster(lagop).to(device)


# %%
# Train function (simple loop)
def train_model(model, ins, tgts, epochs=30, lr=0.001, batch_size=32):
    opt = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    for ep in range(epochs):
        time_start = time.time()
        for i in range(0, len(ins), batch_size):  # Batch size
            batch_in = ins[i : i + batch_size].to(device)
            batch_tgt = tgts[i : i + batch_size].squeeze(-1).to(device)
            preds = model(batch_in)
            loss = loss_fn(preds, batch_tgt)
            opt.zero_grad()
            loss.backward()
            opt.step()
        print(f"Epoch {ep}: Loss {loss.item():.4f}")
        time_end = time.time()
        print(f"Epoch {ep}: Training time {time_end - time_start:.2f} seconds")


# Train
train_model(model_hippo, train_in, train_tgt)
train_model(model_lagop, train_in, train_tgt)
# %%
# Test
with torch.no_grad():
    preds_hippo = model_hippo(test_in.to(device)).cpu()
    preds_lagop = model_lagop(test_in.to(device)).cpu()


# Metrics
def nrmse(pred, true):
    rmse = torch.sqrt(((pred - true) ** 2).mean())
    return rmse / true.std()


print("HiPPO NRMSE:", nrmse(preds_hippo, test_tgt.squeeze(-1)))
print("LagOp NRMSE:", nrmse(preds_lagop, test_tgt.squeeze(-1)))

# plot reconstructed
# concatinate input (test_in) and predictoin (preds_)
hippo_recon = torch.cat([test_in[0, :, :].squeeze(), preds_hippo[0, :]], dim=0)
lagop_recon = torch.cat([test_in[0, :, :].squeeze(), preds_lagop[0, :]], dim=0)
vals = torch
    recon_list = [
        hippo_recon.numpy(),
        lagop_recon.numpy(),
    ]
    reconstruction_labels = [
        "HiPPO-LegS",
        "LagOp HiPPO-ZOH",
    ]
    recon_linestyle_list = [
        "-",
        "-.",
    ]
    
    plot_reconstructed_signal(
        vals,
        u,
        recon_list,
        data_name="",
        domain_name="s",
        title=f"Input Frequency = {freq} Hz",
        paper_ready=PAPER_READY_PLOTS,
        recon_labels_list=reconstruction_labels,
        recon_linestyle_list=recon_linestyle_list,
        recon_color_list=None,
        function_name="u",
        legend_loc="upper left",
        xlim=(vals[0] - 1, vals[-1]),
        ylim=(-3.5, 3.5),
    )
# %%
