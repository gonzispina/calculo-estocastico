"""Generate the English OU trajectory figure used in the presentation."""
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

np.random.seed(0)
SIGMA, T, N_STEPS, N_TRAJ = 1.0, 12.0, 1200, 8
dt = T / N_STEPS
time = np.linspace(0, T, N_STEPS + 1)
x0_values = np.linspace(-3, 3, N_TRAJ)
trajs = np.zeros((N_TRAJ, N_STEPS + 1))
trajs[:, 0] = x0_values
for k in range(N_STEPS):
    trajs[:, k + 1] = trajs[:, k] * (1 - dt) + SIGMA * np.sqrt(dt) * np.random.randn(N_TRAJ)

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
colors = plt.cm.coolwarm(np.linspace(0, 1, N_TRAJ))
for i in range(N_TRAJ):
    axes[0].plot(time, trajs[i], alpha=0.75, lw=1.2, color=colors[i])
sigma_inf = SIGMA / np.sqrt(2)
axes[0].axhline(2 * sigma_inf, color="gray", ls="--", lw=1.2, label=r"$\pm 2\,\sigma_\infty$")
axes[0].axhline(-2 * sigma_inf, color="gray", ls="--", lw=1.2)
axes[0].set_xlabel("time $t$")
axes[0].set_ylabel("$X_t$")
axes[0].set_title("Ornstein–Uhlenbeck trajectories")
axes[0].legend(fontsize=9)

x_grid = np.linspace(-4, 4, 400)
x0_fixed = 2.0
for t, ls in zip([0.5, 1.0, 2.0, 4.0], ["-", "--", "-.", ":"]):
    mean = x0_fixed * np.exp(-t)
    var = (SIGMA**2 / 2) * (1 - np.exp(-2 * t))
    axes[1].plot(x_grid, stats.norm.pdf(x_grid, loc=mean, scale=np.sqrt(var)), lw=1.8, ls=ls, label=f"$t = {t}$")
axes[1].plot(x_grid, stats.norm.pdf(x_grid, loc=0, scale=sigma_inf), lw=2.5, color="black", label=r"$\pi=\mathcal{N}(0,\,\sigma^2/2)$")
axes[1].set_xlabel("state $x$")
axes[1].set_ylabel("$p(t,x)$")
axes[1].set_title("Density evolution from $x_0=2$")
axes[1].legend(fontsize=9)
plt.tight_layout()
plt.savefig("ou_trajectories_en.png", dpi=180, bbox_inches="tight")
