import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

np.random.seed(0)

SIGMA = 1.0
T = 12.0
N_STEPS = 1200
dt = T / N_STEPS
N_TRAJ = 8

time = np.linspace(0, T, N_STEPS + 1)

# Simular N_TRAJ trayectorias con condiciones iniciales distintas
x0_values = np.linspace(-3, 3, N_TRAJ)
trajs = np.zeros((N_TRAJ, N_STEPS + 1))
trajs[:, 0] = x0_values

for k in range(N_STEPS):
    z = np.random.randn(N_TRAJ)
    trajs[:, k + 1] = trajs[:, k] * (1 - dt) + SIGMA * np.sqrt(dt) * z

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# ── Panel izquierdo: trayectorias ──────────────────────────────────────────────
colors = plt.cm.coolwarm(np.linspace(0, 1, N_TRAJ))
for i in range(N_TRAJ):
    axes[0].plot(time, trajs[i], alpha=0.75, lw=1.2, color=colors[i])

sigma_inf = SIGMA / np.sqrt(2)
axes[0].axhline( 2 * sigma_inf, color="gray", ls="--", lw=1.2,
                 label=r"$\pm 2\,\sigma_\infty$")
axes[0].axhline(-2 * sigma_inf, color="gray", ls="--", lw=1.2)
axes[0].set_xlabel("$t$", fontsize=12)
axes[0].set_ylabel("$X_t$", fontsize=12)
axes[0].set_title("Trayectorias del proceso OU", fontsize=12)
axes[0].legend(fontsize=10)

# ── Panel derecho: evolución de la densidad ────────────────────────────────────
x_grid = np.linspace(-4, 4, 400)
x0_fixed = 2.0
t_show = [0.5, 1.0, 2.0, 4.0]
ls_cycle = ["-", "--", "-.", ":"]

for t, ls in zip(t_show, ls_cycle):
    mean = x0_fixed * np.exp(-t)
    var  = (SIGMA**2 / 2) * (1 - np.exp(-2 * t))
    pdf  = stats.norm.pdf(x_grid, loc=mean, scale=np.sqrt(var))
    axes[1].plot(x_grid, pdf, lw=1.8, ls=ls, label=f"$t = {t}$")

pdf_inf = stats.norm.pdf(x_grid, loc=0, scale=sigma_inf)
axes[1].plot(x_grid, pdf_inf, lw=2.5, color="black",
             label=r"$\pi = \mathcal{N}(0,\,\sigma^2/2)$")

axes[1].set_xlabel("$x$", fontsize=12)
axes[1].set_ylabel("$p(t,x)$", fontsize=12)
axes[1].set_title(r"Evolución de la densidad ($x_0 = 2$)", fontsize=12)
axes[1].legend(fontsize=9)

plt.tight_layout()
plt.savefig("ou_trajectories.png", dpi=150, bbox_inches="tight")
plt.show()
print("Guardado en ou_trajectories.png")
