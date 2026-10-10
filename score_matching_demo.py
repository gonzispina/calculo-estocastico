import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from scipy import stats

torch.manual_seed(42)
np.random.seed(42)

# ── p0: Gamma desplazada para tener media 0 ────────────────────────────────────
ALPHA, SCALE = 2.0, 1.0   # Gamma(shape, scale),  media = ALPHA * SCALE = 2
SHIFT = ALPHA * SCALE

def sample_p0(n):
    x = np.random.gamma(ALPHA, SCALE, size=n) - SHIFT
    return torch.tensor(x, dtype=torch.float32).unsqueeze(1)

def p0_pdf(x):
    return stats.gamma.pdf(x + SHIFT, a=ALPHA, scale=SCALE)

# ── Proceso OU: dX_t = -X_t dt + sigma dB_t ───────────────────────────────────
# Distribucion estacionaria: N(0, sigma^2/2)
# X_t | X_0 = x0  ~  N(x0 * e^{-t},  sigma^2/2 * (1 - e^{-2t}))
SIGMA = 1.0
T = 5.0  # suficientemente grande para que p_T ≈ N(0, 1/2)

def ou_params(x0, t):
    mean = x0 * torch.exp(-t)
    var  = (SIGMA**2 / 2) * (1 - torch.exp(-2 * t))
    return mean, var.clamp(min=1e-5)

def sample_xt(x0, t):
    mean, var = ou_params(x0, t)
    return mean + torch.sqrt(var) * torch.randn_like(x0)

def score_condicional(xt, x0, t):
    """∇_{x_t} log p(x_t | x_0) = -(x_t - media) / varianza"""
    mean, var = ou_params(x0, t)
    return -(xt - mean) / var

# ── Red neuronal: s_theta(t, x) ───────────────────────────────────────────────
class ScoreNet(nn.Module):
    def __init__(self, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, t, x):
        return self.net(torch.cat([t, x], dim=1))

# ── Entrenamiento: denoising score matching ────────────────────────────────────
def train(n_epochs=4000, batch=512, lr=2e-3):
    model = ScoreNet()
    opt   = optim.Adam(model.parameters(), lr=lr)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_epochs)
    losses = []

    for epoch in range(n_epochs):
        x0 = sample_p0(batch)
        t  = torch.rand(batch, 1) * T * 0.999 + T * 0.001  # t in (0, T)

        with torch.no_grad():
            xt  = sample_xt(x0, t)
            tgt = score_condicional(xt, x0, t)
            # lambda(t) = Var(t): pondera el loss uniformemente en todas las escalas
            _, var = ou_params(x0, t)

        pred = model(t, xt)
        loss = (var * (pred - tgt) ** 2).mean()

        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()

        losses.append(loss.item())
        if epoch % 500 == 0:
            print(f"[{epoch:5d}] loss = {loss.item():.5f}")

    return model, losses

# ── Generacion: Euler-Maruyama sobre la EDE inversa ───────────────────────────
# EDE inversa: dX_hat = [X_hat + sigma^2 * s_theta(T-t, X_hat)] dt + sigma dB_hat
# Integramos de t=0 a t=T (tiempo invertido).
@torch.no_grad()
def generate(model, n_samples=5000, n_steps=1000):
    model.eval()
    dt = T / n_steps
    x  = torch.randn(n_samples, 1) * (SIGMA / 2**0.5)  # x_T ~ N(0, sigma^2/2)

    for k in range(n_steps):
        t_fwd = T - k * dt                              # tiempo forward: T -> 0
        t_ten = torch.full((n_samples, 1), t_fwd)
        s     = model(t_ten, x)
        drift = x + SIGMA**2 * s                        # -u(T-t,x) + sigma^2 s_theta
        z     = torch.randn_like(x)
        x     = x + drift * dt + SIGMA * dt**0.5 * z

    return x.numpy().flatten()

# ── Main ───────────────────────────────────────────────────────────────────────
print("Entrenando score network...")
model, losses = train(n_epochs=10000)

print("Generando muestras...")
gen  = generate(model, n_samples=5000, n_steps=1000)
real = sample_p0(5000).numpy().flatten()

# KS test como verificacion cuantitativa
ks_stat, ks_p = stats.ks_2samp(real, gen)
print(f"\nKS test: estadistico = {ks_stat:.4f},  p-valor = {ks_p:.4f}")

# ── Figura ─────────────────────────────────────────────────────────────────────
x_grid   = np.linspace(-3, 8, 500)
true_pdf = p0_pdf(x_grid)

fig, axes = plt.subplots(1, 2, figsize=(10, 4))

axes[0].hist(real, bins=60, density=True, alpha=0.6, color="steelblue",
             label="Muestras de $p_0$")
axes[0].plot(x_grid, true_pdf, "k-", lw=2, label="Densidad teórica")
axes[0].set_title("Distribución original $p_0$  [Gamma desplazada]")
axes[0].legend()

axes[1].hist(gen, bins=60, density=True, alpha=0.6, color="darkorange",
             label="Muestras generadas")
axes[1].plot(x_grid, true_pdf, "k-", lw=2, label="Densidad teórica")
axes[1].set_title(f"Generadas vs. $p_0$   (KS $p={ks_p:.3f}$)")
axes[1].legend()

plt.tight_layout()
plt.savefig("score_matching_demo.png", dpi=150, bbox_inches="tight")
plt.show()
print("Guardado en score_matching_demo.png")
