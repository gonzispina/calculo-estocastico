"""
4 figuras: proceso OU (Gamma → Gaussiana), score, EDE inversa, probability flow.
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.colors import LinearSegmentedColormap
from matplotlib import cm
from scipy import stats
from scipy.interpolate import RegularGridInterpolator
from PIL import Image

np.random.seed(42)

def crop_whitespace(filename, pad=12):
    """Recorta márgenes blancos de un PNG guardado."""
    img = Image.open(filename).convert('RGB')
    arr = np.array(img)
    non_white = np.any(arr < 248, axis=2)
    rows = np.where(non_white.any(axis=1))[0]
    cols = np.where(non_white.any(axis=0))[0]
    if len(rows) == 0 or len(cols) == 0:
        return
    r0 = max(0, rows[0]  - pad)
    r1 = min(arr.shape[0], rows[-1] + pad)
    c0 = max(0, cols[0]  - pad)
    c1 = min(arr.shape[1], cols[-1] + pad)
    img.crop((c0, r0, c1, r1)).save(filename)

# ── Parámetros ────────────────────────────────────────────────────────────────
ALPHA, SCALE = 2.0, 1.0
SHIFT  = ALPHA * SCALE          # traslada Gamma para tener media 0
SIGMA  = 5.0                    # σ_∞ = SIGMA/√2 ≈ 3.54 — Gaussiana ancha
T      = 1.0
sig2   = SIGMA**2               # = 25
sigma_stat = np.sqrt(sig2 / 2)  # desviación estacionaria ≈ 3.54

N_SLICES = 9
N_TRAJ   = 5

x_range = np.linspace(-12, 12, 180)

# ── Densidad marginal ─────────────────────────────────────────────────────────
def p0_pdf(x):
    return stats.gamma.pdf(x + SHIFT, a=ALPHA, scale=SCALE)

def p_marginal(x_arr, t, n_y=500):
    if t < 1e-6:
        return p0_pdf(x_arr)
    mu  = np.exp(-t)
    var = sig2 / 2 * (1 - np.exp(-2*t))
    y   = np.linspace(-SHIFT + 1e-3, 14.0, n_y)
    p0y = stats.gamma.pdf(y + SHIFT, a=ALPHA, scale=SCALE)
    X, Y = x_arr[:, None], y[None, :]
    G = stats.norm.pdf(X, loc=Y * mu, scale=np.sqrt(var))
    return np.trapezoid(G * p0y, y, axis=1)

def score_fn(x_arr, t, eps=0.12):
    lp = np.log(np.clip(p_marginal(x_arr + eps, t), 1e-15, None))
    lm = np.log(np.clip(p_marginal(x_arr - eps, t), 1e-15, None))
    return (lp - lm) / (2 * eps)

# ── Pre-cómputo + interpolador ────────────────────────────────────────────────
print("Pre-computando densidades y score...")
t_grid = np.linspace(0, T, 60)
x_grid = np.linspace(-13, 13, 180)
S_grid = np.column_stack([score_fn(x_grid, t) for t in t_grid])

_interp = RegularGridInterpolator(
    (x_grid, t_grid), S_grid, method='linear',
    bounds_error=False, fill_value=0.0)

def fast_score(x_val, t_val):
    xc = float(np.clip(x_val, x_grid[0], x_grid[-1]))
    tc = float(np.clip(t_val, 0.0, T))
    return _interp([[xc, tc]])[0]

print("Listo.")

# ── Cortes temporales ─────────────────────────────────────────────────────────
T_SLICES = np.linspace(0, T, N_SLICES)
_fwd = LinearSegmentedColormap.from_list('fwd', ['#c44020', '#2060a8'])
_rev = LinearSegmentedColormap.from_list('rev', ['#2060a8', '#c44020'])
COLORS_FWD = [_fwd(i / (N_SLICES - 1)) for i in range(N_SLICES)]
COLORS_REV = [_rev(i / (N_SLICES - 1)) for i in range(N_SLICES)]

def _poly(t_pos, t_dens, color, alpha=0.50):
    p  = p_marginal(x_range, t_dens)
    vs = ([[t_pos, x_range[0], 0.0]] +
          [[t_pos, x, z] for x, z in zip(x_range, p)] +
          [[t_pos, x_range[-1], 0.0]])
    return Poly3DCollection([vs], alpha=alpha,
                             facecolor=color, edgecolor=color, linewidth=0.9)

def draw_slices_fwd(ax):
    for t_s, clr in zip(T_SLICES, COLORS_FWD):
        ax.add_collection3d(_poly(t_s, t_s, clr))
        ax.plot([t_s]*len(x_range), x_range, p_marginal(x_range, t_s),
                color=clr, lw=1.6)

def draw_slices_rev(ax):
    for tau_s, clr in zip(T_SLICES, COLORS_REV):
        ax.add_collection3d(_poly(tau_s, T - tau_s, clr))
        ax.plot([tau_s]*len(x_range), x_range, p_marginal(x_range, T - tau_s),
                color=clr, lw=1.6)

# ── Config ejes 3D ────────────────────────────────────────────────────────────
def setup_3d(ax, title, azim=-55, xlabel='Tiempo  $t$'):
    for pane in [ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane]:
        pane.fill = False
        pane.set_edgecolor('none')
    ax.grid(False)
    ax.set_xlabel(xlabel, labelpad=10, fontsize=10)
    ax.set_ylabel('$x$',         labelpad=8,  fontsize=10)
    ax.set_zlabel('$p_t(x)$',   labelpad=8,  fontsize=10)
    ax.set_title(title, fontsize=10, pad=2)
    ax.view_init(elev=22, azim=azim)
    ax.set_box_aspect([2.5, 1.5, 1])
    ax.set_zlim(-0.005, None)

# ── Simulaciones ──────────────────────────────────────────────────────────────
def sim_forward(x0, n=400):
    dt = T / n
    x = np.zeros(n+1); x[0] = x0
    for k in range(n):
        x[k+1] = x[k] * (1-dt) + SIGMA * np.sqrt(dt) * np.random.randn()
    return np.linspace(0, T, n+1), x

def sim_ou_bridge(xT, x0=0.0, n=400):
    """OU condicionado en X_T = xT (h-transform de Doob).
    drift = -x + σ² · ∇_x log p(xT | x, t)
          = -x + σ² · (xT - x·e^{-(T-t)}) · e^{-(T-t)} / (σ²/2·(1-e^{-2(T-t)}))
    """
    dt = T / n
    t_arr = np.linspace(0, T, n+1)
    x = np.zeros(n+1); x[0] = x0
    for k in range(n - 1):
        tau = T - t_arr[k]
        var_tau = sig2 / 2 * (1 - np.exp(-2 * tau))
        mu_tau  = x[k] * np.exp(-tau)
        guidance = (xT - mu_tau) * np.exp(-tau) / var_tau
        x[k+1] = x[k] + (-x[k] + sig2 * guidance) * dt + SIGMA * np.sqrt(dt) * np.random.randn()
    x[-1] = xT
    return t_arr, x

def sim_reverse_sde(x0, n=500):
    """d X̂_τ = [X̂ + σ²·score(X̂, T-τ)]dτ + σ dB̄."""
    dt = T / n
    tau = np.linspace(0, T, n+1)
    x = np.zeros(n+1); x[0] = x0
    for k in range(n):
        s = float(np.clip(fast_score(x[k], T - tau[k]), -5, 5))
        x[k+1] = x[k] + (x[k] + sig2 * s) * dt + SIGMA * np.sqrt(dt) * np.random.randn()
    return tau, x

def sim_pflow(x0, n=500):
    """d X̂_τ = [X̂ + (σ²/2)·score(X̂, T-τ)]dτ  (sin ruido)."""
    dt = T / n
    tau = np.linspace(0, T, n+1)
    x = np.zeros(n+1); x[0] = x0
    for k in range(n):
        s = float(np.clip(fast_score(x[k], T - tau[k]), -5, 5))
        x[k+1] = x[k] + (x[k] + (sig2/2) * s) * dt
    return tau, x

# Puntos iniciales: cuantiles representativos de cada distribución
_q = np.linspace(0.15, 0.85, N_TRAJ)
x0_fwd = stats.gamma.ppf(_q, a=ALPHA, scale=SCALE) - SHIFT
x0_rev = stats.norm.ppf(_q, loc=0, scale=sigma_stat)
TRAJ_C  = [cm.tab10(i / N_TRAJ) for i in range(N_TRAJ)]

print("Simulando trayectorias...")
trajs_fwd = [sim_forward(x0) for x0 in x0_fwd]
trajs_rev = [sim_reverse_sde(x0) for x0 in x0_rev]
_q_pf  = np.linspace(0.05, 0.95, N_TRAJ)   # más dispersas sobre la Gaussiana
x0_pf  = stats.norm.ppf(_q_pf, loc=0, scale=sigma_stat)
trajs_pf  = [sim_pflow(x0) for x0 in x0_pf]
print("Listo.")

# ════════════════════════════════════════════════════════════════════════════════
# Figura 1: Trayectorias forward
# ════════════════════════════════════════════════════════════════════════════════
fig1, ax1 = plt.subplots(subplot_kw={'projection': '3d'}, figsize=(13, 7))
draw_slices_fwd(ax1)
for (t_t, x_t), clr in zip(trajs_fwd, TRAJ_C):
    ax1.plot(t_t, x_t, np.zeros_like(t_t), color=clr, lw=1.8, alpha=0.88)
setup_3d(ax1, '')
fig1.savefig('fig1_forward.png', dpi=150, bbox_inches='tight', pad_inches=0.05)
crop_whitespace('fig1_forward.png')
print("Guardado: fig1_forward.png")

# ════════════════════════════════════════════════════════════════════════════════
# Figura 2: Score + marginales laterales + trayectoria
# ════════════════════════════════════════════════════════════════════════════════
ylims = (-12, 12)
x_plot = np.linspace(ylims[0], ylims[1], 300)
p0_plot = p0_pdf(x_plot)
pT_plot = stats.norm.pdf(x_plot, loc=0, scale=sigma_stat)

t_q = np.linspace(0, T, 32)
x_q = np.linspace(-10, 10, 22)
T_q, X_q = np.meshgrid(t_q, x_q)
P_q = np.column_stack([p_marginal(x_q, t) for t in t_q])
S_q = np.column_stack([score_fn(x_q, t) for t in t_q])   # sin máscara de densidad
sc95 = np.percentile(np.abs(S_q[np.isfinite(S_q)]), 92)
arrow_len = np.clip(S_q, -sc95, sc95) * (0.65 / sc95)

fig2 = plt.figure(figsize=(16, 9))
gs2  = gridspec.GridSpec(1, 3, width_ratios=[1, 5, 1], wspace=0.04)
ax_gam = fig2.add_subplot(gs2[0, 0])
ax_qv  = fig2.add_subplot(gs2[0, 1], sharey=ax_gam)
ax_nor = fig2.add_subplot(gs2[0, 2], sharey=ax_gam)

# Gamma — densidad a la izquierda, pico apunta hacia el centro
ax_gam.fill_betweenx(x_plot, p0_plot, 0, color='#c44020', alpha=0.35)
ax_gam.plot(p0_plot, x_plot, color='#c44020', lw=2)
ax_gam.invert_xaxis()
ax_gam.set_ylim(*ylims)
ax_gam.axhline(0, color='gray', lw=0.7, ls=':', alpha=0.5)
ax_gam.set_ylabel('$x$', fontsize=12)
ax_gam.set_title('$p_0(x)$\nGamma', fontsize=10, color='#c44020')
ax_gam.tick_params(labelbottom=False, bottom=False)

# Centro — campo del score + trayectoria
ax_qv.contourf(T_q, X_q, P_q, levels=20, cmap='Blues', alpha=0.28)
ax_qv.quiver(
    T_q, X_q,
    np.zeros_like(S_q), arrow_len,
    np.abs(S_q),
    cmap='YlOrRd', clim=(0, sc95),
    units='xy', scale=1,
    width=0.06, headwidth=4, headlength=5, alpha=0.85)

# Bridges condicionados en xT: endpoint controlado dentro del rango visible
for xT_target in [-7.5, -2.5, 3.5, 7.5]:
    tt, xt = sim_ou_bridge(xT_target, x0=0.0)
    ax_qv.plot(tt, xt, color='k', lw=1.8, alpha=0.80, zorder=5)
    ax_qv.plot(T, xT_target, 'o', color='k', ms=5, zorder=6)
ax_qv.axhline(0, color='gray', lw=0.7, ls=':', alpha=0.5)
ax_qv.set_xlabel('Tiempo $t$', fontsize=11)
ax_qv.set_xlim(0, T)
ax_qv.set_ylim(*ylims)
ax_qv.tick_params(labelleft=False)
ax_qv.set_title('', fontsize=10)
ax_qv.legend(fontsize=9, loc='upper right', handles=[])

# Gaussiana — densidad a la derecha, pico apunta hacia el centro
ax_nor.fill_betweenx(x_plot, 0, pT_plot, color='#2060a8', alpha=0.35)
ax_nor.plot(pT_plot, x_plot, color='#2060a8', lw=2)
ax_nor.axhline(0, color='gray', lw=0.7, ls=':', alpha=0.5)
ax_nor.set_title('$p_T(x)$\nGaussiana', fontsize=10, color='#2060a8')
ax_nor.tick_params(labelbottom=False, bottom=False, labelleft=False)

fig2.savefig('fig2_score.png', dpi=150, bbox_inches='tight', pad_inches=0.05)
crop_whitespace('fig2_score.png')
print("Guardado: fig2_score.png")

# ════════════════════════════════════════════════════════════════════════════════
# Figura 3: EDE inversa de Anderson
# ════════════════════════════════════════════════════════════════════════════════
fig3, ax3 = plt.subplots(subplot_kw={'projection': '3d'}, figsize=(13, 7))
draw_slices_rev(ax3)
for (tau_t, x_t), clr in zip(trajs_rev, TRAJ_C):
    ax3.plot(tau_t, x_t, np.zeros_like(tau_t), color=clr, lw=1.8, alpha=0.88)
setup_3d(ax3, '', azim=-145, xlabel=r'$\tau$ (tiempo inverso)')
fig3.savefig('fig3_reverse_sde.png', dpi=150, bbox_inches='tight', pad_inches=0.05)
crop_whitespace('fig3_reverse_sde.png')
print("Guardado: fig3_reverse_sde.png")

# ════════════════════════════════════════════════════════════════════════════════
# Figura 4: Probability flow ODE
# ════════════════════════════════════════════════════════════════════════════════
fig4, ax4 = plt.subplots(subplot_kw={'projection': '3d'}, figsize=(13, 7))
draw_slices_rev(ax4)
for (tau_t, x_t), clr in zip(trajs_pf, TRAJ_C):
    ax4.plot(tau_t, x_t, np.zeros_like(tau_t), color=clr, lw=2.0, alpha=0.92)
setup_3d(ax4, '', azim=-145, xlabel=r'$\tau$ (tiempo inverso)')
fig4.savefig('fig4_pflow.png', dpi=150, bbox_inches='tight', pad_inches=0.05)
crop_whitespace('fig4_pflow.png')
print("Guardado: fig4_pflow.png")

print("\nTodos guardados: fig1–fig4.png")
