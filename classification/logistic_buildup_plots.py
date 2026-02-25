"""
Generate plots for the 3-column logistic classifier buildup slide.

Produces 6 PNGs:
  1. z_1d.png        — z = θx + θ₀ (straight line)
  2. z_2d.png        — z = θ₁x₁ + θ₂x₂ + θ₀ (3D plane)
  3. sigmoid_1d.png  — σ(θx + θ₀) vs x  (1D feature)
  4. sigmoid_2d.png  — 3D surface of σ(θ₁x₁ + θ₂x₂ + θ₀)
  5. data_1d.png     — 1D data on number line with decision point
  6. data_2d.png     — 2D scatter with linear decision boundary
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import LinearSegmentedColormap
import matplotlib
matplotlib.rcParams.update({
    'font.size': 12,
    'axes.labelsize': 14,
    'mathtext.fontset': 'cm',
    'figure.facecolor': '#fafafa',
})

# --- Parameters ---
# 1D: positive θ for standard increasing S-curve
theta_1d = 3.0
theta0_1d = 0.0

# 2D: parameters for a nice-looking 3D surface
theta1_2d = 3.0
theta2_2d = 2.0
theta0_2d = 0.0

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Custom colorscale matching the demo: blue → white → red
demo_cmap = LinearSegmentedColormap.from_list('demo', [
    (0.0, '#7eafd4'),
    (0.4, '#c8dce8'),
    (0.5, '#f0f0f0'),
    (0.6, '#e8c8c8'),
    (1.0, '#d47e7e'),
])


# ===================== 0a. z 1D (straight line) =====================
fig, ax = plt.subplots(figsize=(5, 3))
x = np.linspace(-5, 5, 300)
z = theta_1d * x + theta0_1d

ax.plot(x, z, color='#3a7ebf', linewidth=2.5)
ax.axhline(0, color='#999', linewidth=1, linestyle='--', alpha=0.5)
boundary = -theta0_1d / theta_1d if theta_1d != 0 else None
if boundary is not None and -5 <= boundary <= 5:
    ax.axvline(boundary, color='#999', linewidth=1, linestyle='--', alpha=0.5)

ax.set_xlabel('$x$')
ax.set_ylabel('$z$')
ax.set_xlim(-5, 5)
ax.set_xticks([-4, -2, 0, 2, 4])
ax.set_facecolor('#fafafa')
ax.grid(True, color='#eee', linewidth=0.5)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
fig.tight_layout()
fig.savefig('z_1d.png', dpi=200, bbox_inches='tight', facecolor='#fafafa')
plt.close(fig)
print("Saved z_1d.png")


# ===================== 0b. z 2D (3D plane) =====================
fig = plt.figure(figsize=(5.5, 4.5))
ax = fig.add_subplot(111, projection='3d')

N = 40
g = np.linspace(-2, 2, N)
X1, X2 = np.meshgrid(g, g)
Z_plane = theta1_2d * X1 + theta2_2d * X2 + theta0_2d

# Color plane by sign: blue for negative z, red for positive z
Z_norm = (Z_plane - Z_plane.min()) / (Z_plane.max() - Z_plane.min())
ax.plot_surface(X1, X2, Z_plane, facecolors=demo_cmap(Z_norm),
                alpha=0.85, edgecolor='none', antialiased=True, shade=False)
# z=0 contour on the plane
if theta2_2d != 0:
    bx = np.linspace(-2, 2, 100)
    by = -(theta1_2d * bx + theta0_2d) / theta2_2d
    mask = (by >= -2) & (by <= 2)
    ax.plot(bx[mask], by[mask], np.zeros_like(bx[mask]), 'k--', linewidth=1.5, alpha=0.6)

ax.set_xlabel('$x_1$', labelpad=4)
ax.set_ylabel('$x_2$', labelpad=4)
ax.set_zlabel('$z$', labelpad=2)
ax.set_xlim(-2, 2)
ax.set_ylim(-2, 2)
ax.set_xticks([-2, 0, 2])
ax.set_yticks([-2, 0, 2])
ax.set_box_aspect([1, 1, 0.6])
ax.view_init(elev=35, azim=-75)
ax.xaxis.pane.fill = False
ax.yaxis.pane.fill = False
ax.zaxis.pane.fill = False
ax.grid(True, color='#e0e0e0', linewidth=0.5)
fig.tight_layout()
fig.savefig('z_2d.png', dpi=200, bbox_inches='tight', facecolor='#fafafa')
plt.close(fig)
print("Saved z_2d.png")


# ===================== 1. Sigmoid 1D =====================
fig, ax = plt.subplots(figsize=(5, 3))
x = np.linspace(-5, 5, 800)
z = theta_1d * x + theta0_1d
sig = sigmoid(z)

ax.plot(x, sig, color='#3a7ebf', linewidth=2.5)
# decision boundary line
boundary = -theta0_1d / theta_1d if theta_1d != 0 else None
if boundary is not None and -5 <= boundary <= 5:
    ax.axvline(boundary, color='#999', linewidth=1.5, linestyle='--')

ax.set_xlabel('$x$')
ax.set_ylabel(r'$\sigma$')
ax.set_ylim(-0.1, 1.1)
ax.set_xlim(-5, 5)
ax.set_xticks([-4, -2, 0, 2, 4])
ax.set_yticks([0, 0.5, 1])
ax.set_facecolor('#fafafa')
ax.grid(True, color='#eee', linewidth=0.5)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
fig.tight_layout()
fig.savefig('sigmoid_1d.png', dpi=200, bbox_inches='tight', facecolor='#fafafa')
plt.close(fig)
print("Saved sigmoid_1d.png")


# ===================== 2. Sigmoid 2D (3D surface) =====================
fig = plt.figure(figsize=(5.5, 4.5))
ax = fig.add_subplot(111, projection='3d')

N = 80
g = np.linspace(-2, 2, N)
X1, X2 = np.meshgrid(g, g)
Z = sigmoid(theta1_2d * X1 + theta2_2d * X2 + theta0_2d)

ax.plot_surface(X1, X2, Z, cmap=demo_cmap, vmin=0, vmax=1,
                alpha=0.85, edgecolor='none', antialiased=True)

ax.set_xlabel('$x_1$', labelpad=4)
ax.set_ylabel('$x_2$', labelpad=4)
ax.set_zlabel('')
ax.set_xlim(-2, 2)
ax.set_ylim(-2, 2)
ax.set_zlim(-0.1, 1.1)
ax.set_xticks([-2, 0, 2])
ax.set_yticks([-2, 0, 2])
ax.set_zticks([0, 0.5, 1])
ax.set_box_aspect([1, 1, 0.6])
ax.view_init(elev=35, azim=-75)
ax.xaxis.pane.fill = False
ax.yaxis.pane.fill = False
ax.zaxis.pane.fill = False
ax.grid(True, color='#e0e0e0', linewidth=0.5)
fig.tight_layout()
fig.savefig('sigmoid_2d.png', dpi=200, bbox_inches='tight', facecolor='#fafafa')
plt.close(fig)
print("Saved sigmoid_2d.png")


# ===================== 3. Data 1D =====================
# Use different θ for the separator images (cleaner separation)
theta_sep = 2.0
theta0_sep = -1.0
dec_x = -theta0_sep / theta_sep  # = 0.5

np.random.seed(42)
n_pos = 8
n_neg = 8
pos_x = np.random.normal(1.5, 0.6, n_pos)
neg_x = np.random.normal(-1.0, 0.7, n_neg)

fig, ax = plt.subplots(figsize=(5, 2.0))
ax.scatter(pos_x, np.zeros_like(pos_x), marker='+', color='red', s=100, linewidths=2, zorder=5, label='1')
ax.scatter(neg_x, np.zeros_like(neg_x), marker='o', color='#3c78d8', s=60, facecolors='none', linewidths=1.5, zorder=5, label='0')
ax.axvline(dec_x, color='#3c78d8', linewidth=2, linestyle='-', alpha=0.8)
ax.annotate(r'$\theta x + \theta_0 = 0$', xy=(dec_x, 0), xytext=(dec_x + 0.8, 0.15),
            fontsize=11, color='#3c78d8',
            arrowprops=dict(arrowstyle='->', color='#3c78d8', lw=1.2))
ax.axhline(0, color='gray', linewidth=0.5)
ax.set_xlabel('$x$')
ax.set_xlim(-4, 4)
ax.set_xticks([-3, -1, 1, 3])
ax.set_ylim(-0.3, 0.35)
ax.set_yticks([])
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.legend(loc='upper left', fontsize=10, framealpha=0.8)
fig.tight_layout()
fig.savefig('data_1d.png', dpi=200, bbox_inches='tight', facecolor='white')
plt.close(fig)
print("Saved data_1d.png")


# ===================== 4. Data 2D =====================
theta_2d_sep = np.array([1.5, 1.0])
theta0_2d_sep = -0.5

np.random.seed(7)
n2 = 15
pos_2d = np.random.normal([1.2, 1.0], 0.7, (n2, 2))
neg_2d = np.random.normal([-0.8, -0.5], 0.7, (n2, 2))

fig, ax = plt.subplots(figsize=(5, 4.5))
ax.scatter(pos_2d[:, 0], pos_2d[:, 1], marker='+', color='red', s=100, linewidths=2, zorder=5, label='1')
ax.scatter(neg_2d[:, 0], neg_2d[:, 1], marker='o', color='#3c78d8', s=60, facecolors='none', linewidths=1.5, zorder=5, label='0')

bx = np.linspace(-3, 3, 100)
by = -(theta_2d_sep[0] * bx + theta0_2d_sep) / theta_2d_sep[1]
mask = (by >= -3) & (by <= 3)
ax.plot(bx[mask], by[mask], color='#3c78d8', linewidth=2.5, label=r'$\theta^\top x + \theta_0 = 0$')

ax.set_xlabel('$x_1$')
ax.set_ylabel('$x_2$')
ax.set_xlim(-3, 3)
ax.set_ylim(-3, 3)
ax.set_xticks([-2, 0, 2])
ax.set_yticks([-2, 0, 2])
ax.set_aspect('equal')
ax.legend(loc='upper left', fontsize=10, framealpha=0.8)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
fig.tight_layout()
fig.savefig('data_2d.png', dpi=200, bbox_inches='tight', facecolor='white')
plt.close(fig)
print("Saved data_2d.png")

print("\nDone! All 4 plots saved.")
