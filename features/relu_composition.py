"""
Generate 5 matched 3D surface plots for the 'compositions of ReLU(s)' slide.

4 individual ReLU surfaces + their sum, all in a unified viridis style
matching the lecture's other 3D plots.

Output PNGs go directly into the slide images folder.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os

OUTDIR = os.path.join(
    os.path.expanduser("~"),
    "code/slides/390/spring26-slides/slides_introml-sp26-lec05/introml-sp26-lec05",
)

matplotlib.rcParams.update({
    'font.size': 12,
    'axes.labelsize': 14,
    'mathtext.fontset': 'cm',
    'figure.facecolor': 'white',
})

FIGSIZE = (5.5, 4.5)
PAD = dict(left=0.02, right=0.92, bottom=0.05, top=0.95)

# ReLU coefficients (from the existing slide images)
RELUS = [
    ([-1.0, -1.0], -1.0),   # ReLU(-x1 - x2 - 1)
    ([ 1.0, -1.0], -0.5),   # ReLU( x1 - x2 - 0.5)
    ([-1.0,  1.0], -0.5),   # ReLU(-x1 + x2 - 0.5)
    ([ 1.0,  1.0],  0.0),   # ReLU( x1 + x2)
]


def relu(z):
    return np.maximum(z, 0)


def make_surface(ax, Z, z_max):
    """Draw a viridis surface normalized to [0, z_max]."""
    norm = plt.Normalize(vmin=0, vmax=z_max)
    ax.plot_surface(
        X1, X2, Z,
        facecolors=plt.cm.viridis(norm(Z)),
        alpha=0.85,
        edgecolor='none',
        antialiased=True,
        shade=False,
    )


def style_ax(ax, z_max):
    ax.set_xlabel('$x_1$', labelpad=4)
    ax.set_ylabel('$x_2$', labelpad=4)
    ax.set_xlim(-2, 2)
    ax.set_ylim(-2, 2)
    ax.set_zlim(0, z_max)
    ax.set_xticks([-2, 0, 2])
    ax.set_yticks([-2, 0, 2])
    ax.set_zticks([])
    ax.set_box_aspect([1, 1, 0.6])
    ax.view_init(elev=25, azim=-60)
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.grid(True, color='#e0e0e0', linewidth=0.5)


N = 200
g = np.linspace(-2, 2, N)
X1, X2 = np.meshgrid(g, g)

# Compute all individual surfaces and the sum
surfaces = []
for (w, b) in RELUS:
    Z = relu(w[0] * X1 + w[1] * X2 + b)
    surfaces.append(Z)

Z_sum = sum(surfaces)
z_max_sum = float(Z_sum.max())
# Use same z_max for all plots so color scale is consistent
z_max = z_max_sum

# --- Individual plots ---
for i, Z in enumerate(surfaces, start=1):
    fig = plt.figure(figsize=FIGSIZE)
    ax = fig.add_subplot(111, projection='3d')
    make_surface(ax, Z, z_max)
    style_ax(ax, z_max)
    fig.subplots_adjust(**PAD)
    out = os.path.join(OUTDIR, f"relu_comp_{i}.png")
    fig.savefig(out, dpi=300, facecolor='white')
    plt.close(fig)
    print(f"Saved relu_comp_{i}.png")

# --- Sum plot ---
fig = plt.figure(figsize=FIGSIZE)
ax = fig.add_subplot(111, projection='3d')
make_surface(ax, Z_sum, z_max)
style_ax(ax, z_max)
fig.subplots_adjust(**PAD)
out = os.path.join(OUTDIR, "relu_comp_sum.png")
fig.savefig(out, dpi=300, facecolor='white')
plt.close(fig)
print("Saved relu_comp_sum.png")

print("\nDone! All 5 ReLU composition plots saved.")
