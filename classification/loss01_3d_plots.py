"""
Generate two 3D hyperplane plots for the '0-1 loss' slide,
matching the interactive demo's color scheme.

Plot 1 (top): hyperplane correctly classifies both points -> L_01 = 0
Plot 2 (bottom): hyperplane misclassifies one point -> one has L_01 = 1

Theta values are chosen so the plane fills the z-range [-5, 5] without
clipping at any corner of the x1, x2 in [-2, 2] domain.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import os

OUTDIR = os.path.dirname(os.path.abspath(__file__))

# Demo colorscale: blue -> white -> red
demo_cmap = LinearSegmentedColormap.from_list('demo', [
    (0.0, '#7eafd4'),
    (0.4, '#c8dce8'),
    (0.5, '#f0f0f0'),
    (0.6, '#e8c8c8'),
    (1.0, '#d47e7e'),
])

# Marker colors matching demo
POS_COLOR = '#e74c3c'   # red -- positive class (+)
NEG_COLOR = '#3a7ebf'   # blue -- negative class (o)

matplotlib.rcParams.update({
    'font.size': 12,
    'axes.labelsize': 14,
    'mathtext.fontset': 'cm',
    'figure.facecolor': 'white',
})

# Fixed layout so both PNGs have identical pixel dimensions
FIGSIZE = (5.5, 4.5)
PAD = dict(left=0.02, right=0.92, bottom=0.05, top=0.95)


def make_3d_plot(theta, theta0, pos_pt, neg_pt, elev, azim, filename):
    """
    Draw a 3D surface z = theta . x + theta0, with one + and one o marker.
    """
    fig = plt.figure(figsize=FIGSIZE)
    ax = fig.add_subplot(111, projection='3d')

    N = 50
    g = np.linspace(-2, 2, N)
    X1, X2 = np.meshgrid(g, g)
    Z = theta[0] * X1 + theta[1] * X2 + theta0

    # Normalize Z for colormap: map z=0 to colormap center (0.5)
    # so blue/red split always corresponds to negative/positive z
    z_absmax = max(abs(Z.min()), abs(Z.max()), 1e-6)
    Z_norm = 0.5 + 0.5 * Z / z_absmax
    Z_norm = np.clip(Z_norm, 0, 1)

    ax.plot_surface(X1, X2, Z, facecolors=demo_cmap(Z_norm),
                    alpha=0.82, edgecolor='none', antialiased=True, shade=False)

    # Decision boundary: z = 0 contour on the plane
    # Handle both axis-aligned and general cases
    if abs(theta[1]) > 1e-10:
        bx = np.linspace(-2, 2, 200)
        by = -(theta[0] * bx + theta0) / theta[1]
        mask = (by >= -2) & (by <= 2)
        bz = np.zeros_like(bx)
        ax.plot(bx[mask], by[mask], bz[mask], 'k-', linewidth=2.5, alpha=0.7)
    elif abs(theta[0]) > 1e-10:
        by = np.linspace(-2, 2, 200)
        bx_val = -theta0 / theta[0]
        if -2 <= bx_val <= 2:
            bx = np.full_like(by, bx_val)
            bz = np.zeros_like(by)
            ax.plot(bx, by, bz, 'k-', linewidth=2.5, alpha=0.7)

    # Data points -- place above surface so they pop
    LIFT = 0.5
    px, py = pos_pt
    pz = theta[0] * px + theta[1] * py + theta0
    # White halo then red + on top
    ax.plot([px], [py], [pz + LIFT], marker='+', color='white',
            ms=22, mew=7, zorder=9, linestyle='none')
    ax.plot([px], [py], [pz + LIFT], marker='+', color=POS_COLOR,
            ms=20, mew=5, zorder=10, linestyle='none')

    nx, ny = neg_pt
    nz = theta[0] * nx + theta[1] * ny + theta0
    ax.plot([nx], [ny], [nz + LIFT], marker='o', color=NEG_COLOR,
            ms=14, mew=2, mfc=NEG_COLOR, mec='white',
            zorder=10, linestyle='none')

    ax.set_xlabel('$x_1$', labelpad=4)
    ax.set_ylabel('$x_2$', labelpad=4)
    ax.set_zlabel('$z$', labelpad=2)
    ax.set_xlim(-2, 2)
    ax.set_ylim(-2, 2)
    ax.set_zlim(-5, 5)
    ax.set_xticks([-2, -1, 0, 1, 2])
    ax.set_yticks([-2, -1, 0, 1, 2])
    ax.set_box_aspect([1, 1, 0.8])
    ax.view_init(elev=elev, azim=azim)
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.grid(True, color='#e0e0e0', linewidth=0.5)

    fig.subplots_adjust(**PAD)
    out = os.path.join(OUTDIR, filename)
    fig.savefig(out, dpi=200, facecolor='white')
    plt.close(fig)
    print(f"Saved {filename}")


# --- Scene parameters ---
pos_pt = (1.2, 0.8)    # positive class (+)
neg_pt = (-0.8, -1.2)  # negative class (o)

# Plot 1: "Good" theta -- correctly separates both points
# Corners: (2,2)->2.5, (2,-2)->-0.5, (-2,2)->-1.9, (-2,-2)->-4.9
# All within [-5, 5] -- no clipping!
# z_pos = 1.1*1.2 + 0.75*0.8 - 1.2 = 0.72 (positive, correct)
# z_neg = 1.1*(-0.8) + 0.75*(-1.2) - 1.2 = -2.98 (negative, correct)
theta_good = np.array([1.1, 0.75])
theta0_good = -1.2
make_3d_plot(theta_good, theta0_good, pos_pt, neg_pt,
             elev=22, azim=-55,
             filename="loss01_good.png")

# Plot 2: "Bad" theta -- the + ends up on the negative (blue) side
# Corners: (2,2)->-0.5, (2,-2)->-4.5, (-2,2)->3.5, (-2,-2)->-0.5
# All within [-5, 5] -- no clipping!
# z_pos = -1*1.2 + 1*0.8 - 0.5 = -0.9 (negative, misclassified!)
# z_neg = -1*(-0.8) + 1*(-1.2) - 0.5 = -0.9 (negative, correct)
theta_bad = np.array([-1.0, 1.0])
theta0_bad = -0.5
make_3d_plot(theta_bad, theta0_bad, pos_pt, neg_pt,
             elev=22, azim=-55,
             filename="loss01_bad.png")

print("\nDone! Both plots saved.")
