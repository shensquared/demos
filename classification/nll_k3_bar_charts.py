"""
Generate K=3 bar charts for the NLL loss visual slide.
Matches the Isola slide's black-on-white, serif-font style.

Classes: hot dog, pizza, veggie
True label: hot dog  =>  y = [1, 0, 0]
Softmax output: g = [0.1, 0.7, 0.2]  (wrong prediction — pizza highest)
-log(g) ≈ [2.30, 0.36, 1.61]
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import os

# Output directory (same as this script)
OUTDIR = os.path.dirname(os.path.abspath(__file__))

# Style constants
FONT_FAMILY = "serif"
BAR_COLOR = "black"
RED_COLOR = "#cc3333"
BG_COLOR = "white"
LABEL_SIZE = 28
TICK_SIZE = 24
BAR_HEIGHT = 0.55

plt.rcParams.update({
    "font.family": FONT_FAMILY,
    "text.color": "black",
    "axes.labelcolor": "black",
    "xtick.color": "black",
    "ytick.color": "black",
    "figure.facecolor": BG_COLOR,
    "axes.facecolor": BG_COLOR,
    "savefig.facecolor": BG_COLOR,
})

labels = ["hot dog", "pizza", "veggie"]
y_positions = [2, 1, 0]  # top to bottom

# Probabilities and derived values
g = [0.1, 0.7, 0.2]
log_g = [np.log(v) for v in g]        # [-0.357, -1.609, -2.303]
neg_log_g = [-v for v in log_g]        # [ 0.357,  1.609,  2.303]

# Boundaries
LEFT_EDGE = -2.8   # represents -∞ on log(g) charts
RIGHT_EDGE = 2.8   # represents +∞ on -log(g) charts

# ── Fixed layout for all 3-bar charts ────────────────────────────────
# Use identical figsize, ylim, and subplots_adjust so the bar rows
# land at exactly the same pixel position across all PNGs.
THREE_BAR_FIGSIZE = (5.2, 4.5)
THREE_BAR_YLIM = (-0.5, 2.8)
THREE_BAR_ADJUST = dict(left=0.38, right=0.95, top=0.97, bottom=0.10)


def make_three_bar_ax():
    """Create a figure+axes with the shared 3-bar layout."""
    fig, ax = plt.subplots(figsize=THREE_BAR_FIGSIZE)
    ax.set_ylim(*THREE_BAR_YLIM)
    ax.set_yticks([])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(1.2)
    ax.spines["bottom"].set_linewidth(1.2)
    ax.tick_params(axis="x", length=4, width=1.2)
    ax.tick_params(axis="y", length=0)
    return fig, ax


def add_class_labels(ax, bold_index=0):
    """Add class name labels to the left of the y-axis."""
    for i, (label, yp) in enumerate(zip(labels, y_positions)):
        weight = "bold" if i == bold_index else "normal"
        ax.text(-0.04, yp, label, ha="right", va="center",
                fontsize=LABEL_SIZE, fontweight=weight,
                transform=ax.get_yaxis_transform())


def save_three_bar(fig, filename):
    """Save a 3-bar chart with the shared layout."""
    fig.subplots_adjust(**THREE_BAR_ADJUST)
    fig.savefig(os.path.join(OUTDIR, filename), dpi=200, transparent=False)
    plt.close(fig)
    print(f"Saved {filename}")


# ── Chart 1: y one-hot ──────────────────────────────────────────────

fig, ax = make_three_bar_ax()
ax.barh(y_positions, [1, 0, 0], height=BAR_HEIGHT, color=BAR_COLOR)
add_class_labels(ax)
ax.set_xlim(0, 1)
ax.set_xticks([0, 1])
ax.set_xticklabels(["0", "1"], fontsize=TICK_SIZE)
save_three_bar(fig, "nll_k3_y_onehot.png")


# ── Chart 2: log(g)  (Isola-style, -∞ → 0) ─────────────────────────

fig, ax = make_three_bar_ax()
widths = [v - LEFT_EDGE for v in log_g]
ax.barh(y_positions, widths, height=BAR_HEIGHT, color=BAR_COLOR, left=LEFT_EDGE)
add_class_labels(ax)
ax.set_xlim(LEFT_EDGE, 0)
ax.set_xticks([LEFT_EDGE, 0])
ax.set_xticklabels([r"$-\infty$", "0"], fontsize=TICK_SIZE)
save_three_bar(fig, "nll_k3_log_g.png")


# ── Chart 2b: g softmax output (0 → 1) ──────────────────────────────

fig, ax = make_three_bar_ax()
ax.barh(y_positions, g, height=BAR_HEIGHT, color=BAR_COLOR)
add_class_labels(ax)
ax.set_xlim(0, 1)
ax.set_xticks([0, 1])
ax.set_xticklabels(["0", "1"], fontsize=TICK_SIZE)
save_three_bar(fig, "nll_k3_g_softmax.png")


# ── Chart 3: -log(g)  (no-sign-flip version, 0 → +∞) ───────────────

fig, ax = make_three_bar_ax()
ax.barh(y_positions, neg_log_g, height=BAR_HEIGHT, color=BAR_COLOR)
add_class_labels(ax)
ax.set_xlim(0, RIGHT_EDGE)
ax.set_xticks([0, RIGHT_EDGE])
ax.set_xticklabels(["0", r"$\infty$"], fontsize=TICK_SIZE)
save_three_bar(fig, "nll_k3_neglog_g.png")


# ── Fixed layout for single-bar loss charts ──────────────────────────
# Same height and vertical layout as three-bar charts so the hot-dog
# bar lands at the exact same pixel row.  Narrower width.
LOSS_FIGSIZE = (3.8, 4.5)
LOSS_ADJUST = dict(left=0.48, right=0.92, top=0.97, bottom=0.10)

HOTDOG_Y = 2  # same row as in three-bar charts


def make_loss_ax():
    """Create a figure+axes that shares the three-bar vertical layout."""
    fig, ax = plt.subplots(figsize=LOSS_FIGSIZE)
    ax.set_ylim(*THREE_BAR_YLIM)
    ax.set_yticks([])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(1.2)
    ax.spines["bottom"].set_linewidth(1.2)
    ax.tick_params(axis="x", length=4, width=1.2)
    ax.tick_params(axis="y", length=0)
    # "hot dog" label on the single bar
    ax.text(-0.04, HOTDOG_Y, "hot dog", ha="right", va="center",
            fontsize=LABEL_SIZE, fontweight="bold",
            transform=ax.get_yaxis_transform())
    return fig, ax


def add_brace_below(ax, x_left, x_right, label, italic=False, text_x=None):
    """Draw a brace + label below the hot-dog bar."""
    brace_y = HOTDOG_Y - BAR_HEIGHT / 2 - 0.25
    mid = (x_left + x_right) / 2
    if text_x is None:
        text_x = mid
    span = x_right - x_left
    brace_xs = np.linspace(x_left, x_right, 200)
    brace_ys = brace_y - 0.10 * np.sin(np.pi * (brace_xs - x_left) / span)
    ax.plot(brace_xs, brace_ys, color="black", lw=1.8)
    ax.plot([x_left, x_left], [brace_y + 0.03, brace_y - 0.05],
            color="black", lw=1.8)
    ax.plot([x_right, x_right], [brace_y + 0.03, brace_y - 0.05],
            color="black", lw=1.8)
    ax.plot([mid, mid], [brace_y - 0.10, brace_y - 0.30],
            color="black", lw=1.8)
    style = dict(ha="center", va="top", fontsize=18)
    if italic:
        style["fontstyle"] = "italic"
    else:
        style["fontweight"] = "bold"
    ax.text(text_x, brace_y - 0.35, label, **style)


def save_loss(fig, filename):
    """Save a loss chart with the shared vertical layout."""
    fig.subplots_adjust(**LOSS_ADJUST)
    fig.savefig(os.path.join(OUTDIR, filename), dpi=200, transparent=False)
    plt.close(fig)
    print(f"Saved {filename}")


# ── Chart 4: loss (Isola-style, black+red split on -∞ → 0) ──────────

hotdog_log_g = log_g[0]   # ≈ -0.357
loss_val = -hotdog_log_g   # ≈ 0.357
LOSS_LEFT = -0.9

fig, ax = make_loss_ax()
black_width = hotdog_log_g - LOSS_LEFT
ax.barh(HOTDOG_Y, black_width, height=BAR_HEIGHT, color=BAR_COLOR, left=LOSS_LEFT)
ax.barh(HOTDOG_Y, loss_val, height=BAR_HEIGHT, color=RED_COLOR, left=hotdog_log_g)
ax.set_xlim(LOSS_LEFT, 0)
ax.set_xticks([LOSS_LEFT, 0])
ax.set_xticklabels([r"$-\infty$", "0"], fontsize=TICK_SIZE)
add_brace_below(ax, hotdog_log_g, 0,
                "How much better\nyou could have done", italic=True)
save_loss(fig, "nll_k3_loss.png")


# ── Chart 5: direct loss (no-sign-flip version, 0 → +∞) ────────────

hotdog_loss = neg_log_g[0]  # ≈ 0.357

fig, ax = make_loss_ax()
ax.barh(HOTDOG_Y, hotdog_loss, height=BAR_HEIGHT, color=RED_COLOR)
ax.set_xlim(0, RIGHT_EDGE)
ax.set_xticks([0, RIGHT_EDGE])
ax.set_xticklabels(["0", r"$\infty$"], fontsize=TICK_SIZE)
add_brace_below(ax, 0, hotdog_loss, "Loss", text_x=hotdog_loss + 0.3)
save_loss(fig, "nll_k3_loss_direct.png")


print("\nAll 5 PNGs generated successfully.")
