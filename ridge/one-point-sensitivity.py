import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for saving
matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['font.serif'] = ['Palatino', 'Palatino Linotype', 'DejaVu Serif']
matplotlib.rcParams['mathtext.fontset'] = 'stix'
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

# The one-point case for FA26 lec02, drawn where we actually live, in (x, y).
#
#   one-point-fit.png     the single training point and the line the formula returns
#   one-point-nudge.png   the same, plus an x a thousandth to the right, and the
#                         vertical gap that one thousandth opens in the prediction
#
# With n = 1, d = 1 and y = 1 the closed form collapses to theta* = 1/x. At
# x = 0.0002 that is 5,000, which is a perfectly legal answer to a perfectly
# well-posed problem: X is full column rank, nothing is singular. The second figure
# is the cost of that legality. Moving x by 0.001, far below anything an instrument
# would resolve, moves the prediction from 1 to 6. The gap is theta* times the
# nudge, so the size of theta is the whole story.
X_TRAIN, Y_TRAIN = 0.0002, 1.0
THETA = 1.0 / X_TRAIN                 # 5,000
DELTA = 0.001                         # the nudge, a thousandth
X_NEW = X_TRAIN + DELTA               # 0.0012
Y_NEW = THETA * X_NEW                 # 6.0

LINE, POINT, NUDGE = '#3c78d8', '#3c78d8', '#cc3b2f'
XMAX, YMAX = 0.0014, 7.0


def save(fig, out):
    plt.tight_layout()
    fig.savefig(out, dpi=300, transparent=True)
    plt.close(fig)
    img = Image.open(out)
    bbox = img.getbbox()
    if bbox:
        cropped = img.crop(bbox)
        cropped.save(out.replace('.png', '_cropped.png'))
        print('  %-26s -> %s %s' % (out, out.replace('.png', '_cropped.png'), cropped.size))


def frame():
    fig, ax = plt.subplots(figsize=(9, 6.5))
    xs = np.linspace(0, XMAX, 200)
    ax.plot(xs, THETA * xs, color=LINE, linewidth=2.8, zorder=3)
    ax.scatter([X_TRAIN], [Y_TRAIN], s=200, color=POINT, edgecolors='black',
               linewidth=1.8, zorder=6)
    # the slope is the answer the formula returned, so name it on the line itself
    ax.annotate(r'slope $= \theta^* = 5{,}000$', xy=(0.00055, THETA * 0.00055),
                xytext=(0.00062, 1.55), fontsize=17, color=LINE,
                arrowprops=dict(arrowstyle='-', color=LINE, linewidth=1.2), zorder=7)
    ax.set_xlim(0, XMAX)
    ax.set_ylim(0, YMAX)
    ax.set_xlabel(r'$x$', fontsize=17)
    ax.set_ylabel(r'$y$', fontsize=17)
    ax.set_xticks([0, 0.0002, 0.0006, 0.0010, 0.0014])
    ax.set_yticks([0, 1, 2, 3, 4, 5, 6])
    ax.tick_params(labelsize=12)
    ax.grid(True, color='#e8e8e8', linewidth=0.8, zorder=0)
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    return fig, ax


def fit_only(out):
    """The point, and the line the closed form hands back through it."""
    fig, ax = frame()
    ax.annotate(r'$(0.0002,\; 1)$', xy=(X_TRAIN, Y_TRAIN), xytext=(0.000265, 0.62),
                fontsize=15, color='black', zorder=7)
    save(fig, out)


def with_nudge(out):
    """The same line, and what one thousandth of x does to the prediction."""
    fig, ax = frame()
    ax.annotate(r'$(0.0002,\; 1)$', xy=(X_TRAIN, Y_TRAIN), xytext=(0.000265, 0.62),
                fontsize=15, color='black', zorder=7)
    # where the nudged x lands on the same line
    ax.scatter([X_NEW], [Y_NEW], s=200, facecolors='white', edgecolors=NUDGE,
               linewidth=2.4, zorder=6)
    ax.plot([X_NEW, X_NEW], [0, Y_NEW], color=NUDGE, linestyle='--',
            dashes=(4, 3), linewidth=1.6, zorder=4)
    # the gap itself, measured from the old prediction up to the new one
    ax.annotate('', xy=(X_NEW, Y_NEW), xytext=(X_NEW, Y_TRAIN),
                arrowprops=dict(arrowstyle='<->', color=NUDGE, linewidth=2.4), zorder=7)
    ax.plot([X_TRAIN, X_NEW], [Y_TRAIN, Y_TRAIN], color='#999999',
            linestyle='--', dashes=(4, 3), linewidth=1.3, zorder=2)
    ax.annotate(r'prediction moves by $5$', xy=(X_NEW, 3.4), xytext=(0.00050, 3.9),
                fontsize=17, color=NUDGE, ha='left', zorder=7)
    ax.annotate(r'$x + 0.001$', xy=(X_NEW, 0), xytext=(0.001045, 0.22),
                fontsize=15, color=NUDGE, zorder=7)
    save(fig, out)


print('figures:')
fit_only('one-point-fit.png')
with_nudge('one-point-nudge.png')
assert abs(THETA - 5000.0) < 1e-9 and abs(Y_NEW - 6.0) < 1e-9
print('  x = %g, theta* = 1/x = %s' % (X_TRAIN, '{:,.0f}'.format(THETA)))
print('  nudging x by %g moves the prediction from %g to %g, a gap of %g'
      % (DELTA, Y_TRAIN, Y_NEW, Y_NEW - Y_TRAIN))
