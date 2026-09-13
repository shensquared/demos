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
# x = 0.0002 that is 5,000, a perfectly legal answer to a perfectly well-posed
# problem: X is full column rank and nothing is singular.
#
# x runs from 0 to 1, an ordinary feature range, and that choice is the argument.
# At this scale the fitted line is a vertical wall against the y axis, and the two
# x values, 0.0002 and 0.0012, land within a few pixels of each other. A viewer
# cannot tell them apart. The predictions they produce differ by 5. Everything
# interesting is therefore squeezed against the left edge, so the labels ride out
# into the empty space on leaders, and the gap is measured by a bracket set to the
# right rather than drawn at its true x, where the line already is.
X_TRAIN, Y_TRAIN = 0.0002, 1.0
THETA = 1.0 / X_TRAIN                 # 5,000
DELTA = 0.001                         # the nudge, a thousandth
X_NEW = X_TRAIN + DELTA               # 0.0012
Y_NEW = THETA * X_NEW                 # 6.0

LINE, POINT, NUDGE, LEAD = '#3c78d8', '#3c78d8', '#cc3b2f', '#9a9a9a'
XMAX, YMAX = 1.0, 7.0
BRACKET = 0.30                        # where the gap is measured, out in clear space


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
    """The shared frame, so the two figures sit still on consecutive slides."""
    fig, ax = plt.subplots(figsize=(9, 6.5))
    xs = np.linspace(0, XMAX, 400)
    ax.plot(xs, THETA * xs, color=LINE, linewidth=2.8, zorder=3)
    ax.scatter([X_TRAIN], [Y_TRAIN], s=200, color=POINT, edgecolors='black',
               linewidth=1.8, zorder=6)
    ax.annotate(r'slope $= \theta^* = 5{,}000$', xy=(0.0008, 4.0), xytext=(0.26, 4.15),
                fontsize=17, color=LINE, va='center',
                arrowprops=dict(arrowstyle='-', color=LINE, linewidth=1.2), zorder=7)
    ax.annotate(r'the one data point, $(0.0002,\; 1)$', xy=(X_TRAIN, Y_TRAIN),
                xytext=(0.14, 0.55), fontsize=15, color='black', va='center',
                arrowprops=dict(arrowstyle='-', color='black', linewidth=1.0), zorder=7)
    # Start a hair left of zero. Both x values sit within a pixel of the axis, and
    # against the spine itself the markers render as half circles, which reads as a
    # drawing fault rather than as the point about scale.
    ax.set_xlim(-0.03, XMAX)
    ax.set_ylim(0, YMAX)
    ax.set_xlabel(r'$x$', fontsize=17)
    ax.set_ylabel(r'$y$', fontsize=17)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_yticks([0, 1, 2, 3, 4, 5, 6])
    ax.tick_params(labelsize=12)
    ax.grid(True, color='#e8e8e8', linewidth=0.8, zorder=0)
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    return fig, ax


def fit_only(out):
    """The point, and the line the closed form hands back through it."""
    fig, ax = frame()
    save(fig, out)


def with_nudge(out):
    """The same line, and what one thousandth of x does to the prediction."""
    fig, ax = frame()
    ax.scatter([X_NEW], [Y_NEW], s=200, facecolors='white', edgecolors=NUDGE,
               linewidth=2.4, zorder=6)
    # The two x values are a few pixels apart at this scale, so say so rather than
    # pretend the picture can separate them.
    ax.annotate(r'$x+0.001$ lands here too', xy=(X_NEW, Y_NEW), xytext=(0.16, 6.55),
                fontsize=15, color=NUDGE, va='center',
                arrowprops=dict(arrowstyle='-', color=NUDGE, linewidth=1.0), zorder=7)
    # Faint leaders carry the two heights out to where the gap can be drawn.
    for yy in (Y_TRAIN, Y_NEW):
        ax.plot([X_NEW, BRACKET], [yy, yy], color=LEAD, linestyle='--',
                dashes=(4, 3), linewidth=1.2, zorder=2)
    ax.annotate('', xy=(BRACKET, Y_NEW), xytext=(BRACKET, Y_TRAIN),
                arrowprops=dict(arrowstyle='<->', color=NUDGE, linewidth=2.4), zorder=7)
    ax.annotate(r'prediction moves by $5$', xy=(BRACKET, 3.5), xytext=(BRACKET + 0.04, 3.5),
                fontsize=17, color=NUDGE, va='center', ha='left', zorder=7)
    save(fig, out)


print('figures:')
fit_only('one-point-fit.png')
with_nudge('one-point-nudge.png')
assert abs(THETA - 5000.0) < 1e-9 and abs(Y_NEW - 6.0) < 1e-9
print('  x = %g, theta* = 1/x = %s' % (X_TRAIN, '{:,.0f}'.format(THETA)))
print('  nudging x by %g moves the prediction from %g to %g, a gap of %g'
      % (DELTA, Y_TRAIN, Y_NEW, Y_NEW - Y_TRAIN))
print('  on a 0 to 1 axis the two x values are %.3f%% and %.3f%% across'
      % (100 * X_TRAIN / XMAX, 100 * X_NEW / XMAX))
