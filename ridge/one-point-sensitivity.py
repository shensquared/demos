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
# At this scale the fitted line is a vertical wall against the y axis and the two x
# values land a few pixels apart, indistinguishable. Their predictions differ by 5.
#
# These sit on the slide about 400px wide, a downscale of roughly six, so type is
# set around 40pt to arrive near 26px. That size only works with very few labels:
# five of them collide with each other and with the data. Each figure therefore
# carries the one thing it is for. The fit names theta*, the nudge names the gap,
# and every sentence belongs in a slide text block beside the picture.
X_TRAIN, Y_TRAIN = 0.0002, 1.0
THETA = 1.0 / X_TRAIN                 # 5,000
DELTA = 0.001                         # the nudge, a thousandth
X_NEW = X_TRAIN + DELTA               # 0.0012
Y_NEW = THETA * X_NEW                 # 6.0

LINE, POINT, NUDGE, LEAD = '#3c78d8', '#3c78d8', '#cc3b2f', '#9a9a9a'
XMAX, YMAX = 1.0, 7.0
BRACKET = 0.34                        # where the gap is measured, out in clear space
BIG, TICK = 40, 28                    # sized for a 400px-wide placement


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
    ax.plot(xs, THETA * xs, color=LINE, linewidth=4.0, zorder=3)
    ax.scatter([X_TRAIN], [Y_TRAIN], s=380, color=POINT, edgecolors='black',
               linewidth=2.4, zorder=6)
    # Start a hair left of zero. Both x values sit within a pixel of the axis, and
    # against the spine itself the markers render as half circles, which reads as a
    # drawing fault rather than as the point about scale.
    ax.set_xlim(-0.03, XMAX)
    ax.set_ylim(0, YMAX)
    ax.set_xlabel(r'$x$', fontsize=BIG)
    ax.set_ylabel(r'$y$', fontsize=BIG)
    ax.set_xticks([0, 0.5, 1.0])
    ax.set_yticks([0, 1, 6])
    ax.tick_params(labelsize=TICK)
    ax.grid(True, color='#e8e8e8', linewidth=1.2, zorder=0)
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    return fig, ax


def fit_only(out):
    """The point, the line, and the one number worth naming."""
    fig, ax = frame()
    ax.annotate(r'$\theta^* = 5{,}000$', xy=(0.0009, 4.5), xytext=(0.26, 3.6),
                fontsize=BIG, color=LINE, va='center',
                arrowprops=dict(arrowstyle='-', color=LINE, linewidth=1.8), zorder=7)
    save(fig, out)


def with_nudge(out):
    """The same line, and what one thousandth of x does to the prediction.

    theta* is not renamed here. It is on the previous slide, and at this type size
    a second label collides with the gap arrow.
    """
    fig, ax = frame()
    ax.scatter([X_NEW], [Y_NEW], s=380, facecolors='white', edgecolors=NUDGE,
               linewidth=3.2, zorder=6)
    ax.annotate(r'$x + 0.001$', xy=(X_NEW, Y_NEW), xytext=(0.13, 6.3),
                fontsize=BIG, color=NUDGE, va='center',
                arrowprops=dict(arrowstyle='-', color=NUDGE, linewidth=1.6), zorder=7)
    # The gap cannot be drawn at its true x, because that is exactly where the line
    # is, so faint leaders carry the two heights out to clear ground.
    for yy in (Y_TRAIN, Y_NEW):
        ax.plot([X_NEW, BRACKET], [yy, yy], color=LEAD, linestyle='--',
                dashes=(4, 3), linewidth=1.8, zorder=2)
    ax.annotate('', xy=(BRACKET, Y_NEW), xytext=(BRACKET, Y_TRAIN),
                arrowprops=dict(arrowstyle='<->', color=NUDGE, linewidth=3.4), zorder=7)
    ax.annotate(r'$+5$', xy=(BRACKET, 3.5), xytext=(BRACKET + 0.06, 3.5),
                fontsize=BIG, color=NUDGE, va='center', ha='left', zorder=7)
    save(fig, out)


print('figures:')
fit_only('one-point-fit.png')
with_nudge('one-point-nudge.png')
assert abs(THETA - 5000.0) < 1e-9 and abs(Y_NEW - 6.0) < 1e-9
print('  x = %g, theta* = 1/x = %s' % (X_TRAIN, '{:,.0f}'.format(THETA)))
print('  nudging x by %g moves the prediction from %g to %g, a gap of %g'
      % (DELTA, Y_TRAIN, Y_NEW, Y_NEW - Y_TRAIN))
print('  label type: %dpt = %dpx in source' % (BIG, BIG * 300 / 72))
