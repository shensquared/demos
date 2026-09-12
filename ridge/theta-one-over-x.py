import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for saving
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

# Two views of the same one-feature, one-point sensitivity story for FA26 lec02.
# With n = 1 and y = 1 the closed form collapses to theta* = 1/x, so a feature
# value near zero sends theta* off to infinity and flips its sign across zero.
#
#   theta-one-over-x.png  theta* against x: the hyperbola, with both cases marked
#   theta-xy-view.png     the same two cases in (x, y): the two fitted lines
#
# The second view shows what the hyperbola cannot. Both data points sit at the same
# height y = 1, a hair apart in x, yet one fitted line is moderate and the other is
# nearly vertical and sloping the other way. Colour carries the pairing between the
# two figures: a point marked blue on the curve is the blue line here.
CASES = [
    # (x, theta* = 1/x, colour)
    (0.002, 500.0, '#3c78d8'),
    (-0.0002, -5000.0, '#674ea7'),
]
Y_OBS = 1.0          # the single label, fixed by the slide
CURVE = '#cc3b2f'    # matches the red curve the slide's original raster used


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


def hyperbola(out):
    """theta* = 1/x, zoomed so both cases are markable.

    The window is the whole point. Over a normal range like x in [-8, 10] both
    0.002 and -0.0002 sit on top of the origin and 500 and -5000 are nowhere near
    the vertical axis, so neither case can be shown at all.
    """
    XMIN, XMAX = -0.0011, 0.0032
    fig, ax = plt.subplots(figsize=(9, 6.5))
    # Each branch separately, so the asymptote is not drawn as a join across zero
    for lo, hi in ((XMIN, -1e-6), (1e-6, XMAX)):
        xs = np.linspace(lo, hi, 4000)
        ax.plot(xs, 1.0 / xs, color=CURVE, linewidth=2.6, zorder=3)
    ax.axvline(0.0, color='#999999', linestyle='--', dashes=(5, 4), linewidth=1.4, zorder=2)
    ax.axhline(0.0, color='black', linewidth=1.4, zorder=2)
    # Markers only, no captions: slide 25 carries its own text blocks for both cases
    for x, th, col in CASES:
        ax.plot([x, x], [0, th], color=col, linestyle='--', dashes=(4, 3),
                linewidth=1.8, zorder=4)
        ax.scatter([x], [th], s=170, color=col, edgecolors='black',
                   linewidth=1.8, zorder=6)
    ax.set_xlim(XMIN, XMAX)
    ax.set_ylim(-6200.0, 2600.0)
    ax.set_xlabel(r'$x$', fontsize=16)
    ax.set_ylabel(r'$\theta^* = 1/x$', fontsize=16)
    ax.set_xticks([-0.001, 0.0, 0.001, 0.002, 0.003])
    ax.set_yticks([-6000, -4000, -2000, 0, 2000])
    ax.tick_params(labelsize=12)
    ax.grid(True, color='#e8e8e8', linewidth=0.8, zorder=0)
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    save(fig, out)


def xy_view(out):
    """The same two cases in (x, y), as fitted lines through the origin.

    Both points sit at y = 1 and differ only slightly in x, so the eye compares the
    two slopes directly. The -5000 line leaves the frame almost vertically, which is
    the sensitivity made visible rather than merely stated.
    """
    XMIN, XMAX = -0.0011, 0.0032
    YMIN, YMAX = -0.35, 2.2
    fig, ax = plt.subplots(figsize=(9, 6.5))
    ax.axhline(Y_OBS, color='#bbbbbb', linestyle='--', dashes=(5, 4),
               linewidth=1.3, zorder=1)
    ax.axhline(0.0, color='black', linewidth=1.4, zorder=2)
    ax.axvline(0.0, color='black', linewidth=1.4, zorder=2)
    xs = np.linspace(XMIN, XMAX, 400)
    for x, th, col in CASES:
        ax.plot(xs, th * xs, color=col, linewidth=2.6, zorder=3)
        ax.scatter([x], [Y_OBS], s=190, color=col, edgecolors='black',
                   linewidth=1.8, zorder=6)
    ax.set_xlim(XMIN, XMAX)
    ax.set_ylim(YMIN, YMAX)
    ax.set_xlabel(r'$x$', fontsize=16)
    ax.set_ylabel(r'$y$', fontsize=16)
    ax.set_xticks([-0.001, 0.0, 0.001, 0.002, 0.003])
    ax.set_yticks([0, 1, 2])
    ax.tick_params(labelsize=12)
    ax.grid(True, color='#e8e8e8', linewidth=0.8, zorder=0)
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    save(fig, out)


print('figures:')
hyperbola('theta-one-over-x.png')
xy_view('theta-xy-view.png')
for x, th, col in CASES:
    print('  x = %-9g theta* = 1/x = %-8s colour %s' % (x, '{:,.0f}'.format(1.0 / x), col))
