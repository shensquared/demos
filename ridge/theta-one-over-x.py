import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for saving
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

# Two views of the same one-feature, one-point sensitivity story for FA26 lec02.
# With n = 1 and y = 1 the closed form collapses to theta* = 1/x, so a feature
# value near zero sends theta* off to infinity and flips its sign across zero.
#
#   theta-one-over-x*.png  theta* against x: the hyperbola, with both cases marked
#   theta-xy-view*.png     the same two cases in (x, y): the two fitted lines
#
# The second view shows what the hyperbola cannot. Both data points sit at the same
# height y = 1, a hair apart in x, yet one fitted line is moderate and the other is
# nearly vertical and sloping the other way. Colour carries the pairing between the
# two figures: a point marked blue on the curve is the blue line here.
#
# Two variants, because the slide deck carries both and they make different points.
# The original pair sits at x = 0.002 and x = -0.0002, so the two answers differ by
# a factor of ten as well as a sign, which weakens the claim that a tiny change did
# it. The symmetric pair sits at x = +/-0.0002, where the two points differ by
# 0.0004 and are visually indistinguishable, yet theta* lands on +5,000 and -5,000.
# Same magnitude, opposite sign, from a change nothing could measure.
BLUE, PURPLE = '#3c78d8', '#674ea7'
Y_OBS = 1.0          # the single label, fixed by the slide
CURVE = '#cc3b2f'    # matches the red curve the slide's original raster used

VARIANTS = [
    dict(suffix='',
         cases=[(0.002, 500.0, BLUE), (-0.0002, -5000.0, PURPLE)],
         xlim=(-0.0011, 0.0032), ylim=(-6200.0, 2600.0),
         xticks=[-0.001, 0.0, 0.001, 0.002, 0.003],
         yticks=[-6000, -4000, -2000, 0, 2000]),
    dict(suffix='-symmetric',
         cases=[(0.0002, 5000.0, BLUE), (-0.0002, -5000.0, PURPLE)],
         xlim=(-0.0007, 0.0007), ylim=(-7000.0, 7000.0),
         xticks=[-0.0006, -0.0002, 0.0, 0.0002, 0.0006],
         yticks=[-5000, 0, 5000]),
]


def save(fig, out):
    plt.tight_layout()
    fig.savefig(out, dpi=300, transparent=True)
    plt.close(fig)
    img = Image.open(out)
    bbox = img.getbbox()
    if bbox:
        cropped = img.crop(bbox)
        cropped.save(out.replace('.png', '_cropped.png'))
        print('  %-30s -> %s %s' % (out, out.replace('.png', '_cropped.png'), cropped.size))


def hyperbola(v, out):
    """theta* = 1/x, zoomed so both cases are markable.

    The window is the whole point. Over a normal range like x in [-8, 10] both
    cases sit on top of the origin and their theta* values are nowhere near the
    vertical axis, so neither case could be shown at all.
    """
    xmin, xmax = v['xlim']
    fig, ax = plt.subplots(figsize=(9, 6.5))
    # Each branch separately, so the asymptote is not drawn as a join across zero
    for lo, hi in ((xmin, -1e-6), (1e-6, xmax)):
        xs = np.linspace(lo, hi, 4000)
        ax.plot(xs, 1.0 / xs, color=CURVE, linewidth=2.6, zorder=3)
    ax.axvline(0.0, color='#999999', linestyle='--', dashes=(5, 4), linewidth=1.4, zorder=2)
    ax.axhline(0.0, color='black', linewidth=1.4, zorder=2)
    # Markers only, no captions: the slide carries its own text blocks for both cases
    for x, th, col in v['cases']:
        ax.plot([x, x], [0, th], color=col, linestyle='--', dashes=(4, 3),
                linewidth=1.8, zorder=4)
        ax.scatter([x], [th], s=170, color=col, edgecolors='black',
                   linewidth=1.8, zorder=6)
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(*v['ylim'])
    ax.set_xlabel(r'$x$', fontsize=16)
    ax.set_ylabel(r'$\theta^* = 1/x$', fontsize=16)
    ax.set_xticks(v['xticks'])
    ax.set_yticks(v['yticks'])
    ax.tick_params(labelsize=12)
    ax.grid(True, color='#e8e8e8', linewidth=0.8, zorder=0)
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    save(fig, out)


def xy_view(v, out):
    """The same two cases in (x, y), as fitted lines through the origin.

    Both points sit at y = 1 and differ only slightly in x, so the eye compares the
    two slopes directly. The steeper line leaves the frame almost vertically, which
    is the sensitivity made visible rather than merely stated.
    """
    xmin, xmax = v['xlim']
    fig, ax = plt.subplots(figsize=(9, 6.5))
    ax.axhline(Y_OBS, color='#bbbbbb', linestyle='--', dashes=(5, 4),
               linewidth=1.3, zorder=1)
    ax.axhline(0.0, color='black', linewidth=1.4, zorder=2)
    ax.axvline(0.0, color='black', linewidth=1.4, zorder=2)
    xs = np.linspace(xmin, xmax, 400)
    for x, th, col in v['cases']:
        ax.plot(xs, th * xs, color=col, linewidth=2.6, zorder=3)
        ax.scatter([x], [Y_OBS], s=190, color=col, edgecolors='black',
                   linewidth=1.8, zorder=6)
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(-0.35, 2.2)
    ax.set_xlabel(r'$x$', fontsize=16)
    ax.set_ylabel(r'$y$', fontsize=16)
    ax.set_xticks(v['xticks'])
    ax.set_yticks([0, 1, 2])
    ax.tick_params(labelsize=12)
    ax.grid(True, color='#e8e8e8', linewidth=0.8, zorder=0)
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    save(fig, out)


for v in VARIANTS:
    print('figures%s:' % (' ' + v['suffix'].lstrip('-') if v['suffix'] else ''))
    hyperbola(v, 'theta-one-over-x%s.png' % v['suffix'])
    xy_view(v, 'theta-xy-view%s.png' % v['suffix'])
    for x, th, col in v['cases']:
        assert abs(th - 1.0 / x) < 1e-6, 'theta* mislabelled for x = %g' % x
        print('  x = %-9g theta* = 1/x = %-8s colour %s'
              % (x, '{:,.0f}'.format(1.0 / x), col))
