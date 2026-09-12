import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for saving
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

# The one-feature, one-point sensitivity picture for FA26 lec02. With n = 1 and
# y = 1 the closed form collapses to theta* = 1/x, so a feature value near zero
# sends theta* off to infinity and flips its sign across zero.
#
# The window is the whole point. Drawn over a normal range like x in [-8, 10] the
# two data points on the slide, 0.002 and -0.0002, both sit visually on top of the
# origin and cannot be marked at all. Zooming to a few thousandths puts them in
# separate branches and makes the blow-up legible.
POINTS = [(0.002, 500.0), (-0.0002, -5000.0)]   # (x, theta* = 1/x)
XMIN, XMAX = -0.0011, 0.0032
YMIN, YMAX = -6200.0, 2600.0

ACCENT = '#674ea7'   # the deck's purple
CURVE = '#cc3b2f'    # matches the red curve the current raster uses

fig, ax = plt.subplots(figsize=(9, 6.5))

# Each branch separately, so the asymptote is not drawn as a join across zero
for lo, hi in ((XMIN, -1e-6), (1e-6, XMAX)):
    xs = np.linspace(lo, hi, 4000)
    ax.plot(xs, 1.0 / xs, color=CURVE, linewidth=2.6, zorder=3)

# The asymptote itself
ax.axvline(0.0, color='#999999', linestyle='--', dashes=(5, 4), linewidth=1.4, zorder=2)

# Axes through the origin rather than around the outside, so the sign flip reads
ax.axhline(0.0, color='black', linewidth=1.4, zorder=2)

# Markers and drop lines only, no captions. Slide 25 already carries its own text
# blocks for both cases, positioned over the plot area, so baking the same words
# into the raster would duplicate them and make them impossible to reposition.
for x, th in POINTS:
    ax.plot([x, x], [0, th], color=ACCENT, linestyle='--', dashes=(4, 3),
            linewidth=1.8, zorder=4)
    ax.scatter([x], [th], s=170, color=ACCENT, edgecolors='black',
               linewidth=1.8, zorder=6)

ax.set_xlim(XMIN, XMAX)
ax.set_ylim(YMIN, YMAX)
ax.set_xlabel(r'$x$', fontsize=16)
ax.set_ylabel(r'$\theta^* = 1/x$', fontsize=16)
ax.set_xticks([-0.001, 0.0, 0.001, 0.002, 0.003])
ax.set_yticks([-6000, -4000, -2000, 0, 2000])
ax.tick_params(labelsize=12)
ax.grid(True, color='#e8e8e8', linewidth=0.8, zorder=0)
for side in ('top', 'right'):
    ax.spines[side].set_visible(False)

plt.tight_layout()
out = 'theta-one-over-x.png'
plt.savefig(out, dpi=300, transparent=True)
plt.close(fig)

img = Image.open(out)
bbox = img.getbbox()
if bbox:
    img.crop(bbox).save(out.replace('.png', '_cropped.png'))
    print('%s -> %s %s' % (out, out.replace('.png', '_cropped.png'), img.crop(bbox).size))
for x, th in POINTS:
    print('  x = %-9g theta* = 1/x = %s' % (x, '{:,.0f}'.format(1.0 / x)))
