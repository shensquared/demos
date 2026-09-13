import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for saving
matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['font.serif'] = ['Palatino', 'Palatino Linotype', 'DejaVu Serif']
matplotlib.rcParams['mathtext.fontset'] = 'stix'
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

# The two-feature case for FA26 lec02, drawn in feature space rather than in 3D.
#
#   two-feature-collinear.png   the four points, and how little separates them
#   two-feature-nudge.png       the same, with a 0.001 measurement wobble drawn to
#                               scale beside the deviation it would drown
#
# The one-feature slides show the consequence, a huge theta and a prediction that
# swings. This shows the cause. The four points sit within 0.001 of the line
# x2 = x1, so every scrap of information distinguishing the two features lives in a
# band a thousandth wide. To turn that band into labels spanning 5, the fit has to
# multiply it by about a thousand, which is precisely theta = (-999, 1000).
#
# A 3D picture of that plane is unreadable: over this data box it spans about
# +/-5000 in y, so it renders as a vertical wall. The demo on the same slide is the
# better place to watch the plane move. This is here to explain why it moves.
X1 = np.array([-2.0, -1.0, 1.0, 2.0])
X2 = np.array([-1.999, -1.001, 0.999, 2.001])
Y = np.array([-1.0, -2.0, 0.0, 3.0])
DEV = X2 - X1                        # +/- 0.001, the whole of the signal
THETA = np.array([-999.0, 1000.0])
DELTA = 0.001                        # a measurement wobble, the same size

POINT, LINE, NUDGE, FAINT = '#3c78d8', '#9a9a9a', '#cc3b2f', '#e8e8e8'
LIM = 2.6
ZOOM_AT = 3                          # which point the inset magnifies


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


def frame():
    """Feature space, with the four points and the line they almost lie on."""
    fig, ax = plt.subplots(figsize=(9, 6.5))
    ax.plot([-LIM, LIM], [-LIM, LIM], color=LINE, linestyle='--', dashes=(6, 4),
            linewidth=1.6, zorder=2)
    ax.annotate(r'$x_2 = x_1$', xy=(1.55, 1.55), xytext=(1.95, 1.15),
                fontsize=16, color=LINE, va='center', zorder=5)
    ax.scatter(X1, X2, s=200, color=POINT, edgecolors='black', linewidth=1.8, zorder=6)
    ax.set_xlim(-LIM, LIM)
    ax.set_ylim(-LIM, LIM)
    ax.set_xlabel(r'$x_1$', fontsize=17)
    ax.set_ylabel(r'$x_2$', fontsize=17)
    ax.set_xticks([-2, -1, 0, 1, 2])
    ax.set_yticks([-2, -1, 0, 1, 2])
    ax.tick_params(labelsize=12)
    ax.grid(True, color=FAINT, linewidth=0.8, zorder=0)
    ax.set_aspect('equal')
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    return fig, ax


def inset(ax, show_nudge):
    """Magnify one point until the deviation from the line is actually visible."""
    cx, cy = X1[ZOOM_AT], X2[ZOOM_AT]
    half = 0.0028
    ins = ax.inset_axes([0.09, 0.58, 0.36, 0.36])
    ins.plot([cx - half, cx + half], [cx - half, cx + half], color=LINE,
             linestyle='--', dashes=(6, 4), linewidth=1.4, zorder=2)
    ins.scatter([cx], [cy], s=150, color=POINT, edgecolors='black',
                linewidth=1.6, zorder=6)
    # the deviation, drawn as what it is: one thousandth
    ins.annotate('', xy=(cx, cy), xytext=(cx, cx),
                 arrowprops=dict(arrowstyle='<->', color='black', linewidth=1.8), zorder=7)
    ins.annotate(r'$0.001$', xy=(cx, cx + DEV[ZOOM_AT] / 2), xytext=(cx + 0.0004, cx + 0.0005),
                 fontsize=13, color='black', va='center', zorder=7)
    if show_nudge:
        ins.annotate('', xy=(cx + DELTA, cy), xytext=(cx, cy),
                     arrowprops=dict(arrowstyle='->', color=NUDGE, linewidth=2.2), zorder=8)
        ins.annotate('a $0.001$ wobble', xy=(cx + DELTA, cy), xytext=(cx - 0.0025, cy + 0.0013),
                     fontsize=13, color=NUDGE, va='center', zorder=8)
    ins.set_xlim(cx - half, cx + half)
    ins.set_ylim(cx - half, cx + half * 1.4)
    ins.set_xticks([]); ins.set_yticks([])
    for s in ins.spines.values():
        s.set_color('#bbbbbb')
    ins.set_facecolor('white')
    ins.set_title('zoomed in on one point', fontsize=13, color='#555555', pad=6)
    # Connect the inset to the speck it came from. Without this the zoom floats and
    # nothing says which of the four points it magnifies, or how small the region is.
    ax.indicate_inset_zoom(ins, edgecolor='#999999', linewidth=1.2, alpha=0.9)
    return ins


def collinear(out):
    fig, ax = frame()
    inset(ax, show_nudge=False)
    ax.annotate('every point sits within $0.001$ of the line',
                xy=(0, -1.75), xytext=(0, -1.75), fontsize=16, color='black',
                ha='center', va='center', zorder=7)
    ax.annotate(r'so the fit needs $\theta \approx (-999,\; 1000)$ to use it',
                xy=(0, -2.15), xytext=(0, -2.15), fontsize=16, color=POINT,
                ha='center', va='center', zorder=7)
    save(fig, out)


def with_nudge(out):
    fig, ax = frame()
    inset(ax, show_nudge=True)
    ax.annotate('a measurement wobble is the same size as the signal',
                xy=(0, -1.75), xytext=(0, -1.75), fontsize=16, color='black',
                ha='center', va='center', zorder=7)
    ax.annotate(r'nudge $x_1$ by $0.001$ and the prediction moves by $1.0$',
                xy=(0, -2.15), xytext=(0, -2.15), fontsize=16, color=NUDGE,
                ha='center', va='center', zorder=7)
    save(fig, out)


print('figures:')
collinear('two-feature-collinear.png')
with_nudge('two-feature-nudge.png')
pred = THETA @ np.vstack([X1, X2])
assert np.allclose(pred, Y), 'theta does not reproduce the labels'
print('  deviations x2 - x1 : %s' % np.round(DEV, 4).tolist())
print('  ||theta||          : %.2f' % np.hypot(*THETA))
for name, vec in (('x1 only  +0.001', np.array([DELTA, 0.0])),
                  ('x2 only  +0.001', np.array([0.0, DELTA])),
                  ('both     +0.001', np.array([DELTA, DELTA]))):
    print('  %-16s -> prediction moves %+.4f' % (name, THETA @ vec))
