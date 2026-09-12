import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for saving
# Dash lengths are given in points, and by default matplotlib multiplies them by
# the line width. At linewidth 2.5 the gap lines' (4, 3) became a 42px dash with a
# 31px hole, a 73px period, while Chicago's gap leaves only about 40px of line
# visible between its dot and its landing ring. The whole gap fell inside one hole,
# so the smallest of the three gaps drew no dashes at all. Unscaled, (4, 3) is a
# 17px dash and a 13px hole, which fits inside even the shortest gap.
matplotlib.rcParams['lines.scale_dashes'] = False
# The axis labels carry the deck's own words, so lean toward the deck's own look:
# Palatino where it is installed, and a serif math font for the subscripts.
matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['font.serif'] = ['Palatino', 'Palatino Linotype', 'DejaVu Serif']
matplotlib.rcParams['mathtext.fontset'] = 'stix'
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
from PIL import Image
from mpl_toolkits.mplot3d import Axes3D, proj3d
import os

# Figures for the 2-feature running example on the FA26 lec02 slides. All of them
# come out of the one dataset and one viewing angle below, so they cannot drift
# apart and the cities never move between slides:
#
#   3d_scatter_clean.png  the three cities alone
#   3d_plane_clean.png    the three cities and the hypothesis plane
#   3d_plane_bare.png     the hypothesis plane alone, no cities
#   3d_loss_clean.png     the cities, the plane, and the gap from each city to it
#   3d_J_clean.png        the objective J over the two weights, with its minimum
#
# The data matches ridge/d2-unique-solution.html, so the deck's table, its matrices,
# and that demo all run on one example. The two features are 0/1 indicators rather
# than measured quantities, which is why the slide's Temperature and Population
# headers read oddly over them.
#
# theta* = (2, 3) fits all three points exactly, so the least-squares residual is
# zero and J bottoms out at its floor. THETA below is therefore deliberately not the
# fit: the loss picture needs a wrong hypothesis to have any gaps to show.
CITIES = ['Chicago', 'New York', 'Boston']
COLORS = ['red', 'blue', 'green']
TEMP = np.array([1, 0, 1], float)       # x1
POP = np.array([0, 1, 1], float)        # x2
ENERGY = np.array([2, 3, 5], float)     # y

# The hypothesis, in those same units. Deliberately a poor fit, and deliberately
# under every city so all three gaps drop the same direction, which is what keeps
# the loss picture readable at this viewing angle.
#
# Slant and gap size trade off directly here, because theta* = (2, 3) fits all
# three cities exactly: every step toward it tilts the plane and shrinks all three
# gaps at the same rate, and the steepest plane with every city above it is theta*
# itself, where the gaps vanish. This sits 0.7 of the way, steep enough to read as
# a plane rather than an edge-on sliver, shallow enough that Chicago, whose gap is
# the smallest of the three, still has one worth drawing.
THETA = (1.4, 2.1)

# Pure per-axis scaling into the drawing cube. An offset would move the plane off
# the origin and break the "for now, ignoring the offset" framing on the slides.
CUBE = 4.5
# How far the hypothesis plane reaches back past the origin. The axes are drawn only
# in the positive direction: what shows the plane passing through the origin is the
# plane continuing past the point where they meet, not any negative axis furniture.
# The plane sits at 0.805 * PLANE_MIN at the far corner of its patch, so the cube
# floor has to clear that, and reaching further back costs empty space below.
PLANE_MIN = -2.0
s1, s2, sy = CUBE / TEMP.max(), CUBE / POP.max(), CUBE / ENERGY.max()
X1, X2, Y = TEMP * s1, POP * s2, ENERGY * sy
# The same hypothesis expressed in plotted units
T1, T2 = THETA[0] * TEMP.max() / ENERGY.max(), THETA[1] * POP.max() / ENERGY.max()

points = np.column_stack([X1, X2, Y])


def h(px, py):
    return T1 * px + T2 * py


def axes_frame(ax):
    """The shared cube: no ticks, no box, three black axes out of the origin.

    Positive arms only. The limits reach below zero so the plane's patch is never
    clipped, but nothing is drawn down there.
    """
    ax.set_xlim(PLANE_MIN - 0.3, 5.5)
    ax.set_ylim(PLANE_MIN - 0.3, 5.5)
    ax.set_zlim(PLANE_MIN - 0.3, 5.5)
    for setter in ('set_xlabel', 'set_ylabel', 'set_zlabel'):
        getattr(ax, setter)('')
    for setter in ('set_xticks', 'set_yticks', 'set_zticks'):
        getattr(ax, setter)([])
    ax.plot([0, 5], [0, 0], [0, 0], 'k-', linewidth=2, zorder=8)
    ax.plot([0, 0], [0, 5], [0, 0], 'k-', linewidth=2, zorder=8)
    ax.plot([0, 0], [0, 0], [0, 5], 'k-', linewidth=2, zorder=8)
    ax.plot([5, 4.7], [0, 0.2], [0, 0], 'k-', linewidth=2, zorder=8)
    ax.plot([5, 4.7], [0, -0.2], [0, 0], 'k-', linewidth=2, zorder=8)
    ax.plot([0, 0.2], [5, 4.7], [0, 0], 'k-', linewidth=2, zorder=8)
    ax.plot([0, -0.2], [5, 4.7], [0, 0], 'k-', linewidth=2, zorder=8)
    ax.plot([0, 0.2], [0, 0], [5, 4.7], 'k-', linewidth=2, zorder=8)
    ax.plot([0, -0.2], [0, 0], [5, 4.7], 'k-', linewidth=2, zorder=8)
    # azim 35 rather than 45. At 45 any point with x1 == x2 projects straight onto
    # the vertical axis, and one of the three does exactly that.
    ax.view_init(elev=20, azim=35)
    ax.grid(False)
    ax.set_box_aspect([1, 1, 1])
    ax._axis3don = False


def _label_angle(ax, a, b):
    """Screen angle of the segment a to b, in the convention Text.rotation uses.

    Measured off the live projection rather than written down. Typing such angles in
    is what left the deck's own rotated text blocks pointing the wrong way when the
    view moved from azim 45 to azim 35, and a label that lives inside the figure
    cannot drift away from the axis it names.
    """
    def screen(p):
        x, y, _ = proj3d.proj_transform(p[0], p[1], p[2], ax.get_proj())
        return np.array(ax.transData.transform((x, y)))

    d = screen(b) - screen(a)
    # transData is y-up, which is the convention rotation already uses
    angle = np.degrees(np.arctan2(d[1], d[0]))
    return (angle + 90) % 180 - 90  # keep the words upright, never inverted


def axis_labels(ax, fig):
    """Name each arm in the same words the slides use, angled to follow the arm."""
    fig.canvas.draw()  # the projection is only valid once the figure has been drawn
    origin = (0, 0, 0)
    # arm tip, where the words sit, the words, and whether to lie along the arm
    specs = [
        ((5, 0, 0), (5.9, 0, -0.5), 'temperature $x_1$', True),
        ((0, 5, 0), (0, 5.9, -0.5), 'population $x_2$', True),
        ((0, 0, 5), (0, 0, 5.7), 'energy used $y$', False),
    ]
    for tip, at, words, follow in specs:
        angle = _label_angle(ax, origin, tip) if follow else 0.0
        ax.text(at[0], at[1], at[2], words, fontsize=19, rotation=angle,
                rotation_mode='anchor', ha='center', va='center', zorder=12)


def save(fig, out):
    plt.tight_layout()
    fig.savefig(out, dpi=300, bbox_inches='tight', transparent=True)
    plt.close(fig)
    img = Image.open(out)
    bbox = img.getbbox()
    if bbox:
        cropped = img.crop(bbox)
        cropped.save(out.replace('.png', '_cropped.png'))
        print('  %-24s -> %s %s' % (out, out.replace('.png', '_cropped.png'), cropped.size))


def render(show_plane, show_points, show_gaps, out):
    # Every city sits above the plane, so drawing points and gaps over it is the
    # correct order. computed_zorder=False makes matplotlib honor that instead of
    # deriving its own, which otherwise chops the gap lines in half.
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d', computed_zorder=False)

    if show_plane:
        # Reach well back past the origin in both features, so the origin sits well
        # inside the patch rather than near its corner. A patch that starts at or
        # just before x=0, y=0 reads as a quadrant resting on the origin, not as a
        # plane passing through it.
        # Two points per axis, so the mesh is just the four corners. A plane is flat,
        # so that defines it exactly, and it avoids an internal wireframe.
        gx, gy = np.meshgrid(np.linspace(PLANE_MIN, 4.6, 2),
                             np.linspace(PLANE_MIN, 4.6, 2))
        ax.plot_surface(gx, gy, h(gx, gy),
                        color='#cccccc', alpha=0.55, shade=False,
                        edgecolor='#9a9a9a', linewidth=1.2, zorder=1)

    # The gaps stay black and broken, matching the dotted black line blocks the
    # slides already use, and the rings stay unfilled: only the cities carry color,
    # so the four gaps read as four instances of one quantity.
    if show_gaps:
        for px, py, pz in points:
            ax.plot([px, px], [py, py], [pz, h(px, py)],
                    color='black', linestyle='--', dashes=(4, 3),
                    linewidth=2.5, zorder=5)
            ax.scatter([px], [py], [h(px, py)],
                       s=95, facecolors='white', edgecolors='black',
                       linewidth=2, depthshade=False, zorder=6)

    if show_points:
        ax.scatter(points[:, 0], points[:, 1], points[:, 2],
                   s=300, c=COLORS, alpha=1.0,
                   depthshade=False,  # keep the far cities as vivid as the near ones
                   edgecolors='black', linewidth=2, zorder=10)

    axes_frame(ax)
    axis_labels(ax, fig)
    save(fig, out)


def render_J(out):
    """J over the two weights, drawn the way the interactive version draws it.

    ridge/d2-unique-solution.html puts this same surface on the right of its split,
    and this figure is the still of that panel. So it follows the demo rather than
    the arrow-axis style the cube figures use: a warm colourscale, a full axes box
    with ticks and grid, and the demo's own camera. The colour is what carries the
    bowl at a glance; drawn flat grey the same surface reads as a sheet.
    """
    # Raw weights, not the cube-scaled ones. J lives in theta-space, which the
    # drawing cube never touches, and the slide's algebra is in these units:
    # J(theta) = 1/3 [(t1 - 2)^2 + (t2 - 3)^2 + (t1 + t2 - 5)^2], least at (2, 3).
    def J(t1, t2):
        total = 0.0
        for a, b, c in zip(TEMP, POP, ENERGY):
            total = total + (t1 * a + t2 * b - c) ** 2
        return total / len(ENERGY)

    best = np.linalg.lstsq(np.column_stack([TEMP, POP]), ENERGY, rcond=None)[0]
    jmin = J(best[0], best[1])

    # The demo's window, theta1 over -1..5 and theta2 over -1..7, both centred on
    # theta*. No cut: the surface is drawn whole, so its rim stays a clean rectangle.
    g1 = np.linspace(-1.0, 5.0, 80)
    g2 = np.linspace(-1.0, 7.0, 80)
    G1, G2 = np.meshgrid(g1, g2)
    Z = J(G1, G2)

    warm = LinearSegmentedColormap.from_list('warm', ['#ffd966', '#f6b26b', '#cc4125'])

    fig = plt.figure(figsize=(8, 6.5))
    # computed_zorder=False: the marker sits exactly on the surface at the basin, and
    # matplotlib's own depth sort loses it inside the surface it is resting on.
    ax = fig.add_subplot(111, projection='3d', computed_zorder=False)
    ax.plot_surface(G1, G2, Z, cmap=warm, rstride=2, cstride=2,
                    linewidth=0, antialiased=True, alpha=0.9, zorder=1)
    # Lifted a hair off the floor so it does not z-fight with the surface under it.
    ax.scatter([best[0]], [best[1]], [jmin + 0.25], s=150, c='#674ea7',
               edgecolors='black', linewidth=1.5, depthshade=False, zorder=10)

    ax.set_xlim(-1, 5)
    ax.set_ylim(-1, 7)
    ax.set_zlim(0, float(Z.max()))
    ax.set_xlabel(r'$\theta_1$', fontsize=19, labelpad=14)
    ax.set_ylabel(r'$\theta_2$', fontsize=19, labelpad=14)
    # set_zlabel does not survive this figure, at any labelpad or rotation tried, so
    # the name goes in axes fractions instead, beside the tick numbers it belongs to.
    ax.set_zlabel('')
    ax.text2D(1.02, 0.55, r'$J(\theta)$', transform=ax.transAxes,
              fontsize=19, ha='left', va='center')
    ax.set_xticks([0, 2, 4])
    ax.set_yticks([0, 2, 4, 6])
    ax.set_zticks([0, 10, 20])
    ax.tick_params(labelsize=14, colors='#5f6672')
    # The demo's camera sits at eye (1.8, -1.8, 0.8), which is azim 315 and elev 17.
    # Low on purpose: from up high the walls foreshorten and the basin flattens out.
    ax.view_init(elev=18, azim=315)
    # Stretched in z, so the basin reads as a bowl rather than a shallow dish. This
    # changes only how tall the drawing box is; the surface itself is untouched.
    ax.set_box_aspect([1, 1, 1.35])
    # No grid and no panes. The colourscale already carries the height, and the
    # walls boxed the bowl in without adding anything worth reading.
    ax.grid(False)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.set_facecolor((1, 1, 1, 0))
        axis.pane.set_edgecolor('none')
    print('  J minimum at theta = (%.2f, %.2f), J = %.3f, top %.1f' % (
        best[0], best[1], jmin, Z.max()))
    save(fig, out)


print('data, in the slide\'s own units:')
print('  %-11s %6s %6s %7s %8s %8s' % ('city', 'temp', 'pop', 'energy', 'h(x)', 'gap'))
for c, a, b, y in zip(CITIES, TEMP, POP, ENERGY):
    pred = THETA[0] * a + THETA[1] * b
    print('  %-11s %6.0f %6.1f %7.0f %8.2f %+8.2f' % (c, a, b, y, pred, y - pred))
# Spans rather than max/min ratios: both features are 0/1 indicators, so a ratio
# divides by zero and reports inf, which says nothing about the column.
print('  column spans: temp %g..%g  pop %g..%g  energy %g..%g' % (
    TEMP.min(), TEMP.max(), POP.min(), POP.max(), ENERGY.min(), ENERGY.max()))
print('  hypothesis in slide units:   y = %.2f x1 + %.2f x2' % THETA)
print('  hypothesis in plotted units: y = %.3f x1 + %.3f x2  (weights %.1fx apart)' % (
    T1, T2, max(T1 / T2, T2 / T1)))
gaps = Y - h(X1, X2)
print('  plotted gaps: %s' % ', '.join('%s %+.2f' % (c, g) for c, g in zip(CITIES, gaps)))
print('  all cities above the plane: %s, gap range %.2f to %.2f (%.1fx)' % (
    bool((gaps > 0).all()), gaps.min(), gaps.max(), gaps.max() / gaps.min()))
print('figures:')
render(False, True, False, '3d_scatter_clean.png')
render(True, True, False, '3d_plane_clean.png')
render(True, False, False, '3d_plane_bare.png')
render(True, True, True, '3d_loss_clean.png')
render_J('3d_J_clean.png')

# Crop the four cube figures to one shared box. Cropping each to its own content
# frames them differently, because the scatter has no plane and the plane reaches
# further left than the points do, so the cities would shift between slides placed
# at the same width. The J surface keeps its own crop; its aspect is deliberately
# different and it shares no geometry with these.
CUBE_FIGS = ['3d_scatter_clean.png', '3d_plane_clean.png',
             '3d_plane_bare.png', '3d_loss_clean.png']
boxes = [Image.open(f).getbbox() for f in CUBE_FIGS]
union = (min(b[0] for b in boxes), min(b[1] for b in boxes),
         max(b[2] for b in boxes), max(b[3] for b in boxes))
for f in CUBE_FIGS:
    cropped = Image.open(f).crop(union)
    cropped.save(f.replace('.png', '_cropped.png'))
print('  shared crop across the cube figures: %dx%d' % cropped.size)

# The uncropped renders exist only so the pass above can measure one box across
# all four at once. Now that it has, drop them, so the directory holds just the
# five _cropped files that go on slides.
for f in CUBE_FIGS + ['3d_J_clean.png']:
    os.remove(f)
print('  removed %d intermediate renders' % (len(CUBE_FIGS) + 1))
