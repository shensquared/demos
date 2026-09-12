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


def j_axes_frame(ax, g1, g2, ztop, m1, m2):
    """Three black arrowed arms for theta-space, along the two edges facing the viewer.

    The cube figures hang their axes off the origin. That is not available here: the
    window is centred on theta* and the origin of theta-space sits outside it. Running
    all three out of one corner of the window sends one of them straight across the
    bowl, so the two weights follow the near edges instead, meeting at the corner that
    projects to the bottom of the frame, and J rises at the far end of one of them.
    """
    # Pushed well past the surface, so the vertical arm clears the screen column the
    # minimum sits in. At azim 315 that column rises from the middle of the near edge.
    xN, yN = g1.max() + 1.15 * m1, g2.min() - 1.15 * m2
    xF, yF = g1.min(), g2.max()
    style = dict(color='black', linewidth=2, zorder=8)
    ax.plot([xF, xN], [yN, yN], [0, 0], **style)
    ax.plot([xN, xN], [yN, yF], [0, 0], **style)
    # J rises from the far end of the near edge, not from the corner the other two
    # meet at. That corner projects directly below the basin, so an arm there runs up
    # through the bowl and its label lands on the minimum.
    ax.plot([xF, xF], [yN, yN], [0, ztop], **style)
    # Arrowheads sized per axis: the weights span a handful of units while J spans
    # hundreds, so one shared offset would be a speck on one arm and a spike on another.
    hx, hy, hz = 0.06 * (xN - xF), 0.06 * (yF - yN), 0.06 * ztop
    ax.plot([xN, xN - hx], [yN, yN + 0.5 * hy], [0, 0], **style)
    ax.plot([xN, xN - hx], [yN, yN - 0.5 * hy], [0, 0], **style)
    ax.plot([xN + 0.5 * hx, xN], [yF - hy, yF], [0, 0], **style)
    ax.plot([xN - 0.5 * hx, xN], [yF - hy, yF], [0, 0], **style)
    ax.plot([xF + 0.5 * hx, xF], [yN, yN], [ztop - hz, ztop], **style)
    ax.plot([xF - 0.5 * hx, xF], [yN, yN], [ztop - hz, ztop], **style)
    return xF, yF, xN, yN, ztop


def j_axis_labels(ax, fig, frame):
    """Name each arm at its tip, the way the cube figures name theirs."""
    fig.canvas.draw()  # the projection is only valid once the figure has been drawn
    xF, yF, xN, yN, ztop = frame
    dx, dy, dz = 0.07 * (xN - xF), 0.07 * (yF - yN), 0.09 * ztop
    # the arm to lie along, where the words sit, the words
    specs = [
        (((xF, yN, 0), (xN, yN, 0)), (xN + dx, yN - 0.3 * dy, 0), r'$\theta_1$'),
        (((xN, yN, 0), (xN, yF, 0)), (xN + 0.3 * dx, yF + dy, 0), r'$\theta_2$'),
        (None, (xF, yN, ztop + dz), r'$J(\theta)$'),
    ]
    for edge, at, words in specs:
        angle = _label_angle(ax, edge[0], edge[1]) if edge else 0.0
        ax.text(at[0], at[1], at[2], words, fontsize=21, rotation=angle,
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
    """J over the two weights, in plotted units, with its minimum marked."""
    n = len(ENERGY)

    def J(t1, t2):
        total = 0.0
        for a, b, c in zip(X1, X2, Y):
            total = total + (t1 * a + t2 * b - c) ** 2
        return total / n

    # least squares minimum, and a window around it
    A = np.column_stack([X1, X2])
    best = np.linalg.lstsq(A, Y, rcond=None)[0]
    jmin = J(best[0], best[1])
    # theta* plus or minus 3 and 4, the proportions ridge/d2-unique-solution.html
    # uses for the same picture. Sized off the curvature instead, the window hugged
    # the minimum, where a paraboloid is nearly flat, and the surface read as a dish.
    span1, span2 = 3.0, 4.0
    g1 = np.linspace(best[0] - span1, best[0] + span1, 60)
    g2 = np.linspace(best[1] - span2, best[1] + span2, 60)
    G1, G2 = np.meshgrid(g1, g2)
    Z = J(G1, G2)
    # Drawn whole, over its rectangular window. Masking everything above a height
    # instead leaves the rim a ragged sawtooth, because the cut falls between grid
    # cells and every cell is either kept or dropped. The box aspect below
    # compresses the vertical, which is what keeps the basin readable without one.
    ztop = float(Z.max())
    print('  J window: span %.3f by %.3f, floor %.4f, top %.4f' % (
        span1, span2, jmin, ztop))

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d', computed_zorder=False)
    ax.plot_surface(G1, G2, Z, color='#cccccc', alpha=0.5, shade=False,
                    edgecolor='#9a9a9a', linewidth=0.4, zorder=1)
    jmin = J(best[0], best[1])
    ax.scatter([best[0]], [best[1]], [jmin], s=260, c='#674ea7',
               edgecolors='black', linewidth=2, depthshade=False, zorder=10)
    # Hold the limits wider than the surface, so the labels have somewhere to sit.
    # Fitted tight, the bowl fills the frame edge to edge and every label lands on
    # top of it instead of beside it.
    m1 = 0.12 * (g1.max() - g1.min())
    m2 = 0.12 * (g2.max() - g2.min())
    # Asymmetric on purpose: the arms and their labels live past the two near edges.
    ax.set_xlim(g1.min() - m1, g1.max() + 1.9 * m1)
    ax.set_ylim(g2.min() - 1.9 * m2, g2.max() + m2)
    ax.set_zlim(0, ztop)
    for setter in ('set_xlabel', 'set_ylabel', 'set_zlabel'):
        getattr(ax, setter)('')
    for setter in ('set_xticks', 'set_yticks', 'set_zticks'):
        getattr(ax, setter)([])
    # azim 315 looks along the shallow diagonal of the quadratic, the same direction
    # the interactive version looks from. Square to the steep diagonal, at 45 or 225,
    # one corner towers and the surface reads as a ramp rather than a basin.
    # elev 26 keeps the walls inside the frame; lower and they run off the top.
    ax.view_init(elev=26, azim=315)
    ax.grid(False)
    ax.set_box_aspect([1, 1, 0.75])
    ax._axis3don = False
    j_axis_labels(ax, fig, j_axes_frame(ax, g1, g2, ztop, m1, m2))
    print('  J minimum at theta = (%.3f, %.3f), J = %.4f' % (best[0], best[1], jmin))
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
