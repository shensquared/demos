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


def j_axes_frame(ax, g1, g2, cut, m1, m2):
    """The same three black arrowed arms the cube figures carry, for theta-space.

    The cube figures hang their axes off the origin. That is not available here: the
    window sits tight around theta*, and the origin of theta-space is nowhere near
    it. So the arms run out of the low corner of the window instead, through the
    margin the limits already hold open, which keeps them clear of the bowl.
    """
    x0, y0 = g1.min() - 0.6 * m1, g2.min() - 0.6 * m2
    x1, y1 = g1.max(), g2.max()
    style = dict(color='black', linewidth=2, zorder=8)
    ax.plot([x0, x1], [y0, y0], [0, 0], **style)
    ax.plot([x0, x0], [y0, y1], [0, 0], **style)
    ax.plot([x0, x0], [y0, y0], [0, cut], **style)
    # Arrowheads sized per axis. The two weights span about a third of a unit while
    # J spans its cut height, so one shared offset would be a speck on one arm and
    # a spike on another.
    hx, hy, hz = 0.07 * (x1 - x0), 0.07 * (y1 - y0), 0.07 * cut
    ax.plot([x1, x1 - hx], [y0, y0 + 0.5 * hy], [0, 0], **style)
    ax.plot([x1, x1 - hx], [y0, y0 - 0.5 * hy], [0, 0], **style)
    ax.plot([x0 + 0.5 * hx, x0], [y1 - hy, y1], [0, 0], **style)
    ax.plot([x0 - 0.5 * hx, x0], [y1 - hy, y1], [0, 0], **style)
    ax.plot([x0 + 0.5 * hx, x0], [y0, y0], [cut - hz, cut], **style)
    ax.plot([x0 - 0.5 * hx, x0], [y0, y0], [cut - hz, cut], **style)
    return x0, y0, x1, y1, cut


def j_axis_labels(ax, fig, frame):
    """Name each arm at its tip, the way the cube figures name theirs."""
    fig.canvas.draw()  # the projection is only valid once the figure has been drawn
    x0, y0, x1, y1, z1 = frame
    dx, dy, dz = 0.08 * (x1 - x0), 0.08 * (y1 - y0), 0.09 * z1
    # the arm to lie along, where the words sit, the words
    specs = [
        (((x0, y0, 0), (x1, y0, 0)), (x1 + dx, y0 - 0.3 * dy, 0), r'$\theta_1$'),
        (((x0, y0, 0), (x0, y1, 0)), (x0 - 0.3 * dx, y1 + dy, 0), r'$\theta_2$'),
        (None, (x0, y0, z1 + dz), r'$J(\theta)$'),
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
    # Size the window separately per weight, so J rises by the same amount along
    # each axis. A square window renders the surface as a needle: J grows with the
    # square of the step, and the two weights have very different curvature, so one
    # direction towers while the other stays flat.
    curv = A.T @ A / len(Y)
    rise = max(8.0 * jmin, 0.35)
    span1 = np.sqrt(rise / curv[0, 0])
    span2 = np.sqrt(rise / curv[1, 1])
    g1 = np.linspace(best[0] - span1, best[0] + span1, 60)
    g2 = np.linspace(best[1] - span2, best[1] + span2, 60)
    G1, G2 = np.meshgrid(g1, g2)
    Z = J(G1, G2)
    # Cut the walls off above a threshold and drop everything higher. Drawn all the
    # way up its sides a paraboloid hides its own basin: the near wall occludes the
    # floor and the far wall rises behind it, so the surface reads as a peak rather
    # than a minimum. Cutting it leaves an open rim you can see down into.
    cut = jmin + 6.0 * max(jmin, 0.05)
    Z = np.where(Z > cut, np.nan, Z)
    print('  J window: span %.3f by %.3f, floor %.4f, cut at %.4f' % (
        span1, span2, jmin, cut))

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
    ax.set_xlim(g1.min() - m1, g1.max() + m1)
    ax.set_ylim(g2.min() - m2, g2.max() + m2)
    ax.set_zlim(0, cut)
    for setter in ('set_xlabel', 'set_ylabel', 'set_zlabel'):
        getattr(ax, setter)('')
    for setter in ('set_xticks', 'set_yticks', 'set_zticks'):
        getattr(ax, setter)([])
    # Look down into the basin and compress the vertical, so the quadratic reads as
    # a bowl. J grows fast toward the window corners, so at equal aspect and a low
    # elevation the corners tower and the whole surface renders as a spike.
    # azim 215, not 225, for the same reason the cube sits at 35 rather than 45.
    # The window is centred on theta*, so the minimum lies exactly on the diagonal
    # out of the low corner; square to that diagonal it projects onto the vertical
    # arm and the purple dot lands on the axis. Ten degrees off separates them.
    # elev 48 rather than 38. The arm rises from a corner that projects well below
    # the basin, so at 38 the J label and the arrow tip both land on the bowl's
    # lower edge. Lifting the eye closes that gap, and 58 would start flattening
    # the bowl into a disc.
    ax.view_init(elev=48, azim=215)
    ax.grid(False)
    ax.set_box_aspect([1, 1, 0.55])
    ax._axis3don = False
    j_axis_labels(ax, fig, j_axes_frame(ax, g1, g2, cut, m1, m2))
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
