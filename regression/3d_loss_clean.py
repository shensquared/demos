import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for saving
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from mpl_toolkits.mplot3d import Axes3D

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
THETA = (1.0, 1.5)

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


def render(show_plane, show_points, show_gaps, out, floor_drops=False):
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

    # Floor drop lines, on the scatter only. Both features are binary, so the three
    # points sit at corners of the footprint with nothing in between and the picture
    # carries no depth cue at all; anchoring each one to the x1-x2 plane supplies it.
    # The loss figure already has its gap lines, so these would only clutter it.
    if floor_drops:
        for px, py, pz in points:
            ax.plot([px, px], [py, py], [0, pz], color='#9a9a9a',
                    linestyle='--', dashes=(4, 3), linewidth=1.4, zorder=4)
            ax.scatter([px], [py], [0], s=55, facecolors='white',
                       edgecolors='#9a9a9a', linewidth=1.4,
                       depthshade=False, zorder=5)

    if show_points:
        ax.scatter(points[:, 0], points[:, 1], points[:, 2],
                   s=300, c=COLORS, alpha=1.0,
                   depthshade=False,  # keep the far cities as vivid as the near ones
                   edgecolors='black', linewidth=2, zorder=10)

    axes_frame(ax)
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
    ax.set_xlim(g1.min(), g1.max())
    ax.set_ylim(g2.min(), g2.max())
    ax.set_zlim(0, cut)
    for setter in ('set_xlabel', 'set_ylabel', 'set_zlabel'):
        getattr(ax, setter)('')
    for setter in ('set_xticks', 'set_yticks', 'set_zticks'):
        getattr(ax, setter)([])
    # Look down into the basin and compress the vertical, so the quadratic reads as
    # a bowl. J grows fast toward the window corners, so at equal aspect and a low
    # elevation the corners tower and the whole surface renders as a spike.
    ax.view_init(elev=38, azim=45)
    ax.grid(False)
    ax.set_box_aspect([1, 1, 0.55])
    ax._axis3don = False
    print('  J minimum at theta = (%.3f, %.3f), J = %.4f' % (best[0], best[1], jmin))
    save(fig, out)


print('data, in the slide\'s own units:')
print('  %-11s %6s %6s %7s %8s %8s' % ('city', 'temp', 'pop', 'energy', 'h(x)', 'gap'))
for c, a, b, y in zip(CITIES, TEMP, POP, ENERGY):
    pred = THETA[0] * a + THETA[1] * b
    print('  %-11s %6.0f %6.1f %7.0f %8.2f %+8.2f' % (c, a, b, y, pred, y - pred))
print('  column ranges: temp %.2fx  pop %.2fx  energy %.2fx' % (
    TEMP.max() / TEMP.min(), POP.max() / POP.min(), ENERGY.max() / ENERGY.min()))
print('  hypothesis in slide units:   y = %.2f x1 + %.2f x2' % THETA)
print('  hypothesis in plotted units: y = %.3f x1 + %.3f x2  (weights %.1fx apart)' % (
    T1, T2, max(T1 / T2, T2 / T1)))
gaps = Y - h(X1, X2)
print('  plotted gaps: %s' % ', '.join('%s %+.2f' % (c, g) for c, g in zip(CITIES, gaps)))
print('  all cities above the plane: %s, gap range %.2f to %.2f (%.1fx)' % (
    bool((gaps > 0).all()), gaps.min(), gaps.max(), gaps.max() / gaps.min()))
print('figures:')
render(False, True, False, '3d_scatter_clean.png', floor_drops=True)
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
