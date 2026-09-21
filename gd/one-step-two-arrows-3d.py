import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['font.serif'] = ['Palatino', 'Palatino Linotype', 'DejaVu Serif']
matplotlib.rcParams['mathtext.fontset'] = 'stix'
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from mpl_toolkits.mplot3d import proj3d

# One gradient-descent step, for FA26 lec03's two-arrow quiz and its answer.
#
#   bowl-quiz.png      the bowl with one black descent arrow, and nothing to explain it
#   bowl-answer.png    the same step decomposed: green shadow on the floor, yellow drop
#   contour.png        the floor alone: grey rings, grey axes, the green step arrow
#   one-step-two-arrows-3d.png   the answer with every piece labelled, for standalone use
#
# The quiz slide stacks bowl-quiz over contour and asks what the two arrows are. The
# answer slide stacks bowl-answer over contour with the names in its text. Both plots are
# authored at 4:3 so the pair fills the 900x1350 portrait slot the deck already uses,
# with every dot and arrow baked in rather than overlaid as separate blocks.
#
# The descent arrow is the move we watch on the surface, from (theta_prev, J(theta_prev))
# down to (theta_new, J(theta_new)). It decomposes into the step arrow, the horizontal move
# in parameter space that line 5 computes, drawn on the floor as the descent arrow's shadow,
# and the J-drop, the vertical fall in the objective that line 6 watches.
#
# Everything is grey except the arrows and the theta points, so black, green, yellow and
# the deck's theta red are the only colour on the slide. Surface points, which are
# (theta, J(theta)), stay black; floor and contour points, which are theta, are red. The bowl is drawn in polar coordinates so its rim is a clean circle and its
# mesh is rings and spokes. The frame is a triad of arrows from the floor's front-left corner,
# theta_1, theta_2 and J, each named at its tip; no floor square, since its far edges would
# show through the bowl.
#
# J = theta_1^2 + theta_2^2, gradient 2*theta. From theta_prev = (1.6, 1.2) with eta = 0.25
# the step is -(0.8, 0.6), landing at theta_new = (0.8, 0.6). J falls from 4 to 1, so the
# drop is exactly 3. The rings are drawn at half-integer levels so both points sit between
# rings rather than on one.
THETA_PREV = np.array([1.6, 1.2])
ETA = 0.25
def J(t1, t2):
    return t1 ** 2 + t2 ** 2
GRAD = 2 * THETA_PREV
THETA_NEW = THETA_PREV - ETA * GRAD
J_PREV, J_NEW = J(*THETA_PREV), J(*THETA_NEW)

BOWL, MESH = '#e4e4e4', '#cdcdcd'
RING, FRAME = '#b8b8b8', '#c8c8c8'
DESCENT, STEP, DROP = '#000000', '#6aa84f', '#f1c232'
THETA_DOT = '#e06666'            # the deck's red for theta, rgb(224, 102, 102)
DROP_TEXT = '#bf9000'
GUIDE = '#8a8a8a'
AXIS_TEXT = '#8a8a8a'            # axis names, grey like the frame, dark enough to read
LIM, ZMAX = 2.2, 5.3
RIM = np.sqrt(ZMAX)              # the bowl is drawn to this radius, so its rim is a circle
LEVELS = (0.5, 1.5, 2.5, 3.5, 4.5)   # offset from 1 and 4, so neither point sits on a ring
BIG, MID = 30, 22


def save(fig, out):
    fig.savefig(out, dpi=300, transparent=True, bbox_inches='tight')
    plt.close(fig)
    img = Image.open(out)
    bbox = img.getbbox()
    if bbox:
        cropped = img.crop(bbox)
        cropped.save(out.replace('.png', '_cropped.png'))
        print('  %-30s -> %s %s' % (out, out.replace('.png', '_cropped.png'), cropped.size))


def arrow3(ax, p, q, color, lw, ratio=0.18):
    p, q = np.asarray(p, float), np.asarray(q, float)
    ax.quiver(*p, *(q - p), color=color, linewidth=lw, arrow_length_ratio=ratio,
              normalize=False, zorder=8)


def bowl(out, variant):
    """variant: 'quiz' (descent only), 'answer' (decomposed, unlabelled), 'full' (labelled)."""
    fig = plt.figure(figsize=(11.0, 8.25))          # 4:3, authored for a 900x675 slot
    ax = fig.add_subplot(111, projection='3d', computed_zorder=False)

    # Frame: an axis triad from the floor's front-left corner. Three arrows, theta_1 along
    # the front edge, theta_2 along the left edge receding, J straight up past the rim, each
    # named at its tip. No floor square: its back and right edges pass behind the translucent
    # bowl and show through it as two straight streaks, lighter than the wall on any surround
    # darker than white.
    O = (-LIM, -LIM, 0.0)
    T1_TIP = (LIM * 1.18, -LIM, 0.0)
    T2_TIP = (-LIM, LIM * 1.18, 0.0)
    J_TIP = (-LIM, -LIM, ZMAX * 1.75)     # past the far rim, which rises highest on screen
    for tip in (T1_TIP, T2_TIP, J_TIP):
        arrow3(ax, O, tip, AXIS_TEXT, 2.0, ratio=0.05)

    # The bowl: polar parametrisation, one flat grey, rings and spokes.
    # A fine grid: at 26 by 73 the translucent quads leave a lighter seam down the middle of
    # the bowl where anti-aliasing thins them; at 60 by 181 the seams fall below a pixel.
    rr = np.linspace(0, RIM, 60)
    aa = np.linspace(0, 2 * np.pi, 181)
    Rg, Ag = np.meshgrid(rr, aa)
    T1, T2 = Rg * np.cos(Ag), Rg * np.sin(Ag)
    ax.plot_surface(T1, T2, Rg ** 2, color=BOWL, linewidth=0, alpha=0.40,
                    shade=False, antialiased=True, zorder=1)
    # Strides are in grid indices: 181 angles / 15 gives twelve spokes, 60 radii / 5 twelve rings.
    ax.plot_wireframe(T1, T2, Rg ** 2, rstride=15, cstride=5, color=MESH, linewidth=0.5, zorder=2)

    # Floor rings, the same levels the contour figure draws.
    ang = np.linspace(0, 2 * np.pi, 200)
    for lv in LEVELS:
        r = np.sqrt(lv)
        ax.plot(r * np.cos(ang), r * np.sin(ang), np.zeros_like(ang),
                color=RING, linewidth=0.8, zorder=2)

    A = (*THETA_PREV, J_PREV)
    B = (*THETA_NEW, J_NEW)
    A_over_B = (*THETA_NEW, J_PREV)
    A_floor = (*THETA_PREV, 0.0)
    B_floor = (*THETA_NEW, 0.0)

    if variant != 'quiz':
        for top, base in ((A, A_floor), (B, B_floor)):
            ax.plot([top[0], base[0]], [top[1], base[1]], [top[2], base[2]],
                    color=GUIDE, linestyle=':', linewidth=1.4, zorder=5)
        ax.plot([A[0], A_over_B[0]], [A[1], A_over_B[1]], [A[2], A_over_B[2]],
                color=GUIDE, linestyle='--', dashes=(5, 3), linewidth=1.4, zorder=5)
        arrow3(ax, A_floor, B_floor, STEP, 3.0)
        arrow3(ax, A_over_B, B, DROP, 3.0)
        ax.scatter(*zip(A_floor, B_floor), s=90, c=THETA_DOT, depthshade=False, zorder=9)
    arrow3(ax, A, B, DESCENT, 3.0)
    ax.scatter(*zip(A, B), s=120, c='black', depthshade=False, zorder=9)

    # Limits before the projection is read for label placement: quiver does not feed the
    # autoscaler, so without this the J tip projects against a z axis that stops at the rim.
    ax.set_xlim(-LIM * 1.05, LIM * 1.25)
    ax.set_ylim(-LIM * 1.05, LIM * 1.25)
    ax.set_zlim(0, ZMAX * 1.82)
    ax.set_box_aspect((1, 1, 0.62))
    # Camera nearly facing the theta_2 axis, so the floor matches the contour figure below it:
    # theta_1 runs left to right along the front edge and theta_2 recedes into the page.
    ax.view_init(elev=28, azim=-72)
    fig.canvas.draw()
    P = ax.get_proj()
    def label(anchor, text, color, dx, dy, size=MID, ha='left'):
        x2, y2, _ = proj3d.proj_transform(*anchor, P)
        ax.annotate(text, xy=(x2, y2), xytext=(dx, dy), textcoords='offset points',
                    color=color, fontsize=size, ha=ha, va='center', zorder=10)
    if variant == 'full':
        mid_step = (np.array(A_floor) + np.array(B_floor)) / 2
        mid_drop = (np.array(A_over_B) + np.array(B)) / 2
        label(tuple(mid_step), 'step arrow', STEP, 0, -24, ha='center')
        label(tuple(mid_drop), r'$J$-drop', DROP_TEXT, -78, 0)
        label(A_floor, r'$\theta^{(t-1)}$', 'black', 16, 2)
        label(B_floor, r'$\theta^{(t)}$', 'black', -58, -14)
    # Axis names on the frame: theta_1 along the front-left floor edge, theta_2 along the
    # front-right one, J beside the left vertical.
    label(T1_TIP, r'$\theta_1$', AXIS_TEXT, 14, -4, size=BIG)
    label(T2_TIP, r'$\theta_2$', AXIS_TEXT, -10, 14, size=BIG, ha='right')
    label(J_TIP, r'$J(\theta)$', AXIS_TEXT, 0, 16, size=BIG, ha='center')

    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.fill = False
        axis.pane.set_edgecolor('none')
        axis.line.set_linewidth(0)
    ax.grid(False)
    save(fig, out)


def contour(out):
    """The floor alone: grey rings and axes, the two points, the green step arrow."""
    fig = plt.figure(figsize=(8.0, 6.0))            # 4:3, the same 900x675 slot
    ax = fig.add_subplot(111)
    ang = np.linspace(0, 2 * np.pi, 400)
    for lv in LEVELS:
        r = np.sqrt(lv)
        ax.plot(r * np.cos(ang), r * np.sin(ang), color=RING, linewidth=1.2, zorder=1)
    # Axes through the origin with arrowheads, the way the deck's contour plot draws them,
    # in grey so they stay behind the arrow.
    for dx, dy, name, off in ((1, 0, r'$\theta_1$', (8, -6)), (0, 1, r'$\theta_2$', (-6, 8))):
        ax.annotate('', xy=(dx * LIM, dy * LIM), xytext=(-dx * LIM, -dy * LIM),
                    arrowprops=dict(arrowstyle='-|>', color=FRAME, lw=1.4,
                                    mutation_scale=18), zorder=2)
        ax.annotate(name, xy=(dx * LIM, dy * LIM), xytext=off, textcoords='offset points',
                    fontsize=BIG, color=AXIS_TEXT,
                    ha='left' if dx else 'right', va='top' if dy else 'bottom')
    ax.annotate('', xy=THETA_NEW, xytext=THETA_PREV,
                arrowprops=dict(arrowstyle='-|>', color=STEP, lw=3.0, mutation_scale=26),
                zorder=5)
    ax.scatter(*zip(THETA_PREV, THETA_NEW), s=140, c=THETA_DOT, zorder=6)
    ax.set_xlim(-LIM * 1.08, LIM * 1.08)
    ax.set_ylim(-LIM * 1.08, LIM * 1.08)
    ax.set_aspect('equal')
    ax.axis('off')
    save(fig, out)


print('figures:')
bowl('bowl-quiz.png', 'quiz')
bowl('bowl-answer.png', 'answer')
bowl('one-step-two-arrows-3d.png', 'full')
contour('contour.png')
print('  theta_prev %s  J %.1f' % (THETA_PREV, J_PREV))
print('  theta_new  %s  J %.1f   drop %.1f' % (THETA_NEW, J_NEW, J_PREV - J_NEW))
