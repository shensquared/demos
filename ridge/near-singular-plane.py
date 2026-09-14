import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for saving
matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['font.serif'] = ['Palatino', 'Palatino Linotype', 'DejaVu Serif']
matplotlib.rcParams['mathtext.fontset'] = 'stix'
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

# The near-singular two-feature case for FA26 lec02, drawn as the plane it returns.
#
#   near-singular-plane.png   the four points and the plane OLS fits through them
#
# This is the still of ridge/ill-conditioned-two-planes.html, which the deck embeds on
# the same slide. two-feature-collinear.py draws the same data in feature space and
# explains the cause; this shows the consequence.
#
# The data sits within 0.00006 of x2 = 2*x1, so XᵀX is nearly singular, condition number
# about 1.6e12, and the exact optimum is theta = (-60000, 30000), norm 67082, fitting all
# four points. That condition number is past what float64 holds: solving the normal
# equations in double precision returns about (-59998.93, 29999.46), so THETA below is the
# exact rational answer, written down rather than computed.
#
# The z range is the whole design problem. Over the x box the plane spans about
# +/-300000, while the labels y run -1.8 to 1.8. Drawn to the plane's own scale the four
# points collapse into a single dot; drawn to the data's scale the plane is a wall with
# no visible structure. ZMAX below is a compromise: wide enough that the points stay
# distinct and their heights readable, tight enough that the plane still crosses the
# frame within a sliver of x and so reads as near-vertical, which is the point.
#
# x1 and x2 need separate limits. x2 reaches 4, so a single square box would cut the data
# off; LIM_X2 is twice LIM_X1, matching the x2 = 2*x1 ridge the data lies along.
#
# Its partner figure is not a second plane. The ridge fit at lambda = 0.1 is
# theta = (0.190, 0.381), norm 0.43, spanning only +/-2 over the same box, so the two
# planes differ by a factor of 150000 in height and no single axis shows both. The demo
# toggles between them instead.
X1 = np.array([-2.0, -1.0, 1.0, 2.0])
X2 = np.array([-4.00006, -2.00004, 2.00004, 4.00006])
Y = np.array([-1.8, -1.2, 1.2, 1.8])
THETA = np.array([-60000.0, 30000.0])
NORM = float(np.hypot(*THETA))        # 67082.0, shown as 67,082

PLANE, POINT, EDGE = '#c9daf8', '#3c78d8', '#1a4a8a'
STEM = '#9a9a9a'
LIM_X1, LIM_X2, ZMAX = 2.5, 5.0, 4.0
BIG, TICK = 40, 26                    # sized for a 400px-wide placement


def save(fig, out):
    fig.savefig(out, dpi=300, transparent=True, bbox_inches='tight')
    plt.close(fig)
    img = Image.open(out)
    bbox = img.getbbox()
    if bbox:
        cropped = img.crop(bbox)
        cropped.save(out.replace('.png', '_cropped.png'))
        print('  %-26s -> %s %s' % (out, out.replace('.png', '_cropped.png'), cropped.size))


def plane(out):
    # Wider than tall, with room at the right: the z label sits outside the axes in a
    # 3D projection and was being clipped off the canvas before the crop ever ran.
    fig = plt.figure(figsize=(11.5, 7.5))
    ax = fig.add_subplot(111, projection='3d', computed_zorder=False)

    # Parametrised by (x1, y), not masked off a rectangular grid. Writing x2 = 2*x1 + d
    # gives y = 30000*d, so inside this z window the plane is a ribbon about 0.00027 wide
    # in x2, far under one cell of any practical mask grid, which would alias it into
    # disconnected slabs. Solving x2 from y instead lays grid lines exactly along the
    # ribbon, and the surface comes out exact and unbroken at any thinness.
    u = np.linspace(-LIM_X1, LIM_X1, 160)    # along x1
    w = np.linspace(-ZMAX, ZMAX, 160)        # along y
    U, W = np.meshgrid(u, w)
    V = (W - THETA[0] * U) / THETA[1]        # the x2 putting the plane at height W
    ax.plot_surface(U, V, W, color=PLANE, alpha=0.85, linewidth=0,
                    antialiased=False, shade=False, zorder=2)

    # A stem from each point to the floor. The plane is translucent and passes exactly
    # through every point, so with nothing anchoring them they read as floating in
    # front of it. There is no residual to draw here, since the fit is exact; the stem
    # fixes each point in (x1, x2) so the eye can see point and plane coincide.
    for a, b, c in zip(X1, X2, Y):
        ax.plot([a, a], [b, b], [-ZMAX, c], color=STEM, linestyle='--',
                dashes=(4, 3), linewidth=1.8, zorder=4)
    ax.scatter(X1, X2, Y, s=260, c=POINT, edgecolors=EDGE, linewidth=2.0,
               depthshade=False, zorder=6)

    ax.set_xlim(-LIM_X1, LIM_X1)
    ax.set_ylim(-LIM_X2, LIM_X2)
    ax.set_zlim(-ZMAX, ZMAX)
    ax.set_xlabel(r'$x_1$', fontsize=BIG, labelpad=24)
    ax.set_ylabel(r'$x_2$', fontsize=BIG, labelpad=42)
    ax.set_zlabel(r'$y$', fontsize=BIG, labelpad=18)
    ax.set_xticks([-2, 0, 2])
    ax.set_yticks([-4, 4])
    ax.set_zticks([-ZMAX, 0, ZMAX])
    ax.tick_params(labelsize=TICK, pad=6)
    ax.tick_params(axis='y', pad=16)
    # Swept ten combinations of angle and clamp. Looking along the hinge line x2 = x1,
    # near azim 45, turns the sheet edge-on: it vanishes to a hairline and the four
    # points stack into a single column. azim 80 shows the sheet at a steep oblique
    # with the points strung along it, and is the only view where the y label renders.
    ax.view_init(elev=16, azim=80)
    # The one number worth naming. Everything else belongs in a slide text block, the
    # same rule the one-point figures follow.
    ax.text2D(0.52, 0.86, r'$\|\theta^*\| = 67{,}082$', transform=ax.transAxes,
              fontsize=BIG, color=EDGE, ha='left', va='center')
    save(fig, out)


print('figures:')
plane('near-singular-plane.png')
print('  theta* = (%.0f, %.0f), norm %.1f' % (THETA[0], THETA[1], NORM))
print('  fit at the four points: %s' % np.round(THETA[0] * X1 + THETA[1] * X2, 6))
print('  labels:                 %s' % Y)
# atol is loose because the residual is float64 cancellation, not model error: the terms
# are near 120000 and cancel to 1.8, so the exact fit still shows about 3e-5 of noise.
assert np.allclose(THETA[0] * X1 + THETA[1] * X2, Y, atol=1e-4), 'the plane must pass through every point'
print('  plane spans y in [%.0f, %.0f] over the box; frame clamps to +/-%g'
      % ((THETA[0] * -LIM_X1 + THETA[1] * LIM_X2), (THETA[0] * LIM_X1 + THETA[1] * -LIM_X2), ZMAX))
