import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for saving
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from mpl_toolkits.mplot3d import Axes3D

# Two figures for the 2-feature running example, sharing one geometry so they cannot
# drift apart. Both match 3d_scatter_clean.py in view angle, limits, colors and point
# positions, so every slide shows the same four cities:
#
#   3d_plane_clean.png  the four cities and the hypothesis plane, nothing else
#   3d_loss_clean.png   the same, plus the vertical gap from each city to the plane
#
# The hypothesis is h(x) = theta1*x1 + theta2*x2, a hyperplane through the origin
# with no offset, matching the "for now, ignoring the offset" framing on the linear
# hypothesis class slide. theta is deliberately a poor fit so all four gaps stay
# visible, and it sits under every city so all four gaps drop the same direction.
# The two weights differ so the plane tilts visibly. Weighting both equally makes it
# rise toward the far corner, which this viewing angle foreshortens into something
# that reads as horizontal. Blue, at (3.5, 1.5, 1.5), is the binding city: it has the
# highest x of the four, so the plane rises fastest toward it and its gap closes
# first, which caps the slant. theta1 is pushed right up to what blue's gap tolerates
# while still reading as a thin dash. Steeper than this and the plane climbs through
# blue, flipping that one gap upward while the other three still point down, and that
# consistency is what makes the picture legible at this viewing angle.
THETA = (0.32, 0.02)

# Points and colors from 3d_scatter_clean.py: Chicago, New York, Boston, San Diego
points = np.array([
    [2, 1.5, 1.5],    # Point 1 - center bottom area
    [3.5, 1.5, 1.5],    # Point 2 - right bottom area
    [1.5, 3.5, 1.5],    # Point 3 - back bottom area
    [3.5, 2, 3.5]     # Point 4 - top back right area
])
colors = ['red', 'blue', 'green', 'orange']


def h(px, py):
    return THETA[0] * px + THETA[1] * py


def render(show_gaps, out):
    # Every city sits above the plane, so drawing the points and gap lines over it is
    # the geometrically correct order. computed_zorder=False makes matplotlib honor
    # that instead of deriving its own, which otherwise chops the gap lines in half.
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d', computed_zorder=False)

    # Hypothesis plane through the origin. Gray at low opacity, matching the plane on
    # the linear hypothesis class slide.
    # Stops just past the farthest city at 3.5 instead of running out toward the axis
    # arrows, so the patch reads as a plane through the origin rather than as a floor.
    gx, gy = np.meshgrid(np.linspace(0, 3.9, 2), np.linspace(0, 3.9, 2))
    ax.plot_surface(gx, gy, h(gx, gy),
                    color='#cccccc', alpha=0.39, shade=False,
                    edgecolor='#9a9a9a', linewidth=1.2, zorder=1)

    # Vertical gap from each city down to the plane, then a hollow ring where it
    # lands. A plane seen in projection gives no cue for where a vertical meets it, so
    # without the ring each gap reads as stopping in mid-air. The gaps stay black and
    # broken, matching the dotted black line blocks the slide already uses for the 2D
    # version, and the rings stay unfilled: only the cities carry color, so the four
    # gaps read as four instances of one quantity rather than four different things.
    if show_gaps:
        for px, py, pz in points:
            ax.plot([px, px], [py, py], [pz, h(px, py)],
                    color='black', linestyle='--', dashes=(4, 3),
                    linewidth=2.5, zorder=5)
            ax.scatter([px], [py], [h(px, py)],
                       s=95, facecolors='white', edgecolors='black',
                       linewidth=2, depthshade=False, zorder=6)

    # Create scatter plot with large, easy-to-see points
    ax.scatter(points[:, 0], points[:, 1], points[:, 2],
               s=300,           # Large marker size
               c=colors,        # Different colors
               alpha=1.0,
               depthshade=False,  # keep the far cities as vivid as the near ones
               edgecolors='black',  # Black edges for contrast
               linewidth=2,      # Edge line width
               zorder=10)

    # Set axis limits with extra padding for arrows
    ax.set_xlim(-0.5, 5.5)
    ax.set_ylim(-0.5, 5.5)
    ax.set_zlim(-0.5, 5.5)

    # Clean up the axes - no labels, no tick numbers
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.set_zlabel('')

    # Remove tick labels but keep tick marks
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])

    # Remove tick marks as well for completely clean look
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])

    # Draw coordinate axes INSIDE the plot boundaries, over the plane so the frame of
    # reference stays visible through it
    # X-axis from origin to x=5
    ax.plot([0, 5], [0, 0], [0, 0], 'k-', linewidth=2, zorder=8)
    # Y-axis from origin to y=5
    ax.plot([0, 0], [0, 5], [0, 0], 'k-', linewidth=2, zorder=8)
    # Z-axis from origin to z=5
    ax.plot([0, 0], [0, 0], [0, 5], 'k-', linewidth=2, zorder=8)

    # Add simple arrowheads using small lines
    # X-axis arrowhead
    ax.plot([5, 4.7], [0, 0.2], [0, 0], 'k-', linewidth=2, zorder=8)
    ax.plot([5, 4.7], [0, -0.2], [0, 0], 'k-', linewidth=2, zorder=8)

    # Y-axis arrowhead
    ax.plot([0, 0.2], [5, 4.7], [0, 0], 'k-', linewidth=2, zorder=8)
    ax.plot([0, -0.2], [5, 4.7], [0, 0], 'k-', linewidth=2, zorder=8)

    # Z-axis arrowhead
    ax.plot([0, 0.2], [0, 0], [5, 4.7], 'k-', linewidth=2, zorder=8)
    ax.plot([0, -0.2], [0, 0], [5, 4.7], 'k-', linewidth=2, zorder=8)

    # Set a good viewing angle
    ax.view_init(elev=20, azim=45)

    # Remove grid lines for cleaner appearance
    ax.grid(False)

    # Disable the 3D box and planes to remove thin lines
    ax.set_box_aspect([1, 1, 1])
    ax._axis3don = False

    # Adjust layout to prevent clipping
    plt.tight_layout()
    plt.savefig(out, dpi=300, bbox_inches='tight', transparent=True)
    plt.close(fig)

    # Trim the transparent border so the drawing fills the frame, matching how the
    # scatter already on the slide was cropped before upload.
    cropped_name = out.replace('.png', '_cropped.png')
    img = Image.open(out)
    bbox = img.getbbox()
    if bbox:
        cropped = img.crop(bbox)
        cropped.save(cropped_name)
        print(f"  {out} -> {cropped_name} {cropped.size}")


print(f"theta = {THETA}")
for (px, py, pz), c in zip(points, colors):
    print(f"  {c:<7} y={pz:<4} h(x)={h(px, py):.3f}  gap={pz - h(px, py):+.3f}")

render(False, '3d_plane_clean.png')
render(True, '3d_loss_clean.png')
