import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for saving
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from mpl_toolkits.mplot3d import Axes3D

# Loss picture for the 2-feature running example. Geometry, view angle, colors and
# point positions match 3d_scatter_clean.py so both slides show the same four
# cities; this one adds the hypothesis as a plane and the vertical gap from each
# city down to it.

# The hypothesis is h(x) = theta1*x1 + theta2*x2, a hyperplane through the origin
# with no offset, matching the "for now, ignoring the offset" framing on the linear
# hypothesis class slide. theta is deliberately a poor fit so all four gaps stay
# visible, and it sits under every city so all four gaps drop the same direction.
# The two weights differ so the plane tilts visibly. Weighting both equally makes it
# rise toward the far corner, which this viewing angle foreshortens into something
# that reads as horizontal. Blue, at (3.5, 1.5, 1.5), caps how far theta1 can go
# before the plane climbs above that city and the gaps start pointing both ways.
THETA = (0.35, 0.05)

# Points and colors from 3d_scatter_clean.py: Chicago, New York, Boston, San Diego
points = np.array([
    [2, 1.5, 1.5],    # Point 1 - center bottom area
    [3.5, 1.5, 1.5],    # Point 2 - right bottom area
    [1.5, 3.5, 1.5],    # Point 3 - back bottom area
    [3.5, 2, 3.5]     # Point 4 - top back right area
])
colors = ['red', 'blue', 'green', 'orange']

x = points[:, 0]
y = points[:, 1]
z = points[:, 2]


def h(px, py):
    return THETA[0] * px + THETA[1] * py


# Every city sits above the plane, so drawing the points and gap lines over it is
# the geometrically correct order. computed_zorder=False makes matplotlib honor
# that instead of deriving its own, which otherwise chops the gap lines in half.
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d', computed_zorder=False)

# Hypothesis plane through the origin. Gray at low opacity, matching the plane on
# the linear hypothesis class slide.
gx, gy = np.meshgrid(np.linspace(0, 4.6, 2), np.linspace(0, 4.6, 2))
ax.plot_surface(gx, gy, h(gx, gy),
                color='#cccccc', alpha=0.39, shade=False,
                edgecolor='#9a9a9a', linewidth=1.2, zorder=1)

# Vertical gap from each city down to the plane, then a hollow ring where it lands.
# A plane seen in projection gives no cue for where a vertical meets it, so without
# the ring each gap reads as stopping in mid-air. Point 4 is the worked example
# named in the slide's callout and carries the heaviest line.
for (px, py, pz), c in zip(points, colors):
    highlight = (pz == 3.5)
    ax.plot([px, px], [py, py], [pz, h(px, py)],
            color=c,
            linewidth=5.5 if highlight else 4,
            solid_capstyle='butt',
            zorder=5)
    ax.scatter([px], [py], [h(px, py)],
               s=130, facecolors='white', edgecolors=c,
               linewidth=3, depthshade=False, zorder=6)

# Create scatter plot with large, easy-to-see points
scatter = ax.scatter(x, y, z,
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

print(f"theta = {THETA}")
for (px, py, pz), c in zip(points, colors):
    print(f"  {c:<7} y={pz:<4} h(x)={h(px, py):.3f}  gap={pz - h(px, py):+.3f}")

plt.savefig('3d_loss_clean.png', dpi=300, bbox_inches='tight', transparent=True)

# Trim the transparent border so the drawing fills the frame, matching how the
# scatter already on the slide was cropped before upload.
img = Image.open('3d_loss_clean.png')
bbox = img.getbbox()
if bbox:
    cropped = img.crop(bbox)
    cropped.save('3d_loss_clean_cropped.png')
    print(f"Cropped {img.size} to {cropped.size} as '3d_loss_clean_cropped.png'")
