import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for saving
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

# Create figure and 3D axis
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Define 4 points in 3D space - positioned off-axis for maximum visibility
points = np.array([
    [2, 1.5, 1.5],    # Point 1 - center bottom area
    [3.5, 1.5, 1.5],    # Point 2 - right bottom area
    [1.5, 3.5, 1.5],    # Point 3 - back bottom area
    [3.5, 2, 3.5]     # Point 4 - top back right area
])

# Extract x, y, z coordinates
x = points[:, 0]
y = points[:, 1]
z = points[:, 2]

print(f"Plotting points: {points}")

# Create scatter plot with large, easy-to-see points
scatter = ax.scatter(x, y, z, 
                     s=300,           # Large marker size
                     c=['red', 'blue', 'green', 'orange'],  # Different colors
                     alpha=0.8,        # Slight transparency
                     edgecolors='black',  # Black edges for contrast
                     linewidth=2)      # Edge line width

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

# Draw coordinate axes INSIDE the plot boundaries
# X-axis from origin to x=5
ax.plot([0, 5], [0, 0], [0, 0], 'k-', linewidth=2)
# Y-axis from origin to y=5  
ax.plot([0, 0], [0, 5], [0, 0], 'k-', linewidth=2)
# Z-axis from origin to z=5
ax.plot([0, 0], [0, 0], [0, 5], 'k-', linewidth=2)

# Add simple arrowheads using small lines
# X-axis arrowhead
ax.plot([5, 4.7], [0, 0.2], [0, 0], 'k-', linewidth=2)
ax.plot([5, 4.7], [0, -0.2], [0, 0], 'k-', linewidth=2)

# Y-axis arrowhead
ax.plot([0, 0.2], [5, 4.7], [0, 0], 'k-', linewidth=2)
ax.plot([0, -0.2], [5, 4.7], [0, 0], 'k-', linewidth=2)

# Z-axis arrowhead
ax.plot([0, 0.2], [0, 0], [5, 4.7], 'k-', linewidth=2)
ax.plot([0, -0.2], [0, 0], [5, 4.7], 'k-', linewidth=2)

# Set a good viewing angle
ax.view_init(elev=20, azim=45)

# Remove grid lines for cleaner appearance
ax.grid(False)

# Disable the 3D box and planes to remove thin lines
ax.set_box_aspect([1,1,1])
ax._axis3don = False

# Adjust layout to prevent clipping
plt.tight_layout()

print("Saving plot...")
# Save the plot
plt.savefig('3d_scatter_clean.png', dpi=300, bbox_inches='tight')
print("Plot saved as '3d_scatter_clean.png'")

# Also try to show the plot if possible
try:
    plt.show()
except:
    print("Could not display plot interactively, but image was saved")
