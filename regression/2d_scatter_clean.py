import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for saving
import matplotlib.pyplot as plt
import numpy as np

# Create figure and 2D axis
fig, ax = plt.subplots(figsize=(10, 8))

# Define 4 points in 2D space with slightly linear trend
# Adjusting positions to create a more linear relationship between temperature and energy
points_2d = np.array([
    [1.5, 1.8],  # Point 1 - lower temperature, lower energy
    [2.5, 2.2],  # Point 2 - medium-low temperature, medium-low energy
    [3.5, 2.8],  # Point 3 - medium-high temperature, medium-high energy
    [4.5, 3.4]   # Point 4 - higher temperature, higher energy
])

# Extract x, y coordinates (x_1 and energy)
x = points_2d[:, 0]
y = points_2d[:, 1]

print(f"Plotting 2D points: {points_2d}")

# Create scatter plot with large, easy-to-see points
scatter = ax.scatter(x, y, 
                     s=300,           # Large marker size
                     c=['red', 'blue', 'green', 'orange'],  # Different colors
                     alpha=0.8,        # Slight transparency
                     edgecolors='black',  # Black edges for contrast
                     linewidth=2)      # Edge line width

# Set axis limits with extra padding for arrows
ax.set_xlim(-0.5, 5.5)
ax.set_ylim(-0.5, 5.5)

# Clean up the axes - no labels, no tick numbers
ax.set_xlabel('')
ax.set_ylabel('')

# Remove tick labels but keep tick marks
ax.set_xticklabels([])
ax.set_yticklabels([])

# Remove tick marks as well for completely clean look
ax.set_xticks([])
ax.set_yticks([])

# Draw coordinate axes with arrows at the end
# X-axis from origin to x=5
ax.plot([0, 5], [0, 0], 'k-', linewidth=2)
# Y-axis from origin to y=5  
ax.plot([0, 0], [0, 5], 'k-', linewidth=2)

# Add simple arrowheads using small lines
# X-axis arrowhead
ax.plot([5, 4.7], [0, 0.2], 'k-', linewidth=2)
ax.plot([5, 4.7], [0, -0.2], 'k-', linewidth=2)

# Y-axis arrowhead
ax.plot([0, 0.2], [5, 4.7], 'k-', linewidth=2)
ax.plot([0, -0.2], [5, 4.7], 'k-', linewidth=2)

# Remove grid lines for cleaner appearance
ax.grid(False)

# Adjust layout to prevent clipping
plt.tight_layout()

print("Saving 2D plot...")
# Save the plot
plt.savefig('2d_scatter_clean.png', dpi=300, bbox_inches='tight')
print("Plot saved as '2d_scatter_clean.png'")

# Also try to show the plot if possible
try:
    plt.show()
except:
    print("Could not display plot interactively, but image was saved")
