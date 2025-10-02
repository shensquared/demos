import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for saving
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

# Generate data points with higher resolution for smoother surface
x1_range = np.linspace(-3, 3, 200)
x2_range = np.linspace(-3, 3, 200)
X1, X2 = np.meshgrid(x1_range, x2_range)
phi = np.abs(X1 - X2)

# Create 2D contour plot
fig, ax = plt.subplots(figsize=(10, 8))

# Create custom levels with emphasis on 2.5
levels = np.concatenate([np.linspace(0, 2.4, 13), [2.5], np.linspace(2.6, 6, 18)])
print(f"Levels include 2.5: {2.5 in levels}")
print(f"All levels: {levels}")

# Plot only contour lines (no filled contours)
contour = ax.contour(X1, X2, phi, levels=levels, colors='black', alpha=0.8, linewidths=1.2)
# Highlight the 2.5 level with darker, thicker lines
contour_25 = ax.contour(X1, X2, phi, levels=[2.5], colors='black', alpha=1.0, linewidths=3)
ax.clabel(contour, inline=True, fontsize=8)

ax.set_xlim(-3.5, 3.5)
ax.set_ylim(-3.5, 3.5)
ax.set_xlabel('')
ax.set_ylabel('')
ax.set_title('')
ax.grid(True, alpha=0.2, linewidth=0.5)
ax.set_xticks(np.arange(-3, 4, 1))
ax.set_yticks(np.arange(-3, 4, 1))
ax.set_aspect('equal')

plt.tight_layout()
plt.savefig('features/absolute-feature-2d.png', dpi=600, bbox_inches='tight')
print("2D plot saved as 'features/absolute-feature-2d.png'")

# Create 3D surface plot
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Plot surface with colormap (no separate colorbar) - smoother rendering
surface = ax.plot_surface(X1, X2, phi, cmap='viridis', alpha=0.8, linewidth=0, antialiased=True, shade=True, rstride=1, cstride=1)

# Add grey horizontal plane at level 2.5
plane_level = 2.5
plane = ax.plot_surface(X1, X2, np.full_like(X1, plane_level), color='grey', alpha=0.3, linewidth=0)

# Add intersection lines between the surface and the plane
# Find where phi = plane_level (i.e., |x1 - x2| = 2.5)
# This gives us x1 - x2 = ±2.5, so x2 = x1 ± 2.5
x1_line = np.linspace(-3, 3, 100)
x2_line1 = x1_line + 2.5  # x2 = x1 + 2.5
x2_line2 = x1_line - 2.5  # x2 = x1 - 2.5

# Filter points that are within our plot bounds
mask1 = (x2_line1 >= -3) & (x2_line1 <= 3)
mask2 = (x2_line2 >= -3) & (x2_line2 <= 3)

# Plot the intersection lines
ax.plot(x1_line[mask1], x2_line1[mask1], plane_level, 'k-', linewidth=3, alpha=0.8)
ax.plot(x1_line[mask2], x2_line2[mask2], plane_level, 'k-', linewidth=3, alpha=0.8)

ax.set_xlim(-3.5, 3.5)
ax.set_ylim(-3.5, 3.5)
ax.set_zlim(0, 6)
ax.set_xlabel('x₁', fontsize=14)
ax.set_ylabel('x₂', fontsize=14)
ax.set_zlabel('φ = |x₁ - x₂|', fontsize=14)
ax.set_title('')
ax.grid(False)
ax.view_init(elev=20, azim=75)

plt.tight_layout()
plt.savefig('features/absolute-feature-3d-level-3d.png', dpi=600, bbox_inches='tight')
print("3D plot with level plane saved as 'features/absolute-feature-3d-level-3d.png'")

# Create 3D surface plot without level plane
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Plot surface with colormap (no separate colorbar) - smoother rendering
surface = ax.plot_surface(X1, X2, phi, cmap='viridis', alpha=0.8, linewidth=0, antialiased=True, shade=True, rstride=1, cstride=1)

ax.set_xlim(-3.5, 3.5)
ax.set_ylim(-3.5, 3.5)
ax.set_zlim(0, 6)
ax.set_xlabel('x₁', fontsize=14)
ax.set_ylabel('x₂', fontsize=14)
ax.set_zlabel('φ = |x₁ - x₂|', fontsize=14)
ax.set_title('')
ax.grid(False)
ax.view_init(elev=20, azim=75)

plt.tight_layout()
plt.savefig('features/absolute-feature-3d.png', dpi=600, bbox_inches='tight')
print("3D plot without level plane saved as 'features/absolute-feature-3d.png'")

print("All plots generated successfully!")
