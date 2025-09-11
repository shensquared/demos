import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

# Coordinates of the points
x = [-2, -1, 0, 1, 2]
y = 2 * np.array([-2, -1, 0, 1, 2])
z = [-2, -1, 0, 1, 2]

# Create a new figure
fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")

# Plot the points
ax.scatter(x, y, z, c="r", marker="o")

# Set labels
# ax.set_xlabel("X Label")
# ax.set_ylabel("Y Label")
# ax.set_zlabel("Z Label")

# Set the title
ax.set_title("3D Line Plot Centered at Origin")
# Define the planes
xx, yy = np.meshgrid(range(-2, 3), range(-2, 3))
# Adjust the meshgrid to ensure the planes pass through the points (-1, -1, -1), (0, 0, 0), and (1, 1, 1)
xx, yy = np.meshgrid(np.linspace(-1, 1, 10), np.linspace(-1, 1, 10))

# Plane equation: x + y = z
# zz = -xx - yy
# ax.plot_surface(xx, yy, zz, alpha=0.5, color="purple")
# Plane 1: x = y
# zz1 = xx
# ax.plot_surface(xx, yy, zz1, alpha=0.5, color="blue")

# # Plane 2: x = z
# zz2 = xx
# ax.plot_surface(xx, zz2, yy, alpha=0.5, color="green")

# # Plane 3: y = z
# zz3 = yy
# ax.plot_surface(zz3, yy, xx, alpha=0.5, color="yellow")

# # Plane 4: x + y + z = 0
# zz4 = -xx - yy
# ax.plot_surface(xx, yy, zz4, alpha=0.5, color="purple")
# Show the plot
plt.show()
