import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Define grid for x1 and x2
x1 = np.linspace(-2, 2, 50)
x2 = np.linspace(-2, 2, 50)
X1, X2 = np.meshgrid(x1, x2)

# Define arbitrary coefficients for the first ReLU function
a1, b1 = -3, -0.8
Z1 = np.maximum(a1 * X1 + b1 * X2, 0)

# Plot first ReLU function
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X1, X2, Z1, cmap='viridis', edgecolor='k', alpha=0.8)
ax.set_xticks([])  # Remove x-axis ticks
ax.set_yticks([])  # Remove y-axis ticks
ax.set_zticks([])  # Remove z-axis ticks
ax.set_xlabel("$x_1$")
ax.set_ylabel("$x_2$")
ax.set_zlabel("$z = ReLU(-3x_1 - 0.8x_2)$")
# ax.set_title("3D Plot of ReLU Function (First)")
plt.show()

# Define arbitrary coefficients for the second ReLU function
a2, b2 = 1.2, 0.9
Z2 = np.maximum(a2 * X1 + b2 * X2, 0)

# Plot second ReLU function
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X1, X2, Z2, cmap='plasma', edgecolor='k', alpha=0.8)
ax.set_xticks([])  # Remove x-axis ticks
ax.set_yticks([])  # Remove y-axis ticks
ax.set_zticks([])  # Remove z-axis ticks
ax.set_xlabel("$x_1$")
ax.set_ylabel("$x_2$")
ax.set_zlabel("$z = ReLU(1.2x_1 + 0.9x_2, 0)$")
# ax.set_title("3D Plot of ReLU Function (Second)")
plt.show()

# Compute the sum of the two ReLU functions
Z_sum = Z1 + Z2

# Plot sum of the two ReLU functions
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X1, X2, Z_sum, cmap='inferno', edgecolor='k', alpha=0.8)
ax.set_xticks([])  # Remove x-axis ticks
ax.set_yticks([])  # Remove y-axis ticks
ax.set_zticks([])  # Remove z-axis ticks
ax.set_xlabel("$x_1$")
ax.set_ylabel("$x_2$")
ax.set_zlabel("$z = z_1 + z_2)$")
# ax.set_title("3D Plot of the Sum of Two ReLU Functions")
plt.show()