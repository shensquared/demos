import numpy as np
import io
import pyperclip
from PIL import Image

import matplotlib.pyplot as plt

center = True

n = 1
# Generate random data for 5 points
np.random.seed(0)  # Set seed for reproducibility
x_1 = np.random.rand(n)
x_2 = np.random.rand(n)
y = np.random.rand(n)


# Create a 3D scatter plot
fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")


if center:
    # Center the data
    x_1 = x_1 - np.mean(x_1)
    x_2 = x_2 - np.mean(x_2)
    y = y - np.mean(y)

ax.scatter(x_1, x_2, y, color="b", marker="o")

# Mark the origin with a cross
ax.scatter(0, 0, 0, color="r", marker="x", s=100)

# if n == 1:
#     # Draw random planes that go through the origin
#     for _ in range(3):
#         # Random coefficients for the plane equation ax + by + cz = 0
#         a, b, c = np.random.rand(3) - 0.5
#         # Create a meshgrid for the plane
#         xx, yy = np.meshgrid(np.linspace(-0.5, 0.5, 10), np.linspace(-0.5, 0.5, 10))
#         zz = (-a * xx - b * yy) / c
#         ax.plot_surface(xx, yy, zz, alpha=0.5)

# ax.set_xlabel("$x_1$ (Temperature)")
# ax.set_ylabel("$x_2$ (Population)")
# ax.set_zlabel("$y$ (Pollution)")

# Set the number of ticks for each axis
# ax.set_xticks(np.linspace(0, 1, 3))
# ax.set_yticks(np.linspace(0, 1, 3))
ax.set_zticks(np.linspace(0, 1, 3))
# Display the plot
plt.show()
