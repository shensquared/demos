import numpy as np
import matplotlib.pyplot as plt

# Create a grid of x1 and x2 values
x1_points = np.linspace(-3, 3, 9)
x2_points = np.linspace(-3, 3, 9)
x1_grid, x2_grid = np.meshgrid(x1_points, x2_points)

# Define the decision boundaries
boundary1_points = x1_grid - x2_grid + 1
boundary2_points = x1_grid - x2_grid - 1

# Create label set: positive labels outside boundaries, negative in the stripe
labels_grid = np.ones_like(x1_grid)
labels_grid[(boundary1_points > 0) & (boundary2_points < 0)] = -1

# Create a fine grid for plotting decision boundaries
x1 = np.linspace(-3, 3, 400)
x2 = np.linspace(-3, 3, 400)
x1, x2 = np.meshgrid(x1, x2)

# Define decision boundaries for the fine grid
boundary1 = x1 - x2 + 1
boundary2 = x1 - x2 - 1

# Plot the data
plt.figure(figsize=(8, 8))

# Plot decision boundaries
plt.contour(
    x1, x2, boundary1, levels=[0], colors="black", linestyles="--", linewidths=0
)
# plt.contour(
#     x1, x2, boundary2, levels=[0], colors="black", linestyles="--", linewidths=2
# )


# Mark grid points with '+' for positive and '-' for negative
for i in range(len(x1_points)):
    for j in range(len(x2_points)):
        label = "+" if labels_grid[i, j] == 1 else "-"
        color = "red" if label == "+" else "green"
        plt.text(
            x1_grid[i, j],
            x2_grid[i, j],
            label,
            fontsize=14,
            color=color,
            ha="center",
            va="center",
        )

plt.xlabel("$x_1$")
plt.ylabel("$x_2$")
plt.show()
