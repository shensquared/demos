import matplotlib.pyplot as plt
import numpy as np
import io
import pyperclip
from PIL import Image
import argparse

center = True
# Generate random data for 5 points
np.random.seed(0)  # Set seed for reproducibility
x_1 = np.random.rand(5)
y = np.random.rand(5)

# Create a scatter plot
# plt.scatter(x_1, y, color="#3c78d8", marker="o")

# Label the axes
x_1_centered = x_1 - np.mean(x_1)
y_centered = y - np.mean(y)

if center:
    # Center the data
    x_1_centered = x_1 - np.mean(x_1)
    y_centered = y - np.mean(y)
    # Create a new scatter plot with centered data
    plt.scatter(x_1_centered, y_centered, color="#6aa84f", marker="o")
    # Label the axes
    plt.xlabel("$x_1$ (centered)")
    plt.ylabel("$y$ (centered)")
    # Add a title
    # plt.title("Scatter plot of Centered Data")
else:
    # Create a scatter plot with original data
    plt.scatter(x_1, y, color="#3c78d8", marker="o")
    # Label the axes
    plt.xlabel("$x_1$")
    plt.ylabel("$y$")
    # Add a title
    # plt.title("Scatter plot of Original Data")

# Label the axes
plt.xlabel("")
plt.ylabel("")

# Add a title
# plt.title("Scatter plot of Temperature vs Pollution")
plt.axhline(0, color="black", linewidth=0.5)
plt.axvline(0, color="black", linewidth=0.5)
plt.gca().spines["top"].set_visible(False)
plt.gca().spines["right"].set_visible(False)
plt.gca().spines["left"].set_position("zero")
plt.gca().spines["bottom"].set_position("zero")
# Draw the axis with arrows
plt.gca().spines["left"].set_position(("data", 0))
plt.gca().spines["bottom"].set_position(("data", 0))
plt.gca().spines["left"].set_color("none")
plt.gca().spines["bottom"].set_color("none")

# Add arrows
plt.gca().xaxis.set_major_locator(plt.MaxNLocator(3))
plt.gca().yaxis.set_major_locator(plt.MaxNLocator(3))
plt.plot(1, 0, ">k", transform=plt.gca().get_yaxis_transform(), clip_on=False)
plt.plot(0, 1, "^k", transform=plt.gca().get_xaxis_transform(), clip_on=False)

# Display the plot
plt.show()
