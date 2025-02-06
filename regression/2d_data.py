import matplotlib.pyplot as plt
import numpy as np
import io
import pyperclip
from PIL import Image

# Generate random data for 5 points
np.random.seed(0)  # Set seed for reproducibility
x_1 = np.random.rand(5)
y = np.random.rand(5)

# Create a scatter plot
plt.scatter(x_1, y, color="#3c78d8", marker="o")

# Label the axes
plt.xlabel("$x_1$")
plt.ylabel("$y$")

# Add a title

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

# Save the plot to a BytesIO object
# buf = io.BytesIO()
# plt.savefig(buf, format="png")
# buf.seek(0)

# # Open the image with PIL and copy to clipboard
# image = Image.open(buf)
# output = io.BytesIO()
# image.convert("RGB").save(output, "BMP")
# data = output.getvalue()[14:]
# output.close()
# pyperclip.copy(data)

plt.show()
