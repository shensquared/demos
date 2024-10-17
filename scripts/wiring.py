import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os

blue = "#9fc5e8"
red = "#ffe599"


def plot_circles_and_edges(ax, layers, connections):
    ax.clear()  # Clear the previous plot before drawing a new one

    # Set some parameters for the circle size and spacing
    circle_radius = 0.5
    vertical_spacing = 2.0

    # To store the circle positions for connecting them
    layer_positions = []

    # Step 1: Calculate all the positions first (so we can plot edges behind circles)
    x_position = 0
    for num_circles in layers:
        y_start = (
            (num_circles - 1) * vertical_spacing / 2
        )  # Center the circles, first at the top
        current_layer_positions = []
        for i in range(num_circles):
            y = y_start - i * vertical_spacing  # Vertical position, top to bottom
            current_layer_positions.append((x_position, y))  # Store position for edges
        layer_positions.append(current_layer_positions)
        x_position += vertical_spacing * 2  # Move to the right for the next layer

    # Step 2: Draw the edges (this happens first, so they appear behind the circles)
    if connections == "fc":
        # Full connection for all layers
        for layer_idx in range(
            len(layer_positions) - 1
        ):  # For each layer except the last one
            for circle_idx in range(
                len(layer_positions[layer_idx])
            ):  # Each circle in current layer
                start_circle = layer_positions[layer_idx][circle_idx]
                for next_circle_idx in range(
                    len(layer_positions[layer_idx + 1])
                ):  # Each circle in next layer
                    end_circle = layer_positions[layer_idx + 1][next_circle_idx]
                    # Draw black edges between every circle in the current layer to every circle in the next layer
                    ax.plot(
                        [start_circle[0], end_circle[0]],
                        [start_circle[1], end_circle[1]],
                        color="black",
                        zorder=1,
                    )
    else:
        # Handle custom connections (previous behavior)
        for layer_idx, layer_connection in enumerate(connections):
            if layer_idx + 1 < len(
                layer_positions
            ):  # Ensure we don't go beyond the last layer
                for circle_idx, connection_list in enumerate(layer_connection):
                    start_circle = layer_positions[layer_idx][circle_idx]

                    if connection_list == "fc":
                        # Create full connections between this circle and all circles in the next layer
                        for next_circle_idx in range(
                            len(layer_positions[layer_idx + 1])
                        ):
                            end_circle = layer_positions[layer_idx + 1][next_circle_idx]
                            ax.plot(
                                [start_circle[0], end_circle[0]],
                                [start_circle[1], end_circle[1]],
                                color="black",
                                zorder=1,
                            )
                    else:
                        # Handle individual connections
                        for connect_item in connection_list:
                            if (
                                isinstance(connect_item, list)
                                and len(connect_item) == 2
                            ):
                                connect_to_idx = connect_item[0]  # Get the index
                                color = connect_item[1]  # Get the color
                            else:
                                connect_to_idx = connect_item
                                color = "black"

                            if connect_to_idx - 1 < len(layer_positions[layer_idx + 1]):
                                end_circle = layer_positions[layer_idx + 1][
                                    connect_to_idx - 1
                                ]
                                # Draw a line between the circles with the specified or default color
                                ax.plot(
                                    [start_circle[0], end_circle[0]],
                                    [start_circle[1], end_circle[1]],
                                    color=color,
                                    zorder=1,
                                )

    # Step 3: Draw the circles (this happens after edges, so circles appear in front)
    for num_circles, current_layer_positions in zip(layers, layer_positions):
        for x, y in current_layer_positions:
            # Draw the circle with a higher z-order to ensure it is drawn on top of the edges
            circle = plt.Circle(
                (x, y),
                circle_radius,
                fill=True,
                facecolor="white",
                edgecolor="black",
                zorder=2,
            )
            ax.add_artist(circle)

    # Remove axis spines, ticks, and labels
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_frame_on(False)  # Remove the box boundary

    # Adjust axis limits to prevent cutting off the plot
    x_min = min([pos[0] for layer in layer_positions for pos in layer]) - circle_radius
    x_max = max([pos[0] for layer in layer_positions for pos in layer]) + circle_radius
    y_min = min([pos[1] for layer in layer_positions for pos in layer]) - circle_radius
    y_max = max([pos[1] for layer in layer_positions for pos in layer]) + circle_radius

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)

    # Set equal aspect ratio to ensure circles remain circles, not ovals
    ax.set_aspect("equal")  # This line ensures that the aspect ratio is equal


def create_gif_or_svg(
    layers, connections_list, output_filename="network_animation.gif", save_as_gif=True
):
    fig, ax = plt.subplots()

    # Remove margins and padding from the figure layout
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

    if save_as_gif:
        # Generate a GIF
        def update(frame):
            plot_circles_and_edges(ax, layers, connections_list[frame])

        # Create an animation object with the update function
        ani = animation.FuncAnimation(
            fig, update, frames=len(connections_list), interval=500
        )

        # Save the animation as a GIF
        ani.save(output_filename, writer="imagemagick", fps=2)

        plt.close(fig)  # Close the figure after saving the animation

    else:
        # Save each frame as an SVG
        if not os.path.exists("svg_frames"):
            os.makedirs("svg_frames")  # Create directory to store the SVG frames

        for i, connections in enumerate(connections_list):
            plot_circles_and_edges(ax, layers, connections)
            svg_filename = f"svg_frames/frame_{i+1}.png"
            # Save the current frame as an SVG with tight bounding box and no margins
            fig.savefig(svg_filename, bbox_inches="tight", pad_inches=0.2, format="png")
            print(f"Saved {svg_filename}")

        plt.close(fig)  # Close the figure after saving all frames


# Example usage
layers = [6, 4, 4, 3]  # Fixed number of circles in each vertical layer

# Varying connections over time (frames), with 'fc' for full connection
fc_connect_list = [
    [
        [[1], [1], [1], [1], [1]],
    ],
    [
        [[1, 2], [1, 2], [1, 2], [1, 2], [1, 2]],
    ],
    [
        [[1, 2, 3], [1, 2, 3], [1, 2, 3], [1, 2, 3], [1, 2, 3]],
    ],
    "fc",
]

c_conncect_list = [
    [
        [[1], [1], [], [], []],
    ],
    [
        [[1], [1, 2], [2], [], []],
    ],
    [
        [[1], [1, 2], [2, 3], [3], []],
    ],
    [
        [[1], [1, 2], [2, 3], [3, 4], [4]],
    ],
]

max_pool = [
    [
        [[], [2], [2], [], []],
    ],
    [[1], [2], [3], [4]],
    [[], [2], [2], []],
]
# Choose between creating a GIF or SVGs for each frame
create_gif_or_svg(
    layers, max_pool, output_filename="network_animation.gif", save_as_gif=False
)
