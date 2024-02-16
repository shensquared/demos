import plotly.graph_objects as go
import numpy as np


# Define the function and its derivative
def f(x):
    return x**2


def df(x):
    return 2 * x


# Initial parameters
x_start = 9
learning_rates = np.linspace(0.05, 0.95, 5)  # Example learning rates
iterations_options = np.arange(1, 11, 1)  # Example iteration counts

# Generate data for the function graph
x_range = np.linspace(-10, 10, 400)
y_range = f(x_range)

# Initialize figure
fig = go.Figure()

# Add trace for the function
fig.add_trace(go.Scatter(x=x_range, y=y_range, mode="lines", name="f(x) = x^2"))

# Pre-compute gradient descent paths for all combinations of learning rates and iteration counts
for lr in learning_rates:
    for iterations in iterations_options:
        x_values = [x_start]
        y_values = [f(x_start)]
        for _ in range(iterations):
            x_new = x_values[-1] - lr * df(x_values[-1])
            x_values.append(x_new)
            y_values.append(f(x_new))

        fig.add_trace(
            go.Scatter(
                x=x_values,
                y=y_values,
                mode="markers+lines",
                name=f"LR: {lr:.2f}, Iterations: {iterations}",
                visible=False,
            )
        )

# Set the first gradient descent path to be visible
fig.data[1].visible = True

# Create steps for the learning rate slider
lr_steps = []
for i, lr in enumerate(learning_rates, start=1):
    step = dict(
        method="update",
        args=[
            {"visible": [False] * (len(fig.data))},
            {"title": f"Gradient Descent: Learning Rate = {lr:.2f}"},
        ],
        label=f"{lr:.2f}",
    )
    step["args"][0]["visible"][0] = True  # Always show the function
    for j in range(1, len(fig.data)):
        if f"LR: {lr:.2f}" in fig.data[j].name:
            step["args"][0]["visible"][j] = True
    lr_steps.append(step)

# Create steps for the iteration slider
iter_steps = []
for i, iterations in enumerate(iterations_options, start=1):
    step = dict(
        method="update",
        args=[
            {"visible": [False] * (len(fig.data))},
            {"title": f"Gradient Descent: Iterations = {iterations}"},
        ],
        label=f"{iterations}",
    )
    step["args"][0]["visible"][0] = True  # Always show the function
    for j in range(1, len(fig.data)):
        if f"Iterations: {iterations}" in fig.data[j].name:
            step["args"][0]["visible"][j] = True
    iter_steps.append(step)

fig.data
# Add sliders to the figure
fig.update_layout(
    sliders=[
        dict(
            active=0,
            currentvalue={"prefix": "Learning Rate: "},
            pad={"t": 50},
            steps=lr_steps,
            x=0.0,
            len=0.5,
            xanchor="left",
            y=0,
            yanchor="top",
        ),
        dict(
            active=0,
            currentvalue={"prefix": "Iterations: "},
            pad={"t": 50},
            steps=iter_steps,
            x=0.5,
            len=0.5,
            xanchor="left",
            y=0,
            yanchor="top",
        ),
    ]
)
fig.data

fig.show()
