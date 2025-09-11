import plotly.graph_objects as go
import numpy as np

# Three data points
x1 = np.array([2, 4, 6])
x2 = np.array([3, 6, 9])
y = np.array([7, 8, 9])

# Create grid of theta1 and theta2 values (smaller range to show half-pipe clearly)
theta1_range = np.linspace(0, 1, 50)
theta2_range = np.linspace(0, 1.5, 50)
theta1_grid, theta2_grid = np.meshgrid(theta1_range, theta2_range)

# Calculate MSE for each (theta1, theta2) pair
mse_values = np.zeros_like(theta1_grid)

for i in range(len(theta1_range)):
    for j in range(len(theta2_range)):
        theta1 = theta1_grid[i, j]
        theta2 = theta2_grid[i, j]
        
        # Calculate predictions: y_pred = theta1*x1 + theta2*x2
        y_pred = theta1 * x1 + theta2 * x2
        
        # Calculate MSE
        mse = np.mean((y - y_pred)**2)
        mse_values[i, j] = mse

# Create the 3D surface plot
fig = go.Figure(data=[
    go.Surface(
        x=theta1_grid,
        y=theta2_grid,
        z=mse_values,
        colorscale='viridis',
        opacity=0.8,
        showscale=False,
        hovertemplate='θ₁: %{x:.2f}<br>θ₂: %{y:.2f}<br>J: %{z:.2f}<extra></extra>'
    )
])

# Add contour lines to show the minimum
fig.add_trace(go.Surface(
    x=theta1_grid,
    y=theta2_grid,
    z=np.zeros_like(mse_values),  # Contour at z=0
    surfacecolor=mse_values,
    colorscale='viridis',
    opacity=0.3,
    showscale=False,
    hovertemplate='θ₁: %{x:.2f}<br>θ₂: %{y:.2f}<br>J: %{surfacecolor:.2f}<extra></extra>',
    contours=dict(
        x=dict(show=True, color="black", width=1),
        y=dict(show=True, color="black", width=1),
        z=dict(show=False)
    )
))

# Update layout
fig.update_layout(
    title='J(θ₁,θ₂) = (1/3)[(7-2θ₁-3θ₂)² + (8-4θ₁-6θ₂)² + (9-6θ₁-9θ₂)²]',
    scene=dict(
        xaxis_title='θ₁',
        yaxis_title='θ₂',
        zaxis_title='J',
        xaxis=dict(range=[0, 1], showticklabels=False),
        yaxis=dict(range=[0, 1.5], showticklabels=False),
        zaxis=dict(range=[0, 20], showticklabels=False),
        camera=dict(
            eye=dict(x=1.5, y=1.5, z=1.2)
        )
    ),
    width=800,
    height=600
)

# Save as HTML
fig.write_html('colinear_MSE.html')

# Show the plot
fig.show()
