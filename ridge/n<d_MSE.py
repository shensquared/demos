import plotly.graph_objects as go
import numpy as np

# Create a grid of theta1 and theta2 values
theta1 = np.linspace(-1, 3, 50)
theta2 = np.linspace(-1, 3, 50)
theta1_grid, theta2_grid = np.meshgrid(theta1, theta2)

# Calculate the function J(theta1, theta2) = (2*theta1 + 3*theta2 - 4)^2
J = (2 * theta1_grid + 3 * theta2_grid - 4)**2

# Create the 3D surface plot
fig = go.Figure(data=[
    go.Surface(
        x=theta1_grid,
        y=theta2_grid,
        z=J,
        colorscale='viridis',
        opacity=0.8,
        showscale=False,
        name='J(θ₁, θ₂) = (2θ₁ + 3θ₂ - 4)²',
        hovertemplate='θ₁: %{x:.2f}<br>θ₂: %{y:.2f}<br>J: %{z:.2f}<extra></extra>'
    )
])

# Add contour lines
fig.add_trace(go.Surface(
    x=theta1_grid,
    y=theta2_grid,
    z=np.zeros_like(J),  # Contour at z=0
    surfacecolor=J,
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
    title='J(θ₁, θ₂) = (2θ₁ + 3θ₂ - 4)²',
    scene=dict(
        xaxis_title='θ₁',
        yaxis_title='θ₂',
        zaxis_title='J(θ₁, θ₂)',
        xaxis=dict(range=[-1, 3], showticklabels=False),
        yaxis=dict(range=[-1, 3], showticklabels=False),
        zaxis=dict(range=[0, 20], showticklabels=False),
        camera=dict(
            eye=dict(x=1.5, y=1.5, z=1.2)
        )
    ),
    width=800,
    height=600
)

# Save as HTML
fig.write_html('n<d_MSE.html')

# Show the plot
fig.show()
