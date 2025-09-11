import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np

# Three data points
x1 = np.array([2, 4, 6])
x2 = np.array([3, 6, 9])
y = np.array([7, 8, 9])

# Create grid of theta1 and theta2 values (smaller range to show half-pipe clearly)
theta1_range = np.linspace(0, 1, 50)
theta2_range = np.linspace(0, 1.5, 50)
theta1_grid, theta2_grid = np.meshgrid(theta1_range, theta2_range)

# Function to calculate ridge regression cost
def calculate_ridge_cost(theta1, theta2, lambda_val):
    # Calculate predictions: y_pred = theta1*x1 + theta2*x2
    y_pred = theta1 * x1 + theta2 * x2
    
    # Calculate MSE
    mse = np.mean((y - y_pred)**2)
    
    # Add L2 regularization term
    regularization = lambda_val * (theta1**2 + theta2**2)
    
    return mse + regularization

# Calculate cost for each (theta1, theta2) pair with lambda=0 (MSE only)
mse_values = np.zeros_like(theta1_grid)

for i in range(len(theta1_range)):
    for j in range(len(theta2_range)):
        theta1 = theta1_grid[i, j]
        theta2 = theta2_grid[i, j]
        mse_values[i, j] = calculate_ridge_cost(theta1, theta2, 0)

# Create frames for different lambda values
frames = []
lambda_values = np.linspace(0, 20, 21)

for lambda_val in lambda_values:
    # Calculate ridge cost for this lambda
    ridge_values = np.zeros_like(theta1_grid)
    for i in range(len(theta1_range)):
        for j in range(len(theta2_range)):
            theta1 = theta1_grid[i, j]
            theta2 = theta2_grid[i, j]
            ridge_values[i, j] = calculate_ridge_cost(theta1, theta2, lambda_val)
    
    # Create frame data
    frame_data = [
        go.Surface(
            x=theta1_grid,
            y=theta2_grid,
            z=ridge_values,
            colorscale='viridis',
            opacity=0.8,
            showscale=False,
            hovertemplate='θ₁: %{x:.2f}<br>θ₂: %{y:.2f}<br>J: %{z:.2f}<extra></extra>'
        ),
        go.Surface(
            x=theta1_grid,
            y=theta2_grid,
            z=np.zeros_like(ridge_values),
            surfacecolor=ridge_values,
            colorscale='viridis',
            opacity=0.3,
            showscale=False,
            hovertemplate='θ₁: %{x:.2f}<br>θ₂: %{y:.2f}<br>J: %{surfacecolor:.2f}<extra></extra>',
            contours=dict(
                x=dict(show=True, color="black", width=1),
                y=dict(show=True, color="black", width=1),
                z=dict(show=False)
            )
        )
    ]
    
    frames.append(go.Frame(data=frame_data, name=str(lambda_val)))

# Create the 3D surface plot with initial frame (lambda=0)
fig = go.Figure(
    data=frames[0].data,
    frames=frames
)

# Add animation controls
fig.update_layout(
    title='Ridge Regression: J(θ₁,θ₂) = MSE + λ(θ₁² + θ₂²)',
    scene=dict(
        xaxis_title='θ₁',
        yaxis_title='θ₂',
        zaxis_title='J',
        xaxis=dict(range=[0, 1], showticklabels=False),
        yaxis=dict(range=[0, 1.5], showticklabels=False),
        zaxis=dict(range=[4, 30], showticklabels=False),
        camera=dict(
            eye=dict(x=1.5, y=1.5, z=1.2)
        )
    ),
    width=800,
    height=600,
    updatemenus=[{
        'type': 'buttons',
        'showactive': False,
        'buttons': [
            {
                'label': 'Play',
                'method': 'animate',
                'args': [None, {'frame': {'duration': 500, 'redraw': True}, 'fromcurrent': True}]
            },
            {
                'label': 'Pause',
                'method': 'animate',
                'args': [[None], {'frame': {'duration': 0, 'redraw': False}, 'mode': 'immediate'}]
            }
        ]
    }],
    sliders=[{
        'active': 0,
        'yanchor': 'top',
        'xanchor': 'left',
        'currentvalue': {
            'font': {'size': 20},
            'prefix': 'λ: ',
            'visible': True,
            'xanchor': 'right'
        },
        'transition': {'duration': 300, 'easing': 'cubic-in-out'},
        'pad': {'b': 10, 't': 50},
        'len': 0.9,
        'x': 0.1,
        'y': 0,
        'steps': [
            {
                'args': [[str(lambda_val)], {'frame': {'duration': 300, 'redraw': True}, 'mode': 'immediate'}],
                'label': f'{lambda_val:.1f}',
                'method': 'animate'
            } for lambda_val in lambda_values
        ]
    }]
)

# Save as HTML
fig.write_html('ridge.html')
