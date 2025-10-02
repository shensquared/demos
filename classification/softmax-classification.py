import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for saving
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import plotly.graph_objects as go
import plotly.offline as pyo

# Generate grid
x1 = np.linspace(-5, 5, 100)
x2 = np.linspace(-5, 5, 100)
X1, X2 = np.meshgrid(x1, x2)

# Compute logits
z1 = X1 + X2
z2 = 2*X1 - X2
z3 = 3*X1 + X2

# Stack logits and compute softmax probabilities
logits = np.stack([z1, z2, z3], axis=0)
# Softmax computation: exp(z_i) / sum(exp(z_j))
exp_logits = np.exp(logits - np.max(logits, axis=0, keepdims=True))  # Numerical stability
softmax_probs = exp_logits / np.sum(exp_logits, axis=0, keepdims=True)

# Extract individual softmax probabilities
p1 = softmax_probs[0]  # P(class 1)
p2 = softmax_probs[1]  # P(class 2) 
p3 = softmax_probs[2]  # P(class 3)

# Create 3D plot showing all three softmax probability surfaces
fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection='3d')

# Plot each softmax probability as a surface
surface1 = ax.plot_surface(X1, X2, p1, alpha=0.7, color='red', 
                          linewidth=0, antialiased=True, shade=True, label='P(class 1)')
surface2 = ax.plot_surface(X1, X2, p2, alpha=0.7, color='green', 
                          linewidth=0, antialiased=True, shade=True, label='P(class 2)')
surface3 = ax.plot_surface(X1, X2, p3, alpha=0.7, color='blue', 
                          linewidth=0, antialiased=True, shade=True, label='P(class 3)')

# Add some styling
ax.set_xlabel('$x_1$', fontsize=12)
ax.set_ylabel('$x_2$', fontsize=12)
ax.set_zlabel('$P$ (probability)', fontsize=12)
ax.set_title('Softmax Probability Surfaces', fontsize=14)

# Set viewing angle
ax.view_init(elev=20, azim=45)

plt.tight_layout()
plt.savefig('classification/softmax-3d-probabilities.png', dpi=300, bbox_inches='tight')
print("3D softmax probability surfaces plot saved as 'classification/softmax-3d-probabilities.png'")
plt.close()

# Create individual 3D plots for each softmax probability
components = [
    (p1, 'red', 'P(class 1)', 'softmax-p1-3d'),
    (p2, 'green', 'P(class 2)', 'softmax-p2-3d'),
    (p3, 'blue', 'P(class 3)', 'softmax-p3-3d')
]

for prob, color, title, filename in components:
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot surface
    surface = ax.plot_surface(X1, X2, prob, alpha=0.8, color=color, 
                             linewidth=0, antialiased=True, shade=True)
    
    # Add wireframe for better definition
    ax.plot_wireframe(X1, X2, prob, alpha=0.3, color='black', linewidth=0.5, rstride=5, cstride=5)
    
    ax.set_xlabel('$x_1$', fontsize=12)
    ax.set_ylabel('$x_2$', fontsize=12)
    ax.set_zlabel('$P$', fontsize=12)
    ax.set_title(f'Softmax Probability: {title}', fontsize=14)
    ax.set_zlim(0, 1)  # Probabilities are between 0 and 1
    
    ax.view_init(elev=20, azim=45)
    
    plt.tight_layout()
    plt.savefig(f'classification/{filename}.png', dpi=300, bbox_inches='tight')
    print(f"Individual probability plot saved as 'classification/{filename}.png'")
    plt.close()

# Create refined 2D decision boundary plot
fig = plt.figure(figsize=(12, 10))

# Get the decision regions
decision_map = np.argmax(np.stack([z1, z2, z3]), axis=0)

# Create filled contours with better colors
plt.contourf(X1, X2, decision_map, levels=3, alpha=0.4, colors=['#FF6B6B', '#4ECDC4', '#45B7D1'])

# Add contour lines for boundaries
contour = plt.contour(X1, X2, decision_map, levels=3, colors=['black'], linewidths=2, linestyles='-')

# Add the actual boundary lines for reference
x_line = np.linspace(-5, 5, 100)
plt.plot(x_line, x_line/2, 'k--', linewidth=2, alpha=0.8, label='z₁ = z₂')
plt.axvline(x=0, color='black', linestyle='--', linewidth=2, alpha=0.8, label='z₁ = z₃')
plt.plot(x_line, -x_line/2, 'k--', linewidth=2, alpha=0.8, label='z₂ = z₃')

# Add some sample points to show the regions
sample_points = [
    (-3, 2, 'Class 1', 'red'),
    (2, -1, 'Class 2', 'green'), 
    (2, 2, 'Class 3', 'blue')
]

for x, y, label, color in sample_points:
    plt.scatter(x, y, s=100, c=color, edgecolors='black', linewidth=2, zorder=5)
    plt.annotate(label, (x, y), xytext=(10, 10), textcoords='offset points', 
                fontsize=12, fontweight='bold', bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

plt.xlabel('$x_1$', fontsize=14, fontweight='bold')
plt.ylabel('$x_2$', fontsize=14, fontweight='bold')
plt.title('Decision Boundaries for 3-class Softmax Classification', fontsize=16, fontweight='bold')
plt.legend(fontsize=12, loc='upper right')
plt.grid(True, alpha=0.3)
plt.xlim(-5, 5)
plt.ylim(-5, 5)
plt.axis('equal')
plt.savefig('classification/softmax-classification.png', dpi=300, bbox_inches='tight')
print("Softmax classification plot saved as 'classification/softmax-classification.png'")

# Create interactive 3D plot with Plotly
print("Creating interactive 3D plot...")

# Create the interactive 3D surface plot
fig = go.Figure()

# Add each softmax probability surface
fig.add_trace(go.Surface(
    x=X1, y=X2, z=p1,
    colorscale='Reds',
    name='P(class 1)',
    opacity=0.7,
    showscale=False
))

fig.add_trace(go.Surface(
    x=X1, y=X2, z=p2,
    colorscale='Greens', 
    name='P(class 2)',
    opacity=0.7,
    showscale=False
))

fig.add_trace(go.Surface(
    x=X1, y=X2, z=p3,
    colorscale='Blues',
    name='P(class 3)', 
    opacity=0.7,
    showscale=False
))

# Update layout
fig.update_layout(
    title='Interactive Softmax Probability Surfaces (Rotatable)',
    scene=dict(
        xaxis_title='x₁',
        yaxis_title='x₂', 
        zaxis_title='P (probability)',
        zaxis=dict(range=[0, 1]),
        camera=dict(
            eye=dict(x=1.5, y=1.5, z=1.5)
        )
    ),
    width=800,
    height=600
)

# Save as HTML file
pyo.plot(fig, filename='classification/softmax-3d-interactive.html', auto_open=False)
print("Interactive 3D plot saved as 'classification/softmax-3d-interactive.html'")
print("Open the HTML file in a web browser to interact with the 3D plot!")