import numpy as np
import matplotlib.pyplot as plt

# ========================
# Dataset (non-collinear)
# ========================
X1 = np.array([1, 2, 3])
X2 = np.array([2, 1, 4])
y = np.array([3, 2, 6])
n = len(y)
X = np.column_stack([X1, X2])

# ========================
# Closed-form OLS solution
# ========================
w_opt = np.linalg.inv(X.T @ X) @ (X.T @ y)
w1_opt, w2_opt = w_opt
y_pred = X @ w_opt
min_mse = np.mean((y_pred - y) ** 2)

print("Closed-form solution (no offset):")
print(f"w₁ = {w1_opt:.4f}, w₂ = {w2_opt:.4f}, MSE = {min_mse:.4f}")

# ========================
# Define search grid around optimum
# ========================
w1_range = np.linspace(w1_opt - 1, w1_opt + 1, 100)
w2_range = np.linspace(0, 2.5, 100)
W1, W2 = np.meshgrid(w1_range, w2_range)

# Compute MSE across grid
MSE = np.zeros_like(W1)
for i in range(len(w1_range)):
    for j in range(len(w2_range)):
        w1 = W1[j, i]
        w2 = W2[j, i]
        y_pred = w1 * X1 + w2 * X2
        MSE[j, i] = np.mean((y_pred - y) ** 2)

# ========================
# Contour plot with gradient arrow
# ========================
fig, ax = plt.subplots(figsize=(10, 8))
contour = ax.contour(W1, W2, MSE, levels=30, colors='black', alpha=0.6)
ax.clabel(contour, inline=True, fontsize=8)

im = ax.imshow(MSE, extent=[w1_range.min(), w1_range.max(),
                            w2_range.min(), w2_range.max()],
               origin='lower', cmap='viridis', alpha=0.8, aspect='auto')

# Plot global optimum
ax.scatter([w1_opt], [w2_opt], color='red', s=100)

# ========================
# Test point close to global minimum
# ========================
w_test = np.array([0.25, 1.2])
y_test_pred = X @ w_test
mse_test = np.mean((y_test_pred - y) ** 2)

# Plot the test point
ax.scatter([w_test[0]], [w_test[1]], color='cyan', s=100, edgecolor='black')

# Add contour line through the test point
ax.contour(W1, W2, MSE, levels=[mse_test], colors='black', linewidths=2)

# Compute gradient at test point
grad = (2/n) * X.T @ (X @ w_test - y)

# ========================
# Individual gradient components
# ========================
# Compute individual gradients for each data point
# For MSE loss: ∇J_i = -2 * (y_i - ŷ_i) * x_i
# where ŷ_i = w₁ * x₁ᵢ + w₂ * x₂ᵢ

# Data point 1: x₁=1, x₂=2, y=3
y_pred_1 = w_test[0] * 1 + w_test[1] * 2
residual_1 = 3 - y_pred_1
grad1 = np.array([-2 * residual_1 * 1, -2 * residual_1 * 2])

# Data point 2: x₁=2, x₂=1, y=2  
y_pred_2 = w_test[0] * 2 + w_test[1] * 1
residual_2 = 2 - y_pred_2
grad2 = np.array([-2 * residual_2 * 2, -2 * residual_2 * 1])

# Data point 3: x₁=3, x₂=4, y=6
y_pred_3 = w_test[0] * 3 + w_test[1] * 4
residual_3 = 6 - y_pred_3
grad3 = np.array([-2 * residual_3 * 3, -2 * residual_3 * 4])

# Overall gradient should be the average of individual gradients
grad_avg = (grad1 + grad2 + grad3) / 3

# Verify that our manual computation matches the matrix computation
print(f"\nVerification:")
print(f"Matrix gradient: {grad}")
print(f"Manual gradient: {grad_avg}")
print(f"Difference: {np.linalg.norm(grad - grad_avg)}")

# Use same scaling for all arrows
scale = 0.2

# Individual gradient arrows
ax.arrow(w_test[0], w_test[1],
         scale * grad1[0], scale * grad1[1],
         head_width=0.03, head_length=0.06, fc='red', ec='darkred', alpha=0.7)
ax.text(w_test[0] + scale * grad1[0] + 0.05, w_test[1] + scale * grad1[1] + 0.05, 
        '$\\nabla J_1$', fontsize=10, color='red')

ax.arrow(w_test[0], w_test[1],
         scale * grad2[0], scale * grad2[1],
         head_width=0.03, head_length=0.06, fc='green', ec='darkgreen', alpha=0.7)
ax.text(w_test[0] + scale * grad2[0] + 0.05, w_test[1] + scale * grad2[1] + 0.05, 
        '$\\nabla J_2$', fontsize=10, color='green')

ax.arrow(w_test[0], w_test[1],
         scale * grad3[0], scale * grad3[1],
         head_width=0.03, head_length=0.06, fc='blue', ec='darkblue', alpha=0.7)
ax.text(w_test[0] + scale * grad3[0] + 0.05, w_test[1] + scale * grad3[1] + 0.05, 
        '$\\nabla J_3$', fontsize=10, color='blue')

# Overall gradient arrow (use the correct gradient from line 68)
ax.arrow(w_test[0], w_test[1],
         scale * grad[0], scale * grad[1],
         head_width=0.05, head_length=0.1, fc='cyan', ec='black', linewidth=2)
ax.text(w_test[0] + scale * grad[0] + 0.05, w_test[1] + scale * grad[1] + 0.05, 
        '$\\nabla J$', fontsize=12, color='cyan', weight='bold')

# Print gradient information for debugging
print(f"\nGradient at test point ({w_test[0]:.2f}, {w_test[1]:.2f}):")
print(f"Computed gradient: [{grad[0]:.4f}, {grad[1]:.4f}]")
print(f"Averaged gradient: [{grad_avg[0]:.4f}, {grad_avg[1]:.4f}]")
print(f"Gradient magnitude: {np.linalg.norm(grad):.4f}")

# ========================
# Verify gradient is perpendicular to level sets
# ========================
# The gradient should be perpendicular to the level set
# We can verify this by checking that the gradient points in the direction
# of steepest ascent (perpendicular to the contour lines)

# Compute numerical gradient to verify our analytical gradient
def compute_numerical_gradient(w, X, y, h=1e-6):
    """Compute numerical gradient using finite differences"""
    grad_num = np.zeros_like(w)
    for i in range(len(w)):
        w_plus = w.copy()
        w_plus[i] += h
        w_minus = w.copy()
        w_minus[i] -= h
        
        mse_plus = np.mean((X @ w_plus - y) ** 2)
        mse_minus = np.mean((X @ w_minus - y) ** 2)
        
        grad_num[i] = (mse_plus - mse_minus) / (2 * h)
    return grad_num

grad_numerical = compute_numerical_gradient(w_test, X, y)
print(f"Numerical gradient: [{grad_numerical[0]:.4f}, {grad_numerical[1]:.4f}]")
print(f"Analytical vs Numerical difference: {np.linalg.norm(grad - grad_numerical):.6f}")

# The gradient should point in the direction of steepest increase
# Let's also add a small arrow showing the direction perpendicular to the gradient
# (which should be tangent to the level set)
tangent = np.array([-grad[1], grad[0]])  # 90-degree rotation
tangent = tangent / np.linalg.norm(tangent)  # normalize

# Add tangent vector arrow (perpendicular to gradient)
ax.arrow(w_test[0], w_test[1],
         scale * tangent[0], scale * tangent[1],
         head_width=0.03, head_length=0.06, fc='orange', ec='darkorange', alpha=0.7)
ax.text(w_test[0] + scale * tangent[0] + 0.05, w_test[1] + scale * tangent[1] + 0.05, 
        'Tangent', fontsize=10, color='orange')

print(f"Gradient: [{grad[0]:.4f}, {grad[1]:.4f}]")
print(f"Tangent:  [{tangent[0]:.4f}, {tangent[1]:.4f}]")
print(f"Dot product (should be ~0): {np.dot(grad, tangent):.6f}")

# ========================
# Labels and formatting
# ========================
ax.set_xlabel('θ₁', fontsize=12)
ax.set_ylabel('θ₂', fontsize=12)

plt.tight_layout()
plt.savefig('/Users/shenshen/code/demos/mse_contour_with_gradient.png', dpi=300, bbox_inches='tight')
plt.show()