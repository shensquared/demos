import numpy as np
import matplotlib.pyplot as plt

# ========================
# Dataset
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

# Scale gradient for plotting (so arrow is visible)
scale = 0.2
ax.arrow(w_test[0], w_test[1],
         scale * grad[0], scale * grad[1],  # gradient points toward increase
         head_width=0.05, head_length=0.1, fc='cyan', ec='black')

# Add gradient label
ax.text(w_test[0] + scale * grad[0] + 0.05, w_test[1] + scale * grad[1] + 0.05, 
        '$\\nabla J$', fontsize=12, color='cyan', weight='bold')

# ========================
# Labels and formatting
# ========================
ax.set_xlabel('θ₁', fontsize=12)
ax.set_ylabel('θ₂', fontsize=12)

plt.tight_layout()
plt.savefig('/Users/shenshen/code/demos/mse_overall_only.png', dpi=300, bbox_inches='tight')
plt.show()
