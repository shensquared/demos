import numpy as np
import matplotlib.pyplot as plt

# ========================
# Dataset
# ========================
X1 = np.array([1, 2, 3])
X2 = np.array([2, 1, 4])
y = np.array([3, 2, 6])

# ========================
# Define search grid
# ========================
w1_range = np.linspace(-0.5, 1.5, 100)
w2_range = np.linspace(0, 2.5, 100)
W1, W2 = np.meshgrid(w1_range, w2_range)

# ========================
# Individual squared error terms
# ========================
# Term 1: (3 - θ₁ - 2θ₂)²
term1 = (3 - W1 - 2*W2)**2

# Term 2: (2 - 2θ₁ - θ₂)²  
term2 = (2 - 2*W1 - W2)**2

# Term 3: (6 - 3θ₁ - 4θ₂)²
term3 = (6 - 3*W1 - 4*W2)**2

# ========================
# Create three subplots
# ========================
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

terms = [term1, term2, term3]
titles = ['$J_1(\\theta) = (3 - \\theta_1 - 2\\theta_2)^2$', 
          '$J_2(\\theta) = (2 - 2\\theta_1 - \\theta_2)^2$', 
          '$J_3(\\theta) = (6 - 3\\theta_1 - 4\\theta_2)^2$']
colors = ['Reds', 'Greens', 'Blues']

for i, (term, title, cmap) in enumerate(zip(terms, titles, colors)):
    ax = axes[i]
    
    # Contour plot
    contour = ax.contour(W1, W2, term, levels=20, colors='black', alpha=0.6)
    ax.clabel(contour, inline=True, fontsize=8)
    
    # Heatmap
    im = ax.imshow(term, extent=[w1_range.min(), w1_range.max(),
                                w2_range.min(), w2_range.max()],
                   origin='lower', cmap=cmap, alpha=0.8, aspect='auto')
    
    # Find minimum point for this term
    min_idx = np.unravel_index(np.argmin(term), term.shape)
    min_w1 = W1[min_idx]
    min_w2 = W2[min_idx]
    min_val = term[min_idx]
    
    # Plot minimum point
    ax.scatter([min_w1], [min_w2], color='red', s=100)
    
    # Test point
    w_test = np.array([0.25, 1.2])
    test_val = term[np.argmin(np.abs(w2_range - w_test[1])), 
                    np.argmin(np.abs(w1_range - w_test[0]))]
    ax.scatter([w_test[0]], [w_test[1]], color='cyan', s=100, edgecolor='black')
    
    # Add contour line through test point
    ax.contour(W1, W2, term, levels=[test_val], colors='black', linewidths=2)
    
    # Compute gradient at test point for this term
    if i == 0:  # (3 - θ₁ - 2θ₂)²
        grad = np.array([-2*(3 - w_test[0] - 2*w_test[1]), -4*(3 - w_test[0] - 2*w_test[1])])
    elif i == 1:  # (2 - 2θ₁ - θ₂)²
        grad = np.array([-4*(2 - 2*w_test[0] - w_test[1]), -2*(2 - 2*w_test[0] - w_test[1])])
    else:  # (6 - 3θ₁ - 4θ₂)²
        grad = np.array([-6*(6 - 3*w_test[0] - 4*w_test[1]), -8*(6 - 3*w_test[0] - 4*w_test[1])])
    
    # Scale gradient for plotting
    scale = 0.1
    ax.arrow(w_test[0], w_test[1],
             scale * grad[0], scale * grad[1],
             head_width=0.02, head_length=0.05, fc='cyan', ec='black')
    
    # Labels
    ax.set_xlabel('$\\theta_1$', fontsize=12)
    ax.set_ylabel('$\\theta_2$', fontsize=12)
    ax.set_title(title, fontsize=14)

plt.tight_layout()
plt.savefig('/Users/shenshen/code/demos/individual_terms.png', dpi=300, bbox_inches='tight')
plt.show()

# Print minimum values for each term
print("Minimum values for each term:")
print(f"Term 1: {np.min(term1):.4f} at θ₁={min_w1:.4f}, θ₂={min_w2:.4f}")
min_idx2 = np.unravel_index(np.argmin(term2), term2.shape)
print(f"Term 2: {np.min(term2):.4f} at θ₁={W1[min_idx2]:.4f}, θ₂={W2[min_idx2]:.4f}")
min_idx3 = np.unravel_index(np.argmin(term3), term3.shape)
print(f"Term 3: {np.min(term3):.4f} at θ₁={W1[min_idx3]:.4f}, θ₂={W2[min_idx3]:.4f}")
