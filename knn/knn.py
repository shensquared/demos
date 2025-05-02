import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import KDTree
import argparse
import os

# Parse command line arguments
parser = argparse.ArgumentParser(description='KNN Visualization')
parser.add_argument('--k', type=int, default=5, help='Number of nearest neighbors to show (default: 5)')
args = parser.parse_args()

# Create output directory
os.makedirs('knn', exist_ok=True)

# Set random seed for reproducibility
np.random.seed(390)

# Generate random points
num_points = 15
x = np.random.rand(num_points) * 10  # x coordinates between 0 and 10
y = np.random.rand(num_points) * 10  # y coordinates between 0 and 10

# Randomly assign colors (0 for blue, 1 for red)
colors = np.random.randint(0, 2, num_points)

# Create KDTree for efficient nearest neighbor search
points = np.column_stack((x, y))
tree = KDTree(points)

# Create random test points
num_test_points = 5
test_points = np.random.rand(num_test_points, 2) * 10  # Random points between 0 and 10

# First create and save the training points plot
fig_train = plt.figure(figsize=(8, 8))
ax_train = fig_train.add_subplot(111)
ax_train.set_xlim(0, 10)
ax_train.set_ylim(0, 10)
ax_train.set_xlabel('x₁')
ax_train.set_ylabel('x₂')
ax_train.grid(True, alpha=0.3)

# Plot the training points
blue_points = ax_train.scatter(x[colors == 0], y[colors == 0], color='blue', label='Class 0')
red_points = ax_train.scatter(x[colors == 1], y[colors == 1], color='red', label='Class 1')
ax_train.legend()

# Save the training data points plot
plt.tight_layout()
plt.savefig('knn/training_points.png', dpi=100)
print('Saved training_points.png')
plt.close(fig_train)

# Now create the figure for the animation frames
fig, ax = plt.subplots(figsize=(8, 8))
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
ax.set_xlabel('x₁')
ax.set_ylabel('x₂')
ax.grid(True, alpha=0.3)

# Plot the training points
blue_points = ax.scatter(x[colors == 0], y[colors == 0], color='blue', label='Class 0')
red_points = ax.scatter(x[colors == 1], y[colors == 1], color='red', label='Class 1')

# Initialize the test point and lines
test_point = ax.scatter([], [], color='green', marker='*', s=150, label='Test Point')
# Create lines for all distances
all_lines = [ax.plot([], [], 'k-', alpha=0.3, linewidth=.5)[0] for _ in range(num_points)]

def create_legend(test_point_color='green'):
    legend_elements = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=10, label='Class 0'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=10, label='Class 1'),
        plt.Line2D([0], [0], marker='*', color='w', markerfacecolor=test_point_color, markersize=15, label='Test Point')
    ]
    return ax.legend(handles=legend_elements)

# Create initial legend
legend = create_legend()

def update(frame):
    if frame == 0:
        # For frame 0, just plot the training points
        test_point.set_visible(False)  # Hide test point
        for line in all_lines:
            line.set_visible(False)  # Hide all lines
    else:
        test_point.set_visible(True)  # Show test point
        for line in all_lines:
            line.set_visible(False)  # Hide all lines
            
        # Use frame-1 to index test_points since frame 0 is for training points
        test_point.set_offsets([test_points[frame-1]])
        
        # Save frame with just training points and test point
        plt.tight_layout()
        plt.savefig(f'knn/k{args.k}_frame_{frame:03d}_1testpoint.png', dpi=100)
        print(f'Saved k{args.k}_frame_{frame:03d}_1testpoint.png')
        
        # Find all distances and nearest neighbors
        distances, indices = tree.query(test_points[frame-1], k=num_points)
        
        # Show all lines as gray dashed first
        for i, idx in enumerate(indices):
            neighbor_point = points[idx]
            all_lines[i].set_visible(True)
            all_lines[i].set_linestyle('--')
            all_lines[i].set_linewidth(1)
            all_lines[i].set_alpha(0.3)
            all_lines[i].set_color('gray')
            all_lines[i].set_data(
                [test_points[frame-1][0], neighbor_point[0]],
                [test_points[frame-1][1], neighbor_point[1]]
            )
        
        # Save frame with all gray dashed lines
        plt.tight_layout()
        plt.savefig(f'knn/k{args.k}_frame_{frame:03d}_2alllines.png', dpi=100)
        print(f'Saved k{args.k}_frame_{frame:03d}_2alllines.png')
        
        # Now highlight k nearest neighbors
        k_indices = set(indices[:args.k])  # Convert to set for faster lookup
        
        # Then show only the k nearest neighbors with solid lines
        for i, idx in enumerate(indices[:args.k]):
            neighbor_point = points[idx]
            all_lines[i].set_visible(True)
            all_lines[i].set_linestyle('--')  
            all_lines[i].set_linewidth(2)
            all_lines[i].set_alpha(1.0)
            all_lines[i].set_color('black')
            all_lines[i].set_data(
                [test_points[frame-1][0], neighbor_point[0]],
                [test_points[frame-1][1], neighbor_point[1]]
            )
        
        # Show other points with dashed lines
        for i, idx in enumerate(indices[args.k:], start=args.k):
            neighbor_point = points[idx]
            all_lines[i].set_visible(True)
            all_lines[i].set_linestyle('--')  # Dashed line
            all_lines[i].set_linewidth(1)
            all_lines[i].set_alpha(0.3)
            all_lines[i].set_color('black')
            all_lines[i].set_data(
                [test_points[frame-1][0], neighbor_point[0]],
                [test_points[frame-1][1], neighbor_point[1]]
            )
        
        # Calculate majority vote
        k_nearest_colors = colors[indices[:args.k]]
        majority_vote = np.bincount(k_nearest_colors).argmax()
        predicted_color = 'blue' if majority_vote == 0 else 'red'
        
        # Save the frame with green test point
        plt.tight_layout()
        plt.savefig(f'knn/k{args.k}_frame_{frame:03d}_3knearest.png', dpi=100)
        print(f'Saved k{args.k}_frame_{frame:03d}_3knearest.png')
        
        # Update test point color and save prediction frame
        test_point.set_color(predicted_color)
        # Update legend with predicted color
        legend = create_legend(predicted_color)
        plt.savefig(f'knn/k{args.k}_frame_{frame:03d}_4prediction.png', dpi=100)
        print(f'Saved k{args.k}_frame_{frame:03d}_4prediction.png')
        
        # Reset test point color to green for next frame
        test_point.set_color('green')
        # Reset legend to green
        legend = create_legend('green')
    
    return [test_point] + all_lines

# Generate and save all frames
for frame in range(num_test_points + 1):  # +1 for the training points frame
    update(frame)

plt.close() 