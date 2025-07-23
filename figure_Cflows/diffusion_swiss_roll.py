import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform

# Generate Swiss Roll data
def generate_swiss_roll_2d(n_samples=80, noise=0.1):
    t = 1.5 * np.pi * (1 + 2 * np.random.rand(n_samples))
    x = t * np.cos(t)
    z = t * np.sin(t)

    # Add some noise
    x += noise * np.random.randn(n_samples)
    z += noise * np.random.randn(n_samples)

    return x, z, t

# Compute diffusion matrix (transition probabilities)
def compute_diffusion_matrix(x, z, t, sigma=1.0):
    # Order points by manifold parameter t for better matrix visualization
    sorted_indices = np.argsort(t)
    x_sorted = x[sorted_indices]
    z_sorted = z[sorted_indices]
    t_sorted = t[sorted_indices]

    # Compute pairwise distances in ambient space
    points = np.column_stack([x_sorted, z_sorted])
    distances = squareform(pdist(points))

    # Compute manifold distances (along parameter t)
    t_distances = np.abs(t_sorted[:, None] - t_sorted[None, :])

    # Combine spatial and manifold distances (emphasize manifold structure)
    combined_distances = distances + 0.5 * t_distances

    # Gaussian kernel for transition probabilities
    P = np.exp(-combined_distances**2 / (2 * sigma**2))

    # Make matrix sparser by thresholding small values
    P[P < 0.01] = 0

    # Normalize rows to make it a proper transition matrix
    row_sums = P.sum(axis=1)
    row_sums[row_sums == 0] = 1  # Avoid division by zero
    P = P / row_sums[:, None]

    return P, sorted_indices

if __name__ == '__main__':
    # Create two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

    # Generate swiss roll point cloud (smaller for visualization clarity)
    x, z, t = generate_swiss_roll_2d(n_samples=500, noise=0.5)

    # Compute diffusion matrix
    P, sorted_indices = compute_diffusion_matrix(x, z, t, sigma=2)

    # Left plot: Diffusion Matrix
    im = ax1.imshow(P, cmap='Reds', aspect='equal', origin='upper')
    ax1.axis('off')
    ax1.set_facecolor('white')

    # Right plot: Swiss Roll with probability-weighted connections
    # Use original (unsorted) coordinates for the swiss roll plot
    x_orig, z_orig, t_orig = x, z, t

    # Draw line segments between points with opacity = transition probability
    threshold = 0.02  # Only draw lines above this probability threshold

    for i in range(len(x_orig)):
        for j in range(i+1, len(x_orig)):
            # Find corresponding indices in sorted matrix
            orig_i_in_sorted = np.where(sorted_indices == i)[0][0]
            orig_j_in_sorted = np.where(sorted_indices == j)[0][0]

            # Get transition probability from matrix
            prob = max(P[orig_i_in_sorted, orig_j_in_sorted], P[orig_j_in_sorted, orig_i_in_sorted])

            if prob > threshold:
                ax2.plot([x_orig[i], x_orig[j]], [z_orig[i], z_orig[j]],
                        color='black', linewidth=2, alpha=prob*2, zorder=1)

    # Plot the swiss roll points on top
    scatter = ax2.scatter(x_orig, z_orig, c=t_orig, cmap='viridis', s=100,
                          alpha=0.5, edgecolors='white', linewidth=1, zorder=2)

    # Styling for swiss roll plot
    ax2.set_aspect('equal')
    ax2.axis('off')
    ax2.set_facecolor('white')

    # Clean overall styling
    fig.patch.set_facecolor('white')
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0.02)

    plt.savefig('figures/diffusion_swiss_roll.png', dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none', pad_inches=1)
