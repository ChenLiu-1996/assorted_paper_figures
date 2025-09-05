import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def function(x, y):
    z = 0.6 * np.exp(-((x - 1)**2 + (y + 1)**2))
    z += 0.5 * np.exp(-((x - 1)**2 + (y - 4)**2))
    z += 0.3 * np.exp(-((x - 2)**2 + (y - 2)**2))
    z += 0.2 * np.exp(-((x + 3)**2 + (y + 1)**2))
    z += 0.3 * np.exp(-((x + 1)**2 + (y + 1)**2))
    z -= 0.1 * np.exp(-((x + 1)**2 + (y - 2)**2))
    z += 0.3 * np.exp(-((x + 2)**2 + (y - 2)**2))
    z += 0.3 * np.exp(-((x + 2)**2 + (y - 1)**2))
    return z

if __name__ == '__main__':
    # Generate coordinates
    x = np.linspace(-3, 3, 200)
    y = np.linspace(-3, 3, 200)
    x, y = np.meshgrid(x, y)

    # Define a multi-well "energy" function (inverted to form valleys)
    z = function(x, y)

    # Set up plot
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(1, 1, 1, projection='3d')

    # Plot the surface with smooth shading
    ax.plot_surface(
        x, y, z,
        cmap='coolwarm',
        edgecolor='none',
        linewidth=0,
        antialiased=True,
        alpha=0.95,
    )

    # # Plot descent path
    # path_x = np.linspace(-2.5, 1, 100)
    # path_y = np.linspace(2.5, -1, 100)
    # path_z = function(path_x, path_y)
    # ax.plot(path_x, path_y, path_z, color='red', linestyle='--', linewidth=3)

    # Aesthetics
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.xaxis.pane.set_visible(False)
    ax.yaxis.pane.set_visible(False)
    ax.zaxis.pane.set_visible(False)
    ax.xaxis.line.set_color((0.0, 0.0, 0.0, 0.0))
    ax.yaxis.line.set_color((0.0, 0.0, 0.0, 0.0))
    ax.zaxis.line.set_color((0.0, 0.0, 0.0, 0.0))
    ax.set_box_aspect([1, 1, 0.5])
    ax.view_init(elev=20, azim=50)

    fig.tight_layout(pad=2)
    fig.savefig('manifold.png')
    # plt.show()