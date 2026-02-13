import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

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

x = np.linspace(-3, 3, 200)
y = np.linspace(-3, 3, 200)
X, Y = np.meshgrid(x, y)
Z = function(X, Y)

peak_i, peak_j = np.unravel_index(np.argmax(Z), Z.shape)
x_peak, y_peak = X[peak_i, peak_j], Y[peak_i, peak_j]

rng = np.random.default_rng(42)
num_patches = 30
r = 0.3
r_forbid = 0.9
pad = r + 0.1

centers = []
attempts = 0
while len(centers) < num_patches and attempts < 5000:
    cx = rng.uniform(x.min()+pad, x.max()-pad)
    cy = rng.uniform(y.min()+pad, y.max()-pad)
    if (cx - x_peak)**2 + (cy - y_peak)**2 < r_forbid**2:
        attempts += 1
        continue
    ok = True
    for px, py in centers:
        if (cx - px)**2 + (cy - py)**2 < (1.6*r)**2:
            ok = False
            break
    if ok:
        centers.append((cx, cy))
    attempts += 1

mask = np.zeros_like(Z, dtype=bool)
for cx, cy in centers:
    mask |= (X - cx)**2 + (Y - cy)**2 <= r**2

cmap = LinearSegmentedColormap.from_list(
    "softgreen", ["#e9f5ec", "#d9f0e1", "#c9e5d3", "#a9cbb8", "#7f9e8a", "#4f5c4f"], N=256
)

norm = plt.Normalize(Z.min(), Z.max())
facecolors = cmap(norm(Z))
facecolors_with_gray = facecolors.copy()
facecolors_with_gray[mask] = [0.7, 0.7, 0.7, 0.5]

fig = plt.figure(figsize=(14, 6))
for i, (fc, title) in enumerate([(facecolors, "Smooth Manifold"),
                                 (facecolors_with_gray, "Manifold with Gray Patches")], 1):
    ax = fig.add_subplot(1, 2, i, projection="3d")
    ax.plot_surface(
        X, Y, Z,
        facecolors=fc,
        rstride=4, cstride=4,
        linewidth=0.05, edgecolor="k",
        antialiased=True, shade=False, alpha=0.95
    )
    ax.set_title(title, fontsize=14)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
    for a in (ax.xaxis, ax.yaxis, ax.zaxis):
        a.pane.set_visible(False)
        a.line.set_color((0,0,0,0))
    ax.set_box_aspect([1, 1, 0.5])
    ax.view_init(elev=20, azim=50)

fig.tight_layout(pad=2)
os.makedirs("./figures", exist_ok=True)
fig.savefig("./figures/manifold_holes.png", dpi=300)
