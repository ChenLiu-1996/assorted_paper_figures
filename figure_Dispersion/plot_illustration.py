import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d
import matplotlib.cm as cm


EPSILON = 1e-6

def pairwise_sqdist(X: np.ndarray) -> np.ndarray:
    X2 = np.sum(X**2, axis=1, keepdims=True)
    D2 = X2 + X2.T - 2 * (X @ X.T)
    return np.maximum(D2, 0.0)

def safe_scale(V, s=0.6, eps=EPSILON):
    n = np.linalg.norm(V, axis=1, keepdims=True)
    return V / (eps + n) / s

def nice_axes(ax, L):
    y_scale = 0.6
    z_scale = 1.2
    ax.quiver(0, 0, 0, L, 0, 0, color="black", linewidth=2, arrow_length_ratio=0.1)
    ax.quiver(0, 0, 0, 0, -y_scale*L, 0, color="black", linewidth=2, arrow_length_ratio=0.1)
    ax.quiver(0, 0, 0, 0, 0, z_scale*L, color="black", linewidth=2, arrow_length_ratio=0.1)
    ax.text(L*1.02, 0, 0, "x", color="black", fontsize=36)
    ax.text(0, -y_scale*L*1.25, 0, "y", color="black", fontsize=36)
    ax.text(0, 0, z_scale*L*1.05, "z", color="black", fontsize=36)
    ax.grid(False)
    ax.xaxis.pane.set_visible(False)
    ax.yaxis.pane.set_visible(False)
    ax.zaxis.pane.set_visible(False)
    ax.xaxis.line.set_color((1, 1, 1, 0))
    ax.yaxis.line.set_color((1, 1, 1, 0))
    ax.zaxis.line.set_color((1, 1, 1, 0))
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    return ax

def _to3d_xy(xy):
    x, y = xy[:,0], xy[:,1]
    z = np.sqrt(np.clip(1.0 - x*x - y*y, 0.0, 1.0))
    P = np.stack([x, y, z], axis=1)
    # normalize (robust against tiny numerical drift)
    P /= np.linalg.norm(P, axis=1, keepdims=True) + EPSILON
    return P

def _slerp_arc(p, q, n=200):
    p = p / (np.linalg.norm(p) + EPSILON)
    q = q / (np.linalg.norm(q) + EPSILON)
    dot = np.clip(np.dot(p, q), -1.0, 1.0)
    theta = np.arccos(dot)
    if theta < EPSILON:  # nearly identical points
        return np.repeat(p[None, :], n, axis=0)
    # great-circle via SLERP
    t = np.linspace(0.0, 1.0, n)
    s = np.sin
    arc = (s((1-t)*theta)[:,None]*p + s(t*theta)[:,None]*q) / (s(theta) + EPSILON)
    # normalize for safety
    arc /= np.linalg.norm(arc, axis=1, keepdims=True) + EPSILON
    return arc

def draw_geodesic(ax, a2d, b2d, linestyle='-', draw_arrow=True, alpha=0.8,
                  num_points_grid=300, color='blue', lw=2.0, arrow_scale=30, shorten=0.04):
    A3, B3 = _to3d_xy(np.array([a2d])), _to3d_xy(np.array([b2d]))
    arc = _slerp_arc(A3[0], B3[0], n=num_points_grid)
    x_full, y_full = arc[:,0], arc[:,1]

    if draw_arrow:
        k0 = int(2 * shorten * num_points_grid)
        k1 = num_points_grid - k0
    else:
        k0 = int(shorten * num_points_grid)
        k1 = num_points_grid - k0
    x, y = x_full[k0:k1], y_full[k0:k1]
    ax.plot(x, y, color=color, lw=lw, solid_capstyle='round', alpha=alpha, linestyle=linestyle)

    if draw_arrow:
        # Add arrowheads at both ends
        k0 = int(shorten * num_points_grid)
        k1 = num_points_grid - k0
        x, y = x_full[k0:k1], y_full[k0:k1]
        arrow1 = FancyArrowPatch(
            (x[1], y[1]), (x[0], y[0]),
            arrowstyle='-|>', color=color,
            mutation_scale=arrow_scale, lw=0
        )
        arrow2 = FancyArrowPatch(
            (x[-2], y[-2]), (x[-1], y[-1]),
            arrowstyle='-|>', color=color,
            mutation_scale=arrow_scale, lw=0
        )
        ax.add_patch(arrow1)
        ax.add_patch(arrow2)
    return ax

class Arrow3D(FancyArrowPatch):
    def __init__(self, xs, ys, zs, *args, **kwargs):
        super().__init__((0, 0), (0, 0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def do_3d_projection(self, renderer=None):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, self.axes.get_proj())
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        return np.min(zs)

def plot_covariance(ax):
    num_points_grid = 512
    num_points_on_ellipsoid = 18
    image_scale = 2

    xs = np.linspace(-2, 2, num_points_grid)
    ys = np.linspace(-2, 2, num_points_grid)
    x, y = np.meshgrid(xs, ys)

    r2 = x**2 + y**2
    mask_s = r2 <= 1.0
    z_s = np.zeros_like(x)
    z_s[mask_s] = np.sqrt(1.0 - r2[mask_s])

    nx, ny, nz = x.copy(), y.copy(), z_s.copy()
    nrm = np.sqrt(nx**2 + ny**2 + nz**2) + EPSILON
    nx, ny, nz = nx/nrm, ny/nrm, nz/nrm

    light_dir = np.array([-0.5, -0.5, 0.8])
    light_dir /= np.linalg.norm(light_dir)
    intensity = np.maximum(0.0, nx*light_dir[0] + ny*light_dir[1] + nz*light_dir[2])

    img_s = np.ones_like(x)
    img_s[mask_s] = np.clip(0.2 + 0.9*intensity[mask_s], 0, 1)
    ax.imshow(img_s, cmap='gray',
              extent=[-image_scale, image_scale, -image_scale, image_scale],
              vmin=0, vmax=1, alpha=1)

    a, b, c = 1.50, 0.60, 0.40
    theta = np.deg2rad(30)
    ct, st = np.cos(theta), np.sin(theta)

    xL =  ct*x + st*y
    yL = -st*x + ct*y
    vL = (xL/a)**2 + (yL/b)**2
    mask_e = vL <= 1.0
    zL = np.zeros_like(xL)
    zL[mask_e] = c * np.sqrt(1.0 - vL[mask_e])

    nxL = xL/(a*a)
    nyL = yL/(b*b)
    nzL = np.zeros_like(zL)
    nzL[mask_e] = zL[mask_e]/(c*c)
    nrm = np.sqrt(nxL**2 + nyL**2 + nzL**2) + EPSILON
    nxL, nyL, nzL = nxL/nrm, nyL/nrm, nzL/nrm
    nxE = ct*nxL - st*nyL
    nyE = st*nxL + ct*nyL
    nzE = nzL

    light_dir = np.array([0.5, 0.5, -0.8])
    light_dir /= np.linalg.norm(light_dir)
    intensity_e = np.maximum(0.0, nxE*light_dir[0] + nyE*light_dir[1] + nzE*light_dir[2])
    img_e = np.full_like(x, np.nan, dtype=float)
    img_e[mask_e] = np.clip(0.5 + 0.9*intensity_e[mask_e], 0, 1)
    cmap = cm.Blues.copy()
    cmap.set_bad(color="white")
    ax.imshow(img_e, cmap=cmap, origin='lower',
              extent=[-image_scale, image_scale, -image_scale, image_scale],
              vmin=0, vmax=1, alpha=0.4)

    # Pick points on ellipsoid uniformly.
    phi = np.linspace(0, 2*np.pi, num_points_on_ellipsoid, endpoint=False)

    # Ellipsoid rim in local coords.
    xL_rim = a*np.cos(phi)
    yL_rim = b*np.sin(phi)
    zL_rim = np.zeros_like(phi)

    # rotate back to world
    xw = ct*xL_rim - st*yL_rim
    yw = st*xL_rim + ct*yL_rim
    zw = zL_rim
    Pellip = np.stack([xw, yw, zw], axis=1)

    # matching sphere rim points: same world direction in xy, unit radius, z=0
    rxy = np.sqrt(xw**2 + yw**2) + EPSILON
    Psphere = np.stack([xw/rxy, yw/rxy, np.zeros_like(rxy)], axis=1)

    ax.scatter(Pellip[:,0], Pellip[:,1], s=80, color='#0c2458', alpha=0.5)
    ax.scatter(Psphere[:,0], Psphere[:,1], s=80, facecolors='#b64342', alpha=0.5, linewidths=2)

    for p0, p1 in zip(Pellip, Psphere):
        ax.annotate("", xy=(p1[0], p1[1]), xytext=(p0[0], p0[1]),
                    arrowprops=dict(arrowstyle="->", color="#b64342", lw=3, mutation_scale=20))

    ax.set_xlim([-1.6, 1.6])
    ax.set_ylim([-2, 1.6])
    ax.set_axis_off()

    arrow_cov = Line2D([], [], color="#b64342", alpha=0.8,
                       marker=r'$\rightarrow$', linestyle="None", markersize=35, label="Covariance")
    ax.legend(handles=[arrow_cov], frameon=False, loc="lower center", fontsize=24, bbox_to_anchor=(0.5, 0.1))
    return ax

def plot_hinge(ax):
    num_points_grid = 512

    xs = np.linspace(-1, 1, num_points_grid)
    ys = np.linspace(-1, 1, num_points_grid)
    x, y = np.meshgrid(xs, ys)
    r2 = x**2 + y**2
    mask = r2 <= 1.0
    z = np.zeros_like(x)
    z[mask] = np.sqrt(1.0 - r2[mask])

    nx, ny, nz = x.copy(), y.copy(), z.copy()
    norm = np.sqrt(nx**2 + ny**2 + nz**2) + EPSILON
    nx, ny, nz = nx / norm, ny / norm, nz / norm

    # light from top left.
    light_dir = np.array([-0.5, 0.5, 0.8])
    light_dir /= np.linalg.norm(light_dir)
    intensity = np.maximum(0.0, nx*light_dir[0] + ny*light_dir[1] + nz*light_dir[2])
    ambient = 0.3
    shade = np.clip(ambient + 0.9*intensity, 0, 1)

    img = np.ones((num_points_grid, num_points_grid))
    img[mask] = shade[mask]

    ax.imshow(img, cmap='gray', origin='lower', extent=[-1, 1, -1, 1], vmin=0, vmax=1, alpha=0.5)
    ax.set_ylim([-2.1, 1.5])
    ax.set_axis_off()

    # add 3 fixed blue points on sphere
    pts = np.array([
        [-0.2, 0.6],
        [0.9, 0.0],
        [-0.75, -0.4],
    ])
    ax.scatter(pts[:,0], pts[:,1], s=80, color='#0c2458', alpha=0.5)
    ax = draw_geodesic(ax, pts[0], pts[1], color='#b64342', lw=4)
    ax = draw_geodesic(ax, pts[0], pts[2], color='#b64342', lw=4)
    ax = draw_geodesic(ax, pts[1], pts[2], linestyle='--', draw_arrow=False, color='#42949e', lw=4, alpha=0.5)
    ax = draw_geodesic(ax, pts[0], [0, 0], linestyle='--', draw_arrow=False, color='black', lw=1, alpha=0.8, shorten=0)
    ax = draw_geodesic(ax, pts[1], [0, 0], linestyle='--', draw_arrow=False, color='black', lw=1, alpha=0.8, shorten=0)
    ax = draw_geodesic(ax, pts[2], [0, 0], linestyle='--', draw_arrow=False, color='black', lw=1, alpha=0.8, shorten=0)

    ax.text(0.45, 0.4, "acute angle,\ndisperse", color="#b64342", fontsize=24, ha="center", va="center",
            bbox=dict(facecolor="white", alpha=1, edgecolor="none", boxstyle="round,pad=0.2"))
    ax.text(-0.6, 0.15, "acute angle,\ndisperse", color="#b64342", fontsize=24, ha="center", va="center",
            bbox=dict(facecolor="white", alpha=1, edgecolor="none", boxstyle="round,pad=0.2"))
    ax.text(0.15, -0.36, "obtuse angle,\ndo nothing", color="#42949e", fontsize=24, ha="center", va="center",
            bbox=dict(facecolor="white", alpha=1, edgecolor="none", boxstyle="round,pad=0.2"))

    arrow_disp = Line2D([], [], color="#b64342", alpha=0.8,
                        marker=r'$\leftarrow\rightarrow$', linestyle="None", markersize=50, label="Hinge")
    ax.legend(handles=[arrow_disp], frameon=False, loc="lower center", fontsize=24, bbox_to_anchor=(0.5, 0.15))
    return ax

def plot_infonce_l2(ax):
    num_points = 10
    tau = 0.5
    spread = 2.9

    Z = spread * np.random.randn(num_points, 3)
    d = Z.shape[1]

    D2 = pairwise_sqdist(Z) / d
    W = np.exp(-D2 / tau)

    G_disp = np.zeros_like(Z)
    for i in range(num_points):
        diff = Z[i] - Z
        G_disp[i] = (2.0 / tau) * (W[i][:, None] * diff).sum(axis=0)

    A_disp = safe_scale(G_disp)
    Vn = -Z / (np.linalg.norm(Z, axis=1, keepdims=True) + EPSILON)

    ax.view_init(elev=30, azim=-60)
    ax = nice_axes(ax, np.abs(Z).max())
    ax.set_xlim([-4, 4])
    ax.set_ylim([-3, 3])
    ax.set_zlim([-6, 4])
    ax.scatter(Z[:,0], Z[:,1], Z[:,2], s=80, color='#0c2458', alpha=0.5)

    for i in range(num_points):
        p0 = Z[i]
        p1 = Z[i] + A_disp[i] * 1.25
        arrow = Arrow3D([p0[0], p1[0]], [p0[1], p1[1]], [p0[2], p1[2]],
                        mutation_scale=16, lw=4, arrowstyle='->', color='#b64342', alpha=0.8)
        ax.add_artist(arrow)

    for i in range(num_points):
        p0 = Z[i]
        p1 = Z[i] + Vn[i] * 1.2
        arrow = Arrow3D([p0[0], p1[0]], [p0[1], p1[1]], [p0[2], p1[2]],
                        mutation_scale=10, lw=2, arrowstyle='->', color='#9a4d8e', alpha=0.8)
        ax.add_artist(arrow)

    for i in range(num_points):
        ax.plot([Z[i,0], 0], [Z[i,1], 0], [Z[i,2], 0],
                linestyle="--", color="black", linewidth=1, alpha=0.8)

    arrow_disp = Line2D([], [], color="#b64342", alpha=0.8,
                        marker=r'$\rightarrow$', linestyle="None", markersize=25,
                        label=r"InfoNCE$_{\ell_2}$")
    arrow_norm = Line2D([0, 1], [0, 0], color="#9a4d8e", alpha=0.8,
                        marker=r'$\rightarrow$', linestyle="None", markersize=25,
                        label="norm regularization")
    ax.legend(handles=[arrow_disp, arrow_norm], frameon=False, loc="lower center",
              fontsize=24, bbox_to_anchor=(0.5, 0))

    ax.text(0.8, 0.0, 8.0, "pairwise dispersion", color="#b64342", fontsize=24, ha="left", va="center",
            bbox=dict(facecolor="white", alpha=1, edgecolor="none", boxstyle="round,pad=0.2"))
    ax.text(0.8, 0.0, 6.5, "norm reduction", color="#9a4d8e", fontsize=24, ha="left", va="center",
            bbox=dict(facecolor="white", alpha=1, edgecolor="none", boxstyle="round,pad=0.2"))

    return ax


def plot_infonce_cossim(ax):
    num_points_grid = 512

    xs = np.linspace(-1, 1, num_points_grid)
    ys = np.linspace(-1, 1, num_points_grid)
    x, y = np.meshgrid(xs, ys)
    r2 = x**2 + y**2
    mask = r2 <= 1.0
    z = np.zeros_like(x)
    z[mask] = np.sqrt(1.0 - r2[mask])

    nx, ny, nz = x.copy(), y.copy(), z.copy()
    norm = np.sqrt(nx**2 + ny**2 + nz**2) + EPSILON
    nx, ny, nz = nx / norm, ny / norm, nz / norm

    # light from top left.
    light_dir = np.array([-0.5, 0.5, 0.8])
    light_dir /= np.linalg.norm(light_dir)
    intensity = np.maximum(0.0, nx*light_dir[0] + ny*light_dir[1] + nz*light_dir[2])
    ambient = 0.3
    shade = np.clip(ambient + 0.9*intensity, 0, 1)

    img = np.ones((num_points_grid, num_points_grid))
    img[mask] = shade[mask]

    ax.imshow(img, cmap='gray', origin='lower', extent=[-1, 1, -1, 1], vmin=0, vmax=1, alpha=0.5)
    ax.set_ylim([-2.1, 1.5])
    ax.set_axis_off()

    # add 3 fixed blue points on sphere
    pts = np.array([
        [-0.2, 0.6],
        [0.9, 0.0],
        [-0.75, -0.4],
    ])
    ax.scatter(pts[:,0], pts[:,1], s=80, color='#0c2458', alpha=0.5)
    ax = draw_geodesic(ax, pts[0], pts[1], color='#b64342', lw=4)
    ax = draw_geodesic(ax, pts[0], pts[2], color='#b64342', lw=4)
    ax = draw_geodesic(ax, pts[1], pts[2], color='#b64342', lw=4)
    ax = draw_geodesic(ax, pts[0], [0, 0], linestyle='--', draw_arrow=False, color='black', lw=1, alpha=0.8, shorten=0)
    ax = draw_geodesic(ax, pts[1], [0, 0], linestyle='--', draw_arrow=False, color='black', lw=1, alpha=0.8, shorten=0)
    ax = draw_geodesic(ax, pts[2], [0, 0], linestyle='--', draw_arrow=False, color='black', lw=1, alpha=0.8, shorten=0)

    ax.text(0.45, 0.4, "disperse", color="#b64342", fontsize=24, ha="center", va="center",
            bbox=dict(facecolor="white", alpha=1, edgecolor="none", boxstyle="round,pad=0.2"))
    ax.text(-0.6, 0.15, "disperse", color="#b64342", fontsize=24, ha="center", va="center",
            bbox=dict(facecolor="white", alpha=1, edgecolor="none", boxstyle="round,pad=0.2"))
    ax.text(0.15, -0.36, "disperse", color="#b64342", fontsize=24, ha="center", va="center",
            bbox=dict(facecolor="white", alpha=1, edgecolor="none", boxstyle="round,pad=0.2"))

    arrow_disp = Line2D([], [], color="#b64342", alpha=0.8,
                        marker=r'$\leftarrow\rightarrow$', linestyle="None", markersize=50, label=r"InfoNCE$_{\ell_2}$")
    ax.legend(handles=[arrow_disp], frameon=False, loc="lower center", fontsize=24, bbox_to_anchor=(0.5, 0.14))
    return ax


if __name__ == "__main__":
    save_path = './figures/illustration.png'
    np.random.seed(1)
    plt.rcParams['text.usetex'] = True
    plt.rcParams['font.family'] = 'sans-serif'
    fig = plt.figure(figsize=(24, 8))

    ax = fig.add_subplot(1, 4, 1)
    plot_covariance(ax)

    ax = fig.add_subplot(1, 4, 2)
    plot_hinge(ax)

    ax = fig.add_subplot(1, 4, 3, projection="3d")
    plot_infonce_l2(ax)

    ax = fig.add_subplot(1, 4, 4)
    plot_infonce_cossim(ax)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.tight_layout(pad=2)
    fig.savefig(save_path, dpi=300)
