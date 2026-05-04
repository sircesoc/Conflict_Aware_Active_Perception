# astar_safe_ref.py
"""
Safe A* reference path planning on an occupancy grid derived from
3D Gaussian splat uncertainty. Provides grid construction, heading-aware
A* search, and spline smoothing.
"""

import argparse
import heapq
import math
import numpy as np
from scipy.io import loadmat
from scipy.ndimage import binary_dilation, distance_transform_edt

# ----------------------------
# Robot band + grid params
# ----------------------------
GROUND_Z = 2.2
ROBOT_HEIGHT = 0.5
ROBOT_RADIUS = 0.2

GRID_RESOLUTION = 0.01
OBSTACLE_PADDING = 2

# Safe A*
USE_SAFE_ASTAR = True
SAFE_RADIUS = 0.3
SAFETY_WEIGHT = 2.0

# Heading discretization for A* (independent of MPC's 15deg)
N_HEADINGS = 16
MOVE_COST = 1.0
TURN_COST = 0.5

# Visualization roof cut

# Path smoothing
SMOOTH_PATH = True
SMOOTH_NUM_POINTS = None
_EPS = 1e-12


def load_point_cloud(file_name):
    data = loadmat(file_name)
    means3D = np.asarray(data["means3D"]).copy()
    rgb_colors = np.asarray(data["rgb_colors"])
    means3D[:, [1, 2]] = means3D[:, [2, 1]]
    rows = np.all((rgb_colors >= 0) & (rgb_colors <= 1), axis=1)
    return means3D[rows], rgb_colors[rows]


def slice_at_z_range(means3D, rgb_colors, z_min, z_max):
    z = means3D[:, 2]
    mask = (z >= z_min) & (z <= z_max)
    return means3D[mask, :2], rgb_colors[mask]


def normalize_uncertainty(uncertainty):
    """Normalize uncertainty with the same log-percentile rule as Documents/cbf."""
    uncertainty = np.nan_to_num(np.asarray(uncertainty, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    safe = np.maximum(uncertainty, _EPS)
    log_values = np.log10(safe)
    lo = float(np.percentile(log_values, 1.0))
    hi = float(np.percentile(log_values, 99.0))
    if hi <= lo:
        return np.zeros_like(log_values)
    return np.clip((log_values - lo) / (hi - lo), 0.0, 1.0)


def compute_occupancy_radii_from_uncertainty(
    uncertainty,
    *,
    radius_min,
    radius_max,
    robot_radius=0.0,
    base_radius=0.0,
):
    """Map per-splat uncertainty to occupancy radii like Documents/cbf/cbf_safety.py."""
    if radius_min <= 0.0 or radius_max <= 0.0:
        raise ValueError("radius_min and radius_max must be positive")
    if radius_max < radius_min:
        raise ValueError("radius_max must be >= radius_min")
    if robot_radius < 0.0 or base_radius < 0.0:
        raise ValueError("robot_radius and base_radius must be non-negative")

    uncertainty = np.asarray(uncertainty, dtype=np.float64).reshape(-1)
    if uncertainty.size == 0:
        return np.zeros(0, dtype=np.float64)

    normalized = normalize_uncertainty(uncertainty)
    mapped = radius_min + normalized * max(0.0, radius_max - radius_min)
    return mapped + float(robot_radius) + float(base_radius)


def build_uncertainty_occupancy_grid(
    xy,
    uncertainty,
    resolution,
    *,
    radius_min,
    radius_max,
    robot_radius=0.0,
    base_radius=0.0,
):
    """
    Rasterize the occupancy grid using uncertainty-radius circles.

    The returned arrays are transposed to this module's legacy grid[ix, iy]
    layout so the heading-aware A* implementation can stay unchanged.
    """
    xy = np.asarray(xy, dtype=np.float64).reshape(-1, 2)
    if xy.shape[0] == 0:
        raise ValueError("Slice has no points")
    if resolution <= 0.0:
        raise ValueError("resolution must be positive")

    if uncertainty is None:
        uncertainty_values = np.ones(xy.shape[0], dtype=np.float64)
        normalized = np.ones(xy.shape[0], dtype=np.float64)
    else:
        uncertainty_values = np.asarray(uncertainty, dtype=np.float64).reshape(-1)
        if uncertainty_values.shape[0] != xy.shape[0]:
            raise ValueError("uncertainty length must match xy point count")
        normalized = normalize_uncertainty(uncertainty_values)

    radii = compute_occupancy_radii_from_uncertainty(
        uncertainty_values,
        radius_min=radius_min,
        radius_max=radius_max,
        robot_radius=robot_radius,
        base_radius=base_radius,
    )
    score_values = np.maximum(normalized, 1e-6).astype(np.float32)
    padding = float(np.max(radii)) + float(resolution)
    x = xy[:, 0]
    y = xy[:, 1]

    x_min = float(np.min(x) - padding)
    y_min = float(np.min(y) - padding)
    x_max = float(np.max(x) + padding)
    y_max = float(np.max(y) + padding)

    nx = max(1, int(np.ceil((x_max - x_min) / resolution)))
    ny = max(1, int(np.ceil((y_max - y_min) / resolution)))
    x_max = x_min + nx * resolution
    y_max = y_min + ny * resolution

    score = np.zeros((ny, nx), dtype=np.float32)
    x_centers = x_min + (np.arange(nx, dtype=np.float64) + 0.5) * resolution
    y_centers = y_min + (np.arange(ny, dtype=np.float64) + 0.5) * resolution

    for (px, py), radius, value in zip(xy, radii, score_values):
        ix0 = max(0, int(np.floor((px - radius - x_min) / resolution)))
        ix1 = min(nx - 1, int(np.floor((px + radius - x_min) / resolution)))
        iy0 = max(0, int(np.floor((py - radius - y_min) / resolution)))
        iy1 = min(ny - 1, int(np.floor((py + radius - y_min) / resolution)))

        xs = x_centers[ix0:ix1 + 1] - px
        ys = y_centers[iy0:iy1 + 1] - py
        mask = (ys[:, None] ** 2 + xs[None, :] ** 2) <= float(radius * radius)
        sub = score[iy0:iy1 + 1, ix0:ix1 + 1]
        np.maximum(sub, np.where(mask, value, 0.0), out=sub)

    occupied = score > 0.0
    return occupied.T.copy(), score.T.copy(), x_min, y_min, float(resolution)


def build_occupancy_grid(xy, resolution, padding=0):
    if xy.size == 0:
        raise ValueError("Slice has no points")
    x_min, x_max = xy[:, 0].min(), xy[:, 0].max()
    y_min, y_max = xy[:, 1].min(), xy[:, 1].max()

    margin = resolution * 4
    x_min -= margin
    y_min -= margin
    x_max += margin
    y_max += margin

    nx = max(1, int((x_max - x_min) / resolution) + 1)
    ny = max(1, int((y_max - y_min) / resolution) + 1)

    grid = np.zeros((nx, ny), dtype=bool)
    for i in range(xy.shape[0]):
        ix = int((xy[i, 0] - x_min) / resolution)
        iy = int((xy[i, 1] - y_min) / resolution)
        if 0 <= ix < nx and 0 <= iy < ny:
            grid[ix, iy] = True

    if padding > 0:
        grid = binary_dilation(grid, structure=np.ones((2 * padding + 1, 2 * padding + 1)))

    return grid, x_min, y_min, resolution


def build_clearance_map(grid, res):
    dist_cells = distance_transform_edt(~grid)
    return dist_cells.astype(np.float64) * res


def safety_cost(clearance_m):
    if clearance_m >= SAFE_RADIUS:
        return 0.0
    return SAFETY_WEIGHT * (SAFE_RADIUS - clearance_m)


def phi_to_index(phi, n_headings):
    phi = phi % (2 * math.pi)
    return int(round(phi / (2 * math.pi) * n_headings)) % n_headings


def index_to_phi(i, n_headings):
    return (i + 0.5) / n_headings * 2 * math.pi


def heuristic(ix, iy, goal_ix, goal_iy):
    return math.sqrt((goal_ix - ix) ** 2 + (goal_iy - iy) ** 2)


def get_neighbors_forward(ix, iy, i_phi, n_headings, grid):
    phi = index_to_phi(i_phi, n_headings)
    dx, dy = math.cos(phi), math.sin(phi)
    primary_ix = ix + (1 if dx >= 0.5 else -1 if dx <= -0.5 else 0)
    primary_iy = iy + (1 if dy >= 0.5 else -1 if dy <= -0.5 else 0)

    neighbors = []
    added = set()

    if (primary_ix != ix or primary_iy != iy) and 0 <= primary_ix < grid.shape[0] and 0 <= primary_iy < grid.shape[1] and not grid[primary_ix, primary_iy]:
        neighbors.append((primary_ix, primary_iy, i_phi, MOVE_COST))
        added.add((primary_ix, primary_iy))

    # fallback 4-neighbor
    for di, dj in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
        ixn, iyn = ix + di, iy + dj
        if (ixn, iyn) in added:
            continue
        if 0 <= ixn < grid.shape[0] and 0 <= iyn < grid.shape[1] and not grid[ixn, iyn]:
            neighbors.append((ixn, iyn, i_phi, MOVE_COST * 1.1))
            added.add((ixn, iyn))
    return neighbors


def get_neighbors_turn(ix, iy, i_phi, n_headings, grid):
    neighbors = []
    for di in (-1, 1):
        i_phi2 = (i_phi + di) % n_headings
        if not grid[ix, iy]:
            neighbors.append((ix, iy, i_phi2, TURN_COST))
    return neighbors


def get_all_successors(ix, iy, i_phi, n_headings, grid):
    out = []
    for (ix2, iy2, ip2, cost) in get_neighbors_forward(ix, iy, i_phi, n_headings, grid):
        out.append(((ix2, iy2, ip2), cost))
    for (ix2, iy2, ip2, cost) in get_neighbors_turn(ix, iy, i_phi, n_headings, grid):
        out.append(((ix2, iy2, ip2), cost))
    return out


def astar(grid, start_ix, start_iy, start_i_phi, goal_ix, goal_iy, n_headings, clearance_map=None):
    use_safe = clearance_map is not None and USE_SAFE_ASTAR
    open_set = []
    came_from = {}
    g_score = {}

    start = (start_ix, start_iy, start_i_phi)
    g_score[start] = 0.0

    h0 = heuristic(start_ix, start_iy, goal_ix, goal_iy)
    heapq.heappush(open_set, (h0, 0.0, start))
    closed = set()

    while open_set:
        _, g, (ix, iy, ip) = heapq.heappop(open_set)
        if (ix, iy, ip) in closed:
            continue
        closed.add((ix, iy, ip))

        if ix == goal_ix and iy == goal_iy:
            path = []
            cur = (ix, iy, ip)
            while cur is not None:
                path.append(cur)
                cur = came_from.get(cur)
            path.reverse()
            return path

        for (succ, step_cost) in get_all_successors(ix, iy, ip, n_headings, grid):
            ix2, iy2, ip2 = succ
            extra = safety_cost(clearance_map[ix2, iy2]) if use_safe else 0.0
            tg = g + step_cost + extra

            if tg < g_score.get(succ, float("inf")):
                g_score[succ] = tg
                came_from[succ] = (ix, iy, ip)
                h = heuristic(ix2, iy2, goal_ix, goal_iy)
                heapq.heappush(open_set, (tg + h, tg, succ))

    return None


def smooth_path_spline(xw, yw, zw, num_points=None):
    if not xw or len(xw) < 3:
        return xw, yw, zw
    n = len(xw)
    num_points = num_points or max(n * 3, 50)
    try:
        from scipy.interpolate import splprep, splev
        tck, _ = splprep([xw, yw], s=0.0, k=min(3, n - 1))
        u = np.linspace(0, 1, num_points)
        xs, ys = splev(u, tck)
        zs = [zw[0]] * num_points
        return list(xs), list(ys), zs
    except Exception:
        return xw, yw, zw


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("file", nargs="?", default="params/yq_large_cloud.mat")
    args = parser.parse_args()

    means3D, rgb_colors = load_point_cloud(args.file)

    z_band_min = GROUND_Z
    z_band_max = GROUND_Z + ROBOT_HEIGHT
    xy, rgb_slice = slice_at_z_range(means3D, rgb_colors, z_band_min, z_band_max)

    robot_radius_cells = max(1, int(math.ceil(ROBOT_RADIUS / GRID_RESOLUTION)))
    padding_cells = robot_radius_cells + OBSTACLE_PADDING
    grid, x_min, y_min, res = build_occupancy_grid(xy, GRID_RESOLUTION, padding_cells)

    clearance_map = build_clearance_map(grid, res) if USE_SAFE_ASTAR else None

    nx, ny = grid.shape
    x_max = x_min + (nx - 1) * res
    y_max = y_min + (ny - 1) * res

    # default start/goal
    start_x = x_min + 2 * res
    start_y = y_min + 2 * res
    start_phi = 0.0
    goal_x = x_max - 2 * res
    goal_y = y_max - 2 * res
    goal_phi = 0.0

    start_ix = max(0, min(nx - 1, int((start_x - x_min) / res)))
    start_iy = max(0, min(ny - 1, int((start_y - y_min) / res)))
    goal_ix = max(0, min(nx - 1, int((goal_x - x_min) / res)))
    goal_iy = max(0, min(ny - 1, int((goal_y - y_min) / res)))

    start_ip = phi_to_index(start_phi, N_HEADINGS)

    # shift start/goal if occupied
    if grid[start_ix, start_iy]:
        for r in range(1, 50):
            found = False
            for dx in range(-r, r + 1):
                for dy in range(-r, r + 1):
                    ix = start_ix + dx
                    iy = start_iy + dy
                    if 0 <= ix < nx and 0 <= iy < ny and not grid[ix, iy]:
                        start_ix, start_iy = ix, iy
                        found = True
                        break
                if found:
                    break
            if found:
                break
    if grid[goal_ix, goal_iy]:
        for r in range(1, 50):
            found = False
            for dx in range(-r, r + 1):
                for dy in range(-r, r + 1):
                    ix = goal_ix + dx
                    iy = goal_iy + dy
                    if 0 <= ix < nx and 0 <= iy < ny and not grid[ix, iy]:
                        goal_ix, goal_iy = ix, iy
                        found = True
                        break
                if found:
                    break
            if found:
                break

    print("Running Safe A* ..." if USE_SAFE_ASTAR else "Running A* ...")
    path = astar(grid, start_ix, start_iy, start_ip, goal_ix, goal_iy, N_HEADINGS, clearance_map=clearance_map)
    if not path:
        print("No path found.")
        return
    print("A* path length:", len(path))

    path_z = GROUND_Z + ROBOT_HEIGHT / 2.0

    path_xw = [x_min + ix * res for ix, iy, ip in path]
    path_yw = [y_min + iy * res for ix, iy, ip in path]
    path_zw = [path_z] * len(path_xw)

    if SMOOTH_PATH and len(path_xw) >= 3:
        n_smooth = SMOOTH_NUM_POINTS or (len(path_xw) * 3)
        path_xw, path_yw, path_zw = smooth_path_spline(path_xw, path_yw, path_zw, n_smooth)
        print("Smoothed path points:", len(path_xw))

    # 2D plot
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(
        grid.T,
        extent=[x_min, x_min + nx * res, y_min, y_min + ny * res],
        origin="lower",
        cmap="Greys",
        alpha=0.4,
        aspect="equal",
    )
    step_slice = max(1, xy.shape[0] // 80000)
    ax.scatter(xy[::step_slice, 0], xy[::step_slice, 1], c=rgb_slice[::step_slice], s=0.5, alpha=0.8)

    ax.plot(path_xw, path_yw, "b-", lw=2, label="A* ref")
    ax.scatter([path_xw[0]], [path_yw[0]], c="g", s=120, label="start")
    ax.scatter([path_xw[-1]], [path_yw[-1]], c="r", s=120, label="goal")
    ax.set_title("Safe A* reference path")
    ax.legend()
    ax.axis("equal")
    plt.show()

    # Keep 2D window alive until user closes it
    plt.show(block=True)


if __name__ == "__main__":
    main()
