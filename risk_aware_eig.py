"""
Risk-aware Expected Information Gain (EIG) computation, gradient, and simulated perception.

- Trajectory-dependent masking and proximity weighting
- EIG proxy with smooth FOV
- Numerical EIG gradient for information-ascent direction n(p)
- Jacobian of n(p) for perception barrier Lie derivatives
- Simulated observation model (distance+angle weighted uncertainty decay)
"""

import math
import numpy as np
from scipy.io import loadmat

# ──────────────────────────────────────────────────────────────
# Parameters
# ──────────────────────────────────────────────────────────────
FOV_DEG = 40.0              # camera FOV (degrees) — ±20° from heading
MAX_RANGE = 2.5             # maximum sensing range (m) — reduced for speed
BETA_PROX = 1.0             # proximity decay in EIG: exp(-beta*dist)
GAMMA1_PROX = 1.0           # proximity weight scale
GAMMA2_PROX = 1.0           # proximity weight decay
FOV_SIGMOID_K = 30.0        # sigmoid sharpness at FOV boundary

EIG_GRAD_DELTA = 0.05       # finite-difference step for EIG gradient (m)

# Simulated observation parameters
OBS_FOV_DEG = 45.0          # observation FOV half-angle (degrees)
OBS_MAX_RANGE = 2.5         # observation max range (m)
OBS_DECAY_BASE = 0.0001     # exponential decay rate at distance 0 (half-life ≈ 5000 steps)
OBS_DECAY_RANGE = 2.0       # decay length scale (m)
OBS_ANGULAR_WEIGHT = 0.00005  # angular decay rate for FOV-center Gaussians
SIGMA_MIN = 0.05            # floor for uncertainty — keeps splats as meaningful obstacles


# ──────────────────────────────────────────────────────────────
# GaussianSplatField: mutable representation of scene
# ──────────────────────────────────────────────────────────────
GRID_CELL_SIZE = 1.0        # spatial grid cell size for fast neighbor queries (m)


class GaussianSplatField:
    """Holds Gaussian splat data. sigma is mutable (updated by observations).
    Uses a spatial grid index for fast neighborhood queries."""

    def __init__(self, means3D, sigma, means_xy=None):
        """
        means3D: (N, 3) Gaussian means in world frame
        sigma:   (N,) per-Gaussian uncertainty (mutable)
        means_xy: (N, 2) optional precomputed 2D projection
        """
        self.means3D = np.asarray(means3D, dtype=np.float64)
        self.sigma = np.asarray(sigma, dtype=np.float64).copy()
        self.sigma_initial = self.sigma.copy()  # store initial uncertainty for decay floor
        self.means_xy = means_xy if means_xy is not None else self.means3D[:, :2].copy()
        self.N = self.means3D.shape[0]
        self._build_spatial_index()

    def _build_spatial_index(self):
        """Build a grid-based spatial index for fast radius queries."""
        self._cell = GRID_CELL_SIZE
        if self.N == 0:
            self._grid = {}
            return
        ix = np.floor(self.means_xy[:, 0] / self._cell).astype(int)
        iy = np.floor(self.means_xy[:, 1] / self._cell).astype(int)
        self._grid = {}
        for i in range(self.N):
            key = (int(ix[i]), int(iy[i]))
            if key not in self._grid:
                self._grid[key] = []
            self._grid[key].append(i)

    def query_radius(self, px, py, radius):
        """Return indices of Gaussians within radius of (px, py), using spatial grid."""
        r_cells = int(math.ceil(radius / self._cell))
        cx = int(math.floor(px / self._cell))
        cy = int(math.floor(py / self._cell))
        candidates = []
        for dx in range(-r_cells, r_cells + 1):
            for dy in range(-r_cells, r_cells + 1):
                key = (cx + dx, cy + dy)
                if key in self._grid:
                    candidates.extend(self._grid[key])
        if not candidates:
            return np.array([], dtype=int)
        candidates = np.array(candidates, dtype=int)
        # Exact distance filter
        d2 = (self.means_xy[candidates, 0] - px) ** 2 + (self.means_xy[candidates, 1] - py) ** 2
        return candidates[d2 <= radius ** 2]

    @classmethod
    def from_mat(cls, mat_file, z_band=None):
        """Load from .mat using same conventions as astar_safe_ref.load_point_cloud.
        z_band: optional (z_min, z_max) to filter to robot band (reduces Gaussians ~5-10x).
        """
        data = loadmat(mat_file)
        means3D = np.asarray(data["means3D"]).copy()
        means3D[:, [1, 2]] = means3D[:, [2, 1]]
        rgb_colors = np.asarray(data["rgb_colors"])
        rows = np.all((rgb_colors >= 0) & (rgb_colors <= 1), axis=1)
        means3D = means3D[rows]

        if "Uncertainty" in data:
            unc = np.asarray(data["Uncertainty"]).flatten()
            unc = unc[rows]
            unc = np.nan_to_num(unc, nan=0.0)
            umin, umax = float(unc.min()), float(unc.max())
            if umax > umin:
                unc = (unc - umin) / (umax - umin)
            else:
                unc = np.ones(means3D.shape[0]) * 0.5
        else:
            unc = np.ones(means3D.shape[0]) * 0.5

        # Filter to robot z-band
        if z_band is not None:
            z_min, z_max = z_band
            z = means3D[:, 2]
            z_mask = (z >= z_min) & (z <= z_max)
            means3D = means3D[z_mask]
            unc = unc[z_mask]

        return cls(means3D, unc)

    def filter_to_obstacle_gaussians(self, grid, x_min, y_min, res):
        """
        Return a new GaussianSplatField containing only Gaussians that fall
        in occupied grid cells. These represent actual obstacles (not floor).
        The grid is from astar_safe_ref.build_occupancy_grid.
        """
        nx, ny = grid.shape
        ix = ((self.means_xy[:, 0] - x_min) / res).astype(int)
        iy = ((self.means_xy[:, 1] - y_min) / res).astype(int)
        valid = (ix >= 0) & (ix < nx) & (iy >= 0) & (iy < ny)
        is_obstacle = np.zeros(self.N, dtype=bool)
        valid_idx = np.where(valid)[0]
        for i in valid_idx:
            if grid[ix[i], iy[i]]:
                is_obstacle[i] = True
        return GaussianSplatField(
            self.means3D[is_obstacle],
            self.sigma[is_obstacle],
        )


# ──────────────────────────────────────────────────────────────
# Angle utilities
# ──────────────────────────────────────────────────────────────
def _wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def _sigmoid(x, k=FOV_SIGMOID_K):
    """Smooth step: 1 at x>>0, 0 at x<<0."""
    return 1.0 / (1.0 + np.exp(-k * x))


# ──────────────────────────────────────────────────────────────
# EIG at a single pose
# ──────────────────────────────────────────────────────────────
EIG_NEAREST_K = 500  # max splats for EIG computation (nearest by distance)


def eig_at_pose(px, py, theta, gsplat, fov_deg=FOV_DEG, max_range=MAX_RANGE,
                nearest_k=EIG_NEAREST_K):
    """
    Risk-aware EIG proxy with proximity weighting and smooth FOV.
    Uses spatial index for fast neighbor lookup.

    EIG = sum_visible( vi * exp(-beta*dist) * sigma_i * fov_weight_i )
    """
    # Fast spatial query
    idx = gsplat.query_radius(px, py, max_range)
    if len(idx) == 0:
        return 0.0

    mu_xy = gsplat.means_xy[idx]
    sigma = gsplat.sigma[idx]

    dx = mu_xy[:, 0] - px
    dy = mu_xy[:, 1] - py
    dist = np.sqrt(dx * dx + dy * dy)

    # Subsample to nearest_k closest splats for speed
    if nearest_k is not None and len(idx) > nearest_k:
        topk = np.argpartition(dist, nearest_k)[:nearest_k]
        idx = idx[topk]
        mu_xy = mu_xy[topk]
        sigma = sigma[topk]
        dx = dx[topk]
        dy = dy[topk]
        dist = dist[topk]

    # Angular deviation from heading
    ang = np.arctan2(dy, dx)
    rel = ang - theta
    # Vectorized wrap
    rel = (rel + math.pi) % (2 * math.pi) - math.pi
    half_fov = math.radians(fov_deg) * 0.5

    # Smooth FOV weight: sigmoid at boundary
    fov_weight = _sigmoid(half_fov - np.abs(rel))

    # Proximity weight
    prox_weight = GAMMA1_PROX * np.exp(-GAMMA2_PROX * dist)

    # Exponential distance decay
    dist_weight = np.exp(-BETA_PROX * dist)

    val = prox_weight * dist_weight * fov_weight * sigma
    return float(np.sum(val))


# ──────────────────────────────────────────────────────────────
# EIG gradient via central differences
# ──────────────────────────────────────────────────────────────
def eig_gradient_numerical(px, py, theta, gsplat, delta=EIG_GRAD_DELTA):
    """
    Compute nabla_p I(p) via central differences.
    Returns (2,) gradient vector.
    """
    Ix_p = eig_at_pose(px + delta, py, theta, gsplat)
    Ix_m = eig_at_pose(px - delta, py, theta, gsplat)
    Iy_p = eig_at_pose(px, py + delta, theta, gsplat)
    Iy_m = eig_at_pose(px, py - delta, theta, gsplat)

    return np.array([
        (Ix_p - Ix_m) / (2 * delta),
        (Iy_p - Iy_m) / (2 * delta)
    ])


def information_ascent_direction(px, py, theta, gsplat, delta=EIG_GRAD_DELTA):
    """
    Normalized EIG gradient: n(p) = nabla_p I / ||nabla_p I||.
    Falls back to heading direction if gradient is near-zero.
    Returns (2,) unit vector.
    """
    grad = eig_gradient_numerical(px, py, theta, gsplat, delta)
    norm = np.linalg.norm(grad)
    if norm < 1e-8:
        return np.array([np.cos(theta), np.sin(theta)])
    return grad / norm


def eig_gradient_and_magnitude(px, py, theta, gsplat, delta=EIG_GRAD_DELTA):
    """
    Compute EIG gradient, its magnitude, and the normalized direction in one call.
    Avoids the redundant 4 EIG evaluations that would occur if calling both
    eig_gradient_numerical() and information_ascent_direction() separately.

    Returns: (grad (2,), norm scalar, direction (2,) unit vector)
    """
    grad = eig_gradient_numerical(px, py, theta, gsplat, delta)
    norm = float(np.linalg.norm(grad))
    if norm < 1e-8:
        direction = np.array([np.cos(theta), np.sin(theta)])
    else:
        direction = grad / norm
    return grad, norm, direction


EIG_HEADING_DELTA = 0.01  # finite-difference step for heading gradient (rad)


def eig_heading_gradient(px, py, theta, gsplat, delta=EIG_HEADING_DELTA):
    """
    Compute <nabla_eta I, eta_perp> = dI/dtheta via central differences.

    Uses the directional FOV-weighted EIG so the heading gradient captures
    which turn direction brings more uncertain splats into view.

    Returns: (dI_dtheta, I_max) — gradient and max of the two endpoint EIG values.
    """
    I_plus = eig_at_pose(px, py, theta + delta, gsplat)
    I_minus = eig_at_pose(px, py, theta - delta, gsplat)
    return (I_plus - I_minus) / (2.0 * delta), max(I_plus, I_minus)


# ──────────────────────────────────────────────────────────────
# Simulated observation (replaces real camera rendering)
# ──────────────────────────────────────────────────────────────
def simulate_observation(px, py, theta, gsplat,
                         fov_deg=OBS_FOV_DEG,
                         max_range=OBS_MAX_RANGE,
                         decay_base=OBS_DECAY_BASE,
                         decay_range=OBS_DECAY_RANGE,
                         angular_weight=OBS_ANGULAR_WEIGHT,
                         sigma_min=SIGMA_MIN):
    """
    Simulate active perception: reduce sigma for Gaussians in the robot's FOV
    using exponential decay toward sigma_min.

    Model:
        σ_i ← σ_min + (σ_i - σ_min) · exp(-λ_i)

    where the per-splat decay rate λ_i depends on distance and viewing angle:
        λ_i = decay_base · exp(-dist_i / decay_range) + angular_weight · cos(rel_i)

    This ensures σ → σ_min exponentially with a controllable rate,
    rather than crashing to the floor in a few steps.

    Modifies gsplat.sigma in-place.
    """
    # Fast spatial query
    idx = gsplat.query_radius(px, py, max_range)
    if len(idx) == 0:
        return

    mu_xy = gsplat.means_xy[idx]
    dx = mu_xy[:, 0] - px
    dy = mu_xy[:, 1] - py
    dist = np.sqrt(dx * dx + dy * dy)

    ang = np.arctan2(dy, dx)
    rel = (ang - theta + math.pi) % (2 * math.pi) - math.pi
    half_fov = math.radians(fov_deg)
    in_fov = np.abs(rel) <= half_fov

    if not np.any(in_fov):
        return

    vis_idx = idx[in_fov]
    dist_vis = dist[in_fov]
    rel_vis = np.abs(rel[in_fov])

    # Per-splat decay rate: stronger for closer, more centered splats
    lam = decay_base * np.exp(-dist_vis / decay_range) + angular_weight * np.cos(rel_vis)
    lam = np.maximum(lam, 0.0)

    # Per-splat floor: max decay is 80% of original uncertainty
    floor = np.maximum(0.8 * gsplat.sigma_initial[vis_idx], sigma_min)

    # Exponential decay toward per-splat floor
    excess = gsplat.sigma[vis_idx] - floor
    gsplat.sigma[vis_idx] = floor + excess * np.exp(-lam)
    np.maximum(gsplat.sigma[vis_idx], floor, out=gsplat.sigma[vis_idx])
