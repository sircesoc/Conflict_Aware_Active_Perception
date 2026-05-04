"""
Compare the two CBF formulations on the SAME scene + start/goal:

      —  single h_p = I(π,η) - I_c  with ONE slack δ
       —  h_π (spatial) + h_η (angular) with TWO slacks δ_s, δ_η

Metrics compared:
  • Motion smoothness:  velocity, acceleration, jerk, snap
  • Total scene uncertainty reduction (Σ σ_init − Σ σ_final)
  • Trajectory length (sum of Euclidean step distances)
  • Number of steps and wall time
  • Safety margins (h_s min/mean, true min ρ_i)

Usage:
  python compare_formulations.py params/yq_large_cloud.mat \
      --dt 0.05 --beta-lse 200 --i-c 0.3 \
      --safety-horizon-s 0.3 --safety-horizon-n 4 \
      --start 0.0 -10 --goal 2.5 -0.8 --start-phi 1.3
"""

import argparse
import math
import time
import numpy as np
import matplotlib.pyplot as plt

from astar_safe_ref import (
    load_point_cloud,
    slice_at_z_range,
    build_uncertainty_occupancy_grid,
    build_clearance_map,
    astar,
    smooth_path_spline,
    phi_to_index,
    GROUND_Z,
    ROBOT_HEIGHT,
    GRID_RESOLUTION,
    N_HEADINGS,
)
from unicycle_dynamics import UnicycleDynamics, wrap_angle
from risk_aware_eig import (
    GaussianSplatField,
    simulate_observation,
)
from conflict_cbf import (
    ConflictAwareCBFController,
    Equation32Controller,
    compute_occupancy_radii_from_uncertainty,
    OCCUPANCY_RADIUS_MIN,
    OCCUPANCY_RADIUS_MAX,
    OCCUPANCY_NEAREST_K,
)


# ──────────────────────────────────────────────────────────────
# Metrics
# ──────────────────────────────────────────────────────────────
def compute_motion_derivatives(xs, ys, dt):
    """
    Given position traces and dt, compute:
        v (speed)           = ‖(dx, dy)‖ / dt
        a (accel magnitude) = ‖d²(x,y)/dt²‖
        j (jerk magnitude)  = ‖d³(x,y)/dt³‖
        s (snap magnitude)  = ‖d⁴(x,y)/dt⁴‖
    via successive finite differences.
    Returns four 1-D arrays (each shorter than xs by one more than the prev).
    """
    xy = np.column_stack([np.asarray(xs, float), np.asarray(ys, float)])
    v_vec = np.diff(xy, axis=0) / dt
    a_vec = np.diff(v_vec, axis=0) / dt
    j_vec = np.diff(a_vec, axis=0) / dt
    s_vec = np.diff(j_vec, axis=0) / dt
    v = np.linalg.norm(v_vec, axis=1)
    a = np.linalg.norm(a_vec, axis=1)
    j = np.linalg.norm(j_vec, axis=1)
    s = np.linalg.norm(s_vec, axis=1)
    return v, a, j, s


def path_arc_lengths(poly_xy):
    """Cumulative arc lengths at each vertex. First is 0."""
    poly_xy = np.asarray(poly_xy, dtype=float)
    if len(poly_xy) < 2:
        return np.array([0.0])
    seg_lens = np.sqrt(np.sum(np.diff(poly_xy, axis=0) ** 2, axis=1))
    return np.concatenate([[0.0], np.cumsum(seg_lens)])


def trajectory_length(xs, ys):
    xy = np.column_stack([np.asarray(xs), np.asarray(ys)])
    return float(np.sum(np.linalg.norm(np.diff(xy, axis=0), axis=1)))


# ──────────────────────────────────────────────────────────────
# Reference controller
# ──────────────────────────────────────────────────────────────
KP_POS = 1.5
KP_HEADING = 2.0
STEP_AHEAD_M = 0.25
GOAL_TOL = 0.15


# ──────────────────────────────────────────────────────────────
# Reference helpers
# ──────────────────────────────────────────────────────────────
def point_at_arc_length(poly_xy, arc_lengths, s):
    """Interpolate (x,y) on polyline at arc length s."""
    poly_xy = np.asarray(poly_xy, dtype=float)
    L = arc_lengths[-1]
    s = max(0.0, min(s, L))
    if L <= 1e-12:
        return poly_xy[0].copy()
    for i in range(len(arc_lengths) - 1):
        if s <= arc_lengths[i + 1]:
            t = (s - arc_lengths[i]) / (arc_lengths[i + 1] - arc_lengths[i] + 1e-12)
            return (1 - t) * poly_xy[i] + t * poly_xy[i + 1]
    return poly_xy[-1].copy()


def reference_point_and_heading(ref_xy, x0, y0, step_ahead_m, path_progress):
    """
    Get reference (x_des, y_des, phi_des) from arc-length-based lookahead.
        """
    ref_xy = np.asarray(ref_xy, dtype=float)
    if ref_xy.shape[0] < 2:
        return float(ref_xy[0, 0]), float(ref_xy[0, 1]), 0.0

    arc = path_arc_lengths(ref_xy)
    L = arc[-1]
    s = path_progress[0]

    best_s = s
    best_d2 = float("inf")
    for i in range(ref_xy.shape[0] - 1):
        a, b = ref_xy[i], ref_xy[i + 1]
        ab = b - a
        denom = float(np.dot(ab, ab)) + 1e-12
        t = float(np.dot([x0 - a[0], y0 - a[1]], ab) / denom)
        t = max(0.0, min(1.0, t))
        proj = a + t * ab
        seg_s = arc[i] + t * (arc[i + 1] - arc[i])
        d2 = (x0 - proj[0]) ** 2 + (y0 - proj[1]) ** 2
        if d2 < best_d2:
            best_d2 = d2
            best_s = seg_s

    path_progress[0] = min(best_s + 0.05, L)
    s_ref = min(path_progress[0] + step_ahead_m, L)
    pt = point_at_arc_length(ref_xy, arc, s_ref)
    x_des, y_des = float(pt[0]), float(pt[1])

    # Heading = path tangent at reference point
    idx = 0
    for i in range(len(arc) - 1):
        if arc[i + 1] >= s_ref:
            idx = i
            break
        idx = i
    if idx + 1 < ref_xy.shape[0]:
        a, b = ref_xy[idx], ref_xy[idx + 1]
        phi_des = math.atan2(b[1] - a[1], b[0] - a[0])
    else:
        phi_des = math.atan2(
            ref_xy[-1, 1] - ref_xy[-2, 1],
            ref_xy[-1, 0] - ref_xy[-2, 0],
        )
    return x_des, y_des, phi_des


def reference_controller(x, x_des, y_des, phi_des, dynamics):
    """
    Unicycle reference controller.
    phi_des is accepted for signature match; the current implementation there
    uses only (x_des, y_des).
    """
    dx = x_des - x[0]
    dy = y_des - x[1]
    dist = math.hypot(dx, dy)
    angle_to_target = math.atan2(dy, dx)
    angle_error = wrap_angle(angle_to_target - x[2])
    v_ref = max(dynamics.v_min, KP_POS * dist * math.cos(angle_error))
    omega_ref = KP_HEADING * angle_error
    v_ref = float(np.clip(v_ref, -dynamics.v_max, dynamics.v_max))
    omega_ref = float(np.clip(omega_ref, -dynamics.omega_max, dynamics.omega_max))
    return np.array([v_ref, omega_ref])


# ──────────────────────────────────────────────────────────────
# Run one trajectory
# ──────────────────────────────────────────────────────────────
DT_SAFETY = 0.001


def run_one(controller, gsplat, dynamics, ref_xy, start, goal, dt, max_steps,
            obs_decay=None, sigma_min=0.05, no_obs=False,
            goal_tol=GOAL_TOL, step_ahead_m=STEP_AHEAD_M, verbose=False):
    """
    Simulation loop: integrate unicycle under CBF-QP filter along the reference path.
    """
    # Set trajectory mask for goal-focused EIG
    ref_xy_arr = np.asarray(ref_xy, dtype=float)
    if hasattr(controller, "set_path_mask"):
        controller.set_path_mask(ref_xy_arr, mask_radius=1.5)

    path_progress = [0.0]
    x = np.array([start[0], start[1], start[2]])

    ref_arc = path_arc_lengths(ref_xy_arr)
    total_arc_length = ref_arc[-1] if ref_arc[-1] > 1e-9 else 1.0

    xs, ys, thetas = [x[0]], [x[1]], [x[2]]
    hs_trace, hp_trace, slack_trace, slack2_trace, hp2_trace = [], [], [], [], []
    rho_min_trace, clearance_trace = [], []
    sigma_sum_trace = [float(np.sum(gsplat.sigma))]
    u_ref_trace, u_safe_trace = [], []
    progress_trace = []

    t_start = time.time()
    reached = False
    for step in range(max_steps):
        d_goal = math.hypot(x[0] - goal[0], x[1] - goal[1])
        if d_goal <= goal_tol:
            reached = True
            if verbose:
                print(f"  Goal reached at step {step} (d={d_goal:.3f} m)")
            break

        # Observe BEFORE QP so barrier sees current sigma.
        # Only observe when safety has margin (hs > 0.1) to avoid mutation
        # pushing hs negative.
        obs_kwargs = {'sigma_min': sigma_min}
        if obs_decay is not None:
            obs_kwargs['decay_base'] = obs_decay
        if not no_obs:
            last_hs = getattr(controller, 'last_hs', None)
            if last_hs is None or last_hs > 0.1:
                simulate_observation(x[0], x[1], x[2], gsplat, **obs_kwargs)

        # Reference point on path (with phi_des, matching run_conflict_cbf)
        x_des, y_des, phi_des = reference_point_and_heading(
            ref_xy_arr, x[0], x[1], step_ahead_m, path_progress
        )
        u_ref = reference_controller(x, x_des, y_des, phi_des, dynamics)
        u_qp = controller.solve(x, u_ref)

        # Fine sub-steps for Euler accuracy — no safety filter (QP is sole enforcer)
        n_inner = max(1, int(round(dt / DT_SAFETY)))
        dt_inner = dt / n_inner
        for _ in range(n_inner):
            x = dynamics.integrate(x, u_qp, dt_inner)

        # Record
        xs.append(x[0]); ys.append(x[1]); thetas.append(x[2])
        hs_trace.append(controller.last_hs)
        hp_trace.append(controller.last_hp if controller.last_hp is not None else 0.0)
        slack_trace.append(controller.last_slack if controller.last_slack is not None else 0.0)
        slack2_trace.append(controller.last_slack2 if controller.last_slack2 is not None else 0.0)
        hp2_trace.append(
            controller.last_dI_dtheta_raw
            if hasattr(controller, 'last_dI_dtheta_raw')
               and controller.last_dI_dtheta_raw is not None
            else 0.0)
        u_ref_trace.append(u_ref.copy())
        u_safe_trace.append(u_qp.copy())
        sigma_sum_trace.append(float(np.sum(gsplat.sigma)))
        progress_trace.append(100.0 * min(path_progress[0] / total_arc_length, 1.0))

        # True (non-LSE) safety diagnostic
        _idx = gsplat.query_radius(x[0], x[1], radius=2.0)
        if len(_idx) > 0:
            _d = np.linalg.norm(gsplat.means_xy[_idx] - x[:2], axis=1)
            _r = compute_occupancy_radii_from_uncertainty(
                gsplat.sigma[_idx],
                radius_min=controller.safety.occupancy_radius_min,
                radius_max=controller.safety.occupancy_radius_max,
                robot_radius=controller.safety.occupancy_robot_radius,
                base_radius=controller.safety.occupancy_base_radius,
            )
            rho_min_trace.append(float(np.min(_d - _r)))
            clearance_trace.append(float(np.min(_d)))
        else:
            rho_min_trace.append(10.0)
            clearance_trace.append(10.0)

    wall_time = time.time() - t_start

    return {
        "xs": np.asarray(xs), "ys": np.asarray(ys), "thetas": np.asarray(thetas),
        "hs": np.asarray(hs_trace),
        "hp": np.asarray(hp_trace),
        "hp2": np.asarray(hp2_trace),
        "slack": np.asarray(slack_trace),
        "slack2": np.asarray(slack2_trace),
        "rho_min": np.asarray(rho_min_trace),
        "clearance": np.asarray(clearance_trace),
        "sigma_sum": np.asarray(sigma_sum_trace),
        "u_ref": np.asarray(u_ref_trace),
        "u_safe": np.asarray(u_safe_trace),
        "progress": np.asarray(progress_trace),
        "n_steps": len(xs) - 1,
        "wall_time_s": wall_time,
        "reached": reached,
        "final_d_goal": float(math.hypot(x[0] - goal[0], x[1] - goal[1])),
    }


# ──────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Compare CBF  (h_p) vs  (h_π + h_η)"
    )
    parser.add_argument("file", nargs="?", default="params/yq_large_cloud.mat")
    parser.add_argument("--dt", type=float, default=0.05)
    parser.add_argument("--max-steps", type=int, default=10000)
    parser.add_argument("--interactive", action="store_true",
                        help="Click start and goal on the planning grid (2 clicks).")
    parser.add_argument("--start", nargs=2, type=float, default=None,
                        metavar=("X", "Y"))
    parser.add_argument("--goal", nargs=2, type=float, default=None,
                        metavar=("X", "Y"))
    parser.add_argument("--start-phi", type=float, default=None,
                        help="Start heading (rad). With --interactive: defaults "
                             "to the angle from start→goal.")
    # ── CBF / safety ──
    parser.add_argument("--beta-lse", type=float, default=200.0,
                        help="LSE sharpness β (higher = closer to hard min, "
                             "smaller bias). Default 200 for clean safety plots.")
    parser.add_argument("--adaptive-bias-target", type=float, default=None,
                        help="Adapt β per-call so log(M)/β equals this value. "
                             "Pins LSE bias to a fixed magnitude regardless "
                             "of cluster density. Try 0.005 m.")
    parser.add_argument("--phi-max", type=float, default=45.0,
                        help="Perception barrier cone half-angle φ_max (deg).")
    parser.add_argument("--alpha-p", type=float, default=0.5,
                        help="Class-K coefficient α_p for perception barrier.")
    parser.add_argument("--a-max", type=float, default=0.8,
                        help="Max deceleration (m/s²) for safety class-K "
                             "α_s(h_s) = √(2·a_max·h_s).")
    parser.add_argument("--safety-lookahead", type=float, default=0.0,
                        help="Kinematic lookahead distance L (m). h_s evaluated "
                             "at π + L·[cosθ, sinθ]. Compensates Euler overshoot.")
    parser.add_argument("--safety-horizon-s", type=float, default=0.3,
                        help="Predictive horizon (s) for h_s. Enforce soft-min "
                             "of h_s across N+1 points along forward arc over T s.")
    parser.add_argument("--safety-horizon-n", type=int, default=4,
                        help="Number of prediction points along horizon.")
    parser.add_argument("--freeze-sigma", action="store_true",
                        help="Freeze CBF Rᵢ at initial σ (disable renormalization).")
    # ── Occupancy radii ──
    parser.add_argument("--radius-min", type=float, default=OCCUPANCY_RADIUS_MIN,
                        help="Min occupancy radius Rᵢ_min at σ=0.")
    parser.add_argument("--radius-max", type=float, default=OCCUPANCY_RADIUS_MAX,
                        help="Max occupancy radius Rᵢ_max at σ=1.")
    parser.add_argument("--occupancy-radius-scale", "--occupancy-scale",
                        type=float, default=1.0, dest="occupancy_radius_scale",
                        help="Default scale for both A* and safety radii. "
                             "Overridden by --astar-occupancy-scale / "
                             "--safety-occupancy-scale if given. "
                             "(alias: --occupancy-scale)")
    parser.add_argument("--astar-occupancy-scale", type=float, default=None,
                        help="Scale A* inflation radii by this factor.")
    parser.add_argument("--safety-occupancy-scale", type=float, default=None,
                        help="Scale CBF safety radii by this factor.")
    parser.add_argument("--splat-nearest-k", type=int, default=OCCUPANCY_NEAREST_K,
                        help="Number of nearest splats in the LSE / CBF query.")
    parser.add_argument("--splat-robot-radius", type=float, default=0.0,
                        help="Physical robot radius added to every Rᵢ.")
    parser.add_argument("--splat-base-radius", type=float, default=0.0,
                        help="Extra safety buffer added to every Rᵢ.")
    parser.add_argument("--occupancy-inflate", type=float, default=0.0,
                        help="Uniform extra inflation (m) added to the A* grid.")
    parser.add_argument("--deflate", type=int, default=0,
                        help="Erode the A* grid by N cells after planning "
                             "(gives CBF more room on tight passages).")
    parser.add_argument("--astar-uniform-max-radius", action="store_true",
                        help="Inflate every splat in the A* grid to radius_max. "
                             "More conservative path, but ensures no CBF ball "
                             "can grow to block it.")
    parser.add_argument("--astar-mode", choices=["safe", "plain"], default="safe",
                        help="A* cost mode: 'safe' adds clearance-based penalty.")
    # ── Perception (slack weights, I_c, etc.) ──
    parser.add_argument("--i-c", type=float, default=0.3,
                        help="EIG threshold I_c for .")
    parser.add_argument("--eig-fov-deg", type=float, default=180.0,
                        help="EIG FOV (deg) for 's h_p barrier. "
                             "Wider than camera FOV (--fov-deg) to capture "
                             "info along trajectory direction, not just ahead. "
                             "60=same as camera, 120=wider, 180=forward hemisphere "
                             "(default), 360=omnidirectional.")
    parser.add_argument("--heading-smooth", type=float, default=0.85,
                        help="EMA smoothing for  heading gradient. "
                             "0=raw (chatters), 0.85=smooth (default), "
                             "0.95=strong commitment.")
    parser.add_argument("--w-track", type=float, default=1.0,
                        help="Tracking weight for  (ConflictAwareCBFController). "
                             "Lower = robot cares less about following u_ref → "
                             "deviates more for perception. Try 0.1–0.5.")
    parser.add_argument("--k-slack", type=float, default=0.3,
                        help="Primary perception slack weight (w_p for , "
                             "w_s for ).")
    parser.add_argument("--w2-slack", type=float, default=0.1,
                        help="Secondary slack weight for  (δ_η).")
    parser.add_argument("--q-slack", type=int, default=2, choices=[1, 2],
                        help="Slack norm: 1=L1 (sparse), 2=L2 (smooth). Both "
                             "formulations use the same q for fair comparison.")
    parser.add_argument("--adaptive-penalty", action="store_true",
                        help="Use MapStatePenalty to scale slacks based on "
                             "goal distance and scene uncertainty.")
    # ── EIG / FOV ──
    parser.add_argument("--fov-deg", type=float, default=40.0,
                        help="Camera FOV half-width for EIG computation (deg).")
    parser.add_argument("--fov-sigmoid-k", type=float, default=30.0,
                        help="Sharpness of FOV sigmoid boundary for EIG.")
    # ── Observation / uncertainty mutation ──
    parser.add_argument("--obs-decay", type=float, default=0.003,
                        help="Observation decay rate (λ base).")
    parser.add_argument("--sigma-min", type=float, default=0.05,
                        help="σ floor (sensor noise). Monotone update prevents "
                             "σ from ever increasing.")
    parser.add_argument("--no-obs", action="store_true",
                        help="Disable simulate_observation — σ never changes.")
    # ── Scene / grid ──
    parser.add_argument("--z-thickness", type=float, default=0.5,
                        help="Z-band thickness (m), centered at GROUND_Z+ROBOT_HEIGHT/2. "
                             "Default 0.5.")
    parser.add_argument("--grid-res", type=float, default=0.05,
                        help="A* grid resolution (m).")
    parser.add_argument("--save-gif", action="store_true",
                        help="Save an animated GIF of the trajectory rollout "
                             "(compare_formulations.gif).")
    parser.add_argument("--gif-fps", type=int, default=20,
                        help="Frames per second for the GIF (default 20).")
    parser.add_argument("--gif-skip", type=int, default=None,
                        help="Only render every N-th simulation step in the GIF. "
                             "Default: auto-chosen so total frames ≈ 200.")
    parser.add_argument("--formulations", nargs="+", default=["eq32", "eq2"],
                        choices=["eq32", "eq2"],
                        help="Which formulations to run (default: both). "
                             "eq32 =  (h_p), eq2 =  (h_π + h_η).")
    parser.add_argument("--k-slack-sweep", nargs="+", type=float, default=None,
                        metavar="W",
                        help="Run the selected formulation(s) once per slack weight. "
                             "Overrides --k-slack. E.g. --k-slack-sweep 0.3 0.7")
    args = parser.parse_args()

    # ── Load scene (z-slice centered on robot height) ──
    print(f"Loading {args.file}")
    z_center = GROUND_Z + ROBOT_HEIGHT / 2.0
    z_band_min = z_center - args.z_thickness / 2.0
    z_band_max = z_center + args.z_thickness / 2.0
    print(f"z-slice: [{z_band_min:.3f}, {z_band_max:.3f}] m  "
          f"(center {z_center:.3f}, thickness {args.z_thickness:.3f})")
    gsplat_all = GaussianSplatField.from_mat(args.file,
                                              z_band=(z_band_min, z_band_max))
    print(f"Scene: {gsplat_all.N} splats, mean σ = {np.mean(gsplat_all.sigma):.4f}")

    # ── Build A* grid once ──
    # Thread FOV / sigmoid config into risk_aware_eig module
    import risk_aware_eig as _eig_mod
    _eig_mod.FOV_DEG = args.fov_deg
    _eig_mod.FOV_SIGMOID_K = args.fov_sigmoid_k
    _eig_mod.SIGMA_MIN = args.sigma_min

    # Separate A* vs safety inflation scales (falls back to --occupancy-radius-scale)
    astar_scale  = (args.occupancy_radius_scale if args.astar_occupancy_scale  is None
                    else args.astar_occupancy_scale)
    safety_scale = (args.occupancy_radius_scale if args.safety_occupancy_scale is None
                    else args.safety_occupancy_scale)
    astar_r_min  = args.radius_min * astar_scale
    astar_r_max  = args.radius_max * astar_scale
    safety_r_min = args.radius_min * safety_scale
    safety_r_max = args.radius_max * safety_scale

    grid_rmin = astar_r_max if args.astar_uniform_max_radius else astar_r_min
    grid, _score, x_min, y_min, res = build_uncertainty_occupancy_grid(
        gsplat_all.means_xy, gsplat_all.sigma, args.grid_res,
        radius_min=grid_rmin,
        radius_max=astar_r_max,
    )
    # Optional extra uniform inflation on the A* grid (matches run_conflict_cbf)
    if args.occupancy_inflate > 0.0:
        inflate_cells = max(0, int(math.ceil(args.occupancy_inflate / args.grid_res)))
        if inflate_cells > 0:
            from scipy.ndimage import binary_dilation
            struct = np.ones((2 * inflate_cells + 1, 2 * inflate_cells + 1), dtype=bool)
            grid = binary_dilation(grid, structure=struct).astype(bool)
    # Optional erosion (gives CBF more room)
    if args.deflate > 0:
        from scipy.ndimage import binary_erosion
        struct = np.ones((2 * args.deflate + 1, 2 * args.deflate + 1), dtype=bool)
        grid = binary_erosion(grid, structure=struct).astype(bool)
        print(f"Deflated A* grid by {args.deflate} cells "
              f"({args.deflate * args.grid_res:.3f} m)")

    nx, ny = grid.shape
    # Safe mode (clearance penalty) is default; 'plain' skips it
    clearance_map = build_clearance_map(grid, res) if args.astar_mode == "safe" else None
    print(f"Grid: {nx}×{ny} cells @ {res:.3f} m, free cells = {(~grid).sum()} "
          f"({100*(~grid).sum()/grid.size:.1f}%)")

    # ── Pick start/goal: interactive or from CLI ──
    if args.interactive:
        from astar_safe_ref import slice_at_z_range, load_point_cloud
        means3D, rgb = load_point_cloud(args.file)
        xy_disp, rgb_disp = slice_at_z_range(
            means3D, rgb, z_band_min, z_band_max)
        fig_pick, ax_pick = plt.subplots(figsize=(10, 8))
        ax_pick.imshow(grid.T,
                       extent=[x_min, x_min + nx * res, y_min, y_min + ny * res],
                       origin="lower", cmap="Greys", alpha=0.4, aspect="equal")
        sub = max(1, xy_disp.shape[0] // 80000)
        ax_pick.scatter(xy_disp[::sub, 0], xy_disp[::sub, 1],
                        c=rgb_disp[::sub], s=0.5, alpha=0.8)
        ax_pick.set_title("Click START point, then GOAL point (2 clicks)")
        print("Click START point on the image, then click GOAL point...")
        pts = plt.ginput(2, timeout=0)
        plt.close(fig_pick)
        if len(pts) < 2:
            raise RuntimeError("Need 2 clicks (start and goal). Aborting.")
        start_xy = np.array(pts[0])
        goal_xy = np.array(pts[1])
        if args.start_phi is None:
            start_phi = math.atan2(goal_xy[1] - start_xy[1],
                                   goal_xy[0] - start_xy[0])
        else:
            start_phi = args.start_phi
        print(f"Selected start=({start_xy[0]:.2f}, {start_xy[1]:.2f}), "
              f"goal=({goal_xy[0]:.2f}, {goal_xy[1]:.2f}), "
              f"phi={math.degrees(start_phi):.1f}°")
    else:
        if args.start is None or args.goal is None:
            raise RuntimeError("Provide --start X Y --goal X Y, or use --interactive")
        start_xy = np.array(args.start)
        goal_xy = np.array(args.goal)
        start_phi = args.start_phi if args.start_phi is not None else 0.0

    # ── Snap start/goal to nearest free cell if blocked ──
    def snap_to_free(xy_world):
        ix = int((xy_world[0] - x_min) / res)
        iy = int((xy_world[1] - y_min) / res)
        if 0 <= ix < nx and 0 <= iy < ny and not grid[ix, iy]:
            return ix, iy
        # Search outward in expanding rings for a free cell
        for r in range(1, max(nx, ny)):
            for dx in range(-r, r + 1):
                for dy in (-r, r):
                    nix, niy = ix + dx, iy + dy
                    if 0 <= nix < nx and 0 <= niy < ny and not grid[nix, niy]:
                        return nix, niy
                for dx in (-r, r):
                    nix, niy = ix + dx, iy + dy
                    if 0 <= nix < nx and 0 <= niy < ny and not grid[nix, niy]:
                        return nix, niy
        raise RuntimeError(f"No free cell found near {xy_world}")

    start_ix, start_iy = snap_to_free(start_xy)
    goal_ix, goal_iy = snap_to_free(goal_xy)
    snapped_start = np.array([x_min + (start_ix + 0.5) * res,
                               y_min + (start_iy + 0.5) * res])
    snapped_goal = np.array([x_min + (goal_ix + 0.5) * res,
                              y_min + (goal_iy + 0.5) * res])
    if not np.allclose(snapped_start, start_xy, atol=res):
        print(f"  start snapped {start_xy} → {snapped_start} (was in occupied cell)")
        start_xy = snapped_start
    if not np.allclose(snapped_goal, goal_xy, atol=res):
        print(f"  goal snapped {goal_xy} → {snapped_goal} (was in occupied cell)")
        goal_xy = snapped_goal

    start_phi_idx = phi_to_index(start_phi, N_HEADINGS)
    path_cells = astar(grid, start_ix, start_iy, start_phi_idx,
                       goal_ix, goal_iy,
                       N_HEADINGS, clearance_map=clearance_map)
    if path_cells is None:
        raise RuntimeError(
            f"A* failed from cell ({start_ix},{start_iy}) to ({goal_ix},{goal_iy}). "
            f"Try smaller --occupancy-scale or different start/goal.")
    path_xw = [x_min + (c[0] + 0.5) * res for c in path_cells]
    path_yw = [y_min + (c[1] + 0.5) * res for c in path_cells]
    path_zw = [0.0] * len(path_xw)
    path_xw, path_yw, _ = smooth_path_spline(path_xw, path_yw, path_zw)
    path_xw = np.asarray(path_xw)
    path_yw = np.asarray(path_yw)
    ref_xy = np.column_stack([path_xw, path_yw])
    print(f"A* path: {len(ref_xy)} waypoints, length {trajectory_length(path_xw, path_yw):.2f} m")

    # ── Run each formulation ──
    dynamics = UnicycleDynamics()
    start_state = (start_xy[0], start_xy[1], start_phi)

    # Controller kwargs shared by both formulations
    goal_xy_arr = np.array([goal_xy[0], goal_xy[1]]) if args.adaptive_penalty else None
    common_kwargs = dict(
        dynamics=dynamics,
        a_max=args.a_max,
        beta_lse=args.beta_lse,
        adaptive_bias_target=args.adaptive_bias_target,
        q_slack=args.q_slack,
        occupancy_radius_min=safety_r_min,
        occupancy_radius_max=safety_r_max,
        occupancy_nearest_k=args.splat_nearest_k,
        occupancy_robot_radius=args.splat_robot_radius,
        occupancy_base_radius=args.splat_base_radius,
        freeze_sigma=args.freeze_sigma,
        safety_lookahead_m=args.safety_lookahead,
        safety_horizon_s=args.safety_horizon_s,
        safety_horizon_n=args.safety_horizon_n,
        goal_xy=goal_xy_arr,
    )

    results = {}

    # ── Build the list of (run_key, controller_factory) pairs ──
    slack_weights = args.k_slack_sweep if args.k_slack_sweep is not None else [args.k_slack]
    run_list = []
    for w in slack_weights:
        if "eq32" in args.formulations:
            tag = f"eq32_w{w}" if len(slack_weights) > 1 else "eq32"
            run_list.append((tag, "eq32", w,
                lambda gs, _w=w: Equation32Controller(
                    gsplat=gs, I_c=args.i_c, w_p_slack=_w,
                    w_track=args.w_track,
                    alpha_p=args.alpha_p,
                    eig_fov_deg=args.eig_fov_deg,
                    heading_smooth=args.heading_smooth,
                    **common_kwargs)))
        if "eq2" in args.formulations:
            tag = f"eq2_w{w}" if len(slack_weights) > 1 else "eq2"
            run_list.append((tag, "eq2", w,
                lambda gs, _w=w: ConflictAwareCBFController(
                    gsplat=gs, alpha_p=args.alpha_p,
                    phi_max_deg=args.phi_max,
                    k_slack=_w, w2_slack=args.w2_slack,
                    w_track=args.w_track,
                    **common_kwargs)))

    for name, base, w, make_ctrl in run_list:
        print(f"\n=== Running {name} ({base}, k_slack={w}) ===")
        # Fresh copy of the scene for a fair comparison (same initial σ)
        gs = GaussianSplatField(gsplat_all.means3D.copy(),
                                 gsplat_all.sigma.copy())
        ctrl = make_ctrl(gs)
        res_dict = run_one(ctrl, gs, dynamics, ref_xy, start_state, goal_xy,
                           args.dt, args.max_steps,
                           obs_decay=args.obs_decay, sigma_min=args.sigma_min,
                           no_obs=args.no_obs, verbose=True)
        res_dict["name"] = name
        res_dict["gs"] = gs
        res_dict["sigma_init_sum"] = float(np.sum(gsplat_all.sigma))
        res_dict["sigma_final_sum"] = float(np.sum(gs.sigma))
        results[name] = res_dict

        v, a, j, s = compute_motion_derivatives(res_dict["xs"], res_dict["ys"],
                                                 args.dt)
        res_dict["v"] = v; res_dict["a"] = a; res_dict["j"] = j; res_dict["s"] = s

        print(f"  steps={res_dict['n_steps']}, reached={res_dict['reached']}, "
              f"d_goal={res_dict['final_d_goal']:.3f} m, "
              f"wall={res_dict['wall_time_s']:.1f}s")

    # ── Summary table (works for any number of runs) ──
    names = list(results.keys())
    col_w = max(20, max(len(n) + 2 for n in names))
    header = f"{'Metric':<38}" + "".join(f"{n:>{col_w}}" for n in names)
    sep = "=" * len(header)
    print(f"\n{sep}\n{header}\n{sep}")

    def row(label, fmt, key):
        vals = "".join(f"{fmt.format(results[n][key]):>{col_w}}" for n in names)
        print(f"{label:<38}{vals}")

    def row_stat(label, key, stat):
        vals = "".join(f"{getattr(np, stat)(results[n][key]):>{col_w}.4f}" for n in names)
        print(f"{label:<38}{vals}")

    row("Number of steps",         "{:d}",    "n_steps")
    row("Goal reached",            "{}",      "reached")
    row("Final d_goal (m)",        "{:.3f}",  "final_d_goal")
    row("Wall time (s)",           "{:.2f}",  "wall_time_s")
    print("-" * len(header))
    print("MOTION SMOOTHNESS")
    row_stat("  mean |v|  (m/s)",             "v", "mean")
    row_stat("  mean |a|  (m/s²)",            "a", "mean")
    row_stat("  mean |j|  (m/s³)",            "j", "mean")
    row_stat("  mean |s|  (m/s⁴)",            "s", "mean")
    row_stat("  max  |v|  (m/s)",             "v", "max")
    row_stat("  max  |a|  (m/s²)",            "a", "max")
    row_stat("  max  |j|  (m/s³)",            "j", "max")
    row_stat("  max  |s|  (m/s⁴)",            "s", "max")
    print("-" * len(header))
    print("TRAJECTORY")
    traj_vals = "".join(
        f"{trajectory_length(results[n]['xs'], results[n]['ys']):>{col_w}.3f}"
        for n in names)
    print(f"{'  Path length (m)':<38}{traj_vals}")
    ref_len = trajectory_length(path_xw, path_yw)
    ref_vals = "".join(f"{ref_len:>{col_w}.3f}" for _ in names)
    print(f"{'  A* reference length (m)':<38}{ref_vals}")
    print("-" * len(header))
    print("UNCERTAINTY REDUCTION")
    sig_init = list(results.values())[0]["sigma_init_sum"]
    sig_init_vals = "".join(f"{sig_init:>{col_w}.3f}" for _ in names)
    print(f"{'  Σ σ initial':<38}{sig_init_vals}")
    sig_final_vals = "".join(
        f"{results[n]['sigma_final_sum']:>{col_w}.3f}" for n in names)
    print(f"{'  Σ σ final':<38}{sig_final_vals}")
    dsig_vals = "".join(
        f"{sig_init - results[n]['sigma_final_sum']:>{col_w}.3f}" for n in names)
    print(f"{'  Σ σ reduction (bigger=better)':<38}{dsig_vals}")
    pct_vals = "".join(
        f"{100*(sig_init - results[n]['sigma_final_sum'])/max(sig_init,1e-9):>{col_w-1}.2f}%"
        for n in names)
    print(f"{'  % reduction':<38}{pct_vals}")
    print("-" * len(header))
    print("SAFETY (h_s and true ρ_i)")
    row_stat("  h_s min (LSE)",                "hs", "min")
    row_stat("  h_s mean",                     "hs", "mean")
    row_stat("  true min ρ_i  (min)",          "rho_min", "min")
    row_stat("  clearance     (min, m)",       "clearance", "min")
    print(sep)

    # ── Plots ──
    make_plots(results, ref_xy, args.dt, path_xw, path_yw, start_xy, goal_xy,
               grid=grid, x_min=x_min, y_min=y_min, res=res, gsplat=gsplat_all,
               z_thickness=args.z_thickness, mat_file=args.file)
    if args.save_gif:
        make_gif(results, ref_xy, path_xw, path_yw, start_xy, goal_xy,
                 grid=grid, x_min=x_min, y_min=y_min, res=res, gsplat=gsplat_all,
                 z_thickness=args.z_thickness, mat_file=args.file,
                 fps=args.gif_fps, skip=args.gif_skip, dt=args.dt)
        print("Saved: compare_formulations.gif")

    print("\nSaved: compare_formulations.png and compare_formulations_metrics.png")


# ──────────────────────────────────────────────────────────────
# Animated GIF
# ──────────────────────────────────────────────────────────────
def _run_colors_labels(results):
    """Generate color and label dicts for arbitrary run keys."""
    palette = ["#1f77b4", "#d62728", "#2ca02c", "#ff7f0e", "#9467bd",
               "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]
    colors, labels = {}, {}
    for i, name in enumerate(results):
        colors[name] = palette[i % len(palette)]
        if name == "eq32":
            labels[name] = r"  ($h_p$)"
        elif name == "eq2":
            labels[name] = r"  ($h_\pi + h_\eta$)"
        elif name.startswith("eq32_w"):
            w = name.split("eq32_w")[1]
            labels[name] = rf"  ($h_p$, $w_p$={w})"
        elif name.startswith("eq2_w"):
            w = name.split("eq2_w")[1]
            labels[name] = rf"  ($h_\pi + h_\eta$, $w_s$={w})"
        else:
            labels[name] = name
    return colors, labels


def make_gif(results, ref_xy, path_xw, path_yw, start_xy, goal_xy,
             grid=None, x_min=None, y_min=None, res=None, gsplat=None,
             z_thickness=0.5, mat_file=None, fps=20, skip=None, dt=0.05):
    """Render an animated GIF showing trajectories growing over time."""
    import matplotlib.animation as animation

    colors, labels = _run_colors_labels(results)

    # Determine the longest trajectory (in steps)
    max_len = max(len(r["xs"]) for r in results.values())

    # Auto-choose skip so we get roughly 200 frames
    if skip is None:
        skip = max(1, max_len // 200)

    frame_indices = list(range(0, max_len, skip))
    if frame_indices[-1] != max_len - 1:
        frame_indices.append(max_len - 1)

    # ── Set up the static background ──
    fig, ax = plt.subplots(figsize=(14, 10))

    if grid is not None:
        nx, ny = grid.shape
        ax.imshow(grid.T,
                  extent=[x_min, x_min + nx * res, y_min, y_min + ny * res],
                  origin="lower", cmap="Greys", alpha=0.25, aspect="equal")

    if mat_file is not None:
        try:
            from astar_safe_ref import (
                load_point_cloud, slice_at_z_range, GROUND_Z, ROBOT_HEIGHT)
            means3D, rgb = load_point_cloud(mat_file)
            z_center = GROUND_Z + ROBOT_HEIGHT / 2.0
            xy_disp, rgb_disp = slice_at_z_range(
                means3D, rgb,
                z_center - z_thickness / 2.0,
                z_center + z_thickness / 2.0)
            sub = max(1, xy_disp.shape[0] // 60000)
            ax.scatter(xy_disp[::sub, 0], xy_disp[::sub, 1],
                       c=rgb_disp[::sub], s=0.4, alpha=0.7, zorder=1)
        except Exception:
            pass

    ax.plot(path_xw, path_yw, "k--", lw=1.5, alpha=0.6,
            label="A* reference", zorder=8)
    ax.scatter([start_xy[0]], [start_xy[1]], c="g", s=120, zorder=11,
               edgecolors="white", linewidths=1.5, label="start")
    ax.scatter([goal_xy[0]], [goal_xy[1]], c="purple", marker="*", s=200,
               zorder=11, edgecolors="white", linewidths=1, label="goal")
    ax.set_aspect("equal")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title("CBF Formulation Comparison — Trajectories")
    ax.grid(True, alpha=0.3)

    # Dynamic artists: trail line + current-position marker per formulation
    artists = {}
    for name in results:
        line, = ax.plot([], [], color=colors[name], lw=2.5,
                        label=labels[name], zorder=10)
        dot, = ax.plot([], [], "o", color=colors[name], ms=8,
                       markeredgecolor="white", markeredgewidth=1.2, zorder=12)
        artists[name] = (line, dot)

    ax.legend(fontsize=11, loc="best")
    time_text = ax.text(0.02, 0.97, "", transform=ax.transAxes,
                        fontsize=11, verticalalignment="top",
                        bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))

    def update(frame_idx):
        i = frame_indices[frame_idx]
        for name, r in results.items():
            n = min(i + 1, len(r["xs"]))
            line, dot = artists[name]
            line.set_data(r["xs"][:n], r["ys"][:n])
            dot.set_data([r["xs"][n - 1]], [r["ys"][n - 1]])
        time_text.set_text(f"t = {i * dt:.2f} s")
        return [a for pair in artists.values() for a in pair] + [time_text]

    anim = animation.FuncAnimation(
        fig, update, frames=len(frame_indices), blit=True, interval=1000 // fps)
    anim.save("compare_formulations.gif", writer="pillow", fps=fps)
    plt.close(fig)


# ──────────────────────────────────────────────────────────────
# Plotting (standard)
# ──────────────────────────────────────────────────────────────
def make_plots(results, ref_xy, dt, path_xw, path_yw, start_xy, goal_xy,
               grid=None, x_min=None, y_min=None, res=None, gsplat=None,
               z_thickness=0.5, mat_file=None):
    """
    Two figures, standard:
      (1) compare_formulations.png — trajectory overlay on scene background
      (2) compare_formulations_metrics.png — multi-panel metrics
    """
    colors, labels = _run_colors_labels(results)

    # ─── Figure 1: Trajectory overlay (standard) ───
    fig1, ax1 = plt.subplots(figsize=(14, 10))

    # Background: occupancy grid + RGB scatter of the scene
    if grid is not None:
        nx, ny = grid.shape
        ax1.imshow(grid.T,
                   extent=[x_min, x_min + nx * res, y_min, y_min + ny * res],
                   origin="lower", cmap="Greys", alpha=0.25, aspect="equal")
    if mat_file is not None:
        try:
            from astar_safe_ref import (
                load_point_cloud, slice_at_z_range, GROUND_Z, ROBOT_HEIGHT)
            means3D, rgb = load_point_cloud(mat_file)
            # Same centered z-band
            z_center = GROUND_Z + ROBOT_HEIGHT / 2.0
            xy_disp, rgb_disp = slice_at_z_range(
                means3D, rgb,
                z_center - z_thickness / 2.0,
                z_center + z_thickness / 2.0)
            sub = max(1, xy_disp.shape[0] // 60000)
            ax1.scatter(xy_disp[::sub, 0], xy_disp[::sub, 1],
                        c=rgb_disp[::sub], s=0.4, alpha=0.7, zorder=1)
        except Exception:
            pass

    ax1.plot(path_xw, path_yw, "k--", lw=1.5, alpha=0.6,
             label="A* reference", zorder=8)
    for name, r in results.items():
        ax1.plot(r["xs"], r["ys"], color=colors[name], lw=2.5,
                 label=labels[name], zorder=10)
    ax1.scatter([start_xy[0]], [start_xy[1]], c="g", s=120, zorder=11,
                edgecolors="white", linewidths=1.5, label="start")
    ax1.scatter([goal_xy[0]], [goal_xy[1]], c="purple", marker="*", s=200,
                zorder=11, edgecolors="white", linewidths=1, label="goal")
    ax1.set_aspect("equal")
    ax1.set_xlabel("x (m)"); ax1.set_ylabel("y (m)")
    ax1.set_title("CBF Formulation Comparison — Trajectories")
    ax1.legend(fontsize=11, loc="best")
    ax1.grid(True, alpha=0.3)
    plt.savefig("compare_formulations.png", dpi=130, bbox_inches="tight")
    plt.close(fig1)

    # ─── Figure 2: Multi-panel metrics (x-axis = progress along trajectory %) ───
    # Compute per-run progress arrays from cumulative arc length
    progress = {}
    for name, r in results.items():
        xy = np.column_stack([r["xs"], r["ys"]])
        seg = np.linalg.norm(np.diff(xy, axis=0), axis=1)
        arc_full = np.concatenate([[0.0], np.cumsum(seg)])
        total = arc_full[-1] if arc_full[-1] > 1e-9 else 1.0
        prog_full = arc_full / total * 100.0   # length n_steps+1
        progress[name] = {
            "step": prog_full[1:],                                # n_steps
            "step_full": prog_full,                               # n_steps+1
            "v": (prog_full[:-1] + prog_full[1:]) / 2,           # n_steps
            "a": ((prog_full[:-1] + prog_full[1:]) / 2)[:-1],    # n_steps-1
            "j": ((prog_full[:-1] + prog_full[1:]) / 2)[:-2],    # n_steps-2
            "s": ((prog_full[:-1] + prog_full[1:]) / 2)[:-3],    # n_steps-3
        }

    XLABEL = "Progress along trajectory (%)"

    fig2 = plt.figure(figsize=(16, 14))
    gs = fig2.add_gridspec(4, 2, hspace=0.45, wspace=0.25)

    # (A) Safety: h_s and true min ρ_i
    ax = fig2.add_subplot(gs[0, 0])
    for name, r in results.items():
        p = progress[name]["step"]
        ax.plot(p, r["hs"], color=colors[name], lw=1.4,
                label=f"{name} h_s (LSE)")
        ax.plot(p, r["rho_min"], color=colors[name], lw=0.9, ls="--", alpha=0.6,
                label=f"{name} true min ρ")
    ax.axhline(0, color="k", ls=":", lw=0.8)
    ax.set_xlabel(XLABEL); ax.set_ylabel("Safety")
    ax.set_title("h_s (LSE) vs true min ρ_i")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    # (B) Perception
    ax = fig2.add_subplot(gs[0, 1])
    for name, r in results.items():
        p = progress[name]["step"]
        lab = (r"$h_p = I - I_c$" if name.startswith("eq32") else r"$h_\pi$")
        ax.plot(p, r["hp"], color=colors[name], lw=1.4, label=f"{name}: {lab}")
    ax.axhline(0, color="k", ls=":", lw=0.8)
    ax.set_xlabel(XLABEL); ax.set_ylabel("Perception barrier")
    ax.set_title("Primary perception barrier")
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

    # (C) Total scene uncertainty
    ax = fig2.add_subplot(gs[1, 0])
    for name, r in results.items():
        p = progress[name]["step_full"]
        ax.plot(p, r["sigma_sum"], color=colors[name], lw=1.5, label=labels[name])
    ax.set_xlabel(XLABEL); ax.set_ylabel(r"$\sum_i \sigma_i$")
    ax.set_title("Scene uncertainty (lower = more learned)")
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

    # (D) Slack variables
    ax = fig2.add_subplot(gs[1, 1])
    for name, r in results.items():
        p = progress[name]["step"]
        ax.plot(p, r["slack"], color=colors[name], lw=1.3,
                label=f"{name} δ" + (" (δ_s)" if name.startswith("eq2") else ""))
        if name.startswith("eq2") and np.max(r["slack2"]) > 1e-6:
            ax.plot(p, r["slack2"], color=colors[name], lw=1.0, ls="--", alpha=0.6,
                    label=f"{name} δ_η")
    ax.set_xlabel(XLABEL); ax.set_ylabel("Slack")
    ax.set_title("Slack variables")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    # (E)–(H) Motion smoothness: v, a, j, s
    # Savitzky-Golay smoothing — heavier for higher derivatives (noisier).
    from scipy.signal import savgol_filter
    smooth_windows = {"v": 11, "a": 21, "j": 41, "s": 61}
    for row_idx, (key, ylabel, title) in enumerate([
        ("v", "|v|  (m/s)",   "Speed"),
        ("a", "|a|  (m/s²)",  "Acceleration"),
        ("j", "|j|  (m/s³)",  "Jerk"),
        ("s", "|s|  (m/s⁴)",  "Snap"),
    ]):
        ax = fig2.add_subplot(gs[2 + row_idx // 2, row_idx % 2])
        for name, r in results.items():
            arr = r[key]
            p = progress[name][key]
            if len(arr) < 5:
                ax.plot(p, arr, color=colors[name], lw=1.2, label=labels[name])
                continue
            # Raw trace (faint)
            ax.plot(p, arr, color=colors[name], lw=0.3, alpha=0.25)
            # Smoothed trace (bold)
            win = min(smooth_windows[key], len(arr)) | 1  # must be odd
            poly = min(3, win - 1)
            arr_smooth = savgol_filter(arr, window_length=win, polyorder=poly)
            arr_smooth = np.maximum(arr_smooth, 0.0)  # magnitudes can't be negative
            ax.plot(p, arr_smooth, color=colors[name], lw=1.8, label=labels[name])
        ax.set_xlabel(XLABEL); ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    run_names = "  vs  ".join(labels[n] for n in results)
    fig2.suptitle(
        f"CBF Formulation Comparison — Metrics:  {run_names}",
        fontsize=13, fontweight="bold")
    plt.savefig("compare_formulations_metrics.png", dpi=130, bbox_inches="tight")
    plt.close(fig2)


if __name__ == "__main__":
    main()
