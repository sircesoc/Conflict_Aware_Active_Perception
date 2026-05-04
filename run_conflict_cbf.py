"""
Run Safe A* to get a reference path, then follow it using the conflict-aware CBF
controller from the .

Pipeline:
  1. A* planning (astar_safe_ref): grid, clearance, smoothing, waypoints -> reference path
  2. Reference controller: unicycle tracking of path + heading alignment
  3. Conflict-aware CBF-QP filter: safety (hard) + perception (soft) barriers
  4. Forward Euler integration of unicycle dynamics
  5. Simulated observation: reduce uncertainty for Gaussians in FOV each timestep

Usage:
  python run_conflict_cbf.py [params/yq_large_cloud.mat]
  python run_conflict_cbf.py params/yq_large_cloud.mat --dt 0.05 --phi-max 60
"""

import argparse
import math
import numpy as np

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
    ROBOT_RADIUS,
    N_HEADINGS,
    SMOOTH_PATH,
    SMOOTH_NUM_POINTS,
)
def path_arc_lengths(poly_xy):
    """Cumulative arc lengths at each vertex. First is 0."""
    poly_xy = np.asarray(poly_xy, dtype=float)
    if len(poly_xy) < 2:
        return np.array([0.0])
    seg_lens = np.sqrt(np.sum(np.diff(poly_xy, axis=0) ** 2, axis=1))
    return np.concatenate([[0.0], np.cumsum(seg_lens)])

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


def build_uncertainty_display_grid_on_bounds(
    xy,
    uncertainty,
    x_min,
    y_min,
    res,
    nx,
    ny,
    *,
    radius_min,
    radius_max,
    robot_radius=0.0,
    base_radius=0.0,
):
    """Rasterize occupancy circles on fixed planner bounds for trajectory display."""
    xy = np.asarray(xy, dtype=np.float64).reshape(-1, 2)
    if xy.shape[0] == 0:
        return np.zeros((nx, ny), dtype=bool)

    radii = compute_occupancy_radii_from_uncertainty(
        uncertainty,
        radius_min=radius_min,
        radius_max=radius_max,
        robot_radius=robot_radius,
        base_radius=base_radius,
    )
    grid = np.zeros((ny, nx), dtype=bool)
    x_centers = x_min + (np.arange(nx, dtype=np.float64) + 0.5) * res
    y_centers = y_min + (np.arange(ny, dtype=np.float64) + 0.5) * res

    for (px, py), radius in zip(xy, radii):
        ix0 = max(0, int(np.floor((px - radius - x_min) / res)))
        ix1 = min(nx - 1, int(np.floor((px + radius - x_min) / res)))
        iy0 = max(0, int(np.floor((py - radius - y_min) / res)))
        iy1 = min(ny - 1, int(np.floor((py + radius - y_min) / res)))
        if ix0 > ix1 or iy0 > iy1:
            continue

        xs = x_centers[ix0:ix1 + 1] - px
        ys = y_centers[iy0:iy1 + 1] - py
        mask = (ys[:, None] ** 2 + xs[None, :] ** 2) <= float(radius * radius)
        sub = grid[iy0:iy1 + 1, ix0:ix1 + 1]
        np.logical_or(sub, mask, out=sub)

    return grid.T.copy()


def hp_segment_palette(q=None):
    """Trajectory segment colors for h_p <= 0 vs h_p > 0."""
    if q == 1:
        return {"negative": "tab:blue", "positive": "mediumseagreen"}
    if q == 2:
        return {"negative": "#4B0082", "positive": "goldenrod"}
    return {"negative": "tab:orange", "positive": "mediumseagreen"}


def add_hp_colored_trajectory(
    ax,
    xs,
    ys,
    hp_values,
    *,
    positive_color,
    negative_color,
    lw,
    alpha,
    label_prefix,
    zorder=4,
):
    """Draw a trajectory with segment colors switching on the sign of h_p."""
    from matplotlib.collections import LineCollection
    from matplotlib.lines import Line2D

    points = np.column_stack([np.asarray(xs, dtype=np.float64), np.asarray(ys, dtype=np.float64)])
    hp = np.asarray(hp_values, dtype=np.float64).reshape(-1)
    segment_count = min(max(points.shape[0] - 1, 0), hp.size)
    if segment_count <= 0:
        return []

    segments = np.stack([points[:segment_count], points[1:segment_count + 1]], axis=1)
    positive = hp[:segment_count] > 0.0
    handles = []

    if np.any(~positive):
        ax.add_collection(
            LineCollection(
                segments[~positive],
                colors=negative_color,
                linewidths=lw,
                alpha=alpha,
                zorder=zorder,
            )
        )
        handles.append(Line2D([0], [0], color=negative_color, lw=lw, label=f"{label_prefix} ($h_p\\leq0$)"))

    if np.any(positive):
        ax.add_collection(
            LineCollection(
                segments[positive],
                colors=positive_color,
                linewidths=lw,
                alpha=alpha,
                zorder=zorder + 0.1,
            )
        )
        handles.append(Line2D([0], [0], color=positive_color, lw=lw, label=f"{label_prefix} ($h_p>0$)"))

    return handles


from unicycle_dynamics import UnicycleDynamics, wrap_angle
from risk_aware_eig import GaussianSplatField, eig_at_pose, simulate_observation
from conflict_cbf import (
    ConflictAwareCBFController,
    Equation2Controller,
    Equation32Controller,
    OCCUPANCY_RADIUS_MIN,
    OCCUPANCY_RADIUS_MAX,
    OCCUPANCY_NEAREST_K,
    compute_occupancy_radii_from_uncertainty,
)

# ──────────────────────────────────────────────────────────────
# Simulation parameters
# ──────────────────────────────────────────────────────────────
DT = 0.01               # integration timestep (s) — small for continuous-time CBF guarantee — 20 Hz (QP rate)
DT_SAFETY = 0.001       # safety inner loop timestep (s) — 1 kHz
STEP_AHEAD_M = 0.4      # look-ahead distance on path for reference
MAX_STEPS = 100000        # maximum simulation steps
GOAL_TOL = 0.05         # goal reached tolerance (m)

# Reference controller gains
KP_POS = 1.5
KP_HEADING = 1.0


# ──────────────────────────────────────────────────────────────
# Reference tracking
# ──────────────────────────────────────────────────────────────
def reference_point_and_heading(ref_xy, x0, y0, step_ahead_m, path_progress):
    """
    Get reference (x_des, y_des, phi_des) from arc-length-based lookahead on ref_xy.
    path_progress: mutable [current_arc_length].
    """
    ref_xy = np.asarray(ref_xy, dtype=float)
    if ref_xy.shape[0] < 2:
        return float(ref_xy[0, 0]), float(ref_xy[0, 1]), 0.0

    arc = path_arc_lengths(ref_xy)
    L = arc[-1]
    s = path_progress[0]

    # Find nearest point on path to update progress
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
    v_ref = kp_pos * dist * cos(angle_error)
    omega_ref = kp_heading * angle_error
    """
    dx = x_des - x[0]
    dy = y_des - x[1]
    dist = math.sqrt(dx ** 2 + dy ** 2)
    angle_to_target = math.atan2(dy, dx)
    angle_error = wrap_angle(angle_to_target - x[2])

    # Always forward — cos(angle_error) can reduce speed but v_ref >= v_min
    v_ref = max(dynamics.v_min, KP_POS * dist * math.cos(angle_error))
    omega_ref = KP_HEADING * angle_error

    v_ref = np.clip(v_ref, -dynamics.v_max, dynamics.v_max)
    omega_ref = np.clip(omega_ref, -dynamics.omega_max, dynamics.omega_max)

    return np.array([float(v_ref), float(omega_ref)])


# ──────────────────────────────────────────────────────────────
# Main control loop
# ──────────────────────────────────────────────────────────────
def run_conflict_cbf_along_path(
    ref_xy, start_x, start_y, start_phi, goal_x, goal_y,
    gsplat, dynamics,
    dt=DT, step_ahead_m=STEP_AHEAD_M, max_steps=MAX_STEPS, goal_tol=GOAL_TOL,
    alpha_s=1.0, alpha_p=0.5, a_max=0.8, beta_lse=10.0, phi_max_deg=45.0,
    k_slack=0.01, w2_slack=None, q_slack=1, w_track=None,
    radius_min=OCCUPANCY_RADIUS_MIN,
    radius_max=OCCUPANCY_RADIUS_MAX,
    splat_nearest_k=OCCUPANCY_NEAREST_K,
    splat_robot_radius=0.0,
    splat_base_radius=0.0,
    obs_decay=None,
    sigma_min=0.05,
    no_obs=False,
    adaptive_penalty=False,
    formulation="conflict",
    i_c=0.1,
    safety_lookahead_m=0.0,
    safety_horizon_s=0.0,
    safety_horizon_n=4,
    freeze_sigma=False,
    adaptive_bias_target=None,
    verbose=True,
):
    """
    Simulate conflict-aware CBF control along ref_xy.

    Safety uses the occupancy-radius method 
    for every splat in the sliced Gaussian field.

    formulation:
        'conflict' — existing ConflictAwareCBFController (default)
        'eq2'      — (h_pi + h_eta, two perception slacks)
        'eq32'     — (h_p = I - I_c, one perception slack)

    Returns dict with trajectory and diagnostics.
    """
    goal_xy_arr = np.array([goal_x, goal_y]) if adaptive_penalty else None
    common_kwargs = dict(
        gsplat=gsplat,
        dynamics=dynamics,
        a_max=a_max,
        q_slack=q_slack,
        beta_lse=beta_lse,
        occupancy_radius_min=radius_min,
        occupancy_radius_max=radius_max,
        occupancy_nearest_k=splat_nearest_k,
        occupancy_robot_radius=splat_robot_radius,
        occupancy_base_radius=splat_base_radius,
        safety_lookahead_m=safety_lookahead_m,
        safety_horizon_s=safety_horizon_s,
        safety_horizon_n=safety_horizon_n,
        freeze_sigma=freeze_sigma,
        adaptive_bias_target=adaptive_bias_target,
        goal_xy=goal_xy_arr,
    )

    if formulation == "eq2":
        controller = Equation2Controller(
            phi_max_deg=phi_max_deg,
            w_s_slack=k_slack,
            w_eta_slack=w2_slack if w2_slack is not None else k_slack,
            **common_kwargs,
        )
    elif formulation == "eq32":
        controller = Equation32Controller(
            I_c=i_c,
            alpha_p=alpha_p,
            w_p_slack=k_slack,
            **common_kwargs,
        )
    else:  # "conflict" — existing controller
        controller = ConflictAwareCBFController(
            alpha_s=alpha_s,
            alpha_p=alpha_p,
            k_slack=k_slack,
            w2_slack=w2_slack,
            phi_max_deg=phi_max_deg,
            w_track=w_track,
            **common_kwargs,
        )

    ref_xy_arr = np.asarray(ref_xy, dtype=float)

    # Set trajectory mask for goal-focused EIG
    controller.set_path_mask(ref_xy_arr, mask_radius=1.5)
    if verbose:
        n_masked = len(controller._path_mask_idx) if hasattr(controller, '_path_mask_idx') else 0
        print(f"  Path-masked EIG: {n_masked} Gaussians near path (of {gsplat.N} total)")

    path_progress = [0.0]

    x = np.array([start_x, start_y, start_phi])

    # Compute total path arc length for progress %
    ref_arc = path_arc_lengths(ref_xy_arr)
    total_arc_length = ref_arc[-1]

    xs, ys, thetas = [x[0]], [x[1]], [x[2]]
    hs_trace, hp_trace, slack_trace, slack2_trace, hp2_trace = [], [], [], [], []
    rho_min_trace, clearance_trace = [], []  # true (non-LSE) diagnostics
    eig_trace, mean_sigma_trace = [], []
    n_p_trace = []
    u_ref_trace, u_safe_trace = [], []
    progress_trace = []  # arc-length progress along A* path (%)

    for step in range(max_steps):
        d_goal = math.hypot(x[0] - goal_x, x[1] - goal_y)
        if d_goal <= goal_tol:
            if verbose:
                print(f"  Goal reached at step {step} (d={d_goal:.3f} m)")
            break

        # FIX 1+2: Observe BEFORE QP so barrier sees current sigma.
        # Only observe when safety has margin (hs > 0.1) to prevent
        # sigma mutation from pushing hs negative.
        obs_kwargs = {'sigma_min': sigma_min}
        if obs_decay is not None:
            obs_kwargs['decay_base'] = obs_decay
        if not no_obs:
            if not hasattr(controller, 'last_hs') or controller.last_hs is None or controller.last_hs > 0.1:
                simulate_observation(x[0], x[1], x[2], gsplat, **obs_kwargs)

        # Reference point on path
        x_des, y_des, phi_des = reference_point_and_heading(
            ref_xy_arr, x[0], x[1], step_ahead_m, path_progress
        )

        # Reference control (unicycle)
        u_ref = reference_controller(x, x_des, y_des, phi_des, dynamics)

        # Conflict-aware CBF-QP filter — QP is the sole safety enforcement
        u_qp = controller.solve(x, u_ref)

        # Integrate with fine sub-steps for Euler accuracy (still no safety filter)
        n_inner = max(1, int(round(dt / DT_SAFETY)))
        dt_inner = dt / n_inner
        for _ in range(n_inner):
            x = dynamics.integrate(x, u_qp, dt_inner)

        # Record diagnostics
        xs.append(x[0])
        ys.append(x[1])
        thetas.append(x[2])
        hs_trace.append(controller.last_hs)
        hp_trace.append(controller.last_hp)
        slack_trace.append(controller.last_slack if controller.last_slack is not None else 0.0)
        slack2_trace.append(controller.last_slack2 if controller.last_slack2 is not None else 0.0)
        hp2_trace.append(controller.last_dI_dtheta_raw if hasattr(controller, 'last_dI_dtheta_raw') and controller.last_dI_dtheta_raw is not None else 0.0)
        eig_trace.append(eig_at_pose(x[0], x[1], x[2], gsplat))
        mean_sigma_trace.append(float(np.mean(gsplat.sigma)))
        u_ref_trace.append(u_ref.copy())
        u_safe_trace.append(u_qp.copy())
        # Arc-length progress along A* path (monotonically increasing, 0-100%)
        progress_trace.append(100.0 * min(path_progress[0] / total_arc_length, 1.0))
        if controller.last_n_p is not None:
            n_p_trace.append(controller.last_n_p.copy())

        # True (non-LSE) diagnostics: min ρᵢ = min_i(‖π-μᵢ‖ - Rᵢ) and raw clearance
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
            rho_min_trace.append(float(np.min(_d - _r)))     # true closest-ball dist
            clearance_trace.append(float(np.min(_d)))         # dist to nearest center
        else:
            rho_min_trace.append(10.0)
            clearance_trace.append(10.0)

        if verbose and step % 200 == 0:
            print(f"  step {step}: pos=({x[0]:.2f},{x[1]:.2f}) theta={math.degrees(x[2]):.1f}° "
                  f"hs={controller.last_hs:+.4f} min_ρ={rho_min_trace[-1]:+.4f} "
                  f"clear={clearance_trace[-1]:.3f} "
                  f"hp={controller.last_hp:.4f} "
                  f"δ₁={slack_trace[-1]:.3f} d_goal={d_goal:.2f}")

    # Print an honest safety summary comparing LSE h_s vs true ρ_min
    if verbose and len(rho_min_trace) > 0:
        _hs_arr = np.asarray(hs_trace)
        _rho_arr = np.asarray(rho_min_trace)
        _clear_arr = np.asarray(clearance_trace)
        print(f"  SAFETY DIAGNOSTICS (LSE-h_s vs true metrics):")
        print(f"    h_s            min={_hs_arr.min():+.4f}  mean={_hs_arr.mean():+.4f}  (LSE soft-min with bias)")
        print(f"    true min ρᵢ    min={_rho_arr.min():+.4f}  mean={_rho_arr.mean():+.4f}  (closest inflated-ball dist)")
        print(f"    clearance      min={_clear_arr.min():.4f}  mean={_clear_arr.mean():.4f}  (dist to nearest splat CENTER)")
        print(f"    if min ρᵢ > 0 but h_s < 0 → LSE bias, not a real violation")

    return {
        "xs": xs, "ys": ys, "thetas": thetas,
        "hs": hs_trace, "hp": hp_trace, "hp2": hp2_trace,
        "rho_min": rho_min_trace, "clearance": clearance_trace,
        "slack": slack_trace, "slack2": slack2_trace,
        "eig": eig_trace, "mean_sigma": mean_sigma_trace,
        "n_p": n_p_trace,
        "u_ref": u_ref_trace, "u_safe": u_safe_trace,
        "progress": progress_trace,
        "formulation": formulation,
    }


# ──────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="A* path planning + Conflict-Aware CBF controller ()"
    )
    parser.add_argument("file", nargs="?", default="params/yq_large_cloud.mat")
    parser.add_argument("--dt", type=float, default=DT)
    parser.add_argument("--phi-max", type=float, default=45.0,
                        help="Max perception angle from EIG gradient (degrees)")
    parser.add_argument("--fov-deg", type=float, default=40.0,
                        help="Camera FOV for EIG computation (degrees)")
    parser.add_argument("--fov-sigmoid-k", type=float, default=30.0,
                        help="Sigmoid sharpness at FOV boundary (higher = sharper)")
    parser.add_argument("--beta-lse", type=float, default=10.0,
                        help="Log-sum-exp sharpness for safety barrier")
    parser.add_argument("--alpha-s", type=float, default=1.0,
                        help="Safety CBF class-K coefficient (linear fallback)")
    parser.add_argument("--alpha-p", type=float, default=0.5,
                        help="Perception CBF class-K coefficient")
    parser.add_argument("--a-max", type=float, default=0.8,
                        help="Max deceleration for safety class-K α(hs)=√(2·amax·hs)")
    parser.add_argument("--k-slack", type=float, default=0.01,
                        help="Slack penalty weight w1 for δ₁ (perception heading)")
    parser.add_argument("--w2-slack", type=float, default=None,
                        help="Slack penalty weight w2 for δ₂ (EIG heading gradient, default: same as k-slack)")
    parser.add_argument("--w-track", type=float, default=None,
                        help="Reference tracking weight (default: 10.0). Lower = more control deviation")
    parser.add_argument("--radius-min", type=float, default=OCCUPANCY_RADIUS_MIN,
                        help="Minimum per-splat occupancy radius")
    parser.add_argument("--radius-max", type=float, default=OCCUPANCY_RADIUS_MAX,
                        help="Maximum per-splat occupancy radius")
    parser.add_argument("--occupancy-radius-scale", type=float, default=1.0,
                        help="Shared multiplicative scale applied to occupancy radii unless planner/safety overrides are set")
    parser.add_argument("--astar-occupancy-scale", type=float, default=None,
                        help="Optional planner-only multiplicative scale for occupancy radii")
    parser.add_argument("--safety-occupancy-scale", type=float, default=None,
                        help="Optional h_s-only multiplicative scale for occupancy radii")
    parser.add_argument("--splat-nearest-k", type=int, default=OCCUPANCY_NEAREST_K,
                        help="Restrict h_s to the nearest K splats; use 0 for all")
    parser.add_argument("--splat-robot-radius", type=float, default=0.0,
                        help="Extra robot-radius inflation added to each splat occupancy radius")
    parser.add_argument("--splat-base-radius", type=float, default=0.0,
                        help="Extra fixed inflation added to each splat occupancy radius")
    parser.add_argument("--obs-decay", type=float, default=None,
                        help="Observation decay rate (0-1). How much sigma drops per step at distance 0. Default: 0.3")
    parser.add_argument("--sigma-min", type=float, default=0.05,
                        help="Minimum sigma floor — splats never decay below this (default: 0.05)")
    parser.add_argument("--adaptive-bias-target", type=float, default=None,
                        help="Target LSE bias (m) — adapt β per-call so that "
                             "log(M_active)/β ≈ this value, regardless of how "
                             "many splats are nearby. Pins the worst-case h_s "
                             "negative dip to this magnitude. Try 0.005 (5 mm).")
    parser.add_argument("--freeze-sigma", action="store_true",
                        help="Freeze CBF safety radii Rᵢ at the initial σ values. "
                             "Prevents percentile renormalization from shifting Rᵢ "
                             "of unobserved splats as simulate_observation decays σ. "
                             "Matches the theorem's assumption of a static safe set.")
    parser.add_argument("--astar-uniform-max-radius", action="store_true",
                        help="Build A* grid with UNIFORM max(Rᵢ) inflation for every "
                             "splat (not per-splat Rᵢ). Guarantees any A* path stays "
                             "outside all possible CBF balls, even after percentile "
                             "renormalization shifts Rᵢ via simulate_observation.")
    parser.add_argument("--safety-lookahead", type=float, default=0.0,
                        help="Kinematic lookahead distance for safety CBF (m). "
                             "h_s evaluated at π + L·[cosθ, sinθ] instead of π. "
                             "Helps compensate forward-Euler overshoot. Try 0.05-0.15.")
    parser.add_argument("--safety-horizon-s", type=float, default=0.0,
                        help="Predictive safety horizon T (seconds). If > 0, the "
                             "safety CBF is enforced on a soft-min of h_s across "
                             "N+1 points along the predicted forward arc over T s. "
                             "Gives the CBF time to react BEFORE a violation. "
                             "Try 0.1-0.5 s.")
    parser.add_argument("--safety-horizon-n", type=int, default=4,
                        help="Number of prediction points along --safety-horizon-s. "
                             "More points = smoother but more compute. Default 4.")
    parser.add_argument("--formulation", choices=["conflict", "eq2", "eq32"],
                        default="conflict",
                        help="CBF formulation: 'conflict' (existing, default), "
                             "'eq2' (with h_pi + h_eta), "
                             "'eq32' (with single h_p = I - I_c)")
    parser.add_argument("--i-c", type=float, default=0.1,
                        help="EIG threshold I_c for --formulation eq32")
    parser.add_argument("--q-slack", type=int, default=1, choices=[1, 2],
                        help="Slack penalty exponent q: 1=L1 (sparse), 2=L2 (quadratic)")
    parser.add_argument("--max-steps", type=int, default=MAX_STEPS)
    parser.add_argument("--z-thickness", type=float, default=0.5,
                        help="Z-band thickness (m) for slicing splats at robot height (default: 0.5)")
    parser.add_argument("--interactive", action="store_true",
                        help="Click start and goal on the 2D slice image")
    parser.add_argument("--planner", choices=["astar", "rrt"], default="astar",
                        help="Path planner: astar (grid-based) or rrt (grid-free)")
    parser.add_argument("--grid-res", type=float, default=None,
                        help="Override grid resolution (m). Default: 0.05 for astar, unused for rrt")
    parser.add_argument("--astar-mode", choices=["safe", "plain"], default="safe",
                        help="A* mode: safe (clearance cost) or plain (no clearance cost)")
    parser.add_argument("--occupancy-inflate", type=float, default=0.0,
                        help="Extra world-space inflation shared by A* planning occupancy and CBF h_s radii")
    parser.add_argument("--deflate", type=int, default=0,
                        help="Erode occupancy grid by N cells after A* planning (more free space for CBF)")
    parser.add_argument("--no-obs", action="store_true",
                        help="Disable simulated observation (no uncertainty reduction)")
    parser.add_argument("--adaptive-penalty", action="store_true",
                        help="Enable map-state adaptive slack penalty P(x) (default: off, use constant k_slack, w2_slack)")
    parser.add_argument("--hs-margin", type=float, default=0.0,
                        help="Safety buffer: enforce hs >= margin instead of hs >= 0 (pushes robot away from obstacles)")
    parser.add_argument("--compare-norms", action="store_true",
                        help="Run both q=1 (L1) and q=2 (L2) and generate separate diagnostic plots")
    args = parser.parse_args()

    # ── Load point cloud and build grid ──
    means3D, rgb_colors = load_point_cloud(args.file)

    # Thin z-slice: exact robot plane (thinner = fewer Gaussians, more clearance)
    z_center = GROUND_Z + ROBOT_HEIGHT / 2.0
    z_thickness = args.z_thickness
    z_band_min = z_center - z_thickness / 2.0
    z_band_max = z_center + z_thickness / 2.0
    xy, rgb_slice = slice_at_z_range(means3D, rgb_colors, z_band_min, z_band_max)

    # Build the shared Gaussian field before occupancy so A* and CBF use the same splats.
    gsplat = GaussianSplatField.from_mat(args.file, z_band=(z_band_min, z_band_max))
    print(f"Perception field: {gsplat.N} Gaussians, mean sigma = {np.mean(gsplat.sigma):.4f}")

    # Grid resolution: use override if given, else default
    grid_res = args.grid_res if args.grid_res is not None else GRID_RESOLUTION
    print(f"Grid resolution: {grid_res} m")
    astar_scale = args.occupancy_radius_scale if args.astar_occupancy_scale is None else args.astar_occupancy_scale
    safety_scale = args.occupancy_radius_scale if args.safety_occupancy_scale is None else args.safety_occupancy_scale
    astar_radius_min = args.radius_min * astar_scale
    astar_radius_max = args.radius_max * astar_scale
    safety_radius_min = args.radius_min * safety_scale
    safety_radius_max = args.radius_max * safety_scale

    # Base occupancy: rasterize per-splat uncertainty radii.
    # If --astar-uniform-max-radius, use radius_max for EVERY splat instead of
    # scaling by per-splat σᵢ. This bounds the A* free-space by the largest
    # possible CBF ball, so any A* path remains outside every CBF ball — even
    # after simulate_observation shifts the percentile bounds and makes some
    # previously-small Rᵢ larger.
    if args.astar_uniform_max_radius:
        # Setting radius_min == radius_max makes the mapped radius constant
        # regardless of per-splat σᵢ (see compute_occupancy_radii_from_uncertainty).
        astar_rmin_build, astar_rmax_build = astar_radius_max, astar_radius_max
        print(f"A* grid: uniform max inflation Rᵢ = {astar_radius_max:.3f} m per splat")
    else:
        astar_rmin_build, astar_rmax_build = astar_radius_min, astar_radius_max
    base_grid, _grid_score, x_min, y_min, res = build_uncertainty_occupancy_grid(
        gsplat.means_xy,
        gsplat.sigma,
        grid_res,
        radius_min=astar_rmin_build,
        radius_max=astar_rmax_build,
    )
    grid = base_grid.copy()
    extra_inflate_cells = max(0, int(math.ceil(args.occupancy_inflate / grid_res)))
    if extra_inflate_cells > 0:
        from scipy.ndimage import binary_dilation

        structure = np.ones((2 * extra_inflate_cells + 1, 2 * extra_inflate_cells + 1), dtype=bool)
        grid = binary_dilation(grid, structure=structure).astype(bool)
    use_safe = (args.astar_mode == "safe")
    clearance_map = build_clearance_map(grid, res) if use_safe else None

    # Deflate: erode the grid after A* planning to give CBF more room
    if args.deflate > 0:
        from scipy.ndimage import binary_erosion
        d = args.deflate
        struct = np.ones((2 * d + 1, 2 * d + 1))
        grid = binary_erosion(grid, structure=struct).astype(grid.dtype)
        clearance_map = build_clearance_map(grid, res) if use_safe else None
        print(f"Deflated grid by {d} cells ({d * grid_res:.3f}m)")

    print(
        "Occupancy map: Documents/cbf uncertainty radii "
        f"(radius_min={astar_radius_min}, radius_max={astar_radius_max}, "
        f"scale={astar_scale})"
    )
    if args.occupancy_inflate > 0.0:
        print(
            f"A* occupancy: added extra inflation of {args.occupancy_inflate:.3f} m "
            f"({extra_inflate_cells} cells) after uncertainty rasterization"
        )
    print(
        "CBF safety: occupancy radii per splat "
        f"(radius_min={safety_radius_min}, radius_max={safety_radius_max}, "
        f"scale={safety_scale}, nearest_k={args.splat_nearest_k}, "
        f"robot_radius={args.splat_robot_radius}, base_radius={args.splat_base_radius}, "
        f"shared_inflate={args.occupancy_inflate})"
    )

    nx, ny = grid.shape
    x_max = x_min + nx * res
    y_max = y_min + ny * res
    planning_grid_viz = base_grid
    trajectory_grid_viz = build_uncertainty_display_grid_on_bounds(
        gsplat.means_xy,
        gsplat.sigma,
        x_min,
        y_min,
        res,
        nx,
        ny,
        radius_min=safety_radius_min,
        radius_max=safety_radius_max,
        robot_radius=args.splat_robot_radius,
        base_radius=args.splat_base_radius + args.occupancy_inflate,
    )

    # ── Start and goal ──
    if args.interactive:
        import matplotlib.pyplot as plt
        fig_pick, ax_pick = plt.subplots(figsize=(10, 8))
        ax_pick.imshow(
            planning_grid_viz.T,
            extent=[x_min, x_min + nx * res, y_min, y_min + ny * res],
            origin="lower", cmap="Greys", alpha=0.4, aspect="equal",
        )
        step_slice_pick = max(1, xy.shape[0] // 80000)
        ax_pick.scatter(xy[::step_slice_pick, 0], xy[::step_slice_pick, 1],
                        c=rgb_slice[::step_slice_pick], s=0.5, alpha=0.8)
        ax_pick.set_title("Click START point, then GOAL point (2 clicks)")
        print("Click START point on the image, then click GOAL point...")
        pts = plt.ginput(2, timeout=0)
        plt.close(fig_pick)
        if len(pts) < 2:
            print("Need 2 clicks (start and goal). Aborting.")
            return
        start_x, start_y = pts[0]
        goal_x, goal_y = pts[1]
        start_phi = math.atan2(goal_y - start_y, goal_x - start_x)
        print(f"Selected start=({start_x:.2f}, {start_y:.2f}), "
              f"goal=({goal_x:.2f}, {goal_y:.2f}), phi={math.degrees(start_phi):.1f}°")
    else:
        start_x = x_min + 2 * res
        start_y = y_min + 2 * res
        start_phi = 0.0
        goal_x = x_max - 2 * res
        goal_y = y_max - 2 * res

    # ── Path planning ──
    if args.planner == "rrt":
        from rrt_planner import rrt_star, smooth_rrt_path, rrt_path_to_xy
        gsplat_obs = gsplat.filter_to_obstacle_gaussians(grid, x_min, y_min, res)
        print(f"RRT obstacle field: {gsplat_obs.N} Gaussians")

        print("Running RRT* (grid-free) ...")
        rrt_path = rrt_star(
            (start_x, start_y), (goal_x, goal_y),
            gsplat_obs, ROBOT_RADIUS,
            bounds=((x_min, y_min), (x_max, y_max)),
        )
        if rrt_path is None:
            print("RRT* failed to find a path.")
            return

        rrt_path = smooth_rrt_path(rrt_path, gsplat_obs, ROBOT_RADIUS)
        ref_xy = rrt_path_to_xy(rrt_path)
        path_xw = list(ref_xy[:, 0])
        path_yw = list(ref_xy[:, 1])
        start_x_w, start_y_w = path_xw[0], path_yw[0]
        goal_x_w, goal_y_w = path_xw[-1], path_yw[-1]

    else:
        # A* on occupancy grid
        start_ix = max(0, min(nx - 1, int((start_x - x_min) / res)))
        start_iy = max(0, min(ny - 1, int((start_y - y_min) / res)))
        goal_ix = max(0, min(nx - 1, int((goal_x - x_min) / res)))
        goal_iy = max(0, min(ny - 1, int((goal_y - y_min) / res)))
        start_ip = phi_to_index(start_phi, N_HEADINGS)

        # Shift if occupied
        for target, tix, tiy in [("start", start_ix, start_iy), ("goal", goal_ix, goal_iy)]:
            ix_ref, iy_ref = tix, tiy
            if grid[ix_ref, iy_ref]:
                for r in range(1, 50):
                    found = False
                    for ddx in range(-r, r + 1):
                        for ddy in range(-r, r + 1):
                            ixx, iyy = ix_ref + ddx, iy_ref + ddy
                            if 0 <= ixx < nx and 0 <= iyy < ny and not grid[ixx, iyy]:
                                if target == "start":
                                    start_ix, start_iy = ixx, iyy
                                else:
                                    goal_ix, goal_iy = ixx, iyy
                                found = True
                                break
                        if found:
                            break
                    if found:
                        break

        print("Running Safe A* ..." if use_safe else "Running A* (no clearance cost) ...")
        path = astar(grid, start_ix, start_iy, start_ip, goal_ix, goal_iy,
                     N_HEADINGS, clearance_map=clearance_map)
        if not path:
            print("No path found.")
            return
        print(f"A* path length: {len(path)}")

        path_z = GROUND_Z + ROBOT_HEIGHT / 2.0
        path_xw = [x_min + (ix + 0.5) * res for ix, iy, ip in path]
        path_yw = [y_min + (iy + 0.5) * res for ix, iy, ip in path]
        path_zw = [path_z] * len(path_xw)

        if SMOOTH_PATH and len(path_xw) >= 3:
            n_smooth = SMOOTH_NUM_POINTS or (len(path_xw) * 3)
            path_xw, path_yw, path_zw = smooth_path_spline(path_xw, path_yw, path_zw, n_smooth)
            print(f"Smoothed path points: {len(path_xw)}")

        ref_xy = np.column_stack([path_xw, path_yw])
        start_x_w = path_xw[0]
        start_y_w = path_yw[0]
        goal_x_w = path_xw[-1]
        goal_y_w = path_yw[-1]

    # ── Override EIG FOV parameters from CLI ──
    import risk_aware_eig as _eig_mod
    _eig_mod.FOV_DEG = args.fov_deg
    _eig_mod.FOV_SIGMOID_K = args.fov_sigmoid_k
    _eig_mod.SIGMA_MIN = args.sigma_min

    # ── Run conflict-aware CBF ──
    dynamics = UnicycleDynamics()
    print(f"Running conflict-aware CBF [formulation={args.formulation}] "
          f"(dt={args.dt}, phi_max={args.phi_max}°, "
          f"fov={args.fov_deg}°, fov_k={args.fov_sigmoid_k}, "
          f"beta={args.beta_lse}, alpha_s={args.alpha_s}, alpha_p={args.alpha_p}"
          + (f", I_c={args.i_c}" if args.formulation == 'eq32' else '')
          + ")...")

    results = run_conflict_cbf_along_path(
        ref_xy, start_x_w, start_y_w, start_phi, goal_x_w, goal_y_w,
        gsplat, dynamics,
        dt=args.dt, max_steps=args.max_steps,
        alpha_s=args.alpha_s, alpha_p=args.alpha_p, a_max=args.a_max,
        beta_lse=args.beta_lse, phi_max_deg=args.phi_max,
        k_slack=args.k_slack, w2_slack=args.w2_slack, q_slack=args.q_slack,
        w_track=args.w_track,
        radius_min=safety_radius_min,
        radius_max=safety_radius_max,
        splat_nearest_k=args.splat_nearest_k,
        splat_robot_radius=args.splat_robot_radius,
        splat_base_radius=args.splat_base_radius + args.occupancy_inflate,
        obs_decay=args.obs_decay,
        sigma_min=args.sigma_min,
        no_obs=args.no_obs,
        adaptive_penalty=args.adaptive_penalty,
        formulation=args.formulation,
        i_c=args.i_c,
        safety_lookahead_m=args.safety_lookahead,
        safety_horizon_s=args.safety_horizon_s,
        safety_horizon_n=args.safety_horizon_n,
        freeze_sigma=args.freeze_sigma,
        adaptive_bias_target=args.adaptive_bias_target,
    )

    # ── Compare norms mode: run both L1 and L2 ──
    if args.compare_norms:
        results_q1 = results  # already ran with whatever q-slack was set
        print(f"\n{'='*50}")
        print(f"Compare mode: re-running with q={3 - args.q_slack}...")
        print(f"{'='*50}")
        gsplat2 = GaussianSplatField.from_mat(args.file, z_band=(z_band_min, z_band_max))
        q_other = 3 - args.q_slack  # if user chose 1, run 2; if 2, run 1
        results_q2 = run_conflict_cbf_along_path(
            ref_xy, start_x_w, start_y_w, start_phi, goal_x_w, goal_y_w,
            gsplat2, dynamics,
            dt=args.dt, max_steps=args.max_steps,
            alpha_s=args.alpha_s, alpha_p=args.alpha_p, a_max=args.a_max,
            beta_lse=args.beta_lse, phi_max_deg=args.phi_max,
            k_slack=args.k_slack, w2_slack=args.w2_slack, q_slack=q_other,
            w_track=args.w_track,
            radius_min=safety_radius_min,
            radius_max=safety_radius_max,
            splat_nearest_k=args.splat_nearest_k,
            splat_robot_radius=args.splat_robot_radius,
            splat_base_radius=args.splat_base_radius + args.occupancy_inflate,
            obs_decay=args.obs_decay,
        sigma_min=args.sigma_min,
            adaptive_penalty=args.adaptive_penalty,
            formulation=args.formulation,
            i_c=args.i_c,
            safety_lookahead_m=args.safety_lookahead,
        safety_horizon_s=args.safety_horizon_s,
        safety_horizon_n=args.safety_horizon_n,
        freeze_sigma=args.freeze_sigma,
        adaptive_bias_target=args.adaptive_bias_target,
        )
        # Assign: q1=L1, q2=L2
        if args.q_slack == 1:
            all_runs = {1: results_q1, 2: results_q2}
        else:
            all_runs = {1: results_q2, 2: results_q1}

        import matplotlib.pyplot as plt
        for q, res_q in all_runs.items():
            label = f"q={q}"
            fname = f"conflict_cbf_q{q}.png"
            color = "tab:blue" if q == 1 else "#4B0082"

            T = len(res_q["hs"])
            hs_arr = np.array(res_q["hs"])
            hp_arr = np.array(res_q["hp"])
            sl_arr = np.array(res_q["slack"])
            u_ref_arr = np.array(res_q["u_ref"])
            u_safe_arr = np.array(res_q["u_safe"])
            progress = np.array(res_q["progress"])

            u_diff_l2 = np.sqrt(np.sum((u_safe_arr - u_ref_arr)**2, axis=1))
            hs_dot = np.concatenate([[0.0], np.diff(hs_arr) / args.dt])
            hp_dot = np.concatenate([[0.0], np.diff(hp_arr) / args.dt])
            v_arr = u_safe_arr[:, 0]
            omega_arr = u_safe_arr[:, 1]

            fig, axes = plt.subplots(3, 1, figsize=(14, 14), sharex=True)
            fig.subplots_adjust(hspace=0.3)

            # Formulation-aware labels: eq32 uses a single h_p = I - I_c barrier,
            # while conflict/eq2 have two perception barriers (h_pi + h_eta).
            _formulation = res_q.get("formulation", "conflict")
            if _formulation == "eq32":
                hp_label = r"$h_p = I - I_c$ perception"
                slack_label = r"$\delta$"
                show_second_curve = False
            elif _formulation == "eq2":
                hp_label = r"$h_{\pi}$ spatial perception"
                slack_label = r"$\delta_s$"
                show_second_curve = True
                hp2_label = r"$h_{\eta}$ angular perception"
                slack2_label = r"$\delta_{\eta}$"
            else:  # conflict
                hp_label = r"$h_{\pi}$ perception"
                slack_label = r"$\delta_1$"
                show_second_curve = True
                hp2_label = r"$h_{\eta}$ perception"
                slack2_label = r"$\delta_2$"

            # 1. hs, hp1, hp2 on the SAME plot
            ax = axes[0]
            hp2_arr = np.array(res_q.get("hp2", np.zeros_like(hp_arr)))
            hp2_absmax = np.max(np.abs(hp2_arr)) if np.max(np.abs(hp2_arr)) > 1e-8 else 1.0
            hp2_arr = hp2_arr / hp2_absmax
            hs_pos_pct = 100 * np.mean(hs_arr >= 0)
            hp1_pos_pct = 100 * np.mean(hp_arr >= 0)
            ax.plot(progress, hs_arr, color="darkgreen", lw=1.2,
                    label=r"$h_s$ safety ($h_{s,\min}$=" + f"{hs_arr.min():.4f}, $>0$: {hs_pos_pct:.0f}%)")
            ax.plot(progress, hp_arr, color="tab:blue", lw=1,
                    label=hp_label + r" ($>0$: " + f"{hp1_pos_pct:.0f}%)")
            if show_second_curve:
                hp2_pos_pct = 100 * np.mean(hp2_arr >= 0)
                ax.plot(progress, hp2_arr, color="#4B0082", lw=1,
                        label=hp2_label + r" ($>0$: " + f"{hp2_pos_pct:.0f}%)")
            ax.axhline(0, color="k", ls="--", lw=1)
            ax.fill_between(progress, 0, hs_arr, where=(hs_arr < 0), color="#4B0082", alpha=0.2)
            ax.set_ylabel(f"Barrier value (q={q})", fontsize=13)
            ax.legend(fontsize=11, loc="lower right"); ax.grid(True, alpha=0.3)

            # 2. Slack (raw)
            ax = axes[1]
            sl2_arr_plot = np.array(res_q.get("slack2", np.zeros_like(sl_arr)))
            ax.plot(progress, sl_arr, color="tab:blue", lw=1, label=slack_label)
            ax.fill_between(progress, 0, sl_arr, color="tab:blue", alpha=0.15)
            if show_second_curve:
                ax.plot(progress, sl2_arr_plot, color="#4B0082", lw=1, label=slack2_label)
                ax.fill_between(progress, 0, sl2_arr_plot, color="#4B0082", alpha=0.15)
            ax.set_ylabel(f"Slack variable (q={q})", fontsize=13)
            ax.legend(fontsize=10)
            ax.grid(True, alpha=0.3)

            # 3. Slack (smoothed — Savitzky-Golay preserves drops)
            ax = axes[2]
            from scipy.signal import savgol_filter
            _sg_win = max(5, len(sl_arr) // 50) | 1  # must be odd
            sl_arr_smooth = np.maximum(savgol_filter(sl_arr, window_length=_sg_win, polyorder=3), 0)
            ax.plot(progress, sl_arr_smooth, color="tab:blue", lw=1.5, label=slack_label)
            ax.fill_between(progress, 0, sl_arr_smooth, color="tab:blue", alpha=0.15)
            if show_second_curve:
                sl2_smooth = np.maximum(savgol_filter(sl2_arr_plot, window_length=_sg_win, polyorder=3), 0)
                ax.plot(progress, sl2_smooth, color="#4B0082", lw=1.5, label=slack2_label)
                ax.fill_between(progress, 0, sl2_smooth, color="#4B0082", alpha=0.15)
            ax.set_ylabel(f"Slack variable (q={q})", fontsize=13)
            ax.set_xlabel("Progress along the path (%)", fontsize=12)
            ax.legend(fontsize=10)
            ax.grid(True, alpha=0.3)

            # no suptitle
            plt.savefig(fname, dpi=150, bbox_inches="tight")
            print(f"Saved: {fname}")
            plt.close()

            # Summary
            print(f"\n  {label}: {T} steps")
            print(f"  ||u-u_ref||_2: mean={np.mean(u_diff_l2):.4f} max={np.max(u_diff_l2):.4f}")
            nz1_pct = 100.0 * np.count_nonzero(sl_arr > 1e-6) / max(T, 1)
            print(f"  {slack_label}: mean={np.mean(sl_arr):.4f} max={np.max(sl_arr):.4f} nz={nz1_pct:.0f}%")
            if show_second_curve:
                nz2_pct = 100.0 * np.count_nonzero(sl2_arr_plot > 1e-6) / max(T, 1)
                print(f"  {slack2_label}: mean={np.mean(sl2_arr_plot):.4f} max={np.max(sl2_arr_plot):.4f} nz={nz2_pct:.0f}%")
            print(f"  hs: min={np.min(hs_arr):.4f} mean={np.mean(hs_arr):.4f}")
            print(f"  {hp_label}: min={np.min(hp_arr):.4f} >0={hp1_pos_pct:.0f}%")
            if show_second_curve:
                print(f"  {hp2_label}: min={np.min(hp2_arr):.4f} >0={hp2_pos_pct:.0f}%")
            print(f"  v: mean={np.mean(v_arr):.3f} omega_mean={np.mean(np.abs(omega_arr)):.3f} rad/s")

        # ── Combined trajectory: A*, L1, L2 on one plot ──
        fig_traj, ax_t = plt.subplots(figsize=(14, 8))
        from matplotlib.lines import Line2D

        # Plot with swapped axes (y→horizontal, x→vertical) for landscape orientation
        ax_t.imshow(
            trajectory_grid_viz,
            extent=[y_min, y_min + ny * res, x_min, x_min + nx * res],
            origin="lower", cmap="Greys", alpha=0.35, aspect="equal"
        )
        step_s = max(1, xy.shape[0]//60000)
        ax_t.scatter(xy[::step_s,1], xy[::step_s,0], c=rgb_slice[::step_s], s=0.3, alpha=0.5)
        ax_t.plot(path_yw, path_xw, "k-", lw=2, alpha=0.6)
        legend_handles = [Line2D([0], [0], color="k", lw=2, alpha=0.6, label="Reference path")]
        ax_t.plot(all_runs[1]["ys"], all_runs[1]["xs"], color="tab:blue", lw=2, alpha=1.0, zorder=4)
        legend_handles.append(Line2D([0], [0], color="tab:blue", lw=2, label="q=1"))
        ax_t.plot(all_runs[2]["ys"], all_runs[2]["xs"], color="#4B0082", lw=2, alpha=0.9, zorder=4)
        legend_handles.append(Line2D([0], [0], color="#4B0082", lw=2, label="q=2"))
        ax_t.scatter([start_y_w], [start_x_w], c="g", s=120, zorder=5)
        ax_t.scatter([goal_y_w], [goal_x_w], c="#4B0082", s=120, zorder=5)
        legend_handles.append(Line2D([0], [0], marker="o", color="w", markerfacecolor="g", markersize=9, label="start"))
        legend_handles.append(Line2D([0], [0], marker="o", color="w", markerfacecolor="#4B0082", markersize=9, label="goal"))
        ax_t.legend(handles=legend_handles, fontsize=10, loc="upper left")
        ax_t.set_xlim(-12, 0)
        ax_t.set_ylim(-3, 6)
        ax_t.set_aspect("equal")
        ax_t.set_xticklabels([])
        ax_t.set_yticklabels([])
        plt.savefig("conflict_cbf_trajectories.png", dpi=150, bbox_inches="tight")
        print("Saved: conflict_cbf_trajectories.png")
        plt.close()

        # ── Occupancy radii visualization: initial vs final uncertainty ──
        # Reload fresh sigma for "initial"
        gsplat_init = GaussianSplatField.from_mat(args.file, z_band=(z_band_min, z_band_max))
        sigma_init = gsplat_init.sigma.copy()
        sigma_final = gsplat.sigma.copy()  # gsplat was mutated by first run (L1 or L2)
        mu_xy = gsplat.means_xy

        fig_avar, (ax_i, ax_f) = plt.subplots(1, 2, figsize=(20, 10))
        for ax, sigma, _title in [(ax_i, sigma_init, "Initial Uncertainty"),
                                   (ax_f, sigma_final, "After Active Perception")]:
            ax.imshow(trajectory_grid_viz.T, extent=[x_min, x_min+nx*res, y_min, y_min+ny*res],
                      origin="lower", cmap="Greys", alpha=0.25, aspect="equal")
            ax.plot(path_xw, path_yw, "k-", lw=1, alpha=0.3)
            ax.plot(all_runs[1]["xs"], all_runs[1]["ys"], color="tab:blue", lw=1.5, alpha=0.6, zorder=3, label="q=1")

            # Draw occupancy radii as circles colored by sigma
            radii = compute_occupancy_radii_from_uncertainty(
                sigma,
                radius_min=safety_radius_min,
                radius_max=safety_radius_max,
                robot_radius=args.splat_robot_radius,
                base_radius=args.splat_base_radius + args.occupancy_inflate,
            )
            # Only show Gaussians near the trajectory (within 3m of path)
            ref_xy_2col = np.column_stack([path_xw, path_yw])
            from scipy.spatial import cKDTree
            tree = cKDTree(ref_xy_2col)
            dists, _ = tree.query(mu_xy, k=1)
            near = dists <= 3.0

            # Sample for performance
            near_idx = np.where(near)[0]
            sample = near_idx[::max(1, len(near_idx)//500)]

            for i in sample:
                r = radii[i]
                if r < 0.001:
                    continue
                circle = plt.Circle((mu_xy[i, 0], mu_xy[i, 1]), r,
                                    fill=True, alpha=0.15,
                                    color=plt.cm.RdYlGn_r(min(1.0, sigma[i] / 0.3)),
                                    linewidth=0.3, edgecolor='k')
                ax.add_patch(circle)
                ax.plot(mu_xy[i, 0], mu_xy[i, 1], '.', color='k', markersize=0.5)

            ax.scatter([start_x_w], [start_y_w], c="g", s=80, zorder=5)
            ax.scatter([goal_x_w], [goal_y_w], c="#4B0082", s=80, zorder=5)
            # no title
            ax.axis("equal")
            # Match axes limits
            all_x = list(all_runs[1]["xs"]) + list(path_xw)
            all_y = list(all_runs[1]["ys"]) + list(path_yw)
            margin = 1.5
            ax.set_xlim(-3, 6)
            ax.set_ylim(-12, 0)
            ax.set_xticklabels([])
            ax.set_yticklabels([])

        # Colorbar
        import matplotlib.cm as cm
        sm = plt.cm.ScalarMappable(cmap=plt.cm.RdYlGn_r, norm=plt.Normalize(0, 0.3))
        sm.set_array([])
        cbar = fig_avar.colorbar(sm, ax=[ax_i, ax_f], shrink=0.6, label="σ (uncertainty)")

        # no suptitle
        plt.savefig("conflict_cbf_occupancy_radii.png", dpi=150, bbox_inches="tight")
        print("Saved: conflict_cbf_occupancy_radii.png")
        plt.close()

        # ── q1 vs q2 comparison: hs, hp, slack overlaid ──
        r1, r2 = all_runs[1], all_runs[2]
        p1, p2 = np.array(r1["progress"]), np.array(r2["progress"])
        hs1, hs2 = np.array(r1["hs"]), np.array(r2["hs"])
        hp1, hp2 = np.array(r1["hp"]), np.array(r2["hp"])
        sl1, sl2 = np.array(r1["slack"]), np.array(r2["slack"])
        T1, T2 = len(hs1), len(hs2)

        fig_cmp, axes_cmp = plt.subplots(3, 1, figsize=(14, 14), sharex=True)
        fig_cmp.subplots_adjust(hspace=0.3)

        # 1. hs overlaid
        ax = axes_cmp[0]
        hs_min_both = min(hs1.min(), hs2.min())
        ax.plot(p1, hs1, "tab:blue", lw=1, label=f"q=1 ($h_{{s,\\min}}$={hs1.min():.4f})")
        ax.plot(p2, hs2, "#4B0082", lw=1, alpha=0.7, label=f"q=2 ($h_{{s,\\min}}$={hs2.min():.4f})")
        ax.axhline(0, color="k", ls="--", lw=1)
        ax.set_ylabel(r"$h_s$", fontsize=13)
        ax.legend(fontsize=11); ax.grid(True, alpha=0.3)

        # 2. hp overlaid
        ax = axes_cmp[1]
        hp_pos1 = 100 * np.mean(hp1 >= 0)
        hp_pos2 = 100 * np.mean(hp2 >= 0)
        ax.plot(p1, hp1, "tab:blue", lw=0.8, label=f"q=1 ($h_\\pi>0$: {hp_pos1:.0f}%)")
        ax.plot(p2, hp2, "#4B0082", lw=0.8, alpha=0.7, label=f"q=2 ($h_\\pi>0$: {hp_pos2:.0f}%)")
        ax.axhline(0, color="k", ls="--", lw=1)
        ax.fill_between(p1, 0, hp1, where=(hp1 >= 0), alpha=0.1, color="green")
        ax.fill_between(p1, hp1, 0, where=(hp1 < 0), alpha=0.1, color="#4B0082")
        ax.set_ylabel(r"$h_\pi$", fontsize=13)
        ax.legend(fontsize=11); ax.grid(True, alpha=0.3)

        # 3. slack overlaid (both filled line style)
        ax = axes_cmp[2]
        ax.fill_between(p1, 0, sl1, color="tab:blue", alpha=0.2)
        ax.plot(p1, sl1, "tab:blue", lw=1, label="q=1")
        ax.fill_between(p2, 0, sl2, color="#4B0082", alpha=0.2)
        ax.plot(p2, sl2, "#4B0082", lw=1, label="q=2")
        ax.set_ylabel(r"$\delta_p$", fontsize=13)
        ax.set_xlabel("Progress along the path (%)", fontsize=12)
        # no title
        ax.legend(fontsize=11); ax.grid(True, alpha=0.3)

        # no suptitle
        plt.savefig("conflict_cbf_q1_vs_q2.png", dpi=150, bbox_inches="tight")
        print("Saved: conflict_cbf_q1_vs_q2.png")
        plt.close()

        print("\nDone. Saved conflict_cbf_q1.png, conflict_cbf_q2.png, "
              "conflict_cbf_q1_vs_q2.png, conflict_cbf_trajectories.png, "
              "conflict_cbf_occupancy_radii.png")
        _save_run_summary(args, results_q1, results_q2)
        return

    xs = results["xs"]
    ys = results["ys"]
    thetas = results["thetas"]
    print(f"CBF trajectory: {len(xs)} points")
    print(f"Final mean sigma: {np.mean(gsplat.sigma):.4f} "
          f"(started at ~0.5, reduced by observations)")

    # ── Compute metrics ──
    u_ref_arr = np.array(results["u_ref"])    # (T, 2)
    u_safe_arr = np.array(results["u_safe"])  # (T, 2)
    u_diff = u_safe_arr - u_ref_arr
    u_diff_l2 = np.sqrt(np.sum(u_diff ** 2, axis=1))  # ||u - u_ref||_2 per step

    slack_arr = np.array(results["slack"])
    slack2_arr = np.array(results.get("slack2", np.zeros_like(slack_arr)))
    hs_arr = np.array(results["hs"])
    hp_arr = np.array(results["hp"])
    hp2_arr = np.array(results.get("hp2", np.zeros_like(hp_arr)))
    hp2_absmax = np.max(np.abs(hp2_arr)) if np.max(np.abs(hp2_arr)) > 1e-8 else 1.0
    hp2_arr = hp2_arr / hp2_absmax

    # ── Visualization: stacked vertical panels ──
    import matplotlib.pyplot as plt

    T = len(hs_arr)
    t_sec = np.arange(T) * args.dt

    # Compute progress to goal (%)
    d_start = math.hypot(start_x_w - goal_x_w, start_y_w - goal_y_w)
    d_remaining = np.sqrt((np.array(xs[1:]) - goal_x_w)**2 +
                          (np.array(ys[1:]) - goal_y_w)**2)
    progress = np.clip((1.0 - d_remaining / max(d_start, 1e-6)) * 100, 0, 100)

    fig, axes = plt.subplots(3, 1, figsize=(12, 12), sharex=True)
    fig.subplots_adjust(hspace=0.3)
    x_label = "Progress to Goal (%)"

    # (1) ||u - u_ref||_2
    ax = axes[0]
    ax.plot(progress, u_diff_l2, color="#1f77b4", lw=0.8)
    ax.fill_between(progress, 0, u_diff_l2, color="#1f77b4", alpha=0.2)
    ax.set_ylabel(r"$\|u - u_{\mathrm{ref}}\|_2$")
    # no title
    ax.grid(True, alpha=0.3)

    # (2) Slack δ₁² and δ₂² (L2 / quadratic norm)
    ax = axes[1]
    ax.plot(progress, slack_arr, color="#1f77b4", lw=1, label=r"$\delta_1$")
    ax.fill_between(progress, 0, slack_arr, color="#1f77b4", alpha=0.15)
    ax.plot(progress, slack2_arr, color="#4B0082", lw=1, label=r"$\delta_2$")
    ax.fill_between(progress, 0, slack2_arr, color="#4B0082", alpha=0.15)
    ax.set_ylabel("Slack variable")
    # no title
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # (3) h_s, h_p1, h_p2 together
    ax = axes[2]
    ax.plot(progress, hs_arr, color="darkgreen", lw=1.2,
            label=r"$h_s$ safety ($h_{s,\min}$=" + f"{np.min(hs_arr):.4f})")
    hp1_pos_pct = 100.0 * np.count_nonzero(hp_arr > 0) / max(T, 1)
    ax.plot(progress, hp_arr, color="#1f77b4", lw=1,
            label=r"$h_{\pi}$ perception ($>0$: " + f"{hp1_pos_pct:.0f}%)")
    hp2_pos_pct = 100.0 * np.count_nonzero(hp2_arr > 0) / max(T, 1)
    ax.plot(progress, hp2_arr, color="#4B0082", lw=1,
            label=r"$h_{\eta}$ perception ($>0$: " + f"{hp2_pos_pct:.0f}%)")
    ax.axhline(0, color="k", ls="--", lw=0.8)
    ax.fill_between(progress, 0, hs_arr, where=(np.array(hs_arr) < 0),
                    color="#4B0082", alpha=0.3)
    ax.set_xlabel(x_label)
    ax.set_ylabel(f"Barrier value (q={args.q_slack})")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # ── Print summary metrics ──
    print(f"\n{'='*50}")
    print(f"  ||u - u_ref||_2  mean={np.mean(u_diff_l2):.4f}  max={np.max(u_diff_l2):.4f}")
    print(f"  δ₁  L1 = {np.sum(np.abs(slack_arr)) * args.dt:.4f}   "
          f"L_inf = {np.max(np.abs(slack_arr)):.4f}")
    print(f"  δ₂  L1 = {np.sum(np.abs(slack2_arr)) * args.dt:.4f}   "
          f"L_inf = {np.max(np.abs(slack2_arr)):.4f}")
    print(f"  h_s   min={np.min(hs_arr):.4f}  mean={np.mean(hs_arr):.4f}")
    print(f"  h_pi  min={np.min(hp_arr):.4f}  mean={np.mean(hp_arr):.4f}")
    print(f"  h_eta  min={np.min(hp2_arr):.4f}  mean={np.mean(hp2_arr):.4f}")
    print(f"{'='*50}")

    # no suptitle
    plt.savefig("conflict_cbf_diagnostics.png", dpi=150, bbox_inches="tight")
    print("Saved diagnostics to conflict_cbf_diagnostics.png")

    # ── Trajectory figure (separate) ──
    fig2, ax2 = plt.subplots(figsize=(10, 8))
    from matplotlib.lines import Line2D

    ax2.imshow(
        trajectory_grid_viz.T,
        extent=[x_min, x_min + nx * res, y_min, y_min + ny * res],
        origin="lower", cmap="Greys", alpha=0.4, aspect="equal",
    )
    step_slice = max(1, xy.shape[0] // 80000)
    ax2.scatter(xy[::step_slice, 0], xy[::step_slice, 1],
                c=rgb_slice[::step_slice], s=0.5, alpha=0.8)
    ax2.plot(path_xw, path_yw, "b-", lw=1.5, alpha=0.5)
    traj_palette = hp_segment_palette()
    legend_handles = [Line2D([0], [0], color="b", lw=1.5, alpha=0.5, label="Reference path")]
    legend_handles.extend(
        add_hp_colored_trajectory(
            ax2,
            xs,
            ys,
            hp_arr,
            positive_color=traj_palette["positive"],
            negative_color=traj_palette["negative"],
            lw=2,
            alpha=1.0,
            label_prefix="CBF traj",
        )
    )
    arrow_len = 0.15
    step_arrows = max(1, len(xs) // 25)
    for i in range(0, len(xs), step_arrows):
        ax2.arrow(xs[i], ys[i],
                  arrow_len * math.cos(thetas[i]),
                  arrow_len * math.sin(thetas[i]),
                  head_width=0.04, head_length=0.02, fc="k", ec="k")
    ax2.scatter([start_x_w], [start_y_w], c="g", s=100, zorder=5)
    ax2.scatter([goal_x_w], [goal_y_w], c="#4B0082", s=100, zorder=5)
    legend_handles.append(Line2D([0], [0], marker="o", color="w", markerfacecolor="g", markersize=8, label="start"))
    legend_handles.append(Line2D([0], [0], marker="o", color="w", markerfacecolor="#4B0082", markersize=8, label="goal"))
    # no title
    ax2.legend(handles=legend_handles, fontsize=8)
    ax2.set_xlim(-3, 6)
    ax2.set_ylim(-12, 0)
    ax2.set_aspect("equal")
    ax2.set_xticklabels([])
    ax2.set_yticklabels([])
    plt.savefig("conflict_cbf_trajectory.png", dpi=150, bbox_inches="tight")
    print("Saved trajectory to conflict_cbf_trajectory.png")
    _save_run_summary(args, results)
    plt.show()


def _save_run_summary(args, results, results_q2=None):
    """Save flags + metrics to a timestamped text file."""
    import datetime, sys
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    fname = f"run_summary_{ts}.txt"

    lines = []
    lines.append(f"Command: {' '.join(sys.argv)}")
    lines.append(f"Timestamp: {datetime.datetime.now().isoformat()}")
    lines.append("")
    lines.append("=== Flags ===")
    for k, v in sorted(vars(args).items()):
        lines.append(f"  {k}: {v}")
    lines.append("")

    for label, res in [("L1" if args.q_slack == 1 else "L2", results)] + \
                       ([("L2" if args.q_slack == 1 else "L1", results_q2)] if results_q2 else []):
        if res is None:
            continue
        sl1 = np.array(res["slack"])
        sl2 = np.array(res.get("slack2", [0.0]))
        hs = np.array(res["hs"])
        hp = np.array(res["hp"])
        hp2 = np.array(res.get("hp2", [0.0]))
        T = len(hs)
        u_ref_arr = np.array(res["u_ref"])
        u_safe_arr = np.array(res["u_safe"])
        u_diff_l2 = np.sqrt(np.sum((u_safe_arr - u_ref_arr)**2, axis=1))

        lines.append(f"=== {label} (q={'1' if 'L1' in label else '2'}) ===")
        lines.append(f"  Steps: {T}")
        lines.append(f"  ||u-u_ref||_2: mean={np.mean(u_diff_l2):.4f} max={np.max(u_diff_l2):.4f}")
        lines.append(f"  δ₁: L1={np.sum(np.abs(sl1))*args.dt:.4f} Linf={np.max(np.abs(sl1)):.4f} nz={100*np.mean(sl1>1e-8):.0f}%")
        lines.append(f"  δ₂: L1={np.sum(np.abs(sl2))*args.dt:.4f} Linf={np.max(np.abs(sl2)):.4f} nz={100*np.mean(sl2>1e-8):.0f}%")
        lines.append(f"  h_s: min={np.min(hs):.4f} mean={np.mean(hs):.4f}")
        lines.append(f"  h_pi: min={np.min(hp):.4f} mean={np.mean(hp):.4f} >0={100*np.mean(np.array(hp)>0):.0f}%")
        lines.append(f"  h_eta: min={np.min(hp2):.4f} mean={np.mean(hp2):.4f} >0={100*np.mean(np.array(hp2)>0):.0f}%")
        lines.append("")

    with open(fname, "w") as f:
        f.write("\n".join(lines))
    print(f"Saved run summary to {fname}")


if __name__ == "__main__":
    main()
