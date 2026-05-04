"""
Conflict-Aware Control Barrier Functions for safety + active perception.

The safety barrier uses the occupancy-radius mapping 
 for each splat:
    r_i(p) = ||p - mu_i|| - R_i

The barrier itself is the soft minimum over those per-splat margins:
    hs(p) = -(1/beta) * log(sum_i exp(-beta * r_i(p)))
"""

import math
import numpy as np
from scipy import sparse
import clarabel
import cvxpy as cp

import risk_aware_eig as eig_mod

# ──────────────────────────────────────────────────────────────
# Default parameters
# ──────────────────────────────────────────────────────────────
# Safety barrier
BETA_LSE = 10.0           # log-sum-exp sharpness β
BETA2 = 1.0               # barrier scaling: hs = -(1/(β·β₂)) log(Σ exp(-β ρ_i))
ALPHA_S = 0.5             # class-K coefficient for linear safety fallback
A_MAX = 0.8               # max deceleration (m/s²) for class-K α(hs) = √(2·amax·hs)
OCCUPANCY_RADIUS_MIN = 0.01
OCCUPANCY_RADIUS_MAX = 0.08
OCCUPANCY_NEAREST_K = 300
_EPS = 1e-12

# Perception barrier
PHI_MAX_DEG = 45.0        # max angle from EIG gradient — ±45deg
ALPHA_P = 0.5             # class-K coefficient for perception CBF

# QP
K_SLACK = 0.01            # slack penalty weight
Q_SLACK = 1               # slack penalty exponent
W_TRACK = 1.0             # reference tracking weight

# Map-state dependent penalty
P_MIN = 1.0
GAMMA1_PENALTY = 0.5
GAMMA2_PENALTY = 0.3


# ──────────────────────────────────────────────────────────────
#  parameters
# ──────────────────────────────────────────────────────────────
GAMMA_S = 1.0              # safety barrier gain γ_s (, constraint 1)
GAMMA_PI = 1.0             # spatial perception barrier gain γ_π (, constraint 2)
GAMMA_ETA = 1.0            # angular perception barrier gain γ_η (, constraint 3)
TAU_ADAPTIVE = 1.0         # adaptive class-K sigmoid steepness τ
K_ETA = 0.5                # linear class-K coefficient k_η for angular barrier
W_S_SLACK = 0.01           # slack weight w_s for spatial perception
W_ETA_SLACK = 0.01         # slack weight w_η for angular perception

#  parameters (EIG-threshold formulation)
I_C_THRESHOLD = 0.1        # information threshold I_c
GAMMA_P = 1.0              # perception barrier gain γ_p
W_P_SLACK = 0.01           # slack weight w_p


def safety_class_k(hs, a_max=A_MAX):
    """
    Physically consistent class-K function:
        α(hs) = √(2 · amax · max(hs, 0))

    Links the barrier condition to braking capability.
    Clamps hs to 0 when negative (demands ḣs ≥ 0 for recovery).
    """
    return math.sqrt(2.0 * a_max * max(hs, 0.0))


def adaptive_perception_class_k(h_pi, grad_pi_I_norm, tau=TAU_ADAPTIVE):
    """
    Adaptive class-K for spatial perception barrier:
        α_π(h_π) = (1 + exp(-τ · ||∇_π I||)) · h_π

    Scales barrier decay rate with EIG gradient magnitude:
    large ||∇I|| → aggressive perception tracking,
    small ||∇I|| → gentler (factor stays near 2).
    """
    factor = 1.0 + math.exp(-tau * grad_pi_I_norm)
    # Extended class-K: no clamp. When h_pi < 0, RHS demands recovery.
    return factor * h_pi


# ──────────────────────────────────────────────────────────────
# Safety Barrier Function
# ──────────────────────────────────────────────────────────────
def _normalize_nearest_k(nearest_k, count):
    if count <= 0 or nearest_k is None:
        return None
    k = int(nearest_k)
    if k <= 0 or k >= count:
        return None
    return k


def _select_nearest_indices(point_xy, means_xy, nearest_k=None):
    means = np.asarray(means_xy, dtype=np.float64).reshape(-1, 2)
    count = means.shape[0]
    if count == 0:
        return np.zeros(0, dtype=int)

    k = _normalize_nearest_k(nearest_k, count)
    if k is None:
        return np.arange(count, dtype=int)

    query_xy = np.asarray(point_xy, dtype=np.float64).reshape(-1)[:2]
    diff = means - query_xy[np.newaxis, :]
    dist_sq = np.sum(diff * diff, axis=1)
    idx = np.argpartition(dist_sq, k - 1)[:k]
    return idx[np.argsort(dist_sq[idx])]


def stable_logsumexp(values):
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    if arr.size == 0:
        return float(-np.inf)
    vmax = float(np.max(arr))
    if not np.isfinite(vmax):
        return vmax
    shifted = arr - vmax
    return float(vmax + np.log(np.sum(np.exp(shifted))))


def compute_occupancy_radii_from_uncertainty(
    uncertainty,
    radius_min=OCCUPANCY_RADIUS_MIN,
    radius_max=OCCUPANCY_RADIUS_MAX,
    robot_radius=0.0,
    base_radius=0.0,
    use_percentile=False,
):
    """
    Map per-splat uncertainty σᵢ ∈ [0, 1] to occupancy radii Rᵢ.

    Default (use_percentile=False) — DIRECT monotone mapping:
        Rᵢ = radius_min + clip(σᵢ, 0, 1) · (radius_max - radius_min) + ...
    This is strictly monotone in σᵢ, so simulate_observation decreasing σᵢ
    strictly decreases Rᵢ. The initial Rᵢ is always ≥ current Rᵢ for every
    splat, consistent with the physical intuition that observation can only
    reduce uncertainty → reduce required safety inflation.

    Legacy (use_percentile=True) — log-percentile normalization, 
     NOT monotone in σᵢ: when
    some σⱼ decay, the percentile bounds shift and unchanged σᵢ get remapped
    to different Rᵢ. This is the source of the "CBF ball grew around me while
    I stood still" behavior. Kept for backward compatibility.
    """
    if radius_min <= 0.0 or radius_max <= 0.0:
        raise ValueError("radius_min and radius_max must be positive")
    if radius_max < radius_min:
        raise ValueError("radius_max must be >= radius_min")
    if robot_radius < 0.0 or base_radius < 0.0:
        raise ValueError("robot_radius and base_radius must be non-negative")

    uncertainty = np.asarray(uncertainty, dtype=np.float64).reshape(-1)
    if uncertainty.size == 0:
        return np.zeros(0, dtype=np.float64)

    uncertainty = np.nan_to_num(uncertainty, nan=0.0, posinf=0.0, neginf=0.0)

    if use_percentile:
        safe = np.maximum(uncertainty, _EPS)
        log_values = np.log10(safe)
        lo = float(np.percentile(log_values, 1.0))
        hi = float(np.percentile(log_values, 99.0))
        if hi <= lo:
            normalized = np.zeros_like(log_values)
        else:
            normalized = np.clip((log_values - lo) / (hi - lo), 0.0, 1.0)
    else:
        # Direct monotone mapping: σ is already normalized to [0, 1] in
        # GaussianSplatField.from_mat, so use it directly.
        normalized = np.clip(uncertainty, 0.0, 1.0)

    mapped = radius_min + normalized * max(0.0, radius_max - radius_min)
    return mapped + float(robot_radius) + float(base_radius)


class SafetyBarrier:
    """
    Per-splat occupancy soft-min safety barrier.
    """

    def __init__(self, gsplat,
                 beta=BETA_LSE, beta2=BETA2,
                 occupancy_radius_min=OCCUPANCY_RADIUS_MIN,
                 occupancy_radius_max=OCCUPANCY_RADIUS_MAX,
                 occupancy_nearest_k=OCCUPANCY_NEAREST_K,
                 occupancy_robot_radius=0.0,
                 occupancy_base_radius=0.0,
                 freeze_sigma=False,
                 adaptive_bias_target=None,
                 beta_min=10.0,
                 beta_max=10000.0):
        """
        freeze_sigma=False (default): Rᵢ is recomputed every call 
            gsplat.sigma, so as simulate_observation decays σᵢ, the percentile
            normalization shifts and unobserved splats' Rᵢ can change.
        freeze_sigma=True: Rᵢ computed ONCE from gsplat.sigma_initial (or
            gsplat.sigma at construction time). The barrier is static in
            (π, σ)-space; only π changes. This matches the theorem's assumption
            of a static safe set C_s.

        adaptive_bias_target=None (default): Use fixed `beta` for the LSE.
        adaptive_bias_target=B (e.g. 0.005 m): Adapt β per-call so the LSE
            soft-min bias log(M_active)/β stays at the target B regardless
            of how many splats M are active in the local query. This pins
            the worst-case h_s bias to ±B m, independent of cluster density.
            Bounded by [beta_min, beta_max] for numerical safety.
        """
        self.gsplat = gsplat
        self.beta = beta
        self.beta2 = beta2
        self.occupancy_radius_min = occupancy_radius_min
        self.occupancy_radius_max = occupancy_radius_max
        self.occupancy_nearest_k = occupancy_nearest_k
        self.occupancy_robot_radius = occupancy_robot_radius
        self.occupancy_base_radius = occupancy_base_radius
        self.freeze_sigma = freeze_sigma
        self.adaptive_bias_target = adaptive_bias_target
        self.beta_min = beta_min
        self.beta_max = beta_max
        # Diagnostic: last β used (mostly for adaptive mode)
        self.last_beta_used = beta
        # Precompute frozen radii once — uses sigma_initial if present else sigma
        if freeze_sigma:
            sigma_ref = getattr(gsplat, 'sigma_initial', gsplat.sigma)
            self._frozen_radii = compute_occupancy_radii_from_uncertainty(
                sigma_ref,
                radius_min=occupancy_radius_min,
                radius_max=occupancy_radius_max,
                robot_radius=occupancy_robot_radius,
                base_radius=occupancy_base_radius,
            )
        else:
            self._frozen_radii = None

    def _occupancy_terms(self, p):
        idx = _select_nearest_indices(
            p[:2], self.gsplat.means_xy, nearest_k=self.occupancy_nearest_k
        )
        if len(idx) == 0:
            return idx, np.zeros(0, dtype=np.float64), np.zeros((0, 2), dtype=np.float64), np.zeros(0, dtype=np.float64)

        if self._frozen_radii is not None:
            radii = self._frozen_radii
        else:
            radii = compute_occupancy_radii_from_uncertainty(
                self.gsplat.sigma,
                radius_min=self.occupancy_radius_min,
                radius_max=self.occupancy_radius_max,
                robot_radius=self.occupancy_robot_radius,
                base_radius=self.occupancy_base_radius,
            )
        mu_near = self.gsplat.means_xy[idx]
        diff = p[:2] - mu_near
        dists = np.linalg.norm(diff, axis=1)
        rho = dists - radii[idx]
        return idx, rho, diff, dists

    def evaluate(self, p):
        """
        hs(p) = -(1/β) log(Σ exp(-β ρ_i))

        With adaptive_bias_target = B set, β is chosen per-call so the LSE
        bias log(M)/β ≈ B regardless of cluster density:
            β_eff = clip( log(max(M, 2)) / B,  beta_min,  beta_max )
        """
        idx, rho, diff, dists = self._occupancy_terms(p)
        if len(idx) == 0:
            return 10.0, np.zeros(2)

        # Choose effective β
        if self.adaptive_bias_target is not None and self.adaptive_bias_target > 0:
            M = max(len(idx), 2)
            beta_eff = math.log(M) / float(self.adaptive_bias_target)
            beta_eff = float(np.clip(beta_eff, self.beta_min, self.beta_max))
        else:
            beta_eff = self.beta
        self.last_beta_used = beta_eff

        dists_safe = np.maximum(dists, 1e-8)
        logits = -beta_eff * rho
        lse = stable_logsumexp(logits)
        hs = (-lse / beta_eff) / self.beta2
        grad_rho = diff / dists_safe[:, np.newaxis]
        weights = np.exp(logits - lse)
        grad_hs = np.sum(weights[:, np.newaxis] * grad_rho, axis=0) / self.beta2

        return float(hs), grad_hs

    def lie_derivatives(self, x, dynamics, lookahead_m=0.0,
                        horizon_s=0.0, n_horizon_points=1,
                        v_predict=None, beta_horizon=10.0):
        """
        Lie derivatives for unicycle (relative degree 1) with optional
        kinematic lookahead and predictive time-horizon lookahead.

        Three modes:

        A) Single-point lookahead (backward compatible, horizon_s=0):
           h_s evaluated at π_la = π + lookahead_m · c
           Lg_hs = [∇h_s · c,  lookahead_m · ∇h_s · c⊥]

        B) Time-horizon lookahead (horizon_s > 0):
           Predict N+1 points along the forward arc assuming constant heading
           at speed v_predict over T = horizon_s seconds:
               π_i = π + τ_i · v_predict · c,   τ_i = i·T/N, i = 0..N
           Enforce a soft-min (log-sum-exp) of h_s across all points:
               h_s_eff = −(1/β_h) log Σ exp(−β_h · h_s_i)  ≤ min_i h_s_i
           Lie derivative chains through each point's motion:
               dπ_i/dt = v·c + τ_i·v_predict·ω·c⊥
           So  Lg_hs_eff = [Σ w_i · ⟨∇h_s_i, c⟩,
                            Σ w_i · τ_i·v_predict · ⟨∇h_s_i, c⊥⟩]
           where w_i ∝ exp(−β_h · h_s_i) (soft-min weights).

        Args:
            lookahead_m: single-point lookahead distance (m); ignored if horizon_s > 0
            horizon_s: prediction horizon (s); 0 disables horizon mode
            n_horizon_points: number of sample points along horizon (N)
            v_predict: predicted forward speed (m/s); defaults to dynamics.v_max
            beta_horizon: soft-min sharpness β_h (larger → closer to hard min)

        Returns: (hs_eff, Lf=0, Lg (2,))
        """
        # Mode B: time-horizon predictive lookahead
        if horizon_s > 0.0 and n_horizon_points >= 1:
            if v_predict is None:
                v_predict = float(getattr(dynamics, 'v_max', 0.8))
            theta = x[2]
            ct, st = math.cos(theta), math.sin(theta)
            N = int(n_horizon_points)

            # Sample times τ_0=0, τ_1=T/N, ..., τ_N=T
            taus = np.linspace(0.0, horizon_s, N + 1)

            hs_list = []
            grad_list = []
            for tau in taus:
                Ltau = tau * v_predict
                p_tau = np.array([x[0] + Ltau * ct, x[1] + Ltau * st])
                hs_i, grad_i = self.evaluate(p_tau)
                hs_list.append(hs_i)
                grad_list.append(grad_i)
            hs_arr = np.asarray(hs_list)
            grad_arr = np.asarray(grad_list)   # shape (N+1, 2)

            # Soft-min: h_s_eff = -(1/β_h) log(Σ exp(-β_h · h_s_i))
            logits = -beta_horizon * hs_arr
            lse = stable_logsumexp(logits)
            hs_eff = float(-lse / beta_horizon)
            weights = np.exp(logits - lse)   # shape (N+1,), sums to 1

            # Lie derivative (chain rule):
            #   dπ_i/dt  = v·c + τ_i·v_predict·ω·c⊥
            #   dh_i/dt  = ⟨∇h_i, c⟩·v + τ_i·v_predict·⟨∇h_i, c⊥⟩·ω
            # L_g h_s_eff = Σ w_i · [⟨∇h_i,c⟩, τ_i·v_predict·⟨∇h_i,c⊥⟩]
            dot_c     = grad_arr[:, 0] * ct + grad_arr[:, 1] * st          # (N+1,)
            dot_cperp = -grad_arr[:, 0] * st + grad_arr[:, 1] * ct         # (N+1,)
            Lg_v = float(np.sum(weights * dot_c))
            Lg_w = float(np.sum(weights * (taus * v_predict) * dot_cperp))
            return hs_eff, 0.0, np.array([Lg_v, Lg_w])

        # Mode A: single-point lookahead (existing behavior)
        if lookahead_m <= 0.0:
            hs, grad_hs = self.evaluate(x[:2])
            ct, st = math.cos(x[2]), math.sin(x[2])
            Lg_hs = np.array([grad_hs[0] * ct + grad_hs[1] * st, 0.0])
            return hs, 0.0, Lg_hs

        theta = x[2]
        ct, st = math.cos(theta), math.sin(theta)
        p_la = np.array([x[0] + lookahead_m * ct, x[1] + lookahead_m * st])
        hs_la, grad_hs_la = self.evaluate(p_la)
        Lg_v = float(grad_hs_la[0] * ct + grad_hs_la[1] * st)
        Lg_w = float(lookahead_m * (grad_hs_la[0] * (-st) + grad_hs_la[1] * ct))
        return hs_la, 0.0, np.array([Lg_v, Lg_w])


# ──────────────────────────────────────────────────────────────
# Perception Barrier Function
# ──────────────────────────────────────────────────────────────
class PerceptionBarrier:
    """
    Perception barrier:
        hp(p, c) = <c, n(p)> - cos(phi_max)

    Constrains camera heading to stay within phi_max of the EIG gradient.
    """

    def __init__(self, phi_max_deg=PHI_MAX_DEG):
        self.phi_max = math.radians(phi_max_deg)
        self.cos_phi_max = math.cos(self.phi_max)

    def evaluate(self, x, n_p):
        """
        Evaluate hp.

        x: (3,) state [px, py, theta]
        n_p: (2,) normalized EIG gradient direction
        Returns scalar.
        """
        c = np.array([np.cos(x[2]), np.sin(x[2])])
        return float(np.dot(c, n_p) - self.cos_phi_max)

    def lie_derivatives(self, x, n_p):
        """
        Lie derivatives for perception barrier under unicycle.

        dhp/dt = v * W1 + omega * W2

        W1 = c^T @ Jn_p @ c             (translation effect)
        W2 = (Jc)^T @ n(p)              (rotation effect)
           = n2*cos(theta) - n1*sin(theta)

        Returns:
            hp, Lf_hp (=0), Lg_hp (2,)
        """
        theta = x[2]
        c = np.array([np.cos(theta), np.sin(theta)])
        Jc = np.array([-np.sin(theta), np.cos(theta)])

        hp = np.dot(c, n_p) - self.cos_phi_max

        # W2: rotation effect (Jc^T n)
        W2 = float(np.dot(Jc, n_p))

        Lf_hp = 0.0
        Lg_hp = np.array([0.0, W2])

        return float(hp), Lf_hp, Lg_hp


# ──────────────────────────────────────────────────────────────
# Angular Perception Barrier
# ──────────────────────────────────────────────────────────────
class AngularPerceptionBarrier:
    """
    Angular perception barrier:
        h_η(η) = ⟨η, d_η|_{(π₀,η₀)}⟩

    where η = [cos θ, sin θ] is the heading direction and d_η is the
    normalized angular information-ascent direction derived from dI/dθ.

    Class-K function (linear):
        α_η(h_η) = k_η · h_η

    Lie derivatives under unicycle (, no drift):
        ḣ_η = ω · ⟨η⊥, d_η⟩
    where η⊥ = [-sin θ, cos θ].
    """

    def __init__(self, k_eta=K_ETA):
        self.k_eta = k_eta

    def compute_d_eta(self, x, gsplat):
        """
        Compute the angular info-ascent direction d_η.

        d_η = sign(dI/dθ) · η⊥, normalized.
        Points in the heading-rotation direction that increases EIG.

        Returns: (d_eta (2,), dI_dtheta_raw scalar)
        """
        theta = x[2]
        eta_perp = np.array([-np.sin(theta), np.cos(theta)])

        dI_dtheta, I_max = eig_mod.eig_heading_gradient(x[0], x[1], theta, gsplat)

        if abs(dI_dtheta) < 1e-10:
            d_eta = eta_perp
        else:
            d_eta = np.sign(dI_dtheta) * eta_perp

        return d_eta, dI_dtheta

    def evaluate(self, x, d_eta):
        """
        h_η(η) = ⟨η, d_η⟩
        """
        eta = np.array([np.cos(x[2]), np.sin(x[2])])
        return float(np.dot(eta, d_eta))

    def class_k(self, h_eta):
        """
        Extended class-K: α_η(h_η) = k_η · h_η, valid on all of R.
        No clamp — when h_η < 0, RHS -γ_η·α_η becomes positive, forcing recovery.
        """
        return self.k_eta * h_eta

    def lie_derivatives(self, x, d_eta):
        """
        Lie derivatives for angular perception barrier under unicycle.

        ḣ_η = ω · ⟨η⊥, d_η⟩

        Lf_h_η = 0  (no drift)
        Lg_h_η = [0, ⟨η⊥, d_η⟩]  (only ω affects heading)

        Returns: h_eta, Lf_h_eta (=0), Lg_h_eta (2,)
        """
        theta = x[2]
        eta = np.array([np.cos(theta), np.sin(theta)])
        eta_perp = np.array([-np.sin(theta), np.cos(theta)])

        h_eta = float(np.dot(eta, d_eta))
        W_eta = float(np.dot(eta_perp, d_eta))

        return h_eta, 0.0, np.array([0.0, W_eta])


# ──────────────────────────────────────────────────────────────
# Map-State Dependent Penalty
# ──────────────────────────────────────────────────────────────
class MapStatePenalty:
    """
    Adaptive slack penalty:
        P(x) = max(Pmin, exp(gamma1/dg + gamma2 * U))
    """

    def __init__(self, goal_xy, gsplat, p_min=P_MIN,
                 gamma1=GAMMA1_PENALTY, gamma2=GAMMA2_PENALTY):
        self.goal_xy = np.asarray(goal_xy, dtype=np.float64)
        self.gsplat = gsplat
        self.p_min = p_min
        self.gamma1 = gamma1
        self.gamma2 = gamma2

    def evaluate(self, x):
        d_goal = max(np.linalg.norm(x[:2] - self.goal_xy), 0.1)
        U = float(np.mean(self.gsplat.sigma))
        P = np.exp(self.gamma1 / d_goal + self.gamma2 * U)
        return max(self.p_min, float(P))


# ──────────────────────────────────────────────────────────────
# Conflict-Aware CBF Controller (QP, )
# ──────────────────────────────────────────────────────────────
class ConflictAwareCBFController:
    """
    Solve the conflict-aware QP at each timestep:

        min_{u, δ₁, δ₂}  ||u - u_ref||² + w₁·δ₁^q + w₂·δ₂^q
        s.t.  ḣs ≥ -αs·hs                          (safety, hard)
              ḣp ≥ -αp·hp - δ₁                      (perception heading, soft)
              ⟨∇_η I, η⊥⟩ · ω ≥ -δ₂                (EIG heading gradient, soft)
              u_min ≤ u ≤ u_max
              δ₁, δ₂ ≥ 0
    """

    def __init__(self, gsplat, dynamics,
                 alpha_s=ALPHA_S, alpha_p=ALPHA_P, a_max=A_MAX,
                 k_slack=K_SLACK, w2_slack=None, q_slack=Q_SLACK,
                 beta_lse=BETA_LSE,
                 phi_max_deg=PHI_MAX_DEG,
                 occupancy_radius_min=OCCUPANCY_RADIUS_MIN,
                 occupancy_radius_max=OCCUPANCY_RADIUS_MAX,
                 occupancy_nearest_k=OCCUPANCY_NEAREST_K,
                 occupancy_robot_radius=0.0,
                 occupancy_base_radius=0.0,
                 freeze_sigma=False,
                 adaptive_bias_target=None,
                 safety_lookahead_m=0.0,
                 safety_horizon_s=0.0,
                 safety_horizon_n=4,
                 w_track=None,
                 goal_xy=None):

        self.w_track = w_track if w_track is not None else W_TRACK
        self.dynamics = dynamics
        self.gsplat = gsplat  # for EIG/perception
        self.safety_lookahead_m = safety_lookahead_m
        self.safety_horizon_s = safety_horizon_s
        self.safety_horizon_n = safety_horizon_n

        self.safety = SafetyBarrier(
            gsplat,
            beta=beta_lse,
            occupancy_radius_min=occupancy_radius_min,
            occupancy_radius_max=occupancy_radius_max,
            occupancy_nearest_k=occupancy_nearest_k,
            occupancy_robot_radius=occupancy_robot_radius,
            occupancy_base_radius=occupancy_base_radius,
            freeze_sigma=freeze_sigma,
            adaptive_bias_target=adaptive_bias_target,
        )
        self.perception = PerceptionBarrier(phi_max_deg=phi_max_deg)
        self.penalty = MapStatePenalty(goal_xy, gsplat) if goal_xy is not None else None

        self.alpha_s = alpha_s
        self.alpha_p = alpha_p
        self.a_max = a_max
        self.k_slack = k_slack
        self.w2_slack = w2_slack if w2_slack is not None else k_slack
        self.q_slack = q_slack

        # Diagnostics
        self.last_hs = None
        self.last_hp = None
        self.last_slack = None
        self.last_slack2 = None
        self.last_dI_dtheta = None
        self.last_n_p = None
        self.solver_success = True

    def set_path_mask(self, ref_xy, mask_radius=1.0):
        """
        Store the reference path for forward-only EIG masking.
        At each solve(), only Gaussians AHEAD of the robot on the path contribute to EIG.
        """
        self._ref_xy = np.asarray(ref_xy, dtype=np.float64)
        self._mask_radius = mask_radius
        # Precompute arc lengths for fast forward queries
        diffs = np.diff(self._ref_xy, axis=0)
        seg_lens = np.sqrt(np.sum(diffs ** 2, axis=1))
        self._ref_arc = np.concatenate([[0.0], np.cumsum(seg_lens)])
        # Pre-mask: Gaussians within mask_radius of ANY path point (superset)
        mu_xy = self.gsplat.means_xy
        mask = np.zeros(self.gsplat.N, dtype=bool)
        r2 = mask_radius ** 2
        for i in range(self._ref_xy.shape[0] - 1):
            a, b = self._ref_xy[i], self._ref_xy[i + 1]
            ab = b - a
            denom = np.dot(ab, ab) + 1e-12
            ap = mu_xy - a
            t = np.clip(np.dot(ap, ab) / denom, 0.0, 1.0)
            proj = a + t[:, np.newaxis] * ab
            d2 = np.sum((mu_xy - proj) ** 2, axis=1)
            mask |= (d2 <= r2)
        self._full_mask_idx = np.where(mask)[0]

    def solve(self, x, u_ref):
        """
        Solve the conflict-aware QP (, First Formulation).

        min_{u, δ₁, δ₂}  (wt/2)||u - u_ref||² + w1_eff·δ₁^q + w2_eff·δ₂^q
        s.t.  ḣs ≥ -αs·hs                          (safety, hard)
              ḣp ≥ -αp·hp - δ₁                      (perception heading, soft)
              ⟨∇_η I, η⊥⟩ · ω ≥ -δ₂                (EIG heading gradient, soft)
              u_min ≤ u ≤ u_max
              δ₁, δ₂ ≥ 0

        q=1: L1 sparse relaxation (linear penalty on slacks)
        q=2: L2 quadratic relaxation (quadratic penalty on slacks)

        Returns: (2,) optimal control [v, omega]
        """
        # --- Safety barrier (with optional kinematic lookahead) ---
        hs, Lf_hs, Lg_hs = self.safety.lie_derivatives(
            x, self.dynamics,
            lookahead_m=self.safety_lookahead_m,
            horizon_s=self.safety_horizon_s,
            n_horizon_points=self.safety_horizon_n,
        )
        self.last_hs = hs

        # --- EIG direction and perception barrier (δ₁) ---
        n_p = eig_mod.information_ascent_direction(x[0], x[1], x[2], self.gsplat)
        self.last_n_p = n_p

        hp, Lf_hp, Lg_hp = self.perception.lie_derivatives(x, n_p)
        self.last_hp = hp

        # --- EIG heading gradient (δ₂): dI/dtheta = <nabla_eta I, eta_perp> ---
        # Normalized by I_max and clamped to [-1, 1] for QP conditioning.
        # Plot-time scaling is separate (divides recorded trace by its own absmax).
        dI_dtheta_raw, I_max = eig_mod.eig_heading_gradient(x[0], x[1], x[2], self.gsplat)
        dI_dtheta_normalized = float(dI_dtheta_raw / max(I_max, 1e-6))
        dI_dtheta = float(np.clip(dI_dtheta_normalized, -1.0, 1.0))
        self.last_dI_dtheta = dI_dtheta           # clamped, used by QP
        self.last_dI_dtheta_raw = dI_dtheta_normalized  # unclamped, for plotting

        # --- Adaptive penalty ---
        penalty_scale = 1.0
        if self.penalty is not None:
            penalty_scale = self.penalty.evaluate(x)
        w1_eff = self.k_slack * penalty_scale
        w2_eff = self.w2_slack * penalty_scale

        # --- Build QP ---
        # Decision variable: z = [v, omega, δ₁, δ₂]  (dim 4)
        u_min, u_max = self.dynamics.control_bounds()
        wt = self.w_track

        # Objective: (wt/2)||u - u_ref||² + w1_eff·δ₁^q + w2_eff·δ₂^q
        if self.q_slack == 2:
            P_mat = np.diag([wt, wt, w1_eff, w2_eff])
            q_vec = np.array([-wt * u_ref[0], -wt * u_ref[1], 0.0, 0.0])
        else:
            P_mat = np.diag([wt, wt, 1e-6, 1e-6])
            q_vec = np.array([-wt * u_ref[0], -wt * u_ref[1], w1_eff, w2_eff])

        # Constraints: A_our z >= b_our
        rows_A = []
        rows_b = []

        # (1) Safety (hard): Lg_hs @ u >= -α(hs) where α(hs) = √(2·amax·hs)
        rows_A.append([Lg_hs[0], Lg_hs[1], 0.0, 0.0])
        rows_b.append(-safety_class_k(hs, self.a_max))

        # (2) Perception heading (soft): Lg_hp @ u + δ₁ >= -αp·hp
        rows_A.append([Lg_hp[0], Lg_hp[1], 1.0, 0.0])
        rows_b.append(-self.alpha_p * hp)

        # (3) EIG heading gradient (soft): dI/dθ · ω + δ₂ >= 0
        rows_A.append([0.0, dI_dtheta, 0.0, 1.0])
        rows_b.append(0.0)

        # (4-5) Slack non-negativity: δ₁ >= 0, δ₂ >= 0
        rows_A.append([0.0, 0.0, 1.0, 0.0])
        rows_b.append(0.0)
        rows_A.append([0.0, 0.0, 0.0, 1.0])
        rows_b.append(0.0)

        # (6-9) Control bounds
        rows_A.append([1.0, 0.0, 0.0, 0.0])
        rows_b.append(u_min[0])
        rows_A.append([-1.0, 0.0, 0.0, 0.0])
        rows_b.append(-u_max[0])
        rows_A.append([0.0, 1.0, 0.0, 0.0])
        rows_b.append(u_min[1])
        rows_A.append([0.0, -1.0, 0.0, 0.0])
        rows_b.append(-u_max[1])

        A_our = np.array(rows_A)
        b_our = np.array(rows_b)
        n_constraints = A_our.shape[0]

        A_clar = sparse.csc_matrix(-A_our)
        b_clar = -b_our
        P_clar = sparse.csc_matrix(P_mat)

        settings = clarabel.DefaultSettings()
        settings.verbose = False

        solver = clarabel.DefaultSolver(
            P_clar, q_vec,
            A_clar, b_clar,
            [clarabel.NonnegativeConeT(n_constraints)],
            settings
        )
        sol = solver.solve()

        if str(sol.status) == 'Solved':
            self.solver_success = True
            z_opt = np.array(sol.x)
            u_opt = z_opt[:2]
            self.last_slack = float(z_opt[2])
            self.last_slack2 = float(z_opt[3])
        else:
            self.solver_success = False
            self.last_slack = 0.0
            self.last_slack2 = 0.0
            u_opt = np.clip(u_ref, u_min, u_max)

        # Enforce minimum angular velocity (non-convex, applied post-QP)
        u_opt = self.dynamics.enforce_omega_min(u_opt)

        return u_opt


# ──────────────────────────────────────────────────────────────
#  Controller (CVXPY-based, exact formulation)
# ──────────────────────────────────────────────────────────────
class Equation2Controller:
    """
    Exact implementation of Paper  using CVXPY:

        min_{u, δ_s, δ_η}  (1/2)||u - u_ref||² + w_s·δ_s^q + w_η·δ_η^q

        s.t.  ḣ_s  ≥ -γ_s · α_s(h_s)                        [HARD safety]
              ḣ_π  ≥ -γ_π · α_π(h_π) - δ_s                   [SOFT spatial perception]
              ḣ_η  ≥ -γ_η · α_η(h_η) - δ_η                   [SOFT angular perception]
              u_min ≤ u ≤ u_max
              δ_s, δ_η ≥ 0

    Math-to-code mapping:
    ┌────────────────────────────┬──────────────────────────────────────────┐
    │ Paper Symbol               │ Code                                     │
    ├────────────────────────────┼──────────────────────────────────────────┤
    │ u = [v, ω]                 │ u_var (cp.Variable(2))                   │
    │ δ_s                        │ delta_s (cp.Variable, nonneg)            │
    │ δ_η                        │ delta_eta (cp.Variable, nonneg)          │
    │ h_s               │ SafetyBarrier.evaluate()                 │
    │ α_s(h_s) = √(2·a_max·h_s) │ safety_class_k()                         │
    │ γ_s                        │ self.gamma_s                             │
    │ h_π               │ PerceptionBarrier.evaluate()             │
    │ α_π (, adaptive)     │ adaptive_perception_class_k()            │
    │ γ_π                        │ self.gamma_pi                            │
    │ ||∇_π I|| (for )     │ eig_gradient_and_magnitude()[1]          │
    │ h_η               │ AngularPerceptionBarrier.evaluate()      │
    │ α_η = k_η·h_η             │ AngularPerceptionBarrier.class_k()       │
    │ γ_η                        │ self.gamma_eta                           │
    │ ḣ_s, ḣ_π, ḣ_η    │ .lie_derivatives() on each barrier       │
    │ w_s, w_η                   │ self.w_s_slack, self.w_eta_slack         │
    │ q ∈ {1,2}                  │ self.q_slack                             │
    └────────────────────────────┴──────────────────────────────────────────┘
    """

    def __init__(self, gsplat, dynamics,
                 # Safety barrier
                 a_max=A_MAX,
                 gamma_s=GAMMA_S,
                 # Spatial perception
                 gamma_pi=GAMMA_PI,
                 tau_adaptive=TAU_ADAPTIVE,
                 phi_max_deg=PHI_MAX_DEG,
                 # Angular perception
                 gamma_eta=GAMMA_ETA,
                 k_eta=K_ETA,
                 # QP slack
                 w_s_slack=W_S_SLACK,
                 w_eta_slack=W_ETA_SLACK,
                 q_slack=Q_SLACK,
                 # SafetyBarrier params
                 beta_lse=BETA_LSE,
                 occupancy_radius_min=OCCUPANCY_RADIUS_MIN,
                 occupancy_radius_max=OCCUPANCY_RADIUS_MAX,
                 occupancy_nearest_k=OCCUPANCY_NEAREST_K,
                 occupancy_robot_radius=0.0,
                 occupancy_base_radius=0.0,
                 freeze_sigma=False,
                 adaptive_bias_target=None,
                 # Safety lookahead
                 safety_lookahead_m=0.0,
                 safety_horizon_s=0.0,
                 safety_horizon_n=4,
                 # Map-state penalty
                 goal_xy=None):

        self.dynamics = dynamics
        self.gsplat = gsplat
        self.safety_lookahead_m = safety_lookahead_m
        self.safety_horizon_s = safety_horizon_s
        self.safety_horizon_n = safety_horizon_n

        # Reuse existing SafetyBarrier
        self.safety = SafetyBarrier(
            gsplat, beta=beta_lse,
            occupancy_radius_min=occupancy_radius_min,
            occupancy_radius_max=occupancy_radius_max,
            occupancy_nearest_k=occupancy_nearest_k,
            occupancy_robot_radius=occupancy_robot_radius,
            occupancy_base_radius=occupancy_base_radius,
            freeze_sigma=freeze_sigma,
            adaptive_bias_target=adaptive_bias_target,
        )
        # Reuse existing PerceptionBarrier
        self.spatial_perception = PerceptionBarrier(phi_max_deg=phi_max_deg)
        # New: Angular perception barrier
        self.angular_perception = AngularPerceptionBarrier(k_eta=k_eta)

        # Barrier gains
        self.gamma_s = gamma_s
        self.gamma_pi = gamma_pi
        self.gamma_eta = gamma_eta
        self.a_max = a_max
        self.tau_adaptive = tau_adaptive

        # QP weights
        self.w_s_slack = w_s_slack
        self.w_eta_slack = w_eta_slack
        self.q_slack = q_slack

        # Optional map-state penalty
        self.penalty = MapStatePenalty(goal_xy, gsplat) if goal_xy is not None else None

        # Diagnostics
        self.last_hs = None
        self.last_h_pi = None
        self.last_h_eta = None
        self.last_delta_s = None
        self.last_delta_eta = None
        self.last_n_p = None
        self.last_d_eta = None
        self.last_grad_pi_norm = None
        self.solver_success = True

        # Build parametric QP once — avoids CVXPY canonicalization on every solve.
        self._build_parametric_qp()

    def _build_parametric_qp(self):
        """Build the  QP once using cp.Parameter for time-varying terms."""
        u_min, u_max = self.dynamics.control_bounds()

        # Parameters
        self._p_u_ref = cp.Parameter(2)
        self._p_Lg_hs = cp.Parameter(2)
        self._p_Lg_hpi = cp.Parameter(2)
        self._p_Lg_heta = cp.Parameter(2)
        self._p_rhs_s = cp.Parameter()       # -γ_s · α_s(h_s)
        self._p_rhs_pi = cp.Parameter()      # -γ_π · α_π(h_π)
        self._p_rhs_eta = cp.Parameter()     # -γ_η · α_η(h_η)
        self._p_w_s = cp.Parameter(nonneg=True)
        self._p_w_eta = cp.Parameter(nonneg=True)

        # Variables
        self._u_var = cp.Variable(2)
        self._delta_s = cp.Variable(nonneg=True)
        self._delta_eta = cp.Variable(nonneg=True)

        # Objective
        if self.q_slack == 2:
            objective = cp.Minimize(
                0.5 * cp.sum_squares(self._u_var - self._p_u_ref)
                + self._p_w_s * cp.square(self._delta_s)
                + self._p_w_eta * cp.square(self._delta_eta)
            )
        else:
            objective = cp.Minimize(
                0.5 * cp.sum_squares(self._u_var - self._p_u_ref)
                + self._p_w_s * self._delta_s
                + self._p_w_eta * self._delta_eta
            )

        constraints = [
            self._p_Lg_hs @ self._u_var >= self._p_rhs_s,
            self._p_Lg_hpi @ self._u_var + self._delta_s >= self._p_rhs_pi,
            self._p_Lg_heta @ self._u_var + self._delta_eta >= self._p_rhs_eta,
            self._u_var >= u_min,
            self._u_var <= u_max,
        ]
        self._prob = cp.Problem(objective, constraints)

    def solve(self, x, u_ref):
        """
        Solve  via parametric CVXPY problem (fast, compiled once).

        Returns: (2,) optimal control [v, omega]
        """
        u_min, u_max = self.dynamics.control_bounds()

        # ── 1. Safety barrier with optional kinematic lookahead ──
        hs, Lf_hs, Lg_hs = self.safety.lie_derivatives(
            x, self.dynamics,
            lookahead_m=self.safety_lookahead_m,
            horizon_s=self.safety_horizon_s,
            n_horizon_points=self.safety_horizon_n,
        )
        self.last_hs = hs
        alpha_s_val = safety_class_k(hs, self.a_max)

        # ── 2. Spatial perception barrier ──
        # Single call for both direction and magnitude (saves 4 EIG evals)
        _, grad_pi_norm, n_p = eig_mod.eig_gradient_and_magnitude(
            x[0], x[1], x[2], self.gsplat
        )
        self.last_n_p = n_p
        self.last_grad_pi_norm = grad_pi_norm

        h_pi, Lf_h_pi, Lg_h_pi = self.spatial_perception.lie_derivatives(x, n_p)
        self.last_h_pi = h_pi
        alpha_pi_val = adaptive_perception_class_k(
            h_pi, grad_pi_norm, tau=self.tau_adaptive
        )

        # ── 3. Angular perception barrier ──
        d_eta, dI_dtheta = self.angular_perception.compute_d_eta(x, self.gsplat)
        self.last_d_eta = d_eta

        h_eta, Lf_h_eta, Lg_h_eta = self.angular_perception.lie_derivatives(x, d_eta)
        self.last_h_eta = h_eta
        alpha_eta_val = self.angular_perception.class_k(h_eta)

        # ── 4. Adaptive penalty scaling (optional, ) ──
        penalty_scale = 1.0
        if self.penalty is not None:
            penalty_scale = self.penalty.evaluate(x)
        w_s_eff = self.w_s_slack * penalty_scale
        w_eta_eff = self.w_eta_slack * penalty_scale

        # ── 5. Update parameters and solve ──
        self._p_u_ref.value = np.asarray(u_ref, dtype=float).reshape(2)
        self._p_Lg_hs.value = np.asarray(Lg_hs, dtype=float).reshape(2)
        self._p_Lg_hpi.value = np.asarray(Lg_h_pi, dtype=float).reshape(2)
        self._p_Lg_heta.value = np.asarray(Lg_h_eta, dtype=float).reshape(2)
        self._p_rhs_s.value = float(-self.gamma_s * alpha_s_val)
        self._p_rhs_pi.value = float(-self.gamma_pi * alpha_pi_val)
        self._p_rhs_eta.value = float(-self.gamma_eta * alpha_eta_val)
        self._p_w_s.value = float(w_s_eff)
        self._p_w_eta.value = float(w_eta_eff)

        self._prob.solve(solver=cp.CLARABEL, verbose=False)

        if self._prob.status == 'optimal':
            self.solver_success = True
            u_opt = np.array(self._u_var.value).flatten()
            self.last_delta_s = float(self._delta_s.value)
            self.last_delta_eta = float(self._delta_eta.value)
        else:
            self.solver_success = False
            self.last_delta_s = 0.0
            self.last_delta_eta = 0.0
            u_opt = np.clip(u_ref, u_min, u_max)

        # Compatibility aliases for logging
        self.last_hp = self.last_h_pi
        self.last_slack = self.last_delta_s
        self.last_slack2 = self.last_delta_eta
        self.last_dI_dtheta_raw = None  #  doesn't expose the raw heading gradient

        u_opt = self.dynamics.enforce_omega_min(u_opt)
        return u_opt

    def set_path_mask(self, ref_xy, mask_radius=1.0):
        """No-op stub for interface compatibility."""
        pass



# ──────────────────────────────────────────────────────────────
# EIG-Threshold Perception Barrier
# ──────────────────────────────────────────────────────────────
class EIGThresholdBarrier:
    """
    EIG-threshold perception barrier:
        h_p(π, η) = I(π, η) - I_c

    Aggregates spatial and rotational perception into one scalar.
    The robot must maintain EIG above the threshold I_c.

    Lie derivative under unicycle:
        ḣ_p = ⟨∇_π I, π̇⟩ + ⟨∇_η I, η̇⟩
            = v·(∂I/∂x·cosθ + ∂I/∂y·sinθ) + ω·(dI/dθ)

    So L_g h_p = [∂I/∂x·cosθ + ∂I/∂y·sinθ,  dI/dθ]
    """

    def __init__(self, I_c=I_C_THRESHOLD, heading_smooth=0.85,
                 eig_fov_deg=180.0):
        """
        eig_fov_deg: FOV (degrees) used for the EIG barrier evaluation and
            its Lie derivative. This is SEPARATE from the camera/observation
            FOV (--fov-deg) used by simulate_observation. A wider EIG FOV
            captures info along the trajectory direction, not just directly
            ahead. Both h_p and L_g h_p use this consistently.

            180° = forward hemisphere (recommended — trajectory-focused)
            120° = wider than camera but still forward-biased
            360° = omnidirectional (equivalent to old --use-omni-gradient)

        heading_smooth: EMA factor for the ω-component of L_g h_p.
            0.0 = no smoothing (raw dI/dθ, may chatter)
            0.85 = smooth (commits to a direction, default)
        """
        self.I_c = I_c
        self.heading_smooth = heading_smooth
        self.eig_fov_deg = eig_fov_deg
        self._dI_dtheta_ema = 0.0

    def evaluate(self, x, gsplat):
        """
        h_p = I(π, η) - I_c

        Uses the EIG FOV (self.eig_fov_deg), which is typically wider than
        the camera FOV. This captures uncertainty along the trajectory
        direction, not just directly ahead.
        """
        I_val = eig_mod.eig_at_pose(x[0], x[1], x[2], gsplat,
                                     fov_deg=self.eig_fov_deg)
        return float(I_val - self.I_c), float(I_val)

    def lie_derivatives(self, x, gsplat):
        """
        Lie derivatives for h_p under unicycle.

        h_p uses omni EIG (fov=360°) — sees all nearby uncertainty.
        Lie derivative: ḣ_p = v·(∇_π I · η) + ω·(dI/dθ)

        Both components use the SAME eig_fov_deg as h_p for consistency.
        The wider EIG FOV gives meaningful gradients in both v and ω:
          v-component: ∂I/∂x·cosθ + ∂I/∂y·sinθ  (move toward info)
          ω-component: dI/dθ                      (turn toward info)

        Returns: h_p, I_val, Lf_hp (=0), Lg_hp (2,)
        """
        theta = x[2]
        ct, st = np.cos(theta), np.sin(theta)
        fov = self.eig_fov_deg

        # h_p value — uses EIG FOV (wider than camera, captures trajectory info)
        I_val = eig_mod.eig_at_pose(x[0], x[1], theta, gsplat, fov_deg=fov)
        h_p = float(I_val - self.I_c)

        # Lie derivative — same EIG FOV for consistency
        # v-component: spatial gradient of I projected onto heading
        I_xp = eig_mod.eig_at_pose(x[0] + 0.05, x[1], theta, gsplat, fov_deg=fov)
        I_xm = eig_mod.eig_at_pose(x[0] - 0.05, x[1], theta, gsplat, fov_deg=fov)
        I_yp = eig_mod.eig_at_pose(x[0], x[1] + 0.05, theta, gsplat, fov_deg=fov)
        I_ym = eig_mod.eig_at_pose(x[0], x[1] - 0.05, theta, gsplat, fov_deg=fov)
        dI_dx = (I_xp - I_xm) / 0.1
        dI_dy = (I_yp - I_ym) / 0.1
        Lg_hp_v = float(dI_dx * ct + dI_dy * st)

        # ω-component: heading gradient dI/dθ
        delta_th = 0.01
        I_tp = eig_mod.eig_at_pose(x[0], x[1], theta + delta_th, gsplat, fov_deg=fov)
        I_tm = eig_mod.eig_at_pose(x[0], x[1], theta - delta_th, gsplat, fov_deg=fov)
        omega_signal_raw = float((I_tp - I_tm) / (2.0 * delta_th))

        # EMA smoothing on ω-component to prevent chattering
        alpha = 1.0 - self.heading_smooth
        self._dI_dtheta_ema = (alpha * omega_signal_raw
                               + self.heading_smooth * self._dI_dtheta_ema)

        Lg_hp = np.array([Lg_hp_v, float(self._dI_dtheta_ema)])

        return h_p, float(I_val), 0.0, Lg_hp


# ──────────────────────────────────────────────────────────────
#  Controller (EIG-threshold, single perception CBF)
# ──────────────────────────────────────────────────────────────
class Equation32Controller:
    """
    EIG-threshold formulation (Paper ):

        min_{u, δ}  (1/2)||u - u_ref||² + w_p·δ^q

        s.t.  L_g h_s · u ≥ -γ_s · α_s(h_s)              [HARD safety]
              L_g h_p · u ≥ -γ_p · α_p(h_p) - δ            [SOFT perception]
              u_min ≤ u ≤ u_max
              δ ≥ 0

    where h_p = I(π,η) - I_c replaces both h_π and h_η
    with a single EIG threshold constraint.

    Math-to-code mapping:
    ┌──────────────────────────┬──────────────────────────────────────┐
    │ Paper Symbol             │ Code                                 │
    ├──────────────────────────┼──────────────────────────────────────┤
    │ h_s             │ SafetyBarrier.lie_derivatives()      │
    │ α_s             │ safety_class_k()                     │
    │ h_p = I - I_c   │ EIGThresholdBarrier.evaluate()      │
    │ ḣ_p          │ EIGThresholdBarrier.lie_derivatives()│
    │ α_p (extended class-K)   │ gamma_p * alpha_p * h_p    │
    │ γ_s, γ_p                 │ self.gamma_s, self.gamma_p           │
    │ w_p                      │ self.w_p_slack                       │
    │ I_c                      │ self.eig_threshold.I_c               │
    │ δ                        │ delta (cp.Variable, nonneg)          │
    └──────────────────────────┴──────────────────────────────────────┘
    """

    def __init__(self, gsplat, dynamics,
                 # Safety
                 a_max=A_MAX,
                 gamma_s=GAMMA_S,
                 # Perception threshold
                 I_c=I_C_THRESHOLD,
                 eig_fov_deg=180.0,
                 heading_smooth=0.85,
                 gamma_p=GAMMA_P,
                 alpha_p=ALPHA_P,
                 # QP
                 w_p_slack=W_P_SLACK,
                 w_track=1.0,
                 q_slack=Q_SLACK,
                 # SafetyBarrier params
                 beta_lse=BETA_LSE,
                 occupancy_radius_min=OCCUPANCY_RADIUS_MIN,
                 occupancy_radius_max=OCCUPANCY_RADIUS_MAX,
                 occupancy_nearest_k=OCCUPANCY_NEAREST_K,
                 occupancy_robot_radius=0.0,
                 occupancy_base_radius=0.0,
                 freeze_sigma=False,
                 adaptive_bias_target=None,
                 # Safety lookahead
                 safety_lookahead_m=0.0,
                 safety_horizon_s=0.0,
                 safety_horizon_n=4,
                 # Map-state penalty
                 goal_xy=None):

        self.dynamics = dynamics
        self.gsplat = gsplat

        self.safety = SafetyBarrier(
            gsplat, beta=beta_lse,
            occupancy_radius_min=occupancy_radius_min,
            occupancy_radius_max=occupancy_radius_max,
            occupancy_nearest_k=occupancy_nearest_k,
            occupancy_robot_radius=occupancy_robot_radius,
            occupancy_base_radius=occupancy_base_radius,
            freeze_sigma=freeze_sigma,
            adaptive_bias_target=adaptive_bias_target,
        )
        self.eig_threshold = EIGThresholdBarrier(
            I_c=I_c, heading_smooth=heading_smooth,
            eig_fov_deg=eig_fov_deg,
        )

        self.gamma_s = gamma_s
        self.gamma_p = gamma_p
        self.alpha_p = alpha_p
        self.a_max = a_max
        self.safety_lookahead_m = safety_lookahead_m
        self.safety_horizon_s = safety_horizon_s
        self.safety_horizon_n = safety_horizon_n

        self.w_p_slack = w_p_slack
        self.w_track = w_track
        self.q_slack = q_slack

        self.penalty = MapStatePenalty(goal_xy, gsplat) if goal_xy is not None else None

        # Diagnostics
        self.last_hs = None
        self.last_hp = None
        self.last_I_val = None
        self.last_delta = None
        self.solver_success = True

        # Build parametric QP once — avoids CVXPY canonicalization on every solve.
        self._build_parametric_qp()

    def _build_parametric_qp(self):
        """
        Build the  QP once using cp.Parameter for the time-varying
        quantities (u_ref, Lie derivatives, class-K values, penalty weight).
        Only parameter values are updated at each solve, avoiding ~2-3 ms
        of canonicalization overhead per step.
        """
        u_min, u_max = self.dynamics.control_bounds()

        # Parameters (change every solve)
        self._p_u_ref = cp.Parameter(2)
        self._p_Lg_hs = cp.Parameter(2)
        self._p_Lg_hp = cp.Parameter(2)
        self._p_rhs_s = cp.Parameter()      # -γ_s · α_s(h_s)
        self._p_rhs_p = cp.Parameter()      # -γ_p · α_p(h_p)
        self._p_w_p = cp.Parameter(nonneg=True)  # effective slack weight

        # Variables
        self._u_var = cp.Variable(2)
        self._delta = cp.Variable(nonneg=True)

        # Objective: (w_track/2)||u - u_ref||² + w_p·δ^q
        wt = self.w_track
        if self.q_slack == 2:
            objective = cp.Minimize(
                (wt / 2.0) * cp.sum_squares(self._u_var - self._p_u_ref)
                + self._p_w_p * cp.square(self._delta)
            )
        else:
            objective = cp.Minimize(
                (wt / 2.0) * cp.sum_squares(self._u_var - self._p_u_ref)
                + self._p_w_p * self._delta
            )

        constraints = [
            #  line 1 (HARD): L_g h_s · u ≥ -γ_s · α_s(h_s)
            self._p_Lg_hs @ self._u_var >= self._p_rhs_s,
            #  line 2 (SOFT): L_g h_p · u + δ ≥ -γ_p · α_p(h_p)
            self._p_Lg_hp @ self._u_var + self._delta >= self._p_rhs_p,
            # Control bounds (constant — no parameter needed)
            self._u_var >= u_min,
            self._u_var <= u_max,
        ]

        self._prob = cp.Problem(objective, constraints)

    def solve(self, x, u_ref):
        """
        Solve  via parametric CVXPY problem (fast, compiled once).

        Returns: (2,) optimal control [v, omega]
        """
        u_min, u_max = self.dynamics.control_bounds()

        # ── Safety barrier ( + 18) with optional kinematic lookahead ──
        hs, Lf_hs, Lg_hs = self.safety.lie_derivatives(
            x, self.dynamics,
            lookahead_m=self.safety_lookahead_m,
            horizon_s=self.safety_horizon_s,
            n_horizon_points=self.safety_horizon_n,
        )
        self.last_hs = hs
        alpha_s_val = safety_class_k(hs, self.a_max)

        # ── EIG-threshold perception barrier ( + 28-30) ──
        hp, I_val, Lf_hp, Lg_hp = self.eig_threshold.lie_derivatives(x, self.gsplat)
        self.last_hp = hp
        self.last_I_val = I_val
        # Extended class-K: α_p(h_p) = α_p · h_p, valid on all of R.
        # When h_p < 0, α_p·h_p < 0, so RHS -γ_p·α_p(h_p) becomes positive —
        # an active recovery demand that forces EIG to grow. NO clamping to 0.
        alpha_p_val = self.alpha_p * hp

        # ── Adaptive penalty (optional) ──
        penalty_scale = 1.0
        if self.penalty is not None:
            penalty_scale = self.penalty.evaluate(x)
        w_p_eff = self.w_p_slack * penalty_scale

        # ── Update parameters and solve ──
        self._p_u_ref.value = np.asarray(u_ref, dtype=float).reshape(2)
        self._p_Lg_hs.value = np.asarray(Lg_hs, dtype=float).reshape(2)
        self._p_Lg_hp.value = np.asarray(Lg_hp, dtype=float).reshape(2)
        self._p_rhs_s.value = float(-self.gamma_s * alpha_s_val)
        self._p_rhs_p.value = float(-self.gamma_p * alpha_p_val)
        self._p_w_p.value = float(w_p_eff)

        self._prob.solve(solver=cp.CLARABEL, verbose=False)

        if self._prob.status == 'optimal':
            self.solver_success = True
            u_opt = np.array(self._u_var.value).flatten()
            self.last_delta = float(self._delta.value)
        else:
            self.solver_success = False
            self.last_delta = 0.0
            u_opt = np.clip(u_ref, u_min, u_max)

        # Compatibility aliases for logging
        self.last_slack = self.last_delta
        self.last_slack2 = 0.0
        self.last_n_p = None
        self.last_dI_dtheta_raw = None

        u_opt = self.dynamics.enforce_omega_min(u_opt)
        return u_opt

    def set_path_mask(self, ref_xy, mask_radius=1.0):
        """No-op stub for interface compatibility."""
        pass

