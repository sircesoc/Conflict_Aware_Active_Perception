"""
Unicycle dynamics for conflict-aware CBF control.

State: x = [px, py, theta]   (position and heading)
Control: u = [v, omega]      (linear velocity, angular velocity)

Dynamics (control-affine, no drift):
    dx/dt = f(x) + g(x) u
    f(x) = [0, 0, 0]
    g(x) = [[cos(theta), 0],
             [sin(theta), 0],
             [0,          1]]

Both safety and perception barriers have relative degree 1 under these dynamics.
"""

import math
import numpy as np


def wrap_angle(a):
    """Wrap angle to [-pi, pi]."""
    return (a + math.pi) % (2 * math.pi) - math.pi


class UnicycleDynamics:
    """Continuous-time unicycle: dx/dt = g(x) u, no drift."""

    def __init__(self, v_min=0.01, v_max=0.8, omega_min=0.0, omega_max=math.radians(85)):
        self.v_min = v_min   # always forward, never zero or reverse
        self.v_max = v_max
        self.omega_min = omega_min  # minimum |omega| — enforced post-QP
        self.omega_max = omega_max
        self.n_states = 3    # px, py, theta
        self.n_controls = 2  # v, omega
        self.rel_deg = 1     # Both barriers have relative degree 1

    def f(self, x):
        """Drift vector (zero for unicycle). Returns (3,) array."""
        return np.zeros(3)

    def g(self, x):
        """Control matrix. Returns (3, 2) array.
        g(x) = [[cos(theta), 0], [sin(theta), 0], [0, 1]]
        """
        theta = x[2]
        return np.array([
            [np.cos(theta), 0.0],
            [np.sin(theta), 0.0],
            [0.0,           1.0]
        ])

    def heading_vector(self, x):
        """Camera heading direction c = [cos(theta), sin(theta)]."""
        return np.array([np.cos(x[2]), np.sin(x[2])])

    def Jc(self, x):
        """Derivative of heading vector w.r.t. theta: dc/dtheta = [-sin(theta), cos(theta)]."""
        return np.array([-np.sin(x[2]), np.cos(x[2])])

    def integrate(self, x, u, dt):
        """Forward Euler integration. Returns new state (3,)."""
        theta = x[2]
        v, omega = u[0], u[1]
        x_new = np.array([
            x[0] + dt * v * np.cos(theta),
            x[1] + dt * v * np.sin(theta),
            wrap_angle(x[2] + dt * omega)
        ])
        return x_new

    def control_bounds(self):
        """Returns (u_min, u_max) arrays for [v, omega]."""
        return (
            np.array([self.v_min, -self.omega_max]),
            np.array([self.v_max,  self.omega_max])
        )

    def enforce_omega_min(self, u):
        """Snap |omega| to omega_min if below threshold."""
        if self.omega_min > 0.0 and abs(u[1]) < self.omega_min:
            u = u.copy()
            u[1] = self.omega_min * (1.0 if u[1] >= 0 else -1.0)
        return u
