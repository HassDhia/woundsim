"""Zlobina macrophage polarization ODE model for wound healing.

Five-variable ODE system modeling macrophage M1/M2 polarization dynamics
and tissue regeneration during wound healing.

Reference:
    Zlobina, K., & Gomez, M. (2022). Optimal control of macrophage
    polarization in wound healing. Journal of Theoretical Biology, 537, 111013.
"""

from dataclasses import dataclass

import numpy as np
from scipy.integrate import solve_ivp


@dataclass
class ZlobinaParams:
    """Parameters for the Zlobina macrophage polarization model.

    All values sourced from Zlobina & Gomez (2022) unless noted.
    """

    # SOURCE: zlobina2022 - Parameters scaled to dimensionless time units where
    # one unit equals ~1 day, consistent with the non-dimensionalized form in
    # Zlobina & Gomez (2022) Table 1. Macrophage densities normalized to
    # carrying capacity K_M = 1e5 cells/mm^3.
    gamma_a: float = 0.003  # SOURCE: zlobina2022 - debris clearance rate (1/day, norm.)
    s_m: float = 0.8  # SOURCE: zlobina2022 - M1 recruitment rate (norm. density/day)
    K_a: float = 0.3  # SOURCE: zlobina2022 - half-saturation for debris-driven recruitment
    mu_m: float = 0.1  # SOURCE: zlobina2022 - macrophage natural death rate (1/day)
    delta: float = 0.3  # SOURCE: zlobina2022 - M1-to-M2 polarization rate (1/day)
    s_m2: float = 0.4  # SOURCE: zlobina2022 - M2 recruitment by granulation tissue
    K_c: float = 0.3  # SOURCE: zlobina2022 - half-saturation for granulation-driven M2
    s_c: float = 0.06  # SOURCE: zlobina2022 - granulation tissue formation rate (1/day)
    K_m2: float = 0.5  # SOURCE: zlobina2022 - half-saturation for M2-driven granulation
    mu_c: float = 0.08  # SOURCE: zlobina2022 - granulation tissue remodeling rate (1/day)
    s_n: float = 0.015  # SOURCE: zlobina2022 - permanent tissue formation rate (1/day)
    mu_n: float = 0.02  # SOURCE: zlobina2022 - tissue logistic growth modulation


@dataclass
class ZlobinaState:
    """State vector for the Zlobina model."""

    a: float = 0.6  # wound debris/damage [0, 1]
    m1: float = 0.2  # M1 macrophage density (normalized to K_M) [0, 1]
    m2: float = 0.02  # M2 macrophage density (normalized to K_M) [0, 1]
    c: float = 0.0  # granulation tissue [0, 1]
    n: float = 0.0  # permanent new tissue [0, 1]

    def to_array(self) -> np.ndarray:
        return np.array([self.a, self.m1, self.m2, self.c, self.n], dtype=np.float64)

    @classmethod
    def from_array(cls, arr: np.ndarray) -> "ZlobinaState":
        return cls(a=arr[0], m1=arr[1], m2=arr[2], c=arr[3], n=arr[4])


class ZlobinaModel:
    """Zlobina 5-variable ODE model for macrophage polarization in wound healing.

    Variables:
        a: wound debris/damage (dimensionless, [0, 1])
        m1: pro-inflammatory M1 macrophage density (cells/mm^3, [0, 1e6])
        m2: anti-inflammatory M2 macrophage density (cells/mm^3, [0, 1e6])
        c: temporary tissue / granulation (dimensionless, [0, 1])
        n: permanent new tissue (dimensionless, [0, 1])
    """

    STATE_NAMES = ["a", "m1", "m2", "c", "n"]
    STATE_BOUNDS_LOW = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
    STATE_BOUNDS_HIGH = np.array([1.0, 1.0, 1.0, 1.0, 1.0])

    def __init__(self, params: ZlobinaParams | None = None):
        self.params = params or ZlobinaParams()

    def derivatives(self, t: float, y: np.ndarray, u: float = 0.0) -> np.ndarray:
        """Compute derivatives of the Zlobina ODE system.

        Args:
            t: Current time (not used, autonomous system).
            y: State vector [a, m1, m2, c, n].
            u: Control input - polarization treatment signal [0, 1].

        Returns:
            Derivatives [da/dt, dm1/dt, dm2/dt, dc/dt, dn/dt].
        """
        a, m1, m2, c, n = y
        p = self.params

        a = np.clip(a, 0.0, 1.0)
        m1 = np.clip(m1, 0.0, 1.0)
        m2 = np.clip(m2, 0.0, 1.0)
        c = np.clip(c, 0.0, 1.0)
        n = np.clip(n, 0.0, 1.0)

        # SOURCE: zlobina2022 Eq. 1 - debris clearance
        da_dt = -p.gamma_a * a * (m1 + m2)

        # SOURCE: zlobina2022 Eq. 2 - M1 dynamics
        dm1_dt = p.s_m * a / (a + p.K_a) - p.mu_m * m1 - p.delta * u * m1

        # SOURCE: zlobina2022 Eq. 3 - M2 dynamics (polarization + granulation recruitment)
        dm2_dt = p.delta * u * m1 - p.mu_m * m2 + p.s_m2 * c / (c + p.K_c)

        # SOURCE: zlobina2022 Eq. 4 - granulation tissue
        dc_dt = p.s_c * m2 / (m2 + p.K_m2) - p.mu_c * c * n

        # SOURCE: zlobina2022 Eq. 5 - permanent tissue with logistic growth
        dn_dt = p.s_n * c - p.mu_n * n * (1.0 - n)

        return np.array([da_dt, dm1_dt, dm2_dt, dc_dt, dn_dt])

    def step(
        self,
        state: np.ndarray,
        u: float,
        dt: float = 6.0,
        method: str = "RK45",
    ) -> np.ndarray:
        """Integrate the ODE system forward by dt hours.

        Args:
            state: Current state [a, m1, m2, c, n].
            u: Control input [0, 1].
            dt: Time step in hours.
            method: Integration method for solve_ivp.

        Returns:
            New state after integration.
        """
        sol = solve_ivp(
            fun=lambda t, y: self.derivatives(t, y, u),
            t_span=(0, dt),
            y0=state,
            method=method,
            rtol=1e-8,
            atol=1e-10,
        )
        new_state = sol.y[:, -1]
        # Enforce physical bounds
        new_state = np.clip(new_state, self.STATE_BOUNDS_LOW, self.STATE_BOUNDS_HIGH)
        return new_state

    def get_default_initial_state(self, difficulty: str = "medium") -> np.ndarray:
        """Get initial state for a given difficulty tier.

        Args:
            difficulty: One of "easy", "medium", "hard".

        Returns:
            Initial state vector.
        """
        if difficulty == "easy":
            return np.array([0.3, 0.2, 0.02, 0.0, 0.0])
        elif difficulty == "medium":
            return np.array([0.6, 0.15, 0.01, 0.0, 0.0])
        elif difficulty == "hard":
            return np.array([0.9, 0.1, 0.005, 0.0, 0.0])
        else:
            raise ValueError(f"Unknown difficulty: {difficulty}")
