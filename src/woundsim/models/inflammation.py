"""Shared inflammation dynamics for diabetic wound healing.

Extended macrophage model incorporating glucose-insulin dynamics and their
effect on wound healing in diabetic patients.

References:
    Waugh, H. V., & Sherratt, J. A. (2006). Macrophage dynamics in diabetic
    wound dealing. Bulletin of Mathematical Biology, 68(1), 197-207.

    Zlobina, K., & Gomez, M. (2022). Optimal control of macrophage
    polarization in wound healing. J. Theor. Biol., 537, 111013.
"""

from dataclasses import dataclass

import numpy as np
from scipy.integrate import solve_ivp


@dataclass
class InflammationParams:
    """Parameters for the extended diabetic wound inflammation model."""

    # Wound dynamics
    k_wound: float = 0.004  # SOURCE: waugh2006 - wound closure rate
    k_debris: float = 0.05  # SOURCE: zlobina2022 - debris generation from damage

    # Macrophage dynamics (adapted from Zlobina, normalized densities)
    gamma_a: float = 0.08  # SOURCE: zlobina2022 - debris clearance rate (1/day, norm.)
    s_m: float = 0.6  # SOURCE: zlobina2022 - M1 recruitment rate (norm./day)
    K_a: float = 0.3  # SOURCE: zlobina2022 - half-saturation for debris
    mu_m: float = 0.1  # SOURCE: zlobina2022 - macrophage death rate (1/day)
    delta_base: float = 0.25  # SOURCE: zlobina2022 - base M1-to-M2 polarization rate (1/day)

    # Glucose-insulin dynamics
    G_target: float = 100.0  # SOURCE: waugh2006 - target glucose (mg/dL)
    k_insulin: float = 0.05  # SOURCE: waugh2006 - insulin sensitivity
    k_glucose_clear: float = 0.01  # SOURCE: waugh2006 - glucose clearance rate
    G_prod: float = 2.0  # SOURCE: waugh2006 - hepatic glucose production (mg/dL/hr)
    I_half: float = 50.0  # SOURCE: waugh2006 - insulin half-saturation

    # Glucose impairment on wound healing
    G_impair: float = 180.0  # SOURCE: waugh2006 - glucose level above which healing impaired
    k_impair: float = 0.5  # SOURCE: waugh2006 - impairment steepness

    # ECM dynamics (normalized)
    s_E: float = 0.04  # SOURCE: xue2009 - ECM production rate (1/day)
    d_E: float = 0.05  # SOURCE: xue2009 - ECM remodeling rate (1/day)
    K_m2_ecm: float = 0.3  # SOURCE: zlobina2022 - half-sat for M2-driven ECM

    # Growth factor effect
    gf_boost: float = 0.3  # topical growth factor M2 recruitment boost (norm./day)


class InflammationModel:
    """Extended 7-variable ODE model for diabetic wound healing.

    Variables:
        w: wound area [0, 1]
        a: debris/damage [0, 1]
        m1: M1 macrophages [0, 1e6]
        m2: M2 macrophages [0, 1e6]
        G: glucose level (mg/dL, [70, 400])
        I: insulin level (mU/L, [0, 200])
        E: ECM density [0, 1]
    """

    STATE_NAMES = ["w", "a", "m1", "m2", "G", "I", "E"]
    STATE_BOUNDS_LOW = np.array([0.0, 0.0, 0.0, 0.0, 70.0, 0.0, 0.0])
    STATE_BOUNDS_HIGH = np.array([1.0, 1.0, 1.0, 1.0, 400.0, 200.0, 1.0])

    def __init__(self, params: InflammationParams | None = None):
        self.params = params or InflammationParams()

    def glucose_impairment(self, G: float) -> float:
        """Compute glucose-dependent impairment factor [0, 1].

        Returns 1.0 (no impairment) at normal glucose, decreasing toward 0
        at high glucose levels.
        """
        p = self.params
        if p.G_target >= G:
            return 1.0
        excess = (G - p.G_target) / (p.G_impair - p.G_target)
        return 1.0 / (1.0 + p.k_impair * excess * excess)

    def derivatives(
        self,
        t: float,
        y: np.ndarray,
        action: np.ndarray | None = None,
    ) -> np.ndarray:
        """Compute derivatives of the diabetic wound ODE system.

        Args:
            t: Current time.
            y: State vector [w, a, m1, m2, G, I, E].
            action: Control [polarization_signal, growth_factor_dose, insulin_dose].

        Returns:
            Derivatives vector.
        """
        if action is None:
            action = np.array([0.0, 0.0, 0.0])

        w, a, m1, m2, G, I, E = y
        p = self.params
        u_pol, u_gf, u_insulin = action[0], action[1], action[2]

        # Clamp
        w = np.clip(w, 0.0, 1.0)
        a = np.clip(a, 0.0, 1.0)
        m1 = np.clip(m1, 0.0, 1.0)
        m2 = np.clip(m2, 0.0, 1.0)
        G = np.clip(G, 70.0, 400.0)
        I = np.clip(I, 0.0, 200.0)
        E = np.clip(E, 0.0, 1.0)

        # Glucose impairment factor
        gi = self.glucose_impairment(G)

        # Effective polarization rate (impaired by high glucose)
        # SOURCE: waugh2006 - hyperglycemia impairs M1-to-M2 transition
        delta_eff = p.delta_base * gi

        # SOURCE: zlobina2022 adapted - wound area decreases with ECM, increases with debris
        dw_dt = -p.k_wound * E * gi + p.k_debris * a * (1.0 - gi)

        # SOURCE: zlobina2022 - debris clearance (impaired by glucose)
        da_dt = -p.gamma_a * a * (m1 + m2) * gi + p.k_debris * w * (1.0 - gi)

        # SOURCE: zlobina2022 - M1 macrophage dynamics
        dm1_dt = p.s_m * a / (a + p.K_a) - p.mu_m * m1 - delta_eff * u_pol * m1

        # SOURCE: zlobina2022 + waugh2006 - M2 dynamics with growth factor boost
        dm2_dt = (
            delta_eff * u_pol * m1
            - p.mu_m * m2
            + u_gf * p.gf_boost
        )

        # SOURCE: waugh2006 - glucose dynamics with insulin control
        dG_dt = (
            p.G_prod
            - p.k_glucose_clear * G
            - p.k_insulin * I * G / (G + p.G_target)
        )

        # Insulin dynamics (exogenous input + clearance)
        dI_dt = u_insulin * 100.0 - 0.1 * I

        # SOURCE: xue2009 adapted - ECM formation driven by M2, impaired by glucose
        dE_dt = p.s_E * m2 / (m2 + p.K_m2_ecm) * gi - p.d_E * E * (1.0 - E)

        return np.array([dw_dt, da_dt, dm1_dt, dm2_dt, dG_dt, dI_dt, dE_dt])

    def step(
        self,
        state: np.ndarray,
        action: np.ndarray,
        dt: float = 8.0,
        method: str = "RK45",
    ) -> np.ndarray:
        """Integrate the ODE system forward by dt hours."""
        sol = solve_ivp(
            fun=lambda t, y: self.derivatives(t, y, action),
            t_span=(0, dt),
            y0=state,
            method=method,
            rtol=1e-8,
            atol=1e-10,
        )
        new_state = sol.y[:, -1]
        new_state = np.clip(new_state, self.STATE_BOUNDS_LOW, self.STATE_BOUNDS_HIGH)
        return new_state

    def get_default_initial_state(self, difficulty: str = "moderate") -> np.ndarray:
        """Get initial state for a given diabetic severity.

        Args:
            difficulty: One of "well-controlled", "moderate", "uncontrolled".
        """
        if difficulty == "well-controlled":
            return np.array([0.4, 0.5, 0.2, 0.02, 120.0, 50.0, 0.05])
        elif difficulty == "moderate":
            return np.array([0.5, 0.6, 0.15, 0.01, 200.0, 30.0, 0.02])
        elif difficulty == "uncontrolled":
            return np.array([0.7, 0.8, 0.1, 0.005, 350.0, 10.0, 0.01])
        else:
            raise ValueError(f"Unknown difficulty: {difficulty}")
