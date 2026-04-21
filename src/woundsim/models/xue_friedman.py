"""Simplified Xue-Friedman ischemic wound ODE model.

Six-variable ODE system simplified from the 9-variable Xue & Friedman (2009)
PNAS model of ischemic cutaneous wound healing.

Reference:
    Xue, C., Friedman, A., & Sen, C. K. (2009). A mathematical model of
    ischemic cutaneous wounds. PNAS, 106(39), 16782-16787.
"""

from dataclasses import dataclass

import numpy as np
from scipy.integrate import solve_ivp


@dataclass
class XueFriedmanParams:
    """Parameters for the simplified Xue-Friedman model.

    All values sourced from Xue, Friedman & Sen (2009) unless noted.
    """

    k_close: float = 0.05  # SOURCE: xue2009 - wound closure rate coefficient
    K_F: float = 1e5  # SOURCE: xue2009 - half-saturation for fibroblast-driven closure
    k_open: float = 0.02  # SOURCE: xue2009 - ischemic wound opening rate
    O_crit: float = 40.0  # SOURCE: xue2009 - critical oxygen tension (mmHg)
    alpha_O: float = 0.1  # SOURCE: xue2009 - oxygen supply rate from vasculature
    O_blood: float = 80.0  # SOURCE: xue2009 - blood oxygen tension (mmHg)
    beta_O: float = 1e-6  # SOURCE: xue2009 - oxygen consumption by macrophages
    gamma_O: float = 2e-6  # SOURCE: xue2009 - oxygen consumption by fibroblasts
    D_O: float = 5.0  # SOURCE: xue2009 - oxygen diffusion rate from intact tissue
    s_V: float = 0.5  # SOURCE: xue2009 - VEGF production rate by macrophages
    d_V: float = 0.1  # SOURCE: xue2009 - VEGF degradation rate
    s_M: float = 1e4  # SOURCE: xue2009 - macrophage recruitment rate
    K_V: float = 10.0  # SOURCE: xue2009 - half-saturation for VEGF-driven recruitment
    d_M: float = 0.02  # SOURCE: xue2009 - macrophage death rate
    s_F: float = 5e3  # SOURCE: xue2009 - fibroblast recruitment rate
    K_V2: float = 15.0  # SOURCE: xue2009 - half-saturation for fibroblast VEGF response
    K_O: float = 20.0  # SOURCE: xue2009 - half-saturation for oxygen-dependent fibroblasts
    d_F: float = 0.01  # SOURCE: xue2009 - fibroblast death rate
    s_E: float = 1e-5  # SOURCE: xue2009 - ECM production rate by fibroblasts
    d_E: float = 0.005  # SOURCE: xue2009 - ECM remodeling rate


class XueFriedmanModel:
    """Simplified Xue-Friedman 6-variable ODE model for ischemic wound healing.

    Variables:
        w: wound area fraction [0, 1]
        O: oxygen tension (mmHg, [0, 100])
        V: VEGF concentration (ng/mL, [0, 100])
        M: macrophage density [0, 1e6]
        F: fibroblast density [0, 1e6]
        E: ECM density [0, 1]
    """

    STATE_NAMES = ["w", "O", "V", "M", "F", "E"]
    STATE_BOUNDS_LOW = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    STATE_BOUNDS_HIGH = np.array([1.0, 100.0, 100.0, 1e6, 1e6, 1.0])

    def __init__(self, params: XueFriedmanParams | None = None):
        self.params = params or XueFriedmanParams()

    def derivatives(
        self,
        t: float,
        y: np.ndarray,
        action: np.ndarray | None = None,
    ) -> np.ndarray:
        """Compute derivatives of the simplified Xue-Friedman ODE system.

        Args:
            t: Current time.
            y: State vector [w, O, V, M, F, E].
            action: Control input [ischemia_treatment, growth_factor].

        Returns:
            Derivatives [dw/dt, dO/dt, dV/dt, dM/dt, dF/dt, dE/dt].
        """
        if action is None:
            action = np.array([0.0, 0.0])

        w, O, V, M, F, E = y
        p = self.params
        u_ischemia, u_gf = action[0], action[1]

        # Clamp to physical bounds
        w = np.clip(w, 0.0, 1.0)
        O = np.clip(O, 0.0, 100.0)
        V = np.clip(V, 0.0, 100.0)
        M = np.clip(M, 0.0, 1e6)
        F = np.clip(F, 0.0, 1e6)
        E = np.clip(E, 0.0, 1.0)

        # Effective oxygen supply - treatment improves ischemia
        alpha_eff = p.alpha_O * (1.0 + u_ischemia)

        # SOURCE: xue2009 - wound closure driven by ECM and fibroblasts, opened by ischemia
        ischemia_factor = max(0.0, 1.0 - O / p.O_crit)
        dw_dt = -p.k_close * E * F / (F + p.K_F) + p.k_open * ischemia_factor

        # SOURCE: xue2009 - oxygen dynamics
        dO_dt = (
            alpha_eff * (p.O_blood - O)
            - p.beta_O * M
            - p.gamma_O * F
            + p.D_O * (1.0 - w)
        )

        # SOURCE: xue2009 - VEGF production by hypoxic macrophages + exogenous growth factor
        dV_dt = p.s_V * M * ischemia_factor - p.d_V * V + u_gf * 5.0

        # SOURCE: xue2009 - macrophage chemotaxis toward VEGF
        dM_dt = p.s_M * V / (V + p.K_V) - p.d_M * M

        # SOURCE: xue2009 - fibroblast recruitment (VEGF + oxygen dependent)
        dF_dt = p.s_F * V / (V + p.K_V2) * O / (O + p.K_O) - p.d_F * F

        # SOURCE: xue2009 - ECM deposition with logistic saturation
        dE_dt = p.s_E * F - p.d_E * E * (1.0 - E)

        return np.array([dw_dt, dO_dt, dV_dt, dM_dt, dF_dt, dE_dt])

    def step(
        self,
        state: np.ndarray,
        action: np.ndarray,
        dt: float = 12.0,
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
        """Get initial state for a given ischemia severity.

        Args:
            difficulty: One of "mild", "moderate", "severe".
        """
        base = np.array([0.5, 30.0, 5.0, 5e4, 1e4, 0.05])
        if difficulty == "mild":
            base[0] = 0.3  # smaller wound
            base[1] = 50.0  # better oxygenation
            self.params.alpha_O = 0.5
        elif difficulty == "moderate":
            base[0] = 0.5
            base[1] = 30.0
            self.params.alpha_O = 0.3
        elif difficulty == "severe":
            base[0] = 0.8
            base[1] = 15.0
            self.params.alpha_O = 0.1
        else:
            raise ValueError(f"Unknown difficulty: {difficulty}")
        return base
