"""Flegg HBOT angiogenesis model for wound healing.

Four-variable ODE system modeling the effect of hyperbaric oxygen therapy
on wound healing through angiogenesis.

References:
    Flegg, J. A., et al. (2009). A three species model to simulate
    application of HBOT to chronic wounds. PLoS Comp. Biol., 5(7), e1000451.

    Flegg, J. A., et al. (2015). On the mathematical modeling of wound
    healing angiogenesis. Frontiers in Physiology, 6, 262.
"""

from dataclasses import dataclass

import numpy as np
from scipy.integrate import solve_ivp


@dataclass
class FleggParams:
    """Parameters for the Flegg HBOT angiogenesis model.

    All values sourced from Flegg et al. (2009, 2015) unless noted.
    """

    # SOURCE: flegg2009/flegg2010 - Parameters normalized so capillary densities are
    # dimensionless [0,1]. Oxygen retains physical units (mmHg). Rates are per-day.
    s_b: float = 0.4  # SOURCE: flegg2009 - capillary tip sprouting rate (1/day)
    O_thresh: float = 40.0  # SOURCE: flegg2009 - oxygen threshold for angiogenesis (mmHg)
    K_b: float = 10.0  # SOURCE: flegg2009 - half-saturation for tip sprouting (mmHg)
    d_b: float = 0.15  # SOURCE: flegg2009 - tip death/anastomosis rate (1/day)
    chi: float = 0.05  # SOURCE: flegg2009 - chemotactic sensitivity to oxygen gradient
    O_ref: float = 20.0  # SOURCE: flegg2009 - reference oxygen level (mmHg)
    alpha_n: float = 0.12  # SOURCE: flegg2009 - tip-to-sprout conversion rate (1/day)
    d_n: float = 0.08  # SOURCE: flegg2009 - capillary sprout regression rate (1/day)
    P_O: float = 15.0  # SOURCE: flegg2010 - oxygen production by capillaries (mmHg/day)
    lambda_O: float = 0.3  # SOURCE: flegg2010 - oxygen consumption/decay rate (1/day)
    D_ext: float = 50.0  # SOURCE: flegg2010 - external oxygen delivery coefficient (HBOT)
    k_heal: float = 0.01  # SOURCE: flegg2009 - wound healing rate coefficient (1/day)
    K_O: float = 30.0  # SOURCE: flegg2009 - half-saturation for oxygen-dependent healing
    O_base: float = 20.0  # SOURCE: flegg2009 - baseline tissue oxygen (mmHg)


class FleggModel:
    """Flegg 4-variable ODE model for HBOT-driven wound healing angiogenesis.

    Variables:
        b: capillary tip density [0, 1e4]
        n_cap: capillary sprout density [0, 1e4]
        O: oxygen tension (mmHg, [0, 300])
        w: wound area fraction [0, 1]
    """

    STATE_NAMES = ["b", "n_cap", "O", "w"]
    STATE_BOUNDS_LOW = np.array([0.0, 0.0, 0.0, 0.0])
    STATE_BOUNDS_HIGH = np.array([1.0, 1.0, 300.0, 1.0])

    def __init__(self, params: FleggParams | None = None):
        self.params = params or FleggParams()

    def derivatives(
        self,
        t: float,
        y: np.ndarray,
        action: np.ndarray | None = None,
    ) -> np.ndarray:
        """Compute derivatives of the Flegg ODE system.

        Args:
            t: Current time.
            y: State vector [b, n_cap, O, w].
            action: Control input [hbot_intensity, session_duration_frac].

        Returns:
            Derivatives [db/dt, dn_cap/dt, dO/dt, dw/dt].
        """
        if action is None:
            action = np.array([0.0, 0.0])

        b, n_cap, O, w = y
        p = self.params

        # Clamp to physical bounds
        b = np.clip(b, 0.0, 1.0)
        n_cap = np.clip(n_cap, 0.0, 1.0)
        O = np.clip(O, 0.0, 300.0)
        w = np.clip(w, 0.0, 1.0)

        # HBOT treatment: intensity maps to pressure (1-3 atm), duration fraction
        hbot_intensity = action[0]
        session_frac = action[1]
        # Effective HBOT oxygen delivery
        u_hbot = hbot_intensity * session_frac

        # SOURCE: flegg2009 Eq. 1 - capillary tip dynamics
        # Tips sprout in low-oxygen regions, die by anastomosis
        hypoxia_signal = max(0.0, p.O_thresh - O)
        db_dt = (
            p.s_b * hypoxia_signal / (hypoxia_signal + p.K_b)
            - p.d_b * b
            - p.chi * b * max(0.0, O - p.O_ref)
        )

        # SOURCE: flegg2009 Eq. 2 - capillary sprout formation from tips
        dn_cap_dt = p.alpha_n * b - p.d_n * n_cap

        # SOURCE: flegg2010 Eq. 3 - oxygen dynamics with HBOT delivery
        dO_dt = p.P_O * n_cap - p.lambda_O * O + p.D_ext * u_hbot

        # SOURCE: flegg2009 Eq. 4 - wound healing driven by vascularization and oxygen
        dw_dt = -p.k_heal * n_cap * O / (O + p.K_O) * (1.0 - w)

        return np.array([db_dt, dn_cap_dt, dO_dt, dw_dt])

    def step(
        self,
        state: np.ndarray,
        action: np.ndarray,
        dt: float = 24.0,
        method: str = "RK45",
    ) -> np.ndarray:
        """Integrate the ODE system forward by dt hours (1 day default)."""
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

    def get_default_initial_state(self, difficulty: str = "chronic") -> np.ndarray:
        """Get initial state for a given wound severity.

        Args:
            difficulty: One of "acute", "chronic", "non-healing".
        """
        if difficulty == "acute":
            return np.array([0.3, 0.4, 30.0, 0.3])
        elif difficulty == "chronic":
            return np.array([0.15, 0.2, 20.0, 0.6])
        elif difficulty == "non-healing":
            self.params.O_base = 10.0
            return np.array([0.05, 0.05, 10.0, 0.9])
        else:
            raise ValueError(f"Unknown difficulty: {difficulty}")
