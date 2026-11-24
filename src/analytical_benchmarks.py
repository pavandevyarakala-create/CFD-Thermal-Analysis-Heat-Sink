"""Analytical benchmarks and classical heat transfer correlations for laminar channel flow."""

import numpy as np
from typing import Tuple


class AnalyticalBenchmarks:
    """Provides exact closed-form solutions and benchmark correlations for verification."""

    @staticmethod
    def poiseuille_velocity_profile(y: np.ndarray, height: float, u_mean: float) -> np.ndarray:
        """
        Exact fully developed laminar velocity profile between stationary parallel plates:
        u(y) = 6 * u_mean * (y / H) * (1 - y / H)
        Centerline peak velocity: u_max = 1.5 * u_mean.
        """
        eta = y / height
        return 6.0 * u_mean * eta * (1.0 - eta)

    @staticmethod
    def poiseuille_pressure_gradient(mu: float, u_mean: float, height: float) -> float:
        """
        Theoretical pressure gradient for fully developed laminar channel flow:
        dp/dx = -12 * mu * u_mean / (height^2)
        """
        return -12.0 * mu * u_mean / (height ** 2)

    @staticmethod
    def laminar_friction_factor(re_dh: float) -> float:
        """
        Theoretical Darcy-Weisbach friction factor for parallel plates:
        f_Darcy = 96.0 / Re_Dh
        """
        if re_dh <= 0:
            raise ValueError("Reynolds number must be positive.")
        return 96.0 / re_dh

    @staticmethod
    def asymptotic_nusselt(bc_type: str = "isothermal") -> float:
        """
        Theoretical fully developed Nusselt number Nu_Dh,inf for laminar duct flow:
        - Isothermal walls (Dirichlet): Nu_inf = 7.541
        - Isoflux walls (Neumann):      Nu_inf = 8.235
        """
        if bc_type == "isothermal":
            return 7.541
        elif bc_type == "isoflux":
            return 8.235
        elif bc_type == "asymmetric":
            return 4.861 # Single heated wall, other adiabatic
        else:
            raise ValueError(f"Unknown boundary type: {bc_type}")

    @classmethod
    def shah_london_local_nusselt(cls, x: np.ndarray, dh: float, re_dh: float, pr: float, bc_type: str = "isothermal") -> np.ndarray:
        """
        Shah-London empirical / series correlation for local Nusselt number Nu_x
        in the thermally developing entry region between parallel plates.
        
        Dimensionless axial distance: x_plus = x / (dh * Re_dh * Pr)
        """
        x_plus = np.maximum(x / (dh * re_dh * pr), 1e-6)
        nu_local = np.zeros_like(x_plus)

        if bc_type == "isothermal":
            nu_inf = 7.541
            mask1 = x_plus <= 0.001
            nu_local[mask1] = 1.233 * (x_plus[mask1] ** (-1.0 / 3.0)) + 0.4
            mask2 = ~mask1
            nu_local[mask2] = nu_inf + (0.0235 / x_plus[mask2]) / (1.0 + 0.0101 * (x_plus[mask2] ** -0.8))
        else: # isoflux
            nu_inf = 8.235
            mask1 = x_plus <= 0.0002
            nu_local[mask1] = 1.490 * (x_plus[mask1] ** (-1.0 / 3.0))
            mask2 = ~mask1
            nu_local[mask2] = nu_inf + (0.036 / x_plus[mask2]) / (1.0 + 0.011 * (x_plus[mask2] ** -0.8))

        return nu_local

    @staticmethod
    def hydrodynamic_entry_length(re_dh: float, dh: float) -> float:
        """Hydrodynamic entrance length L_h approx 0.05 * Re_Dh * D_h."""
        return 0.05 * re_dh * dh

    @staticmethod
    def thermal_entry_length(re_dh: float, pr: float, dh: float) -> float:
        """Thermal entrance length L_th approx 0.05 * Re_Dh * Pr * D_h."""
        return 0.05 * re_dh * pr * dh
