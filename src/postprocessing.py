"""Post-processing, thermodynamic calculations, and mass/energy conservation verification."""

import numpy as np
from typing import Dict, Any
from .parameters import SimulationCase
from .grid import StructuredGrid2D


class PostProcessor:
    """Calculates engineering heat transfer metrics, pressure drops, and conservation checks."""

    def __init__(self, case: SimulationCase, grid: StructuredGrid2D,
                 u: np.ndarray, v: np.ndarray, p: np.ndarray, T: np.ndarray):
        self.case = case
        self.grid = grid
        self.u = u
        self.v = v
        self.p = p
        self.T = T

        self.dx = grid.dx
        self.dy = grid.dy
        self.rho = case.fluid.rho
        self.cp = case.fluid.cp
        self.k_fluid = case.fluid.k
        self.depth = case.geometry.fin_depth
        self.dh = case.geometry.hydraulic_diameter

    def verify_mass_conservation(self) -> Dict[str, Any]:
        """
        Verify continuity by integrating mass flux across each streamwise vertical slice:
        m_dot(x) = depth * rho * integral(u(x, y) dy)
        """
        m_dot_x = np.zeros(self.grid.nx)
        for i in range(self.grid.nx):
            # Trapezoidal integration across y
            m_dot_x[i] = self.depth * self.rho * np.trapezoid(self.u[:, i], self.grid.y)

        m_dot_inlet = m_dot_x[0]
        rel_errors = np.abs(m_dot_x - m_dot_inlet) / (m_dot_inlet + 1e-12) * 100.0
        max_error = float(np.max(rel_errors))
        mean_error = float(np.mean(rel_errors))

        return {
            "m_dot_inlet": float(m_dot_inlet),
            "m_dot_x": m_dot_x,
            "max_rel_error_percent": max_error,
            "mean_rel_error_percent": mean_error
        }

    def compute_bulk_temperature(self) -> np.ndarray:
        """
        Compute cup-mixing (bulk mean) temperature T_m(x):
        T_m(x) = integral(u * T dy) / integral(u dy)
        """
        t_bulk = np.zeros(self.grid.nx)
        for i in range(self.grid.nx):
            u_slice = self.u[:, i]
            t_slice = self.T[:, i]
            denom = np.trapezoid(u_slice, self.grid.y)
            if denom > 1e-12:
                numer = np.trapezoid(u_slice * t_slice, self.grid.y)
                t_bulk[i] = numer / denom
            else:
                t_bulk[i] = np.mean(t_slice)
        return t_bulk

    def compute_local_heat_transfer(self) -> Dict[str, np.ndarray]:
        """
        Compute wall temperature gradients, heat flux, convective heat transfer coefficient,
        and local Nusselt number Nu_x = h_x * D_h / k.
        """
        t_bulk = self.compute_bulk_temperature()

        # Wall temperature gradients using 2nd-order one-sided differences
        # Bottom wall (j = 0): dT/dy ≈ (-3*T[0] + 4*T[1] - T[2]) / (2*dy)
        dt_dy_bottom = (-3.0 * self.T[0, :] + 4.0 * self.T[1, :] - self.T[2, :]) / (2.0 * self.dy)
        q_flux_bottom = -self.k_fluid * dt_dy_bottom  # Heat flux into fluid [W/m^2]

        # Top wall (j = -1): dT/dy ≈ (3*T[-1] - 4*T[-2] + T[-3]) / (2*dy)
        dt_dy_top = (3.0 * self.T[-1, :] - 4.0 * self.T[-2, :] + self.T[-3, :]) / (2.0 * self.dy)
        q_flux_top = self.k_fluid * dt_dy_top  # Heat flux into fluid [W/m^2]

        # Convective heat transfer coefficient h_x = q"_w / (T_wall - T_bulk)
        delta_t_bottom = np.maximum(np.abs(self.T[0, :] - t_bulk), 0.05)
        h_bottom = np.abs(q_flux_bottom) / delta_t_bottom
        nu_bottom = h_bottom * self.dh / self.k_fluid

        delta_t_top = np.maximum(np.abs(self.T[-1, :] - t_bulk), 0.05)
        h_top = np.abs(q_flux_top) / delta_t_top
        nu_top = h_top * self.dh / self.k_fluid

        nu_mean_local = 0.5 * (nu_bottom + nu_top)

        return {
            "t_bulk": t_bulk,
            "q_flux_bottom": q_flux_bottom,
            "q_flux_top": q_flux_top,
            "h_bottom": h_bottom,
            "h_top": h_top,
            "nu_bottom": nu_bottom,
            "nu_top": nu_top,
            "nu_mean": nu_mean_local
        }

    def compute_global_performance_metrics(self) -> Dict[str, float]:
        """Compute integrated heat dissipation, thermal resistance, pressure drop, and pumping power."""
        mass_info = self.verify_mass_conservation()
        m_dot = mass_info["m_dot_inlet"]
        t_bulk = self.compute_bulk_temperature()

        t_inlet = t_bulk[0]
        t_outlet = t_bulk[-1]

        # Total thermal energy carried away by fluid [W]
        q_enthalpy = m_dot * self.cp * (t_outlet - t_inlet)

        # Total heat dissipated via wall integration [W]
        ht_data = self.compute_local_heat_transfer()
        q_walls = self.depth * np.trapezoid(np.abs(ht_data["q_flux_bottom"]) + np.abs(ht_data["q_flux_top"]), self.grid.x)

        # Thermal resistance [K/W]
        t_wall_max = max(np.max(self.T[0, :]), np.max(self.T[-1, :]))
        thermal_resistance = (t_wall_max - self.case.bc.t_inlet) / (q_enthalpy + 1e-12)

        # Pressure drop across channel
        p_inlet_mean = np.mean(self.p[:, 0])
        p_outlet_mean = np.mean(self.p[:, -1])
        delta_p = float(p_inlet_mean - p_outlet_mean)

        # Volumetric flow rate and pumping power
        vol_flow_rate = m_dot / self.rho
        pumping_power = float(vol_flow_rate * delta_p)

        # Apparent friction factor f_app = (delta_P / (0.5 * rho * U^2)) * (D_h / L_x)
        u_mean = self.case.bc.u_inlet
        f_darcy = (delta_p / (0.5 * self.rho * (u_mean ** 2))) * (self.dh / self.grid.lx)
        f_re = float(f_darcy * self.case.reynolds_dh)

        # Average Nusselt number
        nu_avg = float(np.mean(ht_data["nu_mean"][1:-1]))

        return {
            "m_dot_kg_s": m_dot,
            "t_bulk_inlet_K": float(t_inlet),
            "t_bulk_outlet_K": float(t_outlet),
            "temperature_rise_K": float(t_outlet - t_inlet),
            "heat_dissipated_W": float(q_enthalpy),
            "heat_wall_integrated_W": float(q_walls),
            "energy_balance_error_percent": float(abs(q_enthalpy - q_walls) / (q_enthalpy + 1e-12) * 100.0),
            "thermal_resistance_K_W": float(thermal_resistance),
            "pressure_drop_Pa": delta_p,
            "pumping_power_mW": pumping_power * 1000.0,
            "f_darcy": float(f_darcy),
            "f_Re": f_re,
            "nu_average": nu_avg,
            "max_mass_error_percent": mass_info["max_rel_error_percent"]
        }
