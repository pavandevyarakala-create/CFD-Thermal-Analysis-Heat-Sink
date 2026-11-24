"""2D Energy transport solver coupled with the Navier-Stokes velocity field.

Governing Equation:
-------------------
dT/dt + u * dT/dx + v * dT/dy = alpha * (d^2T/dx^2 + d^2T/dy^2)

Boundary Conditions:
- Inlet: Dirichlet T(0, y) = T_inlet
- Outlet: Convective / Neumann dT/dx(L_x, y) = 0
- Bottom & Top Walls:
  * Isothermal (Dirichlet): T = T_wall
  * Isoflux (Neumann): -k * dT/dn = q"_w
  * Asymmetric: T(bottom) = T_wall, dT/dy(top) = 0 (insulated)
"""

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import splu
from typing import Dict, Tuple, Optional
from .parameters import SimulationCase
from .grid import StructuredGrid2D


class Thermal2DSolver:
    """Solves the 2D thermal convection-diffusion equation coupled to flow velocity."""

    def __init__(self, case: SimulationCase, grid: StructuredGrid2D, u: np.ndarray, v: np.ndarray):
        self.case = case
        self.grid = grid
        self.u = u
        self.v = v

        self.nx = grid.nx
        self.ny = grid.ny
        self.dx = grid.dx
        self.dy = grid.dy
        self.alpha = case.fluid.alpha
        self.k_fluid = case.fluid.k
        self.t_in = case.bc.t_inlet
        self.bc_type = case.bc.thermal_bc_type

        # Temperature field [ny, nx] in Kelvin
        self.T = np.full((self.ny, self.nx), self.t_in, dtype=np.float64)

        # Residuals
        self.residuals = []
        self.is_converged = False
        self.total_iterations = 0

        # Apply initial boundary values
        self._apply_boundary_conditions()

    def _apply_boundary_conditions(self):
        """Enforce inlet and wall boundary conditions."""
        # Inlet (i = 0)
        self.T[:, 0] = self.t_in

        # Outlet (i = -1): initial flat extrapolation
        self.T[:, -1] = self.T[:, -2]

        # Walls (j = 0 and j = -1)
        if self.bc_type == "isothermal":
            self.T[0, :] = self.case.bc.t_wall_bottom
            self.T[-1, :] = self.case.bc.t_wall_top
        elif self.bc_type == "isoflux":
            # q" = -k * dT/dy -> T_wall = T_interior + q" * dy / k
            q_w = self.case.bc.heat_flux_wall
            self.T[0, :] = self.T[1, :] + (q_w * self.dy / self.k_fluid)
            self.T[-1, :] = self.T[-2, :] + (q_w * self.dy / self.k_fluid)
        elif self.bc_type == "asymmetric":
            # Bottom isothermal, top adiabatic (dT/dy = 0)
            self.T[0, :] = self.case.bc.t_wall_bottom
            self.T[-1, :] = self.T[-2, :]
        else:
            raise ValueError(f"Unknown thermal boundary condition type: {self.bc_type}")

    def solve_steady_implicit(self) -> float:
        """
        Solve the steady-state convection-diffusion equation:
        u * dT/dx + v * dT/dy - alpha * (d2T/dx2 + d2T/dy2) = 0
        directly via sparse linear solver (A * T_vec = b) with upwind differencing.
        """
        n_nodes = self.ny * self.nx
        dx = self.dx
        dy = self.dy
        dx2 = dx ** 2
        dy2 = dy ** 2
        alpha = self.alpha

        A = sparse.lil_matrix((n_nodes, n_nodes), dtype=np.float64)
        b = np.zeros(n_nodes, dtype=np.float64)

        for j in range(self.ny):
            for i in range(self.nx):
                idx = j * self.nx + i

                # Boundary nodes
                if j == 0:
                    # Bottom wall
                    if self.bc_type in ["isothermal", "asymmetric"]:
                        A[idx, idx] = 1.0
                        b[idx] = self.case.bc.t_wall_bottom
                    elif self.bc_type == "isoflux":
                        # -k * (T[1] - T[0]) / dy = q" -> T[0] - T[1] = q" * dy / k
                        A[idx, idx] = 1.0
                        A[idx, idx + self.nx] = -1.0
                        b[idx] = self.case.bc.heat_flux_wall * dy / self.k_fluid
                elif j == self.ny - 1:
                    # Top wall
                    if self.bc_type == "isothermal":
                        A[idx, idx] = 1.0
                        b[idx] = self.case.bc.t_wall_top
                    elif self.bc_type == "asymmetric":
                        # Adiabatic top wall: dT/dy = 0 -> T[-1] - T[-2] = 0
                        A[idx, idx] = 1.0
                        A[idx, idx - self.nx] = -1.0
                        b[idx] = 0.0
                    elif self.bc_type == "isoflux":
                        # T[-1] - T[-2] = q" * dy / k
                        A[idx, idx] = 1.0
                        A[idx, idx - self.nx] = -1.0
                        b[idx] = self.case.bc.heat_flux_wall * dy / self.k_fluid
                elif i == 0:
                    # Inlet: Dirichlet T = T_in
                    A[idx, idx] = 1.0
                    b[idx] = self.t_in
                elif i == self.nx - 1:
                    # Outlet: Convective Neumann dT/dx = 0 -> T[-1] - T[-2] = 0
                    A[idx, idx] = 1.0
                    A[idx, idx - 1] = -1.0
                    b[idx] = 0.0
                else:
                    # Interior node
                    u_val = self.u[j, i]
                    v_val = self.v[j, i]

                    # Standard Patankar Upwind Convection-Diffusion Coefficients
                    a_W = (alpha / dx2) + (max(u_val, 0.0) / dx)
                    a_E = (alpha / dx2) + (max(-u_val, 0.0) / dx)
                    a_S = (alpha / dy2) + (max(v_val, 0.0) / dy)
                    a_N = (alpha / dy2) + (max(-v_val, 0.0) / dy)
                    
                    # Strictly diagonally dominant: a_P = a_W + a_E + a_S + a_N
                    a_P = a_W + a_E + a_S + a_N

                    A[idx, idx] = a_P
                    A[idx, idx + 1] = -a_E
                    A[idx, idx - 1] = -a_W
                    A[idx, idx + self.nx] = -a_N
                    A[idx, idx - self.nx] = -a_S
                    b[idx] = 0.0

        A_csc = A.tocsc()
        T_1d = splu(A_csc).solve(b)
        self.T[:, :] = T_1d.reshape((self.ny, self.nx))
        self.is_converged = True
        return 0.0

    def step_transient(self, dt: float, upwind_blend: float = 1.0) -> float:
        """
        Advance one pseudo-transient step to observe thermal boundary layer evolution
        and record convergence residuals.
        """
        T = self.T.copy()
        u = self.u
        v = self.v
        dx = self.dx
        dy = self.dy

        # Convective terms
        dT_dx_upwind = np.where(u > 0,
                                (T - np.roll(T, 1, axis=1)) / dx,
                                (np.roll(T, -1, axis=1) - T) / dx)
        dT_dx_central = (np.roll(T, -1, axis=1) - np.roll(T, 1, axis=1)) / (2.0 * dx)
        dT_dx = upwind_blend * dT_dx_upwind + (1.0 - upwind_blend) * dT_dx_central

        dT_dy_upwind = np.where(v > 0,
                                (T - np.roll(T, 1, axis=0)) / dy,
                                (np.roll(T, -1, axis=0) - T) / dy)
        dT_dy_central = (np.roll(T, -1, axis=0) - np.roll(T, 1, axis=0)) / (2.0 * dy)
        dT_dy = upwind_blend * dT_dy_upwind + (1.0 - upwind_blend) * dT_dy_central

        # Diffusion terms
        d2T_dx2 = (np.roll(T, -1, axis=1) - 2.0 * T + np.roll(T, 1, axis=1)) / (dx ** 2)
        d2T_dy2 = (np.roll(T, -1, axis=0) - 2.0 * T + np.roll(T, 1, axis=0)) / (dy ** 2)

        rhs = -(u * dT_dx + v * dT_dy) + self.alpha * (d2T_dx2 + d2T_dy2)

        # Update interior
        T_new = T.copy()
        T_new[1:-1, 1:-1] = T[1:-1, 1:-1] + dt * rhs[1:-1, 1:-1]

        # Enforce boundary conditions
        self.T[:, :] = T_new
        self._apply_boundary_conditions()

        res = float(np.linalg.norm(self.T[1:-1, 1:-1] - T[1:-1, 1:-1]) / (np.linalg.norm(T[1:-1, 1:-1]) + 1e-12))
        return res

    def solve(self, method: str = "implicit", max_iter: int = 5000, tol: float = 1e-6) -> Dict[str, float]:
        """Solve the thermal field via steady implicit or pseudo-transient stepping."""
        if method == "implicit":
            self.solve_steady_implicit()
            return {"method": "implicit", "converged": True, "final_res": 0.0}
        else:
            for it in range(1, max_iter + 1):
                dt = self.grid.compute_stable_dt(self.u, self.v, self.alpha, self.case.numeric.cfl_safety)
                res = self.step_transient(dt)
                self.residuals.append(res)
                self.total_iterations = it
                if res < tol and it > 50:
                    self.is_converged = True
                    break
            return {"method": "transient", "converged": self.is_converged, "final_res": self.residuals[-1]}
