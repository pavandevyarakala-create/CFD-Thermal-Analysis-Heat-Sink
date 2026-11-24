"""2D Incompressible Navier-Stokes solver using Streamfunction-Vorticity formulation.

Governing Equations:
--------------------
1. Vorticity definition:
   omega = dv/dx - du/dy
2. Streamfunction definition (ensures div(u) = 0 identically):
   u = dpsi/dy,  v = -dpsi/dx
3. Kinematic Poisson equation for streamfunction:
   nabla^2(psi) = -omega
4. Dynamic Vorticity Transport Equation:
   d(omega)/dt + u * d(omega)/dx + v * d(omega)/dy = nu * nabla^2(omega)
5. Pressure Poisson equation for a posteriori pressure field recovery:
   nabla^2(p) = 2 * rho * [ (du/dx)*(dv/dy) - (du/dy)*(dv/dx) ]
"""

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import splu
from typing import Dict, Tuple, Optional
from .parameters import SimulationCase
from .grid import StructuredGrid2D


class NavierStokes2DSolver:
    """Solves the 2D steady/unsteady laminar incompressible Navier-Stokes equations."""

    def __init__(self, case: SimulationCase, grid: StructuredGrid2D):
        self.case = case
        self.grid = grid

        self.nx = grid.nx
        self.ny = grid.ny
        self.dx = grid.dx
        self.dy = grid.dy
        self.nu = case.fluid.nu
        self.rho = case.fluid.rho
        self.u_in = case.bc.u_inlet

        # Field variables (shape: [ny, nx])
        self.psi = np.zeros((self.ny, self.nx), dtype=np.float64)
        self.omega = np.zeros((self.ny, self.nx), dtype=np.float64)
        self.u = np.zeros((self.ny, self.nx), dtype=np.float64)
        self.v = np.zeros((self.ny, self.nx), dtype=np.float64)
        self.p = np.zeros((self.ny, self.nx), dtype=np.float64)

        # Precompute sparse Poisson Laplacian operator for streamfunction
        self._poisson_solver = self._build_poisson_solver()

        # Convergence tracking
        self.residuals_omega = []
        self.residuals_psi = []
        self.is_converged = False
        self.total_iterations = 0

        # Initialize flow fields with realistic boundary conditions
        self.initialize_fields()

    def _build_poisson_solver(self):
        """Build and factorize the 2D discrete 5-point Laplacian matrix with Dirichlet BCs."""
        n_nodes = self.ny * self.nx
        dx2 = self.dx ** 2
        dy2 = self.dy ** 2
        main_diag = -2.0 * (1.0 / dx2 + 1.0 / dy2)

        A = sparse.lil_matrix((n_nodes, n_nodes), dtype=np.float64)

        for j in range(self.ny):
            for i in range(self.nx):
                idx = j * self.nx + i

                # Boundary nodes: enforce Dirichlet values directly
                if i == 0 or i == self.nx - 1 or j == 0 or j == self.ny - 1:
                    A[idx, idx] = 1.0
                else:
                    # Interior nodes: 5-point discrete Laplace operator
                    A[idx, idx] = main_diag
                    A[idx, idx - 1] = 1.0 / dx2          # (i-1, j)
                    A[idx, idx + 1] = 1.0 / dx2          # (i+1, j)
                    A[idx, idx - self.nx] = 1.0 / dy2    # (i, j-1)
                    A[idx, idx + self.nx] = 1.0 / dy2    # (i, j+1)

        A_csc = A.tocsc()
        return splu(A_csc)

    def initialize_fields(self):
        """Initialize the streamfunction and velocity fields."""
        # Inlet: Uniform incoming flow u = u_in -> psi = u_in * y
        for j in range(self.ny):
            self.psi[j, :] = self.u_in * self.grid.y[j]
            self.u[j, :] = self.u_in

        # Top wall: psi = u_in * H
        self.psi[-1, :] = self.u_in * self.grid.ly
        # Bottom wall: psi = 0
        self.psi[0, :] = 0.0

        # No-slip at walls
        self.u[0, :] = 0.0
        self.u[-1, :] = 0.0
        self.v[:, :] = 0.0
        self.omega[:, :] = 0.0

    def update_velocity_from_streamfunction(self):
        """Compute u = dpsi/dy and v = -dpsi/dx using 2nd-order central differences."""
        # Interior points (2nd-order central differences)
        self.u[1:-1, :] = (self.psi[2:, :] - self.psi[:-2, :]) / (2.0 * self.dy)
        self.v[:, 1:-1] = -(self.psi[:, 2:] - self.psi[:, :-2]) / (2.0 * self.dx)

        # Boundary conditions for velocity
        # Walls: No-slip
        self.u[0, :] = 0.0
        self.u[-1, :] = 0.0
        self.v[0, :] = 0.0
        self.v[-1, :] = 0.0

        # Inlet (x = 0): prescribed inflow
        self.u[:, 0] = self.u_in
        self.v[:, 0] = 0.0

        # Outlet (x = L_x): zero normal gradient du/dx = 0, dv/dx = 0
        self.u[:, -1] = self.u[:, -2]
        self.v[:, -1] = self.v[:, -2]

    def update_wall_vorticity_boundary_conditions(self):
        """
        Apply Thom's 2nd-order formula for vorticity at solid no-slip walls:
        omega_wall = -2 * (psi_interior - psi_wall) / (dy^2)
        """
        # Bottom wall (j = 0, psi_wall = 0)
        self.omega[0, :] = -2.0 * (self.psi[1, :] - self.psi[0, :]) / (self.dy ** 2)

        # Top wall (j = -1, psi_wall = u_in * H)
        self.omega[-1, :] = -2.0 * (self.psi[-2, :] - self.psi[-1, :]) / (self.dy ** 2)

        # Inlet (uniform flow u = u_in, v = 0 -> omega = 0)
        self.omega[:, 0] = 0.0

        # Outlet (Neumann: d(omega)/dx = 0)
        self.omega[:, -1] = self.omega[:, -2]

    def solve_streamfunction_poisson(self):
        """Solve nabla^2(psi) = -omega using direct pre-factorized sparse LU decomposition."""
        rhs = np.zeros(self.ny * self.nx, dtype=np.float64)

        for j in range(self.ny):
            for i in range(self.nx):
                idx = j * self.nx + i
                # Boundary conditions for streamfunction
                if j == 0:
                    rhs[idx] = 0.0                            # Bottom wall: psi = 0
                elif j == self.ny - 1:
                    rhs[idx] = self.u_in * self.grid.ly       # Top wall: psi = u_in * L
                elif i == 0:
                    rhs[idx] = self.u_in * self.grid.y[j]     # Inlet: uniform flow
                elif i == self.nx - 1:
                    # Outlet: d^2 psi / dx^2 = 0 -> psi[-1] = 2*psi[-2] - psi[-3]
                    rhs[idx] = 2.0 * self.psi[j, -2] - self.psi[j, -3]
                else:
                    rhs[idx] = -self.omega[j, i]

        psi_new_1d = self._poisson_solver.solve(rhs)
        psi_new = psi_new_1d.reshape((self.ny, self.nx))
        
        # Calculate residual change
        res = np.linalg.norm(psi_new - self.psi) / (np.linalg.norm(psi_new) + 1e-12)
        self.psi[:, :] = psi_new
        return float(res)

    def advance_vorticity(self, dt: float, upwind_blend: float = 1.0) -> float:
        """
        Advance vorticity transport equation by dt using hybrid Upwind/Central convection
        and 2nd-order central diffusion.
        """
        w = self.omega.copy()
        u = self.u
        v = self.v
        dx = self.dx
        dy = self.dy

        # Convective flux calculation with hybrid upwind blending
        # u * d(omega)/dx
        dw_dx_upwind = np.where(u > 0,
                                (w - np.roll(w, 1, axis=1)) / dx,
                                (np.roll(w, -1, axis=1) - w) / dx)
        dw_dx_central = (np.roll(w, -1, axis=1) - np.roll(w, 1, axis=1)) / (2.0 * dx)
        dw_dx = upwind_blend * dw_dx_upwind + (1.0 - upwind_blend) * dw_dx_central

        # v * d(omega)/dy
        dw_dy_upwind = np.where(v > 0,
                                (w - np.roll(w, 1, axis=0)) / dy,
                                (np.roll(w, -1, axis=0) - w) / dy)
        dw_dy_central = (np.roll(w, -1, axis=0) - np.roll(w, 1, axis=0)) / (2.0 * dy)
        dw_dy = upwind_blend * dw_dy_upwind + (1.0 - upwind_blend) * dw_dy_central

        # Diffusion terms: nu * nabla^2(omega)
        d2w_dx2 = (np.roll(w, -1, axis=1) - 2.0 * w + np.roll(w, 1, axis=1)) / (dx ** 2)
        d2w_dy2 = (np.roll(w, -1, axis=0) - 2.0 * w + np.roll(w, 1, axis=0)) / (dy ** 2)

        # Net RHS: - (u dw/dx + v dw/dy) + nu (d2w/dx2 + d2w/dy2)
        rhs = -(u * dw_dx + v * dw_dy) + self.nu * (d2w_dx2 + d2w_dy2)

        # Explicit pseudo-time advancement for interior points
        w_new = w.copy()
        w_new[1:-1, 1:-1] = w[1:-1, 1:-1] + dt * rhs[1:-1, 1:-1]

        # Calculate L2 relative change
        res = np.linalg.norm(w_new[1:-1, 1:-1] - w[1:-1, 1:-1]) / (np.linalg.norm(w_new[1:-1, 1:-1]) + 1e-12)
        self.omega[:, :] = w_new
        return float(res)

    def recover_pressure_field(self):
        """
        Recover the pressure field p(x, y) by integrating the Navier-Stokes momentum equations.
        Streamwise momentum equation:
        -1/rho * dp/dx = u * du/dx + v * du/dy - nu * (d2u/dx2 + d2u/dy2)
        Integrating from the outlet (p = 0 Pa gauge) back to the inlet captures both the
        viscous wall friction (f*Re -> 96 in developed duct) and convective entrance momentum acceleration.
        """
        mu = self.rho * self.nu
        ly = self.grid.ly

        # Wall shear stresses
        # du/dy at bottom wall (j=0) using 2nd-order forward difference
        du_dy_bottom = (-3.0 * self.u[0, :] + 4.0 * self.u[1, :] - self.u[2, :]) / (2.0 * self.dy)
        tau_bottom = mu * np.abs(du_dy_bottom)

        # du/dy at top wall (j=-1) using 2nd-order backward difference
        du_dy_top = (3.0 * self.u[-1, :] - 4.0 * self.u[-2, :] + self.u[-3, :]) / (2.0 * self.dy)
        tau_top = mu * np.abs(du_dy_top)

        # Mean wall shear stress per unit perimeter
        tau_w_mean = 0.5 * (tau_bottom + tau_top)

        # Convective momentum flux: d/dx(integral(rho * u^2 dy)) / ly
        u2_mean = np.array([np.trapezoid(self.u[:, i] ** 2, self.grid.y) / ly for i in range(self.nx)])
        d_momentum_dx = self.rho * np.gradient(u2_mean, self.dx)

        # Streamwise mean pressure gradient: dp/dx = - (2 * tau_w / (L/2) ) / 2 ... for parallel plates:
        # For channel height ly = L, force balance: dp * L = - (tau_bottom + tau_top) * dx - d(momentum)*L
        dp_dx = - (tau_bottom + tau_top) / ly - d_momentum_dx

        # Integrate backwards from outlet (x = L_x, p = 0)
        p_mean = np.zeros(self.nx, dtype=np.float64)
        for i in range(self.nx - 2, -1, -1):
            p_mean[i] = p_mean[i + 1] - 0.5 * (dp_dx[i] + dp_dx[i + 1]) * self.dx

        # Set 2D pressure field with transverse equilibrium
        for j in range(self.ny):
            self.p[j, :] = p_mean

    def solve(self, verbose: bool = False) -> Dict[str, float]:
        """
        Execute coupled iterations until convergence criterion or max_iter is met.
        """
        max_iter = self.case.numeric.max_iter
        tol = self.case.numeric.tolerance
        upwind_blend = self.case.numeric.upwind_blending

        for it in range(1, max_iter + 1):
            # Compute adaptive stable time step
            dt = self.grid.compute_stable_dt(self.u, self.v, self.nu, self.case.numeric.cfl_safety)

            # 1. Advance vorticity
            res_w = self.advance_vorticity(dt, upwind_blend)

            # 2. Update wall vorticity boundary conditions from streamfunction
            self.update_wall_vorticity_boundary_conditions()

            # 3. Solve elliptic streamfunction Poisson equation
            res_psi = self.solve_streamfunction_poisson()

            # 4. Update velocity field from new streamfunction
            self.update_velocity_from_streamfunction()

            self.residuals_omega.append(res_w)
            self.residuals_psi.append(res_psi)
            self.total_iterations = it

            if verbose and (it % 500 == 0 or it == 1):
                print(f"Iter {it:5d} | dt = {dt:.3e} s | Res(omega) = {res_w:.3e} | Res(psi) = {res_psi:.3e}")

            # Check convergence
            if res_w < tol and res_psi < tol and it > 100:
                self.is_converged = True
                if verbose:
                    print(f"Converged at iteration {it} with Res(w)={res_w:.3e}, Res(psi)={res_psi:.3e}")
                break

        # Post-solve: recover pressure field
        self.recover_pressure_field()

        return {
            "iterations": self.total_iterations,
            "converged": self.is_converged,
            "final_res_omega": self.residuals_omega[-1],
            "final_res_psi": self.residuals_psi[-1]
        }
