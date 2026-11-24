"""2D structured Cartesian mesh generation and spatial metrics for channel CFD."""

import numpy as np
from typing import Tuple


class StructuredGrid2D:
    """Represents a 2D uniform structured discretization grid."""

    def __init__(self, lx: float, ly: float, nx: int, ny: int):
        """
        Initialize the 2D grid.
        
        Parameters:
        -----------
        lx : float
            Total domain length in x-direction [m].
        ly : float
            Total domain height in y-direction [m].
        nx : int
            Number of spatial nodes along x-direction.
        ny : int
            Number of spatial nodes along y-direction.
        """
        if nx < 3 or ny < 3:
            raise ValueError(f"Grid dimensions must be at least 3x3, got nx={nx}, ny={ny}")
        if lx <= 0.0 or ly <= 0.0:
            raise ValueError(f"Domain dimensions must be positive, got lx={lx}, ly={ly}")

        self.lx = float(lx)
        self.ly = float(ly)
        self.nx = int(nx)
        self.ny = int(ny)

        self.dx = self.lx / (self.nx - 1)
        self.dy = self.ly / (self.ny - 1)

        self.x = np.linspace(0.0, self.lx, self.nx)
        self.y = np.linspace(0.0, self.ly, self.ny)
        self.X, self.Y = np.meshgrid(self.x, self.y)

    @property
    def shape(self) -> Tuple[int, int]:
        """Return array shape (ny, nx)."""
        return self.ny, self.nx

    def compute_cell_peclet(self, u: np.ndarray, v: np.ndarray, diffusivity: float) -> Tuple[np.ndarray, float]:
        """
        Compute the local cell Peclet number Pe_cell = |u|*dx / diffusivity.
        A cell Peclet number > 2 indicates that central differencing may exhibit unphysical oscillations,
        necessitating upwind differencing or grid refinement.
        """
        if diffusivity <= 0:
            raise ValueError("Diffusivity must be positive.")
        pe_x = np.abs(u) * self.dx / diffusivity
        pe_y = np.abs(v) * self.dy / diffusivity
        pe_max = float(max(np.max(pe_x), np.max(pe_y)))
        return pe_x, pe_max

    def compute_stable_dt(self, u: np.ndarray, v: np.ndarray, diffusion_coeff: float, cfl_safety: float = 0.4) -> float:
        """
        Compute the maximum stable time step considering both CFL convective stability
        and von Neumann diffusive stability limits.
        """
        u_max = float(np.max(np.abs(u)))
        v_max = float(np.max(np.abs(v)))

        # Convective CFL limit
        if u_max > 1e-12 or v_max > 1e-12:
            dt_conv = 1.0 / ( (u_max / self.dx) + (v_max / self.dy) + 1e-14 )
        else:
            dt_conv = 1.0

        # Diffusive von Neumann stability limit: dt <= 0.5 / (diff * (1/dx^2 + 1/dy^2))
        dt_diff = 0.5 / (diffusion_coeff * (1.0 / (self.dx ** 2) + 1.0 / (self.dy ** 2)))

        dt_stable = cfl_safety * min(dt_conv, dt_diff)
        return float(dt_stable)
