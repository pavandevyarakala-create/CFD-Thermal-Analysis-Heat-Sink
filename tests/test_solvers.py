"""Unit and verification tests for the CFD and thermal solvers."""

import numpy as np
import sys
import os

# Add root directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src import (
    SimulationCase,
    StructuredGrid2D,
    NavierStokes2DSolver,
    Thermal2DSolver,
    AnalyticalBenchmarks,
    PostProcessor
)


def test_grid_metrics():
    """Verify grid spatial metrics and dimension calculation."""
    grid = StructuredGrid2D(lx=0.040, ly=0.0029, nx=41, ny=21)
    assert grid.dx > 0.0
    assert grid.dy > 0.0
    assert grid.shape == (21, 41)
    assert np.isclose(grid.x[-1], 0.040)
    assert np.isclose(grid.y[-1], 0.0029)


def test_analytical_benchmarks():
    """Verify exact analytical formulas."""
    y = np.linspace(0, 0.0029, 51)
    u_prof = AnalyticalBenchmarks.poiseuille_velocity_profile(y, 0.0029, 1.5)
    
    # Boundary no-slip
    assert np.isclose(u_prof[0], 0.0)
    assert np.isclose(u_prof[-1], 0.0)
    # Centerline peak
    assert np.isclose(np.max(u_prof), 1.5 * 1.5)

    # Friction factor check: f * Re = 96
    re_test = 500.0
    f = AnalyticalBenchmarks.laminar_friction_factor(re_test)
    assert np.isclose(f * re_test, 96.0)

    # Asymptotic Nusselt numbers
    assert np.isclose(AnalyticalBenchmarks.asymptotic_nusselt("isothermal"), 7.541)
    assert np.isclose(AnalyticalBenchmarks.asymptotic_nusselt("isoflux"), 8.235)


def test_navier_stokes_mass_conservation():
    """Verify mass conservation across channel cross-sections."""
    case = SimulationCase()
    case.numeric.nx = 61
    case.numeric.ny = 25
    case.numeric.max_iter = 1000

    grid = StructuredGrid2D(case.geometry.length, case.geometry.height, case.numeric.nx, case.numeric.ny)
    ns = NavierStokes2DSolver(case, grid)
    ns.solve(verbose=False)

    post = PostProcessor(case, grid, ns.u, ns.v, ns.p, np.zeros_like(ns.u))
    mass_audit = post.verify_mass_conservation()

    # Mass error across all slices must be < 2% on coarse mesh
    assert mass_audit["max_rel_error_percent"] < 2.0


def test_thermal_maximum_principle():
    """Verify temperature obeys the discrete maximum principle (no unphysical overshoot)."""
    case = SimulationCase()
    case.numeric.nx = 61
    case.numeric.ny = 25
    case.numeric.max_iter = 800

    grid = StructuredGrid2D(case.geometry.length, case.geometry.height, case.numeric.nx, case.numeric.ny)
    ns = NavierStokes2DSolver(case, grid)
    ns.solve(verbose=False)

    th = Thermal2DSolver(case, grid, ns.u, ns.v)
    th.solve(method="implicit")

    # For T_in = 300 K and T_wall = 345 K:
    assert np.min(th.T) >= 300.0 - 1e-6
    assert np.max(th.T) <= 345.0 + 1e-6


if __name__ == "__main__":
    print("Running unit tests...")
    test_grid_metrics()
    print("  ✓ test_grid_metrics passed.")
    test_analytical_benchmarks()
    print("  ✓ test_analytical_benchmarks passed.")
    test_navier_stokes_mass_conservation()
    print("  ✓ test_navier_stokes_mass_conservation passed.")
    test_thermal_maximum_principle()
    print("  ✓ test_thermal_maximum_principle passed.")
    print("ALL TESTS PASSED SUCCESSFULLY!")
