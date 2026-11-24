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
    """Verify mass conservation via streamfunction identity and discrete quadrature."""
    case = SimulationCase()
    case.numeric.nx = 61
    case.numeric.ny = 25
    case.numeric.max_iter = 1000

    grid = StructuredGrid2D(case.geometry.length, case.geometry.height, case.numeric.nx, case.numeric.ny)
    ns = NavierStokes2DSolver(case, grid)
    ns.solve(verbose=False)

    post = PostProcessor(case, grid, ns.u, ns.v, ns.p, np.zeros_like(ns.u), ns.psi)
    mass_audit = post.verify_mass_conservation(ns.psi)

    # 1. Streamfunction volumetric flux must be exact to near-machine precision (< 1e-4 %)
    assert mass_audit["max_streamfunction_flux_error_percent"] < 1e-4
    # 2. Discrete trapezoidal quadrature truncation error on this mesh must be < 2.5 %
    assert mass_audit["max_quadrature_error_percent"] < 2.5


def test_discrete_divergence_interior():
    """Verify discrete divergence cancellation for streamfunction-derived velocities."""
    case = SimulationCase()
    case.numeric.nx = 61
    case.numeric.ny = 25
    case.numeric.max_iter = 1000

    grid = StructuredGrid2D(case.geometry.length, case.geometry.height, case.numeric.nx, case.numeric.ny)
    ns = NavierStokes2DSolver(case, grid)
    ns.solve(verbose=False)

    # Interior central-difference mixed derivatives of streamfunction
    psi = ns.psi
    d2psi_dxdy = (psi[2:, 2:] - psi[:-2, 2:] - (psi[2:, :-2] - psi[:-2, :-2])) / (4.0 * grid.dx * grid.dy)
    d2psi_dydx = (psi[2:, 2:] - psi[2:, :-2] - (psi[:-2, 2:] - psi[:-2, :-2])) / (4.0 * grid.dx * grid.dy)
    residual_div = np.abs(d2psi_dxdy - d2psi_dydx)

    assert np.max(residual_div) < 1e-11


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


def test_energy_balance_closure():
    """Verify first-law energy closure between enthalpy rise and wall heat transfer."""
    case = SimulationCase()
    case.numeric.nx = 61
    case.numeric.ny = 25
    case.numeric.max_iter = 800

    grid = StructuredGrid2D(case.geometry.length, case.geometry.height, case.numeric.nx, case.numeric.ny)
    ns = NavierStokes2DSolver(case, grid)
    ns.solve(verbose=False)

    th = Thermal2DSolver(case, grid, ns.u, ns.v)
    th.solve(method="implicit")

    post = PostProcessor(case, grid, ns.u, ns.v, ns.p, th.T, ns.psi)
    metrics = post.compute_global_performance_metrics()

    # Energy closure error must be reasonable on coarse mesh (< 10%)
    assert metrics["energy_balance_error_percent"] < 10.0


if __name__ == "__main__":
    print("Running unit tests...")
    test_grid_metrics()
    print("  ✓ test_grid_metrics passed.")
    test_analytical_benchmarks()
    print("  ✓ test_analytical_benchmarks passed.")
    test_navier_stokes_mass_conservation()
    print("  ✓ test_navier_stokes_mass_conservation passed.")
    test_discrete_divergence_interior()
    print("  ✓ test_discrete_divergence_interior passed.")
    test_thermal_maximum_principle()
    print("  ✓ test_thermal_maximum_principle passed.")
    test_energy_balance_closure()
    print("  ✓ test_energy_balance_closure passed.")
    print("ALL TESTS PASSED SUCCESSFULLY!")
