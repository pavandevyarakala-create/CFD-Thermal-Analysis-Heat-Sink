"""Parametric sweep over velocity, fin gap spacing, inlet temperature, and thermal BCs."""

import sys
import os
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src import (
    SimulationCase,
    StructuredGrid2D,
    NavierStokes2DSolver,
    Thermal2DSolver,
    PostProcessor,
    Visualizer
)


def run_parametric_study():
    print("=" * 75)
    print("PARAMETRIC SENSITIVITY SWEEP (HEAT SINK DESIGN EXPLORATION)")
    print("=" * 75)

    records = []

    # 1. Inlet Velocity Sweep: U in [0.5, 1.0, 1.5, 2.0, 2.5, 3.0] m/s
    velocities = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
    print(f"1. Sweeping Inlet Velocity: {velocities} m/s...")
    for u_val in velocities:
        case = SimulationCase(name=f"Vel_{u_val}")
        case.bc.u_inlet = u_val
        grid = StructuredGrid2D(case.geometry.length, case.geometry.height, case.numeric.nx, case.numeric.ny)

        ns = NavierStokes2DSolver(case, grid)
        ns.solve(verbose=False)

        th = Thermal2DSolver(case, grid, ns.u, ns.v)
        th.solve(method="implicit")

        post = PostProcessor(case, grid, ns.u, ns.v, ns.p, th.T)
        m = post.compute_global_performance_metrics()

        records.append({
            "parameter_type": "velocity",
            "u_inlet": u_val,
            "gap_spacing_mm": case.geometry.height * 1e3,
            "t_inlet_K": case.bc.t_inlet,
            "bc_type": case.bc.thermal_bc_type,
            "Re_Dh": case.reynolds_dh,
            "heat_dissipated_W": m["heat_dissipated_W"],
            "thermal_resistance_K_W": m["thermal_resistance_K_W"],
            "pressure_drop_Pa": m["pressure_drop_Pa"],
            "pumping_power_mW": m["pumping_power_mW"],
            "nu_average": m["nu_average"]
        })
        print(f"   U = {u_val:3.1f} m/s | Re_Dh = {case.reynolds_dh:5.1f} | Q = {m['heat_dissipated_W']:5.2f} W | Delta P = {m['pressure_drop_Pa']:5.2f} Pa | R_th = {m['thermal_resistance_K_W']:5.2f} K/W")

    # 2. Fin Spacing Sweep: s in [1.5, 2.2, 2.9, 3.8, 5.0] mm
    spacings = [0.0015, 0.0022, 0.0029, 0.0038, 0.0050]
    print(f"\n2. Sweeping Fin Gap Spacing: {[s*1e3 for s in spacings]} mm...")
    for s_val in spacings:
        case = SimulationCase(name=f"Gap_{s_val*1e3:.1f}")
        case.geometry.height = s_val
        grid = StructuredGrid2D(case.geometry.length, s_val, case.numeric.nx, case.numeric.ny)

        ns = NavierStokes2DSolver(case, grid)
        ns.solve(verbose=False)

        th = Thermal2DSolver(case, grid, ns.u, ns.v)
        th.solve(method="implicit")

        post = PostProcessor(case, grid, ns.u, ns.v, ns.p, th.T)
        m = post.compute_global_performance_metrics()

        records.append({
            "parameter_type": "gap_spacing",
            "u_inlet": case.bc.u_inlet,
            "gap_spacing_mm": s_val * 1e3,
            "t_inlet_K": case.bc.t_inlet,
            "bc_type": case.bc.thermal_bc_type,
            "Re_Dh": case.reynolds_dh,
            "heat_dissipated_W": m["heat_dissipated_W"],
            "thermal_resistance_K_W": m["thermal_resistance_K_W"],
            "pressure_drop_Pa": m["pressure_drop_Pa"],
            "pumping_power_mW": m["pumping_power_mW"],
            "nu_average": m["nu_average"]
        })
        print(f"   s = {s_val*1e3:3.1f} mm | Re_Dh = {case.reynolds_dh:5.1f} | Q = {m['heat_dissipated_W']:5.2f} W | Delta P = {m['pressure_drop_Pa']:5.2f} Pa | Nu_avg = {m['nu_average']:5.2f}")

    # 3. Inlet Temperature Sweep: T_in in [285, 295, 300, 310, 320] K
    t_inlets = [285.0, 295.0, 300.0, 310.0, 320.0]
    print(f"\n3. Sweeping Inlet Temperature: {t_inlets} K...")
    for t_val in t_inlets:
        case = SimulationCase(name=f"Tin_{int(t_val)}")
        case.bc.t_inlet = t_val
        grid = StructuredGrid2D(case.geometry.length, case.geometry.height, case.numeric.nx, case.numeric.ny)

        ns = NavierStokes2DSolver(case, grid)
        ns.solve(verbose=False)

        th = Thermal2DSolver(case, grid, ns.u, ns.v)
        th.solve(method="implicit")

        post = PostProcessor(case, grid, ns.u, ns.v, ns.p, th.T)
        m = post.compute_global_performance_metrics()

        records.append({
            "parameter_type": "inlet_temp",
            "u_inlet": case.bc.u_inlet,
            "gap_spacing_mm": case.geometry.height * 1e3,
            "t_inlet_K": t_val,
            "bc_type": case.bc.thermal_bc_type,
            "Re_Dh": case.reynolds_dh,
            "heat_dissipated_W": m["heat_dissipated_W"],
            "thermal_resistance_K_W": m["thermal_resistance_K_W"],
            "pressure_drop_Pa": m["pressure_drop_Pa"],
            "pumping_power_mW": m["pumping_power_mW"],
            "nu_average": m["nu_average"]
        })
        print(f"   T_in = {t_val:5.1f} K | delta T_avail = {case.bc.t_wall_bottom - t_val:4.1f} K | Q = {m['heat_dissipated_W']:5.2f} W | R_th = {m['thermal_resistance_K_W']:5.2f} K/W")

    # 4. Thermal Boundary Condition Sweep: Isothermal vs Isoflux vs Asymmetric
    bc_types = ["isothermal", "isoflux", "asymmetric"]
    print(f"\n4. Comparing Thermal Boundary Conditions: {bc_types}...")
    for bc_name in bc_types:
        case = SimulationCase(name=f"BC_{bc_name}")
        case.bc.thermal_bc_type = bc_name
        grid = StructuredGrid2D(case.geometry.length, case.geometry.height, case.numeric.nx, case.numeric.ny)

        ns = NavierStokes2DSolver(case, grid)
        ns.solve(verbose=False)

        th = Thermal2DSolver(case, grid, ns.u, ns.v)
        th.solve(method="implicit")

        post = PostProcessor(case, grid, ns.u, ns.v, ns.p, th.T)
        m = post.compute_global_performance_metrics()

        records.append({
            "parameter_type": "thermal_bc",
            "u_inlet": case.bc.u_inlet,
            "gap_spacing_mm": case.geometry.height * 1e3,
            "t_inlet_K": case.bc.t_inlet,
            "bc_type": bc_name,
            "Re_Dh": case.reynolds_dh,
            "heat_dissipated_W": m["heat_dissipated_W"],
            "thermal_resistance_K_W": m["thermal_resistance_K_W"],
            "pressure_drop_Pa": m["pressure_drop_Pa"],
            "pumping_power_mW": m["pumping_power_mW"],
            "nu_average": m["nu_average"]
        })
        print(f"   BC = {bc_name:10s} | Q = {m['heat_dissipated_W']:5.2f} W | Nu_avg = {m['nu_average']:5.2f}")

    df_sweep = pd.DataFrame(records)
    os.makedirs("results/data", exist_ok=True)
    csv_path = "results/data/parametric_analysis_results.csv"
    df_sweep.to_csv(csv_path, index=False)
    print(f"\nParametric sweep results exported to {csv_path}")

    # Generate summary visualizations
    visualizer = Visualizer(output_dir="results/figures")
    visualizer.plot_parametric_trends(df_sweep)
    print("Parametric plots saved to results/figures/parametric_analysis_summary.png")
    print("=" * 75)
    return df_sweep


if __name__ == "__main__":
    run_parametric_study()
