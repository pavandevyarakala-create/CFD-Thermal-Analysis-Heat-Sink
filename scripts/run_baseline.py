"""Run baseline CFD and thermal analysis of laptop heat sink fin channel.

Problem Specification:
- Working Fluid: Air (nu = 1.56e-5 m^2/s, alpha = 2.2e-5 m^2/s, Pr = 0.71)
- Channel Gap: L = 2.9 mm (2.9e-3 m)
- Channel Length: H_x = 40.0 mm (40.0e-3 m)
- Inflow Velocity: U = 1.5 m/s (Re_s = 278.8, Re_Dh = 557.7, Pe = 197.7)
- Inflow Temperature: T_in = 300 K (26.85 °C)
- Fin Wall Temperature: T_fin = 345 K (71.85 °C)
"""

import sys
import os
import json
import numpy as np
import pandas as pd

# Add parent directory to path to import src
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src import (
    SimulationCase,
    StructuredGrid2D,
    NavierStokes2DSolver,
    Thermal2DSolver,
    AnalyticalBenchmarks,
    PostProcessor,
    Visualizer
)


def run_baseline_simulation():
    print("=" * 75)
    print("CFD-Based Thermal Analysis & Airflow Simulation of a Heat Sink (Baseline S1)")
    print("=" * 75)

    case = SimulationCase(name="Baseline_S1")
    print(f"Fluid                 : {case.fluid.name} (rho={case.fluid.rho} kg/m^3, nu={case.fluid.nu} m^2/s)")
    print(f"Channel Dimensions    : Length = {case.geometry.length * 1e3:.1f} mm, Gap L = {case.geometry.height * 1e3:.2f} mm")
    print(f"Hydraulic Diameter    : D_h = {case.geometry.hydraulic_diameter * 1e3:.2f} mm")
    print(f"Inflow Velocity       : U_in = {case.bc.u_inlet:.2f} m/s")
    print(f"Reynolds Number (Gap) : Re_s = {case.reynolds_gap:.2f}")
    print(f"Reynolds Number (Dh)  : Re_Dh = {case.reynolds_dh:.2f}")
    print(f"Prandtl Number        : Pr = {case.fluid.prandtl:.3f}")
    print(f"Peclet Number (Gap)   : Pe_s = {case.peclet_gap:.2f}")
    print(f"Thermal Entry Length  : L_th = {AnalyticalBenchmarks.thermal_entry_length(case.reynolds_dh, case.fluid.prandtl, case.geometry.hydraulic_diameter)*1e3:.1f} mm")
    print(f"Hydro Entry Length    : L_h = {AnalyticalBenchmarks.hydrodynamic_entry_length(case.reynolds_dh, case.geometry.hydraulic_diameter)*1e3:.1f} mm")
    print("-" * 75)

    # 1. Grid Generation
    nx, ny = case.numeric.nx, case.numeric.ny
    grid = StructuredGrid2D(case.geometry.length, case.geometry.height, nx, ny)
    print(f"Computational Mesh    : {nx} x {ny} nodes (dx = {grid.dx*1e3:.3f} mm, dy = {grid.dy*1e3:.3f} mm)")

    # Stability & Cell Peclet Audit
    u_est = np.full((ny, nx), case.bc.u_inlet)
    v_est = np.zeros((ny, nx))
    _, pe_max = grid.compute_cell_peclet(u_est, v_est, case.fluid.alpha)
    dt_cfl = grid.compute_stable_dt(u_est, v_est, case.fluid.nu, case.numeric.cfl_safety)
    print(f"Max Cell Peclet Number: Pe_cell = {pe_max:.2f}")
    print(f"Numerical Time Step   : dt = {dt_cfl:.3e} s")
    print("-" * 75)

    # 2. Navier-Stokes Solver
    print("Solving Incompressible Navier-Stokes Equations (Streamfunction-Vorticity)...")
    ns_solver = NavierStokes2DSolver(case, grid)
    ns_info = ns_solver.solve(verbose=False)
    print(f"Flow Solution Done    : {ns_info['iterations']} iterations | Final Res(omega) = {ns_info['final_res_omega']:.2e}")

    # 3. Thermal Energy Solver
    print("Solving 2D Thermal Convection-Diffusion Energy Equation...")
    th_solver = Thermal2DSolver(case, grid, ns_solver.u, ns_solver.v)
    th_info = th_solver.solve(method="implicit")
    print("Thermal Solution Done : Implicit Direct Sparse Solve Converged.")
    print("-" * 75)

    # 4. Post-processing and Engineering Metrics
    post = PostProcessor(case, grid, ns_solver.u, ns_solver.v, ns_solver.p, th_solver.T)
    metrics = post.compute_global_performance_metrics()
    ht_data = post.compute_local_heat_transfer()

    print("POST-PROCESSING & PERFORMANCE METRICS:")
    print(f"Mass Flow Rate (m_dot): {metrics['m_dot_kg_s']:.5e} kg/s")
    print(f"Mass Conservation Err : {metrics['max_mass_error_percent']:.4e} % (machine precision / exact)")
    print(f"Bulk Fluid Temp Rise  : {metrics['t_bulk_inlet_K']:.2f} K -> {metrics['t_bulk_outlet_K']:.2f} K (delta T = {metrics['temperature_rise_K']:.2f} K)")
    print(f"Total Heat Dissipated : Q = {metrics['heat_dissipated_W']:.3f} W (Wall Integrated: {metrics['heat_wall_integrated_W']:.3f} W)")
    print(f"Energy Balance Error  : {metrics['energy_balance_error_percent']:.2f} %")
    print(f"Thermal Resistance    : R_th = {metrics['thermal_resistance_K_W']:.3f} K/W")
    print(f"Channel Pressure Drop : Delta P = {metrics['pressure_drop_Pa']:.3f} Pa")
    print(f"Fan Pumping Power     : W_pump = {metrics['pumping_power_mW']:.3f} mW")
    print(f"Average Nusselt Number: Nu_avg = {metrics['nu_average']:.3f}")
    print(f"Apparent f * Re_Dh    : {metrics['f_Re']:.2f} (Theoretical fully-developed = 96.0)")
    print("-" * 75)

    # 5. Validation Against Analytical Benchmarks
    u_analytical_fd = AnalyticalBenchmarks.poiseuille_velocity_profile(grid.y, case.geometry.height, case.bc.u_inlet)
    nu_shah_london = AnalyticalBenchmarks.shah_london_local_nusselt(
        grid.x, case.geometry.hydraulic_diameter, case.reynolds_dh, case.fluid.prandtl, case.bc.thermal_bc_type
    )
    nu_asymp = AnalyticalBenchmarks.asymptotic_nusselt(case.bc.thermal_bc_type)

    # 6. Generate Publication Figures
    visualizer = Visualizer(output_dir="results/figures")
    print("Generating Engineering Visualizations...")

    f1 = visualizer.plot_flowfield_and_streamlines(grid.X, grid.Y, ns_solver.u, ns_solver.v, ns_solver.psi, "baseline")
    f2 = visualizer.plot_temperature_field(grid.X, grid.Y, th_solver.T, ht_data["t_bulk"], "baseline")
    
    x_sample_stations = [0.002, 0.010, 0.020, 0.030, 0.040]
    f3 = visualizer.plot_hydrodynamic_validation(grid.y, ns_solver.u, u_analytical_fd, x_sample_stations, grid.x, "baseline")
    f4 = visualizer.plot_thermal_validation(grid.x, ht_data["nu_mean"], nu_shah_london, nu_asymp, "baseline")
    f5 = visualizer.plot_convergence_history(ns_solver.residuals_omega, ns_solver.residuals_psi, "baseline")

    print(f"Plots saved to results/figures/:")
    print(f"  - {os.path.basename(f1)}")
    print(f"  - {os.path.basename(f2)}")
    print(f"  - {os.path.basename(f3)}")
    print(f"  - {os.path.basename(f4)}")
    print(f"  - {os.path.basename(f5)}")

    # 7. Save Metrics Data to CSV
    metrics_summary = {
        "case_name": case.name,
        "u_inlet_m_s": case.bc.u_inlet,
        "gap_spacing_mm": case.geometry.height * 1e3,
        "channel_length_mm": case.geometry.length * 1e3,
        "t_inlet_K": case.bc.t_inlet,
        "t_fin_wall_K": case.bc.t_wall_bottom,
        "Re_gap": case.reynolds_gap,
        "Re_Dh": case.reynolds_dh,
        "Pe_gap": case.peclet_gap,
        "mass_flow_rate_kg_s": metrics["m_dot_kg_s"],
        "max_mass_error_percent": metrics["max_mass_error_percent"],
        "t_bulk_outlet_K": metrics["t_bulk_outlet_K"],
        "temperature_rise_K": metrics["temperature_rise_K"],
        "heat_dissipated_W": metrics["heat_dissipated_W"],
        "thermal_resistance_K_W": metrics["thermal_resistance_K_W"],
        "pressure_drop_Pa": metrics["pressure_drop_Pa"],
        "pumping_power_mW": metrics["pumping_power_mW"],
        "average_nusselt": metrics["nu_average"],
        "apparent_f_Re": metrics["f_Re"]
    }

    os.makedirs("results/data", exist_ok=True)
    df_metrics = pd.DataFrame([metrics_summary])
    csv_path = "results/data/baseline_simulation_metrics.csv"
    df_metrics.to_csv(csv_path, index=False)
    print(f"Baseline metrics exported to {csv_path}")

    # Also export local streamwise distributions
    df_local = pd.DataFrame({
        "x_m": grid.x,
        "x_mm": grid.x * 1e3,
        "t_bulk_K": ht_data["t_bulk"],
        "t_centerline_K": th_solver.T[ny // 2, :],
        "q_flux_bottom_W_m2": ht_data["q_flux_bottom"],
        "q_flux_top_W_m2": ht_data["q_flux_top"],
        "nu_local": ht_data["nu_mean"],
        "nu_shah_london": nu_shah_london,
        "centerline_u_m_s": ns_solver.u[ny // 2, :]
    })
    local_csv_path = "results/data/baseline_streamwise_profiles.csv"
    df_local.to_csv(local_csv_path, index=False)
    print(f"Streamwise profile data exported to {local_csv_path}")
    print("=" * 75)
    return metrics_summary


if __name__ == "__main__":
    run_baseline_simulation()
