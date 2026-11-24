"""Run conjugate multi-fin heat sink simulation."""

import sys
import os
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src import MultiFinHeatSink2D, Visualizer


def run_multi_fin_simulation():
    print("=" * 75)
    print("MULTI-FIN CONJUGATE HEAT SINK SIMULATION (SOLID + FLUID DOMAIN)")
    print("=" * 75)

    sim = MultiFinHeatSink2D(
        n_fins=4,
        fin_spacing=0.0029,
        fin_thickness=0.0010,
        fin_height=0.0120,
        base_thickness=0.0030,
        channel_length=0.0400,
        u_inlet=1.5,
        t_inlet=300.0,
        q_base_flux=10000.0, # 10 kW/m^2 heat flux from processor
        nx=120,
        ny=60
    )

    print(f"Domain Size        : {sim.lx*1e3:.1f} mm (L) x {sim.ly*1e3:.1f} mm (H)")
    print(f"Baseplate Thickness: {sim.t_base*1e3:.1f} mm (Aluminum k = {sim.k_s} W/(m K))")
    print(f"Airflow Velocity   : {sim.u_inlet:.2f} m/s (Air k = {sim.k_f} W/(m K))")
    print(f"Applied Heat Flux  : {sim.q_base_flux:.0f} W/m^2")

    print("\nSolving coupled conjugate heat transfer equations...")
    res = sim.solve_conjugate_flow_and_heat()

    print("\nCONJUGATE RESULTS:")
    print(f"Max Baseplate Temperature : {res['t_base_max_C']:.2f} °C")
    print(f"Min Baseplate Temperature : {res['t_base_min_C']:.2f} °C")
    print(f"Mean Air Outlet Temp      : {res['t_fluid_out_C']:.2f} °C")
    print(f"Total Heat Injected       : {res['total_heat_input_W_m']:.2f} W/m depth")
    print(f"Effective Thermal Res     : {res['effective_thermal_resistance']:.4f} (K m)/W")

    # Generate Visualization
    visualizer = Visualizer(output_dir="results/figures")
    f_path = visualizer.plot_multi_fin_conjugate(
        sim.X, sim.Y, sim.T, sim.u, sim.is_solid, sim.t_base
    )
    print(f"\nConjugate plot saved to: {f_path}")
    print("=" * 75)
    return res


if __name__ == "__main__":
    run_multi_fin_simulation()
