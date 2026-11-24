"""Master Execution Script: Runs all CFD simulations, validations, and sweeps."""

import time
import sys
import os

from scripts.run_baseline import run_baseline_simulation
from scripts.run_grid_independence import run_grid_refinement_study
from scripts.run_parametric_study import run_parametric_study
from scripts.run_multi_fin_array import run_multi_fin_simulation


def main():
    start_time = time.time()
    print("=" * 80)
    print("    COMPUTATIONAL FLUID DYNAMICS (CFD) & THERMAL ANALYSIS SUITE")
    print("           Project: Airflow Simulation & Heat Sink Optimization")
    print("=" * 80)

    # 1. Baseline Case
    print("\n>>> STEP 1/4: Executing Baseline Single-Channel Simulation (Scenario S1)...")
    run_baseline_simulation()

    # 2. Grid Independence Study
    print("\n>>> STEP 2/4: Executing Grid Refinement Study & GCI Verification...")
    run_grid_refinement_study()

    # 3. Parametric Sensitivity Analysis
    print("\n>>> STEP 3/4: Executing Multi-Parameter Sensitivity Sweeps...")
    run_parametric_study()

    # 4. Multi-Fin Array Conjugate Simulation
    print("\n>>> STEP 4/4: Executing Full Conjugate Multi-Fin Heat Sink Analysis...")
    run_multi_fin_simulation()

    elapsed = time.time() - start_time
    print("\n" + "=" * 80)
    print(f"ALL SIMULATIONS AND VISUALIZATIONS COMPLETED SUCCESSFULLY in {elapsed:.2f} s")
    print("Generated Assets:")
    print("  - Figures directory: results/figures/")
    print("  - Data directory   : results/data/")
    print("=" * 80)


if __name__ == "__main__":
    main()
