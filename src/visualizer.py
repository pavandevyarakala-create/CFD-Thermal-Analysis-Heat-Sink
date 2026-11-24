"""Publication-quality engineering visualizations for heat sink CFD analysis."""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from typing import Dict, List, Optional
import os


class Visualizer:
    """Generates publication-quality 2D flow and thermal contour plots and engineering graphs."""

    def __init__(self, output_dir: str = "results/figures"):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        # Apply clean engineering plotting style
        plt.rcParams.update({
            "font.family": "sans-serif",
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
            "figure.titlesize": 12,
            "figure.dpi": 300,
            "savefig.dpi": 300,
            "savefig.bbox": "tight"
        })

    def plot_flowfield_and_streamlines(self, X: np.ndarray, Y: np.ndarray,
                                       u: np.ndarray, v: np.ndarray, psi: np.ndarray,
                                       case_name: str = "baseline"):
        """Plot 2D velocity magnitude contour with overlaid streamlines."""
        vel_mag = np.sqrt(u ** 2 + v ** 2)

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 5.5), sharex=True)

        # Plot 1: Velocity Magnitude
        c1 = ax1.contourf(X * 1e3, Y * 1e3, vel_mag, levels=50, cmap="viridis")
        cb1 = plt.colorbar(c1, ax=ax1, pad=0.02)
        cb1.set_label("Velocity Magnitude $|\\vec{u}|$ [m/s]")
        ax1.set_ylabel("Gap Height $y$ [mm]")
        ax1.set_title("(a) Velocity Magnitude Field with Streamlines")

        # Overlay streamlines
        ax1.streamplot(X * 1e3, Y * 1e3, u, v, color="white", density=0.8,
                       linewidth=0.7, arrowsize=0.8, arrowstyle="->")

        # Plot 2: Streamfunction Contours
        c2 = ax2.contourf(X * 1e3, Y * 1e3, psi * 1e4, levels=30, cmap="cividis")
        cs2 = ax2.contour(X * 1e3, Y * 1e3, psi * 1e4, levels=15, colors="black", linewidths=0.5, alpha=0.5)
        ax2.clabel(cs2, inline=True, fontsize=7, fmt="%.1f")
        cb2 = plt.colorbar(c2, ax=ax2, pad=0.02)
        cb2.set_label("Streamfunction $\\psi \\times 10^4$ [m$^2$/s]")
        ax2.set_xlabel("Streamwise Distance $x$ [mm]")
        ax2.set_ylabel("Gap Height $y$ [mm]")
        ax2.set_title("(b) Streamfunction $\\psi(x, y)$ Field")

        plt.tight_layout()
        filepath = os.path.join(self.output_dir, f"{case_name}_velocity_streamlines.png")
        plt.savefig(filepath)
        plt.close(fig)
        return filepath

    def plot_temperature_field(self, X: np.ndarray, Y: np.ndarray, T: np.ndarray,
                               t_bulk: np.ndarray, case_name: str = "baseline"):
        """Plot 2D temperature contours and axial temperature distributions."""
        fig = plt.figure(figsize=(10, 6.5))
        gs = fig.add_gridspec(2, 1, height_ratios=[1.2, 1.0])

        ax1 = fig.add_subplot(gs[0])
        # Temperature contour in Celsius
        t_celsius = T - 273.15
        c = ax1.contourf(X * 1e3, Y * 1e3, t_celsius, levels=50, cmap="inferno")
        cs = ax1.contour(X * 1e3, Y * 1e3, t_celsius, levels=10, colors="cyan", linewidths=0.6, alpha=0.6)
        ax1.clabel(cs, inline=True, fontsize=7, fmt="%.1f°C")
        cb = plt.colorbar(c, ax=ax1, pad=0.02)
        cb.set_label("Temperature $T$ [°C]")
        ax1.set_ylabel("Gap Height $y$ [mm]")
        ax1.set_title("(a) Thermal Boundary Layer Development & Temperature Field $T(x,y)$")

        # Mark heat sink fin walls
        ax1.axhline(0, color="gray", linewidth=3)
        ax1.axhline(Y[-1, 0] * 1e3, color="gray", linewidth=3)

        # Plot 2: Axial temperature progression
        ax2 = fig.add_subplot(gs[1])
        x_mm = X[0, :] * 1e3
        t_wall_bottom = T[0, :] - 273.15
        t_centerline = T[T.shape[0] // 2, :] - 273.15
        t_bulk_c = t_bulk - 273.15

        ax2.plot(x_mm, t_wall_bottom, 'r-', linewidth=2, label="Fin Wall Temperature $T_w$")
        ax2.plot(x_mm, t_bulk_c, 'b--', linewidth=2, label="Bulk Mean Fluid Temperature $T_m(x)$")
        ax2.plot(x_mm, t_centerline, 'g-.', linewidth=1.8, label="Channel Centerline Temperature $T(x, L/2)$")

        ax2.set_xlabel("Streamwise Distance $x$ [mm]")
        ax2.set_ylabel("Temperature [°C]")
        ax2.set_title("(b) Streamwise Temperature Profiles Across Channel")
        ax2.grid(True, linestyle=":", alpha=0.6)
        ax2.legend(loc="best", framealpha=0.9)

        plt.tight_layout()
        filepath = os.path.join(self.output_dir, f"{case_name}_temperature_contours.png")
        plt.savefig(filepath)
        plt.close(fig)
        return filepath

    def plot_hydrodynamic_validation(self, y: np.ndarray, u: np.ndarray,
                                     u_analytical: np.ndarray, x_stations: List[float],
                                     x_coords: np.ndarray, case_name: str = "baseline"):
        """Plot velocity development at multiple stations against analytical Poiseuille flow."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5))

        y_mm = y * 1e3
        u_in = u[:, 0][len(y) // 2]

        colors = ["#2b5c8f", "#3b82f6", "#10b981", "#f59e0b", "#ef4444"]

        # Profiles across y
        for idx, (x_pos, col) in enumerate(zip(x_stations, colors)):
            col_idx = int(np.argmin(np.abs(x_coords - x_pos)))
            x_actual_mm = x_coords[col_idx] * 1e3
            ax1.plot(u[:, col_idx], y_mm, color=col, linewidth=1.8,
                     label=f"$x = {x_actual_mm:.1f}$ mm")

        # Analytical Poiseuille fully developed profile
        ax1.plot(u_analytical, y_mm, 'k--', linewidth=2.2, label="Analytical Poiseuille ($u_{fd}$)")
        ax1.set_xlabel("Streamwise Velocity $u(y)$ [m/s]")
        ax1.set_ylabel("Gap Height $y$ [mm]")
        ax1.set_title("(a) Velocity Profile Development $u(x, y)$")
        ax1.grid(True, linestyle=":", alpha=0.6)
        ax1.legend(loc="center left", bbox_to_anchor=(0.02, 0.5), fontsize=8)

        # Plot 2: Centerline velocity acceleration towards 1.5 * U_inlet
        u_center = u[len(y) // 2, :]
        u_ratio = u_center / (u_analytical[len(y) // 2] + 1e-12)
        ax2.plot(x_coords * 1e3, u_center, 'b-', linewidth=2, label="CFD Centerline Velocity $u(x, L/2)$")
        ax2.axhline(u_analytical[len(y) // 2], color='r', linestyle='--', linewidth=1.8,
                    label=f"Analytical $u_{{max}} = 1.5 U_{{in}}$ ({u_analytical[len(y)//2]:.2f} m/s)")
        ax2.set_xlabel("Streamwise Distance $x$ [mm]")
        ax2.set_ylabel("Centerline Velocity [m/s]")
        ax2.set_title("(b) Centerline Velocity Development")
        ax2.grid(True, linestyle=":", alpha=0.6)
        ax2.legend(loc="best")

        plt.tight_layout()
        filepath = os.path.join(self.output_dir, f"{case_name}_hydrodynamic_validation.png")
        plt.savefig(filepath)
        plt.close(fig)
        return filepath

    def plot_thermal_validation(self, x: np.ndarray, nu_cfd: np.ndarray,
                                nu_analytical: np.ndarray, nu_asymptotic: float,
                                case_name: str = "baseline"):
        """Plot local Nusselt number comparison against Shah-London correlation."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5))

        x_mm = x[1:-1] * 1e3
        nu_c = nu_cfd[1:-1]
        nu_a = nu_analytical[1:-1]

        ax1.plot(x_mm, nu_c, 'b-', linewidth=2.0, label="Present CFD Solver")
        ax1.plot(x_mm, nu_a, 'r--', linewidth=2.0, label="Shah-London Benchmark Correlation")
        ax1.axhline(nu_asymptotic, color='k', linestyle=':', linewidth=1.8,
                    label=f"Asymptotic Fully Developed ($Nu_\\infty = {nu_asymptotic:.2f}$)")

        ax1.set_xlabel("Streamwise Distance $x$ [mm]")
        ax1.set_ylabel("Local Nusselt Number $Nu_x$")
        ax1.set_title("(a) Thermal Verification: Local Nusselt Number")
        ax1.grid(True, linestyle=":", alpha=0.6)
        ax1.legend(loc="upper right")

        # Relative error
        rel_err = np.abs(nu_c - nu_a) / nu_a * 100.0
        ax2.plot(x_mm, rel_err, 'purple', linewidth=1.8)
        ax2.set_xlabel("Streamwise Distance $x$ [mm]")
        ax2.set_ylabel("Relative Discrepancy [%]")
        ax2.set_title(f"(b) CFD Discrepancy vs. Shah-London (Mean Error: {np.mean(rel_err):.2f}%)")
        ax2.grid(True, linestyle=":", alpha=0.6)
        ax2.set_ylim([0, max(15.0, float(np.max(rel_err) * 1.1))])

        plt.tight_layout()
        filepath = os.path.join(self.output_dir, f"{case_name}_thermal_validation.png")
        plt.savefig(filepath)
        plt.close(fig)
        return filepath

    def plot_convergence_history(self, res_omega: List[float], res_psi: List[float],
                                 case_name: str = "baseline"):
        """Plot iteration convergence residuals on semilog scale."""
        fig, ax = plt.subplots(figsize=(7, 4.5))

        iters = np.arange(1, len(res_omega) + 1)
        ax.semilogy(iters, res_omega, 'b-', linewidth=1.6, label="Vorticity Transport Residual $||r_\\omega||_2$")
        ax.semilogy(iters, res_psi, 'r--', linewidth=1.6, label="Streamfunction Poisson Residual $||r_\\psi||_2$")

        ax.set_xlabel("Iteration Number")
        ax.set_ylabel("Normalized $L_2$ Residual")
        ax.set_title("Numerical Convergence History (Navier-Stokes Solver)")
        ax.grid(True, which="both", linestyle=":", alpha=0.6)
        ax.legend(loc="upper right")

        plt.tight_layout()
        filepath = os.path.join(self.output_dir, f"{case_name}_convergence_history.png")
        plt.savefig(filepath)
        plt.close(fig)
        return filepath

    def plot_grid_independence(self, grid_labels: List[str], grid_sizes: List[int],
                               nu_values: List[float], dp_values: List[float]):
        """Plot grid convergence and refinement metrics."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5))

        ax1.plot(grid_sizes, nu_values, 'bo-', linewidth=2, markersize=7)
        ax1.set_xlabel("Number of Grid Nodes $N_x \\times N_y$")
        ax1.set_ylabel("Average Nusselt Number $\\overline{Nu}$")
        ax1.set_title("(a) Grid Independence: Average Nusselt Number")
        ax1.grid(True, linestyle=":", alpha=0.6)

        ax2.plot(grid_sizes, dp_values, 'rs-', linewidth=2, markersize=7)
        ax2.set_xlabel("Number of Grid Nodes $N_x \\times N_y$")
        ax2.set_ylabel("Channel Pressure Drop $\\Delta P$ [Pa]")
        ax2.set_title("(b) Grid Independence: Channel Pressure Drop")
        ax2.grid(True, linestyle=":", alpha=0.6)

        plt.tight_layout()
        filepath = os.path.join(self.output_dir, "grid_independence_study.png")
        plt.savefig(filepath)
        plt.close(fig)
        return filepath

    def plot_parametric_trends(self, df_results):
        """Plot parameter sensitivity studies: velocity, fin gap, inlet temp."""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(11, 8.5))

        # 1. Velocity vs Heat Dissipation & Pressure Drop
        df_v = df_results[df_results["parameter_type"] == "velocity"].sort_values("u_inlet")
        ax1_twin = ax1.twinx()
        l1 = ax1.plot(df_v["u_inlet"], df_v["heat_dissipated_W"], 'b-o', label="Heat Dissipated $Q$ [W]")
        l2 = ax1_twin.plot(df_v["u_inlet"], df_v["pressure_drop_Pa"], 'r--s', label="Pressure Drop $\\Delta P$ [Pa]")
        ax1.set_xlabel("Inlet Velocity $U_{in}$ [m/s]")
        ax1.set_ylabel("Heat Dissipation [W]", color='b')
        ax1_twin.set_ylabel("Pressure Drop [Pa]", color='r')
        ax1.set_title("(a) Thermal vs. Pumping Trade-off with Velocity")
        ax1.grid(True, linestyle=":", alpha=0.5)

        # 2. Velocity vs Thermal Resistance
        ax2.plot(df_v["u_inlet"], df_v["thermal_resistance_K_W"], 'g-^', linewidth=2, markersize=6)
        ax2.set_xlabel("Inlet Velocity $U_{in}$ [m/s]")
        ax2.set_ylabel("Thermal Resistance $R_{th}$ [K/W]")
        ax2.set_title("(b) Thermal Resistance vs. Inflow Velocity")
        ax2.grid(True, linestyle=":", alpha=0.5)

        # 3. Fin Spacing vs Average Nu and Thermal Resistance
        df_g = df_results[df_results["parameter_type"] == "gap_spacing"].sort_values("gap_spacing_mm")
        ax3.plot(df_g["gap_spacing_mm"], df_g["nu_average"], 'm-d', linewidth=2, markersize=6)
        ax3.set_xlabel("Fin Spacing $s$ [mm]")
        ax3.set_ylabel("Average Nusselt Number $\\overline{Nu}$")
        ax3.set_title("(c) Nusselt Number vs. Fin Channel Spacing")
        ax3.grid(True, linestyle=":", alpha=0.5)

        # 4. Thermal Boundary Condition Comparison
        df_bc = df_results[df_results["parameter_type"] == "thermal_bc"]
        bars = ax4.bar(df_bc["bc_type"], df_bc["heat_dissipated_W"], color=["#3b82f6", "#10b981", "#f59e0b"], width=0.5)
        ax4.set_ylabel("Total Heat Dissipated $Q$ [W]")
        ax4.set_title("(d) Thermal Boundary Condition Influence")
        for bar in bars:
            height = bar.get_height()
            ax4.annotate(f'{height:.2f} W',
                         xy=(bar.get_x() + bar.get_width() / 2, height),
                         xytext=(0, 3), textcoords="offset points",
                         ha='center', va='bottom', fontsize=9)
        ax4.grid(True, axis='y', linestyle=":", alpha=0.5)

        plt.tight_layout()
        filepath = os.path.join(self.output_dir, "parametric_analysis_summary.png")
        plt.savefig(filepath)
        plt.close(fig)
        return filepath

    def plot_multi_fin_conjugate(self, X: np.ndarray, Y: np.ndarray, T: np.ndarray,
                                 u: np.ndarray, is_solid: np.ndarray, t_base_height: float):
        """Plot conjugate heat transfer across baseplate and fin channels."""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6.0), sharex=True)

        t_c = T - 273.15

        # 1. Temperature field with solid baseplate region highlighted
        c1 = ax1.contourf(X * 1e3, Y * 1e3, t_c, levels=60, cmap="turbo")
        cb1 = plt.colorbar(c1, ax=ax1, pad=0.02)
        cb1.set_label("Temperature [°C]")
        ax1.axhline(t_base_height * 1e3, color="black", linestyle="--", linewidth=1.5, label="Solid-Fluid Interface")
        ax1.fill_between([X[0, 0]*1e3, X[0, -1]*1e3], 0, t_base_height * 1e3,
                         color="none", hatch="//", edgecolor="white", alpha=0.3, label="Solid Aluminum Baseplate")
        ax1.set_ylabel("Height $y$ [mm]")
        ax1.set_title("(a) Conjugate Heat Transfer: Baseplate Conduction & Inter-Fin Airflow Convection")
        ax1.legend(loc="upper left")

        # 2. Velocity field in fluid
        c2 = ax2.contourf(X * 1e3, Y * 1e3, u, levels=40, cmap="viridis")
        cb2 = plt.colorbar(c2, ax=ax2, pad=0.02)
        cb2.set_label("Streamwise Velocity $u$ [m/s]")
        ax2.axhline(t_base_height * 1e3, color="red", linestyle="--", linewidth=1.2)
        ax2.set_xlabel("Streamwise Distance $x$ [mm]")
        ax2.set_ylabel("Height $y$ [mm]")
        ax2.set_title("(b) Airflow Velocity Distribution Across Heat Sink Channel")

        plt.tight_layout()
        filepath = os.path.join(self.output_dir, "multi_fin_conjugate_temperature.png")
        plt.savefig(filepath)
        plt.close(fig)
        return filepath
