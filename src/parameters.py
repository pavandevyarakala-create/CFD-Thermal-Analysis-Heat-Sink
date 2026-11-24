"""Physical, geometric, and numerical simulation parameters for heat sink CFD analysis."""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class FluidProperties:
    """Thermophysical properties of the working fluid (dry air at ~300 K, 1 atm)."""
    name: str = "Air"
    rho: float = 1.184           # Density [kg/m^3]
    nu: float = 1.56e-5          # Kinematic viscosity [m^2/s]
    cp: float = 1007.0           # Specific heat capacity at constant pressure [J/(kg K)]
    k: float = 0.0262            # Thermal conductivity [W/(m K)]
    alpha: float = 2.197e-5      # Thermal diffusivity alpha = k / (rho * cp) [m^2/s]
    
    @property
    def mu(self) -> float:
        """Dynamic viscosity [Pa s]."""
        return self.rho * self.nu

    @property
    def prandtl(self) -> float:
        """Prandtl number Pr = nu / alpha."""
        return self.nu / self.alpha


@dataclass
class SolidProperties:
    """Thermophysical properties of the heat sink material (Aluminum 6061)."""
    name: str = "Aluminum 6061"
    k: float = 205.0             # Thermal conductivity [W/(m K)]
    rho: float = 2700.0          # Density [kg/m^3]
    cp: float = 900.0            # Specific heat [J/(kg K)]


@dataclass
class ChannelGeometry:
    """Geometric dimensions for the 2D inter-fin channel model."""
    length: float = 0.040        # Channel length H_x [m] (40 mm)
    height: float = 0.0029       # Fin spacing / channel height L [m] (2.9 mm)
    fin_thickness: float = 0.0010# Fin thickness t_fin [m] (1.0 mm)
    fin_depth: float = 0.050     # Transverse fin depth W [m] (50 mm into page)
    
    @property
    def aspect_ratio(self) -> float:
        """Channel length to height aspect ratio."""
        return self.length / self.height

    @property
    def hydraulic_diameter(self) -> float:
        """Hydraulic diameter for wide parallel plates: D_h = 2 * height."""
        return 2.0 * self.height


@dataclass
class BoundaryConditions:
    """Thermal and hydrodynamic boundary specifications."""
    u_inlet: float = 1.5         # Inlet flow velocity [m/s]
    v_inlet: float = 0.0         # Inlet vertical velocity [m/s]
    t_inlet: float = 300.0       # Inlet air temperature [K]
    t_wall_bottom: float = 345.0 # Bottom fin wall temperature [K]
    t_wall_top: float = 345.0    # Top fin wall temperature [K]
    thermal_bc_type: str = "isothermal" # "isothermal", "isoflux", "asymmetric"
    heat_flux_wall: float = 1200.0 # Heat flux [W/m^2] for isoflux condition


@dataclass
class NumericalSettings:
    """Numerical solver convergence, mesh, and relaxation controls."""
    nx: int = 121                # Number of grid points in x-direction
    ny: int = 41                 # Number of grid points in y-direction
    max_iter: int = 8000         # Maximum outer iterations
    tolerance: float = 1e-6      # Convergence tolerance on L2 residual norm
    sor_omega_psi: float = 1.5   # Successive Over-Relaxation factor for streamfunction
    cfl_safety: float = 0.4      # Courant-Friedrichs-Lewy safety factor
    upwind_blending: float = 1.0 # 1.0 = Pure upwind (monotone), 0.0 = Pure central


@dataclass
class SimulationCase:
    """Encapsulates a full simulation setup."""
    name: str = "Baseline_S1"
    fluid: FluidProperties = field(default_factory=FluidProperties)
    solid: SolidProperties = field(default_factory=SolidProperties)
    geometry: ChannelGeometry = field(default_factory=ChannelGeometry)
    bc: BoundaryConditions = field(default_factory=BoundaryConditions)
    numeric: NumericalSettings = field(default_factory=NumericalSettings)

    @property
    def reynolds_gap(self) -> float:
        """Reynolds number based on fin gap height: Re_s = U * L / nu."""
        return (self.bc.u_inlet * self.geometry.height) / self.fluid.nu

    @property
    def reynolds_dh(self) -> float:
        """Reynolds number based on hydraulic diameter: Re_Dh = U * D_h / nu."""
        return (self.bc.u_inlet * self.geometry.hydraulic_diameter) / self.fluid.nu

    @property
    def peclet_gap(self) -> float:
        """Peclet number based on gap height: Pe = U * L / alpha."""
        return (self.bc.u_inlet * self.geometry.height) / self.fluid.alpha

    @property
    def peclet_dh(self) -> float:
        """Peclet number based on hydraulic diameter: Pe_Dh = U * D_h / alpha."""
        return (self.bc.u_inlet * self.geometry.hydraulic_diameter) / self.fluid.alpha
