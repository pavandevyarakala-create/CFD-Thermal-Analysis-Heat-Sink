"""CFD and Thermal Analysis package for Heat Sink Airflow Simulation."""

from .parameters import (
    FluidProperties,
    SolidProperties,
    ChannelGeometry,
    BoundaryConditions,
    NumericalSettings,
    SimulationCase
)
from .grid import StructuredGrid2D
from .navier_stokes_solver import NavierStokes2DSolver
from .thermal_solver import Thermal2DSolver
from .analytical_benchmarks import AnalyticalBenchmarks
from .postprocessing import PostProcessor
from .visualizer import Visualizer
from .multi_fin_solver import MultiFinHeatSink2D

__all__ = [
    "FluidProperties",
    "SolidProperties",
    "ChannelGeometry",
    "BoundaryConditions",
    "NumericalSettings",
    "SimulationCase",
    "StructuredGrid2D",
    "NavierStokes2DSolver",
    "Thermal2DSolver",
    "AnalyticalBenchmarks",
    "PostProcessor",
    "Visualizer",
    "MultiFinHeatSink2D"
]
