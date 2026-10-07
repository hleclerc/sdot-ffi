# the C++ of this package (`sdot/`, `asimd/`): `<repo>/sdot/include` from a checkout, the
# `sdot/_include` tree the wheel ships next to the package otherwise -- registered with loom,
# which compiles the kernels and does not know its users by name
from pathlib import Path as _Path
import os as _os
import loom.compilation as _compilation
_here = _Path( __file__ ).resolve().parent
_compilation.register_include_root( _here / "_include" if ( _here / "_include" ).is_dir() else _here.parents[ 1 ] / "include" )

# the catalogue of precompiled kernels, when there is one: a wheel's `sdot/_catalogue`, built with
# these very headers -- or, for a checkout, the directory `SDOT_CATALOGUE_DIR` names EXPLICITLY (a
# checkout's headers move, a catalogue registered by default would silently serve stale binaries)
# the transport's linear solvers (`sdot/sdotplan/Linear.cpp`): Eigen and AMGCL, header-only,
# which loom downloads once into its cache at the first compiled kernel (`loom/compilation/externals.py`)
_compilation.register_external( "eigen", "3.4.0", "https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.tar.gz",
                                "8586084f71f9bde545ee7fa6d00288b264a2b7ac3607b974e54d13e7162c1c72" )
_compilation.register_external( "amgcl", "1.4.4", "https://github.com/ddemidov/amgcl/archive/refs/tags/1.4.4.tar.gz",
                                "02fd5418e14d669422f65fc739ce72bf9516ced2d8942574d4b8caa05dda9d8c" )

from loom.compilation import catalogue as _catalogue
_catalogue.register_catalogue( _os.environ[ "SDOT_CATALOGUE_DIR" ] if "SDOT_CATALOGUE_DIR" in _os.environ else _here / "_catalogue" )

from .AaBsp import AaBsp as AaBsp
from .Cell import Cell as Cell
from .Cell_1 import Cell_1 as Cell_1
from .Cell_2 import Cell_2 as Cell_2
from .Cell_N import Cell_N as Cell_N
from .CellScratch import CellScratch as CellScratch
from .Cell import set_kernel_dtype as set_kernel_dtype
from .Cell import kernel_dtype as kernel_dtype
# THE TRANSPORT: a problem is POSED ( `OtProblem` ), then asked for a solution -- direct
# in 1D ( `SdotPlan1d` ), iterative beyond ( `SdotPlanNd` ). See the docstring of `OtProblem`.
from .OtProblem import OtProblem as OtProblem
from .OtProblem import Direct as Direct
from .OtProblem import Iterative as Iterative
from .OtProblem import Tuning as Tuning
from .OtProblem import OtNotConverged as OtNotConverged
from .OtProblem import ot_solve as ot_solve
from .SdotPlan1d import SdotPlan1d as SdotPlan1d
from .SdotPlanNd import SdotPlanNd as SdotPlanNd
from .PowerDiagram import PowerDiagram as PowerDiagram
from .PowerDiagram import box_half_spaces as box_half_spaces
from .PowerDiagram_Bsp import PowerDiagram_Bsp as PowerDiagram_Bsp
from .PowerDiagram_Plain import PowerDiagram_Plain as PowerDiagram_Plain
from .SpatialAccelerator import SpatialAccelerator as SpatialAccelerator
from .Voronoi import Voronoi as Voronoi

from .distributions.SumOfDiracs1d import SumOfDiracs1d as SumOfDiracs1d
from .distributions.SumOfDiracs import SumOfDiracs as SumOfDiracs
from .distributions.ProjectedSumOfDiracs import ProjectedSumOfDiracs as ProjectedSumOfDiracs
from .distributions.Image import Image as Image
from .distributions.Box import Box as Box
from .distributions.Mesh import Mesh as Mesh
from .distributions.Polytope import Polytope as Polytope
from .distributions.Polytope import Polygon as Polygon
from .distributions.Polytope import Polyhedron as Polyhedron
from .distributions.SumOfGaussians import SumOfGaussians as SumOfGaussians

from .viz.Visualizer import Visualizer as Visualizer
from .viz.convergence import write_convergence_html as write_convergence_html


# the old names, until callers move to `OtProblem` ( see
# `notes/2026-10-02-sdotplan.md`: `OtPlan1d` did not say it was semi-discrete, and `OtPlan` said
# neither its dimension nor its regime )
def __getattr__( name ):
    import warnings
    deprecated = { "OtPlan": "SdotPlanNd", "OtPlan1d": "SdotPlan1d" }
    if name in deprecated:
        warnings.warn( f"sdot.{ name } is deprecated: use `sdot.{ deprecated[ name ] }`, which "
                       "is obtained through `OtProblem( source, target ).solve( ... )`.",
                       DeprecationWarning, stacklevel = 2 )
        return globals()[ deprecated[ name ] ]
    raise AttributeError( f"module { __name__ !r } has no attribute { name !r}" )
