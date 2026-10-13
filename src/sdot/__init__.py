# the C++ of this package (`sdot/`, `asimd/`): `include/` next to this file, in a checkout, an editable install and a
# wheel alike -- registered with loom, which compiles the kernels and does not know its users by name
from pathlib import Path as _Path
import os as _os
import loom.compilation as _compilation
_here = _Path( __file__ ).resolve().parent
_compilation.register_include_root( _here / "include" )

# the precompiled kernels (`precompiled-kernels/<tag>/<key>.so`, see `loom/compilation/catalogue.py`): always there, empty
# in a checkout, filled in a wheel. Read-only as far as we are concerned -- the user may well not be allowed to write
# in the installation. For a checkout, `SDOT_CATALOGUE_DIR` can name another one EXPLICITLY (a checkout's headers move, so
# a catalogue registered by default would silently serve stale binaries)
# the transport's linear solvers (`sdot/sdotplan/Linear.cpp`): Eigen, header-only,
# which loom downloads once into its cache at the first compiled kernel (`loom/compilation/externals.py`)
_compilation.register_external( "eigen", "3.4.0", "https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.tar.gz",
                                "8586084f71f9bde545ee7fa6d00288b264a2b7ac3607b974e54d13e7162c1c72" )

from loom.compilation import catalogue as _catalogue
_catalogue.register_catalogue( _os.environ[ "SDOT_CATALOGUE_DIR" ] if "SDOT_CATALOGUE_DIR" in _os.environ else _here / "precompiled-kernels" )

# what was measured about which architecture each kernel runs as (see `loom/choices.py`); tracked in git, never written at run time
import loom as _loom
_loom.register_choices( _here / "choices" )

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

from .quantization import quantization as quantization

from .viz.Visualizer import Visualizer as Visualizer
from .viz.convergence import write_convergence_html as write_convergence_html


# the public functions return arrays of the caller's framework ( see `loom/boundary.py` ); internal
# code calls `.internal`, which hands back the loom `Tensor`
@_loom.public
def plan( f, g ):
    pb = OtProblem( f, g )
    return pb.solve()

@_loom.public
def w2( f, g ):
    return plan.internal( f, g ).cost

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
