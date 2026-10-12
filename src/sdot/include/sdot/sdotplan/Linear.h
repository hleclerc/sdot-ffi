#pragma once

// =====================================================================================
// THE LINEAR SOLVERS OF THE TRANSPORT: `L d = b` on the reduced system, the gauge `d_0 = 0` being
// their responsibility. The linear solver is not the subject of the study and must not become
// it: they are header-only libraries, in ONE domain unit compiled once
// ( `Linear.cpp`, named by `FfiCode( sources = ... )` ) and linked by the kernel that calls them --
// Eigen and AMGCL have no business being in every generated unit.
//
//   CHOLESKY  Eigen `SimplicialLDLT`, AMD reordering, the symbolic analysis redone ONLY
//             when the pattern changes. Scalar and sequential, but it wins on the lines cloud
//             at n=1e5, and in 2D up to a few hundred thousand seeds.
//   AMG       AMGCL, algebraic multigrid + CG. The hessian is the laplacian of an almost
//             planar graph: the cost stays O( n ). 4.9x on the total at n=1e6 against
//             Cholesky. The default is that of the old `newton` bench: smoothed aggregation + spai0
//             ( parallel on both sides ), `LinearOptions::amg_variant` selects Ruge-Stuben+GS, which
//             picks its coarse nodes edge by edge and holds on the clouds where smoothed aggregation
//             fails ( incomparable edge weights ). The builtin backend of AMGCL is parallel through
//             OPENMP only: see `Linear.cpp` for what that takes.
//   MG        the in-house multigrid of the old campaign ( `Multigrid.h` ): aggregation by the order of the BSP
//             tree, smoothed prolongation, Chebyshev smoother, direct bottom, and a hierarchy and a recycled subspace
//             REUSED from one Newton iteration to the next. Needs OpenMP to be parallel ( like the AMGCL backend ).
//   CG        the conjugate gradient preconditioned by Jacobi, written here: what remains when neither
//             Eigen nor AMGCL are there ( `__has_include` ). Five times slower than Cholesky, but
//             always available.
//
// Measured in `solvers_des_familles` ( README § 3, § 10 ): Cholesky 11.9 s against Ruge-Stuben
// 15.2 s on the 2D lines at 1e5; on the uniform it is the opposite, and at 1e6 Cholesky does not
// scale. AUTO chooses on the dimension and the size.
// =====================================================================================

#include "Laplacian.h"
#include <memory>

namespace sdot {
namespace sdotplan {

enum class Lin : int { AUTO = 0, CHOLESKY = 1, AMG = 2, CG = 3, MG = 4 };

/// what a linear solver reports, accumulated over all its solves.
struct LinearStats {
    double t_build = 0;        ///< the shaping of the matrix ( CRS, triplets )
    double t_hierarchy  = 0;        ///< the AMG hierarchy, or the symbolic analysis
    double t_res   = 0;        ///< the solve proper ( or factorization + back-substitution )
    int    nb_hierarchies = 0;        ///< how many hierarchies / analyses
    int    nb_iter = 0;        ///< CG iterations, in all
    double worst    = 0;        ///< the worst relative residual returned
    double total() const { return t_build + t_hierarchy + t_res; }
};

struct LinearSolver {
    virtual ~LinearSolver() {}
    /// `d[ 0 ] == 0` on output. Returns `false` if the solver could not.
    virtual bool solves( const Laplacian &L, const std::vector<double> &b, std::vector<double> &d ) = 0;
    virtual const char *name() const = 0;
    /// the ORDER OF THE TREE: `rank_of[ i ]` is the rank of seed `i` in the BSP tree ( consecutive ranks are neighbours in
    /// space ). Only the multigrid uses it ( aggregation by rank ); called once, before the first solve.
    virtual void order( const std::vector<SI> &rank_of ) { (void) rank_of; }
    LinearStats st;
};

/// the settings of the solvers that have some ( `0` / `-1`: the default of the solver )
struct LinearOptions {
    double tol = 0;           ///< AMG, CG: the RELATIVE residual to reach ( AMG default: 1e-6, README § 17.2; the old `newton` bench ran 1e-10 )
    int    amg_variant = -1;  ///< AMG: 0 aggregation + spai0 ( the default ), 1 aggregation + Gauss-Seidel, 2 Ruge-Stuben + Gauss-Seidel
    int    mg_pack = 0;       ///< MG: seeds per aggregate, a power of two ( default 8 )
    int    mg_recycle = -1;   ///< MG: solutions kept for the start by projection ( default 2, 0: off )
    int    mg_nu = 0;         ///< MG: Chebyshev smoothing steps per level ( default 1 in 3D, 3 in 2D )
    int    mg_stop = 0;       ///< MG: coarsening stops under this many unknowns ( default 1000 )
    int    mg_rebuild = 0;    ///< MG: the hierarchy is rebuilt every that many solves ( default 4 )
};

/// THE solver for `method` -- AUTO: the multigrid in 3D; in 2D AMG when it is compiled with OpenMP ( parallel ), else Cholesky up to 3e5
/// seeds, AMG if it is there, CG otherwise. A requested method that is absent falls back on the next available one.
std::unique_ptr<LinearSolver> linear_solver( Lin method, SI n, int dim, const LinearOptions &opts = {} );

/// the compiled methods ( a mask: bit `int( Lin::X )` )
int available_linear_methods();

} // namespace sdotplan
} // namespace sdot
