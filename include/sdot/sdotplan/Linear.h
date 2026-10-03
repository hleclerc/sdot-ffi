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
//             Cholesky. Ruge-Stuben+GS, which picks its coarse nodes edge by edge, holds on
//             the clouds where smoothed aggregation fails ( incomparable edge weights ).
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

enum class Lin : int { AUTO = 0, CHOLESKY = 1, AMG = 2, CG = 3 };

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
    LinearStats st;
};

/// THE solver for `method` -- AUTO: Cholesky in 2D up to 3e5 seeds if it is there, AMG if it is
/// there, CG otherwise. A requested method that is absent falls back on the next available one.
std::unique_ptr<LinearSolver> linear_solver( Lin method, SI n, int dim );

/// the compiled methods ( a mask: bit `int( Lin::X )` )
int available_linear_methods();

} // namespace sdotplan
} // namespace sdot
