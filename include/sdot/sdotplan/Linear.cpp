// THE DOMAIN UNIT of the linear solvers ( see `Linear.h` ): compiled once per
// compiler, linked by the kernels that name `sdot/sdotplan/Linear.cpp` in their `sources`.
// Eigen and AMGCL are those that loom downloads ( `sdot/__init__.py` -> `loom/compilation/externals.py`,
// on the include path ), or failing that those of the system ( `<eigen3/...>` ); what is missing is
// simply not offered.

#include "Linear.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <stdexcept>
#include <tuple>

#if __has_include( <amgcl/make_solver.hpp> )
#  define SDOT_AMGCL 1
#  ifndef AMGCL_NO_BOOST
#    define AMGCL_NO_BOOST
#  endif
#  include <amgcl/adapter/crs_tuple.hpp>
#  include <amgcl/amg.hpp>
#  include <amgcl/backend/builtin.hpp>
#  include <amgcl/coarsening/ruge_stuben.hpp>
#  include <amgcl/coarsening/smoothed_aggregation.hpp>
#  include <amgcl/make_solver.hpp>
#  include <amgcl/relaxation/gauss_seidel.hpp>
#  include <amgcl/relaxation/spai0.hpp>
#  include <amgcl/solver/cg.hpp>
#endif

#ifdef _OPENMP
#  include <omp.h>
#  include <cstdlib>
#endif

#if __has_include( <Eigen/SparseCholesky> )
#  define SDOT_EIGEN 1
#  include <Eigen/SparseCholesky>
#  include <Eigen/SparseCore>
#elif __has_include( <eigen3/Eigen/SparseCholesky> )
#  define SDOT_EIGEN 1
#  include <eigen3/Eigen/SparseCholesky>
#  include <eigen3/Eigen/SparseCore>
#endif

#include "Multigrid.h"                   // the in-house multigrid: needs the OpenMP / Eigen switches above

namespace sdot {
namespace sdotplan {

static double now() {
    using namespace std::chrono;
    return duration<double>( steady_clock::now().time_since_epoch() ).count();
}

// ---- the conjugate gradient, always there ---------------------------------------------------------

/// Jacobi-preconditioned CG on the full system, the gauge held by projecting `d_0 = 0`:
/// we solve on the unknowns `1 .. n-1`, reading the full matrix and ignoring column 0.
struct Cg : LinearSolver {
    double tol = 1e-12;
    int    maxit = 20000;

    const char *name() const override { return "conjugate gradient ( Jacobi )"; }

    bool solves( const Laplacian &L, const std::vector<double> &b, std::vector<double> &d ) override {
        const SI n = L.n;
        const double t0 = now();
        std::vector<double> r( n ), z( n ), p( n ), q( n );
        d.assign( n, 0.0 );
        auto product = [&]( const std::vector<double> &x, std::vector<double> &y ) {
            L.product( x.data(), y.data() );
            y[ 0 ] = 0;
        };
        double nb = 0;
        for ( SI i = 1; i < n; ++i ) { r[ i ] = b[ i ]; nb += b[ i ] * b[ i ]; }
        r[ 0 ] = 0;
        nb = std::sqrt( nb );
        if ( ! ( nb > 0 ) ) { st.t_res += now() - t0; return true; }
        double rz = 0;
        for ( SI i = 1; i < n; ++i ) { z[ i ] = r[ i ] / L.dia[ i ]; p[ i ] = z[ i ]; rz += r[ i ] * z[ i ]; }
        double err = 1;
        int it = 0;
        for ( ; it < maxit; ++it ) {
            product( p, q );
            double pq = 0;
            for ( SI i = 1; i < n; ++i ) pq += p[ i ] * q[ i ];
            if ( ! ( pq > 0 ) ) break;
            const double alpha = rz / pq;
            double nr = 0;
            for ( SI i = 1; i < n; ++i ) { d[ i ] += alpha * p[ i ]; r[ i ] -= alpha * q[ i ]; nr += r[ i ] * r[ i ]; }
            err = std::sqrt( nr ) / nb;
            if ( err <= tol ) break;
            double rz2 = 0;
            for ( SI i = 1; i < n; ++i ) { z[ i ] = r[ i ] / L.dia[ i ]; rz2 += r[ i ] * z[ i ]; }
            const double beta = rz2 / rz;
            rz = rz2;
            for ( SI i = 1; i < n; ++i ) p[ i ] = z[ i ] + beta * p[ i ];
        }
        st.nb_iter += it;
        st.worst = std::max( st.worst, err );
        st.t_res += now() - t0;
        return err < 1;
    }
};

// ---- AMGCL ----------------------------------------------------------------------------------------

#ifdef SDOT_AMGCL
struct Amg : LinearSolver {
    enum Variant : int { SA_SPAI0 = 0, SA_GS = 1, RS_GS = 2 };
    int    variant = SA_SPAI0;     ///< the default of the old `newton` bench
    double tol      = 1e-6;        ///< RELATIVE residual: a DAMPED direction needs no more ( README § 17.2: 1e-4 starts to cost diagrams ); the old `newton` bench ran at 1e-10
    int    maxit    = 20000;

    const char *name() const override {
        return variant == RS_GS ? "AMGCL Ruge-Stuben+GS"
             : variant == SA_GS ? "AMGCL aggregation+GS" : "AMGCL aggregation+spai0";
    }

    bool solves( const Laplacian &L, const std::vector<double> &b, std::vector<double> &d ) override {
#ifdef _OPENMP
        // as many OpenMP threads as the pool has workers, unless the user chose ( `OMP_NUM_THREADS` ): more threads
        // than the pinned pool's cores oversubscribes them ( measured: 3D planes, 12.0 s of linear algebra with 16
        // threads against 2.7 s with 8 )
        if ( ! std::getenv( "OMP_NUM_THREADS" ) )
            if ( const char *nt = std::getenv( "SDOT_NB_THREADS" ) )
                if ( std::atoi( nt ) > 0 )
                    omp_set_num_threads( std::atoi( nt ) );
#endif
        const SI n = L.n, m = n - 1;
        const double t0 = now();
        std::vector<int> ptr, col;
        std::vector<double> val;
        L.reduced_crs( ptr, col, val );
        std::vector<double> rb( m ), sol( m, 0.0 );
        for ( SI i = 1; i < n; ++i )
            rb[ i - 1 ] = b[ i ];
        const double t1 = now();
        st.t_build += t1 - t0;

        using Back = amgcl::backend::builtin<double>;
        int it = 0;
        double err = 0;
        auto launch = [&]( auto tag ) {
            using Solv = typename decltype( tag )::type;
            typename Solv::params prm;
            prm.solver.tol = tol;
            prm.solver.maxiter = maxit;
            Solv so( std::tie( m, ptr, col, val ), prm );
            const double ta = now();
            st.t_hierarchy += ta - t1;
            ++st.nb_hierarchies;
            std::tie( it, err ) = so( rb, sol );
            st.t_res += now() - ta;
        };
        using SaSpai = amgcl::make_solver<amgcl::amg<Back, amgcl::coarsening::smoothed_aggregation, amgcl::relaxation::spai0>, amgcl::solver::cg<Back>>;
        using SaGs   = amgcl::make_solver<amgcl::amg<Back, amgcl::coarsening::smoothed_aggregation, amgcl::relaxation::gauss_seidel>, amgcl::solver::cg<Back>>;
        using RsGs   = amgcl::make_solver<amgcl::amg<Back, amgcl::coarsening::ruge_stuben, amgcl::relaxation::gauss_seidel>, amgcl::solver::cg<Back>>;
        // AMGCL THROWS where the others say `false`: a singular system ( a facet graph split into components, each one
        // without the gauge being a singular block -- a kept start that empties cells of a density zero on most of its
        // domain ) reaches the direct solver of the coarsest level, whose LU stops on a zero pivot ( `precondition`,
        // "Zero sum in skyline_lu factorization" ). Through the FFI call it would be a crash instead of a status.
        try {
            if      ( variant == SA_GS ) launch( std::type_identity<SaGs>{} );
            else if ( variant == RS_GS ) launch( std::type_identity<RsGs>{} );
            else                          launch( std::type_identity<SaSpai>{} );
        } catch ( const std::exception & ) {
            st.t_res += now() - t1;
            return false;
        }
        st.nb_iter += it;
        st.worst = std::max( st.worst, err );

        d.assign( n, 0.0 );
        for ( SI i = 1; i < n; ++i )
            d[ i ] = sol[ i - 1 ];
        return err < 1;                                  // `1`: the solver did nothing at all
    }
};
#endif

// ---- Eigen ----------------------------------------------------------------------------------------

#ifdef SDOT_EIGEN
struct Cholesky : LinearSolver {
    using SpM = Eigen::SparseMatrix<double>;
    Eigen::SimplicialLDLT<SpM, Eigen::Lower, Eigen::AMDOrdering<int>> so;
    std::vector<SI> pattern;         ///< the pattern of the last symbolic analysis

    const char *name() const override { return "sparse Cholesky ( Eigen LDLT, AMD )"; }

    /// THE PATTERN BARELY MOVES: a few edges per iteration at the start, zero at the end, whereas
    /// the renumbering and the symbolic analysis cost a third of the factorization. They are
    /// redone ONLY when the pattern has changed -- compared as is, one linear pass.
    bool solves( const Laplacian &L, const std::vector<double> &b, std::vector<double> &d ) override {
        const SI n = L.n, m = n - 1;
        const double t0 = now();
        std::vector<Eigen::Triplet<double>> tri;
        tri.reserve( size_t( L.row[ n ] ) / 2 + n );
        for ( SI i = 1; i < n; ++i ) {
            tri.emplace_back( int( i - 1 ), int( i - 1 ), L.dia[ i ] );
            for ( SI k = L.row[ i ]; k < L.row[ i + 1 ]; ++k ) {
                const SI j = L.col[ k ];
                if ( j >= 1 && j < i )                   // the LOWER triangle only
                    tri.emplace_back( int( i - 1 ), int( j - 1 ), -L.c[ k ] );
            }
        }
        SpM A( m, m );
        A.setFromTriplets( tri.begin(), tri.end() );
        const double t1 = now();
        st.t_build += t1 - t0;

        std::vector<SI> new_pattern( A.outerIndexPtr(), A.outerIndexPtr() + m + 1 );
        new_pattern.insert( new_pattern.end(), A.innerIndexPtr(), A.innerIndexPtr() + A.nonZeros() );
        if ( new_pattern != pattern ) {
            so.analyzePattern( A );
            pattern.swap( new_pattern );
            ++st.nb_hierarchies;
        }
        const double t2 = now();
        st.t_hierarchy += t2 - t1;

        so.factorize( A );
        if ( so.info() != Eigen::Success ) { st.t_res += now() - t2; return false; }
        Eigen::VectorXd rb( m );
        for ( SI i = 1; i < n; ++i )
            rb[ i - 1 ] = b[ i ];
        const Eigen::VectorXd sol = so.solve( rb );
        st.t_res += now() - t2;
        if ( so.info() != Eigen::Success )
            return false;

        d.assign( n, 0.0 );
        for ( SI i = 1; i < n; ++i )
            d[ i ] = sol[ i - 1 ];
        return true;
    }
};
#endif

// ---- the choice -----------------------------------------------------------------------------------

int available_linear_methods() {
    int res = ( 1 << int( Lin::CG ) ) | ( 1 << int( Lin::MG ) );
#ifdef SDOT_EIGEN
    res |= 1 << int( Lin::CHOLESKY );
#endif
#ifdef SDOT_AMGCL
    res |= 1 << int( Lin::AMG );
#endif
    return res;
}

std::unique_ptr<LinearSolver> linear_solver( Lin method, SI n, int dim, const LinearOptions &opts ) {
    const int available = available_linear_methods();
    // AUTO. 3D: the in-house multigrid ( `Multigrid.h` ), as in the old `newton` bench -- measured, 8 pinned threads, n = 1e5, linear
    // part: uniform 0.53 s against 1.07 s for AMGCL, planes 1.12 s against 1.96 s; at n = 5e5 AMGCL stagnates where it converges.
    // 2D: the builtin backend of AMGCL is parallel through OpenMP ONLY. With it ( `-fopenmp` ), AMG wins from 2D uniform on
    // ( 0.58 s of linear algebra against 0.77 s for MG and 1.78 s for Cholesky, n = 1e5; lines: 2.0 / 3.3 / 2.0 ) -- the old bench chose it
    // in 2D too. Without it, AMG is sequential and 4x slower than that ( 3.6 s ): Cholesky stays the choice in 2D up to a few hundred
    // thousand seeds.
#ifdef _OPENMP
    constexpr bool amg_is_parallel = true;
#else
    constexpr bool amg_is_parallel = false;
#endif
    if ( method == Lin::AUTO )
        method = dim >= 3 ? Lin::MG : ( n <= 300000 && ! amg_is_parallel ) ? Lin::CHOLESKY : Lin::AMG;
    if ( method == Lin::MG ) {
        auto res = std::make_unique<Mg>();
        if ( dim >= 3 ) res->nu = 1;                     // measured: a 15-entry row carries information far enough ( see `Multigrid.h` )
        if ( opts.tol > 0 ) res->tol = opts.tol;
        if ( opts.mg_pack > 0 ) res->pack = opts.mg_pack;
        if ( opts.mg_recycle >= 0 ) res->recycle = opts.mg_recycle;
        if ( opts.mg_nu > 0 ) res->nu = opts.mg_nu;
        if ( opts.mg_stop > 0 ) res->stop = opts.mg_stop;
        if ( opts.mg_rebuild > 0 ) res->rebuild = opts.mg_rebuild;
        return res;
    }
    if ( method == Lin::CHOLESKY && ! ( available & ( 1 << int( Lin::CHOLESKY ) ) ) ) method = Lin::AMG;
    if ( method == Lin::AMG      && ! ( available & ( 1 << int( Lin::AMG      ) ) ) ) method = Lin::CHOLESKY;
    if ( ! ( available & ( 1 << int( method ) ) ) )                                   method = Lin::CG;
#ifdef SDOT_EIGEN
    if ( method == Lin::CHOLESKY ) return std::make_unique<Cholesky>();
#endif
#ifdef SDOT_AMGCL
    if ( method == Lin::AMG ) {
        auto res = std::make_unique<Amg>();
        if ( opts.tol > 0 ) res->tol = opts.tol;
        if ( opts.amg_variant >= 0 ) res->variant = opts.amg_variant;
        return res;
    }
#endif
    auto res = std::make_unique<Cg>();
    if ( opts.tol > 0 ) res->tol = opts.tol;
    return res;
}

} // namespace sdotplan
} // namespace sdot
