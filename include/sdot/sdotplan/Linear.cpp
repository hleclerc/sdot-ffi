// THE DOMAIN UNIT of the linear solvers ( see `Linear.h` ): compiled once per
// compiler, linked by the kernels that name `sdot/sdotplan/Linear.cpp` in their `sources`.
// Eigen and AMGCL are those that loom downloads ( `sdot/__init__.py` -> `loom/compilation/externals.py`,
// on the include path ), or failing that those of the system ( `<eigen3/...>` ); what is missing is
// simply not offered.

#include "Linear.h"
#include <algorithm>
#include <chrono>
#include <cmath>
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

#if __has_include( <Eigen/SparseCholesky> )
#  define SDOT_EIGEN 1
#  include <Eigen/SparseCholesky>
#  include <Eigen/SparseCore>
#elif __has_include( <eigen3/Eigen/SparseCholesky> )
#  define SDOT_EIGEN 1
#  include <eigen3/Eigen/SparseCholesky>
#  include <eigen3/Eigen/SparseCore>
#endif

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
    int    variant = RS_GS;
    double tol      = 1e-10;       ///< RELATIVE residual
    int    maxit    = 20000;

    const char *name() const override {
        return variant == RS_GS ? "AMGCL Ruge-Stuben+GS"
             : variant == SA_GS ? "AMGCL aggregation+GS" : "AMGCL aggregation+spai0";
    }

    bool solves( const Laplacian &L, const std::vector<double> &b, std::vector<double> &d ) override {
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
        if      ( variant == SA_GS ) launch( std::type_identity<SaGs>{} );
        else if ( variant == RS_GS ) launch( std::type_identity<RsGs>{} );
        else                          launch( std::type_identity<SaSpai>{} );
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
    int res = 1 << int( Lin::CG );
#ifdef SDOT_EIGEN
    res |= 1 << int( Lin::CHOLESKY );
#endif
#ifdef SDOT_AMGCL
    res |= 1 << int( Lin::AMG );
#endif
    return res;
}

std::unique_ptr<LinearSolver> linear_solver( Lin method, SI n, int dim ) {
    const int available = available_linear_methods();
    if ( method == Lin::AUTO )
        method = dim <= 2 && n <= 300000 ? Lin::CHOLESKY : Lin::AMG;
    if ( method == Lin::CHOLESKY && ! ( available & ( 1 << int( Lin::CHOLESKY ) ) ) ) method = Lin::AMG;
    if ( method == Lin::AMG      && ! ( available & ( 1 << int( Lin::AMG      ) ) ) ) method = Lin::CHOLESKY;
    if ( ! ( available & ( 1 << int( method ) ) ) )                                   method = Lin::CG;
#ifdef SDOT_EIGEN
    if ( method == Lin::CHOLESKY ) return std::make_unique<Cholesky>();
#endif
#ifdef SDOT_AMGCL
    if ( method == Lin::AMG ) return std::make_unique<Amg>();
#endif
    return std::make_unique<Cg>();
}

} // namespace sdotplan
} // namespace sdot
