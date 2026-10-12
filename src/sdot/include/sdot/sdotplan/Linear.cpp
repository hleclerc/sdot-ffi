// THE DOMAIN UNIT of the linear solvers ( see `Linear.h` ): compiled once per
// compiler, linked by the kernels that name `sdot/sdotplan/Linear.cpp` in their `sources`.
// Eigen is the one that loom downloads ( `sdot/__init__.py` -> `loom/compilation/externals.py`,
// on the include path ), or failing that the system's ( `<eigen3/...>` ); what is missing is
// simply not offered. No OpenMP anywhere: the parallel loops run on loom's thread pool.

#include "Linear.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <stdexcept>
#include <tuple>
#include <cstdio>
#include <cstdint>
#if defined( __APPLE__ )
#  include <sys/types.h>
#  include <sys/sysctl.h>
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

#include "Multigrid.h"                   // the in-house multigrid: needs the Eigen switch above

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
    return res;
}

/// the memory the machine can still give ( bytes; 0: unknown )
static double available_memory() {
#if defined( __linux__ )
    if ( FILE *f = std::fopen( "/proc/meminfo", "r" ) ) {
        char line[ 256 ];
        double kb = 0;
        while ( std::fgets( line, sizeof line, f ) )
            if ( std::sscanf( line, "MemAvailable: %lf kB", &kb ) == 1 )
                break;
        std::fclose( f );
        return kb * 1024;
    }
    return 0;
#elif defined( __APPLE__ )
    std::uint64_t bytes = 0;
    std::size_t size = sizeof bytes;
    return sysctlbyname( "hw.memsize", &bytes, &size, nullptr, 0 ) == 0 ? 0.5 * double( bytes ) : 0;   // no "available" there: half of it
#else
    return 0;
#endif
}

/// the peak memory of the sparse Cholesky on a 2D laplacian of `n` seeds: the factor of a planar graph under the AMD ordering
/// is O( n log n ) ( measured, `CHOLESKY_BYTES_PER_NLOGN` ); in 3D it grows like n^(4/3) ( rough upper bound )
static double cholesky_bytes( SI n, int dim ) {
    constexpr double CHOLESKY_BYTES_PER_NLOGN = 40.0;   // measured ~35 ( peak RSS, 2D uniform, n = 1e6 )
    if ( dim >= 3 )
        return 60.0 * std::pow( double( n ), 4.0 / 3.0 );
    return CHOLESKY_BYTES_PER_NLOGN * double( n ) * std::log2( double( std::max<SI>( n, 2 ) ) );
}

/// the Cholesky is there, and fits in the memory ( at most half of what is available )
static bool cholesky_fits( SI n, int dim ) {
#ifdef SDOT_EIGEN
    const double have = available_memory();
    return have <= 0 || cholesky_bytes( n, dim ) <= 0.5 * have;
#else
    (void) n; (void) dim;
    return false;
#endif
}

std::unique_ptr<LinearSolver> linear_solver( Lin method, SI n, int dim, const LinearOptions &opts ) {
    // AUTO. 3D: the in-house multigrid ( `Multigrid.h` ) -- the Cholesky takes minutes there ( 4 s against 800 s of linear algebra at
    // n = 1e5 ). 2D: the Cholesky up to the crossover below, if its factor fits in the memory; the multigrid otherwise.
    // The crossover in 2D ( 8 pinned threads, whole Newton solves, linear part ): uniform 1.7 s / 3.0 s ( Cholesky / multigrid ) at 1e5,
    // 11.4 / 15.1 at 3e5, 89 / 75 at 1e6 -- the Cholesky is O( n^1.5 ) in time, the multigrid O( n ); the lines clouds keep the Cholesky
    // ahead much longer ( 2.2 s against 12 s at 1e5 ). Its factor weighs ~35 bytes x n log2 n ( 0.7 GB over the multigrid at 1e6 ).
    constexpr SI CHOLESKY_MAX_AUTO_2D = 600000;
    if ( method == Lin::AUTO )
        method = dim >= 3 || n > CHOLESKY_MAX_AUTO_2D ? Lin::MG : Lin::CHOLESKY;
    // `AMG` was AMGCL, which is gone: its place is taken by the multigrid ( `LinearOptions::amg_variant` is ignored )
    if ( method == Lin::AMG )
        method = Lin::MG;
    if ( method == Lin::CHOLESKY && ! cholesky_fits( n, dim ) )
        method = Lin::MG;
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
#ifdef SDOT_EIGEN
    if ( method == Lin::CHOLESKY ) return std::make_unique<Cholesky>();
#endif
    auto res = std::make_unique<Cg>();
    if ( opts.tol > 0 ) res->tol = opts.tol;
    return res;
}

} // namespace sdotplan
} // namespace sdot
