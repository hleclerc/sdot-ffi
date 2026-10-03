#pragma once

// =====================================================================================
// THE IN-HOUSE MULTIGRID of the transport ( `Lin::MG` ), taken from `solvers_des_familles`
// ( `src/solver/Multigrille.h`, README § 17 ) and written for ONE job: the weighted graph
// laplacian of a power diagram, solved over and over along a Newton descent. Included by
// `Linear.cpp` ONLY ( it needs the OpenMP and Eigen switches that unit decides ).
//
// = Aggregation is free
//
// The seeds are numbered in the order of the BSP tree ( `Sweep::rank_of`, `AaBsp` ), which is a
// space-filling curve: consecutive RANKS are neighbours in space. Because the tree is cut at
// the MEDIAN, an aligned window of `2^k` consecutive ranks is ( up to the rounding of odd
// halves ) exactly a subtree, hence a compact box. Aggregating is `rank >> k`: no matching, no
// strength-of-connection graph, no compaction, and the next level aggregates the same way ( `a >> k`
// ). The packet holds 8 seeds ( 2 x 2 x 2 in 3D ); it must stay a power of two, anything else
// would straddle a high-level cut. Without an order ( `order()` not called, or a diagram that has
// no tree ) the identifier order is used, and the aggregation is worth what that order is worth.
//
// The inverse map is free too: packet `a` is the ranks `S a .. S a + S - 1`, so every coarse
// quantity is computed by sweeping coarse rows, one thread per row, with a dense per-thread
// accumulator -- no atomic, no sort, no transposition pass ( `P^t` included ).
//
// = The smoothed prolongation
//
//     P = ( I - w D^-1 A ) P0,    P[ i ][ a ] = ( 1 - w ) [ m_i = a ] + ( w / L_ii ) sum_{ j != i, m_j = a } c_ij
//
// Its rows sum to one, so the constant vector -- the kernel of the laplacian -- is exactly in its
// range, and `A_c = P^t A P` is again a laplacian at every level. `P` has `1 + deg` entries per
// row where `P0` has one, so `A_c` would densify level after level: each row is TRUNCATED to the
// entries worth at least `truncate` ( 0.2 ) of its largest and renormalized ( the renormalization is
// what keeps the constants in the range ). The Galerkin product is two passes, each with a dense
// accumulator per thread: `AP = A P` ( one fine row per thread ), then `A_c = P^t AP` ( one coarse row
// per thread ).
//
// = The smoother is a Chebyshev polynomial
//
// A polynomial of degree `nu` in `M^-1 A`, `M` = the spai0 diagonal ( `m_i = A_ii / sum_j A_ij^2`, the
// diagonal that best approximates `A^-1` in the Frobenius norm ), minimal on `[ lmax / 10, lmax ]` -- the
// part of the spectrum that the coarse level does not correct. `lmax` is BOUNDED, not guessed: Gershgorin
// on `M^-1 A`. Only matrix-vector products, so it is deterministic and a fixed linear operator ( the
// V-cycle is symmetric, as CG as outer iteration requires ). `nu = 1` in 3D ( a 15-entry row already
// carries information far enough: measured 4.8 s against 5.3 s for `nu = 2` at n = 5e5 ), `nu = 3` in 2D.
//
// = The bottom is SOLVED
//
// The coarsest level ( `<= stop` unknowns ) is factorized once per hierarchy by Eigen's sparse LDLT ( seed 0
// struck out: the operator `P A_red^-1 P^t` stays symmetric positive semi-definite, which CG accepts ). Without
// Eigen it is smoothed `bottom_sweeps` times.
//
// = Reused across the Newton iterations ( what makes it cheaper than AMGCL )
//
// * THE HIERARCHY serves `rebuild` ( 4 ) solves: between two Newton iterations the graph moves by a few
//   edges and a preconditioner need not be exact. Only the fine level is re-pointed at the new laplacian
//   and its smoothing coefficients recomputed.
// * THE SUBSPACE: the last `recycle` ( 2 ) solutions are kept and the next solve starts from
//   `x0 = U ( U^t A U )^-1 U^t b`, the best energy approximation in `span( U )` ( Galerkin projection;
//   `k` matrix-vector products and a dense `k x k` system ). Two vectors carry nearly all the information:
//   successive Newton directions span one or two useful degrees of freedom.
//
// = The gauge
//
// The solve is done in the zero-MEAN gauge ( the right-hand side is projected, the preconditioner
// output is centered ): it is symmetric, where striking out a row is not, and the constants are the kernel.
// What comes out is translated to `d[ 0 ] = 0`, the gauge the callers impose ( `w2 = w + t d`, then
// `w2[ 0 ] = 0` ): returning one gauge when the caller expects the other would maim a component and
// Newton stagnates at once ( measured by the old code ).
//
// = What differs from the old code
//
// * Only the winning variant is here: smoothed prolongation, Chebyshev, direct bottom. NOT ported ( research
//   that lost, README § 17 ): the unsmoothed aggregation, the strength filter ( measured useless on a Laguerre
//   graph, whose `c_ij` are all of the same order ), damped Jacobi and plain spai0 smoothing, the K-cycle.
// * The hierarchy and the recycled subspace are keyed on the size of the system, as in the old code; the tree
//   order is given by `LinearSolver::order( rank_of )` ( the rank of each user identifier ) instead of the
//   old `pd.ids` ( the inverse permutation ).
// * Without OpenMP the loops run sequentially ( the pragmas are ignored ).
// =====================================================================================

#include "Linear.h"
#include <algorithm>
#include <chrono>
#include <cmath>

namespace sdot {
namespace sdotplan {

#ifdef _OPENMP
static int mg_nb_threads() { return omp_get_max_threads(); }
static int mg_thread() { return omp_get_thread_num(); }
#else
static int mg_nb_threads() { return 1; }
static int mg_thread() { return 0; }
#endif

static double mg_now() {
    using namespace std::chrono;
    return duration<double>( steady_clock::now().time_since_epoch() ).count();
}

/// under this size a loop stays sequential: the fork/join costs more than the work
static constexpr SI MG_PAR_THRESHOLD = 2048;

/// one level. Level 0 POINTS at the caller's laplacian, the others own their matrix.
/// Convention everywhere: `y_i = dia_i x_i - sum_e val_e x_( col_e )`.
struct MgLevel {
    SI                n = 0, nnz = 0;
    const SI         *row = nullptr, *col = nullptr;
    const double     *val = nullptr, *dia = nullptr;
    std::vector<SI>     orow, ocol;                  ///< ownership, for the coarse levels
    std::vector<double> oval, odia;
    std::vector<double> rlx;                         ///< the spai0 coefficient, ONE PER ROW
    double              lmax = 2;                    ///< Gershgorin bound on `M^-1 A`
    std::vector<double> x, b, r, y, z;               ///< the cycle ( `y`, `z`: the two buffers )
};

/// a rectangular sparse matrix in CSR: `P` ( fine x coarse ) or its transpose
struct MgCsr {
    std::vector<SI>     row, col;
    std::vector<double> val;
};

struct Mg : LinearSolver {
    // ---- the settings; the values are those measured in the old campaign ( README § 17 )
    int    pack      = 8;       ///< seeds per packet -- A POWER OF TWO
    int    nu        = 3;       ///< smoothing steps before and after, per level ( `linear_solver()` sets 1 in 3D )
    double cheb      = 10;      ///< `lmin = lmax / cheb`: the part of the spectrum left to the coarse level
    int    recycle   = 2;       ///< solutions kept for the Galerkin start ( 0: off )
    double omega_p   = 0.7;     ///< the damping of the smoothing of `P`
    double truncate  = 0.2;     ///< the entries of `P` under this fraction of the row's maximum are dropped
    int    stop      = 1000;    ///< coarsening stops under this size
    int    bottom_sweeps = 120; ///< smoothing sweeps at the bottom if there is no factorization
    int    rebuild   = 4;       ///< the hierarchy is rebuilt every `rebuild` solves
    double tol       = 1e-6;    ///< RELATIVE residual
    int    maxit     = 20000;

    const char *name() const override { return "in-house multigrid ( tree aggregation, smoothed prolongation, Chebyshev )"; }

    /// `rank_of[ i ]`: the rank of seed `i` in the tree ( the aggregation is `rank >> log2( pack )` )
    void order( const std::vector<SI> &rank_of ) override {
        rg = rank_of;
        ord.resize( rg.size() );
        for ( SI i = 0; i < SI( rg.size() ); ++i )
            ord[ rg[ i ] ] = i;
        lev.clear();                                 // the aggregation changes: the hierarchy is void
        Uv.clear(); AUv.clear();                     // ... and so is the recycled subspace
    }

    bool solves( const Laplacian &L, const std::vector<double> &b, std::vector<double> &d ) override {
        set_threads();
        const double t0 = mg_now();
        if ( ! lev.empty() && lev[ 0 ].n != L.n ) { Uv.clear(); AUv.clear(); }
        const bool fresh = lev.empty() || lev[ 0 ].n != L.n || lev.size() == 1 || since >= std::max( rebuild, 1 );
        if ( fresh ) { build( L ); since = 0; ++st.nb_hierarchies; }
        else           repoint( L );
        ++since;
        const double t1 = mg_now();
        st.t_hierarchy += t1 - t0;
        const bool ok = cg( b, d );
        st.t_res += mg_now() - t1;
        return ok;
    }

private:
    std::vector<SI>        ord, rg;                  ///< rank -> identifier, and its inverse
    std::vector<MgLevel>   lev;
    std::vector<std::vector<SI>> map;                ///< `map[ l ][ i ]`: the packet of `i` at level `l + 1`
    std::vector<MgCsr>     prol, prolt;              ///< `P` ( fine x coarse ) and `P^t`
    std::vector<double>    cr, cz, cp, cq;           ///< the outer CG
    std::vector<std::vector<double>> Uv, AUv;        ///< the recycled subspace, and `A U` ( redone: `A` changes )
    int                    since = 0;                ///< solves since the last build
    // the buffers of the assemblies, kept from one call to the next
    std::vector<std::vector<double>> acc, tval;
    std::vector<std::vector<SI>>     tcol, lst;
    std::vector<std::vector<char>>   mkc, mkf;
#ifdef SDOT_EIGEN
    Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>, Eigen::Lower, Eigen::AMDOrdering<int>> bottom;
    bool            bottom_ready = false;
    Eigen::VectorXd eb, ex;
#endif

    /// as many OpenMP threads as the pool has workers ( see `Amg::solves` ), unless the user chose
    static void set_threads() {
#ifdef _OPENMP
        if ( ! std::getenv( "OMP_NUM_THREADS" ) )
            if ( const char *nt = std::getenv( "SDOT_NB_THREADS" ) )
                if ( std::atoi( nt ) > 0 )
                    omp_set_num_threads( std::atoi( nt ) );
#endif
    }

    int shift() const {                              // `log2( pack )`, rounded up to a power of two
        int sh = 1;
        while ( ( 1 << sh ) < pack && sh < 16 ) ++sh;
        return sh;
    }
    SI packet_size() const { return SI( 1 ) << shift(); }

    /// the fine index of the `t`-th member of packet `a` at level `l` ( -1 if it does not exist )
    SI member( int l, SI a, SI t, SI nf ) const {
        const SI k = packet_size() * a + t;
        if ( k >= nf ) return -1;
        return ( l == 0 && ! ord.empty() ) ? ord[ k ] : k;
    }

    void prepare_buffers( int T, SI nc, SI nf ) {
        acc.resize( T ); tval.resize( T ); tcol.resize( T ); lst.resize( T );
        mkc.resize( T ); mkf.resize( T );
        for ( int t = 0; t < T; ++t ) {
            acc[ t ].assign( nc, 0.0 );
            mkc[ t ].assign( nc, 0 );
            if ( nf > 0 ) mkf[ t ].assign( nf, 0 );
            tcol[ t ].clear(); tval[ t ].clear(); lst[ t ].clear();
        }
    }

    /// THE CSR WITHOUT SORT OR ATOMIC. Every thread filled its arena in the order of ITS rows: the lengths
    /// are summed and each row knows where it is.
    void assemble_csr( SI nl, const std::vector<SI> &len, const std::vector<SI> &loc, const std::vector<int> &thr,
                       std::vector<SI> &row, std::vector<SI> &col, std::vector<double> &val ) const {
        row.assign( nl + 1, 0 );
        for ( SI i = 0; i < nl; ++i ) row[ i + 1 ] = row[ i ] + len[ i ];
        col.resize( row[ nl ] );
        val.resize( row[ nl ] );
        #pragma omp parallel for schedule( static ) if( nl >= MG_PAR_THRESHOLD )
        for ( SI i = 0; i < nl; ++i ) {
            const int t = thr[ i ];
            const SI  o = loc[ i ], p = row[ i ];
            for ( SI k = 0; k < len[ i ]; ++k ) { col[ p + k ] = tcol[ t ][ o + k ]; val[ p + k ] = tval[ t ][ o + k ]; }
        }
    }

    // ---------------------------------------------------------------- the hierarchy
    void repoint( const Laplacian &L ) {
        lev[ 0 ].row = L.row.data(); lev[ 0 ].col = L.col.data();
        lev[ 0 ].val = L.c.data();   lev[ 0 ].dia = L.dia.data();
        lev[ 0 ].nnz = SI( L.col.size() );
        relax_coefficients( lev[ 0 ] );              // the values changed, so did the coefficients
    }

    /// THE RELAXATION COEFFICIENT, one per row: `m_i = A_ii / sum_j A_ij^2` ( spai0 ), and the Gershgorin bound
    /// of `M^-1 A` ( exact, one pass on the edges )
    void relax_coefficients( MgLevel &v ) const {
        const SI n = v.n;
        v.rlx.resize( n );
        double mx = 0;
        #pragma omp parallel for schedule( static ) if( n >= MG_PAR_THRESHOLD )
        for ( SI i = 0; i < n; ++i ) {
            const double d = v.dia[ i ];
            double q = d * d;
            for ( SI e = v.row[ i ]; e < v.row[ i + 1 ]; ++e ) q += v.val[ e ] * v.val[ e ];
            v.rlx[ i ] = q > 0 ? d / q : 0.0;
        }
        #pragma omp parallel for schedule( static ) reduction( max : mx ) if( n >= MG_PAR_THRESHOLD )
        for ( SI i = 0; i < n; ++i ) {
            double s = v.dia[ i ];
            for ( SI e = v.row[ i ]; e < v.row[ i + 1 ]; ++e ) s += std::fabs( v.val[ e ] );
            mx = std::max( mx, v.rlx[ i ] * s );
        }
        v.lmax = mx > 0 ? mx : 2.0;
    }

    /// `r = b - A x`
    static void residual( MgLevel &v ) {
        const SI n = v.n;
        #pragma omp parallel for schedule( static ) if( n >= MG_PAR_THRESHOLD )
        for ( SI i = 0; i < n; ++i ) {
            double s = v.dia[ i ] * v.x[ i ];
            for ( SI e = v.row[ i ]; e < v.row[ i + 1 ]; ++e ) s -= v.val[ e ] * v.x[ v.col[ e ] ];
            v.r[ i ] = v.b[ i ] - s;
        }
    }

    /// THE CHEBYSHEV SMOOTHER, the three-term recurrence. `deg` matrix-vector products, exactly like `deg` sweeps of
    /// Jacobi -- but a polynomial chosen to crush the high part of the spectrum. `fresh`: start from `x = 0`.
    void chebyshev( MgLevel &v, int deg, bool fresh ) const {
        const SI n = v.n;
        if ( deg <= 0 ) { if ( fresh ) std::fill( v.x.begin(), v.x.end(), 0.0 ); return; }
        const double hi = v.lmax, lo = hi / std::max( cheb, 1.01 );
        const double th = ( hi + lo ) / 2, de = ( hi - lo ) / 2;
        const double si = th / de;
        double rh = 1 / si;
        if ( fresh ) {
            std::fill( v.x.begin(), v.x.end(), 0.0 );
            v.r = v.b;
        } else {
            residual( v );
        }
        #pragma omp parallel for schedule( static ) if( n >= MG_PAR_THRESHOLD )
        for ( SI i = 0; i < n; ++i ) v.y[ i ] = v.rlx[ i ] * v.r[ i ] / th;     // `y` carries the direction
        for ( int k = 0; k < deg; ++k ) {
            #pragma omp parallel for schedule( static ) if( n >= MG_PAR_THRESHOLD )
            for ( SI i = 0; i < n; ++i ) v.x[ i ] += v.y[ i ];
            if ( k + 1 == deg ) break;
            // ONE PASS, with a double buffer: the product `A y` of row `i` reads `y` at the neighbours, so the
            // next direction cannot be written into `y`; but it needs no further pass -- we read `y`, update
            // `r` in place ( each its own index ) and write the next direction into `z`, then swap
            const double r2 = 1 / ( 2 * si - rh ), c1 = r2 * rh, c2 = 2 * r2 / de;
            #pragma omp parallel for schedule( static ) if( n >= MG_PAR_THRESHOLD )
            for ( SI i = 0; i < n; ++i ) {
                double s = v.dia[ i ] * v.y[ i ];
                for ( SI e = v.row[ i ]; e < v.row[ i + 1 ]; ++e ) s -= v.val[ e ] * v.y[ v.col[ e ] ];
                const double ri = v.r[ i ] - s;
                v.r[ i ] = ri;
                v.z[ i ] = c1 * v.y[ i ] + c2 * v.rlx[ i ] * ri;
            }
            v.y.swap( v.z );
            rh = r2;
        }
    }

    void build( const Laplacian &L ) {
        lev.clear(); map.clear(); prol.clear(); prolt.clear();

        MgLevel f;
        f.n = L.n; f.nnz = SI( L.col.size() );
        f.row = L.row.data(); f.col = L.col.data(); f.val = L.c.data(); f.dia = L.dia.data();
        lev.push_back( std::move( f ) );

        for ( int l = 0; lev[ l ].n > stop && l < 24; ++l )
            coarsen( l );

        for ( MgLevel &v : lev ) {
            v.x.assign( v.n, 0.0 ); v.b.assign( v.n, 0.0 );
            v.r.assign( v.n, 0.0 ); v.y.assign( v.n, 0.0 ); v.z.assign( v.n, 0.0 );
            relax_coefficients( v );
        }
        factorize_bottom();
    }

    /// level `l` -> level `l + 1`: the map, `P`, `P^t`, `AP = A P` and `A_c = P^t ( AP )`
    void coarsen( int l ) {
        const SI S = packet_size();
        const SI nf = lev[ l ].n, nc = ( nf + S - 1 ) / S;
        const MgLevel &g = lev[ l ];
        const int T = mg_nb_threads();

        std::vector<SI> &m = map.emplace_back();
        m.resize( nf );
        {
            const bool by_rank = ( l == 0 && ! rg.empty() );
            const int sh = shift();
            #pragma omp parallel for schedule( static ) if( nf >= MG_PAR_THRESHOLD )
            for ( SI i = 0; i < nf; ++i ) m[ i ] = ( by_rank ? rg[ i ] : i ) >> sh;
        }
        prol.emplace_back();
        prolt.emplace_back();
        MgCsr &P = prol.back(), &Pt = prolt.back();

        // ---- 1. `P = ( I - w D^-1 A ) P0`, one FINE row per thread, truncated then renormalized
        {
            prepare_buffers( T, nc, 0 );
            std::vector<SI> len( nf ), loc( nf );
            std::vector<int> thr( nf );
            #pragma omp parallel for schedule( static ) if( nf >= MG_PAR_THRESHOLD )
            for ( SI i = 0; i < nf; ++i ) {
                const int t = mg_thread();
                std::vector<double> &ac = acc[ t ];
                std::vector<char> &mk = mkc[ t ];
                std::vector<SI> &tc = tcol[ t ];
                std::vector<double> &tv = tval[ t ];
                const SI beg = SI( tc.size() );
                double dsum = 0;
                for ( SI e = g.row[ i ]; e < g.row[ i + 1 ]; ++e ) dsum += g.val[ e ];
                const SI ai = m[ i ];
                if ( ! mk[ ai ] ) { mk[ ai ] = 1; tc.push_back( ai ); }
                ac[ ai ] += 1 - omega_p;
                const double f = dsum > 0 ? omega_p / dsum : 0.0;
                for ( SI e = g.row[ i ]; e < g.row[ i + 1 ]; ++e ) {
                    const SI b = m[ g.col[ e ] ];
                    if ( ! mk[ b ] ) { mk[ b ] = 1; tc.push_back( b ); }
                    ac[ b ] += f * g.val[ e ];
                }
                // THE TRUNCATION, THEN THE RENORMALIZATION: what weighs is kept, and the sum is put back to
                // one -- which is what leaves the constant vector in the range of `P`
                double mx = 0;
                for ( SI k = beg; k < SI( tc.size() ); ++k ) mx = std::max( mx, std::fabs( ac[ tc[ k ] ] ) );
                const double thresh = truncate * mx;
                SI kept = beg;
                double sum = 0;
                for ( SI k = beg; k < SI( tc.size() ); ++k ) {
                    const SI b = tc[ k ];
                    const double v = ac[ b ];
                    ac[ b ] = 0; mk[ b ] = 0;
                    if ( ! ( std::fabs( v ) >= thresh ) || v == 0 ) continue;
                    tc[ kept ] = b; tv.push_back( v ); ++kept; sum += v;
                }
                tc.resize( kept );
                if ( sum > 0 )
                    for ( SI k = beg; k < SI( tv.size() ); ++k ) tv[ k ] /= sum;
                loc[ i ] = beg; len[ i ] = kept - beg; thr[ i ] = t;
            }
            assemble_csr( nf, len, loc, thr, P.row, P.col, P.val );
        }

        // ---- 2. `P^t`, WITHOUT A TRANSPOSITION PASS. `P[ i ][ a ] != 0` needs `m_i = a` or a neighbour of `i` in packet
        //         `a`: row `a` of `P^t` is carried by the members of the packet and their neighbours, which the
        //         inverse map gives for free
        {
            prepare_buffers( T, nc, nf );
            std::vector<SI> len( nc ), loc( nc );
            std::vector<int> thr( nc );
            #pragma omp parallel for schedule( static ) if( nc >= MG_PAR_THRESHOLD )
            for ( SI a = 0; a < nc; ++a ) {
                const int t = mg_thread();
                std::vector<char> &mk = mkf[ t ];
                std::vector<SI> &ls = lst[ t ];
                std::vector<SI> &tc = tcol[ t ];
                std::vector<double> &tv = tval[ t ];
                ls.clear();
                for ( SI q = 0; q < S; ++q ) {
                    const SI i = member( l, a, q, nf );
                    if ( i < 0 ) continue;
                    if ( ! mk[ i ] ) { mk[ i ] = 1; ls.push_back( i ); }
                    for ( SI e = g.row[ i ]; e < g.row[ i + 1 ]; ++e ) {
                        const SI j = g.col[ e ];
                        if ( ! mk[ j ] ) { mk[ j ] = 1; ls.push_back( j ); }
                    }
                }
                const SI beg = SI( tc.size() );
                for ( SI i : ls ) {
                    mk[ i ] = 0;
                    for ( SI e = P.row[ i ]; e < P.row[ i + 1 ]; ++e )
                        if ( P.col[ e ] == a ) { tc.push_back( i ); tv.push_back( P.val[ e ] ); break; }
                }
                loc[ a ] = beg; len[ a ] = SI( tc.size() ) - beg; thr[ a ] = t;
            }
            assemble_csr( nc, len, loc, thr, Pt.row, Pt.col, Pt.val );
        }

        // ---- 3. `AP = A P`, one fine row per thread
        MgCsr AP;
        {
            prepare_buffers( T, nc, 0 );
            std::vector<SI> len( nf ), loc( nf );
            std::vector<int> thr( nf );
            #pragma omp parallel for schedule( static ) if( nf >= MG_PAR_THRESHOLD )
            for ( SI i = 0; i < nf; ++i ) {
                const int t = mg_thread();
                std::vector<double> &ac = acc[ t ];
                std::vector<char> &mk = mkc[ t ];
                std::vector<SI> &tc = tcol[ t ];
                std::vector<double> &tv = tval[ t ];
                const SI beg = SI( tc.size() );
                for ( SI e = P.row[ i ]; e < P.row[ i + 1 ]; ++e ) {    // `dia_i P[ i ][ . ]`
                    const SI b = P.col[ e ];
                    if ( ! mk[ b ] ) { mk[ b ] = 1; tc.push_back( b ); }
                    ac[ b ] += g.dia[ i ] * P.val[ e ];
                }
                for ( SI e = g.row[ i ]; e < g.row[ i + 1 ]; ++e ) {    // `- sum_j c_ij P[ j ][ . ]`
                    const SI j = g.col[ e ];
                    const double c = g.val[ e ];
                    for ( SI f = P.row[ j ]; f < P.row[ j + 1 ]; ++f ) {
                        const SI b = P.col[ f ];
                        if ( ! mk[ b ] ) { mk[ b ] = 1; tc.push_back( b ); }
                        ac[ b ] -= c * P.val[ f ];
                    }
                }
                for ( SI k = beg; k < SI( tc.size() ); ++k ) {
                    tv.push_back( ac[ tc[ k ] ] );
                    ac[ tc[ k ] ] = 0; mk[ tc[ k ] ] = 0;
                }
                loc[ i ] = beg; len[ i ] = SI( tc.size() ) - beg; thr[ i ] = t;
            }
            assemble_csr( nf, len, loc, thr, AP.row, AP.col, AP.val );
        }

        // ---- 4. `A_c = P^t ( AP )`, one COARSE row per thread
        MgLevel c;
        c.n = nc;
        {
            prepare_buffers( T, nc, 0 );
            std::vector<SI> len( nc ), loc( nc );
            std::vector<int> thr( nc );
            std::vector<double> dia( nc, 0.0 );
            #pragma omp parallel for schedule( static ) if( nc >= MG_PAR_THRESHOLD )
            for ( SI a = 0; a < nc; ++a ) {
                const int t = mg_thread();
                std::vector<double> &ac = acc[ t ];
                std::vector<char> &mk = mkc[ t ];
                std::vector<SI> &tc = tcol[ t ];
                std::vector<double> &tv = tval[ t ];
                const SI beg = SI( tc.size() );
                for ( SI k = Pt.row[ a ]; k < Pt.row[ a + 1 ]; ++k ) {
                    const SI i = Pt.col[ k ];
                    const double w = Pt.val[ k ];
                    for ( SI f = AP.row[ i ]; f < AP.row[ i + 1 ]; ++f ) {
                        const SI b = AP.col[ f ];
                        if ( ! mk[ b ] ) { mk[ b ] = 1; tc.push_back( b ); }
                        ac[ b ] += w * AP.val[ f ];
                    }
                }
                // THE DIAGONAL LEAVES THE LIST and the off-diagonals change sign: the convention is
                // `y = dia x - sum val x`, so `val_ab = - A_c[ a ][ b ]`
                SI kept = beg;
                for ( SI k = beg; k < SI( tc.size() ); ++k ) {
                    const SI b = tc[ k ];
                    const double v = ac[ b ];
                    ac[ b ] = 0; mk[ b ] = 0;
                    if ( b == a ) { dia[ a ] = v; continue; }
                    tc[ kept ] = b; tv.push_back( -v ); ++kept;
                }
                tc.resize( kept );
                loc[ a ] = beg; len[ a ] = kept - beg; thr[ a ] = t;
            }
            assemble_csr( nc, len, loc, thr, c.orow, c.ocol, c.oval );
            c.nnz = c.orow[ nc ];
            c.odia.resize( nc );
            #pragma omp parallel for schedule( static ) if( nc >= MG_PAR_THRESHOLD )
            for ( SI a = 0; a < nc; ++a ) c.odia[ a ] = dia[ a ] > 0 ? dia[ a ] : 1.0;
        }
        lev.push_back( std::move( c ) );
        MgLevel &v = lev.back();                      // after the move: the buffers have moved
        v.row = v.orow.data(); v.col = v.ocol.data(); v.val = v.oval.data(); v.dia = v.odia.data();
    }

    /// the coarsest level, factorized ONCE per hierarchy; the gauge is `x[ 0 ] = 0` there ( row and column
    /// struck out ): the coarse laplacian is singular, and that suffices
    void factorize_bottom() {
#ifdef SDOT_EIGEN
        bottom_ready = false;
        const MgLevel &g = lev.back();
        const SI m = g.n - 1;
        if ( m <= 0 )
            return;
        std::vector<Eigen::Triplet<double>> tr;
        tr.reserve( size_t( g.nnz + m ) );
        for ( SI i = 1; i < g.n; ++i ) {
            tr.emplace_back( int( i - 1 ), int( i - 1 ), g.dia[ i ] );
            for ( SI e = g.row[ i ]; e < g.row[ i + 1 ]; ++e )
                if ( g.col[ e ] >= 1 )
                    tr.emplace_back( int( i - 1 ), int( g.col[ e ] - 1 ), -g.val[ e ] );
        }
        Eigen::SparseMatrix<double> A( m, m );
        A.setFromTriplets( tr.begin(), tr.end() );
        bottom.compute( A );
        bottom_ready = bottom.info() == Eigen::Success;
        eb.resize( m ); ex.resize( m );
#endif
    }

    // ---------------------------------------------------------------- the blocks
    static void matvec( const MgLevel &v, const std::vector<double> &x, std::vector<double> &y ) {
        const SI n = v.n;
        #pragma omp parallel for schedule( static ) if( n >= MG_PAR_THRESHOLD )
        for ( SI i = 0; i < n; ++i ) {
            double s = v.dia[ i ] * x[ i ];
            for ( SI e = v.row[ i ]; e < v.row[ i + 1 ]; ++e ) s -= v.val[ e ] * x[ v.col[ e ] ];
            y[ i ] = s;
        }
    }
    static double dot( const std::vector<double> &u, const std::vector<double> &v ) {
        const SI n = SI( u.size() );
        double s = 0;
        #pragma omp parallel for schedule( static ) reduction( + : s ) if( n >= MG_PAR_THRESHOLD )
        for ( SI i = 0; i < n; ++i ) s += u[ i ] * v[ i ];
        return s;
    }

    // ---------------------------------------------------------------- the V-cycle
    void cycle( int l ) {
        MgLevel &g = lev[ l ];
        if ( l + 1 == int( lev.size() ) ) {
#ifdef SDOT_EIGEN
            if ( bottom_ready ) {
                for ( SI i = 1; i < g.n; ++i ) eb[ i - 1 ] = g.b[ i ];
                ex = bottom.solve( eb );
                g.x[ 0 ] = 0;
                for ( SI i = 1; i < g.n; ++i ) g.x[ i ] = ex[ i - 1 ];
                return;
            }
#endif
            chebyshev( g, bottom_sweeps, true );
            return;
        }
        MgLevel &c = lev[ l + 1 ];
        const SI nf = g.n, ncc = c.n;

        chebyshev( g, nu, true );
        residual( g );
        // THE RESTRICTION IS A GATHER, never a scatter: no atomic. `P^t` is a CSR product.
        const MgCsr &Pt = prolt[ l ];
        #pragma omp parallel for schedule( static ) if( ncc >= MG_PAR_THRESHOLD )
        for ( SI a = 0; a < ncc; ++a ) {
            double s = 0;
            for ( SI k = Pt.row[ a ]; k < Pt.row[ a + 1 ]; ++k ) s += Pt.val[ k ] * g.r[ Pt.col[ k ] ];
            c.b[ a ] = s;
        }
        cycle( l + 1 );
        const MgCsr &P = prol[ l ];
        #pragma omp parallel for schedule( static ) if( nf >= MG_PAR_THRESHOLD )
        for ( SI i = 0; i < nf; ++i ) {
            double s = 0;
            for ( SI e = P.row[ i ]; e < P.row[ i + 1 ]; ++e ) s += P.val[ e ] * c.x[ P.col[ e ] ];
            g.x[ i ] += s;
        }
        chebyshev( g, nu, false );
    }

    /// the small dense system `G y = f`, `G` symmetric positive definite, `k <= 32`. Hand-written Cholesky with a
    /// safety ridge: two successive solutions can be nearly collinear and `G` singular -- we decline rather than return
    /// anything, the start being only a bonus.
    static bool solve_dense( std::vector<double> &G, std::vector<double> &f, int k ) {
        double tr = 0;
        for ( int j = 0; j < k; ++j ) tr += G[ size_t( j ) * k + j ];
        if ( ! ( tr > 0 ) ) return false;
        const double eps = tr / double( k ) * 1e-12;
        for ( int j = 0; j < k; ++j ) G[ size_t( j ) * k + j ] += eps;
        for ( int j = 0; j < k; ++j ) {                  // Cholesky in place, lower triangle
            double s = G[ size_t( j ) * k + j ];
            for ( int q = 0; q < j; ++q ) s -= G[ size_t( j ) * k + q ] * G[ size_t( j ) * k + q ];
            if ( ! ( s > 0 ) ) return false;
            const double dj = std::sqrt( s );
            G[ size_t( j ) * k + j ] = dj;
            for ( int i = j + 1; i < k; ++i ) {
                double t = G[ size_t( i ) * k + j ];
                for ( int q = 0; q < j; ++q ) t -= G[ size_t( i ) * k + q ] * G[ size_t( j ) * k + q ];
                G[ size_t( i ) * k + j ] = t / dj;
            }
        }
        for ( int i = 0; i < k; ++i ) {                  // forward
            double t = f[ i ];
            for ( int q = 0; q < i; ++q ) t -= G[ size_t( i ) * k + q ] * f[ q ];
            f[ i ] = t / G[ size_t( i ) * k + i ];
        }
        for ( int i = k - 1; i >= 0; --i ) {             // backward
            double t = f[ i ];
            for ( int q = i + 1; q < k; ++q ) t -= G[ size_t( q ) * k + i ] * f[ q ];
            f[ i ] = t / G[ size_t( i ) * k + i ];
        }
        return true;
    }

    /// the gauge of the solve: zero MEAN, the kernel of the laplacian
    static void center( std::vector<double> &v ) {
        const SI n = SI( v.size() );
        if ( n <= 0 ) return;
        double s = 0;
        #pragma omp parallel for schedule( static ) reduction( + : s ) if( n >= MG_PAR_THRESHOLD )
        for ( SI i = 0; i < n; ++i ) s += v[ i ];
        const double mu = s / double( n );
        #pragma omp parallel for schedule( static ) if( n >= MG_PAR_THRESHOLD )
        for ( SI i = 0; i < n; ++i ) v[ i ] -= mu;
    }

    // ---------------------------------------------------------------- the outer CG
    bool cg( const std::vector<double> &b, std::vector<double> &d ) {
        const SI n = lev[ 0 ].n;
        d.assign( n, 0.0 );
        cr.assign( b.begin(), b.begin() + n );
        center( cr );
        cz.assign( n, 0.0 ); cp.assign( n, 0.0 ); cq.assign( n, 0.0 );

        auto precond = [&]( const std::vector<double> &rr, std::vector<double> &zz ) {
            lev[ 0 ].b = rr;
            cycle( 0 );                                  // with a single level, the direct solve of the bottom
            zz = lev[ 0 ].x;
            center( zz );
        };

        const double bb = dot( cr, cr );
        if ( ! ( bb > 0 ) ) return true;                 // nothing to solve

        // ---- THE GALERKIN START on the recycled subspace: `x0 = U ( U^t A U )^-1 U^t b` is the best
        // approximation in `span( U )` in the energy norm, and its residual comes out of the same
        // computation: `r0 = b - ( AU ) y`. The price is `k` matrix-vector products.
        if ( recycle > 0 && ! Uv.empty() ) {
            const int k = int( Uv.size() );
            AUv.resize( k );
            for ( int j = 0; j < k; ++j ) {
                AUv[ j ].resize( n );
                matvec( lev[ 0 ], Uv[ j ], AUv[ j ] );
            }
            std::vector<double> G( size_t( k ) * k ), f( k );
            for ( int j = 0; j < k; ++j ) {
                f[ j ] = dot( Uv[ j ], cr );
                for ( int l = 0; l <= j; ++l ) {
                    const double g = dot( Uv[ j ], AUv[ l ] );
                    G[ size_t( j ) * k + l ] = g;
                    G[ size_t( l ) * k + j ] = g;
                }
            }
            if ( solve_dense( G, f, k ) ) {
                #pragma omp parallel for schedule( static ) if( n >= MG_PAR_THRESHOLD )
                for ( SI i = 0; i < n; ++i ) {
                    double sx = 0, sr = 0;
                    for ( int j = 0; j < k; ++j ) { sx += f[ j ] * Uv[ j ][ i ]; sr += f[ j ] * AUv[ j ][ i ]; }
                    d[ i ] = sx;
                    cr[ i ] -= sr;
                }
                center( d );
                center( cr );
            }
        }

        precond( cr, cz );
        cp = cz;
        double rz = dot( cr, cz );
        const double target = tol * tol * bb;
        double rr = dot( cr, cr );
        int it = 0;
        for ( ; it < maxit && rr > target; ++it ) {
            matvec( lev[ 0 ], cp, cq );
            const double pq = dot( cp, cq );
            if ( ! ( pq > 0 ) ) break;                   // the direction is in the kernel: done
            const double al = rz / pq;
            double s = 0;
            #pragma omp parallel for schedule( static ) reduction( + : s ) if( n >= MG_PAR_THRESHOLD )
            for ( SI i = 0; i < n; ++i ) {
                d[ i ] += al * cp[ i ];
                cr[ i ] -= al * cq[ i ];
                s += cr[ i ] * cr[ i ];
            }
            rr = s;
            if ( rr <= target ) { ++it; break; }
            precond( cr, cz );
            const double rz2 = dot( cr, cz );
            const double be = rz != 0 ? rz2 / rz : 0.0;
            rz = rz2;
            #pragma omp parallel for schedule( static ) if( n >= MG_PAR_THRESHOLD )
            for ( SI i = 0; i < n; ++i ) cp[ i ] = cz[ i ] + be * cp[ i ];
        }
        st.nb_iter += it;
        // THE SOLUTION JOINS THE SUBSPACE, in the zero-mean gauge: the one it is solved in, so the one `U` lives in
        if ( recycle > 0 ) {
            if ( int( Uv.size() ) >= recycle ) Uv.erase( Uv.begin() );
            Uv.push_back( d );
        }
        // WE RETURN THE GAUGE `d[ 0 ] = 0`: see the header
        const double d0 = d[ 0 ];
        #pragma omp parallel for schedule( static ) if( n >= MG_PAR_THRESHOLD )
        for ( SI i = 0; i < n; ++i ) d[ i ] -= d0;
        const double err = std::sqrt( rr / bb );
        st.worst = std::max( st.worst, err );
        return err < 1;
    }
};

} // namespace sdotplan
} // namespace sdot
