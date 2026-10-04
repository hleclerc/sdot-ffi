#pragma once

// =====================================================================================
// NEWTON'S LINEAR SYSTEM ON THE CARD: `L d = b`, `L` the laplacian of the Laguerre graph in CSR and IN RANKS
// ( `Laplacian2D.cuh`: `y_i = dia_i x_i - sum_e val_e x_( col_e )`, `L = L^T`, `L 1 = 0` to the bit ), `b` of zero sum.
// The gauge is the ZERO MEAN ( the kernel of `L` is the constants ): what comes out is centred, and the caller
// moves it to the gauge it wants.
//
// Three ways, chosen per call ( `SdotPlanNd`'s `linear_solver`, `Tuning.linear_host` ):
//
//   CG     the conjugate gradient preconditioned by Jacobi, all on the card. `sqrt( n )` iterations ( thousands at
//          1e6 ): the baseline, always there.
//   MG     the multigrid of the old GPU campaign ( `gpu_des_familles/src/gpu/Amg2D.cuh`, doc/06 ): AGGREGATION BY TREE
//          RANK ( `rank >> shift`, packets of 4 by default: consecutive ranks are neighbours in space, an aligned
//          window is a subtree, so the hierarchy is a shift ), the unsmoothed Galerkin product ( on a laplacian it is
//          a laplacian again: the sum of the edge weights between packets ), a CHEBYSHEV smoother of degree `nu` in
//          `M^-1 A` ( `M` the spai0 diagonal, `lmax` bounded by Gershgorin; at degree 1, the default, pre- and
//          post-smoothing are one fused kernel each ), the K-CYCLE ( levels 1 .. `kcycle` accelerated by two flexible-CG
//          steps preconditioned by the level below: the remedy of the old campaign to the weak coarse correction of
//          unsmoothed aggregation ), the coarsest level ( <= 64 unknowns ) SOLVED EXACTLY ( a dense inverse built in one
//          block ), RECYCLING ( the last solutions as a Galerkin start ), a hierarchy REUSED for `rebuild` solves ( only
//          the fine level is repointed ), and every iteration REPLAYED FROM CUDA GRAPHS ( some 150 small launches each ).
//          The outer iteration is a FLEXIBLE CG ( Polak-Ribiere `beta` ): the K-cycle is not a fixed linear operator.
//          Measured at 1e6 ( `calibration_lmo_today.md`, GPU step 4 ): levels in float ( `TV`, the outer iteration in
//          double ) and four lanes per CSR row were both SLOWER -- the cycle is bound by its many small dependent
//          kernels, not by bytes -- they stay as an option and a constant.
//   HOST   the CSR copied to the host, one of the CPU solvers of `sdotplan/Linear.cpp` ( Cholesky, AMGCL, the CPU
//          multigrid, CG ), `d` copied back: what the card does not have, and the reference.
//
// The coarse assembly MERGES the fine rows of a packet ( each sorted by column, so `S` sorted runs once mapped to packets ),
// sums the duplicates and drops the packet itself: no atomic, no sort, the same CSR at every run. Every scalar the
// iterations need ( `alpha`, `beta`, the K-cycle's coefficients, `lmax` ) stays on the card, computed by the kernels that
// read it; the host reads back the residual of the outer iteration only ( every `check` iterations for CG ), and one count
// per coarse level when a hierarchy is built.
// =====================================================================================

#include "Laplacian2D.cuh"
#include "Reduce.cuh"
#include "../sdotplan/Linear.h"
#include <chrono>
#include <cstdlib>
#include <memory>
#include <string>
#include <vector>

namespace sdot::gpu2d {

inline double wall_now() {
    using namespace std::chrono;
    return duration<double>( steady_clock::now().time_since_epoch() ).count();
}

/// a laplacian in CSR ( off-diagonal `val > 0`, the minus sign in the product )
/// ( `TV`: the type of the values -- double for Newton's system, float for the levels of a single-precision multigrid )
template<class TC,class TV = double>
struct CsrView {
    SI        n   = 0;
    const SI *row = nullptr;
    const TC *col = nullptr;
    const TV *val = nullptr, *dia = nullptr;
};

/// `( A x )_i`, accumulated in double whatever the storage
template<class TC,class TV,class TX>
__device__ __forceinline__ double row_product( const CsrView<TC,TV> &A, const TX *x, SI i ) {
    double s = double( A.dia[ i ] ) * double( x[ i ] );
    for ( SI e = A.row[ i ]; e < A.row[ i + 1 ]; ++e )
        s -= double( A.val[ e ] ) * double( x[ A.col[ e ] ] );
    return s;
}

/// THE ROW PRODUCT BY `LANES` LANES: thread `t` takes the entries `sub, sub + LANES, ...` of row `t / LANES`, shuffles sum
/// them ( all the lanes get the result ). Every lane of the warp must call it. MEASURED, and kept at ONE lane per row: four
/// lanes ( a warp reading eight consecutive rows nearly contiguously ) made the 1e6 solve's linear part 10 % SLOWER
/// ( 0.745 s against 0.68 ), the coarse levels paying for four times the threads.
constexpr int LANES = 1;

template<class TC,class TV,class TX>
__device__ __forceinline__ double row_product_lanes( const CsrView<TC,TV> &A, const TX *x, SI i, int sub, bool valid ) {
    double s = 0;
    if ( valid ) {
        if ( sub == 0 )
            s = double( A.dia[ i ] ) * double( x[ i ] );
        for ( SI e = A.row[ i ] + sub; e < A.row[ i + 1 ]; e += LANES )
            s -= double( A.val[ e ] ) * double( x[ A.col[ e ] ] );
    }
    for ( int o = 1; o < LANES; o *= 2 )
        s += __shfl_xor_sync( 0xffffffffu, s, o );
    return s;
}

/// the threads of a lane kernel over `n` rows: a whole number of warps ( every lane reaches the shuffles )
inline SI lane_items( SI n ) { return ( LANES * n + 31 ) / 32 * 32; }

struct LinOptions {
    int    method        = 1;      ///< 0 CG ( Jacobi ), 1 MG
    double tol           = 1e-6;   ///< relative residual
    int    maxit         = 20000;
    int    shift         = 2;      ///< MG: packets of `2^shift` ranks
    int    nu            = 1;      ///< MG: Chebyshev degree, before and after
    double cheb          = 10;     ///< MG: `lmin = lmax / cheb`
    int    recycle       = 2;      ///< MG: solutions kept for the Galerkin start
    int    rebuild       = 1;      ///< MG: solves per hierarchy
    int    stop          = 64;     ///< MG: coarsening stops under this size ( `<= BOTTOM_DENSE`: an exact bottom; else smoothed, at most 2048 )
    int    kcycle        = 2;      ///< MG: levels 1 .. kcycle are K-cycled
    int    bottom_sweeps = 60;     ///< MG: damped Jacobi sweeps at the bottom
    double bottom_omega  = 0.7;
    int    check         = 4;      ///< CG: the residual is read back every that many iterations
    bool   trace         = false;
};

struct LinStats {
    double t_hierarchy = 0, t_res = 0;
    int    nb_hierarchies = 0, nb_iter = 0;
    double worst = 0;
    double total() const { return t_hierarchy + t_res; }
};

// ---- the conjugate gradient ( Jacobi ) ------------------------------------------------------------------------------

struct CgScalars {
    double rz, alpha, beta, rr;
    int    stop;
};

/// `r = b`, `p = b / dia`, `x = 0`; `( r.r, r.z )`
struct CgInit {
    const double *b, *dia;
    double       *r, *p, *x;
    __device__ void operator()( SI i, Sum2 &acc ) const {
        const double ri = b[ i ], zi = ri / dia[ i ];
        r[ i ] = ri; p[ i ] = zi; x[ i ] = 0;
        acc.s0 += ri * ri; acc.s1 += ri * zi;
    }
};

/// `y = A x`; `x . y` ( four lanes per row: over `lane_items( n )` items )
template<class TC>
struct SpmvDot {
    CsrView<TC>   A;
    const double *x;
    double       *y;
    __device__ void operator()( SI t, Sum1 &acc ) const {
        const SI i = t / LANES;
        const int sub = int( t % LANES );
        const bool valid = i < A.n;
        const double v = row_product_lanes( A, x, i, sub, valid );
        if ( valid && sub == 0 ) {
            y[ i ] = v;
            acc.s += x[ i ] * v;
        }
    }
};

/// `x += alpha p`, `r -= alpha q`; `( r.r, r.( r / dia ) )`
struct CgUpdate {
    const double    *p, *q, *dia;
    double          *x, *r;
    const CgScalars *sc;
    __device__ void operator()( SI i, Sum2 &acc ) const {
        const double a = sc->alpha;
        x[ i ] += a * p[ i ];
        const double ri = r[ i ] - a * q[ i ];
        r[ i ] = ri;
        acc.s0 += ri * ri;
        if ( dia ) acc.s1 += ri * ri / dia[ i ];
    }
};

__global__ void cg_start( CgScalars *sc, const Sum2 *s ) { sc->rr = s->s0; sc->rz = s->s1; sc->alpha = sc->beta = 0; sc->stop = 0; }
__global__ void cg_alpha( CgScalars *sc, const Sum1 *pq ) {
    if ( pq->s > 0 ) sc->alpha = sc->rz / pq->s;
    else { sc->alpha = 0; sc->stop = 1; }               // the direction is in the kernel: done
}
__global__ void cg_beta( CgScalars *sc, const Sum2 *s ) {
    sc->beta = sc->rz != 0 ? s->s1 / sc->rz : 0.0;
    sc->rz = s->s1;
    sc->rr = s->s0;
}
__global__ void __launch_bounds__( BLOCK ) cg_direction( SI n, const double *r, const double *dia, double *p, const CgScalars *sc ) {
    const SI i = SI( blockIdx.x ) * BLOCK + threadIdx.x;
    if ( i < n ) p[ i ] = r[ i ] / dia[ i ] + sc->beta * p[ i ];
}

// ---- the multigrid: smoother, transfers, bottom, K-cycle -------------------------------------------------------------

/// spai0 and the Gershgorin bound of `M^-1 A`
template<class TC,class TV>
struct Relax {
    CsrView<TC,TV> A;
    TV            *rlx;
    __device__ void operator()( SI i, Max1 &acc ) const {
        const double d = A.dia[ i ];
        double q = d * d, s = d;
        for ( SI e = A.row[ i ]; e < A.row[ i + 1 ]; ++e ) { const double v = A.val[ e ]; q += v * v; s += fabs( v ); }
        const double rl = q > 0 ? d / q : 0.0;
        rlx[ i ] = TV( rl );
        acc.m = fmax( acc.m, rl * s );
    }
};

/// the Chebyshev interval from the bound read on the card
__device__ __forceinline__ void cheb_interval( const Max1 *lmax, double cheb, double &th, double &de ) {
    const double hi = lmax->m > 0 ? lmax->m : 2.0, lo = hi / fmax( cheb, 1.01 );
    th = ( hi + lo ) / 2;
    de = ( hi - lo ) / 2;
}

/// the first step: `r = b - A x` ( `b` if `fresh`, then `x = 0` ), `y = M r / theta`
template<class TC,class TV>
__global__ void __launch_bounds__( BLOCK ) cheb_init( CsrView<TC,TV> A, const TV *b, TV *x, TV *r, TV *y, const TV *rlx,
                                                     const Max1 *lmax, double cheb, int fresh ) {
    const SI i = SI( blockIdx.x ) * BLOCK + threadIdx.x;
    if ( i >= A.n )
        return;
    double th, de;
    cheb_interval( lmax, cheb, th, de );
    const double ri = fresh ? b[ i ] : b[ i ] - row_product( A, x, i );
    if ( fresh ) x[ i ] = TV( 0 );
    r[ i ] = TV( ri );
    y[ i ] = TV( rlx[ i ] * ri / th );
}

/// step `k`: `x += y`, `r -= A y`, `z = c1 y + c2 M r` -- ONE kernel ( `y` is read at the neighbours, so the next direction
/// goes to `z` and the caller swaps )
template<class TC,class TV>
__global__ void __launch_bounds__( BLOCK ) cheb_step( CsrView<TC,TV> A, TV *x, TV *r, const TV *y, TV *z, const TV *rlx,
                                                     const Max1 *lmax, double cheb, int k ) {
    const SI i = SI( blockIdx.x ) * BLOCK + threadIdx.x;
    if ( i >= A.n )
        return;
    double th, de;
    cheb_interval( lmax, cheb, th, de );
    const double si = th / de;
    double rh = 1 / si;
    for ( int j = 0; j < k; ++j )
        rh = 1 / ( 2 * si - rh );
    const double r2 = 1 / ( 2 * si - rh ), c1 = r2 * rh, c2 = 2 * r2 / de;
    const double yi = y[ i ];
    x[ i ] += yi;
    const double ri = r[ i ] - row_product( A, y, i );
    r[ i ] = TV( ri );
    z[ i ] = TV( c1 * yi + c2 * rlx[ i ] * ri );
}

/// DEGREE ONE, FUSED ( the default ): the smoother is then `x += M ( b - A x ) / theta`. Before the coarse correction, from zero:
/// `x = M b / theta`; after it, OUT OF PLACE ( `x` is read at the neighbours ): `x = xp + M ( b - A xp ) / theta`, `xp` the
/// prolonged iterate. Two kernels per visit instead of six.
template<class TV>
__global__ void __launch_bounds__( BLOCK ) smooth_fresh( SI n, const TV *b, const TV *rlx, const Max1 *lmax, double cheb, TV *x ) {
    const SI i = SI( blockIdx.x ) * BLOCK + threadIdx.x;
    if ( i >= n )
        return;
    double th, de;
    cheb_interval( lmax, cheb, th, de );
    x[ i ] = TV( double( rlx[ i ] ) * b[ i ] / th );
}

template<class TC,class TV>
__global__ void __launch_bounds__( BLOCK ) smooth_post( CsrView<TC,TV> A, const TV *b, const TV *xp, const TV *rlx, const Max1 *lmax,
                                                       double cheb, TV *x ) {
    const SI t = SI( blockIdx.x ) * BLOCK + threadIdx.x, i = t / LANES;
    const int sub = int( t % LANES );
    const bool valid = i < A.n;
    const double ax = row_product_lanes( A, xp, i, sub, valid );
    if ( ! valid || sub )
        return;
    double th, de;
    cheb_interval( lmax, cheb, th, de );
    x[ i ] = TV( xp[ i ] + double( rlx[ i ] ) * ( b[ i ] - ax ) / th );
}

/// `dst = src + e[ i >> sh ]` ( the prolongation, out of place )
template<class TV>
__global__ void __launch_bounds__( BLOCK ) prolong_into( SI n, const TV *src, const TV *e, int sh, TV *dst ) {
    const SI i = SI( blockIdx.x ) * BLOCK + threadIdx.x;
    if ( i < n ) dst[ i ] = TV( src[ i ] + e[ i >> sh ] );
}

template<class TV>
__global__ void __launch_bounds__( BLOCK ) add_into( SI n, TV *x, const TV *y ) {
    const SI i = SI( blockIdx.x ) * BLOCK + threadIdx.x;
    if ( i < n ) x[ i ] = TV( x[ i ] + y[ i ] );
}

/// `bc[ a ] = sum_{ i in packet a } ( b - A x )_i`: one thread per FINE row ( neighbouring threads read neighbouring rows ),
/// the `2^sh` lanes of a packet summed by shuffles ( a packet is aligned on its lanes: `2^sh <= 32` divides the block )
template<class TC,class TV>
__global__ void __launch_bounds__( BLOCK ) restrict_residual( CsrView<TC,TV> A, const TV *b, const TV *x, TV *bc, int sh ) {
    const SI t = SI( blockIdx.x ) * BLOCK + threadIdx.x, i = t / LANES;
    const int sub = int( t % LANES );
    const bool valid = i < A.n;
    const double ax = row_product_lanes( A, x, i, sub, valid );
    double v = valid && sub == 0 ? b[ i ] - ax : 0.0;
    const int W = LANES << sh;                           // the lanes of a packet ( `<= 32` )
    for ( int o = W / 2; o > 0; o /= 2 )
        v += __shfl_down_sync( 0xffffffffu, v, o, W );
    if ( valid && ( t & ( W - 1 ) ) == 0 )
        bc[ i >> sh ] = TV( v );
}

template<class TV>
__global__ void __launch_bounds__( BLOCK ) prolong_add( SI n, TV *x, const TV *e, int sh ) {
    const SI i = SI( blockIdx.x ) * BLOCK + threadIdx.x;
    if ( i < n ) x[ i ] = TV( x[ i ] + e[ i >> sh ] );
}

/// THE COARSEST LEVEL IN ONE BLOCK: damped Jacobi from zero, the iterates in shared memory ( a sweep is a loop with a
/// barrier, not a launch )
template<class TC,class TV>
__global__ void __launch_bounds__( 256 ) bottom_jacobi( CsrView<TC,TV> A, const TV *b, TV *x, double omega, int sweeps ) {
    extern __shared__ double shm[];
    const int n = int( A.n );
    double *u = shm, *v = shm + n;
    for ( int i = threadIdx.x; i < n; i += blockDim.x ) u[ i ] = 0;
    __syncthreads();
    for ( int k = 0; k < sweeps; ++k ) {
        for ( int i = threadIdx.x; i < n; i += blockDim.x ) {
            double s = double( A.dia[ i ] ) * u[ i ];
            for ( SI e = A.row[ i ]; e < A.row[ i + 1 ]; ++e ) s -= double( A.val[ e ] ) * u[ A.col[ e ] ];
            v[ i ] = u[ i ] + omega * ( double( b[ i ] ) - s ) / double( A.dia[ i ] );
        }
        __syncthreads();
        double *t = u; u = v; v = t;
    }
    for ( int i = threadIdx.x; i < n; i += blockDim.x ) x[ i ] = TV( u[ i ] );
}

/// THE COARSEST LEVEL SOLVED EXACTLY ( `m <= BOTTOM_DENSE` unknowns ): `( A + c 1 1^T )^-1`, built once per hierarchy in ONE
/// block ( Gauss-Jordan in place, in shared memory ), applied as a dense product. `A + c 1 1^T` is positive definite on a
/// connected laplacian, and for `b` of zero sum its solution has zero sum and solves `A x = b`: the zero-mean gauge,
/// exactly. ( The old campaign smoothed the bottom: 60 Jacobi sweeps in one block cost a quarter of the solve at 1e5. )
constexpr int BOTTOM_DENSE = 64;

template<class TC,class TV>
__global__ void __launch_bounds__( 256 ) bottom_invert( CsrView<TC,TV> A, double *inv ) {
    __shared__ double M[ BOTTOM_DENSE * BOTTOM_DENSE ];
    __shared__ double col[ BOTTOM_DENSE ];
    const int m = int( A.n );
    double c = 0;
    for ( int i = 0; i < m; ++i ) c += double( A.dia[ i ] );
    c /= double( m ) * m;
    for ( int q = threadIdx.x; q < m * m; q += blockDim.x ) M[ q ] = c;
    __syncthreads();
    for ( int i = threadIdx.x; i < m; i += blockDim.x ) {
        M[ i * m + i ] += double( A.dia[ i ] );
        for ( SI e = A.row[ i ]; e < A.row[ i + 1 ]; ++e ) M[ i * m + int( A.col[ e ] ) ] -= double( A.val[ e ] );
    }
    __syncthreads();
    for ( int k = 0; k < m; ++k ) {
        const double p = 1 / M[ k * m + k ];
        __syncthreads();
        for ( int j = threadIdx.x; j < m; j += blockDim.x ) M[ k * m + j ] = j == k ? p : M[ k * m + j ] * p;
        for ( int i = threadIdx.x; i < m; i += blockDim.x ) col[ i ] = i == k ? 0.0 : M[ i * m + k ];
        __syncthreads();
        for ( int q = threadIdx.x; q < m * m; q += blockDim.x ) {
            const int i = q / m, j = q % m;
            if ( i != k ) M[ q ] = ( j == k ? 0.0 : M[ q ] ) - col[ i ] * M[ k * m + j ];
        }
        __syncthreads();
    }
    for ( int q = threadIdx.x; q < m * m; q += blockDim.x ) inv[ q ] = M[ q ];
}

template<class TV>
__global__ void __launch_bounds__( 64 ) bottom_apply( int m, const double *inv, const TV *b, TV *x ) {
    __shared__ double sb[ BOTTOM_DENSE ];
    for ( int i = threadIdx.x; i < m; i += blockDim.x ) sb[ i ] = b[ i ];
    __syncthreads();
    for ( int i = threadIdx.x; i < m; i += blockDim.x ) {
        double s = 0;
        for ( int j = 0; j < m; ++j ) s += inv[ i * m + j ] * sb[ j ];
        x[ i ] = TV( s );
    }
}

/// K-cycle, first step: `t = A v1`; `( v1 . t, v1 . b )`
template<class TC,class TV>
struct KStep1 {
    CsrView<TC,TV> A;
    const TV *v1, *bk;
    TV           *t;
    __device__ void operator()( SI q, Sum2 &acc ) const {
        const SI i = q / LANES;
        const int sub = int( q % LANES );
        const bool valid = i < A.n;
        const double ti = row_product_lanes( A, v1, i, sub, valid );
        if ( ! valid || sub ) return;
        t[ i ] = TV( ti );
        acc.s0 += double( v1[ i ] ) * ti;
        acc.s1 += double( v1[ i ] ) * double( bk[ i ] );
    }
};

template<class TV>
__global__ void __launch_bounds__( BLOCK ) kcycle_rc( SI n, const TV *bk, const TV *t, TV *rc, const Sum2 *s ) {
    const SI i = SI( blockIdx.x ) * BLOCK + threadIdx.x;
    if ( i >= n )
        return;
    const double c = s->s0 != 0 ? s->s1 / s->s0 : 0.0;
    rc[ i ] = TV( bk[ i ] - c * t[ i ] );
}

/// K-cycle, second step: `( v2 . A v1, v2 . A v2, v2 . rc )`
template<class TC,class TV>
struct KStep2 {
    CsrView<TC,TV> A;
    const TV *v2, *t, *rc;
    __device__ void operator()( SI q, Sum3 &acc ) const {
        const SI i = q / LANES;
        const int sub = int( q % LANES );
        const bool valid = i < A.n;
        const double t2 = row_product_lanes( A, v2, i, sub, valid );
        if ( ! valid || sub ) return;
        acc.s0 += double( v2[ i ] ) * double( t[ i ] );
        acc.s1 += double( v2[ i ] ) * t2;
        acc.s2 += double( v2[ i ] ) * double( rc[ i ] );
    }
};

template<class TV>
__global__ void __launch_bounds__( BLOCK ) kcycle_combine( SI n, const TV *v1, const TV *v2, TV *e, const Sum2 *s1, const Sum3 *s2 ) {
    const SI i = SI( blockIdx.x ) * BLOCK + threadIdx.x;
    if ( i >= n )
        return;
    const double rho1 = s1->s0, a1 = s1->s1, g2 = s2->s0, b2 = s2->s1, a2 = s2->s2;
    const double rho2 = b2 - g2 * g2 / ( rho1 != 0 ? rho1 : 1.0 );
    const double c2 = rho2 != 0 ? a2 / rho2 : 0.0;
    const double c1 = ( rho1 != 0 ? a1 / rho1 : 0.0 ) - ( rho1 != 0 && rho2 != 0 ? g2 * a2 / ( rho1 * rho2 ) : 0.0 );
    e[ i ] = TV( c1 * v1[ i ] + c2 * v2[ i ] );
}

// ---- the coarse assembly ---------------------------------------------------------------------------------------------

/// THE COARSE ROW `a` AS A MERGE: the fine rows of packet `a` are each sorted by column ( `lap_sort`, and `coarse_fill` writes
/// in key order ), so their columns mapped to packets ( `col >> sh` ) are `S` sorted runs -- merged by their heads, the
/// duplicates summed in the order of the runs, the packet itself dropped. No sort and no buffer: the count pass and the fill
/// pass run the same merge. `f( key, sum )` per distinct coarse column.
template<class TC,class TV,class F>
__device__ __forceinline__ void merge_packet( const CsrView<TC,TV> &A, int sh, SI a, F &&f ) {
    constexpr int MAXS = 32;
    SI p[ MAXS ], e[ MAXS ];
    const int S = 1 << sh;
    const SI r0 = a << sh;
    int nr = 0;
    for ( int q = 0; q < S && r0 + q < A.n; ++q, ++nr ) { p[ q ] = A.row[ r0 + q ]; e[ q ] = A.row[ r0 + q + 1 ]; }
    while ( true ) {
        SI k = -1;
        for ( int q = 0; q < nr; ++q )
            if ( p[ q ] < e[ q ] ) {
                const SI c = SI( A.col[ p[ q ] ] ) >> sh;
                if ( k < 0 || c < k ) k = c;
            }
        if ( k < 0 )
            return;
        double s = 0;
        for ( int q = 0; q < nr; ++q )
            for ( ; p[ q ] < e[ q ] && ( SI( A.col[ p[ q ] ] ) >> sh ) == k; ++p[ q ] )
                s += A.val[ p[ q ] ];
        if ( k != a )
            f( k, s );
    }
}

template<class TC,class TV>
__global__ void __launch_bounds__( BLOCK ) coarse_count( CsrView<TC,TV> A, int sh, SI nc, SI *cnt ) {
    const SI a = SI( blockIdx.x ) * BLOCK + threadIdx.x;
    if ( a >= nc )
        return;
    SI c = 0;
    merge_packet( A, sh, a, [&]( SI, double ) { ++c; } );
    cnt[ a ] = c;
}

template<class TC,class TV>
__global__ void __launch_bounds__( BLOCK ) coarse_fill( CsrView<TC,TV> A, int sh, SI nc, const SI *crow, TC *ccol, TV *cval, TV *cdia ) {
    const SI a = SI( blockIdx.x ) * BLOCK + threadIdx.x;
    if ( a >= nc )
        return;
    SI q = crow[ a ];
    double d = 0;
    merge_packet( A, sh, a, [&]( SI k, double s ) { ccol[ q ] = TC( k ); cval[ q ] = TV( s ); ++q; d += s; } );
    cdia[ a ] = TV( d > 0 ? d : 1.0 );
}

// ---- the solver ------------------------------------------------------------------------------------------------------

/// `x = sum_j c_j U_j`, `r = b - sum_j c_j AU_j` ( the recycled start, `k <= 4` coefficients as arguments )
__global__ void __launch_bounds__( BLOCK ) recycle_start( SI n, int k, double c0, double c1, double c2, double c3, const double *U, const double *AU,
                                                         const double *b, double *x, double *r ) {
    const SI i = SI( blockIdx.x ) * BLOCK + threadIdx.x;
    if ( i >= n )
        return;
    const double c[ 4 ] = { c0, c1, c2, c3 };
    double sx = 0, sr = 0;
    for ( int j = 0; j < k; ++j ) { sx += c[ j ] * U[ SI( j ) * n + i ]; sr += c[ j ] * AU[ SI( j ) * n + i ]; }
    x[ i ] = sx;
    r[ i ] = b[ i ] - sr;
}

/// `x += alpha p`, `r -= alpha q`; `r . r`
struct FcgUpdate {
    const double    *p, *q;
    double          *x, *r;
    const CgScalars *sc;
    __device__ void operator()( SI i, Sum1 &acc ) const {
        const double a = sc->alpha;
        x[ i ] += a * p[ i ];
        const double ri = r[ i ] - a * q[ i ];
        r[ i ] = ri;
        acc.s += ri * ri;
    }
};

/// `( z . q, r . z )` -- the flexible `beta = - alpha z.q / rz`
struct FcgDots {
    const double *z, *q, *r;
    __device__ void operator()( SI i, Sum2 &acc ) const { acc.s0 += z[ i ] * q[ i ]; acc.s1 += r[ i ] * z[ i ]; }
};

__global__ void fcg_rr( CgScalars *sc, const Sum1 *s ) { sc->rr = s->s; }
__global__ void fcg_rz( CgScalars *sc, const Sum1 *s ) { sc->rz = s->s; sc->stop = 0; }
__global__ void fcg_beta( CgScalars *sc, const Sum2 *s ) {
    sc->beta = sc->rz != 0 ? - sc->alpha * s->s0 / sc->rz : 0.0;
    sc->rz = s->s1;
}
__global__ void __launch_bounds__( BLOCK ) fcg_direction( SI n, const double *z, double *p, const CgScalars *sc ) {
    const SI i = SI( blockIdx.x ) * BLOCK + threadIdx.x;
    if ( i < n ) p[ i ] = z[ i ] + sc->beta * p[ i ];
}

template<class TA,class TB>
__global__ void __launch_bounds__( BLOCK ) convert_values( SI n, const TA *src, TB *dst ) {
    const SI i = SI( blockIdx.x ) * BLOCK + threadIdx.x;
    if ( i < n ) dst[ i ] = TB( src[ i ] );
}

template<class TV>
__global__ void __launch_bounds__( BLOCK ) fill_values( TV *x, TV v, SI n ) {
    const SI i = SI( blockIdx.x ) * BLOCK + threadIdx.x;
    if ( i < n ) x[ i ] = v;
}

/// A FIXED SEQUENCE OF LAUNCHES, captured once and replayed whole ( a CUDA graph ): what the multigrid's iteration is --
/// some 150 launches per iteration ( a K-cycle visits its coarse levels several times, and every visit is a handful of
/// small kernels ), each costing more to launch than to run below the finest levels. Captured again only when what the
/// launches point to changes ( a coarse level reallocated, another output vector ). `SDOT_CARD_GRAPHS=0`: plain launches.
struct Graph {
    cudaGraphExec_t exec = nullptr;
    const void     *key = nullptr;
    int             generation = -1;

    static bool enabled() {
        static const bool on = ! ( std::getenv( "SDOT_CARD_GRAPHS" ) && std::string( std::getenv( "SDOT_CARD_GRAPHS" ) ) == "0" );
        return on;
    }
    bool valid_for( const void *k, int g ) const { return exec && key == k && generation == g; }
    template<class F>
    void capture( const CudaQueue &q, const void *k, int g, F &&body ) {
        release();
        cudaGraph_t graph;
        cuda_check( cudaStreamBeginCapture( q.stream, cudaStreamCaptureModeThreadLocal ), "begin capture" );
        body();
        cuda_check( cudaStreamEndCapture( q.stream, &graph ), "end capture" );
        cuda_check( cudaGraphInstantiate( &exec, graph, 0 ), "graph instantiate" );
        cudaGraphDestroy( graph );
        key = k;
        generation = g;
    }
    /// `body` replayed from the graph ( captured first if needed ), or launched as is
    template<class F>
    void run( const CudaQueue &q, const void *k, int g, F &&body ) {
        if ( ! enabled() ) { body(); return; }
        if ( ! valid_for( k, g ) )
            capture( q, k, g, body );
        launch_graph( q, exec );
    }
    void release() { if ( exec ) cudaGraphExecDestroy( exec ); exec = nullptr; }
};

/// THE SOLVER ON THE CARD ( `CG` or `MG`, see the header ). Everything is taken from the call's pool once ( `prepare`,
/// then the coarse levels at their first build ); `solve` takes nothing.
/// `TV`: the precision of the multigrid's levels ( the preconditioner ); the outer iteration is in double whatever it is. In
/// float, the fine level is a float copy of Newton's laplacian, made at each solve.
template<class TC,class TV = double>
struct CardLinear {
    static constexpr bool SAME = std::is_same_v<TV,double>;
    struct Level {
        CsrView<TC,TV> A;
        SI         *row = nullptr;
        TC         *col = nullptr;
        TV         *val = nullptr, *dia = nullptr;
        SI          nnz_cap = 0;
        TV         *rlx = nullptr, *r = nullptr, *y = nullptr, *z = nullptr;
        TV         *bk = nullptr, *e = nullptr, *v1 = nullptr, *v2 = nullptr, *t = nullptr, *rc = nullptr;
        RedSlot<Max1> lmax;
        RedSlot<Sum2> s2;
        RedSlot<Sum3> s3;
    };

    LinOptions         o;
    LinStats           st;
    SI                 n = 0, nnz_cap = 0;
    double            *x0 = nullptr, *r = nullptr, *z = nullptr, *p = nullptr, *q = nullptr;
    CgScalars         *sc = nullptr;
    RedSlot<Sum1>      s1;
    RedSlot<Sum2>      s2;
    std::vector<Level> lev;
    LapWork<SI>        scan;
    double            *U = nullptr, *AU = nullptr;       ///< the recycled solutions ( `recycle x n` ), and `A U`
    double            *bottom = nullptr;                 ///< the exact inverse of the coarsest level ( `BOTTOM_DENSE^2` )
    TV                *val0 = nullptr, *dia0 = nullptr;  ///< the fine level in `TV` ( when it is not double )
    TV                *rf = nullptr, *zf = nullptr;      ///< the cycle's input and output on the fine level ( idem )
    int                nb_u = 0, since = 0;
    bool               built = false;
    int                generation = 0;                   ///< bumped when a level's storage moves: the graphs are captured again
    Graph              g_start, g_step, g_precond;

    ~CardLinear() { g_start.release(); g_step.release(); g_precond.release(); }

    /// `n` unknowns, `nnz_cap` the capacity of the fine CSR
    bool prepare( auto &allocator, SI n_, SI nnz_cap_, const LinOptions &o_ ) {
        o = o_;
        o.stop = std::min( std::max( o.stop, 16 ), 2048 );
        o.recycle = std::min( std::max( o.recycle, 0 ), 4 );
        o.shift = std::min( std::max( o.shift, 1 ), LANES == 1 ? 5 : 3 );   // a packet's lanes are at most a warp ( `LANES << shift`: the restriction's shuffles )
        n = n_;
        nnz_cap = nnz_cap_;
        auto vec = [&]( SI m ) { return static_cast<double *>( take( allocator, SI( sizeof( double ) ) * std::max<SI>( m, 1 ) ) ); };
        auto tvec = [&]( SI m ) { return static_cast<TV *>( take( allocator, SI( sizeof( TV ) ) * std::max<SI>( m, 1 ) ) ); };
        r = vec( n ); z = vec( n ); p = vec( n ); q = vec( n );
        sc = static_cast<CgScalars *>( take( allocator, SI( sizeof( CgScalars ) ) ) );
        bool ok = r && z && p && q && sc && s1.take_from( allocator ) && s2.take_from( allocator );
        if ( o.method == 1 ) {
            bottom = vec( SI( BOTTOM_DENSE ) * BOTTOM_DENSE );
            ok = ok && bottom && scan.take_from( allocator, n );
            if constexpr ( ! SAME ) {
                val0 = tvec( nnz_cap ); dia0 = tvec( n ); rf = tvec( n ); zf = tvec( n );
                ok = ok && val0 && dia0 && rf && zf;
            }
            if ( o.recycle > 0 ) {
                U = vec( SI( o.recycle ) * n );
                AU = vec( SI( o.recycle ) * n );
                ok = ok && U && AU;
            }
            // the levels: their sizes depend on `n` alone
            SI m = n;
            const int S = 1 << o.shift;
            lev.clear();
            while ( true ) {
                Level L;
                L.A.n = m;
                L.rlx = tvec( m ); L.r = tvec( m ); L.y = tvec( m ); L.z = tvec( m );
                ok = ok && L.rlx && L.r && L.y && L.z && L.lmax.take_from( allocator ) && L.s2.take_from( allocator ) && L.s3.take_from( allocator );
                if ( ! lev.empty() ) {                   // a coarse level: its own CSR rows and the K-cycle's vectors
                    L.row = static_cast<SI *>( take( allocator, SI( sizeof( SI ) ) * ( m + 1 ) ) );
                    L.dia = tvec( m );
                    L.bk = tvec( m ); L.e = tvec( m ); L.v1 = tvec( m ); L.v2 = tvec( m ); L.t = tvec( m ); L.rc = tvec( m );
                    ok = ok && L.row && L.dia && L.bk && L.e && L.v1 && L.v2 && L.t && L.rc;
                }
                lev.push_back( L );
                if ( m <= o.stop || lev.size() >= 24 )
                    break;
                m = ( m + S - 1 ) / S;
            }
        }
        return ok;
    }

    void relax( const CudaQueue &queue, Level &L ) {
        reduce( queue, L.A.n, Relax<TC,TV>{ L.A, L.rlx }, L.lmax.partials, L.lmax.out );
    }

    /// the coarse levels from the fine one ( one count read back per level ); `false` if the pool said no
    bool build( const CudaQueue &queue, auto &allocator ) {
        const int sh = o.shift;
        relax( queue, lev[ 0 ] );
        for ( size_t l = 0; l + 1 < lev.size(); ++l ) {
            const Level &F = lev[ l ];
            Level &C = lev[ l + 1 ];
            const SI nc = C.A.n;
            launch_kernel( queue, &coarse_count<TC,TV>, blocks_for( nc ), BLOCK, 0, F.A, sh, nc, scan.cnt );
            exclusive_scan( queue, scan.sums, static_cast<const SI *>( scan.cnt ), C.row, nc + 1 );
            SI nnz = 0;
            read_back( queue, &nnz, static_cast<const SI *>( C.row + nc ), 1 );
            if ( nnz > C.nnz_cap ) {                     // the first build, or a coarse level that grew: room from the pool
                C.nnz_cap = nnz + nnz / 4 + 64;
                ++generation;
                C.col = static_cast<TC *>( take( allocator, SI( sizeof( TC ) ) * C.nnz_cap ) );
                C.val = static_cast<TV *>( take( allocator, SI( sizeof( TV ) ) * C.nnz_cap ) );
                if ( ! C.col || ! C.val )
                    return false;
            }
            launch_kernel( queue, &coarse_fill<TC,TV>, blocks_for( nc ), BLOCK, 0, F.A, sh, nc, ( const SI * ) C.row, C.col, C.val, C.dia );
            C.A = CsrView<TC,TV>{ nc, C.row, C.col, C.val, C.dia };
            relax( queue, C );
        }
        if ( lev.back().A.n <= BOTTOM_DENSE )
            launch_kernel( queue, &bottom_invert<TC,TV>, 1, 256, 0, lev.back().A, bottom );
        return true;
    }

    /// `x = S^deg ... `: the Chebyshev smoother on level `L` for `rhs`
    void chebyshev( const CudaQueue &queue, Level &L, const TV *rhs, TV *x, bool fresh ) {
        const int deg = o.nu;
        const SI m = L.A.n;
        if ( deg <= 0 ) {
            if ( fresh ) launch_kernel( queue, &fill_values<TV>, blocks_for( m ), BLOCK, 0, x, TV( 0 ), m );
            return;
        }
        launch_kernel( queue, &cheb_init<TC,TV>, blocks_for( m ), BLOCK, 0, L.A, rhs, x, L.r, L.y, ( const TV * ) L.rlx,
                       ( const Max1 * ) L.lmax.out, o.cheb, int( fresh ) );
        for ( int k = 0; k < deg; ++k ) {
            if ( k + 1 == deg ) {
                launch_kernel( queue, &add_into<TV>, blocks_for( m ), BLOCK, 0, m, x, ( const TV * ) L.y );
                break;
            }
            launch_kernel( queue, &cheb_step<TC,TV>, blocks_for( m ), BLOCK, 0, L.A, x, L.r, ( const TV * ) L.y, L.z, ( const TV * ) L.rlx,
                           ( const Max1 * ) L.lmax.out, o.cheb, k );
            std::swap( L.y, L.z );
        }
    }

    /// THE CYCLE from level `l`: `x ~ A_l^-1 rhs`, from zero
    void cycle( const CudaQueue &queue, int l, const TV *rhs, TV *x ) {
        Level &L = lev[ l ];
        if ( l + 1 == int( lev.size() ) ) {
            if ( L.A.n <= BOTTOM_DENSE ) {
                launch_kernel( queue, &bottom_apply<TV>, 1, 64, 0, int( L.A.n ), ( const double * ) bottom, rhs, x );
                return;
            }
            const int shm = int( 2 * L.A.n * sizeof( double ) );
            launch_kernel( queue, &bottom_jacobi<TC,TV>, 1, 256, shm, L.A, rhs, x, o.bottom_omega, o.bottom_sweeps );
            return;
        }
        Level &C = lev[ l + 1 ];
        const SI nc = C.A.n;
        const bool fused = o.nu == 1;
        TV *xs = fused ? L.y : x;                        // the pre-smoothed iterate
        if ( fused )
            launch_kernel( queue, &smooth_fresh<TV>, blocks_for( L.A.n ), BLOCK, 0, L.A.n, rhs, ( const TV * ) L.rlx, ( const Max1 * ) L.lmax.out, o.cheb, xs );
        else
            chebyshev( queue, L, rhs, x, true );
        launch_kernel( queue, &restrict_residual<TC,TV>, blocks_for( lane_items( L.A.n ) ), BLOCK, 0, L.A, rhs, ( const TV * ) xs, C.bk, o.shift );
        if ( l + 1 <= o.kcycle && l + 2 < int( lev.size() ) ) {
            // two flexible-CG steps on the coarse system, preconditioned by the cycle below
            cycle( queue, l + 1, C.bk, C.v1 );
            reduce( queue, lane_items( nc ), KStep1<TC,TV>{ C.A, C.v1, C.bk, C.t }, C.s2.partials, C.s2.out );
            launch_kernel( queue, &kcycle_rc<TV>, blocks_for( nc ), BLOCK, 0, nc, ( const TV * ) C.bk, ( const TV * ) C.t, C.rc, ( const Sum2 * ) C.s2.out );
            cycle( queue, l + 1, C.rc, C.v2 );
            reduce( queue, lane_items( nc ), KStep2<TC,TV>{ C.A, C.v2, C.t, C.rc }, C.s3.partials, C.s3.out );
            launch_kernel( queue, &kcycle_combine<TV>, blocks_for( nc ), BLOCK, 0, nc, ( const TV * ) C.v1, ( const TV * ) C.v2, C.e,
                           ( const Sum2 * ) C.s2.out, ( const Sum3 * ) C.s3.out );
        } else
            cycle( queue, l + 1, C.bk, C.e );
        if ( fused ) {
            launch_kernel( queue, &prolong_into<TV>, blocks_for( L.A.n ), BLOCK, 0, L.A.n, ( const TV * ) L.y, ( const TV * ) C.e, o.shift, L.z );
            launch_kernel( queue, &smooth_post<TC,TV>, blocks_for( lane_items( L.A.n ) ), BLOCK, 0, L.A, rhs, ( const TV * ) L.z, ( const TV * ) L.rlx,
                           ( const Max1 * ) L.lmax.out, o.cheb, x );
            return;
        }
        launch_kernel( queue, &prolong_add<TV>, blocks_for( L.A.n ), BLOCK, 0, L.A.n, x, ( const TV * ) C.e, o.shift );
        chebyshev( queue, L, rhs, x, false );
    }

    /// `z = M r` ( double in, double out ): the cycle from the fine level, through `TV` copies when it is not double
    void precondition( const CudaQueue &queue ) {
        if constexpr ( SAME )
            cycle( queue, 0, r, z );
        else {
            launch_kernel( queue, &convert_values<double,TV>, blocks_for( n ), BLOCK, 0, n, ( const double * ) r, rf );
            cycle( queue, 0, rf, zf );
            launch_kernel( queue, &convert_values<TV,double>, blocks_for( n ), BLOCK, 0, n, ( const TV * ) zf, z );
        }
    }

    /// the mean out of `v` ( the zero-mean gauge )
    void center( const CudaQueue &queue, double *v ) {
        reduce( queue, n, SumOf{ v }, s1.partials, s1.out );
        launch_kernel( queue, &subtract_scalar, blocks_for( n ), BLOCK, 0, v, reinterpret_cast<const double *>( s1.out ), 1.0 / double( n ), n );
    }

    /// `L x = b` ( `b` of zero sum; `x` comes out centred ). `false`: the solver could not ( the relative residual stayed
    /// above 1 ), or the pool said no.
    bool solve( const CudaQueue &queue, auto &allocator, const CsrView<TC> &A, const double *b, double *x ) {
        if ( o.method == 0 )
            return solve_cg( queue, A, b, x );
        return solve_mg( queue, allocator, A, b, x );
    }

    bool solve_cg( const CudaQueue &queue, const CsrView<TC> &A, const double *b, double *x ) {
        const double t0 = wall_now();
        reduce( queue, n, CgInit{ b, A.dia, r, p, x }, s2.partials, s2.out );
        launch_kernel( queue, &cg_start, 1, 1, 0, sc, ( const Sum2 * ) s2.out );
        CgScalars h;
        read_back( queue, &h, sc, 1 );
        const double bb = h.rr;
        if ( ! ( bb > 0 ) ) { st.t_res += wall_now() - t0; return true; }
        const double target = o.tol * o.tol * bb;
        int it = 0;
        while ( it < o.maxit ) {
            reduce( queue, lane_items( n ), SpmvDot<TC>{ A, p, q }, s1.partials, s1.out );
            launch_kernel( queue, &cg_alpha, 1, 1, 0, sc, ( const Sum1 * ) s1.out );
            reduce( queue, n, CgUpdate{ p, q, A.dia, x, r, sc }, s2.partials, s2.out );
            launch_kernel( queue, &cg_beta, 1, 1, 0, sc, ( const Sum2 * ) s2.out );
            launch_kernel( queue, &cg_direction, blocks_for( n ), BLOCK, 0, n, ( const double * ) r, A.dia, p, ( const CgScalars * ) sc );
            ++it;
            if ( it % std::max( o.check, 1 ) == 0 || it == o.maxit ) {
                read_back( queue, &h, sc, 1 );
                if ( h.rr <= target || h.stop )
                    break;
            }
        }
        read_back( queue, &h, sc, 1 );
        center( queue, x );
        const double err = std::sqrt( std::max( h.rr, 0.0 ) / bb );
        st.nb_iter += it;
        st.worst = std::max( st.worst, err );
        st.t_res += wall_now() - t0;
        if ( o.trace ) std::printf( "      lin cg: %d it, rel. residual %.2e\n", it, err );
        return err < 1;
    }

    bool solve_mg( const CudaQueue &queue, auto &allocator, const CsrView<TC> &A, const double *b, double *x ) {
        double t0 = wall_now();
        if constexpr ( SAME )
            lev[ 0 ].A = A;
        else {                                           // the fine level in `TV`: a copy of the values, at each solve
            // ( the count of entries is on the card: the copy covers the capacity )
            launch_kernel( queue, &convert_values<double,TV>, blocks_for( nnz_cap ), BLOCK, 0, nnz_cap, A.val, val0 );
            launch_kernel( queue, &convert_values<double,TV>, blocks_for( n ), BLOCK, 0, n, A.dia, dia0 );
            lev[ 0 ].A = CsrView<TC,TV>{ n, A.row, A.col, val0, dia0 };
        }
        if ( ! built || since >= std::max( o.rebuild, 1 ) || lev.size() == 1 ) {   // ( one level: the bottom IS the matrix )
            if ( ! build( queue, allocator ) )
                return false;
            built = true;
            since = 0;
            ++st.nb_hierarchies;
        } else
            relax( queue, lev[ 0 ] );                    // the fine level only: repointed, its coefficients redone
        ++since;
        const double t1 = wall_now();
        st.t_hierarchy += t1 - t0;

        // `bb`, and the recycled start ( `x = U ( U^t A U )^-1 U^t b`, `r = b - A x` )
        reduce( queue, n, DotOf{ b, b }, s1.partials, s1.out );
        Sum1 hb;
        read_back( queue, &hb, s1.out, 1 );
        const double bb = hb.s;
        if ( ! ( bb > 0 ) ) {
            launch_kernel( queue, &fill_value, blocks_for( n ), BLOCK, 0, x, 0.0, n );
            st.t_res += wall_now() - t1;
            return true;
        }
        bool started = false;
        if ( nb_u > 0 ) {
            const int k = nb_u;
            for ( int j = 0; j < k; ++j )
                reduce( queue, lane_items( n ), SpmvDot<TC>{ A, U + SI( j ) * n, AU + SI( j ) * n }, s1.partials, s1.out );
            std::vector<double> G( k * k ), f( k );
            for ( int j = 0; j < k; ++j ) {
                Sum1 v;
                reduce( queue, n, DotOf{ U + SI( j ) * n, b }, s1.partials, s1.out );
                read_back( queue, &v, s1.out, 1 );
                f[ j ] = v.s;
                for ( int l = 0; l <= j; ++l ) {
                    reduce( queue, n, DotOf{ U + SI( j ) * n, AU + SI( l ) * n }, s1.partials, s1.out );
                    read_back( queue, &v, s1.out, 1 );
                    G[ j * k + l ] = G[ l * k + j ] = v.s;
                }
            }
            if ( sdotplan_solve_dense( G, f, k ) ) {
                launch_kernel( queue, &recycle_start, blocks_for( n ), BLOCK, 0, n, k, f[ 0 ], k > 1 ? f[ 1 ] : 0.0, k > 2 ? f[ 2 ] : 0.0,
                               k > 3 ? f[ 3 ] : 0.0, ( const double * ) U, ( const double * ) AU, b, x, r );
                center( queue, x );
                center( queue, r );
                started = true;
            }
        }
        if ( ! started ) {
            launch_kernel( queue, &copy_values, blocks_for( n ), BLOCK, 0, r, b, n );
            launch_kernel( queue, &fill_value, blocks_for( n ), BLOCK, 0, x, 0.0, n );
        }

        // the flexible CG ( its fixed sequences of launches replayed from graphs: `Graph` )
        const double target = o.tol * o.tol * bb;
        g_start.run( queue, x, generation, [&] {
            precondition( queue );
            reduce( queue, n, DotOf{ r, z }, s1.partials, s1.out );
            launch_kernel( queue, &fcg_rz, 1, 1, 0, sc, ( const Sum1 * ) s1.out );
            launch_kernel( queue, &copy_values, blocks_for( n ), BLOCK, 0, p, ( const double * ) z, n );
        } );
        CgScalars h;
        h.rr = bb;
        int it = 0;
        while ( it < o.maxit ) {
            g_step.run( queue, x, generation, [&] {
                reduce( queue, lane_items( n ), SpmvDot<TC>{ A, p, q }, s1.partials, s1.out );
                launch_kernel( queue, &cg_alpha, 1, 1, 0, sc, ( const Sum1 * ) s1.out );
                reduce( queue, n, FcgUpdate{ p, q, x, r, sc }, s1.partials, s1.out );
                launch_kernel( queue, &fcg_rr, 1, 1, 0, sc, ( const Sum1 * ) s1.out );
            } );
            ++it;
            read_back( queue, &h, sc, 1 );
            if ( h.rr <= target || h.stop )
                break;
            g_precond.run( queue, x, generation, [&] {
                precondition( queue );
                reduce( queue, n, FcgDots{ z, q, r }, s2.partials, s2.out );
                launch_kernel( queue, &fcg_beta, 1, 1, 0, sc, ( const Sum2 * ) s2.out );
                launch_kernel( queue, &fcg_direction, blocks_for( n ), BLOCK, 0, n, ( const double * ) z, p, ( const CgScalars * ) sc );
            } );
        }
        center( queue, x );
        if ( o.recycle > 0 ) {                           // the solution joins the subspace ( a ring of `recycle` slots )
            const int slot = nb_u < o.recycle ? nb_u : ( since_slot++ % o.recycle );
            launch_kernel( queue, &copy_values, blocks_for( n ), BLOCK, 0, U + SI( slot ) * n, ( const double * ) x, n );
            nb_u = std::min( nb_u + 1, o.recycle );
        }
        const double err = std::sqrt( std::max( h.rr, 0.0 ) / bb );
        st.nb_iter += it;
        st.worst = std::max( st.worst, err );
        st.t_res += wall_now() - t1;
        if ( o.trace ) std::printf( "      lin mg: %d levels, %d it, rel. residual %.2e%s\n", int( lev.size() ), it, err, started ? " ( recycled start )" : "" );
        return err < 1;
    }

    int since_slot = 0;

    /// the small dense system `G y = f` ( Cholesky with a ridge, as `Multigrid.h::solve_dense` ); `false`: declined
    static bool sdotplan_solve_dense( std::vector<double> &G, std::vector<double> &f, int k ) {
        double tr = 0;
        for ( int j = 0; j < k; ++j ) tr += G[ j * k + j ];
        if ( ! ( tr > 0 ) ) return false;
        const double eps = tr / double( k ) * 1e-12;
        for ( int j = 0; j < k; ++j ) G[ j * k + j ] += eps;
        for ( int j = 0; j < k; ++j ) {
            double s = G[ j * k + j ];
            for ( int q = 0; q < j; ++q ) s -= G[ j * k + q ] * G[ j * k + q ];
            if ( ! ( s > 0 ) ) return false;
            const double dj = std::sqrt( s );
            G[ j * k + j ] = dj;
            for ( int i = j + 1; i < k; ++i ) {
                double t = G[ i * k + j ];
                for ( int q = 0; q < j; ++q ) t -= G[ i * k + q ] * G[ j * k + q ];
                G[ i * k + j ] = t / dj;
            }
        }
        for ( int i = 0; i < k; ++i ) {
            double t = f[ i ];
            for ( int q = 0; q < i; ++q ) t -= G[ i * k + q ] * f[ q ];
            f[ i ] = t / G[ i * k + i ];
        }
        for ( int i = k - 1; i >= 0; --i ) {
            double t = f[ i ];
            for ( int q = i + 1; q < k; ++q ) t -= G[ q * k + i ] * f[ q ];
            f[ i ] = t / G[ i * k + i ];
        }
        return true;
    }
};

// ---- the host route ----------------------------------------------------------------------------------------------------

/// THE CPU SOLVERS OF `sdotplan/Linear.cpp` on the card's CSR: copied to the host, solved, `d` copied back ( gauge: zero
/// mean, like the card's solvers ). The times include the copies.
template<class TC>
struct HostLinear {
    std::unique_ptr<sdot::sdotplan::LinearSolver> lin;
    sdot::sdotplan::Laplacian L;
    std::vector<TC>     col;
    std::vector<double> b, d;

    void prepare( int method, SI n, const sdot::sdotplan::LinearOptions &lo ) {
        lin = sdot::sdotplan::linear_solver( sdot::sdotplan::Lin( method ), n, 2, lo );
        std::vector<SI> identity( n );
        for ( SI i = 0; i < n; ++i ) identity[ i ] = i;  // the CSR is in tree ranks already: the multigrid aggregates `i >> k`
        lin->order( identity );
    }

    bool solve( const CudaQueue &queue, const CsrView<TC> &A, const double *b_dev, double *x_dev ) {
        const SI n = A.n;
        L.n = n;
        L.row.resize( n + 1 );
        read_back( queue, L.row.data(), A.row, n + 1 );
        const SI nnz = L.row[ n ];
        col.resize( nnz );
        L.c.resize( nnz );
        L.dia.resize( n );
        b.resize( n );
        read_back( queue, col.data(), A.col, nnz );
        read_back( queue, L.c.data(), A.val, nnz );
        read_back( queue, L.dia.data(), A.dia, n );
        read_back( queue, b.data(), b_dev, n );
        L.col.resize( nnz );
        for ( SI e = 0; e < nnz; ++e ) L.col[ e ] = SI( col[ e ] );
        const bool ok = lin->solves( L, b, d );
        if ( ! ok )
            return false;
        double mean = 0;
        for ( SI i = 0; i < n; ++i ) mean += d[ i ];
        mean /= double( n );
        for ( SI i = 0; i < n; ++i ) d[ i ] -= mean;
        cuda_check( cudaMemcpyAsync( x_dev, d.data(), sizeof( double ) * n, cudaMemcpyHostToDevice, queue.stream ), "copy of the direction" );
        cuda_check( cudaStreamSynchronize( queue.stream ), "sync ( copy of the direction )" );
        return true;
    }
};

} // namespace sdot::gpu2d
