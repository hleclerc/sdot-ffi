#pragma once

// =====================================================================================
// NEWTON'S LINEAR SYSTEM ON THE CARD: `L d = b`, `L` the laplacian of the Laguerre graph in CSR and IN RANKS
// ( `Laplacian2D.cuh`: `y_i = dia_i x_i - sum_e val_e x_( col_e )`, `L = L^T`, `L 1 = 0` to the bit ), `b` of zero sum.
// The gauge is the ZERO MEAN ( the kernel of `L` is the constants ): what comes out is centred, and the caller
// moves it to the gauge it wants.
//
// Three ways, chosen per call ( `SdotPlanNd`'s `linear_solver`, `Tuning.linear_host` ):
//
//   CG     the conjugate gradient preconditioned by Jacobi, all on the card. `sqrt( n )` iterations: the baseline.
//   MG     a multigrid preconditioning a FLEXIBLE CG ( Polak-Ribiere `beta`: the K-cycle is not a fixed operator ), the
//          outer iteration in double, the levels in `TV` ( float by default: Turing's double is 1/32 of its float ):
//
//          * AGGREGATION BY TREE RANK ( `rank >> shift`, packets of 4: consecutive ranks are neighbours in space, an aligned
//            window is a subtree, so the hierarchy is a shift );
//          * the first `smoothed` levels ( 1 by default ) pass to the next by the SMOOTHED AGGREGATION of the CPU's
//            `Multigrid.h`: `P = ( I - w D^-1 A ) P0`, each row truncated at `truncate` of its largest entry and renormalized
//            ( the constants stay in the range of `P` ), `P^t` by a transposition, `A_c = P^t A P` by one WARP per coarse
//            row and a hash table in shared memory summing in FIXED POINT ( integers add in any order to the same sum: the
//            row is the same at every run, whatever the order of the atomics ). The others by the PLAIN aggregation
//            ( `P = P0`: the coarse row is a merge of the sorted rows of its packet, or the same warp product past
//            `plain_warp` entries per fine row );
//          * a CHEBYSHEV smoother of degree `nu` in `M^-1 A` ( `M` the spai0 diagonal, `lmax` bounded by Gershgorin ), its
//            coefficients computed ONCE per level by the thread that finishes the bound's reduction ( no division in the
//            elementwise kernels ), the degree-one smoothing fused with the residual, the restriction and the plain
//            prolongation ( two matrix passes per visit of a level ), `lanes` threads per row in the matrix passes;
//          * the K-CYCLE on `kcycle` levels from the first plain one ( two flexible-CG steps preconditioned by the level
//            below, their scalars on the card ), a V-cycle elsewhere;
//          * the coarsest level ( <= 64 unknowns ) SOLVED EXACTLY ( a dense inverse built in one block );
//          * RECYCLING ( the last solutions as a Galerkin start ), a hierarchy per solve ( `rebuild`: reuse measured worse
//            on the lines ), and every iteration REPLAYED FROM ONE CUDA GRAPH, the host reading back `r.r` only.
//
//          Measured on the dumped systems of the bench's Newton solves ( `calibration_lmo_today.md`, GPU step 5 ): 2x faster
//          than the multigrid of step 4 ( plain aggregation everywhere, double levels, one thread per row ).
//   HOST   the CSR copied to the host, one of the CPU solvers of `sdotplan/Linear.cpp` ( Cholesky, AMGCL, the CPU
//          multigrid, CG ), `d` copied back: what the card does not have, and the reference.
//
// Every scalar the iterations need stays on the card; the host reads back the residual of the outer iteration ( every
// `check` iterations for CG ) and a few counts per level when a hierarchy is built. Two solves of the same systems are
// identical to the bit ( fixed reduction trees, fixed-point sums, rows sorted after any atomic placement ).
// `SDOT_CARD_LIN_DUMP=prefix` writes every system to a file ( `dump_system` ), to replay them outside the Newton.
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

/// `( A x )_i`, accumulated in `TV` ( double for Newton's system; float on float levels: Turing's double is 1/32 of its float )
template<class TC,class TV,class TX>
__device__ __forceinline__ TV row_product( const CsrView<TC,TV> &A, const TX *x, SI i ) {
    TV s = A.dia[ i ] * TV( x[ i ] );
    for ( SI e = A.row[ i ]; e < A.row[ i + 1 ]; ++e )
        s -= A.val[ e ] * TV( x[ A.col[ e ] ] );
    return s;
}

struct LinOptions {
    int    method        = 1;      ///< 0 CG ( Jacobi ), 1 MG
    double tol           = 1e-6;   ///< relative residual
    int    maxit         = 20000;
    int    shift         = 2;      ///< MG: packets of `2^shift` ranks
    int    smoothed      = 1;      ///< MG: the first `smoothed` levels pass to the next by the smoothed aggregation, the others by the plain one
    double omega         = 0.7;    ///< MG smoothed: `P = ( I - omega D^-1 A ) P0`, truncated ...
    double truncate      = 0.3;    ///< MG smoothed: ... of the entries under this fraction of their row's largest, renormalized
    int    nu            = 1;      ///< MG: Chebyshev degree, before and after
    int    nu0           = -1;     ///< MG: the same on the fine level ( -1: `nu` )
    double cheb          = 20;     ///< MG: `lmin = lmax / cheb`
    int    kcycle        = 2;      ///< MG: that many levels are K-cycled ...
    int    kfrom         = -1;     ///< MG: ... from this one ( -1: the first plain one, `max( 1, smoothed + 1 )` )
    int    recycle       = 2;      ///< MG: solutions kept for the Galerkin start
    int    rebuild       = 1;      ///< MG: solves per hierarchy
    int    stop          = 64;     ///< MG: coarsening stops under this size ( `<= BOTTOM_DENSE`: an exact bottom; else smoothed, at most 2048 )
    int    lanes         = 4;      ///< MG: lanes per row of the matrix-vector kernels of the cycle ( 1, 2, 4 )
    int    plain_warp    = 12;     ///< MG: the plain coarse product by warps ( `sa_galerkin` ) above that many entries per fine row
    int    bottom_sweeps = 60;     ///< MG: damped Jacobi sweeps at the bottom ( when it is not dense )
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

// ---- reductions with a finalizer ( the scalars derived from a reduction computed by the thread that writes it: no
//      extra launch, no division in the elementwise kernels ) and a wider first pass ( the matrix-vector products ) ------

struct NoFin {
    template<class A> __device__ void operator()( const A & ) const {}
};

template<class Acc,class Fin>
__global__ void __launch_bounds__( BLOCK ) reduce_last_fin( int nb, const Acc *partials, Acc *out, Fin fin ) {
    Acc acc = Acc::identity();
    for ( int i = threadIdx.x; i < nb; i += BLOCK )
        acc.combine( partials[ i ] );
    acc = block_combine( acc );
    if ( threadIdx.x == 0 ) {
        *out = acc;
        fin( acc );
    }
}

/// blocks of the first pass of the solver's reductions ( several of them are fused matrix-vector products ): measured,
/// 136 / 240 / 340 / 512 / 1024 / 2048 blocks of 128 threads gave 0.400 / 0.395 / 0.394 / 0.412 / 0.413 / 0.414 s of
/// linear solves at 1e6 -- more resident threads gathering at once is NOT faster here. Fixed, so the tree of the sums is
/// the same at every run.
constexpr int WIDE_GRID = 272;

template<class Acc>
struct WideSlot {
    Acc *partials = nullptr, *out = nullptr;
    bool take_from( auto &allocator ) {
        partials = static_cast<Acc *>( take( allocator, SI( sizeof( Acc ) ) * WIDE_GRID ) );
        out      = static_cast<Acc *>( take( allocator, SI( sizeof( Acc ) ) ) );
        return partials && out;
    }
};

template<class Acc,class F,class Fin = NoFin>
void reduce_fin( const CudaQueue &queue, SI n, const F &f, const WideSlot<Acc> &slot, Fin fin = {} ) {
    const int grid = int( std::min<SI>( WIDE_GRID, std::max<SI>( 1, ( n + BLOCK - 1 ) / BLOCK ) ) );
    launch_kernel( queue, &reduce_pass<Acc,F>, grid, BLOCK, 0, n, f, slot.partials );
    launch_kernel( queue, &reduce_last_fin<Acc,Fin>, 1, BLOCK, 0, grid, ( const Acc * ) slot.partials, slot.out, fin );
}

// ---- the conjugate gradient ( Jacobi ) ------------------------------------------------------------------------------

struct CgScalars {
    double rz, alpha, beta, rr;
    int    stop;
    int    it;                                           ///< the iterations of the flexible CG
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

/// `y = A x`; `x . y`
template<class TC>
struct SpmvDot {
    CsrView<TC>   A;
    const double *x;
    double       *y;
    __device__ void operator()( SI i, Sum1 &acc ) const {
        const double v = row_product( A, x, i );
        y[ i ] = v;
        acc.s += x[ i ] * v;
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

/// the finalizers of the ( flexible ) CG: the scalars live on the card
struct FinStart { CgScalars *sc; __device__ void operator()( const Sum2 &s ) const { sc->rr = s.s0; sc->rz = s.s1; sc->alpha = sc->beta = 0; sc->stop = 0; } };
struct FinAlpha {
    CgScalars *sc;
    __device__ void operator()( const Sum1 &pq ) const {
        if ( pq.s > 0 ) sc->alpha = sc->rz / pq.s;
        else { sc->alpha = 0; sc->stop = 1; }           // the direction is in the kernel: done
    }
};
struct FinBetaCg {
    CgScalars *sc;
    __device__ void operator()( const Sum2 &s ) const { sc->beta = sc->rz != 0 ? s.s1 / sc->rz : 0.0; sc->rz = s.s1; sc->rr = s.s0; }
};
/// the end of an iteration of the flexible CG: `r.r`, and the count of the iterations
struct FinCheck {
    CgScalars *sc;
    __device__ void operator()( const Sum1 &s ) const { sc->rr = s.s; ++sc->it; }
};

/// the scalars of a flexible CG from `r.r` and the targets ( `rz = 0`: the first `beta` is zero )
__global__ void fcg_init( CgScalars *sc, double rr ) {
    sc->rz = 0; sc->alpha = 0; sc->beta = 0; sc->rr = rr; sc->stop = 0; sc->it = 0;
}

/// flexible `beta = - alpha z.q / rz` ( Polak-Ribiere: the preconditioner need not be a fixed operator )
struct FinBetaFcg {
    CgScalars *sc;
    __device__ void operator()( const Sum2 &s ) const { sc->beta = sc->rz != 0 ? - sc->alpha * s.s0 / sc->rz : 0.0; sc->rz = s.s1; }
};

__global__ void __launch_bounds__( BLOCK ) cg_direction( SI n, const double *r, const double *dia, double *p, const CgScalars *sc ) {
    const SI i = SI( blockIdx.x ) * BLOCK + threadIdx.x;
    if ( i < n ) p[ i ] = r[ i ] / dia[ i ] + sc->beta * p[ i ];
}

// ---- the multigrid: smoother -----------------------------------------------------------------------------------------

constexpr int MAX_DEG = 8;

/// the coefficients of the Chebyshev smoother of a level, computed ONCE from its Gershgorin bound ( by the thread that
/// finishes the bound's reduction ): `1 / theta` and the three-term recurrence's `c1`, `c2` per step
struct ChebCoef {
    double inv_th;
    double c1[ MAX_DEG ], c2[ MAX_DEG ];
};

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

struct FinCheb {
    double    cheb;
    ChebCoef *out;
    __device__ void operator()( const Max1 &m ) const {
        const double hi = m.m > 0 ? m.m : 2.0, lo = hi / fmax( cheb, 1.01 );
        const double th = ( hi + lo ) / 2, de = ( hi - lo ) / 2, si = th / de;
        out->inv_th = 1 / th;
        double rh = 1 / si;
        for ( int k = 0; k < MAX_DEG; ++k ) {
            const double r2 = 1 / ( 2 * si - rh );
            out->c1[ k ] = r2 * rh;
            out->c2[ k ] = 2 * r2 / de;
            rh = r2;
        }
    }
};

/// `y = M rhs / theta` ( the first Chebyshev direction from zero: `x = y` after it )
template<class TV>
__global__ void __launch_bounds__( BLOCK ) cheb_fresh( SI n, const TV *rhs, const TV *rlx, const ChebCoef *cc, TV *y ) {
    const SI i = SI( blockIdx.x ) * BLOCK + threadIdx.x;
    if ( i < n ) y[ i ] = rlx[ i ] * rhs[ i ] * TV( cc->inv_th );
}

/// the product of row `i` by `LN` lanes ( entries `sub, sub + LN, ...` ), summed by shuffles: every lane of the warp calls it.
/// `f( j )` the value of the vector at `j`.
template<int LN,class TC,class TV,class F>
__device__ __forceinline__ TV row_off_lanes( const CsrView<TC,TV> &A, SI i, int sub, bool valid, F &&f ) {
    TV s = 0;
    if ( valid )
        for ( SI q = A.row[ i ] + sub; q < A.row[ i + 1 ]; q += LN )
            s += A.val[ q ] * f( A.col[ q ] );
    for ( int o = 1; o < LN; o *= 2 )
        s += __shfl_xor_sync( 0xffffffffu, s, o );
    return s;
}

/// the threads of a kernel with `LN` lanes per row: a whole number of blocks
template<int LN>
__device__ __forceinline__ SI lane_threads( SI n ) { return ( n * LN + BLOCK - 1 ) / BLOCK * BLOCK; }

/// `r = rhs - A x0`, `y = M r / theta`, with `x0 = x + ( P e )`: the plain prolongation `e[ j >> sh ]` read at the
/// neighbours ( `sh >= 0` ), or none ( `sh < 0`, `x` already prolonged ). `deg == 1`: `out = x0 + y` directly. `LN` lanes per row.
template<int LN,class TC,class TV>
__global__ void __launch_bounds__( BLOCK ) cheb_start( CsrView<TC,TV> A, const TV *rhs, const TV *x, const TV *e, int sh, const TV *rlx,
                                                      const ChebCoef *cc, TV *r, TV *y, TV *out, int last ) {
    const SI nt = lane_threads<LN>( A.n );
    for ( SI t = SI( blockIdx.x ) * BLOCK + threadIdx.x; t < nt; t += SI( gridDim.x ) * BLOCK ) {
        const SI i = t / LN;
        const int sub = int( t % LN );
        const bool valid = i < A.n;
        auto x0 = [&]( SI j ) { return sh >= 0 ? x[ j ] + e[ j >> sh ] : x[ j ]; };
        const TV off = row_off_lanes<LN>( A, i, sub, valid, x0 );
        if ( ! valid || sub )
            continue;
        const TV xi = x0( i );
        const TV ri = rhs[ i ] - ( A.dia[ i ] * xi - off ), yi = rlx[ i ] * ri * TV( cc->inv_th );
        if ( last ) { out[ i ] = xi + yi; continue; }
        out[ i ] = xi;
        r[ i ] = ri;
        y[ i ] = yi;
    }
}

/// step `k`: `x += y`, `r -= A y`, `z = c1 y + c2 M r` ( `y` is read at the neighbours: the next direction goes to `z`,
/// the caller swaps ). `fresh`: `x` and `r` are not there yet ( `x = 0`, `r = rhs` ).
template<class TC,class TV>
__global__ void __launch_bounds__( BLOCK ) cheb_step( CsrView<TC,TV> A, const TV *rhs, TV *x, TV *r, const TV *y, TV *z, const TV *rlx,
                                                     const ChebCoef *cc, int k, int fresh ) {
    for ( SI i = SI( blockIdx.x ) * BLOCK + threadIdx.x; i < A.n; i += SI( gridDim.x ) * BLOCK ) {
        const TV yi = y[ i ];
        x[ i ] = fresh ? yi : x[ i ] + yi;
        const TV ri = ( fresh ? rhs[ i ] : r[ i ] ) - row_product( A, y, i );
        r[ i ] = ri;
        z[ i ] = TV( cc->c1[ k ] ) * yi + TV( cc->c2[ k ] ) * rlx[ i ] * ri;
    }
}

/// the last direction: `x += y`; and the residual `rho = r - A y` ( = `rhs - A x` ) for the restriction. `fresh` ( degree 1
/// from zero ): `x = y`, `r = rhs`.
template<int LN,class TC,class TV>
__global__ void __launch_bounds__( BLOCK ) cheb_last_residual( CsrView<TC,TV> A, const TV *rhs, TV *x, const TV *r, const TV *y, TV *rho, int fresh ) {
    const SI nt = lane_threads<LN>( A.n );
    for ( SI t = SI( blockIdx.x ) * BLOCK + threadIdx.x; t < nt; t += SI( gridDim.x ) * BLOCK ) {
        const SI i = t / LN;
        const int sub = int( t % LN );
        const bool valid = i < A.n;
        const TV off = row_off_lanes<LN>( A, i, sub, valid, [&]( SI j ) { return y[ j ]; } );
        if ( ! valid || sub )
            continue;
        const TV yi = y[ i ];
        x[ i ] = fresh ? yi : x[ i ] + yi;
        rho[ i ] = ( fresh ? rhs[ i ] : r[ i ] ) - ( A.dia[ i ] * yi - off );
    }
}

/// the same, the residual summed per packet ( the plain restriction: `bc[ a ] = sum_{ i in a } rho_i`, `2^sh` lanes summed by
/// shuffles; a packet is aligned on its lanes: `2^sh <= 32` divides the block ) -- or written to `rho` ( `sh < 0` )
template<int LN,class TC,class TV>
__global__ void __launch_bounds__( BLOCK ) cheb_last_restrict( CsrView<TC,TV> A, const TV *rhs, TV *x, const TV *r, const TV *y, TV *bc, int sh, int fresh ) {
    const SI nt = lane_threads<LN>( A.n );               // every lane of a warp reaches the shuffles
    for ( SI t = SI( blockIdx.x ) * BLOCK + threadIdx.x; t < nt; t += SI( gridDim.x ) * BLOCK ) {
        const SI i = t / LN;
        const int sub = int( t % LN );
        const bool valid = i < A.n;
        const TV off = row_off_lanes<LN>( A, i, sub, valid, [&]( SI j ) { return y[ j ]; } );
        TV v = 0;
        if ( valid && sub == 0 ) {
            const TV yi = y[ i ];
            x[ i ] = fresh ? yi : x[ i ] + yi;
            v = ( fresh ? rhs[ i ] : r[ i ] ) - ( A.dia[ i ] * yi - off );
        }
        const int W = LN << sh;                          // the lanes of a packet ( `<= 32` )
        for ( int o = W / 2; o > 0; o /= 2 )
            v += __shfl_down_sync( 0xffffffffu, v, o, W );
        if ( valid && ( t & ( W - 1 ) ) == 0 )
            bc[ i >> sh ] = v;
    }
}

// ---- the transfers of the smoothed aggregation ------------------------------------------------------------------------

/// a rectangular CSR ( `P`: fine rows -> packets, `P^t`: packets -> fine rows )
template<class TC,class TV>
struct RectView {
    SI        n   = 0;
    const SI *row = nullptr;
    const TC *col = nullptr;
    const TV *val = nullptr;
};

/// `bc[ a ] = sum_k Pt[ a ][ k ] rho[ k ]`
template<class TC,class TV>
__global__ void __launch_bounds__( BLOCK ) restrict_sa( RectView<TC,TV> Pt, const TV *rho, TV *bc ) {
    const SI a = SI( blockIdx.x ) * BLOCK + threadIdx.x;
    if ( a >= Pt.n )
        return;
    TV s = 0;
    for ( SI k = Pt.row[ a ]; k < Pt.row[ a + 1 ]; ++k )
        s += Pt.val[ k ] * rho[ Pt.col[ k ] ];
    bc[ a ] = s;
}

/// `xo = x + P e`
template<class TC,class TV>
__global__ void __launch_bounds__( BLOCK ) prolong_sa( RectView<TC,TV> P, const TV *x, const TV *e, TV *xo ) {
    const SI i = SI( blockIdx.x ) * BLOCK + threadIdx.x;
    if ( i >= P.n )
        return;
    TV s = x[ i ];
    for ( SI k = P.row[ i ]; k < P.row[ i + 1 ]; ++k )
        s += P.val[ k ] * e[ P.col[ k ] ];
    xo[ i ] = s;
}

// ---- the bottom ------------------------------------------------------------------------------------------------------

/// THE COARSEST LEVEL IN ONE BLOCK: damped Jacobi from zero, the iterates in shared memory
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
/// connected laplacian, and for `b` of zero sum its solution has zero sum and solves `A x = b`: the zero-mean gauge, exactly.
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

// ---- the K-cycle ( two flexible-CG steps on a coarse level, preconditioned by the cycle below ) ------------------------

/// the K-cycle's scalars of a level, computed by the threads that finish its reductions
struct KCoef {
    double c, c1, c2;
};

/// first step: `t = A v1`; `( v1 . t, v1 . b )`
template<class TC,class TV>
struct KStep1 {
    CsrView<TC,TV> A;
    const TV *v1, *bk;
    TV           *t;
    __device__ void operator()( SI i, Sum2 &acc ) const {
        const TV ti = row_product( A, v1, i );
        t[ i ] = ti;
        acc.s0 += double( v1[ i ] ) * double( ti );
        acc.s1 += double( v1[ i ] ) * double( bk[ i ] );
    }
};
struct FinK1 { KCoef *k; __device__ void operator()( const Sum2 &s ) const { k->c = s.s0 != 0 ? s.s1 / s.s0 : 0.0; } };

template<class TV>
__global__ void __launch_bounds__( BLOCK ) kcycle_rc( SI n, const TV *bk, const TV *t, TV *rc, const KCoef *k ) {
    const SI i = SI( blockIdx.x ) * BLOCK + threadIdx.x;
    if ( i < n ) rc[ i ] = bk[ i ] - TV( k->c ) * t[ i ];
}

/// second step: `( v2 . A v1, v2 . A v2, v2 . rc )`
template<class TC,class TV>
struct KStep2 {
    CsrView<TC,TV> A;
    const TV *v2, *t, *rc;
    __device__ void operator()( SI i, Sum3 &acc ) const {
        const TV t2 = row_product( A, v2, i );
        acc.s0 += double( v2[ i ] ) * double( t[ i ] );
        acc.s1 += double( v2[ i ] ) * double( t2 );
        acc.s2 += double( v2[ i ] ) * double( rc[ i ] );
    }
};
struct FinK2 {
    KCoef *k;
    const Sum2 *s1;
    __device__ void operator()( const Sum3 &s2 ) const {
        const double rho1 = s1->s0, a1 = s1->s1, g2 = s2.s0, b2 = s2.s1, a2 = s2.s2;
        const double rho2 = b2 - g2 * g2 / ( rho1 != 0 ? rho1 : 1.0 );
        k->c2 = rho2 != 0 ? a2 / rho2 : 0.0;
        k->c1 = ( rho1 != 0 ? a1 / rho1 : 0.0 ) - ( rho1 != 0 && rho2 != 0 ? g2 * a2 / ( rho1 * rho2 ) : 0.0 );
    }
};

template<class TV>
__global__ void __launch_bounds__( BLOCK ) kcycle_combine( SI n, const TV *v1, const TV *v2, TV *e, const KCoef *k ) {
    const SI i = SI( blockIdx.x ) * BLOCK + threadIdx.x;
    if ( i < n ) e[ i ] = TV( k->c1 ) * v1[ i ] + TV( k->c2 ) * v2[ i ];
}

// ---- the coarse assembly of the plain aggregation --------------------------------------------------------------------

/// THE COARSE ROW `a` AS A MERGE: the fine rows of packet `a` are each sorted by column, so their columns mapped to packets
/// ( `col >> SH` ) are `S` sorted runs -- merged by their heads ( in registers: `SH` is a constant ), the duplicates summed in
/// the order of the runs, the packet itself dropped. The count pass and the fill pass run the same merge.
template<int SH,class TC,class TV,class F>
__device__ __forceinline__ void merge_packet( const CsrView<TC,TV> &A, SI a, F &&f ) {
    constexpr int S = 1 << SH;
    SI p[ S ], e[ S ];
    const SI r0 = a << SH;
#pragma unroll
    for ( int q = 0; q < S; ++q ) {
        const bool ok = r0 + q < A.n;
        p[ q ] = ok ? A.row[ r0 + q ] : 0;
        e[ q ] = ok ? A.row[ r0 + q + 1 ] : 0;
    }
    while ( true ) {
        SI k = -1;
#pragma unroll
        for ( int q = 0; q < S; ++q )
            if ( p[ q ] < e[ q ] ) {
                const SI c = SI( A.col[ p[ q ] ] ) >> SH;
                if ( k < 0 || c < k ) k = c;
            }
        if ( k < 0 )
            return;
        double s = 0;
#pragma unroll
        for ( int q = 0; q < S; ++q )
            for ( ; p[ q ] < e[ q ] && ( SI( A.col[ p[ q ] ] ) >> SH ) == k; ++p[ q ] )
                s += A.val[ p[ q ] ];
        if ( k != a )
            f( k, s );
    }
}

template<int SH,class TC,class TV>
__global__ void __launch_bounds__( BLOCK ) coarse_count( CsrView<TC,TV> A, SI nc, SI *cnt ) {
    const SI a = SI( blockIdx.x ) * BLOCK + threadIdx.x;
    if ( a >= nc )
        return;
    SI c = 0;
    merge_packet<SH>( A, a, [&]( SI, double ) { ++c; } );
    cnt[ a ] = c;
}

template<int SH,class TC,class TV>
__global__ void __launch_bounds__( BLOCK ) coarse_fill( CsrView<TC,TV> A, SI nc, const SI *crow, TC *ccol, TV *cval, TV *cdia ) {
    const SI a = SI( blockIdx.x ) * BLOCK + threadIdx.x;
    if ( a >= nc )
        return;
    SI q = crow[ a ];
    double d = 0;
    merge_packet<SH>( A, a, [&]( SI k, double s ) { ccol[ q ] = TC( k ); cval[ q ] = TV( s ); ++q; d += s; } );
    cdia[ a ] = TV( d > 0 ? d : 1.0 );
}

// ---- the hierarchy of the smoothed aggregation ------------------------------------------------------------------------

/// the packets of row `i` of `( I - w D^-1 A ) P0` in increasing order, `f( a, v )` per packet: the columns of a row are
/// sorted, so their packets come in runs; the packet of `i` itself is merged in at its place
template<class TC,class TV,class F>
__device__ __forceinline__ void sa_row_packets( const CsrView<TC,TV> &A, SI i, int sh, TV om, F &&f ) {
    const SI b = A.row[ i ], e = A.row[ i + 1 ];
    TV dsum = 0;
    for ( SI q = b; q < e; ++q ) dsum += A.val[ q ];
    const TV fct = dsum > 0 ? om / dsum : TV( 0 );
    const SI own = i >> sh;
    bool done = false;
    SI q = b;
    while ( q < e ) {
        const SI k = SI( A.col[ q ] ) >> sh;
        TV s = 0;
        for ( ; q < e && ( SI( A.col[ q ] ) >> sh ) == k; ++q )
            s += A.val[ q ];
        if ( ! done && own < k ) { f( own, 1 - om ); done = true; }
        if ( k == own ) { f( own, 1 - om + fct * s ); done = true; }
        else f( k, fct * s );
    }
    if ( ! done ) f( own, 1 - om );
}

/// THE ROW `i` OF `P`, TRUNCATED THEN RENORMALIZED ( what keeps the constants in the range of `P` ): `f( a, v )` per kept
/// entry ( in the precision of the level )
template<class TC,class TV,class F>
__device__ __forceinline__ void sa_row( const CsrView<TC,TV> &A, SI i, int sh, TV om, TV trunc, F &&f ) {
    TV mx = 0;
    sa_row_packets( A, i, sh, om, [&]( SI, TV v ) { const TV av = v < 0 ? -v : v; if ( av > mx ) mx = av; } );
    const TV th = trunc * mx;
    TV sum = 0;
    sa_row_packets( A, i, sh, om, [&]( SI, TV v ) { if ( ( v < 0 ? -v : v ) >= th && v != 0 ) sum += v; } );
    const TV inv = sum > 0 ? 1 / sum : TV( 1 );
    sa_row_packets( A, i, sh, om, [&]( SI a, TV v ) { if ( ( v < 0 ? -v : v ) >= th && v != 0 ) f( a, v * inv ); } );
}

template<class TC,class TV>
__global__ void __launch_bounds__( BLOCK ) sa_p_count( CsrView<TC,TV> A, int sh, TV om, TV trunc, SI *cnt ) {
    const SI i = SI( blockIdx.x ) * BLOCK + threadIdx.x;
    if ( i >= A.n )
        return;
    SI c = 0;
    sa_row( A, i, sh, om, trunc, [&]( SI, TV ) { ++c; } );
    cnt[ i ] = c;
}

template<class TC,class TV>
__global__ void __launch_bounds__( BLOCK ) sa_p_fill( CsrView<TC,TV> A, int sh, TV om, TV trunc, const SI *prow, TC *pcol, TV *pval ) {
    const SI i = SI( blockIdx.x ) * BLOCK + threadIdx.x;
    if ( i >= A.n )
        return;
    SI q = prow[ i ];
    sa_row( A, i, sh, om, trunc, [&]( SI a, TV v ) { pcol[ q ] = TC( a ); pval[ q ] = v; ++q; } );
}

/// `P^t` BY TRANSPOSITION: the entries of `P` counted per column ( atomics ), placed by a cursor per column ( atomics: any
/// order ), then each row of `P^t` sorted by fine index ( `lap_sort` ): the values are copied, not summed, so the result is
/// the same at every run
template<class TC,class TV>
__global__ void __launch_bounds__( BLOCK ) pt_count( RectView<TC,TV> P, SI *cnt ) {
    const SI i = SI( blockIdx.x ) * BLOCK + threadIdx.x;
    if ( i >= P.n )
        return;
    for ( SI q = P.row[ i ]; q < P.row[ i + 1 ]; ++q )
        atomic_inc( cnt + P.col[ q ] );
}

template<class TC,class TV>
__global__ void __launch_bounds__( BLOCK ) pt_fill( RectView<TC,TV> P, SI *at, TC *tcol, TV *tval ) {
    const SI i = SI( blockIdx.x ) * BLOCK + threadIdx.x;
    if ( i >= P.n )
        return;
    for ( SI q = P.row[ i ]; q < P.row[ i + 1 ]; ++q ) {
        const SI pos = atomic_inc( at + P.col[ q ] );
        tcol[ pos ] = TC( i );
        tval[ pos ] = P.val[ q ];
    }
}

/// a product to the fixed point ( in the precision of the level: Turing's double is 1/32 of its float )
__device__ __forceinline__ long long to_fixed( double v ) { return __double2ll_rn( v ); }
__device__ __forceinline__ long long to_fixed( float v ) { return __float2ll_rn( v ); }

/// a signed 64-bit fixed-point sum kept as two 32-bit words ( native shared-memory atomics; the carry of the low word
/// goes to the high one ): the same sum whatever the order of the additions
__device__ __forceinline__ void fx_add( unsigned *lo, int *hi, long long q ) {
    const unsigned l = unsigned( q );
    int h = int( q >> 32 );
    const unsigned old = atomicAdd( lo, l );
    if ( old + l < old ) h += 1;
    if ( h ) atomicAdd( hi, h );
}
__device__ __forceinline__ long long fx_value( unsigned lo, int hi ) {
    return ( long long )( ( ( unsigned long long ) unsigned( hi ) << 32 ) | lo );
}

/// the scale of the fixed point of a coarse row: `2^61` over a power of two above `2 len max_i dia_i` ( the entries of `P` are
/// in `( 0, 1 ]` and its rows sum to one, so this bounds the sum of the magnitudes of the row's contributions )
__device__ __forceinline__ double fx_scale_bound( double md, SI len ) {
    const double bound = 2 * double( len ) * md;
    return bound > 0 ? ldexp( 1.0, 61 - ilogb( bound ) ) : 1.0;
}

/// THE SAME PRODUCT ON A SMALL COARSE LEVEL ( `nc <= GAL_DENSE` ): one BLOCK per coarse row, the row accumulated DENSE in
/// shared memory ( no hashing, and the compaction gives the columns in order ) -- the deep levels, where few coarse rows
/// receive long lists of contributions
constexpr int GAL_DENSE = 4096;
constexpr int GAL_DENSE_BLOCK = 256;

template<class TC,class TV,bool FILL>
__global__ void __launch_bounds__( GAL_DENSE_BLOCK ) sa_galerkin_dense( CsrView<TC,TV> A, RectView<TC,TV> P, RectView<TC,TV> Pt, SI *cnt,
                                                                       const SI *crow, TC *ccol, TV *cval, TV *cdia ) {
    __shared__ unsigned lo[ GAL_DENSE ];
    __shared__ int      hi[ GAL_DENSE ];
    __shared__ unsigned char mk[ FILL ? GAL_DENSE : 1 ];   ///< FILL: the columns touched ( a sum may round to zero, the column stays:
                                                             ///< the count pass saw it -- a gap would leave its slot unwritten )
    __shared__ int      pre[ GAL_DENSE_BLOCK ];
    __shared__ double   red[ GAL_DENSE_BLOCK / 32 ];
    const int t = threadIdx.x;
    const SI a = blockIdx.x, nc = Pt.n;
    for ( SI b = t; b < nc; b += GAL_DENSE_BLOCK ) { lo[ b ] = 0; hi[ b ] = 0; if constexpr ( FILL ) mk[ b ] = 0; }
    const SI tb = Pt.row[ a ], te = Pt.row[ a + 1 ];
    double scale = 1;
    if constexpr ( FILL ) {
        double md = 0;
        for ( SI k = tb + t; k < te; k += GAL_DENSE_BLOCK ) md = fmax( md, double( A.dia[ Pt.col[ k ] ] ) );
        for ( int o = 16; o > 0; o /= 2 ) md = fmax( md, __shfl_xor_sync( 0xffffffffu, md, o ) );
        if ( ( t & 31 ) == 0 ) red[ t >> 5 ] = md;
        __syncthreads();
        md = 0;
        for ( int q = 0; q < GAL_DENSE_BLOCK / 32; ++q ) md = fmax( md, red[ q ] );
        scale = fx_scale_bound( md, te - tb );
    }
    __syncthreads();
    for ( SI k = tb; k < te; ++k ) {
        const SI i = Pt.col[ k ];
        const double wi = Pt.val[ k ];
        const SI rb = A.row[ i ], re = A.row[ i + 1 ];
        for ( SI u = t; u <= re - rb; u += GAL_DENSE_BLOCK ) {
            const SI j = u == 0 ? i : SI( A.col[ rb + u - 1 ] );
            const double c = u == 0 ? double( A.dia[ i ] ) : - double( A.val[ rb + u - 1 ] );
            for ( SI q = P.row[ j ]; q < P.row[ j + 1 ]; ++q ) {
                const SI b = P.col[ q ];
                if constexpr ( FILL ) { fx_add( lo + b, hi + b, __double2ll_rn( wi * c * double( P.val[ q ] ) * scale ) ); mk[ b ] = 1; }
                else hi[ b ] = 1;                        // ( the count marks the columns only )
            }
        }
    }
    __syncthreads();
    // the columns in order: a scan of the marks by chunks of the block
    SI base = FILL ? crow[ a ] : 0;
    double d = 0;
    for ( SI b0 = 0; b0 < nc; b0 += GAL_DENSE_BLOCK ) {
        const SI b = b0 + t;
        bool on = false;
        long long v = 0;
        if ( b < nc ) {
            if constexpr ( FILL ) { v = fx_value( lo[ b ], hi[ b ] ); on = mk[ b ] != 0; }
            else on = hi[ b ] != 0;
        }
        if ( b == a ) { if constexpr ( FILL ) d = double( v ) / scale; on = false; }
        pre[ t ] = on;
        __syncthreads();
        for ( int o = 1; o < GAL_DENSE_BLOCK; o *= 2 ) {
            const int x = t >= o ? pre[ t - o ] : 0;
            __syncthreads();
            pre[ t ] += x;
            __syncthreads();
        }
        const int tot = pre[ GAL_DENSE_BLOCK - 1 ];
        if constexpr ( FILL ) if ( on ) {
            const SI q = base + pre[ t ] - 1;
            ccol[ q ] = TC( b );
            cval[ q ] = TV( - double( v ) / scale );
        }
        base += tot;
        __syncthreads();
    }
    if constexpr ( FILL ) {
        // the diagonal: held by the thread of its column
        if ( a % GAL_DENSE_BLOCK == t ) cdia[ a ] = TV( d > 0 ? d : 1.0 );
    } else if ( t == 0 )
        cnt[ a ] = base;
}

constexpr int GAL_WARPS = 4;                             ///< warps per block of the product

/// `PLAIN`: `P = P0` ( the plain aggregation, `P[ j ] = { ( j >> sh, 1 ) }`, `P^t[ a ]` = the members of `a` ), implicit
template<class TC,class TV,bool PLAIN>
struct Transfer {
    RectView<TC,TV> P, Pt;
    int             sh;
    SI              nf;
    __device__ SI t_begin( SI a ) const { return PLAIN ? a << sh : Pt.row[ a ]; }
    __device__ SI t_end( SI a ) const { return PLAIN ? min( ( a + 1 ) << sh, nf ) : Pt.row[ a + 1 ]; }
    __device__ SI t_col( SI k ) const { return PLAIN ? k : SI( Pt.col[ k ] ); }
    __device__ TV t_val( SI k ) const { return PLAIN ? TV( 1 ) : Pt.val[ k ]; }
    __device__ SI p_begin( SI j ) const { return PLAIN ? j : P.row[ j ]; }
    __device__ SI p_end( SI j ) const { return PLAIN ? j + 1 : P.row[ j + 1 ]; }
    __device__ SI p_col( SI q ) const { return PLAIN ? q >> sh : SI( P.col[ q ] ); }
    __device__ TV p_val( SI q ) const { return PLAIN ? TV( 1 ) : P.val[ q ]; }
};

template<int HC,class TC,class TV,bool FILL,bool PLAIN>
__global__ void __launch_bounds__( 32 * GAL_WARPS ) sa_galerkin( CsrView<TC,TV> A, Transfer<TC,TV,PLAIN> T, SI nc, SI *cnt,
                                                                const SI *crow, TC *ccol, TV *cval, TV *cdia, int *fail ) {
    __shared__ int                keys[ GAL_WARPS ][ HC ];
    __shared__ unsigned           acl[ GAL_WARPS ][ FILL ? HC : 1 ];
    __shared__ int                ach[ GAL_WARPS ][ FILL ? HC : 1 ];
    const int lane = threadIdx.x & 31, w = threadIdx.x >> 5;
    const SI a = SI( blockIdx.x ) * GAL_WARPS + w;
    if ( a >= nc )
        return;
    int *K = keys[ w ];
    unsigned *VL = acl[ w ];
    int *VH = ach[ w ];
    for ( int h = lane; h < HC; h += 32 ) {
        K[ h ] = -1;
        if constexpr ( FILL ) { VL[ h ] = 0; VH[ h ] = 0; }
    }
    const SI tb = T.t_begin( a ), te = T.t_end( a );
    double scale = 0;
    if constexpr ( FILL ) {
        double md = 0;
        for ( SI k = tb + lane; k < te; k += 32 ) md = fmax( md, double( A.dia[ T.t_col( k ) ] ) );
        for ( int o = 16; o > 0; o /= 2 ) md = fmax( md, __shfl_xor_sync( 0xffffffffu, md, o ) );
        scale = fx_scale_bound( md, te - tb );
    }
    __syncwarp();
    bool full = false;
    __shared__ SI     s_i[ GAL_WARPS ][ 32 ];
    __shared__ double s_w[ GAL_WARPS ][ 32 ];
    __shared__ int    s_ex[ GAL_WARPS ][ 33 ];
    for ( SI k0 = tb; k0 < te; k0 += 32 ) {
        // the pairs ( i, j ) of this chunk of the row of `P^t`, numbered by a scan and spread over the lanes
        const SI k = k0 + lane;
        const bool kv = k < te;
        const SI i = kv ? T.t_col( k ) : 0;
        const int len = kv ? int( A.row[ i + 1 ] - A.row[ i ] ) + 1 : 0;
        int incl = len;
        for ( int o = 1; o < 32; o *= 2 ) {
            const int x = __shfl_up_sync( 0xffffffffu, incl, o );
            if ( lane >= o ) incl += x;
        }
        const int total = __shfl_sync( 0xffffffffu, incl, 31 );
        s_i[ w ][ lane ] = i;
        s_w[ w ][ lane ] = kv ? double( T.t_val( k ) ) : 0.0;
        s_ex[ w ][ lane + 1 ] = incl;
        if ( lane == 0 ) s_ex[ w ][ 0 ] = 0;
        __syncwarp();
        for ( int f = lane; f < total; f += 32 ) {
            int lo = 0, hi = 31;
            while ( lo < hi ) {
                const int mid = ( lo + hi + 1 ) / 2;
                if ( s_ex[ w ][ mid ] <= f ) lo = mid; else hi = mid - 1;
            }
            const SI ii = s_i[ w ][ lo ];
            const int u = f - s_ex[ w ][ lo ];
            const SI rb = A.row[ ii ];
            const SI j = u == 0 ? ii : SI( A.col[ rb + u - 1 ] );
            const TV wc = TV( s_w[ w ][ lo ] * scale ) * ( u == 0 ? A.dia[ ii ] : - A.val[ rb + u - 1 ] );
            for ( SI q = T.p_begin( j ); q < T.p_end( j ); ++q ) {
                const int key = int( T.p_col( q ) );
                int h = int( ( unsigned( key ) * 2654435761u ) & unsigned( HC - 1 ) );
                for ( int probe = 0; ; ++probe ) {
                    if ( probe == HC ) { full = true; break; }
                    const int prev = atomicCAS( K + h, -1, key );
                    if ( prev == -1 || prev == key ) {
                        if constexpr ( FILL ) fx_add( VL + h, VH + h, to_fixed( wc * T.p_val( q ) ) );
                        break;
                    }
                    h = ( h + 1 ) & ( HC - 1 );
                }
            }
        }
        __syncwarp();
    }
    if ( __any_sync( 0xffffffffu, full ) ) {
        if ( lane == 0 ) *fail = 1;
        return;
    }
    __syncwarp();
    // compaction of the occupied slots, in slot order ( each chunk read before it is written: the writes go below )
    int m = 0;
    for ( int h0 = 0; h0 < HC; h0 += 32 ) {
        const int key = K[ h0 + lane ];
        unsigned vl = 0;
        int vh = 0;
        if constexpr ( FILL ) { vl = VL[ h0 + lane ]; vh = VH[ h0 + lane ]; }
        const unsigned ball = __ballot_sync( 0xffffffffu, key >= 0 );
        __syncwarp();
        if ( key >= 0 ) {
            const int pos = m + __popc( ball & ( ( 1u << lane ) - 1 ) );
            K[ pos ] = key;
            if constexpr ( FILL ) { VL[ pos ] = vl; VH[ pos ] = vh; }
        }
        m += __popc( ball );
        __syncwarp();
    }
    if constexpr ( ! FILL ) {
        bool has_diag = false;
        for ( int h = lane; h < m; h += 32 ) has_diag |= K[ h ] == int( a );
        const bool d = __any_sync( 0xffffffffu, has_diag );
        if ( lane == 0 ) cnt[ a ] = m - ( d ? 1 : 0 );
        return;
    } else {
        int p2 = 1;
        while ( p2 < m ) p2 *= 2;
        for ( int h = m + lane; h < p2; h += 32 ) K[ h ] = 0x7fffffff;
        __syncwarp();
        for ( int size = 2; size <= p2; size *= 2 )
            for ( int stride = size / 2; stride > 0; stride /= 2 ) {
                for ( int t = lane; t < p2 / 2; t += 32 ) {
                    const int i = 2 * t - ( t & ( stride - 1 ) ), j = i + stride;
                    const bool up = ( i & size ) == 0;
                    const int ki = K[ i ], kj = K[ j ];
                    if ( ( ki > kj ) == up ) {
                        K[ i ] = kj; K[ j ] = ki;
                        const unsigned li = VL[ i ]; VL[ i ] = VL[ j ]; VL[ j ] = li;
                        const int hi = VH[ i ]; VH[ i ] = VH[ j ]; VH[ j ] = hi;
                    }
                }
                __syncwarp();
            }
        // the row: the off-diagonals in column order, the diagonal apart
        int pd = m;
        for ( int h = lane; h < m; h += 32 ) if ( K[ h ] == int( a ) ) pd = h;
        for ( int o = 16; o > 0; o /= 2 ) pd = min( pd, __shfl_xor_sync( 0xffffffffu, pd, o ) );
        const SI q0 = crow[ a ];
        for ( int h = lane; h < m; h += 32 ) {
            const double v = double( fx_value( VL[ h ], VH[ h ] ) ) / scale;
            if ( h == pd ) { cdia[ a ] = TV( v > 0 ? v : 1.0 ); continue; }
            const SI q = q0 + h - ( h > pd ? 1 : 0 );
            ccol[ q ] = TC( K[ h ] );
            cval[ q ] = TV( - v );
        }
        if ( pd == m && lane == 0 ) cdia[ a ] = TV( 1 );
    }
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

/// `x += alpha p`, `r -= alpha q`; `r . r`. `rf`: `r` in the precision of the levels too ( the cycle's input ), if not null.
template<class TV>
struct FcgUpdate {
    const double    *p, *q;
    double          *x, *r;
    TV              *rf;
    const CgScalars *sc;
    __device__ void operator()( SI i, Sum1 &acc ) const {
        const double a = sc->alpha;
        x[ i ] += a * p[ i ];
        const double ri = r[ i ] - a * q[ i ];
        r[ i ] = ri;
        if ( rf ) rf[ i ] = TV( ri );
        acc.s += ri * ri;
    }
};

/// `z = zf` ( the cycle's output, in double ); `( z . q, r . z )`
template<class TV>
struct FcgDots {
    const TV     *zf;
    double       *z;
    const double *q, *r;
    __device__ void operator()( SI i, Sum2 &acc ) const {
        const double zi = zf[ i ];
        z[ i ] = zi;
        acc.s0 += zi * q[ i ];
        acc.s1 += r[ i ] * zi;
    }
};

__global__ void __launch_bounds__( BLOCK ) fcg_direction( SI n, const double *z, double *p, const CgScalars *sc ) {
    const SI i = SI( blockIdx.x ) * BLOCK + threadIdx.x;
    if ( i < n ) p[ i ] = z[ i ] + sc->beta * p[ i ];
}

template<class TA,class TB>
__global__ void __launch_bounds__( BLOCK ) convert_values( SI n, const TA *src, TB *dst ) {
    const SI i = SI( blockIdx.x ) * BLOCK + threadIdx.x;
    if ( i < n ) dst[ i ] = TB( src[ i ] );
}

/// the same over the entries of a CSR only ( their count `row[ nr ]` read on the card: past it, nothing was written )
template<class TA,class TB>
__global__ void __launch_bounds__( BLOCK ) convert_csr_values( SI cap, const SI *row, SI nr, const TA *src, TB *dst ) {
    const SI i = SI( blockIdx.x ) * BLOCK + threadIdx.x;
    if ( i < cap && i < row[ nr ] ) dst[ i ] = TB( src[ i ] );
}

/// A FIXED SEQUENCE OF LAUNCHES, captured once and replayed whole ( a CUDA graph ). Captured again only when what the
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
    template<class F>
    void run( const CudaQueue &q, const void *k, int g, F &&body ) {
        if ( ! enabled() ) { body(); return; }
        if ( ! valid_for( k, g ) )
            capture( q, k, g, body );
        launch_graph( q, exec );
    }
    void release() { if ( exec ) cudaGraphExecDestroy( exec ); exec = nullptr; }
};

/// `SDOT_CARD_LIN_DUMP=prefix`: every system the card solves written to `prefix_<k>.bin` ( a debugging and benchmarking
/// aid: the systems of a Newton solve replayed outside of it ). Layout: `n`, `nnz`, `sizeof( TC )` as int64, then `row`
/// ( int64 ), `col` ( `TC` ), `val`, `dia`, `b` ( double ).
template<class TC>
void dump_system( const CudaQueue &queue, const CsrView<TC> &A, const double *b ) {
    static const char *prefix = std::getenv( "SDOT_CARD_LIN_DUMP" );
    static int count = 0;
    if ( ! prefix || ! *prefix )
        return;
    const SI n = A.n;
    std::vector<SI> row( n + 1 );
    read_back( queue, row.data(), A.row, n + 1 );
    const SI nnz = row[ n ];
    std::vector<TC> col( nnz );
    std::vector<double> val( nnz ), dia( n ), rhs( n );
    read_back( queue, col.data(), A.col, nnz );
    read_back( queue, val.data(), A.val, nnz );
    read_back( queue, dia.data(), A.dia, n );
    read_back( queue, rhs.data(), b, n );
    const std::string name = std::string( prefix ) + "_" + std::to_string( count++ ) + ".bin";
    if ( FILE *f = std::fopen( name.c_str(), "wb" ) ) {
        const SI head[ 3 ] = { n, nnz, SI( sizeof( TC ) ) };
        std::fwrite( head, sizeof( SI ), 3, f );
        std::fwrite( row.data(), sizeof( SI ), n + 1, f );
        std::fwrite( col.data(), sizeof( TC ), nnz, f );
        std::fwrite( val.data(), sizeof( double ), nnz, f );
        std::fwrite( dia.data(), sizeof( double ), n, f );
        std::fwrite( rhs.data(), sizeof( double ), n, f );
        std::fclose( f );
    }
}

/// `f( std::integral_constant<int,SH>() )` for the packet's shift `sh` ( kernels whose registers are sized on it )
template<class F>
void with_shift( int sh, F &&f ) {
    switch ( sh ) {
        case 1:  f( std::integral_constant<int,1>() ); break;
        case 2:  f( std::integral_constant<int,2>() ); break;
        case 3:  f( std::integral_constant<int,3>() ); break;
        case 4:  f( std::integral_constant<int,4>() ); break;
        default: f( std::integral_constant<int,5>() ); break;
    }
}

/// `f( std::integral_constant<int,LN>() )` for `ln` lanes per row
template<class F>
void with_lanes( int ln, F &&f ) {
    switch ( ln ) {
        case 4:  f( std::integral_constant<int,4>() ); break;
        case 2:  f( std::integral_constant<int,2>() ); break;
        default: f( std::integral_constant<int,1>() ); break;
    }
}

template<class TV>
__global__ void __launch_bounds__( BLOCK ) sum_into( SI n, const TV *a, const TV *b, TV *out ) {
    const SI i = SI( blockIdx.x ) * BLOCK + threadIdx.x;
    if ( i < n ) out[ i ] = a[ i ] + b[ i ];
}

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
        int         deg = 1;                             ///< the Chebyshev degree of this level
        bool        sa = false;                          ///< its transfer to the next level is the smoothed one
        int         ln = 1;                              ///< lanes per row of its matrix-vector kernels
        TV         *rlx = nullptr, *x = nullptr, *r = nullptr, *y = nullptr, *z = nullptr, *rho = nullptr, *xp = nullptr;
        TV         *bk = nullptr, *e = nullptr, *v1 = nullptr, *v2 = nullptr, *t = nullptr, *rc = nullptr;
        ChebCoef   *cc = nullptr;
        KCoef      *kc = nullptr;
        WideSlot<Max1> lmax;
        WideSlot<Sum2> s2;
        WideSlot<Sum3> s3;
        // the smoothed aggregation: `P` ( this level -> the next ) and `P^t`
        RectView<TC,TV> P, Pt;
        SI         *prow = nullptr, *trow = nullptr;
        TC         *pcol = nullptr, *tcol = nullptr;
        TV         *pval = nullptr, *tval = nullptr;
        SI          p_cap = 0, t_cap = 0;
    };

    LinOptions         o;
    LinStats           st;
    SI                 n = 0, nnz_cap = 0;
    double            *r = nullptr, *z = nullptr, *p = nullptr, *q = nullptr;
    CgScalars         *sc = nullptr;
    WideSlot<Sum1>     w1;
    WideSlot<Sum2>     w2;
    std::vector<Level> lev;
    LapWork<SI>        scan;
    double            *U = nullptr, *AU = nullptr;       ///< the recycled solutions ( `recycle x n` ), and `A U`
    double            *bottom = nullptr;                 ///< the exact inverse of the coarsest level ( `BOTTOM_DENSE^2` )
    TV                *val0 = nullptr, *dia0 = nullptr;  ///< the fine level in `TV` ( when it is not double )
    TV                *rf = nullptr, *zf = nullptr;      ///< the cycle's input and output on the fine level ( idem )
    int               *fail = nullptr;
    int                nb_u = 0, since = 0, since_slot = 0;
    bool               built = false;
    int                generation = 0;                   ///< bumped when a level's storage moves: the graphs are captured again
    Graph              g_iter;

    ~CardLinear() { g_iter.release(); }

    /// `n` unknowns, `nnz_cap` the capacity of the fine CSR
    bool prepare( auto &allocator, SI n_, SI nnz_cap_, const LinOptions &o_ ) {
        o = o_;
        o.stop = std::min( std::max( o.stop, 16 ), 2048 );
        o.recycle = std::min( std::max( o.recycle, 0 ), 4 );
        o.shift = std::min( std::max( o.shift, 1 ), 5 );
        o.smoothed = std::max( o.smoothed, 0 );
        if ( o.kfrom < 0 ) o.kfrom = std::max( 1, o.smoothed + 1 );
        o.nu = std::min( std::max( o.nu, 1 ), MAX_DEG );
        if ( o.nu0 > 0 ) o.nu0 = std::min( o.nu0, MAX_DEG );
        n = n_;
        nnz_cap = nnz_cap_;
        auto vec = [&]( SI m ) { return static_cast<double *>( take( allocator, SI( sizeof( double ) ) * std::max<SI>( m, 1 ) ) ); };
        auto tvec = [&]( SI m ) { return static_cast<TV *>( take( allocator, SI( sizeof( TV ) ) * std::max<SI>( m, 1 ) ) ); };
        r = vec( n ); z = vec( n ); p = vec( n ); q = vec( n );
        sc = static_cast<CgScalars *>( take( allocator, SI( sizeof( CgScalars ) ) ) );
        fail = static_cast<int *>( take( allocator, SI( sizeof( int ) ) ) );
        bool ok = r && z && p && q && sc && fail && w1.take_from( allocator ) && w2.take_from( allocator );
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
                L.deg = lev.empty() && o.nu0 > 0 ? o.nu0 : o.nu;
                L.sa = int( lev.size() ) < o.smoothed;
                L.rlx = tvec( m ); L.x = tvec( m ); L.r = tvec( m ); L.y = tvec( m ); L.z = tvec( m ); L.rho = tvec( m ); L.xp = tvec( m );
                L.cc = static_cast<ChebCoef *>( take( allocator, SI( sizeof( ChebCoef ) ) ) );
                L.kc = static_cast<KCoef *>( take( allocator, SI( sizeof( KCoef ) ) ) );
                ok = ok && L.rlx && L.x && L.r && L.y && L.z && L.rho && L.xp && L.cc && L.kc && L.lmax.take_from( allocator )
                     && L.s2.take_from( allocator ) && L.s3.take_from( allocator );
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
            for ( size_t l = 0; l + 1 < lev.size(); ++l )
                if ( lev[ l ].sa ) {
                    lev[ l ].prow = static_cast<SI *>( take( allocator, SI( sizeof( SI ) ) * ( lev[ l ].A.n + 1 ) ) );
                    lev[ l ].trow = static_cast<SI *>( take( allocator, SI( sizeof( SI ) ) * ( lev[ l + 1 ].A.n + 1 ) ) );
                    ok = ok && lev[ l ].prow && lev[ l ].trow;
                }
        }
        return ok;
    }

    /// the lanes per row of level `L`'s kernels, from its entries per row
    void set_lanes( const CudaQueue &queue, Level &L ) {
        SI nz = 0;
        read_back( queue, &nz, L.A.row + L.A.n, 1 );
        const double per_row = double( nz ) / double( std::max<SI>( L.A.n, 1 ) );
        L.ln = o.lanes >= 4 && per_row >= 3 ? 4 : o.lanes >= 2 && per_row >= 2 ? 2 : 1;
        while ( ( L.ln << o.shift ) > 32 ) L.ln /= 2;     // ( a packet's lanes in one warp: the restriction's shuffles )
    }

    void relax( const CudaQueue &queue, Level &L ) {
        reduce_fin( queue, L.A.n, Relax<TC,TV>{ L.A, L.rlx }, L.lmax, FinCheb{ o.cheb, L.cc } );
    }

    /// the count of a CSR on the card ( `cnt` -> `row` by a scan ), read back; room from the pool if the capacity is short
    template<class T>
    bool csr_room( const CudaQueue &queue, auto &allocator, SI nr, SI *row, SI &cap, TC *&col, T *&val ) {
        zero_fill( queue, scan.cnt + nr, SI( sizeof( SI ) ) );   // ( the counts are `nr`: the scan reads one more )
        exclusive_scan( queue, scan.sums, static_cast<const SI *>( scan.cnt ), row, nr + 1 );
        SI nnz = 0;
        read_back( queue, &nnz, static_cast<const SI *>( row + nr ), 1 );
        if ( nnz > cap ) {                               // the first build, or a matrix that grew: room from the pool
            cap = nnz + nnz / 4 + 64;
            ++generation;
            col = static_cast<TC *>( take( allocator, SI( sizeof( TC ) ) * cap ) );
            val = static_cast<T *>( take( allocator, SI( sizeof( T ) ) * cap ) );
            if ( ! col || ! val )
                return false;
        }
        return true;
    }

    template<int HC>
    void galerkin_launch( const CudaQueue &queue, Level &F, Level &C, bool fill ) {
        const SI nc = C.A.n;
        const int grid = int( ( nc + GAL_WARPS - 1 ) / GAL_WARPS );
        if ( F.sa ) {
            const Transfer<TC,TV,false> T{ F.P, F.Pt, o.shift, F.A.n };
            if ( fill ) launch_kernel( queue, &sa_galerkin<HC,TC,TV,true,false>, grid, 32 * GAL_WARPS, 0, F.A, T, nc, scan.cnt, ( const SI * ) C.row, C.col, C.val, C.dia, fail );
            else        launch_kernel( queue, &sa_galerkin<HC,TC,TV,false,false>, grid, 32 * GAL_WARPS, 0, F.A, T, nc, scan.cnt, ( const SI * ) C.row, C.col, C.val, C.dia, fail );
        } else {
            const Transfer<TC,TV,true> T{ {}, {}, o.shift, F.A.n };
            if ( fill ) launch_kernel( queue, &sa_galerkin<HC,TC,TV,true,true>, grid, 32 * GAL_WARPS, 0, F.A, T, nc, scan.cnt, ( const SI * ) C.row, C.col, C.val, C.dia, fail );
            else        launch_kernel( queue, &sa_galerkin<HC,TC,TV,false,true>, grid, 32 * GAL_WARPS, 0, F.A, T, nc, scan.cnt, ( const SI * ) C.row, C.col, C.val, C.dia, fail );
        }
    }
    void galerkin_any( int hc, const CudaQueue &queue, Level &F, Level &C, bool fill ) {
        switch ( hc ) {
            case 64:  galerkin_launch<64>( queue, F, C, fill ); break;
            case 128: galerkin_launch<128>( queue, F, C, fill ); break;
            case 256: galerkin_launch<256>( queue, F, C, fill ); break;
            default:  galerkin_launch<512>( queue, F, C, fill ); break;
        }
    }
    bool galerkin( const CudaQueue &queue, auto &allocator, Level &F, Level &C ) {
        const SI nc = C.A.n;
        if ( nc <= GAL_DENSE && F.sa ) {
            launch_kernel( queue, &sa_galerkin_dense<TC,TV,false>, int( nc ), GAL_DENSE_BLOCK, 0, F.A, F.P, F.Pt, scan.cnt, ( const SI * ) C.row, C.col, C.val, C.dia );
            if ( ! csr_room( queue, allocator, nc, C.row, C.nnz_cap, C.col, C.val ) )
                return false;
            launch_kernel( queue, &sa_galerkin_dense<TC,TV,true>, int( nc ), GAL_DENSE_BLOCK, 0, F.A, F.P, F.Pt, scan.cnt, ( const SI * ) C.row, C.col, C.val, C.dia );
            return true;
        }
        // a first size: twice the entries of a fine row times a packet, rounded up ( the coarse rows are wider )
        SI fnnz = 0;
        read_back( queue, &fnnz, F.A.row + F.A.n, 1 );
        int hc = 64;
        while ( hc < 512 && hc < 4 * ( fnnz / std::max<SI>( F.A.n, 1 ) + 1 ) * 2 ) hc *= 2;
        while ( true ) {
            zero_fill( queue, fail, SI( sizeof( int ) ) );
            galerkin_any( hc, queue, F, C, false );
            int h = 0;
            read_back( queue, &h, ( const int * ) fail, 1 );
            if ( ! h ) break;
            if ( hc >= 512 ) {
                std::printf( "sdot: a coarse row of the card's smoothed aggregation has more than 512 entries\n" );
                return false;
            }
            hc *= 2;
        }
        if ( ! csr_room( queue, allocator, nc, C.row, C.nnz_cap, C.col, C.val ) )
            return false;
        galerkin_any( hc, queue, F, C, true );
        return true;
    }

    /// the coarse levels from the fine one ( a few counts read back per level ); `false` if the pool said no
    bool build( const CudaQueue &queue, auto &allocator ) {
        const int sh = o.shift;
        relax( queue, lev[ 0 ] );
        set_lanes( queue, lev[ 0 ] );
        for ( size_t l = 0; l + 1 < lev.size(); ++l ) {
            Level &F = lev[ l ];
            Level &C = lev[ l + 1 ];
            const SI nf = F.A.n, nc = C.A.n;
            SI fnnz = 0;
            if ( ! F.sa && l > 0 ) read_back( queue, &fnnz, F.A.row + nf, 1 );
            if ( ! F.sa && ( l == 0 || fnnz <= o.plain_warp * nf ) ) {
                with_shift( sh, [&]( auto c ) { launch_kernel( queue, &coarse_count<decltype( c )::value,TC,TV>, blocks_for( nc ), BLOCK, 0, F.A, nc, scan.cnt ); } );
                if ( ! csr_room( queue, allocator, nc, C.row, C.nnz_cap, C.col, C.val ) )
                    return false;
                with_shift( sh, [&]( auto c ) {
                    launch_kernel( queue, &coarse_fill<decltype( c )::value,TC,TV>, blocks_for( nc ), BLOCK, 0, F.A, nc, ( const SI * ) C.row, C.col, C.val, C.dia );
                } );
            } else if ( ! F.sa ) {
                if ( ! galerkin( queue, allocator, F, C ) )
                    return false;
            } else {
                // `P`
                launch_kernel( queue, &sa_p_count<TC,TV>, blocks_for( nf ), BLOCK, 0, F.A, sh, TV( o.omega ), TV( o.truncate ), scan.cnt );
                if ( ! csr_room( queue, allocator, nf, F.prow, F.p_cap, F.pcol, F.pval ) )
                    return false;
                launch_kernel( queue, &sa_p_fill<TC,TV>, blocks_for( nf ), BLOCK, 0, F.A, sh, TV( o.omega ), TV( o.truncate ), ( const SI * ) F.prow, F.pcol, F.pval );
                F.P = RectView<TC,TV>{ nf, F.prow, F.pcol, F.pval };
                // `P^t`
                zero_fill( queue, scan.cnt, SI( sizeof( SI ) ) * ( nc + 1 ) );
                launch_kernel( queue, &pt_count<TC,TV>, blocks_for( nf ), BLOCK, 0, F.P, scan.cnt );
                if ( ! csr_room( queue, allocator, nc, F.trow, F.t_cap, F.tcol, F.tval ) )
                    return false;
                cuda_check( cudaMemcpyAsync( scan.at, F.trow, sizeof( SI ) * nc, cudaMemcpyDeviceToDevice, queue.stream ), "copy of the row starts" );
                launch_kernel( queue, &pt_fill<TC,TV>, blocks_for( nf ), BLOCK, 0, F.P, scan.at, F.tcol, F.tval );
                launch_kernel( queue, &lap_sort<SI,TC,TV>, blocks_for( nc ), BLOCK, 0, ( const SI * ) F.trow, F.tcol, F.tval, C.dia, nc );
                F.Pt = RectView<TC,TV>{ nc, F.trow, F.tcol, F.tval };
                // `A_c = P^t A P`: the hash table sized on the rows of the fine level ( retried larger if it was short )
                if ( ! galerkin( queue, allocator, F, C ) )
                    return false;
            }
            C.A = CsrView<TC,TV>{ nc, C.row, C.col, C.val, C.dia };
            relax( queue, C );
            set_lanes( queue, C );
        }
        if ( lev.back().A.n <= BOTTOM_DENSE )
            launch_kernel( queue, &bottom_invert<TC,TV>, 1, 256, 0, lev.back().A, bottom );
        return true;
    }

    /// PRE-SMOOTHING from zero, then the residual restricted into the next level's `bk`
    void pre_restrict( const CudaQueue &queue, Level &L, Level &C, const TV *rhs ) {
        const SI m = L.A.n;
        const int deg = L.deg;
        launch_kernel( queue, &cheb_fresh<TV>, blocks_for( m ), BLOCK, 0, m, rhs, ( const TV * ) L.rlx, ( const ChebCoef * ) L.cc, L.y );
        for ( int k = 0; k + 1 < deg; ++k ) {
            launch_kernel( queue, &cheb_step<TC,TV>, blocks_for( m ), BLOCK, 0, L.A, rhs, L.x, L.r, ( const TV * ) L.y, L.z, ( const TV * ) L.rlx,
                           ( const ChebCoef * ) L.cc, k, int( k == 0 ) );
            std::swap( L.y, L.z );
        }
        const int fresh = deg == 1;
        if ( ! L.sa )
            with_lanes( L.ln, [&]( auto c ) {
                constexpr int LN = decltype( c )::value;
                launch_kernel( queue, &cheb_last_restrict<LN,TC,TV>, blocks_for( m * LN ), BLOCK, 0, L.A, rhs, L.x, ( const TV * ) L.r, ( const TV * ) L.y, C.bk, o.shift, fresh );
            } );
        else {
            with_lanes( L.ln, [&]( auto c ) {
                constexpr int LN = decltype( c )::value;
                launch_kernel( queue, &cheb_last_residual<LN,TC,TV>, blocks_for( m * LN ), BLOCK, 0, L.A, rhs, L.x, ( const TV * ) L.r, ( const TV * ) L.y, L.rho, fresh );
            } );
            launch_kernel( queue, &restrict_sa<TC,TV>, blocks_for( C.A.n ), BLOCK, 0, L.Pt, ( const TV * ) L.rho, C.bk );
        }
    }

    /// PROLONGATION of the coarse correction `e` and POST-SMOOTHING, into `out`. The prolonged iterate `x0` is `L.x + e[ j >> sh ]`
    /// read at the neighbours ( plain aggregation ), or `L.xp = L.x + P e` made first ( smoothed ).
    void prolong_post( const CudaQueue &queue, Level &L, const TV *e, const TV *rhs, TV *out ) {
        const SI m = L.A.n;
        const int deg = L.deg;
        const TV *x0 = L.x;
        int sh = o.shift;
        TV *xo = L.xp;                                   // the iterate of the steps ( never what the start reads )
        if ( L.sa ) {
            launch_kernel( queue, &prolong_sa<TC,TV>, blocks_for( m ), BLOCK, 0, L.P, ( const TV * ) L.x, e, L.xp );
            x0 = L.xp;
            xo = L.x;
            sh = -1;
        }
        with_lanes( L.ln, [&]( auto c ) {
            constexpr int LN = decltype( c )::value;
            launch_kernel( queue, &cheb_start<LN,TC,TV>, blocks_for( m * LN ), BLOCK, 0, L.A, rhs, x0, e, sh, ( const TV * ) L.rlx, ( const ChebCoef * ) L.cc,
                           L.r, L.y, deg == 1 ? out : xo, int( deg == 1 ) );
        } );
        if ( deg == 1 )
            return;
        for ( int k = 0; k + 1 < deg; ++k ) {
            launch_kernel( queue, &cheb_step<TC,TV>, blocks_for( m ), BLOCK, 0, L.A, rhs, xo, L.r, ( const TV * ) L.y, L.z, ( const TV * ) L.rlx,
                           ( const ChebCoef * ) L.cc, k, 0 );
            std::swap( L.y, L.z );
        }
        launch_kernel( queue, &sum_into<TV>, blocks_for( m ), BLOCK, 0, m, ( const TV * ) xo, ( const TV * ) L.y, out );
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
        pre_restrict( queue, L, C, rhs );
        if ( l + 1 >= o.kfrom && l + 1 < o.kfrom + o.kcycle && l + 2 < int( lev.size() ) ) {
            // two flexible-CG steps on the coarse system, preconditioned by the cycle below
            cycle( queue, l + 1, C.bk, C.v1 );
            reduce_fin( queue, nc, KStep1<TC,TV>{ C.A, C.v1, C.bk, C.t }, C.s2, FinK1{ C.kc } );
            launch_kernel( queue, &kcycle_rc<TV>, blocks_for( nc ), BLOCK, 0, nc, ( const TV * ) C.bk, ( const TV * ) C.t, C.rc, ( const KCoef * ) C.kc );
            cycle( queue, l + 1, C.rc, C.v2 );
            reduce_fin( queue, nc, KStep2<TC,TV>{ C.A, C.v2, C.t, C.rc }, C.s3, FinK2{ C.kc, C.s2.out } );
            launch_kernel( queue, &kcycle_combine<TV>, blocks_for( nc ), BLOCK, 0, nc, ( const TV * ) C.v1, ( const TV * ) C.v2, C.e, ( const KCoef * ) C.kc );
        } else
            cycle( queue, l + 1, C.bk, C.e );
        prolong_post( queue, L, C.e, rhs, x );
    }

    /// `z = M r`: the cycle from the fine level ( on `rf`, `r` in `TV`, which the update of the CG wrote; into `zf`, which the
    /// dot products of the CG copy into `z` )
    void precondition( const CudaQueue &queue ) {
        if constexpr ( SAME )
            cycle( queue, 0, r, z );
        else
            cycle( queue, 0, rf, zf );
    }

    /// the mean out of `v` ( the zero-mean gauge )
    void center( const CudaQueue &queue, double *v ) {
        reduce_fin( queue, n, SumOf{ v }, w1 );
        launch_kernel( queue, &subtract_scalar, blocks_for( n ), BLOCK, 0, v, reinterpret_cast<const double *>( w1.out ), 1.0 / double( n ), n );
    }

    /// `L x = b` ( `b` of zero sum; `x` comes out centred ). `false`: the solver could not ( the relative residual stayed
    /// above 1 ), or the pool said no.
    bool solve( const CudaQueue &queue, auto &allocator, const CsrView<TC> &A, const double *b, double *x ) {
        dump_system( queue, A, b );
        if ( o.method == 0 )
            return solve_cg( queue, A, b, x );
        return solve_mg( queue, allocator, A, b, x );
    }

    bool solve_cg( const CudaQueue &queue, const CsrView<TC> &A, const double *b, double *x ) {
        const double t0 = wall_now();
        reduce_fin( queue, n, CgInit{ b, A.dia, r, p, x }, w2, FinStart{ sc } );
        CgScalars h;
        read_back( queue, &h, sc, 1 );
        const double bb = h.rr;
        if ( ! ( bb > 0 ) ) { st.t_res += wall_now() - t0; return true; }
        const double target = o.tol * o.tol * bb;
        int it = 0;
        while ( it < o.maxit ) {
            reduce_fin( queue, n, SpmvDot<TC>{ A, p, q }, w1, FinAlpha{ sc } );
            reduce_fin( queue, n, CgUpdate{ p, q, A.dia, x, r, sc }, w2, FinBetaCg{ sc } );
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
            launch_kernel( queue, &convert_csr_values<double,TV>, blocks_for( nnz_cap ), BLOCK, 0, nnz_cap, ( const SI * ) A.row, n, A.val, val0 );
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
        reduce_fin( queue, n, DotOf{ b, b }, w1 );
        Sum1 hb;
        read_back( queue, &hb, w1.out, 1 );
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
                reduce_fin( queue, n, SpmvDot<TC>{ A, U + SI( j ) * n, AU + SI( j ) * n }, w1 );
            std::vector<double> G( k * k ), f( k );
            for ( int j = 0; j < k; ++j ) {
                Sum1 v;
                reduce_fin( queue, n, DotOf{ U + SI( j ) * n, b }, w1 );
                read_back( queue, &v, w1.out, 1 );
                f[ j ] = v.s;
                for ( int l = 0; l <= j; ++l ) {
                    reduce_fin( queue, n, DotOf{ U + SI( j ) * n, AU + SI( l ) * n }, w1 );
                    read_back( queue, &v, w1.out, 1 );
                    G[ j * k + l ] = G[ l * k + j ] = v.s;
                }
            }
            if ( solve_dense( G, f, k ) ) {
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

        // THE FLEXIBLE CG. An iteration: `z = M r`, `beta`, `p = z + beta p`, `q = A p`, `alpha`, `x += alpha p`, `r -= alpha q`;
        // every scalar on the card, the whole iteration ONE graph ( `Graph` ), the host reads back `r.r` only. ( The loop as a
        // while node of a graph, `cudaGraphSetConditional` from the last finalizer, was measured SLOWER: 0.0767 s against
        // 0.0703 at 1e5. )
        const double target = o.tol * o.tol * bb;
        launch_kernel( queue, &fcg_init, 1, 1, 0, sc, bb );
        launch_kernel( queue, &fill_value, blocks_for( n ), BLOCK, 0, p, 0.0, n );
        launch_kernel( queue, &fill_value, blocks_for( n ), BLOCK, 0, q, 0.0, n );
        if constexpr ( ! SAME )                          // ( the first input of the cycle; then the update writes it )
            launch_kernel( queue, &convert_values<double,TV>, blocks_for( n ), BLOCK, 0, n, ( const double * ) r, rf );
        CgScalars h;
        while ( true ) {
            g_iter.run( queue, x, generation, [&] {
                precondition( queue );
                reduce_fin( queue, n, FcgDots<TV>{ SAME ? ( const TV * ) z : ( const TV * ) zf, z, q, r }, w2, FinBetaFcg{ sc } );
                launch_kernel( queue, &fcg_direction, blocks_for( n ), BLOCK, 0, n, ( const double * ) z, p, ( const CgScalars * ) sc );
                reduce_fin( queue, n, SpmvDot<TC>{ A, p, q }, w1, FinAlpha{ sc } );
                reduce_fin( queue, n, FcgUpdate<TV>{ p, q, x, r, SAME ? nullptr : rf, sc }, w1, FinCheck{ sc } );
            } );
            read_back( queue, &h, sc, 1 );
            if ( h.rr <= target || h.stop || h.it >= o.maxit )
                break;
        }
        const int it = h.it;
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

    /// the small dense system `G y = f` ( Cholesky with a ridge, as `Multigrid.h::solve_dense` ); `false`: declined
    static bool solve_dense( std::vector<double> &G, std::vector<double> &f, int k ) {
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
