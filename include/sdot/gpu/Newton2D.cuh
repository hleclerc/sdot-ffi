#pragma once

// =====================================================================================
// THE TRANSPORT SOLVED ON THE CARD ( 2D, a box, a constant density ): what `sdotplan/Solve.h` + `Newton.h` do on the
// CPU, for the `SdotPlanNd` of a CUDA driver, in ONE ffi call -- so it runs under `jax.jit` like in eager.
//
// The handler is HOST code ( loom's `FfiCode.inline` ): it drives the Newton loop and launches everything on the call's
// stream. The vectors live on the card, IN TREE RANKS ( the order of `sorted_positions`, of the facets, of the
// laplacian ): the user's order comes back only in the outputs. The host reads back SCALARS, and only where it decides
// something: one small report per diagram ( the reductions of the measures: residuals, merit, smallest mass, cells
// under the floor, and the counts of the passes ), the residual of the linear solver's iterations, the step of the
// polynomial passes. Everything else stays on the card.
//
// = One diagram ( `diagram` )
//
//   1. THE WEIGHT MAJORANTS of the tree from the trial weights ( `Majorant2D.cuh`: a few launches over the seeds and the
//      levels, instead of the host's one work item per node ), straight into the card's node records;
//   2. THE CELLS ( `Cell2D.cuh::Card`, `MEASURES | FACETS | EDGES` ): the measures, the COO of the upper facets ( the next
//      Hessian ), the cuts of each cell in polygon order ( the next step's polynomials );
//   3. THE REPORT: deterministic reductions ( `Reduce.cuh` ) and one read back.
//
// Two SLOTS of everything a diagram writes ( measures, COO, counters, edges ): the accepted one and the trial one,
// swapped when a step is accepted -- what `a / fa` and `a2 / fa2` are on the CPU.
//
// = One iteration ( as `Newton.h::solves`, the same options, the same tests )
//
//   the switch of the residual ( log then lin, `switch_residual` ), the right-hand side `b` and its projection on the range,
//   the floor `eps` at the first iteration, the stopping test; then the CSR of the laplacian from the accepted COO
//   ( `Laplacian2D.cuh` ), `L d = b` ( `Linear2D.cuh`: the card's multigrid or CG, or a CPU solver of `Linear.cpp` ), the
//   gauge `d[ seed 0 ] = 0`, and the step:
//
//   TRIALS  the CPU's: `t = min( 1, mult_ok t_last )`, halved while the step is refused ( KMT: the floor `eps`, a strict
//           decrease of the merit by `1 - t / 2` ).
//   LIMITS  the old GPU campaign's exact step ( `gpu_des_familles/src/gpu/Alpha2D.cuh`, doc/06 l.808-835 ) with the CPU's
//           correction rounds: the AREA OF A CELL ALONG `w + t d` IS A POLYNOMIAL OF DEGREE 2 as long as its edges do not
//           change ( a plane's offset is affine in `t`, its normal fixed, so is a vertex ), and its edges are those of the
//           accepted diagram ( `EDGES` ): `alpha* = min_i` of the first root of `mass_i( t ) = eps`, without a walk. The
//           trial is `t = 1` if `alpha* >= 1`, `factor alpha*` otherwise; if the trial diagram still has cells under the
//           floor ( an edge appeared: the polynomial was optimistic ), their polynomials IN THE TRIAL DIAGRAM give where
//           they crossed it going back, `t = factor min`, at most eight rounds ( `Bounds.h`'s "predict, check, correct",
//           the check being the trial diagram ); then the same damping as TRIALS. The CPU's `beta` ( a first trial grown by
//           `mult_lim` ) has no use here: the polynomial pass costs a fraction of a diagram, so it is done every time.
//
// = What it does not do ( yet )
//
//   * a density that is not a constant ( `Image`, gaussians ): the cells only integrate a constant; `SdotPlanNd` refuses;
//   * the width continuation ( it needs a convolved density ); `auto` proceeds without it;
//   * 3D, other domains than a box, the neighbour memory.
//
// = Failures
//
// Per-cell status ( `Cell2D.cuh::Status`, the `status` output in user order ) and loom's error buffer: a cell past
// `max_vertices` is a `KernelFailure`; a capacity that was too small ( the fourth pass, the COO of the facets ) is written
// into its ShapeVar, which records the overflow: loom runs the call again with more ( eagerly ) or raises ( traced ), and
// the solve stops at once with `status = capacity`.
// =====================================================================================

#include "Majorant2D.cuh"
#include "Linear2D.cuh"
#include "../sdotplan/Report.h"
#include <cmath>
#include <cstdio>
#include <vector>

namespace sdot::gpu2d {

namespace sp = sdot::sdotplan;

/// the options: ONE real tensor of the call ( one read back ), in the order of `SdotPlanNd._CARD_OPTIONS`
enum Opt : int {
    O_TOL_ABS = 0, O_TOL_REL, O_T_MIN, O_MULT_OK, O_FACTOR, O_MAXIT, O_MAX_BACKTRACKS, O_STEP, O_RESIDUAL, O_POWER, O_SWITCH,
    O_LIN, O_HOST_METHOD, O_LIN_TOL, O_AMG_VARIANT, O_MG_SHIFT, O_MG_RECYCLE, O_MG_REBUILD, O_MG_STOP, O_MG_NU, O_MG_KCYCLE,
    O_LIN_MAXIT, O_TRACE, O_MG_FLOAT,
    NB_OPTS
};
enum StepKind : int { STEP_TRIALS = 0, STEP_LIMITS = 1 };
enum ResidualKind : int { RES_LIN = 0, RES_LOG = 1, RES_POWER = 2 };
enum LinKind : int { LIN_CG = 0, LIN_MG = 1, LIN_HOST = 2 };

// ---- the residual ( `Newton.h::g_of / gp_of` ) -----------------------------------------------------------------------

__host__ __device__ inline double g_of( double x, int r, double p ) {
    x = fmax( x, 1e-8 );
    if ( r == RES_POWER ) return p == 0 ? log( x ) : ( pow( x, p ) - 1 ) / p;
    return r == RES_LOG ? log( x ) : x - 1;
}
__host__ __device__ inline double gp_of( double x, int r, double p ) {
    x = fmax( x, 1e-8 );
    if ( r == RES_POWER ) return pow( x, p - 1 );
    return r == RES_LOG ? 1 / x : 1.0;
}

// ---- the report of a diagram ---------------------------------------------------------------------------------------

struct DiagRed {
    double sum_a, sum_d2, sum_g, min_a, max_abs, max_rel;
    unsigned long long nb_empty, nb_below;
    __host__ __device__ static DiagRed identity() { return { 0.0, 0.0, 0.0, 1e300, 0.0, 0.0, 0ull, 0ull }; }
    __device__ void combine( const DiagRed &o ) {
        sum_a += o.sum_a; sum_d2 += o.sum_d2; sum_g += o.sum_g;
        min_a = fmin( min_a, o.min_a ); max_abs = fmax( max_abs, o.max_abs ); max_rel = fmax( max_rel, o.max_rel );
        nb_empty += o.nb_empty; nb_below += o.nb_below;
    }
};

struct DiagFn {
    const double *a, *nu;
    double        eps, p;
    int           res;
    __device__ void operator()( SI i, DiagRed &r ) const {
        const double ai = a[ i ], ni = nu[ i ], dd = ai - ni;
        r.sum_a += ai;
        r.sum_d2 += dd * dd;
        r.sum_g += g_of( ai / ni, res, p );
        r.min_a = fmin( r.min_a, ai );
        r.max_abs = fmax( r.max_abs, fabs( dd ) );
        r.max_rel = fmax( r.max_rel, fabs( dd ) / ni );
        r.nb_empty += ! ( ai > 0 );
        r.nb_below += ai < eps;
    }
};

/// `sum ( g - mean g )^2`, the mean read on the card ( the merit of the log / power residuals, in two passes: one pass of
/// `sum g^2 - n m^2` would cancel )
struct CentredG2 {
    const double  *a, *nu;
    const DiagRed *dr;
    double         inv_n, p;
    int            res;
    __device__ void operator()( SI i, Sum1 &acc ) const {
        const double e = g_of( a[ i ] / nu[ i ], res, p ) - dr->sum_g * inv_n;
        acc.s += e * e;
    }
};

/// what the host reads back after a diagram ( one copy )
struct Report {
    DiagRed d;
    double  g2c;
    unsigned long long nb_facets, spill_need, nb_failed, pad;
};

__global__ void gather_report( Report *rep, const DiagRed *d, const Sum1 *g2, const Counters *c ) {
    rep->d = *d;
    rep->g2c = g2->s;
    rep->nb_facets  = c ? c->nb_facets : 0;
    rep->spill_need = c ? c->spill_need : 0;
    rep->nb_failed  = c ? c->nb_failed : 0;
    rep->pad = 0;
}

/// the merit of the damping ( `Newton.h::merit` ) from a report
inline double merit_of( const Report &r, int res ) {
    return std::sqrt( std::max( res == RES_LIN ? r.d.sum_d2 : r.g2c, 0.0 ) );
}

// ---- the right-hand side ( `Newton.h::rhs` ) -------------------------------------------------------------------------

struct RhsSums {
    const double *a, *nu;
    double        p;
    int           res;
    __device__ void operator()( SI i, Sum2 &acc ) const {
        const double x = a[ i ] / nu[ i ], u = nu[ i ] / gp_of( x, res, p );
        acc.s0 += u;
        acc.s1 += u * g_of( x, res, p );
    }
};

__global__ void __launch_bounds__( BLOCK ) rhs_kernel( SI n, const double *a, const double *nu, double p, int res, const Sum2 *s, double *b ) {
    const SI i = SI( blockIdx.x ) * BLOCK + threadIdx.x;
    if ( i >= n )
        return;
    if ( res == RES_LIN ) {
        b[ i ] = nu[ i ] - a[ i ];
        return;
    }
    const double c = s->s1 / s->s0, x = a[ i ] / nu[ i ];
    b[ i ] = nu[ i ] / gp_of( x, res, p ) * ( c - g_of( x, res, p ) );
}

// ---- small kernels ---------------------------------------------------------------------------------------------------

/// `dst[ k ] = src( ids( k ) )`: a user-order input in ranks
template<class TF,class TI>
__global__ void __launch_bounds__( BLOCK ) gather_ranks( SI n, Strided<TF,1> src, Strided<TI,1> ids, double *dst ) {
    const SI k = SI( blockIdx.x ) * BLOCK + threadIdx.x;
    if ( k < n ) dst[ k ] = double( src( SI( ids( k ) ) ) );
}

/// `dst( ids( k ) ) = src[ k ]`: a rank-order vector to a user-order output
template<class TF,class TI>
__global__ void __launch_bounds__( BLOCK ) scatter_ranks( SI n, const double *src, Strided<TI,1> ids, StridedOut<TF,1> dst ) {
    const SI k = SI( blockIdx.x ) * BLOCK + threadIdx.x;
    if ( k < n ) dst( SI( ids( k ) ) ) = TF( src[ k ] );
}

template<class TI>
__global__ void __launch_bounds__( BLOCK ) scatter_status( SI n, const int *src, Strided<TI,1> ids, int *dst ) {
    const SI k = SI( blockIdx.x ) * BLOCK + threadIdx.x;
    if ( k < n ) dst[ SI( ids( k ) ) ] = src[ k ];
}

/// a row of the weights history: `dst( row, ids( k ) ) = w[ k ]`
template<class TF,class TI>
__global__ void __launch_bounds__( BLOCK ) scatter_row( SI n, SI row, const double *w, Strided<TI,1> ids, StridedOut<TF,2> dst ) {
    const SI k = SI( blockIdx.x ) * BLOCK + threadIdx.x;
    if ( k < n ) dst( row, SI( ids( k ) ) ) = TF( w[ k ] );
}

template<class TF>
__global__ void __launch_bounds__( BLOCK ) write_strided( SI n, const double *src, StridedOut<TF,1> dst ) {
    const SI k = SI( blockIdx.x ) * BLOCK + threadIdx.x;
    if ( k < n ) dst( k ) = TF( src[ k ] );
}

template<class TF>
__global__ void __launch_bounds__( BLOCK ) write_rows( SI nr, SI nc, const double *src, StridedOut<TF,2> dst ) {
    const SI k = SI( blockIdx.x ) * BLOCK + threadIdx.x;
    if ( k < nr * nc ) dst( k / nc, k % nc ) = TF( src[ k ] );
}

template<class TF>
__global__ void __launch_bounds__( BLOCK ) read_strided( SI n, Strided<TF,1> src, double *dst ) {
    const SI k = SI( blockIdx.x ) * BLOCK + threadIdx.x;
    if ( k < n ) dst[ k ] = double( src( k ) );
}

/// the rank of user seed 0 ( the gauge )
template<class TI>
__global__ void __launch_bounds__( BLOCK ) find_rank0( SI n, Strided<TI,1> ids, SI *r0 ) {
    const SI k = SI( blockIdx.x ) * BLOCK + threadIdx.x;
    if ( k < n && SI( ids( k ) ) == 0 ) *r0 = k;
}

/// `w2 = w + t d`, `w2[ r0 ] = 0` ( the gauge, imposed and not hoped for )
__global__ void __launch_bounds__( BLOCK ) trial_weights( SI n, const double *w, const double *d, double t, SI r0, double *w2 ) {
    const SI i = SI( blockIdx.x ) * BLOCK + threadIdx.x;
    if ( i < n ) w2[ i ] = i == r0 ? 0.0 : w[ i ] + t * d[ i ];
}

__global__ void pick_value( const double *x, SI i, double *out ) { *out = x[ i ]; }

__global__ void __launch_bounds__( BLOCK ) scale_values( SI n, double *x, double f ) {
    const SI i = SI( blockIdx.x ) * BLOCK + threadIdx.x;
    if ( i < n ) x[ i ] *= f;
}

/// THE SIMILARITY START ( `Solve.h::similarity` ): the Voronoi of the cloud contracted and translated into the box, as weights
template<class TF>
__global__ void __launch_bounds__( BLOCK ) similarity_weights( SI n, Strided<TF,2> pos, double a, double b0, double b1, double *w ) {
    const SI k = SI( blockIdx.x ) * BLOCK + threadIdx.x;
    if ( k >= n )
        return;
    const double x = double( pos( k, 0 ) ), y = double( pos( k, 1 ) );
    const double qx = a * x + b0, qy = a * y + b1;
    w[ k ] = x * x + y * y - ( qx * qx + qy * qy ) / a;
}

template<class TF>
struct ExtentOf {
    Strided<TF,2> pos;
    __device__ void operator()( SI k, Extent2 &acc ) const {
        for ( int d = 0; d < 2; ++d ) {
            const double v = double( pos( k, d ) );
            acc.lo[ d ] = fmin( acc.lo[ d ], v );
            acc.hi[ d ] = fmax( acc.hi[ d ], v );
        }
    }
};

struct MaxAbs {
    const double *x;
    __device__ void operator()( SI i, Max1 &acc ) const { acc.m = fmax( acc.m, fabs( x[ i ] ) ); }
};

struct MinOf {
    const double *x;
    __device__ void operator()( SI i, Min1 &acc ) const { acc.m = fmin( acc.m, x[ i ] ); }
};

template<class SV>
__global__ void set_shape_var( SV sv, SI v ) { sv.set( v ); }

// ---- THE AREA POLYNOMIAL OF A CELL ALONG THE DIRECTION ( the step ) --------------------------------------------------------

/// the line of cut `c` of the cell of rank `k`, in its seed's frame, at weights `w + t0 d`: `n . v = off + s delta` at
/// `w + ( t0 + s ) d` ( a bisector's offset is affine in the weights; a side of the box does not move )
template<class TF,class TR>
__device__ __forceinline__ void line_along( const Strided<TF,2> &pos, const double *w, const double *d, double t0, SI k, TR c,
                                            double px, double py, const double ( &box )[ 4 ], double &nx, double &ny, double &off, double &delta ) {
    if ( c >= 0 ) {
        nx = double( pos( SI( c ), 0 ) ) - px;
        ny = double( pos( SI( c ), 1 ) ) - py;
        off = 0.5 * ( nx * nx + ny * ny ) + 0.5 * ( ( w[ k ] + t0 * d[ k ] ) - ( w[ c ] + t0 * d[ c ] ) );
        delta = 0.5 * ( d[ k ] - d[ c ] );
        return;
    }
    const int f = int( -1 - c );                         // 0 bottom, 1 right, 2 top, 3 left
    nx = ( f & 1 ) ? 1.0 : 0.0;
    ny = ( f & 1 ) ? 0.0 : 1.0;
    off = f == 0 ? box[ 1 ] : f == 1 ? box[ 2 ] : f == 2 ? box[ 3 ] : box[ 0 ];
    delta = 0;
}

/// `area( t0 + s ) = a0 + a1 s + a2 s^2` for the cell of rank `k` whose cuts are `edges[ q * n + k ]`, `q < nb`, while its
/// edges stay the same. The vertex `q` is the crossing of the lines `q - 1` and `q`: affine in `s` ( same matrix, the
/// right-hand side affine ). `false`: two lines nearly parallel -- this cell has no say ( the trial diagram decides ).
template<class TF,class TR>
__device__ bool area_polynomial( const Strided<TF,2> &pos, const Strided<TF,1> &box_min, const Strided<TF,1> &box_max,
                                 const double *w, const double *d, double t0, const TR *edges, SI n, SI k, int nb,
                                 double &a0, double &a1, double &a2 ) {
    const double px = double( pos( k, 0 ) ), py = double( pos( k, 1 ) );
    const double box[ 4 ] = { double( box_min( 0 ) ) - px, double( box_min( 1 ) ) - py, double( box_max( 0 ) ) - px, double( box_max( 1 ) ) - py };
    double ax, ay, ao, ad;
    line_along( pos, w, d, t0, k, edges[ SI( nb - 1 ) * n + k ], px, py, box, ax, ay, ao, ad );
    double c0 = 0, c1 = 0, c2 = 0;
    double fx = 0, fy = 0, fvx = 0, fvy = 0;              // the vertex 0 and its velocity, to close the loop
    double qx = 0, qy = 0, qvx = 0, qvy = 0;              // the previous vertex
    for ( int i = 0; i < nb; ++i ) {
        double bx, by, bo, bd;
        line_along( pos, w, d, t0, k, edges[ SI( i ) * n + k ], px, py, box, bx, by, bo, bd );
        const double det = ax * by - ay * bx;
        if ( ! ( det * det > DET_MIN * DET_MIN * ( ax * ax + ay * ay ) * ( bx * bx + by * by ) ) )
            return false;
        const double inv = 1 / det;
        const double vx = ( ao * by - bo * ay ) * inv, vy = ( ax * bo - bx * ao ) * inv;
        const double ux = ( ad * by - bd * ay ) * inv, uy = ( ax * bd - bx * ad ) * inv;
        if ( i == 0 ) { fx = vx; fy = vy; fvx = ux; fvy = uy; }
        else {
            c0 += qx * vy - vx * qy;
            c1 += qx * uy + qvx * vy - vx * qvy - ux * qy;
            c2 += qvx * uy - ux * qvy;
        }
        qx = vx; qy = vy; qvx = ux; qvy = uy;
        ax = bx; ay = by; ao = bo; ad = bd;
    }
    c0 += qx * fy - fx * qy;
    c1 += qx * fvy + qvx * fy - fx * qvy - fvx * qy;
    c2 += qvx * fvy - fvx * qvy;
    const double sg = c0 < 0 ? -0.5 : 0.5;
    a0 = sg * c0; a1 = sg * c1; a2 = sg * c2;
    return true;
}

/// the real roots of `a2 s^2 + a1 s + c = 0`, sorted ( `Bounds.h::CellPolynomial::roots` )
__device__ __forceinline__ int quadratic_roots( double a2, double a1, double c, double &r1, double &r2 ) {
    if ( fabs( a2 ) <= 1e-300 ) {
        if ( a1 == 0 ) return 0;
        r1 = r2 = -c / a1;
        return 1;
    }
    const double disc = a1 * a1 - 4 * a2 * c;
    if ( disc < 0 ) return 0;
    const double s = sqrt( disc );
    const double q = -0.5 * ( a1 + ( a1 >= 0 ? s : -s ) );
    double x1 = q / a2, x2 = q != 0 ? c / q : x1;
    if ( x1 > x2 ) { const double t = x1; x1 = x2; x2 = t; }
    r1 = x1; r2 = x2;
    return 2;
}

/// FORWARD, every cell of the accepted diagram: the first `t > 0` where its mass reaches `eps` ( `1e300`: never, or no say )
template<class TF,class TR>
struct AlphaForward {
    Strided<TF,2> pos;
    Strided<TF,1> box_min, box_max;
    const double *w, *d;
    const TR     *edges;
    const int    *nb_edges;
    SI            n;
    double        rho, eps;
    __device__ void operator()( SI k, Min1 &acc ) const {
        const int nb = nb_edges[ k ];
        if ( nb < 3 )
            return;
        double a0, a1, a2;
        if ( ! area_polynomial( pos, box_min, box_max, w, d, 0.0, edges, n, k, nb, a0, a1, a2 ) )
            return;
        double r1, r2;
        const int nr = quadratic_roots( rho * a2, rho * a1, fmax( rho * a0 - eps, 0.0 ), r1, r2 );
        double t = 1e300;
        if ( nr >= 1 && r1 > 0 ) t = r1;
        else if ( nr >= 2 && r2 > 0 ) t = r2;
        acc.m = fmin( acc.m, t );
    }
};

/// BACKWARD, the cells of the trial diagram ( at `t_trial` ) under the floor: where they crossed it, going back
/// ( `Bounds.h`'s "bad" branch: the largest negative root ); half the trial when there is no polynomial
template<class TF,class TR>
struct AlphaBackward {
    Strided<TF,2> pos;
    Strided<TF,1> box_min, box_max;
    const double *w, *d, *a;
    const TR     *edges;
    const int    *nb_edges;
    SI            n;
    double        rho, eps, t_trial;
    __device__ void operator()( SI k, MinCount &acc ) const {
        if ( ! ( a[ k ] < eps ) )
            return;
        acc.c += 1;
        double target = 0.5 * t_trial;
        const int nb = nb_edges[ k ];
        double a0, a1, a2;
        if ( nb >= 3 && area_polynomial( pos, box_min, box_max, w, d, t_trial, edges, n, k, nb, a0, a1, a2 ) ) {
            double r1, r2, beta = -1e300;
            const int nr = quadratic_roots( rho * a2, rho * a1, rho * a0 - eps, r1, r2 );
            if ( nr >= 1 && r1 < 0 ) beta = r1;
            if ( nr >= 2 && r2 < 0 ) beta = r2;
            const double tg = t_trial + beta;
            if ( tg > 0 && tg < t_trial )
                target = tg;
        }
        acc.m = fmin( acc.m, target );
    }
};

// ---- the timers ---------------------------------------------------------------------------------------------------------

/// the card's time of a stage: two events around it, read once the stream has passed them ( after a read back )
struct StageTimer {
    cudaEvent_t a = nullptr, b = nullptr;
    bool        pending = false;
    double      total = 0;
    void init() { cuda_check( cudaEventCreate( &a ), "event" ); cuda_check( cudaEventCreate( &b ), "event" ); }
    void start( const CudaQueue &q ) { cuda_check( cudaEventRecord( a, q.stream ), "event record" ); }
    void stop( const CudaQueue &q ) { cuda_check( cudaEventRecord( b, q.stream ), "event record" ); pending = true; }
    void collect() {
        if ( ! pending ) return;
        float ms = 0;
        if ( cudaEventSynchronize( b ) == cudaSuccess && cudaEventElapsedTime( &ms, a, b ) == cudaSuccess )
            total += 1e-3 * ms;
        pending = false;
    }
    void release() { if ( a ) cudaEventDestroy( a ); if ( b ) cudaEventDestroy( b ); a = b = nullptr; }
};

// ---- the solver -----------------------------------------------------------------------------------------------------------

/// what a diagram writes, twice ( the accepted one and the trial )
template<class TR>
struct Slot {
    double   *a = nullptr;                               ///< the measures, ranks
    TR       *fi = nullptr, *fj = nullptr;               ///< the COO of the upper facets
    double   *fc = nullptr;
    Counters *counters = nullptr;
    TR       *edges = nullptr;                           ///< `EDGE_CAP x n`
    int      *nb_edges = nullptr;
    int      *status = nullptr;
    Report    rep{};                                     ///< its last report ( host )
};

/// THE SOLVE ( see the header ). `pd`: the diagram ( its tree, its positions; its weights are not read ); `nu_in`, `w0_in`:
/// user order; `opts_in`: `NB_OPTS` reals. Outputs as `sdotplan::solve`, plus the diagram's `sorted_weights_out`,
/// `node_wa_out`, `node_wb_out` and `work` ( `status`, the capacities `nb_spill` and `nb_facets` ).
template<class V,class VD = V>
void solve( const CudaQueue &queue, const auto &pd, const auto &nu_in, const auto &w0_in, const auto &opts_in,
            auto &&weights, auto &&hist, auto &&stats, auto &&masses, auto &&bary, auto &&cost,
            auto &&sorted_weights_out, auto &&node_wa_out, auto &&node_wb_out, auto &&work,
            const auto &errors_, auto &allocator, const auto &rho_in, int max_vertices ) {
    using PD = std::decay_t<decltype( pd )>;
    using TF = TFOf<PD>;
    using TI = TIOf<PD>;
    using TR = typename V::TR;
    using TN = typename V::TN;
    using TK = typename V::TK;
    using CardT = Card<V,true,MEASURES | FACETS | EDGES,TF,TI>;
    using CardD = Card<VD,true,MEASURES | FACETS | EDGES,TF,TI>;
    using MomT  = Card<VD,true,MEASURES | MOMENTS,TF,TI>;
    constexpr bool MIXED = ! std::is_same_v<V,VD>;       // the float kernel first, the double one once the float stagnates
    static_assert( std::is_same_v<typename V::TR,typename VD::TR> && std::is_same_v<typename V::TN,typename VD::TN>, "one tree, one rank type" );
    static_assert( std::is_same_v<TF,double>, "the card's solver works on float64 positions" );
    static_assert( PD::has_weights, "the solver's diagram carries weights" );

    const double t_begin = wall_now();
    const SI n = SI( pd.nb_seeds() );
    const auto errors = sdot::kernel_form( queue, MutList(), errors_ );
    const auto ids = strided( pd.tree.seed_indices );
    const auto pos = strided( pd.sorted_positions );
    const auto box_min = strided( pd.box_min ), box_max = strided( pd.box_max );
    auto vec = [&]( SI m ) { return static_cast<double *>( take( allocator, SI( sizeof( double ) ) * std::max<SI>( m, 1 ) ) ); };

    // ---- the options ( one read back )
    std::vector<double> o( NB_OPTS, 0.0 );
    {
        double *tmp = vec( NB_OPTS );
        if ( ! tmp ) return;
        launch_kernel( queue, &read_strided<TF>, 1, BLOCK, 0, SI( NB_OPTS ), strided( opts_in ), tmp );
        read_back( queue, o.data(), ( const double * ) tmp, SI( NB_OPTS ) );
    }
    const double tol_abs = o[ O_TOL_ABS ], tol_rel = o[ O_TOL_REL ], t_min = o[ O_T_MIN ], mult_ok = o[ O_MULT_OK ], factor = o[ O_FACTOR ];
    const int maxit = int( o[ O_MAXIT ] ), max_backtracks = int( o[ O_MAX_BACKTRACKS ] ), step_kind = int( o[ O_STEP ] );
    const int residual = int( o[ O_RESIDUAL ] );
    const double power = o[ O_POWER ], switch_residual = o[ O_SWITCH ];
    const int lin_kind = int( o[ O_LIN ] );
    const bool trace = o[ O_TRACE ] != 0;

    // ---- the card's diagram, its two slots, the majorants
    CardT card;
    if ( ! card.prepare( queue, pd, allocator, int( std::min<SI>( work.nb_spill.max, max_vertices ) ), SPILL_WARPS, max_vertices ) )
        return;
    set_density( card.pb, rho_in );
    double rho = 1;
    if constexpr ( requires { rho_in.data().raw; } ) {
        cuda_check( cudaMemcpyAsync( &rho, rho_in.data().raw, sizeof( double ), cudaMemcpyDeviceToHost, queue.stream ), "read of the density" );
        cuda_check( cudaStreamSynchronize( queue.stream ), "sync ( density )" );
    } else
        rho = double( rho_in );
    card.pb.user_order = false;
    const SI fcap = std::max<SI>( SI( work.nb_facets.max ), 1 );
    card.pb.fcap = fcap;
    // MIXED: the double kernel's card, sharing the node records ( the majorants write them for both: they are float records
    // whatever the kernel ); only the seeds' weights are packed per kernel
    CardD cardd;
    bool use_double = ! MIXED;
    int it_double = MIXED ? -1 : 0;
    if constexpr ( MIXED ) {
        if ( ! cardd.prepare( queue, pd, allocator, int( std::min<SI>( work.nb_spill.max, max_vertices ) ), SPILL_WARPS, max_vertices ) )
            return;
        set_density( cardd.pb, rho_in );
        cardd.pb.user_order = false;
        cardd.pb.fcap = fcap;
        cardd.pb.nodes = card.pb.nodes;
    }
    Slot<TR> slots[ 2 ];
    for ( Slot<TR> &s : slots ) {
        s.a = vec( n );
        s.fi = static_cast<TR *>( take( allocator, SI( sizeof( TR ) ) * fcap ) );
        s.fj = static_cast<TR *>( take( allocator, SI( sizeof( TR ) ) * fcap ) );
        s.fc = vec( fcap );
        s.counters = static_cast<Counters *>( take( allocator, SI( sizeof( Counters ) ) ) );
        s.edges = static_cast<TR *>( take( allocator, SI( sizeof( TR ) ) * EDGE_CAP * n ) );
        s.nb_edges = static_cast<int *>( take( allocator, SI( sizeof( int ) ) * n ) );
        s.status = static_cast<int *>( take( allocator, SI( sizeof( int ) ) * n ) );
        if ( ! s.a || ! s.fi || ! s.fj || ! s.fc || ! s.counters || ! s.edges || ! s.nb_edges || ! s.status )
            return;
    }
    Slot<TR> *cur = &slots[ 0 ], *tri = &slots[ 1 ];
    Majorants<TR,TN> maj;
    if ( ! maj.prepare( allocator, pd ) )
        return;

    // ---- the vectors ( ranks ), the reductions, the laplacian, the linear solver
    double *nu = vec( n ), *w = vec( n ), *w2 = vec( n ), *d = vec( n ), *b = vec( n ), *gauge = vec( 1 );
    SI *r0_dev = static_cast<SI *>( take( allocator, SI( sizeof( SI ) ) ) );
    Report *rep_dev = static_cast<Report *>( take( allocator, SI( sizeof( Report ) ) ) );
    RedSlot<DiagRed> red_diag;
    RedSlot<Sum1> red1;
    RedSlot<Sum2> red2;
    RedSlot<Max1> redm;
    RedSlot<Min1> red_min;
    RedSlot<MinCount> red_mc;
    RedSlot<Extent2> red_ext;
    if ( ! nu || ! w || ! w2 || ! d || ! b || ! gauge || ! r0_dev || ! rep_dev || ! red_diag.take_from( allocator ) || ! red1.take_from( allocator )
         || ! red2.take_from( allocator ) || ! redm.take_from( allocator ) || ! red_min.take_from( allocator ) || ! red_mc.take_from( allocator )
         || ! red_ext.take_from( allocator ) )
        return;
    const SI nnz_cap = 2 * fcap;
    SI *lrow = static_cast<SI *>( take( allocator, SI( sizeof( SI ) ) * ( n + 1 ) ) );
    TR *lcol = static_cast<TR *>( take( allocator, SI( sizeof( TR ) ) * nnz_cap ) );
    double *lval = vec( nnz_cap ), *ldia = vec( n );
    LapWork<SI> lws;
    if ( ! lrow || ! lcol || ! lval || ! ldia || ! lws.take_from( allocator, n ) )
        return;
    const CsrView<TR> L{ n, lrow, lcol, lval, ldia };
    CardLinear<TR,double> lin;                           // the card's solver, its levels in double ...
    CardLinear<TR,float>  linf;                          // ... or in float ( `mg_precision`: the outer iteration stays in double )
    const bool lin_float = o[ O_MG_FLOAT ] != 0 && lin_kind == LIN_MG;
    HostLinear<TR> host;
    {
        LinOptions lo;
        lo.method = lin_kind == LIN_CG ? 0 : 1;
        if ( o[ O_LIN_TOL ] > 0 ) lo.tol = o[ O_LIN_TOL ];
        if ( o[ O_LIN_MAXIT ] > 0 ) lo.maxit = int( o[ O_LIN_MAXIT ] );
        if ( o[ O_MG_SHIFT ] > 0 ) lo.shift = int( o[ O_MG_SHIFT ] );
        if ( o[ O_MG_RECYCLE ] >= 0 ) lo.recycle = int( o[ O_MG_RECYCLE ] );
        if ( o[ O_MG_REBUILD ] > 0 ) lo.rebuild = int( o[ O_MG_REBUILD ] );
        if ( o[ O_MG_STOP ] > 0 ) lo.stop = int( o[ O_MG_STOP ] );
        if ( o[ O_MG_NU ] > 0 ) lo.nu = int( o[ O_MG_NU ] );
        if ( o[ O_MG_KCYCLE ] >= 0 ) lo.kcycle = int( o[ O_MG_KCYCLE ] );
        lo.trace = trace;
        if ( lin_kind == LIN_HOST ) {
            sp::LinearOptions hl;
            hl.tol = o[ O_LIN_TOL ];
            hl.amg_variant = int( o[ O_AMG_VARIANT ] );
            hl.mg_pack = o[ O_MG_SHIFT ] > 0 ? 1 << int( o[ O_MG_SHIFT ] ) : 0;
            hl.mg_recycle = int( o[ O_MG_RECYCLE ] );
            hl.mg_rebuild = int( o[ O_MG_REBUILD ] );
            hl.mg_stop = int( o[ O_MG_STOP ] );
            hl.mg_nu = int( o[ O_MG_NU ] );
            host.prepare( int( o[ O_HOST_METHOD ] ), n, hl );
        } else if ( lin_float ? ! linf.prepare( allocator, n, nnz_cap, lo ) : ! lin.prepare( allocator, n, nnz_cap, lo ) )
            return;
    }

    StageTimer tm_maj, tm_diag, tm_asm, tm_lim;
    tm_maj.init(); tm_diag.init(); tm_asm.init(); tm_lim.init();
    double t_lin = 0, t_lim_host = 0;
    int nb_diag = 0;
    unsigned long long max_spill = 0, max_facets = 0;
    bool stop_all = false;                               // a capacity or a failure: the call is run again ( or raises )
    int res_cur = residual;
    double eps = 0;

    // ---- the gauge seed, the target, the start
    zero_fill( queue, r0_dev, SI( sizeof( SI ) ) );
    launch_kernel( queue, &find_rank0<TI>, blocks_for( n ), BLOCK, 0, n, ids, r0_dev );
    SI r0 = 0;
    read_back( queue, &r0, ( const SI * ) r0_dev, 1 );
    launch_kernel( queue, &gather_ranks<TF,TI>, blocks_for( n ), BLOCK, 0, n, strided( nu_in ), ids, nu );

    /// the report of slot `s` against the current target, residual and floor ( reductions, one read back )
    auto report = [&]( Slot<TR> &s ) {
        reduce( queue, n, DiagFn{ s.a, nu, eps, power, res_cur }, red_diag.partials, red_diag.out );
        reduce( queue, n, CentredG2{ s.a, nu, red_diag.out, 1.0 / double( n ), power, res_cur }, red1.partials, red1.out );
        launch_kernel( queue, &gather_report, 1, 1, 0, rep_dev, ( const DiagRed * ) red_diag.out, ( const Sum1 * ) red1.out, ( const Counters * ) s.counters );
        read_back( queue, &s.rep, ( const Report * ) rep_dev, 1 );
    };

    /// THE DIAGRAM of `wt` into slot `s`
    auto diagram = [&]( const double *wt, Slot<TR> &s ) {
        tm_maj.start( queue );
        maj.refresh( queue, pd, wt, const_cast<Node<true,TR> *>( card.pb.nodes ) );
        auto run_on = [&]( auto &c ) {
            using C = std::decay_t<decltype( c )>;
            launch_kernel( queue, &pack_weights<typename C::TK>, blocks_for( n ), BLOCK, 0, n, wt, const_cast<typename C::Wt *>( c.pb.w ) );
            tm_maj.stop( queue );
            c.pb.w64 = Strided<TF,1>{ reinterpret_cast<const char *>( wt ), { SI( sizeof( double ) ) } };
            c.pb.res = StridedOut<TF,1>{ reinterpret_cast<char *>( s.a ), { SI( sizeof( double ) ) } };
            c.pb.status = s.status;
            c.pb.fi = s.fi; c.pb.fj = s.fj; c.pb.fc = s.fc;
            c.counters = s.counters; c.pb.counters = s.counters;
            c.pb.edges = s.edges; c.pb.nb_edges = s.nb_edges;
            tm_diag.start( queue );
            c.run( queue, errors );
            tm_diag.stop( queue );
        };
        if constexpr ( MIXED ) {
            if ( use_double ) run_on( cardd );
            else              run_on( card );
        } else
            run_on( card );
        report( s );
        tm_maj.collect();
        tm_diag.collect();
        ++nb_diag;
        max_spill = std::max( max_spill, s.rep.spill_need );
        max_facets = std::max( max_facets, s.rep.nb_facets );
        if ( s.rep.nb_facets > ( unsigned long long ) fcap || s.rep.spill_need > 0 || s.rep.nb_failed > 0 )
            stop_all = true;
    };

    std::vector<double> rows;                            // the history, written to the card at the end
    const SI cap_steps = SI( hist.rows.shape( 0 ) );
    SI nb_steps = 0;
    auto after_step = [&]( int it, double t, int nb_evals ) {
        if ( nb_steps >= cap_steps ) return;
        const Report &r = cur->rep;
        double row[ sp::NB_HIST ];
        row[ sp::H_STEP ] = it;
        row[ sp::H_T ] = t;
        row[ sp::H_RESIDUAL_L2 ] = std::sqrt( r.d.sum_d2 );
        row[ sp::H_MIN_MASS ] = r.d.min_a;
        row[ sp::H_MAX_RESIDUAL ] = r.d.max_abs;
        row[ sp::H_NB_DIAG ] = nb_diag;
        row[ sp::H_NB_EVALS ] = nb_evals;
        row[ sp::H_S ] = 0;
        rows.insert( rows.end(), row, row + sp::NB_HIST );
        if constexpr ( std::decay_t<decltype( hist.weights )>::is_valid )
            launch_kernel( queue, &scatter_row<TF,TI>, blocks_for( n ), BLOCK, 0, n, nb_steps, ( const double * ) w, ids,
                           strided_out<TF,2>( hist.weights ) );
        ++nb_steps;
    };

    // the given weights ( `w0`, user order ), in ranks; given if one is not zero
    bool given = false;
    if constexpr ( requires { w0_in.data().raw; } ) {
        launch_kernel( queue, &gather_ranks<TF,TI>, blocks_for( n ), BLOCK, 0, n, strided( w0_in ), ids, w );
        reduce( queue, n, MaxAbs{ w }, redm.partials, redm.out );
        Max1 m;
        read_back( queue, &m, ( const Max1 * ) redm.out, 1 );
        given = m.m != 0;
    } else
        launch_kernel( queue, &fill_value, blocks_for( n ), BLOCK, 0, w, 0.0, n );
    int start = given ? sp::START_GIVEN : sp::START_VORONOI;
    reduce( queue, n, SumOf{ nu }, red1.partials, red1.out );
    Sum1 snu;
    read_back( queue, &snu, ( const Sum1 * ) red1.out, 1 );
    double nu_min = 0;
    {
        reduce( queue, n, MinOf{ nu }, red_min.partials, red_min.out );
        Min1 m;
        read_back( queue, &m, ( const Min1 * ) red_min.out, 1 );
        nu_min = m.m;
    }

    diagram( w, *cur );
    if ( ! stop_all && given && cur->rep.d.min_a < 1e-3 * nu_min ) {   // a warm start that empties a cell: the Voronoi, if better
        launch_kernel( queue, &fill_value, blocks_for( n ), BLOCK, 0, w2, 0.0, n );
        diagram( w2, *tri );
        if ( tri->rep.d.min_a > cur->rep.d.min_a ) { std::swap( cur, tri ); std::swap( w, w2 ); start = sp::START_VORONOI; }
    }
    if ( ! stop_all && cur->rep.d.min_a <= 0 ) {         // seeds outside the domain: the similarity
        reduce( queue, n, ExtentOf<TF>{ pos }, red_ext.partials, red_ext.out );
        Extent2 ext;
        read_back( queue, &ext, ( const Extent2 * ) red_ext.out, 1 );
        double lo[ 2 ], hi[ 2 ];
        read_back( queue, lo, reinterpret_cast<const double *>( pd.box_min.data().raw ), 2 );
        read_back( queue, hi, reinterpret_cast<const double *>( pd.box_max.data().raw ), 2 );
        double a = 1;
        for ( int k = 0; k < 2; ++k ) {
            const double span_dom = ( hi[ k ] - lo[ k ] ) * ( 1 - 2 * 0.1 ), span_pts = std::max( ext.hi[ k ] - ext.lo[ k ], 1e-300 );
            a = std::min( a, span_dom / span_pts );
        }
        const double b0 = ( lo[ 0 ] + hi[ 0 ] ) / 2 - a * ( ext.lo[ 0 ] + ext.hi[ 0 ] ) / 2;
        const double b1 = ( lo[ 1 ] + hi[ 1 ] ) / 2 - a * ( ext.lo[ 1 ] + ext.hi[ 1 ] ) / 2;
        launch_kernel( queue, &similarity_weights<TF>, blocks_for( n ), BLOCK, 0, n, pos, a, b0, b1, w2 );
        diagram( w2, *tri );
        if ( tri->rep.d.min_a > cur->rep.d.min_a ) { std::swap( cur, tri ); std::swap( w, w2 ); start = sp::START_SIMILARITY; }
    }
    const double min_start_mass = cur->rep.d.min_a;

    // ---- the target at the scale of what the domain holds, the gauge
    const double domain_mass = cur->rep.d.sum_a;
    if ( domain_mass > 0 && snu.s > 0 && domain_mass != snu.s ) {
        launch_kernel( queue, &scale_values, blocks_for( n ), BLOCK, 0, n, nu, domain_mass / snu.s );
        nu_min *= domain_mass / snu.s;
    }
    launch_kernel( queue, &pick_value, 1, 1, 0, ( const double * ) w, r0, gauge );
    launch_kernel( queue, &subtract_scalar, blocks_for( n ), BLOCK, 0, w, ( const double * ) gauge, 1.0, n );
    report( *cur );

    // ---- THE NEWTON LOOP ( `Newton.h::solves` )
    int status = sp::S_RUNNING, nb_iter = 0, nb_backtracks = 0, it_switch = -1, nb_limit_rounds = 0;
    SI nb_cell_lim = 0;
    double residual0 = 0, residual_max = cur->rep.d.max_abs, t_last = 1;
    if ( stop_all ) status = sp::S_CAPACITY;
    after_step( 0, 0, 1 );
    for ( int it = 0; it < maxit && status == sp::S_RUNNING; ++it ) {
        const double worst = cur->rep.d.max_abs, worst_rel = cur->rep.d.max_rel;
        if ( switch_residual > 0 && res_cur != RES_LIN && worst_rel <= switch_residual ) {
            res_cur = RES_LIN;
            it_switch = it;
            if ( trace ) std::printf( "      switch: residual -> lin ( max|a-nu|/nu %.3e <= %.3e )\n", worst_rel, switch_residual );
        }
        if ( it == 0 ) {
            eps = 0.5 * std::min( nu_min, cur->rep.d.min_a );
            residual0 = worst;
        }
        report( *cur );                                  // its merit in the residual in use, its cells under the floor
        const double nr = merit_of( cur->rep, res_cur );
        residual_max = worst;
        if ( worst <= tol_abs || ( tol_rel > 0 && worst_rel <= tol_rel ) ) {
            if ( trace ) std::printf( "    it %2d  |r|_2 %.3e  max|a-nu| %.3e  CONVERGED\n", it, nr, worst );
            status = sp::S_CONVERGED;
            break;
        }
        ++nb_iter;
        const int g0 = nb_diag;

        // the right-hand side, projected on the range; the laplacian of the accepted diagram
        tm_asm.start( queue );
        if ( res_cur != RES_LIN )
            reduce( queue, n, RhsSums{ cur->a, nu, power, res_cur }, red2.partials, red2.out );
        launch_kernel( queue, &rhs_kernel, blocks_for( n ), BLOCK, 0, n, ( const double * ) cur->a, ( const double * ) nu, power, res_cur,
                       ( const Sum2 * ) red2.out, b );
        reduce( queue, n, SumOf{ b }, red1.partials, red1.out );
        launch_kernel( queue, &subtract_scalar, blocks_for( n ), BLOCK, 0, b, reinterpret_cast<const double *>( red1.out ), 1.0 / double( n ), n );
        card.pb.fi = cur->fi; card.pb.fj = cur->fj; card.pb.fc = cur->fc;
        card.counters = cur->counters; card.pb.counters = cur->counters;
        assemble_laplacian_in( queue, card, lws, lrow, lcol, lval, ldia );
        tm_asm.stop( queue );

        // the direction
        const double tl0 = wall_now();
        const bool solved = lin_kind == LIN_HOST ? host.solve( queue, L, b, d )
                          : lin_float ? linf.solve( queue, allocator, L, b, d ) : lin.solve( queue, allocator, L, b, d );
        t_lin += wall_now() - tl0;
        tm_asm.collect();
        if ( ! solved ) {
            status = sp::S_LINEAR_FAILURE;
            break;
        }
        launch_kernel( queue, &pick_value, 1, 1, 0, ( const double * ) d, r0, gauge );
        launch_kernel( queue, &subtract_scalar, blocks_for( n ), BLOCK, 0, d, ( const double * ) gauge, 1.0, n );

        // ---- the step
        double t = std::min( 1.0, mult_ok * t_last );
        bool already = false;
        double alpha_lim = -1;
        int nb_evals = 0;
        if ( step_kind == STEP_LIMITS ) {
            const double th0 = wall_now();
            tm_lim.start( queue );
            reduce( queue, n, AlphaForward<TF,TR>{ pos, box_min, box_max, w, d, cur->edges, cur->nb_edges, n, rho, eps }, red_min.partials, red_min.out );
            Min1 am;
            read_back( queue, &am, ( const Min1 * ) red_min.out, 1 );
            tm_lim.stop( queue );
            tm_lim.collect();
            t_lim_host += wall_now() - th0;
            t = am.m >= 1 ? 1.0 : factor * am.m;
            double t_done = -1;
            for ( int round = 0; round < 8; ++round ) {
                launch_kernel( queue, &trial_weights, blocks_for( n ), BLOCK, 0, n, ( const double * ) w, ( const double * ) d, t, r0, w2 );
                diagram( w2, *tri );
                ++nb_evals;
                t_done = t;
                if ( stop_all || tri->rep.d.nb_below == 0 )
                    break;
                ++nb_limit_rounds;
                const double th1 = wall_now();
                tm_lim.start( queue );
                reduce( queue, n, AlphaBackward<TF,TR>{ pos, box_min, box_max, w, d, tri->a, tri->edges, tri->nb_edges, n, rho, eps, t },
                        red_mc.partials, red_mc.out );
                MinCount mc;
                read_back( queue, &mc, ( const MinCount * ) red_mc.out, 1 );
                tm_lim.stop( queue );
                tm_lim.collect();
                t_lim_host += wall_now() - th1;
                nb_cell_lim += SI( mc.c );
                const double al = std::min( t, mc.m );
                if ( trace )
                    std::printf( "      trial t %.3e : %llu cells below eps, local limit %.3e\n", t, mc.c, al );
                t = factor * al;
                if ( t < t_min ) break;
            }
            if ( t < t_min ) t = t_done / 2;
            already = t == t_done;
            alpha_lim = t;
            if ( trace ) std::printf( "      alpha* %.3e -> trial %.3e\n", am.m, t );
        }

        // ---- THE DAMPING
        bool taken = false;
        const double t_lim0 = t;
        for ( int trial = 0; trial < max_backtracks && ! stop_all; ++trial ) {
            if ( ! ( trial == 0 && already ) ) {
                launch_kernel( queue, &trial_weights, blocks_for( n ), BLOCK, 0, n, ( const double * ) w, ( const double * ) d, t, r0, w2 );
                diagram( w2, *tri );
                ++nb_evals;
                if ( stop_all ) break;
            }
            const double m2 = tri->rep.d.min_a, n2r = merit_of( tri->rep, res_cur );
            if ( m2 >= eps && std::isfinite( n2r ) && n2r <= ( 1 - t / 2 ) * nr && n2r < nr ) { taken = true; break; }
            t /= 2;
            ++nb_backtracks;
            if ( t < t_min )
                break;
        }
        if ( trace ) {
            std::printf( "    it %2d  |r|_2 %.3e  max|a-nu| %.3e  %llu empty  step %.2e  %d diag  [majorant %.3f  diag %.3f  asm %.3f  lin %.3f  lim %.3f]",
                         it, nr, worst, cur->rep.d.nb_empty, t, nb_diag - g0, tm_maj.total, tm_diag.total, tm_asm.total, t_lin, tm_lim.total );
            if ( alpha_lim >= 0 ) std::printf( "  alpha* %.2e%s", alpha_lim, t < t_lim0 ? " REFUSED" : "" );
            std::printf( "\n" );
            std::fflush( stdout );
        }
        if ( stop_all ) { status = sp::S_CAPACITY; break; }
        if constexpr ( MIXED ) {
            if ( ! taken && ! use_double ) {
                // THE FLOAT KERNEL STAGNATES ( a cut decided in float on a degenerate cloud: its merit stops decreasing ):
                // the double kernel from here on, the accepted diagram measured again with it, and the iteration done again
                use_double = true;
                it_double = it;
                if ( trace ) std::printf( "      switch: kernel float -> double ( the float step stagnates )\n" );
                diagram( w, *cur );
                if ( stop_all ) { status = sp::S_CAPACITY; break; }
                continue;
            }
        }
        if ( ! taken ) { status = sp::S_STAGNATION; break; }
        t_last = t;
        std::swap( w, w2 );
        std::swap( cur, tri );
        after_step( it + 1, t, nb_evals );
    }
    if ( status == sp::S_RUNNING ) {
        status = sp::S_MAX_ITERATIONS;
        residual_max = cur->rep.d.max_abs;
    } else if ( status == sp::S_CONVERGED || status == sp::S_STAGNATION || status == sp::S_LINEAR_FAILURE )
        residual_max = cur->rep.d.max_abs;

    // ---- what comes out: the weights and the measures ( user order ), the diagram's weights and majorants
    launch_kernel( queue, &scatter_ranks<TF,TI>, blocks_for( n ), BLOCK, 0, n, ( const double * ) w, ids, strided_out<TF,1>( weights ) );
    launch_kernel( queue, &scatter_ranks<TF,TI>, blocks_for( n ), BLOCK, 0, n, ( const double * ) cur->a, ids, strided_out<TF,1>( masses ) );
    launch_kernel( queue, &write_strided<TF>, blocks_for( n ), BLOCK, 0, n, ( const double * ) w, strided_out<TF,1>( sorted_weights_out ) );
    launch_kernel( queue, &scatter_status<TI>, blocks_for( n ), BLOCK, 0, n, ( const int * ) cur->status, ids,
                   reinterpret_cast<int *>( const_cast<void *>( ( const void * ) work.status.data().raw ) ) );
    maj.refresh( queue, pd, w, const_cast<Node<true,TR> *>( card.pb.nodes ), strided_out<TF,2>( node_wa_out ), strided_out<TF,1>( node_wb_out ) );
    CardD *cmom_p = nullptr;                             // the moments in the double kernel's form ( mixed ), or the one kernel's
    if constexpr ( MIXED ) cmom_p = &cardd; else cmom_p = &card;
    CardD &cmom = *cmom_p;
    launch_kernel( queue, &pack_weights<typename VD::TK>, blocks_for( n ), BLOCK, 0, n, ( const double * ) w, const_cast<typename CardD::Wt *>( cmom.pb.w ) );

    // ---- THE MOMENTS at the fitted weights ( barycentres, the cost ), a last walk
    if ( ! stop_all ) {
        MomT mom;
        double *mres = vec( n ), *mcost = vec( n );
        if ( mres && mcost && mom.prepare( queue, pd, allocator, int( std::min<SI>( work.nb_spill.max, max_vertices ) ), SPILL_WARPS, max_vertices ) ) {
            set_density( mom.pb, rho_in );
            mom.pb.nodes = card.pb.nodes;
            mom.pb.w = cmom.pb.w;
            mom.pb.w64 = Strided<TF,1>{ reinterpret_cast<const char *>( w ), { SI( sizeof( double ) ) } };
            mom.pb.user_order = true;
            mom.pb.res = StridedOut<TF,1>{ reinterpret_cast<char *>( mres ), { SI( sizeof( double ) ) } };
            mom.pb.status = nullptr;
            mom.pb.bary = strided_out<TF,2>( bary );
            mom.pb.cost = StridedOut<TF,1>{ reinterpret_cast<char *>( mcost ), { SI( sizeof( double ) ) } };
            mom.run( queue, errors );
            reduce( queue, n, SumOf{ mcost }, red1.partials, red1.out );
            cuda_check( cudaMemcpyAsync( const_cast<void *>( ( const void * ) cost.data().raw ), red1.out, sizeof( double ),
                                         cudaMemcpyDeviceToDevice, queue.stream ), "copy of the cost" );
            Counters mc;
            read_back( queue, &mc, ( const Counters * ) mom.counters, 1 );
            max_spill = std::max( max_spill, mc.spill_need );
        }
    }

    // ---- the capacities ( written into their ShapeVars: past them, loom runs the call again or raises )
    launch_kernel( queue, &set_shape_var<std::decay_t<decltype( sdot::kernel_form( queue, MutList(), work.nb_spill ) )>>, 1, 1, 0,
                   sdot::kernel_form( queue, MutList(), work.nb_spill ), SI( max_spill ) );
    launch_kernel( queue, &set_shape_var<std::decay_t<decltype( sdot::kernel_form( queue, MutList(), work.nb_facets ) )>>, 1, 1, 0,
                   sdot::kernel_form( queue, MutList(), work.nb_facets ), SI( max_facets ) );

    // ---- the history and the stats
    launch_kernel( queue, &set_shape_var<std::decay_t<decltype( sdot::kernel_form( queue, MutList(), hist.nb_steps ) )>>, 1, 1, 0,
                   sdot::kernel_form( queue, MutList(), hist.nb_steps ), nb_steps );
    const LinStats ls = lin_kind == LIN_HOST ? LinStats{ host.lin->st.t_hierarchy + host.lin->st.t_build, host.lin->st.t_res,
                                                         host.lin->st.nb_hierarchies, host.lin->st.nb_iter, host.lin->st.worst }
                         : lin_float ? linf.st : lin.st;
    std::vector<double> st( sp::NB_STATS, 0.0 );
    st[ sp::STATUS ] = status;
    st[ sp::RESIDUAL ] = residual_max;
    st[ sp::RESIDUAL0 ] = residual0;
    st[ sp::NB_ITER ] = nb_iter;
    st[ sp::NB_DIAG ] = nb_diag;
    st[ sp::NB_BACKTRACKS ] = nb_backtracks;
    st[ sp::T_MAJORANT ] = tm_maj.total;
    st[ sp::T_DIAG ] = tm_diag.total;
    st[ sp::T_ASM ] = tm_asm.total;
    st[ sp::T_LIN ] = t_lin;
    st[ sp::T_LIM ] = tm_lim.total;
    st[ sp::EPS ] = eps;
    st[ sp::DOMAIN_MASS ] = domain_mass;
    st[ sp::NB_OVERFLOWED ] = 0;
    st[ sp::NB_CELL_LIM ] = double( nb_cell_lim );
    st[ sp::NB_LIMIT_ROUNDS ] = nb_limit_rounds;
    st[ sp::LIN_NB_HIERARCHIES ] = ls.nb_hierarchies;
    st[ sp::LIN_NB_ITER ] = ls.nb_iter;
    st[ sp::LIN_WORST ] = ls.worst;
    st[ sp::START ] = start;
    st[ sp::NB_CONTINUATION_STEPS ] = 1;
    st[ sp::MIN_START_MASS ] = min_start_mass;
    st[ sp::IT_SWITCH ] = it_switch;
    st[ sp::IT_DOUBLE ] = it_double;
    ( void ) t_lim_host;
    double *tmp = vec( std::max<SI>( SI( rows.size() ), sp::NB_STATS ) );
    if ( tmp ) {
        if ( nb_steps > 0 ) {
            cuda_check( cudaMemcpyAsync( tmp, rows.data(), sizeof( double ) * rows.size(), cudaMemcpyHostToDevice, queue.stream ), "copy of the history" );
            launch_kernel( queue, &write_rows<TF>, blocks_for( SI( rows.size() ) ), BLOCK, 0, nb_steps, SI( sp::NB_HIST ), ( const double * ) tmp,
                           strided_out<TF,2>( hist.rows ) );
            cuda_check( cudaStreamSynchronize( queue.stream ), "sync ( history )" );
        }
        st[ sp::T_TOTAL ] = wall_now() - t_begin;
        cuda_check( cudaMemcpyAsync( tmp, st.data(), sizeof( double ) * sp::NB_STATS, cudaMemcpyHostToDevice, queue.stream ), "copy of the stats" );
        launch_kernel( queue, &write_strided<TF>, 1, BLOCK, 0, SI( sp::NB_STATS ), ( const double * ) tmp, strided_out<TF,1>( stats ) );
        cuda_check( cudaStreamSynchronize( queue.stream ), "sync ( stats )" );
    }
    tm_maj.release(); tm_diag.release(); tm_asm.release(); tm_lim.release();
}

} // namespace sdot::gpu2d
