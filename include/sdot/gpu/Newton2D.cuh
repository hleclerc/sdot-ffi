#pragma once

// =====================================================================================
// THE TRANSPORT SOLVED ON THE CARD ( 2D or 3D, a box; in 2D a constant density, an image or gaussians, in 3D a constant ):
// what `sdotplan/Solve.h` + `Newton.h` do on the CPU, for the `SdotPlanNd` of a CUDA driver, in ONE ffi call -- so it runs
// under `jax.jit` like in eager.
//
// ONE SOLVER FOR BOTH DIMENSIONS ( `solve`, the dimension read from the diagram ): what depends on it is the cells
// ( `Cell2D.cuh`'s or `Cell3D.cuh`'s `Card`, the same interface: `prepare`, `lend`, `run`; `SolveCard` picks one ), the
// majorants' records ( `Majorants`, `gpu3d::MajorantsN` ), the weights packed for the kernel ( `pack_card_weights` ), the
// box's corners. The laplacian ( `Laplacian2D.cuh` ), the linear solver ( `Linear2D.cuh`, any CSR ), the damping, the
// aggregation are the same code. In 3D: the step is `trials` ( the cells keep no edges for the volume polynomials ), the
// density a constant ( `SdotPlanNd._build_card` refuses the rest ).
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
// = The density and the width continuation ( `Density2D.cuh`, `DensityHost2D.cuh` )
//
//   A constant ( the cells' closed forms ), an `Image` on a regular grid ( Green on its rows ) or isotropic gaussians ( the
//   polar corners, `erf`s ): integrated on the cells' edges, the mass, the facets of the laplacian and the moments. The
//   WIDTH CONTINUATION is `Solve.h`'s, stage for stage: the same widths ( `s0` half the diameter of the box or given, divided by
//   `ratio` down to the distribution's scale, then 0 ), the same triggers ( `always`; `auto` when the best start leaves a
//   cell under `threshold` times the smallest target ), the target rescaled to the domain's mass of each stage, a Newton per
//   stage from the previous weights, the moments on the true density. The density of a stage is made on the card ( the
//   image blurred by the CPU's filter, the gaussians widened ).
//
//   The step `limits` with a density: the mass along `w + t d` is no polynomial, but the cell with its edges FROZEN is still
//   known at every `t`, so the forward and backward passes bisect its mass ( `AlphaForwardDens`, `AlphaBackwardDens` ).
//
// = What it does not do ( yet )
//
//   * other domains than a box, the neighbour memory; an image on a rotated or irregular grid; in 3D the step `limits`
//     ( the volume along the direction is cubic per cell while its topology holds, but the cells would have to keep their
//     faces ) and the densities.
//
// = Failures
//
// Loom's error buffer: a cell past `max_vertices` is a `KernelFailure` naming its seed ( eager and traced ), and the solve
// stops at once with `status = failure`. The cells' fourth pass has a FIXED budget ( `Cell2D.cuh::Overflow`, taken once
// and shared by every card of the solve ): it never asks for room. The one capacity left is the COO of the facets, written
// into its ShapeVar, which records the overflow: loom runs the call again with more ( eagerly ) or raises ( traced ), and
// the solve stops at once with `status = capacity`.
// =====================================================================================

#include "Majorant2D.cuh"
#include "Majorant3D.cuh"
#include "Cell3D.cuh"
#include "Linear2D.cuh"
#include "DensityHost2D.cuh"
#include "../sdotplan/Report.h"
#include "../sdotplan/Continuation.h"
#include "../sdotplan/Aggregation.h"
#include <cmath>
#include <cstdio>
#include <type_traits>
#include <vector>

namespace sdot::gpu2d {

namespace sp = sdot::sdotplan;

/// the options: ONE real tensor of the call ( one read back ), in the order of `SdotPlanNd._CARD_OPTIONS`
enum Opt : int {
    O_TOL_ABS = 0, O_TOL_REL, O_T_MIN, O_MULT_OK, O_FACTOR, O_MAXIT, O_MAX_BACKTRACKS, O_STEP, O_RESIDUAL, O_POWER, O_SWITCH,
    O_LIN, O_HOST_METHOD, O_LIN_TOL, O_AMG_VARIANT, O_MG_SHIFT, O_MG_RECYCLE, O_MG_REBUILD, O_MG_STOP, O_MG_NU, O_MG_KCYCLE,
    O_LIN_MAXIT, O_TRACE, O_MG_FLOAT, O_MG_SMOOTHED,
    O_CONTINUATION, O_CONV_THRESHOLD, O_CONV_S0, O_CONV_RATIO, O_CONV_MIN, O_CONV_POSSIBLE, O_MIN_SCALE,
    O_IMG_X0, O_IMG_Y0, O_IMG_HX, O_IMG_HY, O_IMG_NX, O_IMG_NY,
    O_AGG_MARGIN, O_AGG_GAP, O_AGG_NB_DUPS,
    NB_OPTS
};
enum ContinuationKind : int { CONT_NEVER = 0, CONT_AUTO = 1, CONT_ALWAYS = 2 };
enum StepKind : int { STEP_TRIALS = 0, STEP_LIMITS = 1 };
enum ResidualKind : int { RES_LIN = 0, RES_LOG = 1, RES_POWER = 2 };
enum LinKind : int { LIN_CG = 0, LIN_MG = 1, LIN_HOST = 2 };

// ---- the residual ( `Newton.h::g_of / gp_of` ) -----------------------------------------------------------------------

__host__ __device__ inline double g_of( double x, int r, double p ) {
    x = fmax( x, 1e-300 );
    if ( r == RES_POWER ) return p == 0 ? log( x ) : ( pow( x, p ) - 1 ) / p;
    return r == RES_LOG ? log( x ) : x - 1;
}
__host__ __device__ inline double gp_of( double x, int r, double p ) {
    x = fmax( x, 1e-300 );
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
    const double *ar  = nullptr;                         ///< the masses the residual reads ( the clusters' shares, `sdotplan/Aggregation.h` ); null: `a`
    const double *nue = nullptr;                         ///< the targets, an exact duplicate's at 0: left out of the floor ( its cell is empty )
    __device__ void operator()( SI i, DiagRed &r ) const {
        const double ai = a[ i ], ni = nu[ i ], ri = ar ? ar[ i ] : ai, dd = ri - ni;
        r.sum_a += ai;
        r.sum_d2 += dd * dd;
        r.sum_g += g_of( ri / ni, res, p );
        r.max_abs = fmax( r.max_abs, fabs( dd ) );
        r.max_rel = fmax( r.max_rel, fabs( dd ) / ni );
        if ( nue && ! ( nue[ i ] > 0 ) )
            return;
        r.min_a = fmin( r.min_a, ai );
        r.nb_empty += ! ( ai > 0 );
        r.nb_below += ai < eps;
    }
};

/// THE FULL PROBLEM'S RESIDUAL ( `Aggregation::full_residual` ): `max |a - nu_e|` and its relative form, the exact duplicates
/// ( `nu_e = 0` ) left out
struct FullResFn {
    const double *a, *nue;
    __device__ void operator()( SI i, DiagRed &r ) const {
        if ( ! ( nue[ i ] > 0 ) )
            return;
        const double dd = fabs( a[ i ] - nue[ i ] );
        r.max_abs = fmax( r.max_abs, dd );
        r.max_rel = fmax( r.max_rel, dd / nue[ i ] );
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
    unsigned long long nb_facets, nb_failed;
};

__global__ void gather_report( Report *rep, const DiagRed *d, const Sum1 *g2, const Counters *c ) {
    rep->d = *d;
    rep->g2c = g2->s;
    rep->nb_facets = c ? c->nb_facets : 0;
    rep->nb_failed = c ? c->nb_failed : 0;
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
        if ( ! ( nu[ i ] > 0 ) )                         // an exact duplicate ( `nu_e = 0` ): its weight follows its representative
            return;
        const double x = a[ i ] / nu[ i ], u = nu[ i ] / gp_of( x, res, p );
        acc.s0 += u;
        acc.s1 += u * g_of( x, res, p );
    }
};

__global__ void __launch_bounds__( BLOCK ) rhs_kernel( SI n, const double *a, const double *nu, double p, int res, const Sum2 *s, double *b ) {
    const SI i = SI( blockIdx.x ) * BLOCK + threadIdx.x;
    if ( i >= n )
        return;
    if ( ! ( nu[ i ] > 0 ) ) {                           // an exact duplicate ( `nu_e = 0` )
        b[ i ] = 0;
        return;
    }
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

/// a translation, by value ( a kernel argument )
struct Shift3 { double v[ 3 ]; };

/// THE SIMILARITY START ( `Solve.h::similarity` ): the Voronoi of the cloud contracted and translated into the box, as weights
template<int DIM,class TF>
__global__ void __launch_bounds__( BLOCK ) similarity_weights( SI n, Strided<TF,2> pos, double a, Shift3 b, double *w ) {
    const SI k = SI( blockIdx.x ) * BLOCK + threadIdx.x;
    if ( k >= n )
        return;
    // ( the expressions written out per dimension: the same roundings, hence the same start, as the CPU's `Solve.h::similarity`
    // -- which of the Voronoi and the similarity starts may hang on a cell of 1e-17 on an image with a hole )
    const double x = double( pos( k, 0 ) ), y = double( pos( k, 1 ) );
    const double qx = a * x + b.v[ 0 ], qy = a * y + b.v[ 1 ];
    if constexpr ( DIM == 2 )
        w[ k ] = x * x + y * y - ( qx * qx + qy * qy ) / a;
    else {
        const double z = double( pos( k, 2 ) ), qz = a * z + b.v[ 2 ];
        w[ k ] = x * x + y * y + z * z - ( qx * qx + qy * qy + qz * qz ) / a;
    }
}

template<int DIM,class TF>
struct ExtentOf {
    Strided<TF,2> pos;
    __device__ void operator()( SI k, ExtentN<DIM> &acc ) const {
        for ( int d = 0; d < DIM; ++d ) {
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

// ---- THE AGGREGATION of near-coincident seeds ( `sdotplan/Aggregation.h`; the host decides, these kernels apply ) --------------

/// each member of a cluster gets its share of the cluster's mass, `ae[ m ] = nu[ m ] a_r / nu_r` ( `ae` holds `a` elsewhere )
__global__ void __launch_bounds__( BLOCK ) cluster_shares( SI nc, const SI *beg, const SI *mem, const double *a, const double *nu, double *ae ) {
    const SI c = SI( blockIdx.x ) * BLOCK + threadIdx.x;
    if ( c >= nc )
        return;
    double sa = 0, sn = 0;
    for ( SI k = beg[ c ]; k < beg[ c + 1 ]; ++k ) { sa += a[ mem[ k ] ]; sn += nu[ mem[ k ] ]; }
    const double r = sn > 0 ? sa / sn : 0.0;
    for ( SI k = beg[ c ]; k < beg[ c + 1 ]; ++k ) ae[ mem[ k ] ] = nu[ mem[ k ] ] * r;
}

/// the targets with the exact duplicates' carried by their representatives: `nu_e[ rep ] += nu[ dup ]`, `nu_e[ dup ] = 0`
__global__ void __launch_bounds__( BLOCK ) duplicate_targets( SI k, const SI *dup, const SI *rep, const double *nu, double *nue ) {
    const SI q = SI( blockIdx.x ) * BLOCK + threadIdx.x;
    if ( q >= k )
        return;
    atomicAdd( nue + rep[ q ], nu[ dup[ q ] ] );
    nue[ dup[ q ] ] = 0;
}

/// `w[ dup ] = w[ rep ] - gap`: the cell of an exact duplicate stays empty
__global__ void __launch_bounds__( BLOCK ) tie_duplicates( SI k, const SI *dup, const SI *rep, const double *gap, double *w ) {
    const SI q = SI( blockIdx.x ) * BLOCK + threadIdx.x;
    if ( q < k ) w[ dup[ q ] ] = w[ rep[ q ] ] - gap[ q ];
}

/// the ranks of the user seeds `users` ( sorted, `m` ): `ranks[ q ]` for `users[ q ]`
template<class TI>
__global__ void __launch_bounds__( BLOCK ) ranks_of_users( SI n, Strided<TI,1> ids, SI m, const SI *users, SI *ranks ) {
    const SI k = SI( blockIdx.x ) * BLOCK + threadIdx.x;
    if ( k >= n )
        return;
    const SI u = SI( ids( k ) );
    SI lo = 0, hi = m;
    while ( lo < hi ) { const SI mid = ( lo + hi ) / 2; if ( users[ mid ] < u ) lo = mid + 1; else hi = mid; }
    if ( lo < m && users[ lo ] == u ) ranks[ lo ] = k;
}

/// `out[ q ] = src[ at[ q ] ]` ( a few values read back by the host )
__global__ void __launch_bounds__( BLOCK ) gather_at( SI m, const SI *at, const double *src, double *out ) {
    const SI q = SI( blockIdx.x ) * BLOCK + threadIdx.x;
    if ( q < m ) out[ q ] = src[ at[ q ] ];
}

template<class TI>
__global__ void __launch_bounds__( BLOCK ) users_at( SI m, const SI *at, Strided<TI,1> ids, SI *out ) {
    const SI q = SI( blockIdx.x ) * BLOCK + threadIdx.x;
    if ( q < m ) out[ q ] = SI( ids( at[ q ] ) );
}

struct MaxOfV {
    const double *x;
    __device__ void operator()( SI i, Max1 &acc ) const { acc.m = fmax( acc.m, x[ i ] ); }
};

/// one coefficient of the laplacian above the bound of the detection, read back by the host
struct HeavyFacet { SI i, j; double c, wi, wj, nui, nuj; };

/// THE DETECTION'S SCAN ( `Aggregation::detect` ): the entries `i < j` of the laplacian with `c_ij > c_star`, at most `cap`
template<class TR>
__global__ void __launch_bounds__( BLOCK ) heavy_facets( SI n, const SI *row, const TR *col, const double *val, const double *dia, double c_star,
                                                        const double *w, const double *nu, unsigned long long *count, HeavyFacet *out, SI cap ) {
    const SI i = SI( blockIdx.x ) * BLOCK + threadIdx.x;
    if ( i >= n || ! ( dia[ i ] > c_star ) )
        return;
    for ( SI e = row[ i ]; e < row[ i + 1 ]; ++e ) {
        const SI j = SI( col[ e ] );
        if ( j <= i || ! ( val[ e ] > c_star ) )
            continue;
        const SI q = SI( atomicAdd( count, 1ull ) );
        if ( q < cap ) out[ q ] = HeavyFacet{ i, j, val[ e ], w[ i ], w[ j ], nu[ i ], nu[ j ] };
    }
}

/// THE RE-SPLITTING'S ROWS: for each member `mem[ q ]`, its diagonal and its entries ( at most `deg` ), for the host's blocks
template<class TR>
__global__ void __launch_bounds__( BLOCK ) member_rows( SI m, const SI *mem, const SI *row, const TR *col, const double *val, const double *dia,
                                                       int deg, double *out_dia, SI *out_len, SI *out_col, double *out_val ) {
    const SI q = SI( blockIdx.x ) * BLOCK + threadIdx.x;
    if ( q >= m )
        return;
    const SI i = mem[ q ];
    out_dia[ q ] = dia[ i ];
    out_len[ q ] = row[ i + 1 ] - row[ i ];
    for ( SI e = row[ i ], u = 0; e < row[ i + 1 ] && u < deg; ++e, ++u ) {
        out_col[ q * deg + u ] = SI( col[ e ] );
        out_val[ q * deg + u ] = val[ e ];
    }
}

/// THE LINEAR SYSTEM'S ROWS OF THE DUPLICATES ( `Aggregation::link_duplicates` ): a facet `( dup, rep, c )` appended to the COO
/// after its `nb` entries ( the host checked the room ), counted in by `bump_facets`, taken out again after the assembly
template<class TR>
__global__ void __launch_bounds__( BLOCK ) append_links( SI k, const SI *dup, const SI *rep, double c, unsigned long long nb, TR *fi, TR *fj, double *fc ) {
    const SI q = SI( blockIdx.x ) * BLOCK + threadIdx.x;
    if ( q >= k )
        return;
    fi[ nb + q ] = TR( dup[ q ] );
    fj[ nb + q ] = TR( rep[ q ] );
    fc[ nb + q ] = c;
}
__global__ void bump_facets( Counters *counters, long long delta ) { counters->nb_facets += delta; }

/// THE DIRECTION WITH CLUSTERS ( see `solve` ): the coefficients between two members of cluster `c` capped at `cap` in their
/// rows ( the diagonal following ); one thread per cluster
template<class TR>
__global__ void __launch_bounds__( BLOCK ) cap_internal( SI nc, const SI *beg, const SI *mem, const SI *row, const TR *col, double *val, double *dia, double cap ) {
    const SI c = SI( blockIdx.x ) * BLOCK + threadIdx.x;
    if ( c >= nc )
        return;
    for ( SI k = beg[ c ]; k < beg[ c + 1 ]; ++k ) {
        const SI i = mem[ k ];
        bool capped = false;
        for ( SI e = row[ i ]; e < row[ i + 1 ]; ++e ) {
            if ( ! ( val[ e ] > cap ) ) continue;
            const SI j = SI( col[ e ] );
            SI lo = beg[ c ], hi = beg[ c + 1 ];
            while ( lo < hi ) { const SI mid = ( lo + hi ) / 2; if ( mem[ mid ] < j ) lo = mid + 1; else hi = mid; }
            if ( lo < beg[ c + 1 ] && mem[ lo ] == j ) { val[ e ] = cap; capped = true; }
        }
        if ( capped ) {                                  // the diagonal summed again ( a float cut can make a coefficient infinite )
            double d = 0;
            for ( SI e = row[ i ]; e < row[ i + 1 ]; ++e ) d += val[ e ];
            dia[ i ] = d > 0 ? d : 1.0;
        }
    }
}

/// ... and the members' directions tied to their cluster's, `d_i = sum nu_j d_j / sum nu_j` ( `nu`: the targets, a duplicate's 0 )
__global__ void __launch_bounds__( BLOCK ) tie_direction( SI nc, const SI *beg, const SI *mem, const double *nu, double *d ) {
    const SI c = SI( blockIdx.x ) * BLOCK + threadIdx.x;
    if ( c >= nc )
        return;
    double s = 0, sn = 0;
    for ( SI k = beg[ c ]; k < beg[ c + 1 ]; ++k ) { s += nu[ mem[ k ] ] * d[ mem[ k ] ]; sn += nu[ mem[ k ] ]; }
    const double dm = sn > 0 ? s / sn : 0.0;
    for ( SI k = beg[ c ]; k < beg[ c + 1 ]; ++k ) d[ mem[ k ] ] = dm;
}

/// `w2 = w - shift`, then `w2[ at[ q ] ] += dw[ q ]` ( the re-splitting's step; the shift keeps the gauge )
__global__ void __launch_bounds__( BLOCK ) shifted_copy( SI n, const double *w, double shift, double *w2 ) {
    const SI i = SI( blockIdx.x ) * BLOCK + threadIdx.x;
    if ( i < n ) w2[ i ] = w[ i ] - shift;
}
__global__ void __launch_bounds__( BLOCK ) add_at( SI m, const SI *at, const double *dw, double *w ) {
    const SI q = SI( blockIdx.x ) * BLOCK + threadIdx.x;
    if ( q < m ) w[ at[ q ] ] += dw[ q ];
}

/// the clusters' output, user order: every seed its own index ...
template<class TI,class TO>
__global__ void __launch_bounds__( BLOCK ) write_own_index( SI n, Strided<TI,1> ids, StridedOut<TO,1> out ) {
    const SI k = SI( blockIdx.x ) * BLOCK + threadIdx.x;
    if ( k < n ) out( SI( ids( k ) ) ) = TO( ids( k ) );
}
/// ... then the members of a cluster the smallest index of it
template<class TO>
__global__ void __launch_bounds__( BLOCK ) write_representatives( SI m, const SI *users, const SI *reps, StridedOut<TO,1> out ) {
    const SI q = SI( blockIdx.x ) * BLOCK + threadIdx.x;
    if ( q < m ) out( users[ q ] ) = TO( reps[ q ] );
}

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
    const double *nue = nullptr;                         ///< an exact duplicate ( `nu_e = 0` ): empty by construction, no say
    __device__ void operator()( SI k, MinCount &acc ) const {
        if ( ! ( a[ k ] < eps ) || ( nue && ! ( nue[ k ] > 0 ) ) )
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

// ---- THE MASS OF A CELL ALONG THE DIRECTION, for a density that is not a constant ( the step ) --------------------------------
//
// The mass along `w + t d` is not a polynomial any more, but with its edges frozen the cell is still known at every `t`: its
// vertices are the crossings of consecutive lines, each line's offset affine in `t`. So the step is the same as for a constant
// ( forward on the accepted diagram's edges, backward on the trial's ), each root found by a BISECTION on the frozen cell's
// mass ( the CPU's `Bounds.h` bisects too, on exact cells: `1e-2` relative, the admissible end ).

/// the mass of the cell of rank `k` whose cuts are `edges[ q * n + k ]`, `q < nb`, frozen, at the weights `w + t d`: `false` if
/// two consecutive lines are nearly parallel ( no say ). Counterclockwise as the cell it comes from: a cell that turns inside
/// out has a negative mass ( it is crushed ).
template<class TF,class TR,class D>
__device__ bool frozen_mass( const Strided<TF,2> &pos, const Strided<TF,1> &box_min, const Strided<TF,1> &box_max, const double *w, const double *d,
                             double t, const TR *edges, SI n, SI k, int nb, const D &dens, double &m ) {
    const double px = double( pos( k, 0 ) ), py = double( pos( k, 1 ) );
    const double box[ 4 ] = { double( box_min( 0 ) ) - px, double( box_min( 1 ) ) - py, double( box_max( 0 ) ) - px, double( box_max( 1 ) ) - py };
    double ax, ay, ao, ad;
    line_along( pos, w, d, t, k, edges[ SI( nb - 1 ) * n + k ], px, py, box, ax, ay, ao, ad );
    DensSums acc;
    DensState<D> st;
    double fx = 0, fy = 0, qx = 0, qy = 0;
    for ( int i = 0; i < nb; ++i ) {
        double bx, by, bo, bd;
        line_along( pos, w, d, t, k, edges[ SI( i ) * n + k ], px, py, box, bx, by, bo, bd );
        const double det = ax * by - ay * bx;
        if ( ! ( det * det > DET_MIN * DET_MIN * ( ax * ax + ay * ay ) * ( bx * bx + by * by ) ) )
            return false;
        const double inv = 1 / det;
        const double vx = ( ao * by - bo * ay ) * inv, vy = ( ax * bo - bx * ao ) * inv;
        if ( i == 0 ) { fx = vx; fy = vy; }
        else density_edge_mass( dens, st, px, py, qx, qy, vx, vy, acc );
        qx = vx; qy = vy;
        ax = bx; ay = by; ao = bo;
    }
    density_edge_mass( dens, st, px, py, qx, qy, fx, fy, acc );
    m = acc.m;
    return true;
}

/// the bisection on `[ lo, hi ]`, `mass( lo ) >= eps > mass( hi )`: the admissible end, to `1e-2` relative and above zero
template<class F>
__device__ double mass_bisection( double lo, double hi, double eps, const F &mass ) {
    for ( int round = 0; round < 64; ++round ) {
        if ( hi - lo <= 1e-2 * hi && lo > 0 )
            break;
        const double mid = 0.5 * ( lo + hi );
        double m;
        if ( mass( mid, m ) && m >= eps ) lo = mid;
        else                               hi = mid;
    }
    return lo;
}

/// FORWARD, every cell of the accepted diagram ( a density ): the first `t` in `( 0, 1 ]` where its frozen mass falls under `eps`
/// -- looked for at `t = 1` and at the bottom of its area's parabola ( where a cell that dips then grows back is thinnest );
/// `1e300`: none seen
template<class TF,class TR,class D>
struct AlphaForwardDens {
    Strided<TF,2> pos;
    Strided<TF,1> box_min, box_max;
    const double *w, *d;
    const TR     *edges;
    const int    *nb_edges;
    SI            n;
    D             dens;
    double        eps;
    __device__ void operator()( SI k, Min1 &acc ) const {
        const int nb = nb_edges[ k ];
        if ( nb < 3 )
            return;
        auto mass = [&]( double t, double &m ) { return frozen_mass( pos, box_min, box_max, w, d, t, edges, n, k, nb, dens, m ); };
        double m1, hi = -1;
        if ( ! mass( 1.0, m1 ) )
            return;
        if ( m1 < eps )
            hi = 1;
        else {
            double a0, a1, a2;
            if ( area_polynomial( pos, box_min, box_max, w, d, 0.0, edges, n, k, nb, a0, a1, a2 ) && a2 > 0 ) {
                const double ts = -a1 / ( 2 * a2 );
                double ms;
                if ( ts > 0 && ts < 1 && mass( ts, ms ) && ms < eps )
                    hi = ts;
            }
        }
        if ( hi < 0 )
            return;
        acc.m = fmin( acc.m, mass_bisection( 0.0, hi, eps, mass ) );
    }
};

/// BACKWARD, the cells of the trial diagram ( at `t_trial` ) under the floor ( a density ): where their frozen mass crossed `eps`
/// going back; half the trial when the frozen cell is under the floor at `t = 0` too
template<class TF,class TR,class D>
struct AlphaBackwardDens {
    Strided<TF,2> pos;
    Strided<TF,1> box_min, box_max;
    const double *w, *d, *a;
    const TR     *edges;
    const int    *nb_edges;
    SI            n;
    D             dens;
    double        eps, t_trial;
    const double *nue = nullptr;                         ///< an exact duplicate ( `nu_e = 0` ): empty by construction, no say
    __device__ void operator()( SI k, MinCount &acc ) const {
        if ( ! ( a[ k ] < eps ) || ( nue && ! ( nue[ k ] > 0 ) ) )
            return;
        acc.c += 1;
        double target = 0.5 * t_trial;
        const int nb = nb_edges[ k ];
        auto mass = [&]( double t, double &m ) { return frozen_mass( pos, box_min, box_max, w, d, t, edges, n, k, nb, dens, m ); };
        double m0;
        if ( nb >= 3 && mass( 0.0, m0 ) && m0 >= eps ) {
            const double tg = mass_bisection( 0.0, t_trial, eps, mass );
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

/// an integer input read as `SI` ( the exact duplicates )
template<class T>
__global__ void __launch_bounds__( BLOCK ) read_ints( SI n, Strided<T,1> src, SI *dst ) {
    const SI k = SI( blockIdx.x ) * BLOCK + threadIdx.x;
    if ( k < n ) dst[ k ] = SI( src( k ) );
}

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
    Report    rep{};                                     ///< its last report ( host )
};

/// THE CARDS OF THE SOLVE by dimension: `Cell2D.cuh`'s ( `OUT` as asked, the density `D` ) or `Cell3D.cuh`'s ( a constant
/// density; no `EDGES`: the 3D step is `trials` )
template<int DIM,class V,unsigned OUT,class TF,class TI,class D> struct SolveCard;
template<class V,unsigned OUT,class TF,class TI,class D> struct SolveCard<2,V,OUT,TF,TI,D> { using type = Card<V,true,OUT,TF,TI,D>; };
template<class V,unsigned OUT,class TF,class TI,class D> struct SolveCard<3,V,OUT,TF,TI,D> { using type = gpu3d::Card<V,true,OUT & ~EDGES,TF,TI>; };

/// the walk's node records of a card, writable ( the majorants write them )
template<class C>
auto *node_records( const C &c ) { return const_cast<std::remove_const_t<std::remove_pointer_t<decltype( c.pb.nodes )>> *>( c.pb.nodes ); }

/// the weights `w` ( ranks, doubles ) into the kernel's seeds of card `c` ( 2D: the weights apart; 3D: in the seeds' records )
template<class C>
void pack_card_weights( const CudaQueue &queue, C &c, SI n, const double *w ) {
    using TK = typename C::TK;
    if constexpr ( requires { c.pb.seeds; } )
        launch_kernel( queue, &gpu3d::pack_seed_weights<TK>, blocks_for( n ), BLOCK, 0, n, w, const_cast<gpu3d::Seed<TK> *>( c.pb.seeds ) );
    else
        launch_kernel( queue, &pack_weights<TK>, blocks_for( n ), BLOCK, 0, n, w, const_cast<typename C::Wt *>( c.pb.w ) );
}

/// THE SOLVE ( see the header ). `pd`: the diagram ( its tree, its positions; its weights are not read ); `nu_in`, `w0_in`:
/// user order; `dups_in`: the exact duplicates, `( dup, rep )` user pairs flattened ( the first `O_AGG_NB_DUPS` pairs );
/// `opts_in`: `NB_OPTS` reals. Outputs as `sdotplan::solve`, plus the diagram's `sorted_weights_out`,
/// `node_wa_out`, `node_wb_out` and `work` ( the capacity `nb_facets` ). `dens_in`: the density ( a 0-d tensor: a constant;
/// `image_in( ... )`, `gauss_in( ... )`: `DensityHost2D.cuh` ). `max_vertices`, `overflow_warps`: the cells' fourth pass
/// ( `Cell2D.cuh::Overflow` ).
///
/// THE AGGREGATION ( `sdotplan/Aggregation.h` ): the host keeps the clusters ( in ranks ), the card applies them -- the shares
/// in the reports' residual, the exact duplicates' targets and tied weights, the detection's scan of the laplacian, the
/// re-splitting's step. Nothing of it runs on a cloud without such seeds but two reductions per iteration ( the largest
/// weight and diagonal ).
template<class V,class VD = V>
void solve( const CudaQueue &queue, const auto &pd, const auto &nu_in, const auto &w0_in, const auto &dups_in, const auto &opts_in,
            auto &&weights, auto &&hist, auto &&stats, auto &&masses, auto &&bary, auto &&cost, auto &&clusters_out,
            auto &&sorted_weights_out, auto &&node_wa_out, auto &&node_wb_out, auto &&work,
            const auto &errors_, auto &allocator, const auto &dens_in, int max_vertices, int overflow_warps ) {
    using PD = std::decay_t<decltype( pd )>;
    using TF = TFOf<PD>;
    using TI = TIOf<PD>;
    using TR = typename V::TR;
    using TN = typename V::TN;
    using DH = typename DensityHostOf<std::decay_t<decltype( dens_in )>>::type;
    using D  = typename DH::Dev;
    constexpr bool CONST = std::is_same_v<D,DensConst>;
    constexpr int DIM = PD::ct_dim;                      // ( 3D: `Cell3D.cuh`'s cells, a constant density, the `trials` step )
    static_assert( DIM == 2 || DIM == 3, "the card's solve is 2D or 3D" );
    static_assert( DIM == 2 || CONST, "the card's 3D solve integrates a constant density" );
    using CardT = typename SolveCard<DIM,V,MEASURES | FACETS | EDGES,TF,TI,D>::type;
    using CardD = typename SolveCard<DIM,VD,MEASURES | FACETS | EDGES,TF,TI,D>::type;
    using MomT  = typename SolveCard<DIM,VD,MEASURES | MOMENTS,TF,TI,D>::type;
    constexpr bool MIXED = ! std::is_same_v<V,VD>;       // the float kernel first, the double one once the float stagnates
    static_assert( std::is_same_v<typename V::TR,typename VD::TR> && std::is_same_v<typename V::TN,typename VD::TN>, "one tree, one rank type" );
    static_assert( std::is_same_v<TF,double>, "the card's solver works on float64 positions" );
    static_assert( PD::has_weights, "the solver's diagram carries weights" );

    const double t_begin = wall_now();
    const SI n = SI( pd.nb_seeds() );
    if constexpr ( requires { allocator.why; } )         // what a refusal of the pool says ( loom's `Scratch` )
        allocator.why = "It was the card's Newton solve of " + std::to_string( n ) + " seeds ( `SdotPlanNd`, `gpu/Newton2D.cuh` ), "
                        "whose need `sdot.CardMemory.card_solve_bytes` estimates ( and checks before the call, unless "
                        "SDOT_CARD_MEMORY_CHECK=0 ). Solve fewer seeds, give XLA more of the card ( XLA_PYTHON_CLIENT_MEM_FRACTION, "
                        "0.75 by default ), free the arrays the program keeps, or solve on the CPU ( LOOM_DEVICE=cpu )";
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
    const int continuation = int( o[ O_CONTINUATION ] );
    const SI agg_nb_dups = SI( o[ O_AGG_NB_DUPS ] );

    // ---- the fourth pass's slots, ONE budget for every card of the solve ( they run one after the other ), sized for the
    // largest cell form ( the double kernel's, MIXED )
    using Ovf = std::conditional_t<DIM == 2, Overflow, gpu3d::Slots>;
    Ovf overflow = Ovf::sized( n, overflow_warps, max_vertices );
    overflow.bytes = std::max( overflow.template bytes_for<typename V::TK,TR>(), overflow.template bytes_for<typename VD::TK,TR>() );
    unsigned char *ovf_ptr = static_cast<unsigned char *>( take( allocator, overflow.bytes ) );
    if ( ! ovf_ptr )
        return;
    if constexpr ( DIM == 2 ) overflow.slots = ovf_ptr; else overflow.ptr = ovf_ptr;

    // ---- the density ( `DensityHost2D.cuh` ): a constant, or the image / gaussians as given ( `s = 0` )
    DH dh;
    if constexpr ( CONST ) {
        dh.possible = o[ O_CONV_POSSIBLE ] != 0;
        dh.scale = o[ O_MIN_SCALE ];
    } else if ( ! dh.prepare( queue, allocator, dens_in, &o[ O_IMG_X0 ] ) )
        return;

    // ---- the card's diagram, its two slots, the majorants
    CardT card;
    if ( ! card.prepare( queue, pd, allocator, overflow ) )
        return;
    double rho = 1;
    if constexpr ( CONST ) {
        set_density( card.pb, dens_in );
        if constexpr ( requires { dens_in.data().raw; } ) {
            cuda_check( cudaMemcpyAsync( &rho, dens_in.data().raw, sizeof( double ), cudaMemcpyDeviceToHost, queue.stream ), "read of the density" );
            cuda_check( cudaStreamSynchronize( queue.stream ), "sync ( density )" );
        } else
            rho = double( dens_in );
    }
    card.pb.user_order = false;
    const SI fcap = std::max<SI>( SI( work.nb_facets.max ), 1 );
    card.pb.fcap = fcap;
    // MIXED: the double kernel's card, sharing the node records ( the majorants write them for both: they are float records
    // whatever the kernel ) and the passes' lists; only the seeds are packed per kernel
    CardD cardd;
    bool use_double = ! MIXED;
    int it_double = MIXED ? -1 : 0;
    if constexpr ( MIXED ) {
        if ( ! cardd.prepare( queue, pd, allocator, overflow, card.lend( false ) ) )
            return;
        if constexpr ( CONST )
            set_density( cardd.pb, dens_in );
        cardd.pb.user_order = false;
        cardd.pb.fcap = fcap;
        cardd.pb.nodes = card.pb.nodes;
    }
    /// the density at the width `s` for the cards of the descent ( a constant: nothing to do )
    double s_current = 0;
    auto set_stage = [&]( double s ) {
        s_current = s;
        if constexpr ( ! CONST ) {
            dh.at( queue, s, false );
            card.pb.dens = dh.dev;
            if constexpr ( MIXED )
                cardd.pb.dens = dh.dev;
        }
    };
    // ONE COO for the two slots: only the accepted diagram's is read ( the laplacian, at the start of an iteration ), and
    // the accepted diagram is the last one made but on a refused start or after a linear solve ( `coo_of`: then it is made
    // again ). ONE BLOCK, because between the assembly and the next diagram it is the linear solver's ( `CardLinear::prepare` )
    Slot<TR> slots[ 2 ];
    const SI coo_bytes = ( SI( sizeof( double ) ) + 2 * SI( sizeof( TR ) ) ) * fcap;
    auto *coo = static_cast<unsigned char *>( take( allocator, coo_bytes ) );
    if ( ! coo )
        return;
    double *coo_c = reinterpret_cast<double *>( coo );
    TR *coo_i = reinterpret_cast<TR *>( coo + SI( sizeof( double ) ) * fcap ), *coo_j = coo_i + fcap;
    for ( Slot<TR> &s : slots ) {
        s.fi = coo_i; s.fj = coo_j; s.fc = coo_c;
        if ( ! ( s.a = vec( n ) ) || ! ( s.counters = static_cast<Counters *>( take( allocator, SI( sizeof( Counters ) ) ) ) ) )
            return;
        if constexpr ( DIM == 2 )                        // ( the cuts of each cell, for the step `limits`: 2D only )
            if ( ! ( s.edges = static_cast<TR *>( take( allocator, SI( sizeof( TR ) ) * EDGE_CAP * n ) ) ) ||
                 ! ( s.nb_edges = static_cast<int *>( take( allocator, SI( sizeof( int ) ) * n ) ) ) )
                return;
    }
    Slot<TR> *cur = &slots[ 0 ], *tri = &slots[ 1 ];
    const Slot<TR> *coo_of = nullptr;                    // the slot whose facets the COO holds
    std::conditional_t<DIM == 2, Majorants<TR,TN>, gpu3d::MajorantsN<3,TR,TN>> maj;
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
    RedSlot<ExtentN<DIM>> red_ext;
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
        if ( o[ O_MG_SMOOTHED ] >= 0 ) lo.smoothed = int( o[ O_MG_SMOOTHED ] );
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
        } else if ( lin_float ? ! linf.prepare( allocator, n, nnz_cap, lo, &lws, coo, coo_bytes )
                              : ! lin.prepare( allocator, n, nnz_cap, lo, &lws, coo, coo_bytes ) )
            return;
    }

    StageTimer tm_maj, tm_diag, tm_asm, tm_lim;
    tm_maj.init(); tm_diag.init(); tm_asm.init(); tm_lim.init();
    double t_lin = 0, t_lim_host = 0;
    int nb_diag = 0;
    unsigned long long max_facets = 0;
    bool stop_all = false;                               // a capacity or a failure: the call is run again ( or raises )
    bool failed = false;                                 // ... a failure ( a cell past `max_vertices` )
    int res_cur = residual;
    double eps = 0;

    // ---- the gauge seed, the target, the start
    zero_fill( queue, r0_dev, SI( sizeof( SI ) ) );
    launch_kernel( queue, &find_rank0<TI>, blocks_for( n ), BLOCK, 0, n, ids, r0_dev );
    SI r0 = 0;
    read_back( queue, &r0, ( const SI * ) r0_dev, 1 );
    // the target in ranks ( `nu`: gathered again from `nu_in` at each stage rather than kept twice )
    launch_kernel( queue, &gather_ranks<TF,TI>, blocks_for( n ), BLOCK, 0, n, strided( nu_in ), ids, nu );

    // ---- THE AGGREGATION ( `sdotplan/Aggregation.h` ), its host side in RANKS
    sp::Aggregation agg;
    agg.init( n );
    agg.margin = o[ O_AGG_MARGIN ];
    agg.gap = o[ O_AGG_GAP ];
    agg.tol_abs = tol_abs;
    agg.tol_rel = tol_rel;
    double tau_min = 0;
    int nb_polish = 0;
    SI *dup_dev = nullptr, *drep_dev = nullptr;          // the exact duplicates and their representatives ( ranks )
    double *gap_dev = nullptr;                           // ... `w_rep - w_dup`
    double *nue = nullptr;                               // the targets, the duplicates' carried by their representatives
    double *const ae = b;                                // the clusters' shares: `b`'s room, dead whenever a diagram is reported
    SI *cl_beg_dev = nullptr, *cl_mem_dev = nullptr, cl_cap = 0;
    auto upload_clusters = [&]() -> bool {
        const SI nc = agg.nb_clusters(), nm = agg.nb_aggregated();
        if ( std::max( nc + 1, nm ) > cl_cap ) {
            cl_cap = 2 * std::max( nc + 1, nm ) + 64;
            cl_beg_dev = static_cast<SI *>( take( allocator, SI( sizeof( SI ) ) * cl_cap ) );
            cl_mem_dev = static_cast<SI *>( take( allocator, SI( sizeof( SI ) ) * cl_cap ) );
            if ( ! cl_beg_dev || ! cl_mem_dev )
                return false;
        }
        cuda_check( cudaMemcpyAsync( cl_beg_dev, agg.cl_begin.data(), sizeof( SI ) * ( nc + 1 ), cudaMemcpyHostToDevice, queue.stream ), "copy of the clusters" );
        cuda_check( cudaMemcpyAsync( cl_mem_dev, agg.cl_members.data(), sizeof( SI ) * nm, cudaMemcpyHostToDevice, queue.stream ), "copy of the clusters" );
        cuda_check( cudaStreamSynchronize( queue.stream ), "sync ( clusters )" );
        return true;
    };
    if ( agg_nb_dups > 0 ) {
        // the pairs ( user order ) to ranks: the users sorted, each rank looks itself up
        SI *tmp = static_cast<SI *>( take( allocator, SI( sizeof( SI ) ) * 2 * agg_nb_dups ) );
        if ( ! tmp ) return;
        launch_kernel( queue, &read_ints<std::remove_const_t<typename std::decay_t<decltype( dups_in )>::TF>>, blocks_for( 2 * agg_nb_dups ), BLOCK, 0,
                       2 * agg_nb_dups, strided( dups_in ), tmp );
        std::vector<SI> pairs( 2 * agg_nb_dups );
        read_back( queue, pairs.data(), ( const SI * ) tmp, 2 * agg_nb_dups );
        std::vector<SI> users( pairs );
        std::sort( users.begin(), users.end() );
        users.erase( std::unique( users.begin(), users.end() ), users.end() );
        const SI m = SI( users.size() );
        SI *users_dev = static_cast<SI *>( take( allocator, SI( sizeof( SI ) ) * m ) ), *ranks_dev = static_cast<SI *>( take( allocator, SI( sizeof( SI ) ) * m ) );
        dup_dev = static_cast<SI *>( take( allocator, SI( sizeof( SI ) ) * agg_nb_dups ) );
        drep_dev = static_cast<SI *>( take( allocator, SI( sizeof( SI ) ) * agg_nb_dups ) );
        nue = vec( n );
        gap_dev = vec( agg_nb_dups );
        if ( ! users_dev || ! ranks_dev || ! dup_dev || ! drep_dev || ! nue || ! gap_dev ) return;
        cuda_check( cudaMemcpyAsync( users_dev, users.data(), sizeof( SI ) * m, cudaMemcpyHostToDevice, queue.stream ), "copy of the duplicates" );
        launch_kernel( queue, &ranks_of_users<TI>, blocks_for( n ), BLOCK, 0, n, ids, m, ( const SI * ) users_dev, ranks_dev );
        std::vector<SI> ranks( m );
        read_back( queue, ranks.data(), ( const SI * ) ranks_dev, m );
        auto rank_of = [&]( SI u ) { return ranks[ SI( std::lower_bound( users.begin(), users.end(), u ) - users.begin() ) ]; };
        std::vector<SI> dr( agg_nb_dups ), rr( agg_nb_dups );
        for ( SI q = 0; q < agg_nb_dups; ++q ) { dr[ q ] = rank_of( pairs[ 2 * q ] ); rr[ q ] = rank_of( pairs[ 2 * q + 1 ] ); }
        cuda_check( cudaMemcpyAsync( dup_dev, dr.data(), sizeof( SI ) * agg_nb_dups, cudaMemcpyHostToDevice, queue.stream ), "copy of the duplicates" );
        cuda_check( cudaMemcpyAsync( drep_dev, rr.data(), sizeof( SI ) * agg_nb_dups, cudaMemcpyHostToDevice, queue.stream ), "copy of the duplicates" );
        cuda_check( cudaStreamSynchronize( queue.stream ), "sync ( duplicates )" );
        agg.set_duplicates( dr, rr );
        cuda_check( cudaMemcpyAsync( gap_dev, agg.dup_gap.data(), sizeof( double ) * agg_nb_dups, cudaMemcpyHostToDevice, queue.stream ), "copy of the gaps" );
        if ( ! upload_clusters() ) return;
    }
    /// the targets with the duplicates' carried by their representatives, for the current `nu`
    auto duplicate_targets_now = [&]() {
        if ( ! nue ) return;
        cuda_check( cudaMemcpyAsync( nue, nu, sizeof( double ) * n, cudaMemcpyDeviceToDevice, queue.stream ), "copy of the targets" );
        launch_kernel( queue, &duplicate_targets, blocks_for( agg_nb_dups ), BLOCK, 0, agg_nb_dups, ( const SI * ) dup_dev, ( const SI * ) drep_dev,
                       ( const double * ) nu, nue );
    };
    duplicate_targets_now();

    /// the report of slot `s` against the current target, residual and floor ( reductions, one read back ); with clusters, the
    /// residual reads their shares ( `ae` )
    auto report = [&]( Slot<TR> &s ) {
        const double *ar = nullptr;
        if ( agg.any() ) {
            cuda_check( cudaMemcpyAsync( ae, s.a, sizeof( double ) * n, cudaMemcpyDeviceToDevice, queue.stream ), "copy of the measures" );
            launch_kernel( queue, &cluster_shares, blocks_for( agg.nb_clusters() ), BLOCK, 0, agg.nb_clusters(), ( const SI * ) cl_beg_dev,
                           ( const SI * ) cl_mem_dev, ( const double * ) s.a, ( const double * ) nu, ae );
            ar = ae;
        }
        reduce( queue, n, DiagFn{ s.a, nu, eps, power, res_cur, ar, nue }, red_diag.partials, red_diag.out );
        reduce( queue, n, CentredG2{ ar ? ar : s.a, nu, red_diag.out, 1.0 / double( n ), power, res_cur }, red1.partials, red1.out );
        launch_kernel( queue, &gather_report, 1, 1, 0, rep_dev, ( const DiagRed * ) red_diag.out, ( const Sum1 * ) red1.out, ( const Counters * ) s.counters );
        read_back( queue, &s.rep, ( const Report * ) rep_dev, 1 );
    };

    /// THE DIAGRAM of `wt` into slot `s` ( the exact duplicates' weights tied to their representatives' first )
    auto diagram = [&]( const double *wt, Slot<TR> &s ) {
        if ( agg_nb_dups > 0 )
            launch_kernel( queue, &tie_duplicates, blocks_for( agg_nb_dups ), BLOCK, 0, agg_nb_dups, ( const SI * ) dup_dev, ( const SI * ) drep_dev,
                           ( const double * ) gap_dev, const_cast<double *>( wt ) );
        tm_maj.start( queue );
        maj.refresh( queue, pd, wt, node_records( card ) );
        auto run_on = [&]( auto &c ) {
            pack_card_weights( queue, c, n, wt );
            tm_maj.stop( queue );
            c.pb.w64 = Strided<TF,1>{ reinterpret_cast<const char *>( wt ), { SI( sizeof( double ) ) } };
            c.pb.res = StridedOut<TF,1>{ reinterpret_cast<char *>( s.a ), { SI( sizeof( double ) ) } };
            c.pb.fi = s.fi; c.pb.fj = s.fj; c.pb.fc = s.fc;
            c.counters = s.counters; c.pb.counters = s.counters;
            if constexpr ( DIM == 2 ) { c.pb.edges = s.edges; c.pb.nb_edges = s.nb_edges; }
            tm_diag.start( queue );
            c.run( queue, errors );
            tm_diag.stop( queue );
        };
        if constexpr ( MIXED ) {
            if ( use_double ) run_on( cardd );
            else              run_on( card );
        } else
            run_on( card );
        coo_of = &s;
        report( s );
        tm_maj.collect();
        tm_diag.collect();
        ++nb_diag;
        max_facets = std::max( max_facets, s.rep.nb_facets );
        failed = failed || s.rep.nb_failed > 0;
        if ( s.rep.nb_facets > ( unsigned long long ) fcap || s.rep.nb_failed > 0 )
            stop_all = true;
    };

    std::vector<double> rows;                            // the history, written to the card at the end
    const SI cap_steps = SI( hist.rows.shape( 0 ) );
    SI nb_steps = 0;
    int nb_steps_before = 0;                             // the steps of the previous stages
    auto after_step = [&]( int it, double t, int nb_evals ) {
        if ( nb_steps >= cap_steps ) return;
        const Report &r = cur->rep;
        double row[ sp::NB_HIST ];
        row[ sp::H_STEP ] = nb_steps_before + it;
        row[ sp::H_T ] = t;
        row[ sp::H_RESIDUAL_L2 ] = std::sqrt( r.d.sum_d2 );
        row[ sp::H_MIN_MASS ] = r.d.min_a;
        row[ sp::H_MAX_RESIDUAL ] = r.d.max_abs;
        row[ sp::H_NB_DIAG ] = nb_diag;
        row[ sp::H_NB_EVALS ] = nb_evals;
        row[ sp::H_S ] = s_current;
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
    double nu_min0 = 0;
    {
        reduce( queue, n, MinOf{ nu }, red_min.partials, red_min.out );
        Min1 m;
        read_back( queue, &m, ( const Min1 * ) red_min.out, 1 );
        nu_min0 = m.m;
    }
    double box_lo[ DIM ], box_hi[ DIM ];
    read_back( queue, box_lo, reinterpret_cast<const double *>( pd.box_min.data().raw ), DIM );
    read_back( queue, box_hi, reinterpret_cast<const double *>( pd.box_max.data().raw ), DIM );
    double box_diam2 = 0, box_volume = 1;
    for ( int k = 0; k < DIM; ++k ) {
        box_diam2 += ( box_hi[ k ] - box_lo[ k ] ) * ( box_hi[ k ] - box_lo[ k ] );
        box_volume *= box_hi[ k ] - box_lo[ k ];
    }

    // ---- THE STAGES OF THE WIDTH CONTINUATION ( `Solve.h`, `Continuation.h` ): `s0, s0 / ratio, ... >= s_min`, then `0`
    const double s0 = o[ O_CONV_S0 ] > 0 ? o[ O_CONV_S0 ] : 0.5 * std::sqrt( box_diam2 );
    auto the_steps = [&]() { return sp::continuation_steps( s0, o[ O_CONV_RATIO ], o[ O_CONV_MIN ] > 0 ? o[ O_CONV_MIN ] : dh.min_scale( queue ) ); };
    std::vector<double> scales = continuation == CONT_ALWAYS && dh.possible ? the_steps() : std::vector<double>{ 0.0 };
    set_stage( scales[ 0 ] );

    diagram( w, *cur );
    if ( ! stop_all && given && cur->rep.d.min_a < 1e-3 * nu_min0 ) {   // a warm start that empties a cell: the Voronoi, if better
        launch_kernel( queue, &fill_value, blocks_for( n ), BLOCK, 0, w2, 0.0, n );
        diagram( w2, *tri );
        if ( tri->rep.d.min_a > cur->rep.d.min_a ) { std::swap( cur, tri ); std::swap( w, w2 ); start = sp::START_VORONOI; }
    }
    if ( ! stop_all && cur->rep.d.min_a <= 0 ) {         // seeds outside the domain: the similarity
        reduce( queue, n, ExtentOf<DIM,TF>{ pos }, red_ext.partials, red_ext.out );
        ExtentN<DIM> ext;
        read_back( queue, &ext, ( const ExtentN<DIM> * ) red_ext.out, 1 );
        double a = 1;
        for ( int k = 0; k < DIM; ++k ) {
            const double span_dom = ( box_hi[ k ] - box_lo[ k ] ) * ( 1 - 2 * 0.1 ), span_pts = std::max( ext.hi[ k ] - ext.lo[ k ], 1e-300 );
            a = std::min( a, span_dom / span_pts );
        }
        Shift3 b{};
        for ( int k = 0; k < DIM; ++k )
            b.v[ k ] = ( box_lo[ k ] + box_hi[ k ] ) / 2 - a * ( ext.lo[ k ] + ext.hi[ k ] ) / 2;
        launch_kernel( queue, &similarity_weights<DIM,TF>, blocks_for( n ), BLOCK, 0, n, pos, a, b, w2 );
        diagram( w2, *tri );
        if ( tri->rep.d.min_a > cur->rep.d.min_a ) { std::swap( cur, tri ); std::swap( w, w2 ); start = sp::START_SIMILARITY; }
    }
    const double min_start_mass = cur->rep.d.min_a;
    // AUTO: the density is missing where cells are -> the continuation, from the same start
    if ( ! stop_all && continuation == CONT_AUTO && dh.possible && scales.size() == 1 && min_start_mass < o[ O_CONV_THRESHOLD ] * nu_min0 ) {
        scales = the_steps();
        set_stage( scales[ 0 ] );
        diagram( w, *cur );
    }
    launch_kernel( queue, &pick_value, 1, 1, 0, ( const double * ) w, r0, gauge );
    launch_kernel( queue, &subtract_scalar, blocks_for( n ), BLOCK, 0, w, ( const double * ) gauge, 1.0, n );

    /// the scale of a facet's coefficient ( `rho |facet| / 2 delta` ): in 2D `l ~ delta`, the density, or the mean density of
    /// the box; in 3D `|facet| ~ delta^2`, times the spacing of the seeds `( |box| / n )^( 1 / 3 )`
    const double c_scale = ( CONST ? rho : 1.0 / std::max( box_volume, 1e-300 ) ) * ( DIM == 3 ? std::cbrt( box_volume / double( std::max<SI>( n, 1 ) ) ) : 1.0 );

    // ---- THE AGGREGATION'S DETECTION ( `Aggregation::detect` ) on the laplacian just assembled and the accepted weights
    const SI heavy_cap = 4096;
    HeavyFacet *heavy = nullptr;
    unsigned long long *heavy_count = nullptr;
    auto detect = [&]() -> bool {
        if ( ! ( agg.margin > 0 ) )
            return false;
        reduce( queue, n, MaxAbs{ w }, redm.partials, redm.out );
        Max1 mw, md;
        read_back( queue, &mw, ( const Max1 * ) redm.out, 1 );
        reduce( queue, n, MaxOfV{ ldia }, redm.partials, redm.out );
        read_back( queue, &md, ( const Max1 * ) redm.out, 1 );
        // ON THE CARD, a second bound: a coefficient past 1e9 times the scale of a facet is merged whatever the weights -- the
        // multigrid's float levels do not hold it, and a cut decided by the float kernel can make it infinite ( a pair 1e-12
        // apart: 2.5e10, a linear solver failure at the first iteration, before any weight has grown ). ( 1e6 merged an
        // ordinary pair of a uniform cloud of 4000 seeds under the mixed kernel )
        const double c_cond = 1e9 * c_scale;
        const double c_star = std::min( tau_min / ( agg.margin * sp::ulp_of( mw.m ) ), c_cond );
        if ( ! ( md.m > c_star ) )
            return false;
        if ( ! heavy ) {
            heavy = static_cast<HeavyFacet *>( take( allocator, SI( sizeof( HeavyFacet ) ) * heavy_cap ) );
            heavy_count = static_cast<unsigned long long *>( take( allocator, SI( sizeof( unsigned long long ) ) ) );
            if ( ! heavy || ! heavy_count ) { agg.margin = 0; return false; }
        }
        zero_fill( queue, heavy_count, SI( sizeof( unsigned long long ) ) );
        launch_kernel( queue, &heavy_facets<TR>, blocks_for( n ), BLOCK, 0, n, ( const SI * ) lrow, ( const TR * ) lcol, ( const double * ) lval,
                       ( const double * ) ldia, c_star, ( const double * ) w, ( const double * ) nu, heavy_count, heavy, heavy_cap );
        unsigned long long cnt = 0;
        read_back( queue, &cnt, ( const unsigned long long * ) heavy_count, 1 );
        std::vector<HeavyFacet> hf( size_t( std::min<unsigned long long>( cnt, heavy_cap ) ) );
        if ( ! hf.empty() )
            read_back( queue, hf.data(), ( const HeavyFacet * ) heavy, SI( hf.size() ) );
        bool changed = false;
        for ( const HeavyFacet &h : hf ) {
            const double m = h.c * sp::ulp_of( std::max( std::fabs( h.wi ), std::fabs( h.wj ) ) );
            if ( ( m * agg.margin > agg.tau( std::min( h.nui, h.nuj ) ) || h.c > c_cond ) && agg.unite( h.i, h.j ) ) { changed = true; ++agg.nb_merged_pairs; }
        }
        if ( changed ) {
            agg.rebuild();
            if ( ! upload_clusters() ) { stop_all = failed = true; return false; }   // ( the pool is exhausted )
        }
        return changed;
    };

    // ---- THE CLUSTERS' ROWS OF THE LAPLACIAN, read back for the host's `k x k` blocks ( the re-splitting, and the direction's
    // local correction below )
    const int row_cap = 64;                              // entries per member row read back ( a 2D cell has a handful )
    SI *rs_mem = nullptr, *rs_len = nullptr, *rs_col = nullptr, *rs_at = nullptr;
    double *rs_dia = nullptr, *rs_val = nullptr, *rs_a = nullptr, *rs_nu = nullptr, *rs_dw = nullptr, *rs_dn = nullptr;
    SI rs_cap = 0;
    struct Rows { std::vector<SI> all, beg, len, col; std::vector<double> dia, val; };
    /// the movable members, cluster after cluster, and their rows of the laplacian `( lrow, lcol, lval, ldia )` as it is now
    auto read_rows = [&]( Rows &R ) -> bool {
        std::vector<SI> mem;
        R.all.clear(); R.beg.assign( 1, 0 );
        for ( SI c = 0; c < agg.nb_clusters(); ++c ) {
            agg.movable_members( c, mem );
            R.all.insert( R.all.end(), mem.begin(), mem.end() );
            R.beg.push_back( SI( R.all.size() ) );
        }
        const SI m = SI( R.all.size() );
        if ( m == 0 )
            return false;
        if ( m > rs_cap ) {
            rs_cap = m;
            rs_mem = static_cast<SI *>( take( allocator, SI( sizeof( SI ) ) * m ) );
            rs_len = static_cast<SI *>( take( allocator, SI( sizeof( SI ) ) * m ) );
            rs_col = static_cast<SI *>( take( allocator, SI( sizeof( SI ) ) * m * row_cap ) );
            rs_at = static_cast<SI *>( take( allocator, SI( sizeof( SI ) ) * m * row_cap ) );
            rs_dia = vec( m ); rs_val = vec( m * row_cap ); rs_a = vec( m ); rs_nu = vec( m ); rs_dw = vec( m ); rs_dn = vec( m * row_cap );
            if ( ! rs_mem || ! rs_len || ! rs_col || ! rs_at || ! rs_dia || ! rs_val || ! rs_a || ! rs_nu || ! rs_dw || ! rs_dn ) {
                rs_cap = 0;
                return false;
            }
        }
        cuda_check( cudaMemcpyAsync( rs_mem, R.all.data(), sizeof( SI ) * m, cudaMemcpyHostToDevice, queue.stream ), "copy of the members" );
        launch_kernel( queue, &member_rows<TR>, blocks_for( m ), BLOCK, 0, m, ( const SI * ) rs_mem, ( const SI * ) lrow, ( const TR * ) lcol,
                       ( const double * ) lval, ( const double * ) ldia, row_cap, rs_dia, rs_len, rs_col, rs_val );
        R.dia.resize( m ); R.len.resize( m ); R.col.resize( size_t( m ) * row_cap ); R.val.resize( size_t( m ) * row_cap );
        read_back( queue, R.dia.data(), ( const double * ) rs_dia, m );
        read_back( queue, R.len.data(), ( const SI * ) rs_len, m );
        read_back( queue, R.col.data(), ( const SI * ) rs_col, m * row_cap );
        read_back( queue, R.val.data(), ( const double * ) rs_val, m * row_cap );
        return true;
    };
    /// `L_CC x_C = r_C` per cluster ( `r`, per member, overwritten by `x`; 0 where a block cannot be solved )
    auto block_solves = [&]( const Rows &R, std::vector<double> &r ) {
        std::vector<double> M, x;
        for ( size_t c = 0; c + 1 < R.beg.size(); ++c ) {
            const SI b0 = R.beg[ c ], k = R.beg[ c + 1 ] - b0;
            bool fits = k >= 2 && k <= 64;
            M.assign( size_t( k * k ), 0.0 );
            x.assign( r.begin() + b0, r.begin() + b0 + k );
            for ( SI u = 0; u < k && fits; ++u ) {
                const SI q = b0 + u;
                fits = R.len[ q ] <= row_cap;
                M[ u * k + u ] = R.dia[ q ];
                for ( SI e = 0; e < std::min<SI>( R.len[ q ], row_cap ); ++e ) {
                    const SI j = R.col[ size_t( q ) * row_cap + e ];
                    const auto it = std::lower_bound( R.all.begin() + b0, R.all.begin() + b0 + k, j );
                    if ( it != R.all.begin() + b0 + k && *it == j )
                        M[ u * k + SI( it - ( R.all.begin() + b0 ) ) ] -= R.val[ size_t( q ) * row_cap + e ];
                }
            }
            for ( double v : M ) fits = fits && std::isfinite( v );
            if ( ! fits || ! sp::Aggregation::cholesky_solve( k, M, x ) )
                x.assign( k, 0.0 );
            std::copy( x.begin(), x.end(), r.begin() + b0 );
        }
    };

    // ---- THE DIRECTION WITH CLUSTERS: the laplacian's coefficient between two members of a cluster is `rho l / 2 delta` --
    // 2.5e10 for a pair 1e-12 apart, which the card's multigrid ( float levels ) does not survive. So the system is solved
    // with those coefficients capped at 1e6 times the scale of a facet ( `cap_internal`: the multigrid's floats still hold
    // there, `lines_equal`'s pair is at 4e6 ), the members' directions tied to their cluster's ( `tie_direction`: the
    // direction of the merged seed, to ~1e-6 ), then the split inside each cluster corrected by its exact local system, the
    // rest fixed: `L_CC dd_C = b_C - ( L d )_C` with the TRUE rows, its mean over the cluster ( weighted by the targets )
    // taken out -- one two-level step
    Rows rows_dir;
    bool rows_ok = false;
    auto correct_direction = [&]() {
        if ( ! rows_ok )
            return;
        const SI m = SI( rows_dir.all.size() );
        std::vector<SI> at;                              // the columns of the members' rows, then the members
        for ( SI q = 0; q < m; ++q )
            for ( SI e = 0; e < std::min<SI>( rows_dir.len[ q ], row_cap ); ++e ) at.push_back( rows_dir.col[ size_t( q ) * row_cap + e ] );
        const SI na = SI( at.size() );
        if ( na > m * row_cap ) return;
        cuda_check( cudaMemcpyAsync( rs_at, at.data(), sizeof( SI ) * na, cudaMemcpyHostToDevice, queue.stream ), "copy of the columns" );
        launch_kernel( queue, &gather_at, blocks_for( na ), BLOCK, 0, na, ( const SI * ) rs_at, ( const double * ) d, rs_dn );
        launch_kernel( queue, &gather_at, blocks_for( m ), BLOCK, 0, m, ( const SI * ) rs_mem, ( const double * ) d, rs_a );
        std::vector<double> dn( na ), dm( m ), bm( m );
        read_back( queue, dn.data(), ( const double * ) rs_dn, na );
        read_back( queue, dm.data(), ( const double * ) rs_a, m );
        read_back( queue, bm.data(), ( const double * ) rs_nu, m );   // ( `b` at the members, gathered before the solve )
        std::vector<double> r( m );
        for ( SI q = 0, k = 0; q < m; ++q ) {
            double ld = rows_dir.dia[ q ] * dm[ q ];
            for ( SI e = 0; e < std::min<SI>( rows_dir.len[ q ], row_cap ); ++e, ++k ) ld -= rows_dir.val[ size_t( q ) * row_cap + e ] * dn[ k ];
            r[ q ] = bm[ q ] - ld;
        }
        block_solves( rows_dir, r );
        // only the split inside each cluster: the cluster as a whole moves as the tied solve said ( its share of the targets )
        launch_kernel( queue, &gather_at, blocks_for( m ), BLOCK, 0, m, ( const SI * ) rs_mem, nue ? ( const double * ) nue : ( const double * ) nu, rs_a );
        std::vector<double> tm( m );
        read_back( queue, tm.data(), ( const double * ) rs_a, m );
        for ( size_t c = 0; c + 1 < rows_dir.beg.size(); ++c ) {
            double s = 0, sn = 0;
            for ( SI q = rows_dir.beg[ c ]; q < rows_dir.beg[ c + 1 ]; ++q ) { s += tm[ q ] * r[ q ]; sn += tm[ q ]; }
            for ( SI q = rows_dir.beg[ c ]; q < rows_dir.beg[ c + 1 ]; ++q ) r[ q ] -= sn > 0 ? s / sn : 0.0;
        }
        cuda_check( cudaMemcpyAsync( rs_dw, r.data(), sizeof( double ) * m, cudaMemcpyHostToDevice, queue.stream ), "copy of the correction" );
        launch_kernel( queue, &add_at, blocks_for( m ), BLOCK, 0, m, ( const SI * ) rs_mem, ( const double * ) rs_dw, d );
    };

    // ---- THE RE-SPLITTING ( `Newton.h::resplit` ): local Newton steps on the members of the clusters, one diagram each
    auto resplit = [&]() {
        const double *tg = nue ? nue : nu;
        Rows R;
        for ( int round = 0; round < 4 && ! stop_all; ++round ) {
            reduce( queue, n, FullResFn{ cur->a, tg }, red_diag.partials, red_diag.out );
            DiagRed fr;
            read_back( queue, &fr, ( const DiagRed * ) red_diag.out, 1 );
            if ( agg.converged( fr.max_abs, fr.max_rel ) )
                return;
            // the laplacian of the accepted diagram, and the members' rows
            if ( coo_of != cur ) {
                diagram( w, *cur );
                if ( stop_all ) return;
            }
            card.pb.fi = cur->fi; card.pb.fj = cur->fj; card.pb.fc = cur->fc;
            card.counters = cur->counters; card.pb.counters = cur->counters;
            assemble_laplacian_in( queue, card, lws, lrow, lcol, lval, ldia );
            if ( ! read_rows( R ) )
                return;
            const SI m = SI( R.all.size() );
            launch_kernel( queue, &gather_at, blocks_for( m ), BLOCK, 0, m, ( const SI * ) rs_mem, ( const double * ) cur->a, rs_a );
            launch_kernel( queue, &gather_at, blocks_for( m ), BLOCK, 0, m, ( const SI * ) rs_mem, tg, rs_nu );
            std::vector<double> am( m ), nm( m ), dw( m );
            read_back( queue, am.data(), ( const double * ) rs_a, m );
            read_back( queue, nm.data(), ( const double * ) rs_nu, m );
            double before = 0;
            for ( SI q = 0; q < m; ++q ) { dw[ q ] = nm[ q ] - am[ q ]; before = std::max( before, std::fabs( dw[ q ] ) ); }
            block_solves( R, dw );
            double d0 = 0;                               // the gauge: a shift of everyone if seed 0 moved
            for ( SI q = 0; q < m; ++q ) if ( R.all[ q ] == r0 ) d0 = dw[ q ];
            cuda_check( cudaMemcpyAsync( rs_dw, dw.data(), sizeof( double ) * m, cudaMemcpyHostToDevice, queue.stream ), "copy of the step" );
            launch_kernel( queue, &shifted_copy, blocks_for( n ), BLOCK, 0, n, ( const double * ) w, d0, w2 );
            launch_kernel( queue, &add_at, blocks_for( m ), BLOCK, 0, m, ( const SI * ) rs_mem, ( const double * ) rs_dw, w2 );
            diagram( w2, *tri );
            ++nb_polish;
            if ( stop_all ) return;
            launch_kernel( queue, &gather_at, blocks_for( m ), BLOCK, 0, m, ( const SI * ) rs_mem, ( const double * ) tri->a, rs_a );
            read_back( queue, am.data(), ( const double * ) rs_a, m );
            double after = 0;
            for ( SI q = 0; q < m; ++q ) after = std::max( after, std::fabs( am[ q ] - nm[ q ] ) );
            if ( trace )
                std::printf( "      re-splitting: members' residual %.3e -> %.3e, aggregated %.3e\n", before, after, tri->rep.d.max_abs );
            if ( ! ( after < before ) || ! ( tri->rep.d.min_a > 0 ) || ! agg.converged( tri->rep.d.max_abs, tri->rep.d.max_rel ) )
                return;
            std::swap( w, w2 );
            std::swap( cur, tri );
        }
    };

    // ---- THE STAGES, each a Newton ( `Newton.h::solves` )
    int status = sp::S_RUNNING, nb_iter = 0, nb_backtracks = 0, it_switch = -1, nb_limit_rounds = 0;
    SI nb_cell_lim = 0;
    double residual0 = 0, eps0 = 0, residual_max = cur->rep.d.max_abs, domain_mass = 0;
    if ( stop_all ) status = failed ? sp::S_FAILURE : sp::S_CAPACITY;
    for ( size_t stage = 0; stage < scales.size() && ! stop_all; ++stage ) {
        if ( stage > 0 ) {                               // the next density: the measures of the start redone
            set_stage( scales[ stage ] );
            diagram( w, *cur );
            if ( stop_all ) { status = failed ? sp::S_FAILURE : sp::S_CAPACITY; break; }
        }
        // the target at the scale of what the domain holds of THIS density
        domain_mass = cur->rep.d.sum_a;
        launch_kernel( queue, &gather_ranks<TF,TI>, blocks_for( n ), BLOCK, 0, n, strided( nu_in ), ids, nu );
        double nu_min = nu_min0;
        if ( domain_mass > 0 && snu.s > 0 && domain_mass != snu.s ) {
            launch_kernel( queue, &scale_values, blocks_for( n ), BLOCK, 0, n, nu, domain_mass / snu.s );
            nu_min *= domain_mass / snu.s;
        }
        duplicate_targets_now();
        tau_min = agg.tau( nu_min );
        if ( trace && scales.size() > 1 )
            std::printf( "  stage %d / %d : s = %.4e, domain mass %.6f, smallest mass %.3e\n", int( stage + 1 ), int( scales.size() ), s_current,
                         domain_mass, cur->rep.d.min_a );
        res_cur = residual;
        eps = 0;
        report( *cur );

        int st_status = sp::S_RUNNING;
        double t_last = 1;
        after_step( 0, 0, 1 );
        for ( int it = 0; it < maxit; ++it ) {
            const double worst = cur->rep.d.max_abs, worst_rel = cur->rep.d.max_rel;
            if ( switch_residual > 0 && res_cur != RES_LIN && worst_rel <= switch_residual ) {
                res_cur = RES_LIN;
                if ( stage == 0 ) it_switch = it;
                if ( trace ) std::printf( "      switch: residual -> lin ( max|a-nu|/nu %.3e <= %.3e )\n", worst_rel, switch_residual );
            }
            if ( it == 0 ) {
                eps = 0.5 * std::min( nu_min, cur->rep.d.min_a );
                if ( stage == 0 ) { residual0 = worst; eps0 = eps; }
            }
            report( *cur );                              // its merit in the residual in use, its cells under the floor
            double nr = merit_of( cur->rep, res_cur );
            residual_max = worst;
            if ( worst <= tol_abs || ( tol_rel > 0 && worst_rel <= tol_rel ) ) {
                if ( trace ) std::printf( "    it %2d  |r|_2 %.3e  max|a-nu| %.3e  CONVERGED%s\n", it, nr, worst, agg.any() ? " ( aggregated )" : "" );
                st_status = sp::S_CONVERGED;
                if ( agg.any() )
                    resplit();
                break;
            }
            ++nb_iter;
            const int g0 = nb_diag;

            // the right-hand side, projected on the range; the laplacian of the accepted diagram
            if ( coo_of != cur ) {                       // the COO holds a refused start's facets: the accepted diagram again
                diagram( w, *cur );
                if ( stop_all ) { st_status = failed ? sp::S_FAILURE : sp::S_CAPACITY; break; }
            }
            tm_asm.start( queue );
            card.pb.fi = cur->fi; card.pb.fj = cur->fj; card.pb.fc = cur->fc;
            card.counters = cur->counters; card.pb.counters = cur->counters;
            // the duplicates' rows linked to their representatives ( `Aggregation::link_duplicates`; the coefficient: the
            // scale of a facet, `rho`, or the mean density of the box )
            const bool links = agg_nb_dups > 0 && cur->rep.nb_facets + ( unsigned long long ) agg_nb_dups <= ( unsigned long long ) fcap;
            if ( links ) {
                const double c_link = c_scale;
                launch_kernel( queue, &append_links<TR>, blocks_for( agg_nb_dups ), BLOCK, 0, agg_nb_dups, ( const SI * ) dup_dev, ( const SI * ) drep_dev,
                               c_link, cur->rep.nb_facets, cur->fi, cur->fj, cur->fc );
                launch_kernel( queue, &bump_facets, 1, 1, 0, cur->counters, ( long long ) agg_nb_dups );
            }
            assemble_laplacian_in( queue, card, lws, lrow, lcol, lval, ldia );
            if ( links )
                launch_kernel( queue, &bump_facets, 1, 1, 0, cur->counters, - ( long long ) agg_nb_dups );
            // the pairs whose plane the doubles can no longer place are merged ( `Aggregation.h` ): the merit reads their
            // clusters from here on ( before `b`, whose room the shares take )
            if ( detect() ) {
                report( *cur );
                nr = merit_of( cur->rep, res_cur );
                if ( trace ) std::printf( "      aggregation: %d clusters, %d seeds ( |r|_2 %.3e )\n", int( agg.nb_clusters() ), int( agg.nb_aggregated() ), nr );
            }
            // with clusters: their true rows read back, then their internal coefficients capped ( see `correct_direction` )
            rows_ok = agg.any() && read_rows( rows_dir );
            if ( agg.any() )
                launch_kernel( queue, &cap_internal<TR>, blocks_for( agg.nb_clusters() ), BLOCK, 0, agg.nb_clusters(), ( const SI * ) cl_beg_dev,
                               ( const SI * ) cl_mem_dev, ( const SI * ) lrow, ( const TR * ) lcol, lval, ldia,
                               1e6 * c_scale );
            const double *tg = nue ? nue : nu;           // the targets ( an exact duplicate's carried by its representative )
            if ( res_cur != RES_LIN )
                reduce( queue, n, RhsSums{ cur->a, tg, power, res_cur }, red2.partials, red2.out );
            launch_kernel( queue, &rhs_kernel, blocks_for( n ), BLOCK, 0, n, ( const double * ) cur->a, tg, power, res_cur,
                           ( const Sum2 * ) red2.out, b );
            reduce( queue, n, SumOf{ b }, red1.partials, red1.out );
            launch_kernel( queue, &subtract_scalar, blocks_for( n ), BLOCK, 0, b, reinterpret_cast<const double *>( red1.out ), 1.0 / double( n ), n );
            if ( rows_ok )                               // ( `b` at the members, for the correction after the solve )
                launch_kernel( queue, &gather_at, blocks_for( SI( rows_dir.all.size() ) ), BLOCK, 0, SI( rows_dir.all.size() ), ( const SI * ) rs_mem,
                               ( const double * ) b, rs_nu );
            tm_asm.stop( queue );

            // the direction
            const double tl0 = wall_now();
            const bool solved = lin_kind == LIN_HOST ? host.solve( queue, L, b, d )
                              : lin_float ? linf.solve( queue, allocator, L, b, d ) : lin.solve( queue, allocator, L, b, d );
            t_lin += wall_now() - tl0;
            tm_asm.collect();
            if ( lin_kind != LIN_HOST )
                coo_of = nullptr;                        // ( the card's solver worked in the COO )
            if ( ! solved ) {
                st_status = sp::S_LINEAR_FAILURE;
                break;
            }
            if ( agg.any() ) {
                launch_kernel( queue, &tie_direction, blocks_for( agg.nb_clusters() ), BLOCK, 0, agg.nb_clusters(), ( const SI * ) cl_beg_dev,
                               ( const SI * ) cl_mem_dev, tg, d );
                correct_direction();
            }
            launch_kernel( queue, &pick_value, 1, 1, 0, ( const double * ) d, r0, gauge );
            launch_kernel( queue, &subtract_scalar, blocks_for( n ), BLOCK, 0, d, ( const double * ) gauge, 1.0, n );

            // ---- the step
            double t = std::min( 1.0, mult_ok * t_last );
            bool already = false;
            double alpha_lim = -1;
            int nb_evals = 0;
            if constexpr ( DIM == 2 ) if ( step_kind == STEP_LIMITS ) {   // ( 3D: `trials`, refused in `SdotPlanNd` )
                const double th0 = wall_now();
                tm_lim.start( queue );
                if constexpr ( CONST )
                    reduce( queue, n, AlphaForward<TF,TR>{ pos, box_min, box_max, w, d, cur->edges, cur->nb_edges, n, rho, eps }, red_min.partials, red_min.out );
                else
                    reduce( queue, n, AlphaForwardDens<TF,TR,D>{ pos, box_min, box_max, w, d, cur->edges, cur->nb_edges, n, card.pb.dens, eps },
                            red_min.partials, red_min.out );
                Min1 am;
                read_back( queue, &am, ( const Min1 * ) red_min.out, 1 );
                tm_lim.stop( queue );
                tm_lim.collect();
                t_lim_host += wall_now() - th0;
                t = am.m >= 1 ? 1.0 : factor * am.m;
                if constexpr ( ! CONST ) {               // no positive limit seen: the trials' first step
                    if ( ! ( t >= t_min ) )
                        t = std::min( 1.0, mult_ok * t_last );
                }
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
                    if constexpr ( CONST )
                        reduce( queue, n, AlphaBackward<TF,TR>{ pos, box_min, box_max, w, d, tri->a, tri->edges, tri->nb_edges, n, rho, eps, t, nue },
                                red_mc.partials, red_mc.out );
                    else
                        reduce( queue, n, AlphaBackwardDens<TF,TR,D>{ pos, box_min, box_max, w, d, tri->a, tri->edges, tri->nb_edges, n, card.pb.dens, eps, t, nue },
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
            sp::HopelessDamping hope;                     // ( `sdotplan/Report.h` )
            for ( int trial = 0; trial < max_backtracks && ! stop_all; ++trial ) {
                if ( ! ( trial == 0 && already ) ) {
                    launch_kernel( queue, &trial_weights, blocks_for( n ), BLOCK, 0, n, ( const double * ) w, ( const double * ) d, t, r0, w2 );
                    diagram( w2, *tri );
                    ++nb_evals;
                    if ( stop_all ) break;
                }
                const double m2 = tri->rep.d.min_a, n2r = merit_of( tri->rep, res_cur );
                if ( m2 >= eps && std::isfinite( n2r ) && n2r <= ( 1 - t / 2 ) * nr && n2r < nr ) { taken = true; break; }
                const bool hopeless = hope.refused( t, n2r, nr, m2 >= eps && std::isfinite( n2r ) );
                t /= 2;
                ++nb_backtracks;
                if ( t < t_min || hopeless )
                    break;
            }
            if ( trace ) {
                std::printf( "    it %2d  |r|_2 %.3e  max|a-nu| %.3e  %llu empty  step %.2e  %d diag  [majorant %.3f  diag %.3f  asm %.3f  lin %.3f  lim %.3f]",
                             it, nr, worst, cur->rep.d.nb_empty, t, nb_diag - g0, tm_maj.total, tm_diag.total, tm_asm.total, t_lin, tm_lim.total );
                if ( alpha_lim >= 0 ) std::printf( "  alpha* %.2e%s", alpha_lim, t < t_lim0 ? " REFUSED" : "" );
                std::printf( "\n" );
                std::fflush( stdout );
            }
            if ( stop_all ) { st_status = failed ? sp::S_FAILURE : sp::S_CAPACITY; break; }
            if constexpr ( MIXED ) {
                if ( ! taken && ! use_double ) {
                    // THE FLOAT KERNEL STAGNATES ( a cut decided in float on a degenerate cloud: its merit stops decreasing ):
                    // the double kernel from here on, the accepted diagram measured again with it, and the iteration done again
                    use_double = true;
                    it_double = it;
                    if ( trace ) std::printf( "      switch: kernel float -> double ( the float step stagnates )\n" );
                    diagram( w, *cur );
                    if ( stop_all ) { st_status = failed ? sp::S_FAILURE : sp::S_CAPACITY; break; }
                    // the target, rescaled to the domain's mass as the DOUBLE kernel measures it: the float kernel's sum is off
                    // by ~1e-11, which every cell then carried as a residual the double steps could not remove ( a uniform
                    // 3e-11 relative floor under a tight tolerance )
                    const double dm = cur->rep.d.sum_a;
                    if ( dm > 0 && domain_mass > 0 && dm != domain_mass ) {
                        launch_kernel( queue, &scale_values, blocks_for( n ), BLOCK, 0, n, nu, dm / domain_mass );
                        nu_min *= dm / domain_mass;
                        domain_mass = dm;
                        duplicate_targets_now();
                        tau_min = agg.tau( nu_min );
                        report( *cur );
                    }
                    continue;
                }
            }
            if ( ! taken && res_cur != RES_LIN ) {          // a refused log step: the iteration again in LIN ( `Newton.h` )
                res_cur = RES_LIN;
                if ( stage == 0 ) it_switch = it;
                if ( trace ) std::printf( "      switch: residual -> lin ( the step of the log residual is refused )\n" );
                --nb_iter;
                --it;
                continue;
            }
            if ( ! taken ) { st_status = sp::S_STAGNATION; break; }
            t_last = t;
            std::swap( w, w2 );
            std::swap( cur, tri );
            after_step( it + 1, t, nb_evals );
        }
        if ( st_status == sp::S_RUNNING )
            st_status = sp::S_MAX_ITERATIONS;
        if ( st_status != sp::S_CAPACITY && st_status != sp::S_FAILURE )
            residual_max = cur->rep.d.max_abs;
        status = st_status;
        nb_steps_before = nb_steps > 0 ? int( rows[ SI( nb_steps - 1 ) * sp::NB_HIST + sp::H_STEP ] ) + 1 : 0;
        if ( st_status == sp::S_LINEAR_FAILURE || stop_all )
            break;
    }

    // ---- THE FULL PROBLEM'S RESIDUAL, and the status when only the aggregated problem passed the test
    double full = residual_max;
    if ( agg.any() && ! stop_all ) {
        reduce( queue, n, FullResFn{ cur->a, nue ? nue : nu }, red_diag.partials, red_diag.out );
        DiagRed fr;
        read_back( queue, &fr, ( const DiagRed * ) red_diag.out, 1 );
        full = fr.max_abs;
        residual_max = cur->rep.d.max_abs;
        if ( status == sp::S_CONVERGED && ! agg.converged( fr.max_abs, fr.max_rel ) )
            status = sp::S_CONVERGED_AGGREGATED;
    }
    // ---- the clusters ( user order ): every seed its own index, the members of a cluster the smallest index of it
    {
        using TCl = std::remove_const_t<typename std::decay_t<decltype( clusters_out )>::TF>;
        launch_kernel( queue, &write_own_index<TI,TCl>, blocks_for( n ), BLOCK, 0, n, ids, strided_out<TCl,1>( clusters_out ) );
        const SI m = agg.nb_aggregated();
        SI *mu = m ? static_cast<SI *>( take( allocator, SI( sizeof( SI ) ) * 2 * m ) ) : nullptr;
        if ( mu ) {
            launch_kernel( queue, &users_at<TI>, blocks_for( m ), BLOCK, 0, m, ( const SI * ) cl_mem_dev, ids, mu );
            std::vector<SI> users( m ), reps( m );
            read_back( queue, users.data(), ( const SI * ) mu, m );
            for ( SI c = 0; c < agg.nb_clusters(); ++c ) {
                SI r = users[ agg.cl_begin[ c ] ];
                for ( SI k = agg.cl_begin[ c ]; k < agg.cl_begin[ c + 1 ]; ++k ) r = std::min( r, users[ k ] );
                for ( SI k = agg.cl_begin[ c ]; k < agg.cl_begin[ c + 1 ]; ++k ) reps[ k ] = r;
            }
            cuda_check( cudaMemcpyAsync( mu, users.data(), sizeof( SI ) * m, cudaMemcpyHostToDevice, queue.stream ), "copy of the clusters" );
            cuda_check( cudaMemcpyAsync( mu + m, reps.data(), sizeof( SI ) * m, cudaMemcpyHostToDevice, queue.stream ), "copy of the clusters" );
            launch_kernel( queue, &write_representatives<TCl>, blocks_for( m ), BLOCK, 0, m, ( const SI * ) mu, ( const SI * ) mu + m,
                           strided_out<TCl,1>( clusters_out ) );
            cuda_check( cudaStreamSynchronize( queue.stream ), "sync ( clusters )" );
        }
    }

    // ---- what comes out: the weights and the measures ( user order ), the diagram's weights and majorants
    launch_kernel( queue, &scatter_ranks<TF,TI>, blocks_for( n ), BLOCK, 0, n, ( const double * ) w, ids, strided_out<TF,1>( weights ) );
    launch_kernel( queue, &scatter_ranks<TF,TI>, blocks_for( n ), BLOCK, 0, n, ( const double * ) cur->a, ids, strided_out<TF,1>( masses ) );
    launch_kernel( queue, &write_strided<TF>, blocks_for( n ), BLOCK, 0, n, ( const double * ) w, strided_out<TF,1>( sorted_weights_out ) );
    maj.refresh( queue, pd, w, node_records( card ), strided_out<TF,2>( node_wa_out ), strided_out<TF,1>( node_wb_out ) );
    // ---- THE MOMENTS on the TRUE density ( `s = 0`, whatever stage the solve stopped at ) at the fitted weights ( barycentres,
    // the cost ), a last walk -- by the double kernel ( mixed in 2D, or once a 3D mixed solve switched to it ), or by the one
    // kernel of the solve. A 3D mixed solve that the float kernel finished takes them in float: its vertices are re-solved
    // in double like the cells it converged on, and a double 3D walk costs four float ones ( 56 against 12 ms at 1e5 )
    auto moments_on = [&]( auto &cm, auto mom_type ) {
        using M = typename decltype( mom_type )::type;
        pack_card_weights( queue, cm, n, ( const double * ) w );
        if ( stop_all )
            return;
        // the moments' card borrows the one of its kernel ( nodes, seeds, lists, finish store ), and writes its measures
        // and costs in the trial's measures and the direction, free now
        M mom;
        double *mres = tri->a, *mcost = d;
        if ( ! mom.prepare( queue, pd, allocator, overflow, cm.lend( true ) ) )
            return;
        if constexpr ( CONST )
            set_density( mom.pb, dens_in );
        else {
            dh.at( queue, 0, true );
            mom.pb.dens = dh.dev;
        }
        mom.pb.nodes = card.pb.nodes;
        if constexpr ( DIM == 2 ) mom.pb.w = cm.pb.w;     // ( 3D: the seeds, borrowed )
        mom.pb.w64 = Strided<TF,1>{ reinterpret_cast<const char *>( w ), { SI( sizeof( double ) ) } };
        mom.pb.user_order = true;
        mom.pb.res = StridedOut<TF,1>{ reinterpret_cast<char *>( mres ), { SI( sizeof( double ) ) } };
        mom.pb.bary = strided_out<TF,2>( bary );
        mom.pb.cost = StridedOut<TF,1>{ reinterpret_cast<char *>( mcost ), { SI( sizeof( double ) ) } };
        mom.run( queue, errors );
        reduce( queue, n, SumOf{ mcost }, red1.partials, red1.out );
        cuda_check( cudaMemcpyAsync( const_cast<void *>( ( const void * ) cost.data().raw ), red1.out, sizeof( double ),
                                     cudaMemcpyDeviceToDevice, queue.stream ), "copy of the cost" );
    };
    if constexpr ( ! MIXED )
        moments_on( card, std::type_identity<MomT>{} );
    else if constexpr ( DIM == 3 ) {
        if ( use_double ) moments_on( cardd, std::type_identity<MomT>{} );
        else              moments_on( card, std::type_identity<typename SolveCard<DIM,V,MEASURES | MOMENTS,TF,TI,D>::type>{} );
    } else
        moments_on( cardd, std::type_identity<MomT>{} );

    // ---- the capacity of the facets ( written into its ShapeVar: past it, loom runs the call again or raises )
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
    st[ sp::EPS ] = eps0;
    st[ sp::DOMAIN_MASS ] = domain_mass;
    st[ sp::NB_OVERFLOWED ] = 0;
    st[ sp::NB_CELL_LIM ] = double( nb_cell_lim );
    st[ sp::NB_LIMIT_ROUNDS ] = nb_limit_rounds;
    st[ sp::LIN_NB_HIERARCHIES ] = ls.nb_hierarchies;
    st[ sp::LIN_NB_ITER ] = ls.nb_iter;
    st[ sp::LIN_WORST ] = ls.worst;
    st[ sp::START ] = start;
    st[ sp::NB_CONTINUATION_STEPS ] = double( scales.size() );
    st[ sp::MIN_START_MASS ] = min_start_mass;
    st[ sp::IT_SWITCH ] = it_switch;
    st[ sp::IT_DOUBLE ] = it_double;
    st[ sp::NB_CLUSTERS ] = double( agg.nb_clusters() );
    st[ sp::NB_AGGREGATED ] = double( agg.nb_aggregated() );
    st[ sp::NB_DUPLICATES ] = double( agg.dups.size() );
    st[ sp::RESIDUAL_FULL ] = full;
    st[ sp::NB_POLISH ] = double( nb_polish );
    ( void ) t_lim_host;
    double *tmp = vec( std::max<SI>( SI( rows.size() ), sp::NB_STATS ) );
    if constexpr ( requires { allocator.taken; } )
        st[ sp::SCRATCH_BYTES ] = double( allocator.taken );
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

// ---- the card's linear solver alone ( a test hook: `SdotPlanNd._card_linear_solves` ) -----------------------------------

template<class T,class D>
__global__ void __launch_bounds__( BLOCK ) read_as( SI n, Strided<T,1> src, D *dst ) {
    const SI k = SI( blockIdx.x ) * BLOCK + threadIdx.x;
    if ( k < n ) dst[ k ] = D( src( k ) );
}

/// the options of `linear_solves`, ONE real tensor, in the order of `SdotPlanNd._card_linear_solves`
enum LinTestOpt : int { LT_N = 0, LT_K, LT_METHOD, LT_TOL, LT_SMOOTHED, LT_FLOAT, LT_RECYCLE, LT_REBUILD, LT_KCYCLE, LT_NU, NB_LT_OPTS };

template<class TV>
void linear_solves_in( const CudaQueue &queue, auto &allocator, const CsrView<SI> &L, SI nnz, SI k, const double *b, double *d,
                       std::vector<double> &its, const LinOptions &lo ) {
    CardLinear<SI,TV> lin;
    if ( ! lin.prepare( allocator, L.n, nnz, lo ) )
        return;
    for ( SI j = 0; j < k; ++j ) {
        const int it0 = lin.st.nb_iter;
        const bool ok = lin.solve( queue, allocator, L, b + j * L.n, d + j * L.n );
        its[ j ] = ok ? double( lin.st.nb_iter - it0 ) : -1.0;
    }
}

/// `k` systems `L d_j = b_j` with ONE laplacian ( `row`, `col`, `val`, `dia`: the CSR of `Laplacian2D.cuh` ), solved in sequence
/// by ONE solver, as in a Newton solve ( the recycled start, the hierarchy per solve or reused ). `rhs`: the `k n` values of
/// the `b_j` ( each of zero sum ); `opts`: `NB_LT_OPTS` reals. Out: `d` ( `k n`, each of zero mean ) and `its` ( `k`: the
/// iterations of each solve, -1 for a failure ).
void linear_solves( const CudaQueue &queue, const auto &row_in, const auto &col_in, const auto &val_in, const auto &dia_in, const auto &rhs_in,
                    const auto &opts_in, auto &&d_out, auto &&its_out, auto &allocator ) {
    auto vec = [&]( SI m ) { return static_cast<double *>( take( allocator, SI( sizeof( double ) ) * std::max<SI>( m, 1 ) ) ); };
    std::vector<double> o( NB_LT_OPTS, 0.0 );
    double *tmp = vec( NB_LT_OPTS );
    if ( ! tmp ) return;
    launch_kernel( queue, &read_strided<double>, 1, BLOCK, 0, SI( NB_LT_OPTS ), strided( opts_in ), tmp );
    read_back( queue, o.data(), ( const double * ) tmp, SI( NB_LT_OPTS ) );
    const SI n = SI( o[ LT_N ] ), k = SI( o[ LT_K ] );
    SI *row = static_cast<SI *>( take( allocator, SI( sizeof( SI ) ) * ( n + 1 ) ) );
    if ( ! row ) return;
    launch_kernel( queue, &read_as<std::remove_const_t<typename std::decay_t<decltype( row_in )>::TF>,SI>, blocks_for( n + 1 ), BLOCK, 0, n + 1, strided( row_in ), row );
    SI nnz = 0;
    read_back( queue, &nnz, ( const SI * ) row + n, 1 );
    SI *col = static_cast<SI *>( take( allocator, SI( sizeof( SI ) ) * std::max<SI>( nnz, 1 ) ) );
    double *val = vec( nnz ), *dia = vec( n ), *b = vec( k * n ), *d = vec( k * n );
    if ( ! col || ! val || ! dia || ! b || ! d ) return;
    launch_kernel( queue, &read_as<std::remove_const_t<typename std::decay_t<decltype( col_in )>::TF>,SI>, blocks_for( nnz ), BLOCK, 0, nnz, strided( col_in ), col );
    launch_kernel( queue, &read_strided<double>, blocks_for( nnz ), BLOCK, 0, nnz, strided( val_in ), val );
    launch_kernel( queue, &read_strided<double>, blocks_for( n ), BLOCK, 0, n, strided( dia_in ), dia );
    launch_kernel( queue, &read_strided<double>, blocks_for( k * n ), BLOCK, 0, k * n, strided( rhs_in ), b );
    LinOptions lo;
    lo.method = int( o[ LT_METHOD ] );
    if ( o[ LT_TOL ] > 0 ) lo.tol = o[ LT_TOL ];
    if ( o[ LT_SMOOTHED ] >= 0 ) lo.smoothed = int( o[ LT_SMOOTHED ] );
    if ( o[ LT_RECYCLE ] >= 0 ) lo.recycle = int( o[ LT_RECYCLE ] );
    if ( o[ LT_REBUILD ] > 0 ) lo.rebuild = int( o[ LT_REBUILD ] );
    if ( o[ LT_KCYCLE ] >= 0 ) lo.kcycle = int( o[ LT_KCYCLE ] );
    if ( o[ LT_NU ] > 0 ) lo.nu = int( o[ LT_NU ] );
    const CsrView<SI> L{ n, row, col, val, dia };
    std::vector<double> its( k, -1.0 );
    if ( o[ LT_FLOAT ] != 0 && lo.method == 1 )
        linear_solves_in<float>( queue, allocator, L, nnz, k, b, d, its, lo );
    else
        linear_solves_in<double>( queue, allocator, L, nnz, k, b, d, its, lo );
    launch_kernel( queue, &write_strided<double>, blocks_for( k * n ), BLOCK, 0, k * n, ( const double * ) d, strided_out<double,1>( d_out ) );
    double *hits = vec( k );
    if ( ! hits ) return;
    cuda_check( cudaMemcpyAsync( hits, its.data(), sizeof( double ) * k, cudaMemcpyHostToDevice, queue.stream ), "copy of the counts" );
    launch_kernel( queue, &write_strided<double>, 1, BLOCK, 0, k, ( const double * ) hits, strided_out<double,1>( its_out ) );
    cuda_check( cudaStreamSynchronize( queue.stream ), "sync ( counts )" );
}

} // namespace sdot::gpu2d
