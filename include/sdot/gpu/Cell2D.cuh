#pragma once

// =====================================================================================
// THE 2D CELLS OF THE CARD: power diagrams with weights in a box, on a CUDA device, with the BSP tree --
// what `PowerDiagram.measures` and Newton's iterations of `SdotPlanNd` need of a diagram, and nothing else.
//
// A dedicated path, written for the card ( `PowerDiagram_Bsp._card_variant` decides when it applies; 3D, a
// distribution that is not a constant, a domain that is not a box, the neighbour memory keep the generic path
// of `diagram/Ops.h` ). It is the old GPU campaign's kernel ( `nsdot/gpu_des_familles`, `FilMsk2D.cuh` +
// `Arbre.cuh` + `Mesures.cu` + `Hess2D.cuh` ) brought to this code base.
//
// = WHAT A CELL GIVES ( `Out`, a compile-time set: what is not asked for is not compiled )
//
//   * MEASURES  `rho |cell|`;
//   * FACETS    for each neighbour `j` of rank above the cell's, `c_kj = rho |facet| / ( 2 |p_k - p_j| )`: one
//               entry of a COO list ( rank, rank, value ), which `Laplacian2D.cuh` assembles into the symmetric
//               CSR of the Laguerre graph's laplacian -- Newton's Hessian;
//   * VJP       the adjoint of the measures, `g -> ( dm/dp )^T g, ( dm/dw )^T g`, as a GATHER: the facet ij moves
//               by `( x - p_i ) . dp_i / d` for cell i and the opposite for cell j, so with `c_ij` as above and `x_ij`
//               the middle of the facet,
//                   grad_w_i = sum_j c_ij ( g_i - g_j ),   grad_p_i = sum_j 2 c_ij ( g_i - g_j ) ( x_ij - p_i ),
//               written by cell `i` alone from its own facets ( no atomic, the same order at every run );
//   * MOMENTS   the barycentre and `rho int |x - p_k|^2` ( the transport cost of the cell, whose sum is
//               differentiated by the envelope theorem: `d cost / d p_k = 2 m_k ( p_k - b_k )` ).
//   * EDGES     the cuts of the cell's edges IN POLYGON ORDER ( rank of the neighbour, or a side of the box ), at most
//               `EDGE_CAP` ( more: the count is `-1` ): what the step of Newton's `Newton2D.cuh` rebuilds the cell from,
//               as an area polynomial along the direction, without a walk.
//
// = HOW A CELL IS BUILT
//
//   * ONE THREAD PER CELL, the cell in REGISTERS: `R1 = 8` vertices in three arrays with immediate indices only
//     ( the masks and the single barrel shift of `filmsk`: `RegCell::cut` ), starting from the domain box.
//   * THE OVERFLOW IS REDONE BY LATER PASSES, launched without reading any count back ( each strides over a list
//     whose length only the card knows ): the cells that needed more than `R1` vertices ( a tenth ) by a second pass
//     with `R2 = 16` registers, what overflows that ( a thousandth ) by a third pass with ONE WARP PER CELL
//     ( `WarpCell`, the vertices in shared memory ), and what overflows that ( none on the campaign's clouds ) by a
//     fourth pass, the same warp cell in GLOBAL memory, within a FIXED BUDGET taken once per call ( `Overflow` ): a few
//     slots ( `overflow_warps`, Python's choice ) of the per-cell vertex limit each ( `max_vertices`, bounded by
//     `n + 4`: a cell cannot have more ), and a persistent grid of exactly that many warps striding over the list --
//     the remaining cells go through in successive BATCHES, a warp taking its next cell into the slot it freed. No
//     capacity to guess, no run again, no read back. Past `max_vertices` ( a hard limit, Python's ) a cell is a
//     FAILURE, reported through loom's error buffer ( a `KernelFailure` naming the seed, eager or traced ): never a NaN.
//   * THE FLOAT KERNEL'S FINISH IN A KERNEL OF ITS OWN ( `finish_pass` ): the cells of the first two passes are left in
//     global memory and finished there -- the double arithmetic of the re-solve cost the walk half its occupancy.
//   * THE TREE NODES as ONE aligned record in FLOAT ( `Node`: box rounded outward, majorant slopes and constant rounded
//     up, slice ), rebuilt from the tree's tensors at every call; the tree is the perfect binary tree in PREORDER of
//     `ProviderBsp` ( left child `n + 1`, right child `n + 2^( h - 1 )` ). The pruning is in float for both kernels
//     ( the double one with a margin: `vertex_may_go` ).
//   * THE WALK'S STACK IS INDEXED BY HEIGHT: a depth-first descent pushes at most one node per height, and the heights
//     on the stack decrease from bottom to top, so `stack[ h ]` plus a bit mask of the occupied heights is the stack,
//     the top being the lowest bit. No packing, no depth limit: the node index type ( `TN` ) and the stack's size
//     ( `MAX_HEIGHT`, 32 or 64 ) are template parameters chosen by Python from the tree ( `Variant` ), as is the rank type
//     ( `TR`, 64 bits past 2^31 seeds ).
//   * THE ACCURACY FIXES of the old campaign ( its `doc/04-echelle.md` ): the cell lives in the SEED's frame; the
//     positions and weights reach a float kernel as TWO floats ( `x = xh + xl` ), so that a plane carries the double
//     difference rounded once; at the end every vertex is RE-SOLVED in double from the two planes that carry it, and
//     everything a cell gives ( measure, facets, adjoint, moments ) is taken on these re-solved vertices: the float
//     only decides WHICH cuts apply.
//
// The double kernel ( `TK = double` ) is the same code without the two-float split, the finish pass and the
// re-solve. Diagnosis: `SDOT_CARD_STATS=1` prints how many cells each pass left over ( a read back: a diagnosis only ).
// =====================================================================================

#include <loom/support/kernels/CudaQueue.h>
#include <loom/support/common_types.h>
#include <loom/support/Ct.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <type_traits>
#include <utility>

namespace sdot::gpu2d {

constexpr int    BLOCK      = 128;                       ///< threads per block, every kernel here
constexpr int    R1         = 8;                         ///< vertices in registers, first pass
constexpr int    R2         = 16;                        ///< ... second pass
constexpr int    RD         = 12;                        ///< vertex slots per cell left for the finish ( float kernel )
constexpr int    MINB1      = 1;                         ///< resident blocks per SM asked of the first pass ( `__launch_bounds__` )
constexpr double DET_MIN    = 1e-6;                      ///< under this relative determinant, the float vertex is kept
/// the new vertices of the first and second passes from their two planes ( `refine` ) rather than interpolated:
/// float against double max 1.1e-8 -> 4.6e-14 on the uniform cloud, 2.3e-9 -> 8e-14 on the lines, for +16 %
/// of time ( 9.4 -> 10.9 ns/seed ); the median is 1e-14 either way. Off: the speed is what the card is for.
constexpr bool   EXACT_VERTICES = false;

/// WHAT A CELL GIVES ( see the header ): a bit set, a template parameter of everything below
enum Out : unsigned { MEASURES = 1, FACETS = 2, VJP = 4, MOMENTS = 8, EDGES = 16 };

/// EDGES: the edges kept per cell ( SoA, `edges[ q * n + k ]` ); a cell with more says `-1` ( a hundredth of a percent of
/// the uniform cells, a tenth on the lines: the step treats them as unknown )
constexpr int EDGE_CAP = 16;

/// what goes to loom's error buffer besides the capacities ( kind 2, `ErrorKind::failure`: `id` is the code below,
/// `value` the user index of the cell ); the message is Python's ( `PowerDiagram_Bsp._CARD_FAILURES` )
enum Failure : int { FAIL_TOO_MANY_VERTICES = 1 };
constexpr int ERROR_KIND_FAILURE = 2;

/// THE VARIANT, chosen by Python from the inputs ( `PowerDiagram_Bsp._card_variant` ): the kernel's float, the type of
/// a rank / cut identifier ( `int` up to 2^31 - 5 seeds ), the type of a node index ( `int` up to 2^31 - 1 nodes ), and
/// the walk's stack ( one slot per height: 32 or 64 ).
template<class _TK,class _TR,class _TN,int _MAX_HEIGHT>
struct Variant {
    using TK = _TK;
    using TR = _TR;
    using TN = _TN;
    static constexpr int MAX_HEIGHT = _MAX_HEIGHT;
    static_assert( MAX_HEIGHT == 32 || MAX_HEIGHT == 64, "the stack's mask is a 32 or 64-bit word" );
    using Mask = std::conditional_t<MAX_HEIGHT == 32,unsigned,unsigned long long>;
};

// ---- the kernel's float: how a seed is stored for it, and the seed's frame ---------------------------------

template<class TK> struct KernelSeeds;
template<> struct KernelSeeds<float>  { using Pos = float4;  using Wt = float2; };   ///< ( xh, yh, xl, yl ), ( wh, wl )
template<> struct KernelSeeds<double> { using Pos = double2; using Wt = double; };

/// the seed of the cell, as the cut sees it
template<class TK> struct Frame;
template<> struct Frame<float>  { float  xh, yh, xl, yl, wh, wl; };
template<> struct Frame<double> { double x, y, w; };

/// a half-space `dx . v <= off`, `v` counted from the seed; `id`: a rank ( >= 0 ) or a side of the box ( < 0 )
template<class TK,class TR>
struct Plane { TK dx, dy, off; TR id; };

__device__ __forceinline__ Frame<float>  frame_of( float4 p, float2 w )   { return { p.x, p.y, p.z, p.w, w.x, w.y }; }
__device__ __forceinline__ Frame<double> frame_of( double2 p, double w )  { return { p.x, p.y, w }; }

/// THE POWER BISECTOR in the seed's frame: `|x - p0|^2 - w0 <= |x - q|^2 - wq` is `d . v <= |d|^2 / 2 + ( w0 - wq ) / 2`
/// with `d = q - p0`. In float, `d` is the difference of the two-float positions: the high parts first ( exact
/// as soon as the seeds are close, and rounded relatively to `d` otherwise ), then the low parts.
template<bool W,class TR>
__device__ __forceinline__ Plane<float,TR> bisector( const Frame<float> &f, float4 q, float2 wq, TR id ) {
    Plane<float,TR> p;
    p.dx  = ( q.x - f.xh ) + ( q.z - f.xl );
    p.dy  = ( q.y - f.yh ) + ( q.w - f.yl );
    p.off = 0.5f * ( p.dx * p.dx + p.dy * p.dy );
    if constexpr ( W )
        p.off += 0.5f * ( ( f.wh - wq.x ) + ( f.wl - wq.y ) );
    p.id  = id;
    return p;
}

template<bool W,class TR>
__device__ __forceinline__ Plane<double,TR> bisector( const Frame<double> &f, double2 q, double wq, TR id ) {
    Plane<double,TR> p;
    p.dx  = q.x - f.x;
    p.dy  = q.y - f.y;
    p.off = 0.5 * ( p.dx * p.dx + p.dy * p.dy );
    if constexpr ( W )
        p.off += 0.5 * ( f.w - wq );
    p.id  = id;
    return p;
}

// ---- the tree, as the kernel reads it --------------------------------------------------------------------

/// ONE NODE, one aligned record in float: the threads of a warp read DIFFERENT nodes ( the walks diverge ), so what
/// costs is the number of transactions per node, and a 16-byte aligned record is read in 16-byte loads. 32 bytes
/// for a Voronoi node, 48 weighted ( `int` slices ).
template<bool W,class TR> struct alignas( 16 ) Node;
template<class TR> struct alignas( 16 ) Node<false,TR> { float lo[ 2 ], hi[ 2 ]; TR beg, end; };
template<class TR> struct alignas( 16 ) Node<true,TR>  { float lo[ 2 ], hi[ 2 ], a[ 2 ], b; TR beg, end; };

/// the first bytes of a node: what the ordering of two children reads
struct alignas( 16 ) Box { float lo[ 2 ], hi[ 2 ]; };

/// a node seen from the seed: its box in the seed's frame, and for a weighted diagram the slopes and the
/// constant `c = w0 - b - a . p0` of the majorant ( `w( q ) <= a . q + b` )
struct CBox { float lo[ 2 ], hi[ 2 ], a[ 2 ], c; };

template<bool W,class TR>
__device__ __forceinline__ CBox centred( const Node<W,TR> &nd, const Frame<float> &f ) {
    CBox B;
    // the box is rounded OUTWARD ( `make_nodes` ), and its distance to the seed is taken on both halves of
    // the seed: an error relative to that distance, not to the coordinates
    B.lo[ 0 ] = ( nd.lo[ 0 ] - f.xh ) - f.xl;
    B.lo[ 1 ] = ( nd.lo[ 1 ] - f.yh ) - f.yl;
    B.hi[ 0 ] = ( nd.hi[ 0 ] - f.xh ) - f.xl;
    B.hi[ 1 ] = ( nd.hi[ 1 ] - f.yh ) - f.yl;
    if constexpr ( W ) {
        B.a[ 0 ] = nd.a[ 0 ];
        B.a[ 1 ] = nd.a[ 1 ];
        // the margin of `b` ( `AaBsp._weight_majorant`, 1e-6 of the scale ) is far above these roundings
        B.c = ( ( f.wh - nd.b ) + f.wl ) - ( nd.a[ 0 ] * f.xh + nd.a[ 1 ] * f.yh );
    } else {
        B.a[ 0 ] = B.a[ 1 ] = B.c = 0;
    }
    return B;
}

/// the seed's frame for the PRUNING, which is done in float whatever the kernel's float ( the double kernel
/// only adds a margin, see below ): a double seed split in two floats
__device__ __forceinline__ Frame<float> prune_frame( const Frame<float> &f ) { return f; }
__device__ __forceinline__ Frame<float> prune_frame( const Frame<double> &f ) {
    Frame<float> r;
    r.xh = __double2float_rn( f.x ); r.xl = __double2float_rn( f.x - double( r.xh ) );
    r.yh = __double2float_rn( f.y ); r.yl = __double2float_rn( f.y - double( r.yh ) );
    r.wh = __double2float_rn( f.w ); r.wl = __double2float_rn( f.w - double( r.wh ) );
    return r;
}

/// THE PRUNING TEST for one vertex `v` ( seed's frame ): `<= 0` says "a seed of the box may remove it". The
/// minimum of `|v - q|^2 - a . q` over the box is separable, free at `q = v + a / 2`, one clamp per axis gives it
/// ( `cell/Pruning.h`, the same test in the absolute frame ).
template<bool W>
__device__ __forceinline__ bool vertex_may_go( const CBox &B, float vx, float vy ) {
    float yx = vx, yy = vy;
    if constexpr ( W ) {
        yx += 0.5f * B.a[ 0 ];
        yy += 0.5f * B.a[ 1 ];
    }
    yx = fminf( fmaxf( yx, B.lo[ 0 ] ), B.hi[ 0 ] );
    yy = fminf( fmaxf( yy, B.lo[ 1 ] ), B.hi[ 1 ] );
    const float ux = yx - vx, uy = yy - vy;
    float s = ux * ux + uy * uy - ( vx * vx + vy * vy );
    if constexpr ( W )
        s += B.c - ( B.a[ 0 ] * yx + B.a[ 1 ] * yy );
    return s <= 0.f;
}

/// THE SAME TEST FOR A DOUBLE VERTEX, in float: the pruning only has to be CONSERVATIVE, and the double kernel
/// spent half its time in it at 1/32 of the float rate. The vertex is rounded to float, the test answers "may
/// go" up to a margin of 1e-6 of the magnitude of its terms ( sixteen float epsilons: far above the roundings
/// of the vertex, of the centred box and of the majorant constant ), so that what the float prunes, the double
/// would have pruned too.
template<bool W>
__device__ __forceinline__ bool vertex_may_go( const CBox &B, double vxd, double vyd ) {
    const float vx = __double2float_rn( vxd ), vy = __double2float_rn( vyd );
    float yx = vx, yy = vy;
    if constexpr ( W ) {
        yx += 0.5f * B.a[ 0 ];
        yy += 0.5f * B.a[ 1 ];
    }
    yx = fminf( fmaxf( yx, B.lo[ 0 ] ), B.hi[ 0 ] );
    yy = fminf( fmaxf( yy, B.lo[ 1 ] ), B.hi[ 1 ] );
    const float ux = yx - vx, uy = yy - vy;
    const float p = ux * ux + uy * uy, q = vx * vx + vy * vy;
    float s = p - q, t = p + q;
    if constexpr ( W ) {
        const float ax = B.a[ 0 ] * yx, ay = B.a[ 1 ] * yy;
        s += B.c - ( ax + ay );
        t += fabsf( B.c ) + fabsf( ax ) + fabsf( ay );
    }
    return s <= 1e-6f * t;
}

/// the squared distance from the seed to a box: the ORDER of two children, not a test
__device__ __forceinline__ float proximity( const Box &b, const Frame<float> &f ) {
    const float ex = fmaxf( fmaxf( b.lo[ 0 ] - f.xh, f.xh - b.hi[ 0 ] ), 0.f );
    const float ey = fmaxf( fmaxf( b.lo[ 1 ] - f.yh, f.yh - b.hi[ 1 ] ), 0.f );
    return ex * ex + ey * ey;
}

// ---- what a kernel receives ----------------------------------------------------------------------------

/// a strided read of a tensor of the call ( byte strides, as `TensorView` keeps them )
template<class T,int N>
struct Strided {
    const char *p;
    SI          s[ N ];

    __device__ __forceinline__ T operator()( SI i ) const { return *reinterpret_cast<const T *>( p + i * s[ 0 ] ); }
    __device__ __forceinline__ T operator()( SI i, SI j ) const { return *reinterpret_cast<const T *>( p + i * s[ 0 ] + j * s[ 1 ] ); }
    __device__ __forceinline__ T operator()( SI i, SI j, SI k ) const { return *reinterpret_cast<const T *>( p + i * s[ 0 ] + j * s[ 1 ] + k * s[ 2 ] ); }
};

/// ... and a strided write ( `p == nullptr`: nothing asked )
template<class T,int N>
struct StridedOut {
    char *p = nullptr;
    SI    s[ N ] = {};

    __device__ __forceinline__ T &operator()( SI i ) const { return *reinterpret_cast<T *>( p + i * s[ 0 ] ); }
    __device__ __forceinline__ T &operator()( SI i, SI j ) const { return *reinterpret_cast<T *>( p + i * s[ 0 ] + j * s[ 1 ] ); }
    __device__ __forceinline__ T &operator()( SI i, SI j, SI k ) const { return *reinterpret_cast<T *>( p + i * s[ 0 ] + j * s[ 1 ] + k * s[ 2 ] ); }
};

template<class View>
Strided<std::remove_const_t<typename View::TF>,View::ct_rank> strided( const View &v ) {
    Strided<std::remove_const_t<typename View::TF>,View::ct_rank> res;
    res.p = reinterpret_cast<const char *>( v.data().raw );
    [&]<int... I>( std::integer_sequence<int,I...> ) {
        ( ( res.s[ I ] = SI( v._strides[ Ct<int,I>() ] ) ), ... );
    }( std::make_integer_sequence<int,View::ct_rank>() );
    return res;
}

/// a writable view of an output, or nothing ( an unbound gradient: `NoneTensor` )
template<class T,int N,class View>
StridedOut<T,N> strided_out( const View &v ) {
    StridedOut<T,N> res;
    if constexpr ( requires { v.data().raw; } ) {
        static_assert( View::ct_rank == N );
        res.p = reinterpret_cast<char *>( const_cast<std::remove_const_t<typename View::TF> *>( v.data().raw ) );
        [&]<int... I>( std::integer_sequence<int,I...> ) {
            ( ( res.s[ I ] = SI( v._strides[ Ct<int,I>() ] ) ), ... );
        }( std::make_integer_sequence<int,N>() );
    }
    return res;
}

/// what the passes count on the card ( zeroed at the start of a run )
struct Counters {
    unsigned long long ovf[ 3 ];                         ///< the lists of the second, third and fourth passes
    unsigned long long nb_facets;                        ///< the upper facets the cells WANTED ( maybe more than the COO holds )
    unsigned long long nb_failed;                        ///< the cells past `max_vertices`
    unsigned long long pad[ 3 ];
};

/// EVERYTHING A CELL READS AND WRITES, by value ( kernel parameters )
template<class _V,class TF,class TI,bool _W,unsigned _OUT>
struct Problem {
    using V   = _V;
    using TK  = typename V::TK;
    using TR  = typename V::TR;
    using TN  = typename V::TN;
    using Pos = typename KernelSeeds<TK>::Pos;
    using Wt  = typename KernelSeeds<TK>::Wt;
    static constexpr bool     W   = _W;
    static constexpr unsigned OUT = _OUT;

    // ---- the diagram
    const Node<W,TR> *nodes;                             ///< in float for both kernels: the pruning is in float
    const Pos        *pos;                               ///< the seeds in tree order, in the kernel's form
    const Wt         *w;                                 ///< their weights ( W only )
    Strided<TF,2>     pos64;                             ///< the seeds as given, `[ n, 2 ]`, tree order
    Strided<TF,1>     w64;                               ///< the weights as given ( W only )
    Strided<TF,1>     box_min, box_max;                  ///< the domain
    Strided<TI,1>     ids;                               ///< rank -> the user's index
    TR                n;
    int               depth;
    double            rho;                               ///< the ( constant ) density, if `rho_dev` is null
    const TF         *rho_dev;                           ///< ... or read on the card ( a 0-d tensor of the call )
    bool              user_order;                        ///< per-cell outputs at the user's index ( else at the rank )

    // ---- what the cells write ( see `Out` )
    StridedOut<TF,1>  res;                               ///< MEASURES
    TR               *fi, *fj;                          ///< FACETS: the COO of the upper facets, in ranks
    double           *fc;
    unsigned long long fcap;                             ///< ... its capacity
    Counters         *counters;
    Strided<TF,1>     g;                                 ///< VJP: the cotangent of the measures ( user order )
    StridedOut<TF,2>  grad_pos;                          ///< ... -> the positions, rank order ( may be null )
    StridedOut<TF,1>  grad_w;                            ///< ... -> the weights, rank order ( may be null )
    StridedOut<TF,2>  bary;                              ///< MOMENTS: the barycentre ( the seed if empty )
    StridedOut<TF,1>  cost;                              ///< ... and `rho int_cell |x - p|^2`
    TR               *edges;                             ///< EDGES: the cuts in polygon order, `edges[ q * n + rank ]`
    int              *nb_edges;                          ///< ... their number per rank ( `-1`: more than `EDGE_CAP` )

    __device__ __forceinline__ double density() const { return rho_dev ? double( *rho_dev ) : rho; }
    __device__ __forceinline__ Wt weight( TR q ) const { if constexpr ( W ) return w[ q ]; else return Wt{}; }
    __device__ __forceinline__ SI user( TR k ) const { return user_order ? SI( ids( k ) ) : SI( k ); }
};

// ---- the cell in registers -------------------------------------------------------------------------------

/// no plane to re-read: the new vertices are interpolated along their edge
struct NoEdgePlanes {};

/// A NEW VERTEX FROM ITS TWO PLANES, the cut and the edge it falls on, instead of interpolated along that edge
/// ( `EXACT_VERTICES` ). Kept interpolated where the two planes are nearly parallel.
template<class TK,class TR,class EP>
__device__ __forceinline__ void refine( const Plane<TK,TR> &p, const EP &edge_plane, TR cid, TK &vx, TK &vy ) {
    const Plane<TK,TR> e = edge_plane( cid );
    const TK det = p.dx * e.dy - p.dy * e.dx;
    const TK n2 = ( p.dx * p.dx + p.dy * p.dy ) * ( e.dx * e.dx + e.dy * e.dy );
    if ( det * det > TK( 1e-6 ) * n2 ) {
        const TK inv = TK( 1 ) / det;
        vx = ( p.off * e.dy - e.off * p.dy ) * inv;
        vy = ( p.dx * e.off - e.dx * p.off ) * inv;
    }
}

/// `R` vertices SORTED in `0 .. nb - 1`, counterclockwise, in registers: every index below is a compile-time
/// constant once the loops are unrolled ( a dynamic index would send the arrays to local memory ). Edge `i`
/// goes from vertex `i` to vertex `i + 1` and carries the cut `c[ i ]`; vertex `i` lies on `c[ i - 1 ]` and
/// `c[ i ]`. `nb < 0`: the cell overflowed `R`.
template<class TK,class TR,int R>
struct RegCell {
    static constexpr int SUR = 3;                        ///< a non-empty cell has three vertices at least
    static_assert( R >= 4 && R < 32, "the box fits in the registers, a mask in 32 bits" );

    TK  x[ R ], y[ R ];
    TR  c[ R ];
    int nb;

    /// the domain box `[ x0, x1 ] x [ y0, y1 ]` ( seed's frame ); its sides are `-1` bottom, `-2` right,
    /// `-3` top, `-4` left
    __device__ __forceinline__ void init( TK x0, TK y0, TK x1, TK y1 ) {
#pragma unroll
        for ( int i = 0; i < R; ++i ) {
            x[ i ] = i == 1 || i == 2 ? x1 : x0;
            y[ i ] = i == 2 || i == 3 ? y1 : y0;
            c[ i ] = i < 4 ? TR( -1 - i ) : TR( 0 );
        }
        nb = 4;
    }

    template<bool W>
    __device__ __forceinline__ bool may_be_cut_by( const CBox &B ) const {
        bool res = false;
#pragma unroll
        for ( int i = 0; i < R; ++i ) {
            if ( i >= SUR && i >= nb ) break;
            res |= vertex_may_go<W>( B, x[ i ], y[ i ] );
        }
        return res;
    }

    /// the cyclic run of `m` ( an outside mask with several runs ) that holds the vertex farthest outside --
    /// the rare path of `cut`, written for the registers too ( the argmax by selects, the run by bit tricks on
    /// `m` rotated so that the argmax is bit 0 )
    __device__ __forceinline__ unsigned main_run( const Plane<TK,TR> &p, unsigned m, unsigned valid ) const {
        int a = 0;
        TK best = p.dx * x[ 0 ] + p.dy * y[ 0 ] - p.off;
#pragma unroll
        for ( int i = 1; i < R; ++i ) {
            if ( i >= SUR && i >= nb ) break;
            const TK s = p.dx * x[ i ] + p.dy * y[ i ] - p.off;
            a = s > best ? i : a;
            best = s > best ? s : best;
        }
        const unsigned rot = a ? ( ( m >> a ) | ( m << ( nb - a ) ) ) & valid : m;
        const int up   = __ffs( int( ~rot & valid ) ) - 1;                    // the first inside above bit 0
        const int lead = __clz( ~( rot << ( 32 - nb ) ) );                    // the outside ones at the top
        const unsigned run = ( ( 1u << up ) - 1 ) | ( lead ? ( ( 1u << lead ) - 1 ) << ( nb - lead ) : 0u );
        return a ? ( ( run << a ) | ( run >> ( nb - a ) ) ) & valid : run;
    }

    /// ONE CUT ( `filmsk`'s `coupe_msk`, see the old campaign for the derivation ). Returns `false` when there
    /// is nothing left to do: the cell is empty ( `nb == 0` ) or overflowed ( `nb == -1` ).
    __device__ __forceinline__ bool cut( const Plane<TK,TR> &p ) { return cut( p, NoEdgePlanes{} ); }

    /// `edge_plane( cid )`: the plane of an edge, to compute a new vertex as the intersection of two planes rather
    /// than by interpolation along the edge ( see `refine` )
    template<class EP>
    __device__ __forceinline__ bool cut( const Plane<TK,TR> &p, const EP &edge_plane ) {
        // ---- the mask of the vertices outside
        unsigned m = 0;
#pragma unroll
        for ( int i = 0; i < R; ++i ) {
            if ( i >= SUR && i >= nb ) break;
            m |= unsigned( p.dx * x[ i ] + p.dy * y[ i ] - p.off > TK( 0 ) ) << i;
        }
        if ( __builtin_expect( ! m, 1 ) )
            return true;
        const unsigned valid = ( 1u << nb ) - 1;
        if ( m == valid ) {
            nb = 0;
            return false;
        }

        // ---- four role masks, one bit each ( the outside range is one cyclic run )
        unsigned prev = ( ( m << 1 ) | ( m >> ( nb - 1 ) ) ) & valid;
        unsigned next = ( ( m >> 1 ) | ( m << ( nb - 1 ) ) ) & valid;
        if ( __builtin_expect( __popc( m & ~prev ) > 1, 0 ) ) {
            // SEVERAL outside runs: the plane passes, in the kernel's float, through vertices that are on it
            // in exact arithmetic ( concurrent bisectors: seeds on a circle around another one ), and their
            // signs are noise. The run that holds the farthest vertex is the cut; the others are on the plane.
            m = main_run( p, m, valid );
            prev = ( ( m << 1 ) | ( m >> ( nb - 1 ) ) ) & valid;
            next = ( ( m >> 1 ) | ( m << ( nb - 1 ) ) ) & valid;
        }
        const unsigned r0 = ~m & next & valid;           // inside, the next one outside  -> `v_j0`
        const unsigned r1 =  m & ~prev;                  // outside, the previous inside  -> `v_i1`
        const unsigned r2 =  m & ~next;                  // outside, the next inside      -> `v_j2`
        const unsigned r3 = ~m & prev & valid;           // inside, the previous outside  -> `v_j3`
        const int j0 = __ffs( int( r0 ) ) - 1, i1 = __ffs( int( r1 ) ) - 1;
        const int j2 = __ffs( int( r2 ) ) - 1, j3 = __ffs( int( r3 ) ) - 1;
        const int nb_out = __popc( m );
        const int nn = nb - nb_out + 2;
        if ( nn > R ) {
            nb = -1;                                     // for the next pass
            return false;
        }

        // ---- the four vertices, one loop bounded by `nb`, the compares shared by the selects
        TK x0v = x[ 0 ], y0v = y[ 0 ], x1v = x[ 0 ], y1v = y[ 0 ];
        TK x2v = x[ 0 ], y2v = y[ 0 ], x3v = x[ 0 ], y3v = y[ 0 ];
        TR bid = c[ 0 ], aid = c[ 0 ];
#pragma unroll
        for ( int i = 1; i < R; ++i ) {
            if ( i >= SUR && i >= nb ) break;
            x0v = i == j0 ? x[ i ] : x0v; y0v = i == j0 ? y[ i ] : y0v; aid = i == j0 ? c[ i ] : aid;
            x1v = i == i1 ? x[ i ] : x1v; y1v = i == i1 ? y[ i ] : y1v;
            x2v = i == j2 ? x[ i ] : x2v; y2v = i == j2 ? y[ i ] : y2v; bid = i == j2 ? c[ i ] : bid;
            x3v = i == j3 ? x[ i ] : x3v; y3v = i == j3 ? y[ i ] : y3v;
        }

        // ---- the two new points
        const TK s0 = p.dx * x0v + p.dy * y0v - p.off, s1 = p.dx * x1v + p.dy * y1v - p.off;
        const TK s2 = p.dx * x2v + p.dy * y2v - p.off, s3 = p.dx * x3v + p.dy * y3v - p.off;
        const TK ta = s0 / ( s0 - s1 ), tb = s3 / ( s3 - s2 );
        TK pax = x0v + ( x1v - x0v ) * ta, pay = y0v + ( y1v - y0v ) * ta;
        TK pbx = x3v + ( x2v - x3v ) * tb, pby = y3v + ( y2v - y3v ) * tb;
        if constexpr ( ! std::is_same_v<EP,NoEdgePlanes> ) {
            refine( p, edge_plane, aid, pax, pay );
            refine( p, edge_plane, bid, pbx, pby );
        }

        // ---- the reassembly: `u[ k ] = old[ k - 1 ]` ( free: a renaming ), then ONE barrel shift of `e`
        const bool wraps = r1 > r2;                      // `i1 > j2`: the outside range wraps around
        const int  a = wraps ? 0 : i1;
        const int  e = wraps ? j3 - 1 : nb_out - 1;      // `>= 0`
        TK  ux[ R + 1 ], uy[ R + 1 ];
        TR  uc[ R + 1 ];
        ux[ 0 ] = x[ 0 ]; uy[ 0 ] = y[ 0 ]; uc[ 0 ] = c[ 0 ];   // never read: `o + e >= 1`
#pragma unroll
        for ( int o = 1; o < R + 1; ++o ) { ux[ o ] = x[ o - 1 ]; uy[ o ] = y[ o - 1 ]; uc[ o ] = c[ o - 1 ]; }
#pragma unroll
        for ( int b = 1; b < R + 1; b *= 2 ) {
            const bool on = e & b;
#pragma unroll
            for ( int o = 0; o + b < R + 1; ++o ) { ux[ o ] = on ? ux[ o + b ] : ux[ o ]; uy[ o ] = on ? uy[ o + b ] : uy[ o ]; uc[ o ] = on ? uc[ o + b ] : uc[ o ]; }
        }
#pragma unroll
        for ( int o = 0; o < R; ++o ) {
            if ( o >= SUR && o >= nn ) break;
            x[ o ] = o < a ? x[ o ] : ( o == a ? pax  : ( o == a + 1 ? pbx : ux[ o ] ) );
            y[ o ] = o < a ? y[ o ] : ( o == a ? pay  : ( o == a + 1 ? pby : uy[ o ] ) );
            c[ o ] = o < a ? c[ o ] : ( o == a ? p.id : ( o == a + 1 ? bid : uc[ o ] ) );
        }
        nb = nn;
        return true;
    }

    /// `c[ nb - 1 ]` without a dynamic index
    __device__ __forceinline__ TR last_cut() const {
        TR r = c[ 0 ];
#pragma unroll
        for ( int q = 1; q < R; ++q ) r = q == nb - 1 ? c[ q ] : r;
        return r;
    }

    /// visits the vertices in order: `f( i, x, y, c )`
    template<class F>
    __device__ __forceinline__ void for_each_vertex( F &&f ) const {
#pragma unroll
        for ( int i = 0; i < R; ++i ) {
            if ( i >= SUR && i >= nb ) break;
            f( i, x[ i ], y[ i ], c[ i ] );
        }
    }
};

// ---- the cell of a whole warp ( third and fourth passes ) ------------------------------------------------------

/// warp reductions ( sm_75 has no `__reduce_*_sync` )
__device__ __forceinline__ int warp_sum( int v ) {
#pragma unroll
    for ( int o = 16; o; o /= 2 ) v += __shfl_xor_sync( 0xffffffffu, v, o );
    return v;
}
__device__ __forceinline__ int warp_min( int v ) {
#pragma unroll
    for ( int o = 16; o; o /= 2 ) v = min( v, __shfl_xor_sync( 0xffffffffu, v, o ) );
    return v;
}
/// lane `j`'s `v`, for any value made of 32-bit words ( `float4`, `double2`... )
template<class T>
__device__ __forceinline__ T shfl( const T &v, int j ) {
    static_assert( sizeof( T ) % 4 == 0 );
    int w[ sizeof( T ) / 4 ];
    memcpy( w, &v, sizeof( T ) );
#pragma unroll
    for ( int i = 0; i < int( sizeof( T ) / 4 ); ++i )
        w[ i ] = __shfl_sync( 0xffffffffu, w[ i ], j );
    T r;
    memcpy( &r, w, sizeof( T ) );
    return r;
}

/// the index of the largest `s` ( the smallest index among ties ), on every lane
template<class TK>
__device__ __forceinline__ int warp_argmax( TK s, int i ) {
#pragma unroll
    for ( int o = 16; o; o /= 2 ) {
        const TK  so = __shfl_xor_sync( 0xffffffffu, s, o );
        const int io = __shfl_xor_sync( 0xffffffffu, i, o );
        if ( so > s || ( so == s && io < i ) ) { s = so; i = io; }
    }
    return i;
}

/// A CELL HELD BY A WHOLE WARP: the vertices in shared ( third pass ) or global ( fourth pass ) memory, each lane in
/// charge of one vertex in 32. What a lone thread does in a chain of dependent reads ( the pruning test and the cut
/// both run over all the vertices, several times per cut ), the warp does in one step and a few shuffles: the cells
/// that get here have 16 vertices or more, hundreds to thousands of cuts. Every lane runs the same walk on the same
/// data ( the planes, the nodes: broadcast reads ); the results the walk branches on are reduced over the warp, so all
/// the lanes take the same branches.
template<class TK,class TR>
struct WarpCell {
    TK  *x[ 2 ], *y[ 2 ], *s;
    TR  *c[ 2 ];
    int  cur, cap, nb, lane;

    __device__ __forceinline__ TK  X( int i ) const { return x[ cur ][ i ]; }
    __device__ __forceinline__ TK  Y( int i ) const { return y[ cur ][ i ]; }
    __device__ __forceinline__ TR  C( int i ) const { return c[ cur ][ i ]; }

    /// the five rows of `cap` reals then the two rows of `cap` identifiers, at `base` ( 16-byte aligned )
    __device__ __forceinline__ void attach( unsigned char *base, int cap_ ) {
        cap = cap_;
        TK *ts = reinterpret_cast<TK *>( base );
        TR *cs = reinterpret_cast<TR *>( base + SI( 5 ) * cap * sizeof( TK ) );
        for ( int q = 0; q < 2; ++q ) {
            x[ q ] = ts + SI( q ) * cap;
            y[ q ] = ts + SI( 2 + q ) * cap;
            c[ q ] = cs + SI( q ) * cap;
        }
        s = ts + SI( 4 ) * cap;
    }
    static constexpr SI bytes_for( int cap ) { return ( SI( cap ) * SI( 5 * sizeof( TK ) + 2 * sizeof( TR ) ) + 15 ) / 16 * 16; }

    __device__ __forceinline__ void init( TK x0, TK y0, TK x1, TK y1 ) {
        cur = 0;
        if ( lane < 4 ) {
            x[ 0 ][ lane ] = lane == 1 || lane == 2 ? x1 : x0;
            y[ 0 ][ lane ] = lane == 2 || lane == 3 ? y1 : y0;
            c[ 0 ][ lane ] = TR( -1 - lane );
        }
        nb = 4;
        __syncwarp();
    }

    template<bool W>
    __device__ __forceinline__ bool may_be_cut_by( const CBox &B ) const {
        bool res = false;
        for ( int i = lane; i < nb; i += 32 )
            res |= vertex_may_go<W>( B, X( i ), Y( i ) );
        return __any_sync( 0xffffffffu, res );
    }

    __device__ __forceinline__ bool cut( const Plane<TK,TR> &p ) {
        // the signed distances, kept for the interpolations; how many outside; the farthest
        int nb_out = 0, a = 0;
        TK best = TK( 0 );
        bool first = true;
        for ( int i = lane; i < nb; i += 32 ) {
            const TK si = p.dx * X( i ) + p.dy * Y( i ) - p.off;
            s[ i ] = si;
            nb_out += si > TK( 0 );
            if ( first || si > best ) { best = si; a = i; first = false; }
        }
        nb_out = warp_sum( nb_out );
        if ( nb_out == 0 )
            return true;
        if ( nb_out == nb ) {
            nb = 0;
            return false;
        }
        a = warp_argmax( first ? -TK( 1e30 ) : best, first ? nb : a );
        __syncwarp();

        // the outside run `[ b, e ]` around the farthest vertex ( one run, see `RegCell::main_run` ): the nearest
        // inside vertices before and after it
        int db = nb, df = nb;
        for ( int i = lane; i < nb; i += 32 ) {
            if ( ! ( s[ i ] > TK( 0 ) ) ) {
                db = min( db, ( a - i + nb ) % nb );
                df = min( df, ( i - a + nb ) % nb );
            }
        }
        db = warp_min( db );
        df = warp_min( df );
        const int b = ( a - db + 1 + nb ) % nb, e = ( a + df - 1 ) % nb;
        const int nin = nb - ( db + df - 1 );
        if ( nin + 2 > cap ) {
            nb = -1;
            return false;
        }

        // the new cell: the inside vertices from `e + 1` on, then A ( on the edge that enters the run ), then B
        // ( on the edge that leaves it )
        const int nxt = 1 - cur;
        for ( int t = lane; t < nin; t += 32 ) {
            const int q = ( e + 1 + t ) % nb;
            x[ nxt ][ t ] = X( q );
            y[ nxt ][ t ] = Y( q );
            c[ nxt ][ t ] = C( q );
        }
        if ( lane == 0 ) {
            const int j0 = ( b - 1 + nb ) % nb, j3 = ( e + 1 ) % nb;
            const TK s0 = s[ j0 ], s1 = s[ b ], s2 = s[ e ], s3 = s[ j3 ];
            const TK ta = s0 / ( s0 - s1 ), tb = s3 / ( s3 - s2 );
            x[ nxt ][ nin ] = X( j0 ) + ( X( b ) - X( j0 ) ) * ta;
            y[ nxt ][ nin ] = Y( j0 ) + ( Y( b ) - Y( j0 ) ) * ta;
            c[ nxt ][ nin ] = p.id;
            x[ nxt ][ nin + 1 ] = X( j3 ) + ( X( e ) - X( j3 ) ) * tb;
            y[ nxt ][ nin + 1 ] = Y( j3 ) + ( Y( e ) - Y( j3 ) ) * tb;
            c[ nxt ][ nin + 1 ] = C( e );
        }
        __syncwarp();
        cur = nxt;
        nb = nin + 2;
        return true;
    }

    /// A LEAF AT ONCE: the lanes read its seeds together ( one read each, where the walk read them one after
    /// the other, each a full memory latency for a cell that is alone on its warp ), the cuts take them from
    /// the lanes in rank order. Returns `false` when the walk must stop.
    template<class Pb>
    __device__ __forceinline__ bool cut_leaf( const Pb &pb, const Frame<TK> &f, TR beg, TR end ) {
        for ( TR q0 = beg; q0 < end; q0 += 32 ) {
            typename Pb::Pos P{};
            typename Pb::Wt  Q{};
            if ( q0 + lane < end ) {
                P = pb.pos[ q0 + lane ];
                Q = pb.weight( q0 + lane );
            }
            const int nq = int( min( TR( 32 ), TR( end - q0 ) ) );
            for ( int j = 0; j < nq; ++j ) {
                const typename Pb::Pos Pj = shfl( P, j );
                const typename Pb::Wt  Qj = shfl( Q, j );
                if ( ! cut( bisector<Pb::W>( f, Pj, Qj, TR( q0 + j ) ) ) )
                    return false;
            }
        }
        return true;
    }

    __device__ __forceinline__ TR last_cut() const { return C( nb - 1 ); }

    template<class F>
    __device__ __forceinline__ void for_each_vertex( F &&f ) const {
        for ( int i = 0; i < nb; ++i )
            f( i, X( i ), Y( i ), C( i ) );
    }
};

// ---- the cells of the first pass, kept for the finish ( float kernel ) ---------------------------------------

/// WHERE THE FIRST PASS LEAVES ITS CELLS for `finish_pass`, SoA ( vertex `i` of cell `k` at `i * n + k`: the
/// threads of a warp write consecutive addresses ). The double re-solve of the float kernel needs some seventy
/// registers that the walk does not, and a kernel is sized on its worst moment: in the same kernel it halves
/// the occupancy of the walk ( 127 registers against 57 ); in a kernel of its own it costs ~100 bytes per
/// cell written and read once.
template<class TR>
struct Deferred {
    float *x, *y;
    TR    *c;
    int   *nb;
    SI     n;
};

/// a cell of `Deferred`, seen as a cell ( `nb`, `last_cut`, `for_each_vertex` )
template<class TR>
struct DeferredCell {
    const Deferred<TR> &d;
    SI                  k;
    int                 nb;

    __device__ __forceinline__ TR last_cut() const { return d.c[ SI( nb - 1 ) * d.n + k ]; }

    template<class F>
    __device__ __forceinline__ void for_each_vertex( F &&f ) const {
        for ( int i = 0; i < nb; ++i )
            f( i, d.x[ SI( i ) * d.n + k ], d.y[ SI( i ) * d.n + k ], d.c[ SI( i ) * d.n + k ] );
    }
};

// ---- the walk ----------------------------------------------------------------------------------------------

/// THE DESCENT, depth first, the nearest child first ( `ProviderBsp::next`, in one piece ): a node is pruned
/// at POP time, against the cell as it is then; a leaf hands its seeds to `cell.cut` in rank order. Stops as
/// soon as the cell says so ( empty, or overflowed ).
///
/// THE STACK, BY HEIGHT: going down from a node of height `h`, the farther child waits at height `h - 1` and the
/// walk goes on in the nearer one, whose own waiting children are lower still. So the waiting nodes have distinct
/// heights, decreasing from the bottom of the stack to its top: `stack[ h ]` holds the one of height `h`, the bit
/// `h` of `pending` says it is there, and the top is the LOWEST pending height. Nothing to pack, nothing to bound
/// but the height itself ( `MAX_HEIGHT` ).
template<class Pb,class Cell,class EP = NoEdgePlanes>
__device__ __forceinline__ void walk( const Pb &pb, const Frame<typename Pb::TK> &f, Cell &cell, const EP &edge_plane = {} ) {
    using TN   = typename Pb::TN;
    using TR   = typename Pb::TR;
    using Mask = typename Pb::V::Mask;
    const Frame<float> pf = prune_frame( f );
    TN   stack[ Pb::V::MAX_HEIGHT ];
    Mask pending = 0;
    TN   n = 0;
    int  h = pb.depth;                                   // the root: node 0, height `depth`
    while ( true ) {
        const Node<Pb::W,TR> nd = pb.nodes[ n ];
        // an empty slot ( `beg == end` ), or a subtree that can no longer reach the cell
        if ( nd.beg < nd.end && cell.template may_be_cut_by<Pb::W>( centred( nd, pf ) ) ) {
            if ( h <= 1 ) {                              // a leaf. No "is it me" test: my plane is `0 <= 0`
                if constexpr ( requires { cell.cut_leaf( pb, f, TR( 0 ), TR( 0 ) ); } ) {
                    if ( ! cell.cut_leaf( pb, f, nd.beg, nd.end ) )
                        return;
                } else {
                    for ( TR q = nd.beg; q < nd.end; ++q ) {
                        const Plane<typename Pb::TK,TR> p = bisector<Pb::W>( f, pb.pos[ q ], pb.weight( q ), q );
                        if constexpr ( std::is_same_v<EP,NoEdgePlanes> ) {
                            if ( ! cell.cut( p ) )
                                return;
                        } else {
                            if ( ! cell.cut( p, edge_plane ) )
                                return;
                        }
                    }
                }
            } else {
                // the nearest child is visited at once ( the cell will not change before: no push, no pop ),
                // the other one waits at its height and is tested when it comes out
                const TN lc = n + 1, rc = n + ( TN( 1 ) << ( h - 1 ) );
                const Box bl = *reinterpret_cast<const Box *>( pb.nodes + lc );
                const Box br = *reinterpret_cast<const Box *>( pb.nodes + rc );
                const bool left_first = proximity( bl, pf ) <= proximity( br, pf );
                --h;
                stack[ h ] = left_first ? rc : lc;
                pending |= Mask( 1 ) << h;
                n = left_first ? lc : rc;
                continue;
            }
        }
        if ( ! pending )
            return;
        if constexpr ( sizeof( Mask ) == 4 ) h = __ffs( int( pending ) ) - 1;
        else                                 h = __ffsll( ( long long ) pending ) - 1;
        pending &= pending - 1;
        n = stack[ h ];
    }
}

// ---- the end of a cell: its re-solved edges, and what is made of them -------------------------------------------

/// the seed and the box sides in double, in the seed's frame: what the re-solve reads
struct Origin { double x, y, w, x0, y0, x1, y1; };

template<class Pb>
__device__ __forceinline__ Origin origin_of( const Pb &pb, typename Pb::TR k ) {
    Origin o;
    o.x = double( pb.pos64( k, 0 ) );
    o.y = double( pb.pos64( k, 1 ) );
    if constexpr ( Pb::W ) o.w = double( pb.w64( k ) ); else o.w = 0;
    o.x0 = double( pb.box_min( 0 ) ) - o.x;
    o.y0 = double( pb.box_min( 1 ) ) - o.y;
    o.x1 = double( pb.box_max( 0 ) ) - o.x;
    o.y1 = double( pb.box_max( 1 ) ) - o.y;
    return o;
}

/// THE PLANE OF A CUT, RE-READ in double from its identifier: the bisector from the positions as given ( the
/// rank is the identifier ), or a side of the box. `e`: `|n|^2`, the squared distance of the two seeds for a bisector.
template<class Pb>
__device__ __forceinline__ void plane64( const Pb &pb, const Origin &o, typename Pb::TR cid, double &nx, double &ny, double &off, double &e ) {
    if ( cid >= 0 ) {
        nx  = double( pb.pos64( cid, 0 ) ) - o.x;
        ny  = double( pb.pos64( cid, 1 ) ) - o.y;
        e   = nx * nx + ny * ny;
        off = 0.5 * e;
        if constexpr ( Pb::W )
            off += 0.5 * ( o.w - double( pb.w64( cid ) ) );
        return;
    }
    const int f = int( -1 - cid );                       // 0 bottom, 1 right, 2 top, 3 left
    const bool vert = f & 1;
    nx  = vert ? 1.0 : 0.0;
    ny  = vert ? 0.0 : 1.0;
    e   = 1.0;
    off = f == 0 ? o.y0 : f == 1 ? o.x1 : f == 2 ? o.y1 : o.x0;
}

/// the vertex on planes `a` ( the edge that arrives ) and `b` ( the one that leaves ); `false` if the system is
/// too badly conditioned to gain anything ( the kernel's vertex is then kept )
__device__ __forceinline__ bool cross( double ax, double ay, double ao, double ea, double bx, double by, double bo, double eb,
                                       double &vx, double &vy, double &det2, double &e12 ) {
    const double det = ax * by - ay * bx;
    det2 = det * det;
    e12  = ea * eb;
    if ( ! ( det2 > DET_MIN * DET_MIN * e12 ) )
        return false;
    const double inv = __drcp_rn( det );                 // one reciprocal, no division ( a division is a subroutine )
    vx = ( ao * by - bo * ay ) * inv;
    vy = ( ax * bo - bx * ao ) * inv;
    return true;
}

/// THE EDGES OF A FINISHED CELL, in order, in double and in the seed's frame: `f( ax, ay, bx, by, id, d2 )` for the
/// edge from `a` to `b` carried by the cut `id`, `d2` the squared distance between the two seeds ( a bisector ).
/// Float kernel: every vertex re-solved in double from its two planes, as it comes; double kernel: the vertices.
template<bool WANT_D2,class Pb,class Cell,class F>
__device__ __forceinline__ void for_each_edge( const Pb &pb, const Origin &o, const Cell &cell, F &&f ) {
    using TR = typename Pb::TR;
    if ( cell.nb < 3 )
        return;
    double fx = 0, fy = 0, px = 0, py = 0, pe = 1;
    TR pc = 0;
    if constexpr ( std::is_same_v<typename Pb::TK,float> ) {
        // a re-solved vertex is only trusted NEAR the float one -- near meaning what the float vertex can be
        // off by, `eps L / sin( angle of the two planes )`, with a margin of a thousand: where the float decided
        // a topology that the double would not, the intersection of two planes that are consecutive in float
        // only can be anywhere, and the float vertex is right to its own precision
        //
        // The scale of that error is not the cell's: a float vertex is interpolated from vertices that were
        // once those of the domain box, so it carries `eps L`, `L` the farthest corner of the box from the seed.
        const double L2 = fmax( o.x0 * o.x0, o.x1 * o.x1 ) + fmax( o.y0 * o.y0, o.y1 * o.y1 );
        const double tol2 = 1e-8 * L2;                       // ( 1e3 eps_float L )^2
        double ax, ay, ao, ea;
        plane64( pb, o, cell.last_cut(), ax, ay, ao, ea );
        cell.for_each_vertex( [&]( int i, float x, float y, TR c ) {
            double bx, by, bo, eb, vx, vy, det2, e12;
            plane64( pb, o, c, bx, by, bo, eb );
            if ( ! cross( ax, ay, ao, ea, bx, by, bo, eb, vx, vy, det2, e12 ) || ( ( vx - x ) * ( vx - x ) + ( vy - y ) * ( vy - y ) ) * det2 > tol2 * e12 ) {
                // the float vertex, PROJECTED on the plane of the edge that leaves it: two planes nearly parallel
                // ( a seed duplicated with another weight ) leave the vertex free ALONG them -- where moving it
                // changes no area -- and the float error across them is the one that counts
                const double r = eb > 0 ? ( bo - ( bx * x + by * y ) ) / eb : 0.0;
                vx = x + r * bx;
                vy = y + r * by;
            }
            ax = bx; ay = by; ao = bo; ea = eb;
            if ( i == 0 ) { fx = vx; fy = vy; }
            else f( px, py, vx, vy, pc, pe );
            px = vx; py = vy; pc = c; pe = eb;
        } );
    } else {
        cell.for_each_vertex( [&]( int i, double x, double y, TR c ) {
            if ( i == 0 ) { fx = x; fy = y; }
            else f( px, py, x, y, pc, pe );
            px = x; py = y; pc = c;
            if constexpr ( WANT_D2 ) {
                if ( c >= 0 ) {
                    const typename Pb::Pos q = pb.pos[ c ];
                    const double dx = q.x - o.x, dy = q.y - o.y;
                    pe = dx * dx + dy * dy;
                } else
                    pe = 1;
            }
        } );
    }
    f( px, py, fx, fy, pc, pe );
}

/// `n` slots of the COO for the calling thread, ONE atomic per group of lanes that get here together ( the lanes of
/// `__activemask`, each one's offset the sum of the lower lanes' -- a loop over the active lanes: the cooperative
/// groups' scan would do the same, and their headers do not match the pinned `nvcc` ). Past the COO's capacity the
/// count goes on: it is what tells loom how much was wanted.
__device__ __forceinline__ unsigned long long reserve( unsigned long long *count, unsigned n ) {
    unsigned mask;
    asm volatile( "activemask.b32 %0;" : "=r"( mask ) );   // `__activemask()`: its intrinsic does not resolve with the pinned `nvcc`
    const int lane = threadIdx.x % 32, leader = __ffs( int( mask ) ) - 1;
    unsigned tot = 0, off = 0;
    for ( unsigned m = mask; m; m &= m - 1 ) {
        const int src = __ffs( int( m ) ) - 1;
        const unsigned v = __shfl_sync( mask, n, src );
        tot += v;
        off += src < lane ? v : 0u;
    }
    unsigned long long base = 0;
    if ( lane == leader )
        base = atomicAdd( count, ( unsigned long long ) tot );
    return __shfl_sync( mask, base, leader ) + off;
}

/// THE END OF A CELL: everything the call wants of it ( `Pb::OUT` ), from its edges
template<class Pb,class Cell>
__device__ __forceinline__ void finish_cell( const Pb &pb, typename Pb::TR k, const Cell &cell ) {
    using TR = typename Pb::TR;
    using TF = std::remove_reference_t<decltype( pb.res( 0 ) )>;
    constexpr unsigned OUT = Pb::OUT;
    constexpr bool EDGE_TERMS = OUT & ( FACETS | VJP );
    const Origin o = origin_of( pb, k );
    const double rho = pb.density();

    // the facets: the upper ones only ( the neighbour's rank above the cell's ), whose number the cuts give
    unsigned long long fbase = 0;
    bool fok = false;
    if constexpr ( bool( OUT & FACETS ) ) {
        unsigned nup = 0;
        if ( cell.nb >= 3 )
            cell.for_each_vertex( [&]( int, auto, auto, TR c ) { nup += c > k; } );
        fbase = reserve( &pb.counters->nb_facets, nup );
        fok = fbase + nup <= pb.fcap;                    // else the count says it to loom ( `report_facets` )
    }
    double gk = 0;
    if constexpr ( bool( OUT & VJP ) )
        gk = double( pb.g( SI( pb.ids( k ) ) ) );
    if constexpr ( bool( OUT & EDGES ) ) {
        const SI n = SI( pb.n );
        int ne = cell.nb < 3 ? 0 : cell.nb;
        if ( ne > EDGE_CAP )
            ne = -1;
        else
            cell.for_each_vertex( [&]( int i, auto, auto, TR c ) { pb.edges[ SI( i ) * n + SI( k ) ] = c; } );
        pb.nb_edges[ k ] = ne;
    }

    double a2 = 0, m1x = 0, m1y = 0, m2 = 0, gw = 0, gpx = 0, gpy = 0;
    unsigned t = 0;
    for_each_edge<EDGE_TERMS>( pb, o, cell, [&]( double ax, double ay, double bx, double by, TR c, double d2 ) {
        const double cr = ax * by - bx * ay;
        a2 += cr;
        if constexpr ( bool( OUT & MOMENTS ) ) {
            m1x += cr * ( ax + bx );
            m1y += cr * ( ay + by );
            m2  += cr * ( ax * ax + ax * bx + bx * bx + ay * ay + ay * by + by * by );
        }
        if constexpr ( EDGE_TERMS ) {
            if ( c >= 0 ) {
                const double ex = bx - ax, ey = by - ay;
                const double coef = d2 > 0 ? rho * sqrt( ex * ex + ey * ey ) / ( 2 * sqrt( d2 ) ) : 0.0;
                if constexpr ( bool( OUT & FACETS ) ) {
                    if ( c > k && fok ) {
                        const unsigned long long q = fbase + t++;
                        pb.fi[ q ] = k;
                        pb.fj[ q ] = c;
                        pb.fc[ q ] = coef;
                    }
                }
                if constexpr ( bool( OUT & VJP ) ) {
                    // `2 c ( g_k - g_j ) ( x_kj - p_k )`, the middle of the facet in the seed's frame being `( a + b ) / 2`
                    const double dg = coef * ( gk - double( pb.g( SI( pb.ids( c ) ) ) ) );
                    gw  += dg;
                    gpx += dg * ( ax + bx );
                    gpy += dg * ( ay + by );
                }
            }
        }
    } );

    const double area = 0.5 * fabs( a2 );
    if constexpr ( bool( OUT & MEASURES ) )
        pb.res( pb.user( k ) ) = TF( rho * area );
    if constexpr ( bool( OUT & VJP ) ) {
        if ( pb.grad_pos.p ) {
            pb.grad_pos( SI( k ), 0 ) = TF( gpx );
            pb.grad_pos( SI( k ), 1 ) = TF( gpy );
        }
        if constexpr ( Pb::W )
            if ( pb.grad_w.p )
                pb.grad_w( SI( k ) ) = TF( gw );
    }
    if constexpr ( bool( OUT & MOMENTS ) ) {
        // the triangles ( seed, a, b ): `int x = |T| ( a + b ) / 3`, `int |x|^2 = |T| ( a.a + a.b + b.b ) / 6`, `|T| = cr / 2`,
        // in the seed's frame -- so the second moment IS the cost of the cell
        const SI u = pb.user( k );
        const double sg = a2 < 0 ? -1.0 : 1.0;
        pb.bary( u, 0 ) = TF( area > 0 ? o.x + sg * m1x / ( 6 * area ) : o.x );
        pb.bary( u, 1 ) = TF( area > 0 ? o.y + sg * m1y / ( 6 * area ) : o.y );
        pb.cost( u ) = TF( rho * sg * m2 / 12 );
    }
}

/// A CELL THAT COULD NOT BE DONE: zeros where it writes ( never a NaN: the error buffer says why )
template<class Pb>
__device__ __forceinline__ void finish_failed( const Pb &pb, typename Pb::TR k ) {
    using TF = std::remove_reference_t<decltype( pb.res( 0 ) )>;
    constexpr unsigned OUT = Pb::OUT;
    if constexpr ( bool( OUT & MEASURES ) )
        pb.res( pb.user( k ) ) = TF( 0 );
    if constexpr ( bool( OUT & EDGES ) )
        pb.nb_edges[ k ] = -1;
    if constexpr ( bool( OUT & VJP ) ) {
        if ( pb.grad_pos.p ) { pb.grad_pos( SI( k ), 0 ) = TF( 0 ); pb.grad_pos( SI( k ), 1 ) = TF( 0 ); }
        if constexpr ( Pb::W ) if ( pb.grad_w.p ) pb.grad_w( SI( k ) ) = TF( 0 );
    }
    if constexpr ( bool( OUT & MOMENTS ) ) {
        const SI u = pb.user( k );
        pb.bary( u, 0 ) = TF( 0 ); pb.bary( u, 1 ) = TF( 0 ); pb.cost( u ) = TF( 0 );
    }
}

// ---- the kernels -----------------------------------------------------------------------------------------

/// the seed's frame in the kernel's float, and the domain box in that frame
template<class Pb>
__device__ __forceinline__ void start_of( const Pb &pb, typename Pb::TR k, Frame<typename Pb::TK> &f, typename Pb::TK ( &b )[ 4 ] ) {
    using TK = typename Pb::TK;
    if constexpr ( Pb::W ) f = frame_of( pb.pos[ k ], pb.w[ k ] );
    else                   f = frame_of( pb.pos[ k ], typename Pb::Wt{} );
    const double px = double( pb.pos64( k, 0 ) ), py = double( pb.pos64( k, 1 ) );
    b[ 0 ] = TK( double( pb.box_min( 0 ) ) - px );
    b[ 1 ] = TK( double( pb.box_min( 1 ) ) - py );
    b[ 2 ] = TK( double( pb.box_max( 0 ) ) - px );
    b[ 3 ] = TK( double( pb.box_max( 1 ) ) - py );
}

/// `deferred`: the cell is left in it for `finish_pass` instead of being finished here ( `nullptr`: finished here ).
/// `false`: the cell overflowed `R` vertices, nothing was written
template<int R,class Pb>
__device__ __forceinline__ bool cell_in_registers( const Pb &pb, typename Pb::TR k, const Deferred<typename Pb::TR> *deferred = nullptr ) {
    using TK = typename Pb::TK;
    using TR = typename Pb::TR;
    Frame<TK> f;
    TK b[ 4 ];
    start_of( pb, k, f, b );
    RegCell<TK,TR,R> cell;
    cell.init( b[ 0 ], b[ 1 ], b[ 2 ], b[ 3 ] );
    // the plane of an edge: a bisector re-read from its seed, or a side of the box ( `-1` bottom, `-2` right,
    // `-3` top, `-4` left )
    auto edge_plane = [&]( TR cid ) {
        if ( cid >= 0 )
            return bisector<Pb::W>( f, pb.pos[ cid ], pb.weight( cid ), cid );
        const int s = int( -1 - cid );
        Plane<TK,TR> e;
        e.dx = s & 1 ? TK( 1 ) : TK( 0 );
        e.dy = s & 1 ? TK( 0 ) : TK( 1 );
        e.off = s == 0 ? b[ 1 ] : s == 1 ? b[ 2 ] : s == 2 ? b[ 3 ] : b[ 0 ];
        e.id = cid;
        return e;
    };
    if constexpr ( EXACT_VERTICES )
        walk( pb, f, cell, edge_plane );
    else
        walk( pb, f, cell );
    if ( deferred ) {
        if constexpr ( std::is_same_v<TK,float> ) {
            if ( cell.nb > RD )                          // more than the finish holds: a later pass
                cell.nb = -1;
            deferred->nb[ k ] = cell.nb;
            cell.for_each_vertex( [&]( int i, float x, float y, TR c ) {
                deferred->x[ SI( i ) * deferred->n + k ] = x;
                deferred->y[ SI( i ) * deferred->n + k ] = y;
                deferred->c[ SI( i ) * deferred->n + k ] = c;
            } );
        }
    }
    if ( cell.nb < 0 )
        return false;
    if ( ! deferred )
        finish_cell( pb, k, cell );
    return true;
}

template<class Pb>
constexpr bool deferring() { return std::is_same_v<typename Pb::TK,float>; }

/// FIRST PASS: one thread per cell, rank order ( two neighbouring threads are neighbours in the tree, hence in
/// space: their walks look alike, which is the only thing that limits the divergence -- and it is free )
template<class Pb>
__global__ void __launch_bounds__( BLOCK, MINB1 ) first_pass( Pb pb, typename Pb::TR *ovf_list, Deferred<typename Pb::TR> deferred ) {
    using TR = typename Pb::TR;
    const SI k = SI( blockIdx.x ) * blockDim.x + threadIdx.x;
    if ( k < SI( pb.n ) && ! cell_in_registers<R1>( pb, TR( k ), deferring<Pb>() ? &deferred : nullptr ) )
        ovf_list[ atomicAdd( &pb.counters->ovf[ 0 ], 1ull ) ] = TR( k );
}

/// THE FINISH of the cells the first two passes left in `deferred` ( float kernel )
template<class Pb>
__global__ void __launch_bounds__( BLOCK ) finish_pass( Pb pb, Deferred<typename Pb::TR> deferred ) {
    using TR = typename Pb::TR;
    const SI k = SI( blockIdx.x ) * blockDim.x + threadIdx.x;
    if ( k >= SI( pb.n ) )
        return;
    const int nb = deferred.nb[ k ];
    if ( nb < 0 )                                        // a later pass has it
        return;
    finish_cell( pb, TR( k ), DeferredCell<TR>{ deferred, k, nb } );
}

/// SECOND PASS: the cells the first one could not hold, `R2` registers. Its grid is what the card holds at
/// once and it strides over a count only the card knows: no read back between the passes. ( Tried in rank
/// order instead, one thread per rank and a flag per rank, so that a warp holds neighbouring cells only:
/// 5.4 ms instead of 2.3 on the uniform cloud -- most warps then hold one or two cells. )
template<class Pb>
__global__ void __launch_bounds__( BLOCK ) second_pass( Pb pb, const typename Pb::TR *list, typename Pb::TR *ovf_list, Deferred<typename Pb::TR> deferred ) {
    const unsigned long long m = pb.counters->ovf[ 0 ];
    for ( unsigned long long i = SI( blockIdx.x ) * blockDim.x + threadIdx.x; i < m; i += SI( gridDim.x ) * blockDim.x )
        if ( ! cell_in_registers<R2>( pb, list[ i ], deferring<Pb>() ? &deferred : nullptr ) )
            ovf_list[ atomicAdd( &pb.counters->ovf[ 1 ], 1ull ) ] = list[ i ];
}

/// THE WARP PASSES: what the registers of one thread could not hold, ONE CELL PER WARP ( `WarpCell` ). The cells that
/// get here have nothing in common any more ( not neighbours in the tree ): one per thread, a warp of 32 of them ran
/// them one after the other ( 7 ms for the 50 of the lines cloud ), and one per warp on a single lane each was a chain
/// of dependent reads ( 0.5 to 5 million cycles ). Launched without a read back: the grid is what the card holds, the
/// count is read on the card.
///
/// THIRD PASS: the vertices in SHARED memory, `shared_cap< TK >()` per warp ( 48 KB per block of four warps ).
/// FOURTH PASS ( `slots != nullptr` ): the vertices in GLOBAL memory, one slot of `cap` vertices ( the per-cell limit )
/// per warp of the grid ( `WarpCell::bytes_for( cap )` bytes each, `Overflow` ), the grid being exactly the slots: each
/// warp takes the cells `gwarp, gwarp + nb_warps, ...` one after the other in its slot -- batches of `nb_warps` cells,
/// as many as the list needs. A cell that does not fit in its slot has more than the limit: a FAILURE.
template<class TK>
constexpr int shared_cap() { return sizeof( TK ) == 4 ? 384 : 256; }

template<class TK,class TR,int CAP>
constexpr int warp_bytes() { return ( BLOCK / 32 ) * int( WarpCell<TK,TR>::bytes_for( CAP ) ); }

template<class Pb,class EB>
__global__ void __launch_bounds__( BLOCK ) warp_pass( Pb pb, const typename Pb::TR *list, int in_list, typename Pb::TR *ovf_list,
                                                     unsigned char *slots, int cap, EB errors ) {
    using TK = typename Pb::TK;
    using TR = typename Pb::TR;
    extern __shared__ __align__( 16 ) unsigned char shared_bytes_[];
    const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    const SI  gwarp = ( SI( blockIdx.x ) * blockDim.x + threadIdx.x ) / 32;
    unsigned char *mine = slots ? slots + gwarp * WarpCell<TK,TR>::bytes_for( cap )
                                : shared_bytes_ + SI( warp ) * WarpCell<TK,TR>::bytes_for( cap );
    const unsigned long long m = pb.counters->ovf[ in_list ];
    for ( unsigned long long i = gwarp; i < m; i += SI( gridDim.x ) * blockDim.x / 32 ) {
        const TR k = list[ i ];
        Frame<TK> f;
        TK b[ 4 ];
        start_of( pb, k, f, b );
        WarpCell<TK,TR> cell;
        cell.attach( mine, cap );
        cell.lane = lane;
        cell.init( b[ 0 ], b[ 1 ], b[ 2 ], b[ 3 ] );
        walk( pb, f, cell );
        if ( lane == 0 ) {
            if ( cell.nb >= 0 )
                finish_cell( pb, k, cell );
            else if ( ! slots )                          // the shared pass: the global one has it
                ovf_list[ atomicAdd( &pb.counters->ovf[ 2 ], 1ull ) ] = k;
            else {                                       // the global pass: past the per-cell limit, a failure said as such
                finish_failed( pb, k );
                if ( atomicAdd( &pb.counters->nb_failed, 1ull ) == 0 )
                    errors.record( ERROR_KIND_FAILURE, FAIL_TOO_MANY_VERTICES, SI( pb.ids( k ) ) );   // the user's index, whatever the order of the outputs
            }
        }
        __syncwarp();
    }
}

/// THE KERNEL'S TREE, from the tree's tensors ( one thread per node ). In float the box is rounded OUTWARD and
/// the majorant constant UP: a pruning made on the rounded node is still a pruning of the true one.
template<bool W,class TR,class TB,class TJ>
__global__ void __launch_bounds__( BLOCK ) make_nodes( SI nb_nodes, Strided<TB,3> box, Strided<TB,2> wa, Strided<TB,1> wb,
                                                      Strided<TJ,1> beg, Strided<TJ,1> end, Node<W,TR> *out ) {
    const SI i = SI( blockIdx.x ) * blockDim.x + threadIdx.x;
    if ( i >= nb_nodes )
        return;
    auto down = []( TB v ) -> float { if constexpr ( ! std::is_same_v<TB,float> ) return __double2float_rd( v ); else return v; };
    auto up   = []( TB v ) -> float { if constexpr ( ! std::is_same_v<TB,float> ) return __double2float_ru( v ); else return v; };
    Node<W,TR> nd;
    for ( int d = 0; d < 2; ++d ) {
        nd.lo[ d ] = down( box( i, 0, d ) );
        nd.hi[ d ] = up  ( box( i, 1, d ) );
    }
    if constexpr ( W ) {
        // the slopes are only used to bound: rounded, they give another valid majorant once the constant is
        // taken above it -- the constant already carries a margin far above these roundings
        nd.a[ 0 ] = float( wa( i, 0 ) );
        nd.a[ 1 ] = float( wa( i, 1 ) );
        nd.b = up( wb( i ) );
    }
    nd.beg = TR( beg( i ) );
    nd.end = TR( end( i ) );
    out[ i ] = nd;
}

/// THE KERNEL'S SEEDS: two floats per coordinate ( and per weight ) for the float kernel, the doubles as such for
/// the double one
template<class TK,bool W,class TF>
__global__ void __launch_bounds__( BLOCK ) pack_seeds( SI n, Strided<TF,2> pos, Strided<TF,1> w,
                                                      typename KernelSeeds<TK>::Pos *pos_out, typename KernelSeeds<TK>::Wt *w_out ) {
    const SI k = SI( blockIdx.x ) * blockDim.x + threadIdx.x;
    if ( k >= n )
        return;
    const double x = double( pos( k, 0 ) ), y = double( pos( k, 1 ) );
    if constexpr ( std::is_same_v<TK,float> ) {
        const float xh = __double2float_rn( x ), yh = __double2float_rn( y );
        pos_out[ k ] = make_float4( xh, yh, __double2float_rn( x - double( xh ) ), __double2float_rn( y - double( yh ) ) );
        if constexpr ( W ) {
            const double v = double( w( k ) );
            const float vh = __double2float_rn( v );
            w_out[ k ] = make_float2( vh, __double2float_rn( v - double( vh ) ) );
        }
    } else {
        pos_out[ k ] = make_double2( x, y );
        if constexpr ( W )
            w_out[ k ] = double( w( k ) );
    }
}

/// `count` into a loom ShapeVar ( on the card: the capacity check and the error record are the ShapeVar's own )
template<class SV>
__global__ void report_count( SV sv, const unsigned long long *count, unsigned long long factor ) {
    sv.set( SI( *count * factor ) );
}

// ---- the host side -----------------------------------------------------------------------------------------

/// `nb_bytes` of the call's pool, 16-byte aligned ( `nullptr` if the pool said no: the handler reports it )
inline void *take( auto &allocator, SI nb_bytes ) {
    auto v = allocator.template view<std::int32_t>( ( nb_bytes + 16 + 3 ) / 4 );
    auto p = reinterpret_cast<std::uintptr_t>( v.data().raw );
    if ( ! p )
        return nullptr;
    return reinterpret_cast<void *>( ( p + 15 ) & ~std::uintptr_t( 15 ) );
}

inline int blocks_for( SI n ) { return int( ( n + BLOCK - 1 ) / BLOCK ); }

/// THE FOURTH PASS'S BUDGET: `warps` slots of `cap` vertices in global memory, taken ONCE per call and reused by every
/// diagram of the call ( the cards of a solve share it: they run one after the other on the stream ). `cap` is the
/// per-cell limit, `warps` Python's choice ( `PowerDiagram_Bsp.card_overflow_warps_for` ): what does not fit in the slots
/// at once waits for a slot to be free ( `warp_pass` ), it is never a reason to run again.
struct Overflow {
    unsigned char *slots = nullptr;
    SI             bytes = 0;                            ///< what `slots` holds
    int            cap = 0, warps = 0;

    /// the slots' geometry for `n` seeds: `cap = min( max_vertices, n + 4 )` ( a cell of `n` seeds in a box has at most
    /// `n + 3` vertices ), `warps` rounded up to whole blocks
    static Overflow sized( SI n, int warps, int max_vertices ) {
        Overflow o;
        o.cap   = int( std::max<SI>( 4, std::min<SI>( max_vertices, n + 4 ) ) );
        o.warps = ( std::max( 1, warps ) + BLOCK / 32 - 1 ) / ( BLOCK / 32 ) * ( BLOCK / 32 );
        return o;
    }
    template<class TK,class TR>
    SI bytes_for() const { return WarpCell<TK,TR>::bytes_for( cap ) * warps; }

    /// room for the slots of a `< TK, TR >` cell from the pool ( `false`: the pool said no )
    template<class TK,class TR>
    bool take_from( auto &allocator ) {
        bytes = bytes_for<TK,TR>();
        slots = static_cast<unsigned char *>( take( allocator, bytes ) );
        return slots != nullptr;
    }
};

/// THE DIAGRAM ON THE CARD, for one call: the kernel's tree and seeds, the lists and counters of the passes, the
/// global-memory slots of the fourth pass. `prepare` takes it all from the call's allocator and fills the tree and
/// the seeds; `run` launches the passes -- everything `pb` points to ( the outputs ) is the caller's. Nothing is read
/// back ( but with `SDOT_CARD_STATS=1` ). This is the object a solver keeps across its diagrams ( `SdotPlanNd`, step 2 ):
/// `prepare` again when the weights change ( it rebuilds the nodes from the tree's majorants ), `run` per diagram.
template<class V,bool W,unsigned OUT,class TF,class TI>
struct Card {
    using Pb  = Problem<V,TF,TI,W,OUT>;
    using TK  = typename V::TK;
    using TR  = typename V::TR;
    using Pos = typename Pb::Pos;
    using Wt  = typename Pb::Wt;

    Pb             pb{};
    TR            *lists = nullptr;                      ///< two lists of `n` ranks
    Counters      *counters = nullptr;
    Deferred<TR>   deferred{};
    Overflow       overflow{};                           ///< the fourth pass's slots
    SI             nb_nodes = 0;

    /// `overflow`: the slots of the fourth pass, already taken ( shared with another card of the call: they must hold
    /// this card's cells, `bytes_for< TK, TR >` ), or only sized ( `slots == nullptr` ): taken here. `false`: the pool
    /// said no.
    bool prepare( const CudaQueue &queue, const auto &pd, auto &allocator, Overflow overflow_ ) {
        static_assert( std::decay_t<decltype( pd )>::ct_dim == 2, "the dedicated GPU cell is 2D" );
        const SI n = SI( pd.nb_seeds() );
        nb_nodes = SI( pd.tree.node_begin.shape( 0 ) );
        int depth = 0;
        for ( SI m = nb_nodes; m; m >>= 1 )
            ++depth;
        if ( depth > V::MAX_HEIGHT || ( std::is_same_v<typename V::TN,int> && nb_nodes > SI( 0x7fffffff ) )
                                   || ( std::is_same_v<TR,int> && n > SI( 0x7fffffff ) - 8 ) )
            throw std::runtime_error( "sdot::gpu2d: the variant chosen does not hold this tree ( see `PowerDiagram_Bsp._card_variant` )" );
        overflow = overflow_;
        if ( overflow.slots && overflow.bytes < overflow.template bytes_for<TK,TR>() )
            throw std::runtime_error( "sdot::gpu2d: shared overflow slots too small for this card" );
        if ( ! overflow.slots && ! overflow.template take_from<TK,TR>( allocator ) )
            return false;

        auto *nodes = static_cast<Node<W,TR> *>( take( allocator, SI( sizeof( Node<W,TR> ) ) * nb_nodes ) );
        auto *pos   = static_cast<Pos *>( take( allocator, SI( sizeof( Pos ) ) * n ) );
        Wt   *w     = nullptr;
        if constexpr ( W )
            w = static_cast<Wt *>( take( allocator, SI( sizeof( Wt ) ) * n ) );
        lists    = static_cast<TR *>( take( allocator, SI( sizeof( TR ) ) * 2 * n ) );
        counters = static_cast<Counters *>( take( allocator, SI( sizeof( Counters ) ) ) );
        if ( ! nodes || ! pos || ( W && ! w ) || ! lists || ! counters )
            return false;
        deferred = Deferred<TR>{ nullptr, nullptr, nullptr, nullptr, n };
        if constexpr ( std::is_same_v<TK,float> ) {
            deferred.x  = static_cast<float *>( take( allocator, SI( sizeof( float ) ) * RD * n ) );
            deferred.y  = static_cast<float *>( take( allocator, SI( sizeof( float ) ) * RD * n ) );
            deferred.c  = static_cast<TR *>( take( allocator, SI( sizeof( TR ) ) * RD * n ) );
            deferred.nb = static_cast<int *>( take( allocator, SI( sizeof( int ) ) * n ) );
            if ( ! deferred.x || ! deferred.y || ! deferred.c || ! deferred.nb )
                return false;
        }

        pb.nodes   = nodes;
        pb.pos     = pos;
        pb.w       = w;
        pb.pos64   = strided( pd.sorted_positions );
        if constexpr ( W )
            pb.w64 = strided( pd.sorted_weights );
        pb.box_min = strided( pd.box_min );
        pb.box_max = strided( pd.box_max );
        pb.ids     = strided( pd.tree.seed_indices );
        pb.n       = TR( n );
        pb.depth   = depth;
        pb.rho     = 1;
        pb.user_order = true;
        pb.counters = counters;
        refresh( queue, pd );
        return true;
    }

    /// the kernel's tree and seeds from the diagram's tensors ( again after the weights changed )
    void refresh( const CudaQueue &queue, const auto &pd ) {
        using TB = std::remove_const_t<typename std::decay_t<decltype( pd.tree.node_box )>::TF>;
        using TJ = std::remove_const_t<typename std::decay_t<decltype( pd.tree.node_begin )>::TF>;
        static_assert( std::is_same_v<TB,TF>, "the tree's boxes and the positions share the driver's float" );
        Strided<TF,2> wa{};
        Strided<TF,1> wb{};
        if constexpr ( W ) {
            wa = strided( pd.tree.node_wa );
            wb = strided( pd.tree.node_wb );
            pb.w64 = strided( pd.sorted_weights );
        }
        launch_kernel( queue, &make_nodes<W,TR,TB,TJ>, blocks_for( nb_nodes ), BLOCK, 0,
                       nb_nodes, strided( pd.tree.node_box ), wa, wb, strided( pd.tree.node_begin ), strided( pd.tree.node_end ),
                       const_cast<Node<W,TR> *>( pb.nodes ) );
        launch_kernel( queue, &pack_seeds<TK,W,TF>, blocks_for( SI( pb.n ) ), BLOCK, 0, SI( pb.n ), pb.pos64, pb.w64,
                       const_cast<Pos *>( pb.pos ), const_cast<Wt *>( pb.w ) );
    }

    /// THE PASSES. `errors`: the kernel form of loom's error buffer ( what a failure is recorded in ).
    template<class EB>
    void run( const CudaQueue &queue, const EB &errors ) {
        const SI n = SI( pb.n );
        zero_fill( queue, counters, SI( sizeof( Counters ) ) );
        if ( n == 0 )
            return;

        // first pass, second pass ( no read back between them )
        TR *list1 = lists, *list2 = lists + n;
        launch_kernel( queue, &first_pass<Pb>, blocks_for( n ), BLOCK, 0, pb, list1, deferred );
        static const int grid2 = resident_grid( &second_pass<Pb>, BLOCK );
        launch_kernel( queue, &second_pass<Pb>, grid2, BLOCK, 0, pb, list1, list2, deferred );

        // third pass, one warp per cell in shared memory ( its overflow goes to `list1`, free again )
        constexpr int CAP = shared_cap<TK>();
        auto *k3 = &warp_pass<Pb,EB>;
        constexpr int bytes3 = warp_bytes<TK,TR,CAP>();
        static const int grid3 = [&] {
            cuda_check( cudaFuncSetAttribute( k3, cudaFuncAttributeMaxDynamicSharedMemorySize, bytes3 ), "shared memory of the warp pass" );
            return resident_grid( k3, BLOCK, bytes3 );
        }();
        launch_kernel( queue, k3, grid3, BLOCK, bytes3, pb, list2, 1, list1, ( unsigned char * ) nullptr, CAP, errors );

        if constexpr ( deferring<Pb>() )
            launch_kernel( queue, &finish_pass<Pb>, blocks_for( n ), BLOCK, 0, pb, deferred );

        // fourth pass, one warp per cell in global memory, a persistent grid of exactly the slots: launched whatever the
        // count ( it reads it ), it costs a launch when there is nothing to do
        launch_kernel( queue, k3, overflow.warps * 32 / BLOCK, BLOCK, 0, pb, list1, 2, list2, overflow.slots, overflow.cap, errors );

        static const bool stats = std::getenv( "SDOT_CARD_STATS" ) && *std::getenv( "SDOT_CARD_STATS" ) != '0';
        if ( stats ) {                                   // how many cells each pass left over ( a diagnosis: a read back )
            Counters c;
            read_back( queue, &c, counters, 1 );
            std::printf( "[card cells] n %lld, over %d registers %llu ( %.2f %% ), over %d registers %llu ( %.3f %% ), over %d shared %llu "
                         "( %d slots of %d vertices ), failed %llu, facets %llu\n",
                         ( long long ) n, R1, c.ovf[ 0 ], 100.0 * c.ovf[ 0 ] / n, R2, c.ovf[ 1 ], 100.0 * c.ovf[ 1 ] / n, CAP, c.ovf[ 2 ],
                         overflow.warps, overflow.cap, c.nb_failed, c.nb_facets );
        }
    }

    /// the upper facets the cells wanted, times `factor`, written into a loom ShapeVar ON THE CARD ( `set` checks the
    /// capacity and records the overflow: loom runs the call again with more room )
    void report_facets( const CudaQueue &queue, const auto &sv, unsigned long long factor ) const {
        launch_kernel( queue, &report_count<std::decay_t<decltype( sdot::kernel_form( queue, MutList(), sv ) )>>, 1, 1, 0,
                       sdot::kernel_form( queue, MutList(), sv ), &counters->nb_facets, factor );
    }
};

/// the types a diagram's tensors give
template<class PD> using TFOf = typename PD::TF;
template<class PD> using TIOf = std::remove_const_t<typename std::decay_t<decltype( std::declval<PD>().tree.seed_indices )>::TF>;

/// the constant density: a number, or a 0-d tensor of the call ( read on the card: its value is not a compile-time
/// constant of the kernel )
template<class Pb>
void set_density( Pb &pb, const auto &density ) {
    if constexpr ( requires { density.data().raw; } ) {
        pb.rho_dev = reinterpret_cast<decltype( pb.rho_dev )>( density.data().raw );
        pb.rho = 1;
    } else {
        pb.rho_dev = nullptr;
        pb.rho = double( density );
    }
}

// ---- the entry points of the loom calls ( `PowerDiagram_Bsp.py` ) -------------------------------------------------
//
// `max_vertices` ( the per-cell limit ) and `overflow_warps` ( the fourth pass's slots ) are Python's constants
// ( `card_max_vertices`, `card_overflow_warps_for` ): the forward and the backward use the same scheme.

/// THE MEASURES of `pd` on the call's stream; `rho`: the constant density.
template<class V>
void measures( const CudaQueue &queue, const auto &pd, auto &&res, const auto &errors, auto &allocator, const auto &rho,
               int max_vertices, int overflow_warps ) {
    using PD = std::decay_t<decltype( pd )>;
    Card<V,PD::has_weights,MEASURES,TFOf<PD>,TIOf<PD>> card;
    if ( ! card.prepare( queue, pd, allocator, Overflow::sized( SI( pd.nb_seeds() ), overflow_warps, max_vertices ) ) )
        return;                                          // the pool said no: `allocator.failed` is reported
    set_density( card.pb, rho );
    card.pb.res = strided_out<TFOf<PD>,1>( res );
    card.run( queue, sdot::kernel_form( queue, MutList(), errors ) );
}

/// THE ADJOINT OF THE MEASURES: `grad_res` ( user order ) -> the gradients of the sorted positions and weights. The
/// same passes and the same fourth-pass budget as the forward.
template<class V>
void measures_vjp( const CudaQueue &queue, const auto &pd, const auto &grad_res, auto &&grad_pos, auto &&grad_w, const auto &errors,
                   auto &allocator, const auto &rho, int max_vertices, int overflow_warps ) {
    using PD = std::decay_t<decltype( pd )>;
    using TF = TFOf<PD>;
    constexpr bool W = PD::has_weights;
    constexpr bool has_gp = requires { grad_pos.data().raw; }, has_gw = requires { grad_w.data().raw; };
    if constexpr ( ! has_gp && ! has_gw ) {
        return;
    } else if constexpr ( ! requires { grad_res.data().raw; } ) {
        // a symbolic zero cotangent: zero gradients ( loom seeds the outputs, but say it rather than assume it )
        if constexpr ( has_gp ) zero_fill( queue, const_cast<void *>( ( const void * ) grad_pos.data().raw ), SI( sizeof( TF ) ) * 2 * SI( pd.nb_seeds() ) );
        if constexpr ( has_gw ) zero_fill( queue, const_cast<void *>( ( const void * ) grad_w.data().raw ), SI( sizeof( TF ) ) * SI( pd.nb_seeds() ) );
    } else {
        Card<V,W,VJP,TF,TIOf<PD>> card;
        if ( ! card.prepare( queue, pd, allocator, Overflow::sized( SI( pd.nb_seeds() ), overflow_warps, max_vertices ) ) )
            return;
        set_density( card.pb, rho );
        card.pb.g        = strided( grad_res );
        card.pb.grad_pos = strided_out<TF,2>( grad_pos );
        if constexpr ( W )
            card.pb.grad_w = strided_out<TF,1>( grad_w );
        card.run( queue, sdot::kernel_form( queue, MutList(), errors ) );
    }
}

} // namespace sdot::gpu2d
