#pragma once

// =====================================================================================
// THE 2D CELLS OF THE CARD: `PowerDiagram.measures` on a CUDA device, in 2D, with the BSP tree.
//
// A dedicated path, written for the card and nothing else ( `PowerDiagram_Bsp._measures_on_card` decides when
// it applies; everything else -- 3D, moments, derivatives, facets, a distribution, a domain that is not a box,
// the neighbour memory -- keeps the generic path of `diagram/Ops.h` ). It is the old GPU campaign's kernel
// ( `nsdot/gpu_des_familles`, `FilMsk2D.cuh` + `Arbre.cuh` + `Mesures.cu` ) brought to this code base:
//
//   * ONE THREAD PER CELL, the cell in REGISTERS: `R1 = 8` vertices in three arrays with immediate indices only
//     ( the masks and the single barrel shift of `filmsk`: `RegCell::cut` ), starting from the domain box. No
//     scratch at all.
//   * THE OVERFLOW IS REDONE BY LATER PASSES, launched without reading any count back ( each strides over a list
//     whose length only the card knows ): the cells that needed more than `R1` vertices at some point ( a tenth
//     of them ) by a second pass with `R2 = 16` registers, what overflows that ( a thousandth ) by a third pass
//     with ONE WARP PER CELL ( `WarpCell`, the vertices in shared memory: those cells are big, few and unrelated,
//     a thread each ran them one after the other ), and what overflows that ( none on the campaign's clouds ) by
//     a fourth pass in global memory, sized on a count read back, grown until nothing overflows.
//   * THE FLOAT KERNEL'S FINISH IN A KERNEL OF ITS OWN ( `finish_pass` ): the cells of the first two passes are
//     left in global memory and re-solved there -- the double arithmetic of the re-solve cost the walk half its
//     occupancy when it was in the same kernel.
//   * THE TREE NODES as ONE aligned record in FLOAT ( `Node`: box rounded outward, majorant slopes and constant
//     rounded up, slice ), rebuilt from the tree's tensors at every call ( `make_nodes`: ~1 % of the call, and
//     always in sync with the weights ); the tree is the same perfect binary tree in preorder as `ProviderBsp`
//     walks ( left child `n + 1`, right child `n + 2^( h - 1 )` ), so a node does not store its children. The
//     pruning is in float for both kernels ( the double one with a margin: `vertex_may_go` ).
//   * THE ACCURACY FIXES of the old campaign ( its `doc/04-echelle.md`, "les trois reparations" + "la quatrieme" ):
//       - the cell lives in the SEED's frame ( vertices counted from `p0` ), so the bisector is `|d|^2 / 2`
//         and nothing large is subtracted to make something of the cell's size;
//       - the positions reach a `float` kernel as TWO floats ( `x = xh + xl`, 48 bits ): `dx = ( xh_q - xh_0 ) +
//         ( xl_q - xl_0 )` is the double difference rounded once, in single precision arithmetic only ( the old
//         kernel used 64-bit fixed point for the same effect ); the WEIGHTS likewise, so the plane carries the
//         weight DIFFERENCE, never a weight rounded on its own;
//       - at the end every vertex is RE-SOLVED in double from the two planes that carry it ( re-read from the
//         positions as given, by rank ), and the area is taken on these re-solved vertices, streaming ( the
//         shoelace closes on the first one ): the float only decides WHICH cuts apply.
//   * an int32 stack of 48 entries, node index and height packed ( the old `PILE = 48` ).
//
// The double kernel ( `TK = double` ) is the same code without the two-float split, the finish pass and the
// re-solve. Diagnosis: `SDOT_CARD_STATS=1` prints how many cells each pass left over.
// =====================================================================================

#include <loom/support/kernels/CudaQueue.h>
#include <loom/support/common_types.h>
#include <loom/support/Ct.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <type_traits>
#include <utility>

namespace sdot::gpu2d {

constexpr int    BLOCK      = 128;                       ///< threads per block, every kernel here
constexpr int    STACK      = 48;                        ///< the walk's stack ( depth + 1 entries at most )
constexpr int    H_BITS     = 5;                         ///< the height, packed under the node index
constexpr int    MAX_DEPTH  = 27;                        ///< node indices fit in 32 - H_BITS bits
constexpr int    R1         = 8;                         ///< vertices in registers, first pass
constexpr int    R2         = 16;                        ///< ... second pass
constexpr int    RD         = 12;                        ///< vertex slots per cell left for the finish ( float kernel )
constexpr int    MINB1      = 1;                         ///< resident blocks per SM asked of the first pass ( `__launch_bounds__` )
constexpr double DET_MIN    = 1e-6;                      ///< under this relative determinant, the float vertex is kept
/// the new vertices of the first and second passes from their two planes ( `refine` ) rather than interpolated:
/// float against double max 1.1e-8 -> 4.6e-14 on the uniform cloud, 2.3e-9 -> 8e-14 on the lines, for +16 %
/// of time ( 9.4 -> 10.9 ns/seed ); the median is 1e-14 either way. Off: the speed is what the card is for.
constexpr bool   EXACT_VERTICES = false;

// ---- the kernel's float: how a seed is stored for it, and the seed's frame ---------------------------------

template<class TK> struct KernelSeeds;
template<> struct KernelSeeds<float>  { using Pos = float4;  using Wt = float2; };   ///< ( xh, yh, xl, yl ), ( wh, wl )
template<> struct KernelSeeds<double> { using Pos = double2; using Wt = double; };

/// the seed of the cell, as the cut sees it
template<class TK> struct Frame;
template<> struct Frame<float>  { float  xh, yh, xl, yl, wh, wl; };
template<> struct Frame<double> { double x, y, w; };

/// a half-space `dx . v <= off`, `v` counted from the seed; `id`: a rank ( >= 0 ) or a side of the box ( < 0 )
template<class TK>
struct Plane { TK dx, dy, off; int id; };

__device__ __forceinline__ Frame<float>  frame_of( float4 p, float2 w )   { return { p.x, p.y, p.z, p.w, w.x, w.y }; }
__device__ __forceinline__ Frame<double> frame_of( double2 p, double w )  { return { p.x, p.y, w }; }

/// THE POWER BISECTOR in the seed's frame: `|x - p0|^2 - w0 <= |x - q|^2 - wq` is `d . v <= |d|^2 / 2 + ( w0 - wq ) / 2`
/// with `d = q - p0`. In float, `d` is the difference of the two-float positions: the high parts first ( exact
/// as soon as the seeds are close, and rounded relatively to `d` otherwise ), then the low parts.
template<bool W>
__device__ __forceinline__ Plane<float> bisector( const Frame<float> &f, float4 q, float2 wq, int id ) {
    Plane<float> p;
    p.dx  = ( q.x - f.xh ) + ( q.z - f.xl );
    p.dy  = ( q.y - f.yh ) + ( q.w - f.yl );
    p.off = 0.5f * ( p.dx * p.dx + p.dy * p.dy );
    if constexpr ( W )
        p.off += 0.5f * ( ( f.wh - wq.x ) + ( f.wl - wq.y ) );
    p.id  = id;
    return p;
}

template<bool W>
__device__ __forceinline__ Plane<double> bisector( const Frame<double> &f, double2 q, double wq, int id ) {
    Plane<double> p;
    p.dx  = q.x - f.x;
    p.dy  = q.y - f.y;
    p.off = 0.5 * ( p.dx * p.dx + p.dy * p.dy );
    if constexpr ( W )
        p.off += 0.5 * ( f.w - wq );
    p.id  = id;
    return p;
}

// ---- the tree, as the kernel reads it --------------------------------------------------------------------

/// ONE NODE, one aligned record in the kernel's float: the threads of a warp read DIFFERENT nodes ( the walks
/// diverge ), so what costs is the number of transactions per node, and a 16-byte aligned record is read in
/// 16-byte loads. 32 bytes for a float Voronoi node, 48 weighted.
template<class TK,bool W> struct alignas( 16 ) Node;
template<class TK> struct alignas( 16 ) Node<TK,false> { TK lo[ 2 ], hi[ 2 ]; int beg, end; };
template<class TK> struct alignas( 16 ) Node<TK,true>  { TK lo[ 2 ], hi[ 2 ], a[ 2 ], b; int beg, end; };

/// the first bytes of a node: what the ordering of two children reads
template<class TK> struct alignas( 16 ) Box { TK lo[ 2 ], hi[ 2 ]; };

/// a node seen from the seed: its box in the seed's frame, and for a weighted diagram the slopes and the
/// constant `c = w0 - b - a . p0` of the majorant ( `w( q ) <= a . q + b` )
template<class TK>
struct CBox { TK lo[ 2 ], hi[ 2 ], a[ 2 ], c; };

template<bool W>
__device__ __forceinline__ CBox<float> centred( const Node<float,W> &nd, const Frame<float> &f ) {
    CBox<float> B;
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
template<bool W,class TK>
__device__ __forceinline__ bool vertex_may_go( const CBox<TK> &B, TK vx, TK vy ) {
    TK yx = vx, yy = vy;
    if constexpr ( W ) {
        yx += TK( 0.5 ) * B.a[ 0 ];
        yy += TK( 0.5 ) * B.a[ 1 ];
    }
    yx = fmin( fmax( yx, B.lo[ 0 ] ), B.hi[ 0 ] );
    yy = fmin( fmax( yy, B.lo[ 1 ] ), B.hi[ 1 ] );
    const TK ux = yx - vx, uy = yy - vy;
    TK s = ux * ux + uy * uy - ( vx * vx + vy * vy );
    if constexpr ( W )
        s += B.c - ( B.a[ 0 ] * yx + B.a[ 1 ] * yy );
    return s <= TK( 0 );
}

/// THE SAME TEST FOR A DOUBLE VERTEX, in float: the pruning only has to be CONSERVATIVE, and the double kernel
/// spent half its time in it at 1/32 of the float rate. The vertex is rounded to float, the test answers "may
/// go" up to a margin of 1e-6 of the magnitude of its terms ( sixteen float epsilons: far above the roundings
/// of the vertex, of the centred box and of the majorant constant ), so that what the float prunes, the double
/// would have pruned too.
template<bool W>
__device__ __forceinline__ bool vertex_may_go( const CBox<float> &B, double vxd, double vyd ) {
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
__device__ __forceinline__ float proximity( const Box<float> &b, const Frame<float> &f ) {
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

template<class View>
Strided<std::remove_const_t<typename View::TF>,View::ct_rank> strided( const View &v ) {
    Strided<std::remove_const_t<typename View::TF>,View::ct_rank> res;
    res.p = reinterpret_cast<const char *>( v.data().raw );
    [&]<int... I>( std::integer_sequence<int,I...> ) {
        ( ( res.s[ I ] = SI( v._strides[ Ct<int,I>() ] ) ), ... );
    }( std::make_integer_sequence<int,View::ct_rank>() );
    return res;
}

/// EVERYTHING A CELL READS, by value ( kernel parameters ): the kernel's tree and seeds, the seeds as given
/// ( for the re-solve ), the domain box and where the measure goes
template<class _TK,class TF,class TI,bool _W>
struct Problem {
    using TK  = _TK;
    using Pos = typename KernelSeeds<TK>::Pos;
    using Wt  = typename KernelSeeds<TK>::Wt;
    static constexpr bool W = _W;

    const Node<float,W> *nodes;                        ///< in float for both kernels: the pruning is in float
    const Pos        *pos;                               ///< the seeds in tree order, in the kernel's form
    const Wt         *w;                                 ///< their weights ( W only )
    Strided<TF,2>     pos64;                             ///< the seeds as given, `[ n, 2 ]`
    Strided<TF,1>     w64;                               ///< the weights as given ( W only )
    Strided<TF,1>     box_min, box_max;                  ///< the domain
    Strided<TI,1>     ids;                               ///< rank -> the user's index
    char             *res;                               ///< the measures, user's order
    SI                res_stride;                        ///< in bytes
    int               n, depth;

    __device__ __forceinline__ Wt   weight( int q ) const { if constexpr ( W ) return w[ q ]; else return Wt{}; }
    __device__ __forceinline__ void write( int k, double m ) const { *reinterpret_cast<TF *>( res + SI( ids( k ) ) * res_stride ) = TF( m ); }
};

// ---- the cell in registers -------------------------------------------------------------------------------

/// no plane to re-read: the new vertices are interpolated along their edge
struct NoEdgePlanes {};

/// A NEW VERTEX FROM ITS TWO PLANES, the cut and the edge it falls on, instead of interpolated along that edge.
/// Interpolated, a vertex inherits the rounding of the edge's ends: `eps L` when the edge spans the domain ( the
/// first cuts make such edges, their ends on the far sides of the box ) even if the vertex lands next to the
/// seed -- and a later cut decided on a vertex off by `eps L` can be missed by that much, a first-order error
/// on the area ( 5e-5 on one cell of the lines cloud ). Intersected, it is off by `eps |v| / sin`. Kept
/// interpolated where the two planes are nearly parallel.
template<class TK,class EP>
__device__ __forceinline__ void refine( const Plane<TK> &p, const EP &edge_plane, int cid, TK &vx, TK &vy ) {
    const Plane<TK> e = edge_plane( cid );
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
template<class TK,int R>
struct RegCell {
    static constexpr int SUR = 3;                        ///< a non-empty cell has three vertices at least
    static_assert( R >= 4 && R < 32, "the box fits in the registers, a mask in 32 bits" );

    TK  x[ R ], y[ R ];
    int c[ R ];
    int nb;

    /// the domain box `[ x0, x1 ] x [ y0, y1 ]` ( seed's frame ); its sides are `-1` bottom, `-2` right,
    /// `-3` top, `-4` left
    __device__ __forceinline__ void init( TK x0, TK y0, TK x1, TK y1 ) {
#pragma unroll
        for ( int i = 0; i < R; ++i ) {
            x[ i ] = i == 1 || i == 2 ? x1 : x0;
            y[ i ] = i == 2 || i == 3 ? y1 : y0;
            c[ i ] = i < 4 ? -1 - i : 0;
        }
        nb = 4;
    }

    template<bool W>
    __device__ __forceinline__ bool may_be_cut_by( const CBox<float> &B ) const {
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
    __device__ __forceinline__ unsigned main_run( const Plane<TK> &p, unsigned m, unsigned valid ) const {
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
    __device__ __forceinline__ bool cut( const Plane<TK> &p ) { return cut( p, NoEdgePlanes{} ); }

    /// `edge_plane( cid, plane )`: the plane of an edge, to compute a new vertex as the intersection of two
    /// planes rather than by interpolation along the edge ( see `refine` )
    template<class EP>
    __device__ __forceinline__ bool cut( const Plane<TK> &p, const EP &edge_plane ) {
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
        int bid = c[ 0 ], aid = c[ 0 ];
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
        int uc[ R + 1 ];
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
    __device__ __forceinline__ int last_cut() const {
        int r = c[ 0 ];
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

// ---- the cell in global memory ( fourth pass ) -------------------------------------------------------------

/// `cap` vertices in two sets of rows of global memory ( the cut writes the other one: Sutherland-Hodgman, one
/// plane ), element `v` of the cell of slot `j` at `v * stride + j`, so that the threads of a warp touch
/// consecutive addresses
template<class TK>
struct MemCell {
    TK  *x[ 2 ], *y[ 2 ];
    int *c[ 2 ];
    int  cur, stride, cap, nb;

    __device__ __forceinline__ TK  X( int i ) const { return x[ cur ][ SI( i ) * stride ]; }
    __device__ __forceinline__ TK  Y( int i ) const { return y[ cur ][ SI( i ) * stride ]; }
    __device__ __forceinline__ int C( int i ) const { return c[ cur ][ SI( i ) * stride ]; }

    __device__ __forceinline__ void init( TK x0, TK y0, TK x1, TK y1 ) {
        cur = 0;
        for ( int i = 0; i < 4; ++i ) {
            x[ 0 ][ SI( i ) * stride ] = i == 1 || i == 2 ? x1 : x0;
            y[ 0 ][ SI( i ) * stride ] = i == 2 || i == 3 ? y1 : y0;
            c[ 0 ][ SI( i ) * stride ] = -1 - i;
        }
        nb = 4;
    }

    template<bool W>
    __device__ __forceinline__ bool may_be_cut_by( const CBox<float> &B ) const {
        for ( int i = 0; i < nb; ++i )
            if ( vertex_may_go<W>( B, X( i ), Y( i ) ) )
                return true;
        return false;
    }

    __device__ __forceinline__ bool cut( const Plane<TK> &p ) {
        int nb_out = 0;
        for ( int i = 0; i < nb; ++i )
            nb_out += p.dx * X( i ) + p.dy * Y( i ) - p.off > TK( 0 );
        if ( nb_out == 0 )
            return true;
        if ( nb_out == nb ) {
            nb = 0;
            return false;
        }
        const int nxt = 1 - cur;
        int o = 0;
        auto put = [&]( TK vx, TK vy, int id ) {
            x[ nxt ][ SI( o ) * stride ] = vx;
            y[ nxt ][ SI( o ) * stride ] = vy;
            c[ nxt ][ SI( o ) * stride ] = id;
            ++o;
        };
        // the outside range is ONE cyclic run `[ b, e ]`: the one that holds the farthest vertex ( several runs
        // mean vertices that are on the plane in exact arithmetic, see `RegCell::main_run` )
        auto s_of = [&]( int i ) { return p.dx * X( i ) + p.dy * Y( i ) - p.off; };
        int a = 0;
        TK best = s_of( 0 );
        for ( int i = 1; i < nb; ++i ) { const TK s = s_of( i ); if ( s > best ) { best = s; a = i; } }
        int b = a, e = a;
        while ( true ) { const int q = b ? b - 1 : nb - 1; if ( q == e || ! ( s_of( q ) > 0 ) ) break; b = q; }
        while ( true ) { const int q = e + 1 < nb ? e + 1 : 0; if ( q == b || ! ( s_of( q ) > 0 ) ) break; e = q; }
        auto out = [&]( int i ) { return b <= e ? i >= b && i <= e : i >= b || i <= e; };
        if ( nb - ( e - b + nb ) % nb - 1 + 2 > cap ) {
            nb = -1;
            return false;
        }
        TK xi = X( 0 ), yi = Y( 0 );
        TK si = s_of( 0 );
        for ( int i = 0; i < nb; ++i ) {
            const int j = i + 1 < nb ? i + 1 : 0;
            const TK xj = X( j ), yj = Y( j );
            const TK sj = p.dx * xj + p.dy * yj - p.off;
            const bool in_i = ! out( i ), in_j = ! out( j );
            if ( in_i )
                put( xi, yi, C( i ) );
            if ( in_i != in_j ) {
                // from the inside vertex, as `RegCell::cut`
                const TK xa = in_i ? xi : xj, ya = in_i ? yi : yj, sa = in_i ? si : sj;
                const TK xb = in_i ? xj : xi, yb = in_i ? yj : yi, sb = in_i ? sj : si;
                const TK t = sa / ( sa - sb );
                put( xa + ( xb - xa ) * t, ya + ( yb - ya ) * t, in_i ? p.id : C( i ) );
            }
            xi = xj; yi = yj; si = sj;
        }
        cur = nxt;
        nb = o;
        return true;
    }

    __device__ __forceinline__ int last_cut() const { return C( nb - 1 ); }

    template<class F>
    __device__ __forceinline__ void for_each_vertex( F &&f ) const {
        for ( int i = 0; i < nb; ++i )
            f( i, X( i ), Y( i ), C( i ) );
    }
};

// ---- the cell of a whole warp ( third pass ) -----------------------------------------------------------------

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

/// A CELL HELD BY A WHOLE WARP: the vertices in shared memory, each lane in charge of one vertex in 32. What a
/// lone thread does in a chain of dependent reads ( the pruning test and the cut both run over all the vertices,
/// several times per cut ), the warp does in one step and a few shuffles: the cells that get here have 16
/// vertices or more, hundreds to thousands of cuts, and one thread took 0.5 to 5 million cycles over them.
/// Every lane runs the same walk on the same data ( the planes, the nodes: broadcast reads ); the results the
/// walk branches on are reduced over the warp, so all the lanes take the same branches.
template<class TK>
struct WarpCell {
    TK  *x[ 2 ], *y[ 2 ], *s;
    int *c[ 2 ];
    int  cur, cap, nb, lane;

    __device__ __forceinline__ TK  X( int i ) const { return x[ cur ][ i ]; }
    __device__ __forceinline__ TK  Y( int i ) const { return y[ cur ][ i ]; }
    __device__ __forceinline__ int C( int i ) const { return c[ cur ][ i ]; }

    __device__ __forceinline__ void init( TK x0, TK y0, TK x1, TK y1 ) {
        cur = 0;
        if ( lane < 4 ) {
            x[ 0 ][ lane ] = lane == 1 || lane == 2 ? x1 : x0;
            y[ 0 ][ lane ] = lane == 2 || lane == 3 ? y1 : y0;
            c[ 0 ][ lane ] = -1 - lane;
        }
        nb = 4;
        __syncwarp();
    }

    template<bool W>
    __device__ __forceinline__ bool may_be_cut_by( const CBox<float> &B ) const {
        bool res = false;
        for ( int i = lane; i < nb; i += 32 )
            res |= vertex_may_go<W>( B, X( i ), Y( i ) );
        return __any_sync( 0xffffffffu, res );
    }

    __device__ __forceinline__ bool cut( const Plane<TK> &p ) {
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
    __device__ __forceinline__ bool cut_leaf( const Pb &pb, const Frame<TK> &f, int beg, int end ) {
        for ( int q0 = beg; q0 < end; q0 += 32 ) {
            typename Pb::Pos P{};
            typename Pb::Wt  Q{};
            if ( q0 + lane < end ) {
                P = pb.pos[ q0 + lane ];
                Q = pb.weight( q0 + lane );
            }
            const int nq = min( 32, end - q0 );
            for ( int j = 0; j < nq; ++j ) {
                const typename Pb::Pos Pj = shfl( P, j );
                const typename Pb::Wt  Qj = shfl( Q, j );
                if ( ! cut( bisector<Pb::W>( f, Pj, Qj, q0 + j ) ) )
                    return false;
            }
        }
        return true;
    }

    __device__ __forceinline__ int last_cut() const { return C( nb - 1 ); }

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
struct Deferred {
    float *x, *y;
    int   *c, *nb;
    int    n;
};

/// a cell of `Deferred`, seen as a cell ( `nb`, `last_cut`, `for_each_vertex` )
struct DeferredCell {
    const Deferred &d;
    int             k, nb;

    __device__ __forceinline__ int last_cut() const { return d.c[ SI( nb - 1 ) * d.n + k ]; }

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
template<class Pb,class Cell,class EP = NoEdgePlanes>
__device__ __forceinline__ void walk( const Pb &pb, const Frame<typename Pb::TK> &f, Cell &cell, const EP &edge_plane = {} ) {
    const Frame<float> pf = prune_frame( f );
    int stack[ STACK ];
    int top = 0;
    int n = 0, h = pb.depth;                             // the root: node 0, height `depth`
    while ( true ) {
        const Node<float,Pb::W> nd = pb.nodes[ n ];
        // an empty slot ( `beg == end` ), or a subtree that can no longer reach the cell
        if ( nd.beg < nd.end && cell.template may_be_cut_by<Pb::W>( centred( nd, pf ) ) ) {
            if ( h <= 1 ) {                              // a leaf. No "is it me" test: my plane is `0 <= 0`
                if constexpr ( requires { cell.cut_leaf( pb, f, 0, 0 ); } ) {
                    if ( ! cell.cut_leaf( pb, f, nd.beg, nd.end ) )
                        return;
                } else {
                    for ( int q = nd.beg; q < nd.end; ++q ) {
                        const Plane<typename Pb::TK> p = bisector<Pb::W>( f, pb.pos[ q ], pb.weight( q ), q );
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
                // the other one waits on the stack and is tested when it comes out
                const int lc = n + 1, rc = n + ( 1 << ( h - 1 ) );
                const Box<float> bl = *reinterpret_cast<const Box<float> *>( pb.nodes + lc );
                const Box<float> br = *reinterpret_cast<const Box<float> *>( pb.nodes + rc );
                const bool left_first = proximity( bl, pf ) <= proximity( br, pf );
                --h;
                stack[ top++ ] = ( ( left_first ? rc : lc ) << H_BITS ) | h;
                n = left_first ? lc : rc;
                continue;
            }
        }
        if ( ! top )
            return;
        const int e = stack[ --top ];
        n = e >> H_BITS;
        h = e & ( ( 1 << H_BITS ) - 1 );
    }
}

// ---- the measure -----------------------------------------------------------------------------------------

/// the seed and the box sides in double, in the seed's frame: what the re-solve reads
struct Origin { double x, y, w, x0, y0, x1, y1; };

template<class Pb>
__device__ __forceinline__ Origin origin_of( const Pb &pb, int k ) {
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
/// rank is the identifier ), or a side of the box
template<class Pb>
__device__ __forceinline__ void plane64( const Pb &pb, const Origin &o, int cid, double &nx, double &ny, double &off, double &e ) {
    if ( cid >= 0 ) {
        nx  = double( pb.pos64( cid, 0 ) ) - o.x;
        ny  = double( pb.pos64( cid, 1 ) ) - o.y;
        e   = nx * nx + ny * ny;                         // `|n|^2`, for the conditioning of the vertices
        off = 0.5 * e;
        if constexpr ( Pb::W )
            off += 0.5 * ( o.w - double( pb.w64( cid ) ) );
        return;
    }
    const int f = -1 - cid;                              // 0 bottom, 1 right, 2 top, 3 left
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

/// THE AREA. Float kernel: every vertex re-solved in double from its two planes, the shoelace taken on them as
/// they come ( two doubles for the previous vertex, two for the first one ). Double kernel: on the vertices.
template<class Pb,class Cell>
__device__ __forceinline__ double area_of( const Pb &pb, int k, const Cell &cell ) {
    if ( cell.nb < 3 )
        return 0;
    double a2 = 0, fx = 0, fy = 0, px = 0, py = 0;
    if constexpr ( std::is_same_v<typename Pb::TK,float> ) {
        // a re-solved vertex is only trusted NEAR the float one -- near meaning what the float vertex can be
        // off by, `eps L / sin( angle of the two planes )`, with a margin of a thousand: where the float decided
        // a topology that the double would not, the intersection of two planes that are consecutive in float
        // only can be anywhere, and the float vertex is right to its own precision
        //
        // The scale of that error is not the cell's: a float vertex is interpolated from vertices that were
        // once those of the domain box, so it carries `eps L`, `L` the farthest corner of the box from the seed.
        const Origin o = origin_of( pb, k );
        const double L2 = fmax( o.x0 * o.x0, o.x1 * o.x1 ) + fmax( o.y0 * o.y0, o.y1 * o.y1 );
        const double tol2 = 1e-8 * L2;                       // ( 1e3 eps_float L )^2
        double ax, ay, ao, ea;
        plane64( pb, o, cell.last_cut(), ax, ay, ao, ea );
        cell.for_each_vertex( [&]( int i, float x, float y, int c ) {
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
            else a2 += px * vy - vx * py;
            px = vx; py = vy;
        } );
    } else {
        cell.for_each_vertex( [&]( int i, double x, double y, int ) {
            if ( i == 0 ) { fx = x; fy = y; }
            else a2 += px * y - x * py;
            px = x; py = y;
        } );
    }
    a2 += px * fy - fx * py;
    return 0.5 * fabs( a2 );
}

// ---- the kernels -----------------------------------------------------------------------------------------

/// the seed's frame in the kernel's float, and the domain box in that frame
template<class Pb>
__device__ __forceinline__ void start_of( const Pb &pb, int k, Frame<typename Pb::TK> &f, typename Pb::TK ( &b )[ 4 ] ) {
    using TK = typename Pb::TK;
    if constexpr ( Pb::W ) f = frame_of( pb.pos[ k ], pb.w[ k ] );
    else                   f = frame_of( pb.pos[ k ], typename Pb::Wt{} );
    const double px = double( pb.pos64( k, 0 ) ), py = double( pb.pos64( k, 1 ) );
    b[ 0 ] = TK( double( pb.box_min( 0 ) ) - px );
    b[ 1 ] = TK( double( pb.box_min( 1 ) ) - py );
    b[ 2 ] = TK( double( pb.box_max( 0 ) ) - px );
    b[ 3 ] = TK( double( pb.box_max( 1 ) ) - py );
}

/// `deferred`: the cell is left in it for `finish_pass` instead of being measured here ( `nullptr`: measured here ).
/// `false`: the cell overflowed `R` vertices, nothing was written
template<int R,class Pb>
__device__ __forceinline__ bool cell_in_registers( const Pb &pb, int k, const Deferred *deferred = nullptr ) {
    using TK = typename Pb::TK;
    Frame<TK> f;
    TK b[ 4 ];
    start_of( pb, k, f, b );
    RegCell<TK,R> cell;
    cell.init( b[ 0 ], b[ 1 ], b[ 2 ], b[ 3 ] );
    // the plane of an edge: a bisector re-read from its seed, or a side of the box ( `-1` bottom, `-2` right,
    // `-3` top, `-4` left )
    auto edge_plane = [&]( int cid ) {
        if ( cid >= 0 )
            return bisector<Pb::W>( f, pb.pos[ cid ], pb.weight( cid ), cid );
        const int s = -1 - cid;
        Plane<TK> e;
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
            if ( cell.nb > RD )                          // more than the finish holds: the memory pass
                cell.nb = -1;
            deferred->nb[ k ] = cell.nb;
            cell.for_each_vertex( [&]( int i, float x, float y, int c ) {
                deferred->x[ SI( i ) * deferred->n + k ] = x;
                deferred->y[ SI( i ) * deferred->n + k ] = y;
                deferred->c[ SI( i ) * deferred->n + k ] = c;
            } );
        }
    }
    if ( cell.nb < 0 )
        return false;
    if ( ! deferred )
        pb.write( k, area_of( pb, k, cell ) );
    return true;
}

/// FIRST PASS: one thread per cell, rank order ( two neighbouring threads are neighbours in the tree, hence in
/// space: their walks look alike, which is the only thing that limits the divergence -- and it is free )
template<class Pb>
__global__ void __launch_bounds__( BLOCK, MINB1 ) first_pass( Pb pb, int *ovf_list, int *ovf_count, Deferred deferred ) {
    const int k = blockIdx.x * blockDim.x + threadIdx.x;
    if ( k < pb.n && ! cell_in_registers<R1>( pb, k, std::is_same_v<typename Pb::TK,float> ? &deferred : nullptr ) )
        ovf_list[ atomicAdd( ovf_count, 1 ) ] = k;
}

/// THE FINISH of the cells the first two passes left in `deferred` ( float kernel ): re-solve, area, write
template<class Pb>
__global__ void __launch_bounds__( BLOCK ) finish_pass( Pb pb, Deferred deferred ) {
    const int k = blockIdx.x * blockDim.x + threadIdx.x;
    if ( k >= pb.n )
        return;
    const int nb = deferred.nb[ k ];
    if ( nb < 0 )                                        // a later pass has it
        return;
    pb.write( k, area_of( pb, k, DeferredCell{ deferred, k, nb } ) );
}

/// SECOND PASS: the cells the first one could not hold, `R2` registers. Its grid is what the card holds at
/// once and it strides over a count only the card knows: no read back between the passes. ( Tried in rank
/// order instead, one thread per rank and a flag per rank, so that a warp holds neighbouring cells only:
/// 5.4 ms instead of 2.3 on the uniform cloud -- most warps then hold one or two cells. )
template<class Pb>
__global__ void __launch_bounds__( BLOCK ) second_pass( Pb pb, const int *list, const int *count, int *ovf_list, int *ovf_count, Deferred deferred ) {
    const int m = *count;
    for ( int i = blockIdx.x * blockDim.x + threadIdx.x; i < m; i += gridDim.x * blockDim.x )
        if ( ! cell_in_registers<R2>( pb, list[ i ], std::is_same_v<typename Pb::TK,float> ? &deferred : nullptr ) )
            ovf_list[ atomicAdd( ovf_count, 1 ) ] = list[ i ];
}

/// THE WARP PASS: what the registers of one thread could not hold, ONE CELL PER WARP ( `WarpCell` ), the
/// vertices in shared memory. The cells that get here have nothing in common any more ( not neighbours in the
/// tree ): one per thread, a warp of 32 of them ran them one after the other ( 7 ms for the 50 of the lines
/// cloud ), and one per warp on a single lane each was a chain of dependent reads ( 0.5 to 5 million cycles ).
/// `CAP = shared_cap< TK >()` vertices ( 48 KB per block of four warps ). Launched without a read back: the
/// grid is what the card holds, the count is read on the card. ( Tried as the second pass too, with 64
/// vertices: 8.5 ms instead of 2.3 on the uniform cloud, the tenth of the cells that gets there being too many
/// for one warp each. )
template<class TK>
constexpr int shared_cap() { return sizeof( TK ) == 4 ? 384 : 256; }

template<class TK,int CAP>
constexpr int warp_bytes() { return ( BLOCK / 32 ) * CAP * int( 5 * sizeof( TK ) + 2 * sizeof( int ) ); }

template<class Pb,int CAP>
__global__ void __launch_bounds__( BLOCK ) warp_pass( Pb pb, const int *list, const int *count, int *ovf_list, int *ovf_count ) {
    using TK = typename Pb::TK;
    extern __shared__ __align__( 16 ) unsigned char shared_bytes_[];
    const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    TK  *ts = reinterpret_cast<TK *>( shared_bytes_ ) + warp * 5 * CAP;                       // x0 x1 y0 y1 s
    int *cs = reinterpret_cast<int *>( shared_bytes_ + ( BLOCK / 32 ) * 5 * CAP * sizeof( TK ) ) + warp * 2 * CAP;
    const int m = *count;
    for ( int i = ( blockIdx.x * blockDim.x + threadIdx.x ) / 32; i < m; i += gridDim.x * blockDim.x / 32 ) {
        const int k = list[ i ];
        Frame<TK> f;
        TK b[ 4 ];
        start_of( pb, k, f, b );
        WarpCell<TK> cell;
        for ( int q = 0; q < 2; ++q ) {
            cell.x[ q ] = ts + q * CAP;
            cell.y[ q ] = ts + ( 2 + q ) * CAP;
            cell.c[ q ] = cs + q * CAP;
        }
        cell.s = ts + 4 * CAP;
        cell.cap = CAP;
        cell.lane = lane;
        cell.init( b[ 0 ], b[ 1 ], b[ 2 ], b[ 3 ] );
        walk( pb, f, cell );
        if ( lane == 0 ) {
            if ( cell.nb < 0 )
                ovf_list[ atomicAdd( ovf_count, 1 ) ] = k;
            else
                pb.write( k, area_of( pb, k, cell ) );
        }
        __syncwarp();
    }
}

/// FOURTH PASS: what is left, in global memory, `cap` vertices per cell; `m` is known on the host
template<class Pb>
__global__ void __launch_bounds__( BLOCK ) memory_pass( Pb pb, const int *list, int m, int cap, typename Pb::TK *xs, typename Pb::TK *ys, int *cs,
                                                       int *ovf_list, int *ovf_count, bool last ) {
    using TK = typename Pb::TK;
    const int j = ( blockIdx.x * blockDim.x + threadIdx.x ) / 32;   // one cell per warp, see `warp_pass`
    if ( threadIdx.x % 32 || j >= m )
        return;
    const int k = list[ j ];
    Frame<TK> f;
    TK b[ 4 ];
    start_of( pb, k, f, b );
    MemCell<TK> cell;
    const SI rows = SI( cap ) * m;
    cell.x[ 0 ] = xs + j; cell.x[ 1 ] = xs + rows + j;
    cell.y[ 0 ] = ys + j; cell.y[ 1 ] = ys + rows + j;
    cell.c[ 0 ] = cs + j; cell.c[ 1 ] = cs + rows + j;
    cell.stride = m;
    cell.cap = cap;
    cell.init( b[ 0 ], b[ 1 ], b[ 2 ], b[ 3 ] );
    walk( pb, f, cell );
    if ( cell.nb < 0 ) {
        if ( last )                                      // no room left to grow: say it rather than lie
            pb.write( k, __longlong_as_double( 0x7ff8000000000000ll ) );
        else
            ovf_list[ atomicAdd( ovf_count, 1 ) ] = k;
        return;
    }
    pb.write( k, area_of( pb, k, cell ) );
}

/// THE KERNEL'S TREE, from the tree's tensors ( one thread per node ). In float the box is rounded OUTWARD and
/// the majorant constant UP: a pruning made on the rounded node is still a pruning of the true one.
template<class TK,bool W,class TB,class TI>
__global__ void __launch_bounds__( BLOCK ) make_nodes( int nb_nodes, Strided<TB,3> box, Strided<TB,2> wa, Strided<TB,1> wb,
                                                      Strided<TI,1> beg, Strided<TI,1> end, Node<float,W> *out ) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if ( i >= nb_nodes )
        return;
    auto down = []( TB v ) -> TK { if constexpr ( std::is_same_v<TK,float> && ! std::is_same_v<TB,float> ) return __double2float_rd( v ); else return TK( v ); };
    auto up   = []( TB v ) -> TK { if constexpr ( std::is_same_v<TK,float> && ! std::is_same_v<TB,float> ) return __double2float_ru( v ); else return TK( v ); };
    Node<float,W> nd;
    for ( int d = 0; d < 2; ++d ) {
        nd.lo[ d ] = down( box( i, 0, d ) );
        nd.hi[ d ] = up  ( box( i, 1, d ) );
    }
    if constexpr ( W ) {
        nd.a[ 0 ] = TK( wa( i, 0 ) );
        nd.a[ 1 ] = TK( wa( i, 1 ) );
        nd.b = up( wb( i ) );
    }
    nd.beg = int( beg( i ) );
    nd.end = int( end( i ) );
    out[ i ] = nd;
}

/// THE KERNEL'S SEEDS: two floats per coordinate ( and per weight ) for the float kernel, the doubles as such for
/// the double one
template<class TK,bool W,class TF>
__global__ void __launch_bounds__( BLOCK ) pack_seeds( int n, Strided<TF,2> pos, Strided<TF,1> w,
                                                      typename KernelSeeds<TK>::Pos *pos_out, typename KernelSeeds<TK>::Wt *w_out ) {
    const int k = blockIdx.x * blockDim.x + threadIdx.x;
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

// ---- the handler -------------------------------------------------------------------------------------------

/// `nb_bytes` of the call's pool, 16-byte aligned ( `nullptr` if the pool said no: the handler reports it )
inline void *take( auto &allocator, SI nb_bytes ) {
    auto v = allocator.template view<std::int32_t>( ( nb_bytes + 16 + 3 ) / 4 );
    auto p = reinterpret_cast<std::uintptr_t>( v.data().raw );
    if ( ! p )
        return nullptr;
    return reinterpret_cast<void *>( ( p + 15 ) & ~std::uintptr_t( 15 ) );
}

/// `true` if this call can take the dedicated path ( checked again by the caller, `PowerDiagram_Bsp.py` )
inline bool supports_depth( int depth ) { return depth <= MAX_DEPTH; }

/// THE MEASURES OF `pd` INTO `res`, on the call's stream. `TK`: the kernel's float.
template<class TK>
void measures( const CudaQueue &queue, const auto &pd, auto &&res, auto &allocator ) {
    using PD = std::decay_t<decltype( pd )>;
    using TF = typename PD::TF;
    using TI = std::remove_const_t<typename std::decay_t<decltype( pd.tree.seed_indices )>::TF>;
    constexpr bool W = PD::has_weights;
    static_assert( PD::ct_dim == 2, "the dedicated GPU cell is 2D" );
    using Pb  = Problem<TK,TF,TI,W>;
    using Pos = typename Pb::Pos;
    using Wt  = typename Pb::Wt;

    const int n = int( pd.nb_seeds() );
    const int nb_nodes = int( pd.tree.node_begin.shape( 0 ) );
    if ( n == 0 )
        return;
    int depth = 0;
    for ( int m = nb_nodes; m; m >>= 1 )
        ++depth;

    // what the call allocates: the kernel's tree and seeds, two lists of ranks, the counters
    Pb pb{};
    auto *nodes = static_cast<Node<float,W> *>( take( allocator, SI( sizeof( Node<float,W> ) ) * nb_nodes ) );
    auto *pos   = static_cast<Pos *>( take( allocator, SI( sizeof( Pos ) ) * n ) );
    Wt   *w     = nullptr;
    if constexpr ( W )
        w = static_cast<Wt *>( take( allocator, SI( sizeof( Wt ) ) * n ) );
    int *lists    = static_cast<int *>( take( allocator, SI( sizeof( int ) ) * 2 * n ) );
    int *counters = static_cast<int *>( take( allocator, SI( sizeof( int ) ) * 64 ) );
    if ( ! nodes || ! pos || ( W && ! w ) || ! lists || ! counters )
        return;                                          // the pool said no: `allocator.failed` is reported

    pb.nodes   = nodes;
    pb.pos     = pos;
    pb.w       = w;
    pb.pos64   = strided( pd.sorted_positions );
    if constexpr ( W )
        pb.w64 = strided( pd.sorted_weights );
    pb.box_min = strided( pd.box_min );
    pb.box_max = strided( pd.box_max );
    pb.ids     = strided( pd.tree.seed_indices );
    pb.res     = reinterpret_cast<char *>( res.data().raw );
    pb.res_stride = SI( res._strides[ Ct<int,0>() ] );
    pb.n       = n;
    pb.depth   = depth;

    zero_fill( queue, counters, SI( sizeof( int ) ) * 64 );

    // the tree and the seeds in the kernel's form
    {
        using TB = std::remove_const_t<typename std::decay_t<decltype( pd.tree.node_box )>::TF>;
        using TJ = std::remove_const_t<typename std::decay_t<decltype( pd.tree.node_begin )>::TF>;
        static_assert( std::is_same_v<TB,TF>, "the tree's boxes and the positions share the driver's float" );
        Strided<TF,2> wa{};
        Strided<TF,1> wb{};
        if constexpr ( W ) {
            wa = strided( pd.tree.node_wa );
            wb = strided( pd.tree.node_wb );
        }
        launch_kernel( queue, &make_nodes<float,W,TB,TJ>, ( nb_nodes + BLOCK - 1 ) / BLOCK, BLOCK, 0,
                       nb_nodes, strided( pd.tree.node_box ), wa, wb, strided( pd.tree.node_begin ), strided( pd.tree.node_end ), nodes );
        launch_kernel( queue, &pack_seeds<TK,W,TF>, ( n + BLOCK - 1 ) / BLOCK, BLOCK, 0, n, pb.pos64, pb.w64, pos, w );
    }

    // first pass, second pass ( no read back between them )
    int *list1 = lists, *list2 = lists + n;
    Deferred deferred{ nullptr, nullptr, nullptr, nullptr, n };
    if constexpr ( std::is_same_v<TK,float> ) {
        deferred.x  = static_cast<float *>( take( allocator, SI( sizeof( float ) ) * RD * n ) );
        deferred.y  = static_cast<float *>( take( allocator, SI( sizeof( float ) ) * RD * n ) );
        deferred.c  = static_cast<int *>( take( allocator, SI( sizeof( int ) ) * RD * n ) );
        deferred.nb = static_cast<int *>( take( allocator, SI( sizeof( int ) ) * n ) );
        if ( ! deferred.x || ! deferred.y || ! deferred.c || ! deferred.nb )
            return;
    }
    launch_kernel( queue, &first_pass<Pb>, ( n + BLOCK - 1 ) / BLOCK, BLOCK, 0, pb, list1, counters + 0, deferred );
    // second pass, `R2` registers, without a read back
    static const int grid2 = resident_grid( &second_pass<Pb>, BLOCK );
    launch_kernel( queue, &second_pass<Pb>, grid2, BLOCK, 0, pb, list1, counters + 0, list2, counters + 1, deferred );

    // third pass, one warp per cell
    auto *k3 = &warp_pass<Pb,shared_cap<TK>()>;
    static const int grid3 = resident_grid( k3, BLOCK, warp_bytes<TK,shared_cap<TK>()>() );
    launch_kernel( queue, k3, grid3, BLOCK, warp_bytes<TK,shared_cap<TK>()>(), pb, list2, counters + 1, list1, counters + 2 );

    if constexpr ( std::is_same_v<TK,float> )
        launch_kernel( queue, &finish_pass<Pb>, ( n + BLOCK - 1 ) / BLOCK, BLOCK, 0, pb, deferred );

    // fourth pass, as long as something overflows: the count is read back, the rows sized on it
    int m = 0;
    read_back( queue, &m, counters + 2, 1 );
    static const bool stats = std::getenv( "SDOT_CARD_STATS" ) && *std::getenv( "SDOT_CARD_STATS" ) != '0';
    if ( stats ) {                                       // how many cells each pass left over ( a diagnosis )
        int c[ 2 ] = { 0, 0 };
        read_back( queue, c, counters + 0, 2 );
        std::printf( "[card cells] n %d, over %d registers %d ( %.2f %% ), over %d registers %d ( %.3f %% ), over %d shared %d\n",
                     n, R1, c[ 0 ], 100.0 * c[ 0 ] / n, R2, c[ 1 ], 100.0 * c[ 1 ] / n, shared_cap<TK>(), m );
    }
    int *in = list1, *out = list2;
    for ( int round = 0, cap = 4 * shared_cap<TK>(); m > 0; ++round, cap *= 4 ) {
        const bool last = round == 4;                    // 2^18 or 2^19 vertices
        auto *xs = static_cast<TK *>( take( allocator, SI( sizeof( TK ) ) * 2 * cap * m ) );
        auto *ys = static_cast<TK *>( take( allocator, SI( sizeof( TK ) ) * 2 * cap * m ) );
        auto *cs = static_cast<int *>( take( allocator, SI( sizeof( int ) ) * 2 * cap * m ) );
        if ( ! xs || ! ys || ! cs )
            return;
        launch_kernel( queue, &memory_pass<Pb>, ( 32 * m + BLOCK - 1 ) / BLOCK, BLOCK, 0, pb, in, m, cap, xs, ys, cs, out, counters + 3 + round, last );
        if ( last )
            break;
        read_back( queue, &m, counters + 3 + round, 1 );
        std::swap( in, out );
    }
}

} // namespace sdot::gpu2d
