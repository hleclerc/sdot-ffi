#pragma once

// =====================================================================================
// THE 3D CELLS OF THE CARD: power diagrams with weights in a box, on a CUDA device, with the BSP tree -- what
// `PowerDiagram.measures` and a Newton iteration need of a 3D diagram ( `PowerDiagram_Bsp._card_variant` decides when
// it applies: a box, a constant density, the BSP tree ). The 2D path is `Cell2D.cuh`; this is its 3D sibling, with the
// old GPU campaign's 3D mapping ( `nsdot/gpu_des_familles`, `Voies3D.cuh`, doc/02-mappages.md ): A CELL ON A WARP.
//
// = WHAT A CELL GIVES ( `Out`, the same bits as in 2D: what is not asked for is not compiled )
//
//   * MEASURES  `rho |cell|`;
//   * FACETS    for each neighbour `j` of rank above the cell's, `c_kj = rho |facet| / ( 2 |p_k - p_j| )`, one COO entry
//               ( rank, rank, value ) that `Laplacian2D.cuh::assemble_laplacian_in` mirrors into the symmetric CSR;
//   * VJP       the adjoint of the measures as a GATHER over the cell's own facets ( `x_kj` the facet's centroid ):
//                   grad_w_k = sum_j c_kj ( g_k - g_j ),   grad_p_k = sum_j 2 c_kj ( g_k - g_j ) ( x_kj - p_k );
//   * MOMENTS   the barycentre and `rho int |x - p_k|^2`.
//
// = HOW A CELL IS BUILT
//
//   * ONE WARP PER CELL, the polytope of the CPU ( `cell/Cellule3D.h` ): a SIMPLE polytope, three cuts and three
//     neighbours per vertex, the neighbour `j` OPPOSITE the cut `j` ( the edge to it lies on the two other cuts ). Lane
//     `l` holds the vertices `l, l + 32, ...` in `S` register slots ( compile-time indices only ), the three cuts of a
//     vertex in one word ( a byte each: LOCAL cut indices, into a table of the cell's cuts held the same way, `cid` ),
//     its three neighbours in another. `nv` and `nc` are uniform.
//   * A CUT, done by the warp together ( `RegCell::cut` ): a `fma` and a `ballot` per slot give the masks of the
//     vertices outside -- they ARE the state of the cut; a CANDIDATE per lane ( an outside vertex, one of its three
//     edges ) kept when the other end is inside, compacted by `ballot` / `popc` ( the new vertex `m` lives on lane `m`
//     during the cut ); the new vertices paired by the old cut they share ( two per crossed face ); the survivors
//     renumbered by `popc` of the masks below them, the new ones after them. The cell is only written at the end of a
//     cut: past its slots it is left whole for the next pass.
//   * THE PASSES, launched without reading a count back ( each strides over a list whose length only the card knows ):
//     `S1 = 2` slots ( 64 vertices ) for every cell, `S2 = 4` ( 128 ) for those that overflowed ( 223 cells in 1e6 in the
//     old campaign ), then a warp per cell in GLOBAL memory ( `MemCell`: any size up to the per-cell limit, within a
//     fixed budget of slots taken once per call, the cells going through them in batches -- `Cell2D.cuh::Overflow`'s
//     scheme ). Past the limit, a FAILURE in loom's error buffer, never a NaN. A cut whose pairing is ambiguous ( a face
//     crossed four times: signs at the rounding of the plane through vertices that are on it ) also sends the cell to the
//     global pass, whose new vertices are paired by WALKING the faces: valid whatever the signs say.
//   * THE TREE NODES as one aligned record in float ( `Node`: box rounded outward, majorant rounded up, slice ), the walk
//     depth first, the nearest child first, its stack indexed by height ( `Cell2D.cuh::walk` ), a leaf's seeds read by
//     the lanes at once and handed to the cuts by `shfl`.
//   * THE ACCURACY FIXES of the old campaign ( doc/04-echelle.md ), as in 2D: the cell lives in the SEED's frame; a float
//     kernel reads the positions and weights as TWO floats ( `x = xh + xl` ), so that a plane carries the double
//     difference rounded once; at the end every vertex is RE-SOLVED in double from the three planes that carry it ( a
//     3 x 3 system, kept where it is well conditioned and near the float vertex ), and everything a cell gives is taken
//     on these vertices: the float only decides WHICH cuts apply.
//
// = THE END OF A CELL ( `finish_warp` ), deterministic: the vertices ( re-solved ), their topology and the planes of
// the cuts go to shared memory ( the global slot for the third pass ), then each lane WALKS faces: the fan of
// triangles from the face's lowest vertex, the tetrahedra from the vertex mean `g`, each face's sums signed by its own
// volume ( convex: every pyramid from `g` is positive ) -- volume, moments, facet area and centroid. A butterfly
// reduction gives every lane the same totals, in the same order at every run.
//
// THE TIES are decided once ( `widen` ): a vertex is outside a plane only beyond the reach of the rounding, so that the
// cuts through vertices that are on them in exact arithmetic ( a grid ) leave them inside rather than cut them off at the
// noise -- which in 3D can leave a face in overlapping cycles. The finish sums each face over every cycle if it has
// several ( `finish_warp` ), and the re-solve puts the vertices back on the true planes ( the double kernel re-solves too:
// the widening would otherwise bias its cells by `tol / |d|` ).
//
// The double kernel ( `TK = double` ) is the same code without the two-float split; its pruning is in float with a
// margin ( `vertex_may_go` ). Diagnosis: `SDOT_CARD_STATS=1` prints how many cells each pass left over.
// =====================================================================================

#include "Laplacian2D.cuh"
#include <climits>

namespace sdot::gpu3d {

using gpu2d::BLOCK;
using gpu2d::Counters;
using gpu2d::Strided;
using gpu2d::StridedOut;
using gpu2d::strided;
using gpu2d::strided_out;
using gpu2d::take;
using gpu2d::blocks_for;
using gpu2d::TFOf;
using gpu2d::TIOf;
using gpu2d::ERROR_KIND_FAILURE;

/// the variant, the same parameters as in 2D ( `PowerDiagram_Bsp.card_variant_for` )
template<class TK,class TR,class TN,int MAX_HEIGHT> using Variant = gpu2d::Variant<TK,TR,TN,MAX_HEIGHT>;

/// what a cell gives ( the bits of `gpu2d::Out` )
constexpr unsigned MEASURES = gpu2d::MEASURES, FACETS = gpu2d::FACETS, VJP = gpu2d::VJP, MOMENTS = gpu2d::MOMENTS;

/// the failure codes ( `id` of loom's error record, `value` the user index of the seed; the messages are Python's )
enum Failure : int { FAIL_TOO_MANY_VERTICES = 1, FAIL_TOPOLOGY = 2 };

constexpr unsigned FULL   = 0xffffffffu;
constexpr int      S1     = 2;                           ///< register slots per lane, first pass ( 64 vertices )
constexpr int      S2     = 4;                           ///< ... second pass ( 128 )
constexpr double   DET_MIN = 1e-6;                       ///< under this relative determinant, the float vertex is kept

/// resident blocks per SM asked of the first pass: four ( 128 registers ) in float -- the old campaign's 216 -> 229 ns
/// against three blocks; at 102 registers it spilled ( 251 ). The double kernel holds twice the coordinates.
template<class TK> constexpr int first_blocks = sizeof( TK ) == 4 ? 5 : 4;

/// what a cut did
enum CutState : int { CUT_NONE = 0, CUT_DONE = 1, CUT_EMPTY = 2, CUT_OVERFLOW = 3, CUT_BROKEN = 4 };

// ---- the kernel's float: the seeds and the planes -----------------------------------------------------------------

/// a seed as a float kernel reads it: the high parts, then the low ones ( `x = xh + xl` ), weight included ( 32 bytes )
struct alignas( 16 ) SeedF { float xh, yh, zh, wh, xl, yl, zl, wl; };
/// ... and as a double kernel does
struct alignas( 16 ) SeedD { double x, y, z, w; };

template<class TK> struct SeedOf;
template<> struct SeedOf<float>  { using T = SeedF; };
template<> struct SeedOf<double> { using T = SeedD; };
template<class TK> using Seed = typename SeedOf<TK>::T;

/// a half-space `d . v <= off`, `v` counted from the seed
/// ( `tol`: how far beyond the plane a vertex must be to count as outside, `widen` )
template<class TK> struct Plane { TK dx, dy, dz, off, tol; };

/// THE POWER BISECTOR in the seed's frame, `d = q - p0`, `off = |d|^2 / 2 + ( w0 - wq ) / 2`. In float `d` is the
/// difference of the two-float positions ( the high parts first: exact as soon as the seeds are close ), the weights'
/// difference the same way.
template<bool W>
__device__ __forceinline__ Plane<float> bisector( const SeedF &f, const SeedF &q ) {
    Plane<float> p;
    p.tol = 0;
    p.dx  = ( q.xh - f.xh ) + ( q.xl - f.xl );
    p.dy  = ( q.yh - f.yh ) + ( q.yl - f.yl );
    p.dz  = ( q.zh - f.zh ) + ( q.zl - f.zl );
    p.off = 0.5f * ( p.dx * p.dx + p.dy * p.dy + p.dz * p.dz );
    if constexpr ( W )
        p.off += 0.5f * ( ( f.wh - q.wh ) + ( f.wl - q.wl ) );
    return p;
}

template<bool W>
__device__ __forceinline__ Plane<double> bisector( const SeedD &f, const SeedD &q ) {
    Plane<double> p;
    p.tol = 0;
    p.dx  = q.x - f.x;
    p.dy  = q.y - f.y;
    p.dz  = q.z - f.z;
    p.off = 0.5 * ( p.dx * p.dx + p.dy * p.dy + p.dz * p.dz );
    if constexpr ( W )
        p.off += 0.5 * ( f.w - q.w );
    return p;
}

/// THE TIES, decided once: a vertex counts as outside only beyond `tol = c eps ( |d| ( |d| + L ) + |off| )`, what the
/// rounding of `d . v - off` can reach for a vertex of the cell ( `L` the farthest corner of the box: a float vertex is
/// interpolated from what were once its corners ). A plane through vertices that are on it in exact arithmetic ( a grid:
/// eight cells at each vertex ) then leaves them all inside, instead of cutting them off at the noise of the rounding --
/// which, in 3D, may leave a face in several cycles that overlap. The new vertices are still placed on the TRUE plane
/// ( the interpolation reads `d . v - off` ): a second cut by the same plane ( two seeds at one place, weights within an
/// ulp of the offset ) then finds them within `tol` and leaves them alone. The cell moves by `tol / |d|` at most.
template<class TK>
__device__ __forceinline__ void widen( Plane<TK> &p, TK L ) {
    // ( 8 float epsilons, 32 double ones: the float band is what a sliver may lose, measured on the planes cloud's facets )
    constexpr TK eps = sizeof( TK ) == 4 ? TK( 8 * 1.1920929e-7 ) : TK( 32 * 2.220446049250313e-16 );
    const TK d = sqrt( p.dx * p.dx + p.dy * p.dy + p.dz * p.dz );
    p.tol = eps * ( d * ( d + L ) + fabs( p.off ) );
}

// ---- the tree, as the kernel reads it ( see `Cell2D.cuh` ) --------------------------------------------------------------

/// ONE NODE, one aligned record in float: 32 bytes for a Voronoi node, 48 weighted ( `int` slices )
template<bool W,class TR> struct alignas( 16 ) Node;
template<class TR> struct alignas( 16 ) Node<false,TR> { float lo[ 3 ], hi[ 3 ]; TR beg, end; };
template<class TR> struct alignas( 16 ) Node<true,TR>  { float lo[ 3 ], hi[ 3 ], a[ 3 ], b; TR beg, end; };

/// the first bytes of a node: what the ordering of two children reads
struct Box { float lo[ 3 ], hi[ 3 ]; };

/// a node seen from the seed: its box in the seed's frame, the majorant's slopes and constant `c = w0 - b - a . p0`
struct CBox { float lo[ 3 ], hi[ 3 ], a[ 3 ], c; };

template<bool W,class TR>
__device__ __forceinline__ CBox centred( const Node<W,TR> &nd, const SeedF &f ) {
    CBox B;
    B.lo[ 0 ] = ( nd.lo[ 0 ] - f.xh ) - f.xl;
    B.lo[ 1 ] = ( nd.lo[ 1 ] - f.yh ) - f.yl;
    B.lo[ 2 ] = ( nd.lo[ 2 ] - f.zh ) - f.zl;
    B.hi[ 0 ] = ( nd.hi[ 0 ] - f.xh ) - f.xl;
    B.hi[ 1 ] = ( nd.hi[ 1 ] - f.yh ) - f.yl;
    B.hi[ 2 ] = ( nd.hi[ 2 ] - f.zh ) - f.zl;
    if constexpr ( W ) {
        B.a[ 0 ] = nd.a[ 0 ];
        B.a[ 1 ] = nd.a[ 1 ];
        B.a[ 2 ] = nd.a[ 2 ];
        // the margin of `b` ( 1e-6 of the scale ) is far above these roundings
        B.c = ( ( f.wh - nd.b ) + f.wl ) - ( nd.a[ 0 ] * f.xh + nd.a[ 1 ] * f.yh + nd.a[ 2 ] * f.zh );
    } else {
        B.a[ 0 ] = B.a[ 1 ] = B.a[ 2 ] = B.c = 0;
    }
    return B;
}

/// the seed's frame for the PRUNING, in float whatever the kernel's float
__device__ __forceinline__ SeedF prune_frame( const SeedF &f ) { return f; }
__device__ __forceinline__ SeedF prune_frame( const SeedD &f ) {
    SeedF r;
    r.xh = __double2float_rn( f.x ); r.xl = __double2float_rn( f.x - double( r.xh ) );
    r.yh = __double2float_rn( f.y ); r.yl = __double2float_rn( f.y - double( r.yh ) );
    r.zh = __double2float_rn( f.z ); r.zl = __double2float_rn( f.z - double( r.zh ) );
    r.wh = __double2float_rn( f.w ); r.wl = __double2float_rn( f.w - double( r.wh ) );
    return r;
}

/// THE PRUNING TEST for one vertex `v` ( seed's frame ): `true` says "a seed of the box may remove it" -- the minimum of
/// `|v - q|^2 - a . q` over the box, free at `q = v + a / 2`, one clamp per axis ( `Cell2D.cuh::vertex_may_go` )
template<bool W>
__device__ __forceinline__ bool vertex_may_go( const CBox &B, float vx, float vy, float vz ) {
    float yx = vx, yy = vy, yz = vz;
    if constexpr ( W ) {
        yx += 0.5f * B.a[ 0 ];
        yy += 0.5f * B.a[ 1 ];
        yz += 0.5f * B.a[ 2 ];
    }
    yx = fminf( fmaxf( yx, B.lo[ 0 ] ), B.hi[ 0 ] );
    yy = fminf( fmaxf( yy, B.lo[ 1 ] ), B.hi[ 1 ] );
    yz = fminf( fmaxf( yz, B.lo[ 2 ] ), B.hi[ 2 ] );
    const float ux = yx - vx, uy = yy - vy, uz = yz - vz;
    float s = ux * ux + uy * uy + uz * uz - ( vx * vx + vy * vy + vz * vz );
    if constexpr ( W )
        s += B.c - ( B.a[ 0 ] * yx + B.a[ 1 ] * yy + B.a[ 2 ] * yz );
    return s <= 0.f;
}

/// the same test for a double vertex, in float, CONSERVATIVE: "may go" up to 1e-6 of the magnitude of its terms
template<bool W>
__device__ __forceinline__ bool vertex_may_go( const CBox &B, double vxd, double vyd, double vzd ) {
    const float vx = __double2float_rn( vxd ), vy = __double2float_rn( vyd ), vz = __double2float_rn( vzd );
    float yx = vx, yy = vy, yz = vz;
    if constexpr ( W ) {
        yx += 0.5f * B.a[ 0 ];
        yy += 0.5f * B.a[ 1 ];
        yz += 0.5f * B.a[ 2 ];
    }
    yx = fminf( fmaxf( yx, B.lo[ 0 ] ), B.hi[ 0 ] );
    yy = fminf( fmaxf( yy, B.lo[ 1 ] ), B.hi[ 1 ] );
    yz = fminf( fmaxf( yz, B.lo[ 2 ] ), B.hi[ 2 ] );
    const float ux = yx - vx, uy = yy - vy, uz = yz - vz;
    const float p = ux * ux + uy * uy + uz * uz, q = vx * vx + vy * vy + vz * vz;
    float s = p - q, t = p + q;
    if constexpr ( W ) {
        const float ax = B.a[ 0 ] * yx, ay = B.a[ 1 ] * yy, az = B.a[ 2 ] * yz;
        s += B.c - ( ax + ay + az );
        t += fabsf( B.c ) + fabsf( ax ) + fabsf( ay ) + fabsf( az );
    }
    return s <= 1e-6f * t;
}

/// the squared distance from the seed to a box: the ORDER of two children, not a test
__device__ __forceinline__ float proximity( const Box &b, const SeedF &f ) {
    const float ex = fmaxf( fmaxf( b.lo[ 0 ] - f.xh, f.xh - b.hi[ 0 ] ), 0.f );
    const float ey = fmaxf( fmaxf( b.lo[ 1 ] - f.yh, f.yh - b.hi[ 1 ] ), 0.f );
    const float ez = fmaxf( fmaxf( b.lo[ 2 ] - f.zh, f.zh - b.hi[ 2 ] ), 0.f );
    return ex * ex + ey * ey + ez * ez;
}

// ---- what a kernel receives ---------------------------------------------------------------------------------------

/// EVERYTHING A CELL READS AND WRITES, by value ( kernel parameters )
template<class _V,class TF,class TI,bool _W,unsigned _OUT>
struct Problem {
    using V     = _V;
    using TK    = typename V::TK;
    using TR    = typename V::TR;
    using TN    = typename V::TN;
    using SeedT = Seed<TK>;
    static constexpr bool     W   = _W;
    static constexpr unsigned OUT = _OUT;

    // ---- the diagram
    const Node<W,TR> *nodes;                             ///< in float for both kernels
    const SeedT      *seeds;                             ///< the seeds in tree order, in the kernel's form
    Strided<TF,2>     pos64;                             ///< the seeds as given, `[ n, 3 ]`, tree order
    Strided<TF,1>     w64;                               ///< the weights as given ( W only )
    Strided<TF,1>     box_min, box_max;                  ///< the domain
    Strided<TI,1>     ids;                               ///< rank -> the user's index
    TR                n;
    int               depth;
    double            rho;                               ///< the ( constant ) density, if `rho_dev` is null
    const TF         *rho_dev;                           ///< ... or read on the card
    bool              user_order;                        ///< per-cell outputs at the user's index ( else at the rank )
    bool              only_global;                       ///< every cell through the global pass ( `SDOT_CARD_GLOBAL_ONLY=1`, a test )

    // ---- what the cells write
    StridedOut<TF,1>  res;                               ///< MEASURES
    TR               *fi, *fj;                          ///< FACETS: the COO of the upper facets, in ranks
    double           *fc;
    unsigned long long fcap;
    Counters         *counters;
    Strided<TF,1>     g;                                 ///< VJP: the cotangent of the measures ( user order )
    StridedOut<TF,2>  grad_pos;                          ///< ... -> the positions, rank order ( may be null )
    StridedOut<TF,1>  grad_w;                            ///< ... -> the weights, rank order ( may be null )
    StridedOut<TF,2>  bary;                              ///< MOMENTS: the barycentre ( the seed if empty )
    StridedOut<TF,1>  cost;                              ///< ... and `rho int_cell |x - p|^2`

    __device__ __forceinline__ double density() const { return rho_dev ? double( *rho_dev ) : rho; }
    __device__ __forceinline__ SI user( TR k ) const { return user_order ? SI( ids( k ) ) : SI( k ); }
};

// ---- warp helpers ( the old campaign's: no `__fns`, no local memory, every slot index a compile-time constant ) --------

/// `a[ s ]` with a dynamic `s`, by a chain of selects
template<class T,int S>
__device__ __forceinline__ T sel( const T ( &a )[ S ], int s ) {
    T r = a[ 0 ];
#pragma unroll
    for ( int q = 1; q < S; ++q ) r = s == q ? a[ q ] : r;
    return r;
}

/// THE GATHER: field `a` of vertex ( or cut ) `idx`, wherever it lives ( lane `idx % 32`, slot `idx / 32` ). `su`: the
/// slots in use ( uniform ); at two slots every one is broadcast ( a `shfl` costs less than a branch and its barrier )
template<class T,int S>
__device__ __forceinline__ T gather( const T ( &a )[ S ], int idx, int su ) {
    T r = T( 0 );
#pragma unroll
    for ( int s = 0; s < S; ++s ) {
        if ( S <= 2 || s < su ) {
            const T v = __shfl_sync( FULL, a[ s ], idx & 31 );
            r = ( idx >> 5 ) == s ? v : r;
        }
    }
    return r;
}

/// the `n`-th set bit of `m` ( from 0 ), without a loop: a dichotomy on `popc`
__device__ __forceinline__ int bit_nth( unsigned m, int n ) {
    int pos = 0;
    int c = __popc( m & 0xffffu ); bool h = n >= c; n -= h ? c : 0; pos += h ? 16 : 0; m = h ? m >> 16 : m;
    c = __popc( m & 0xffu );       h = n >= c; n -= h ? c : 0; pos += h ? 8 : 0;  m = h ? m >> 8 : m;
    c = __popc( m & 0xfu );        h = n >= c; n -= h ? c : 0; pos += h ? 4 : 0;  m = h ? m >> 4 : m;
    c = __popc( m & 0x3u );        h = n >= c; n -= h ? c : 0; pos += h ? 2 : 0;  m = h ? m >> 2 : m;
    c = __popc( m & 0x1u );        h = n >= c; pos += h ? 1 : 0;
    return pos;
}

/// the `n`-th set bit of the masks, as a vertex index
template<int S>
__device__ __forceinline__ int nth( const unsigned ( &m )[ S ], int n ) {
    int slot = 0, pre = 0, acc = 0;
#pragma unroll
    for ( int s = 0; s < S - 1; ++s ) {
        acc += __popc( m[ s ] );
        const bool h = n >= acc;
        slot = h ? s + 1 : slot;
        pre  = h ? acc : pre;
    }
    return slot * 32 + bit_nth( sel( m, slot ), n - pre );
}

/// how many set bits of the masks are below index `w`
template<int S>
__device__ __forceinline__ int rank( const unsigned ( &m )[ S ], int w ) {
    int r = 0;
#pragma unroll
    for ( int s = 0; s < S; ++s ) if ( s < ( w >> 5 ) ) r += __popc( m[ s ] );
    return r + __popc( sel( m, w >> 5 ) & ( ( 1u << ( w & 31 ) ) - 1 ) );
}

template<int S>
__device__ __forceinline__ int total( const unsigned ( &m )[ S ] ) {
    int r = 0;
#pragma unroll
    for ( int s = 0; s < S; ++s ) r += __popc( m[ s ] );
    return r;
}

template<int S>
__device__ __forceinline__ bool has_bit( const unsigned ( &m )[ S ], int w ) { return ( sel( m, w >> 5 ) >> ( w & 31 ) ) & 1u; }

__device__ __forceinline__ int byte_of( unsigned w, int j ) { return int( ( w >> ( 8 * j ) ) & 255u ); }
__device__ __forceinline__ unsigned word3( int a, int b, int c ) {
    return ( unsigned( a ) & 255u ) | ( ( unsigned( b ) & 255u ) << 8 ) | ( ( unsigned( c ) & 255u ) << 16 );
}

/// the two faces that carry edge `j` of a vertex: its cuts but the `j`-th
__device__ __forceinline__ void faces_of( unsigned k, int j, int &f0, int &f1 ) {
    f0 = byte_of( k, j == 0 ? 1 : 0 );
    f1 = byte_of( k, j == 2 ? 1 : 2 );
}

__device__ __forceinline__ int warp_sum( int v ) {
#pragma unroll
    for ( int o = 16; o; o /= 2 ) v += __shfl_xor_sync( FULL, v, o );
    return v;
}
/// a butterfly sum: every lane ends with THE SAME bits ( each stage adds the same two numbers, in either order )
__device__ __forceinline__ double warp_sum( double v ) {
#pragma unroll
    for ( int o = 16; o; o /= 2 ) v += __shfl_xor_sync( FULL, v, o );
    return v;
}

/// the distance from the seed to the farthest corner of the box `b` ( seed's frame )
template<class TK>
__device__ __forceinline__ TK corner( const TK ( &b )[ 6 ] ) {
    TK r = 0;
    for ( int d = 0; d < 3; ++d )
        r += fmax( b[ d ] * b[ d ], b[ 3 + d ] * b[ 3 + d ] );
    return sqrt( r );
}

// ---- the cell in registers ( first and second passes ) --------------------------------------------------------------

/// `32 S` vertices and `32 S` cuts at most; the domain box to start with ( `init` )
template<class TK,class TR,int S>
struct RegCell {
    static constexpr int CAP = 32 * S;
    static_assert( S >= 1 && S <= 8, "the cut and neighbour indices are bytes" );

    TK       x[ S ], y[ S ], z[ S ];
    unsigned k[ S ];                                     ///< the three cuts of the vertex, a byte each
    unsigned nn[ S ];                                    ///< its three neighbours, `nn_j` OPPOSITE `k_j`
    TR       cid[ S ];                                   ///< the identifier of cut `s * 32 + lane` ( a rank, or a side of the box )
    TK       L;                                          ///< the farthest corner of the box from the seed ( `widen` )
    int      nv, nc, lane, state;

    __device__ __forceinline__ int su_v() const { return ( nv + 31 ) >> 5; }

    /// THE BOX `b = ( x0, y0, z0, x1, y1, z1 )` ( seed's frame ): vertex `v < 8` on lane `v`, its bits the sides it is on;
    /// cut `2 a + h` is the side `h` of axis `a`, identifier `-1 - ( 2 a + h )`
    __device__ __forceinline__ void init( int l, const TK ( &b )[ 6 ] ) {
        lane = l;
        state = CUT_NONE;
        nv = 8;
        nc = 6;
        L = corner( b );
        const int v = l & 7, i = v & 1, j = ( v >> 1 ) & 1, q = ( v >> 2 ) & 1;
        x[ 0 ] = i ? b[ 3 ] : b[ 0 ];
        y[ 0 ] = j ? b[ 4 ] : b[ 1 ];
        z[ 0 ] = q ? b[ 5 ] : b[ 2 ];
        k[ 0 ] = word3( i, 2 + j, 4 + q );
        nn[ 0 ] = word3( v ^ 1, v ^ 2, v ^ 4 );
        cid[ 0 ] = TR( -1 - l );
#pragma unroll
        for ( int s = 1; s < S; ++s ) {
            x[ s ] = y[ s ] = z[ s ] = TK( 0 );
            k[ s ] = nn[ s ] = 0;
            cid[ s ] = TR( 0 );
        }
    }

    template<bool W>
    __device__ __forceinline__ bool may_be_cut_by( const CBox &B ) const {
        const int su = su_v();
        bool res = false;
#pragma unroll
        for ( int s = 0; s < S; ++s ) {
            if ( s < su ) {
                const bool m = s * 32 + lane < nv && vertex_may_go<W>( B, x[ s ], y[ s ], z[ s ] );
                res |= __any_sync( FULL, m );
            }
        }
        return res;
    }

    /// THE DEAD CUTS out of the table ( no vertex refers to them any more ): the live ones renumbered by `popc` below
    /// them -- monotone, so the order of the bytes does not matter
    __device__ __forceinline__ void compact() {
        const int su = su_v();
        unsigned m[ S ];
#pragma unroll
        for ( int r = 0; r < S; ++r ) {
            unsigned loc = 0;
#pragma unroll
            for ( int s = 0; s < S; ++s ) {
                if ( s * 32 + lane < nv ) {
#pragma unroll
                    for ( int f = 0; f < 3; ++f ) {
                        const int c = byte_of( k[ s ], f );
                        loc |= ( c >> 5 ) == r ? 1u << ( c & 31 ) : 0u;
                    }
                }
            }
#pragma unroll
            for ( int o = 16; o; o /= 2 ) loc |= __shfl_xor_sync( FULL, loc, o );
            m[ r ] = loc;
        }
        const int ncn = total( m );
#pragma unroll
        for ( int s = 0; s < S; ++s )
            if ( s < su && s * 32 + lane < nv )
                k[ s ] = word3( rank( m, byte_of( k[ s ], 0 ) ), rank( m, byte_of( k[ s ], 1 ) ), rank( m, byte_of( k[ s ], 2 ) ) );
        const int suc = ( nc + 31 ) >> 5;
        TR c2[ S ];
#pragma unroll
        for ( int s = 0; s < S; ++s ) {
            const int t = s * 32 + lane;
            const TR v = gather( cid, t < ncn ? nth( m, t ) : 0, suc );
            c2[ s ] = t < ncn ? v : TR( 0 );
        }
#pragma unroll
        for ( int s = 0; s < S; ++s ) cid[ s ] = c2[ s ];
        nc = ncn;
    }

    /// ONE CUT by `p`, whose identifier is `pid`
    __device__ __forceinline__ int cut( const Plane<TK> &p, TR pid ) {
        const int su = su_v();

        // ---- the signed distances, the masks of the vertices outside and inside
        TK       sv[ S ];
        unsigned out[ S ], in[ S ];
#pragma unroll
        for ( int s = 0; s < S; ++s ) {
            sv[ s ] = p.dx * x[ s ] + p.dy * y[ s ] + p.dz * z[ s ] - p.off;
            const bool valid = s * 32 + lane < nv;
            out[ s ] = in[ s ] = 0u;
            if ( s < su ) {
                out[ s ] = __ballot_sync( FULL, valid && sv[ s ] > p.tol );
                in[ s ]  = __ballot_sync( FULL, valid && ! ( sv[ s ] > p.tol ) );
            }
        }
        const int nb_out = total( out );
        if ( __builtin_expect( nb_out == 0, 1 ) )
            return CUT_NONE;
        const int nb_in = total( in );
        if ( nb_in == 0 ) {
            nv = 0;
            return CUT_EMPTY;
        }
        if ( nc >= CAP ) {
            compact();
            if ( nc >= CAP )
                return CUT_OVERFLOW;
        }
        const int knew = nc;

        // ---- THE NEW VERTICES: a candidate per lane ( outside vertex `o`, its edge `j` ), kept if the other end `u` is
        // inside; the new vertex `m` lives on lane `m`
        TK       NX = 0, NY = 0, NZ = 0;
        unsigned N01 = 0;                                // the two cuts it inherits, `f0 | f1 << 8`
        int      RV = 0, RF = 0;                         // the inside end, and the slot of it that pointed outside
        int      nm = 0;
        for ( int base = 0; base < 3 * nb_out; base += 32 ) {
            const int  c = base + lane;
            const bool cand = c < 3 * nb_out;
            const int  io = c / 3, j = c - 3 * io;
            const int  o = cand ? nth( out, io ) : 0;
            const unsigned ko = gather( k, o, su ), nno = gather( nn, o, su );
            const int  u = byte_of( nno, j );
            const bool keep = cand && ! has_bit( out, u );
            const TK   so = gather( sv, o, su ), s_u = gather( sv, u, su );
            const TK   xo = gather( x, o, su ), yo = gather( y, o, su ), zo = gather( z, o, su );
            const TK   xu = gather( x, u, su ), yu = gather( y, u, su ), zu = gather( z, u, su );
            const unsigned nnu = gather( nn, u, su );
            // from the inside end; an inside end within `tol` beyond the plane is the new vertex itself ( the ratio
            // would extrapolate, without bound when both ends are in the band )
            const TK   t = s_u > TK( 0 ) ? TK( 0 ) : s_u / ( s_u - so );
            const TK   nx = xu + ( xo - xu ) * t, ny = yu + ( yo - yu ) * t, nz = zu + ( zo - zu ) * t;
            int f0, f1;
            faces_of( ko, j, f0, f1 );
            const int rf = byte_of( nnu, 0 ) == o ? 0 : ( byte_of( nnu, 1 ) == o ? 1 : 2 );

            const unsigned bal = __ballot_sync( FULL, keep );
            const int cnt = __popc( bal );
            if ( nm + cnt > 32 )
                return CUT_OVERFLOW;
            const int  r = lane - nm;
            const bool mine = r >= 0 && r < cnt;
            const int  src = bit_nth( bal, mine ? r : 0 );
            const TK   gx = __shfl_sync( FULL, nx, src ), gy = __shfl_sync( FULL, ny, src ), gz = __shfl_sync( FULL, nz, src );
            const unsigned g01 = __shfl_sync( FULL, unsigned( f0 ) | ( unsigned( f1 ) << 8 ), src );
            const int  gv = __shfl_sync( FULL, u, src ), gf = __shfl_sync( FULL, rf, src );
            if ( mine ) { NX = gx; NY = gy; NZ = gz; N01 = g01; RV = gv; RF = gf; }
            nm += cnt;
        }
        const int new_nv = nb_in + nm;
        if ( new_nv > CAP )
            return CUT_OVERFLOW;

        // ---- THE NEW VERTICES' NEIGHBOURS AMONG THEMSELVES: the one that shares old cut `f0` is opposite `f1`, and
        // conversely -- exactly one each when every crossed face is crossed twice. Otherwise ( a face crossed four times:
        // signs decided at the rounding of a plane through vertices that are on it ) the pairing is ambiguous, and the
        // global pass, which pairs by walking the faces, has the cell.
        int M0 = -1, M1 = -1, c0 = 0, c1 = 0;
        {
            const int n0 = byte_of( N01, 0 ), n1 = byte_of( N01, 1 );
            for ( int j = 0; j < nm; ++j ) {
                const unsigned nj = __shfl_sync( FULL, N01, j );
                const int a = byte_of( nj, 0 ), b = byte_of( nj, 1 );
                if ( j != lane ) {
                    if ( n0 == a || n0 == b ) { M1 = j; ++c0; }
                    if ( n1 == a || n1 == b ) { M0 = j; ++c1; }
                }
            }
        }
        if ( __any_sync( FULL, lane < nm && ( c0 != 1 || c1 != 1 || M0 == M1 ) ) )
            return CUT_OVERFLOW;

        // ---- THE NEW STATE: the survivors in order, then the new vertices
        const int su2 = ( new_nv + 31 ) >> 5;
        int src[ S ];
#pragma unroll
        for ( int s = 0; s < S; ++s ) {
            const int t = s * 32 + lane;
            src[ s ] = t < nb_in ? nth( in, t ) : 0;
        }
        TK       x2[ S ], y2[ S ], z2[ S ];
        unsigned k2[ S ], nn2[ S ];
#pragma unroll
        for ( int s = 0; s < S; ++s ) {
            x2[ s ] = y2[ s ] = z2[ s ] = TK( 0 );
            k2[ s ] = nn2[ s ] = 0;
            if ( s < su2 ) {
                const int i = src[ s ];
                x2[ s ] = gather( x, i, su );
                y2[ s ] = gather( y, i, su );
                z2[ s ] = gather( z, i, su );
                k2[ s ] = gather( k, i, su );
                const unsigned ni = gather( nn, i, su );
                unsigned w = 0;
#pragma unroll
                for ( int f = 0; f < 3; ++f ) {
                    const int old = byte_of( ni, f );
                    w |= unsigned( has_bit( out, old ) ? 255 : rank( in, old ) ) << ( 8 * f );
                }
                nn2[ s ] = w;
            }
        }
        // a survivor's slot that pointed outside now points at the new vertex born on that edge
        for ( int j = 0; j < nm; ++j ) {
            const int rv = __shfl_sync( FULL, RV, j ), rf = __shfl_sync( FULL, RF, j );
#pragma unroll
            for ( int s = 0; s < S; ++s )
                if ( s * 32 + lane < nb_in && src[ s ] == rv )
                    nn2[ s ] = ( nn2[ s ] & ~( 255u << ( 8 * rf ) ) ) | ( unsigned( nb_in + j ) << ( 8 * rf ) );
        }
#pragma unroll
        for ( int s = 0; s < S; ++s ) {
            if ( s < su2 ) {
                const int  t = s * 32 + lane;
                const bool neu = t >= nb_in && t < new_nv;
                const int  j = neu ? t - nb_in : 0;
                const TK   gx = __shfl_sync( FULL, NX, j ), gy = __shfl_sync( FULL, NY, j ), gz = __shfl_sync( FULL, NZ, j );
                const unsigned n01 = __shfl_sync( FULL, N01, j );
                const int  m0 = __shfl_sync( FULL, M0, j ), m1 = __shfl_sync( FULL, M1, j ), rv = __shfl_sync( FULL, RV, j );
                if ( neu ) {
                    x2[ s ] = gx; y2[ s ] = gy; z2[ s ] = gz;
                    k2[ s ] = n01 | ( unsigned( knew ) << 16 );
                    nn2[ s ] = word3( nb_in + m0, nb_in + m1, rank( in, rv ) );
                }
            }
        }
#pragma unroll
        for ( int s = 0; s < S; ++s ) {
            x[ s ] = x2[ s ]; y[ s ] = y2[ s ]; z[ s ] = z2[ s ]; k[ s ] = k2[ s ]; nn[ s ] = nn2[ s ];
            // ( a select, not a conditional store: the compiler turns the latter into `cid[ knew >> 5 ]`, a dynamic index
            // that sends the whole cell to local memory )
            cid[ s ] = ( ( knew >> 5 ) == s && ( knew & 31 ) == lane ) ? pid : cid[ s ];
        }
        nc = knew + 1;
        nv = new_nv;
        return CUT_DONE;
    }

    /// A LEAF AT ONCE: the lanes read its seeds together and compute their planes, the cuts take them from the lanes in
    /// rank order. `false` when the walk must stop ( the cell is empty, or overflowed: `state` )
    template<class Pb>
    __device__ __forceinline__ bool cut_leaf( const Pb &pb, const Seed<TK> &f, TR beg, TR end ) {
        for ( TR q0 = beg; q0 < end; q0 += 32 ) {
            Plane<TK> P{};
            if ( q0 + lane < end ) {
                P = bisector<Pb::W>( f, pb.seeds[ q0 + lane ] );
                widen( P, L );
            }
            const int nq = int( min( TR( 32 ), TR( end - q0 ) ) );
            for ( int j = 0; j < nq; ++j ) {
                Plane<TK> pj;
                pj.dx = __shfl_sync( FULL, P.dx, j );
                pj.dy = __shfl_sync( FULL, P.dy, j );
                pj.dz = __shfl_sync( FULL, P.dz, j );
                pj.off = __shfl_sync( FULL, P.off, j );
                pj.tol = __shfl_sync( FULL, P.tol, j );
                const int r = cut( pj, TR( q0 + j ) );
                if ( r == CUT_EMPTY || r == CUT_OVERFLOW ) {
                    state = r;
                    return false;
                }
            }
        }
        return true;
    }
};

// ---- the cell in global memory ( third pass ) -----------------------------------------------------------------------

/// A CELL OF ANY SIZE ( up to `capv` vertices ), held by a warp in a slot of global memory: the same polytope with `int`
/// indices, two buffers of vertices ( a cut reads one and writes the other ), the signs, the new indices. What the
/// registers could not hold: each lane loops over the vertices; the new vertices are paired by WALKING the faces, from
/// an outside vertex along the face to the edge where the cut leaves it -- valid whatever the signs decided ( several
/// outside regions, a face crossed four times ).
template<class TK,class TR>
struct MemCell {
    TK     *X[ 2 ], *Y[ 2 ], *Z[ 2 ], *s;
    int    *K[ 2 ], *N[ 2 ], *ren, *aux, *cnt;
    TR     *cid;
    double *pl, *vx, *vy, *vz;
    TK      L;
    int     cur, capv, capc, nv, nc, lane, state;

    static constexpr SI align16( SI b ) { return ( b + 15 ) / 16 * 16; }

    /// the bytes of a slot of `capv` vertices and `capc` cuts
    static constexpr SI bytes_for( int capv, int capc ) {
        return 6 * align16( SI( sizeof( TK ) ) * capv ) + 4 * align16( SI( 12 ) * capv ) + align16( SI( sizeof( TK ) ) * capv )
             + align16( SI( 4 ) * capv ) + 3 * align16( SI( 8 ) * capv )
             + align16( SI( sizeof( TR ) ) * capc ) + 2 * align16( SI( 4 ) * capc ) + align16( SI( 32 ) * capc );
    }

    __device__ __forceinline__ void attach( unsigned char *base, int capv_, int capc_ ) {
        capv = capv_;
        capc = capc_;
        auto next = [&]( SI nb ) { unsigned char *r = base; base += align16( nb ); return r; };
        for ( int q = 0; q < 2; ++q ) {
            X[ q ] = reinterpret_cast<TK *>( next( SI( sizeof( TK ) ) * capv ) );
            Y[ q ] = reinterpret_cast<TK *>( next( SI( sizeof( TK ) ) * capv ) );
            Z[ q ] = reinterpret_cast<TK *>( next( SI( sizeof( TK ) ) * capv ) );
            K[ q ] = reinterpret_cast<int *>( next( SI( 12 ) * capv ) );
            N[ q ] = reinterpret_cast<int *>( next( SI( 12 ) * capv ) );
        }
        s   = reinterpret_cast<TK *>( next( SI( sizeof( TK ) ) * capv ) );
        ren = reinterpret_cast<int *>( next( SI( 4 ) * capv ) );
        vx  = reinterpret_cast<double *>( next( SI( 8 ) * capv ) );
        vy  = reinterpret_cast<double *>( next( SI( 8 ) * capv ) );
        vz  = reinterpret_cast<double *>( next( SI( 8 ) * capv ) );
        cid = reinterpret_cast<TR *>( next( SI( sizeof( TR ) ) * capc ) );
        aux = reinterpret_cast<int *>( next( SI( 4 ) * capc ) );
        cnt = reinterpret_cast<int *>( next( SI( 4 ) * capc ) );
        pl  = reinterpret_cast<double *>( next( SI( 32 ) * capc ) );
    }

    __device__ __forceinline__ void init( int l, const TK ( &b )[ 6 ] ) {
        lane = l;
        state = CUT_NONE;
        cur = 0;
        L = corner( b );
        if ( lane < 8 ) {
            const int v = lane, i = v & 1, j = ( v >> 1 ) & 1, q = ( v >> 2 ) & 1;
            X[ 0 ][ v ] = i ? b[ 3 ] : b[ 0 ];
            Y[ 0 ][ v ] = j ? b[ 4 ] : b[ 1 ];
            Z[ 0 ][ v ] = q ? b[ 5 ] : b[ 2 ];
            K[ 0 ][ 3 * v ] = i; K[ 0 ][ 3 * v + 1 ] = 2 + j; K[ 0 ][ 3 * v + 2 ] = 4 + q;
            N[ 0 ][ 3 * v ] = v ^ 1; N[ 0 ][ 3 * v + 1 ] = v ^ 2; N[ 0 ][ 3 * v + 2 ] = v ^ 4;
        }
        if ( lane < 6 )
            cid[ lane ] = TR( -1 - lane );
        nv = 8;
        nc = 6;
        __syncwarp();
    }

    template<bool W>
    __device__ __forceinline__ bool may_be_cut_by( const CBox &B ) const {
        bool res = false;
        for ( int i = lane; i < nv; i += 32 )
            res |= vertex_may_go<W>( B, X[ cur ][ i ], Y[ cur ][ i ], Z[ cur ][ i ] );
        return __any_sync( FULL, res );
    }

    TK      tol;                                         ///< the current cut's ( `widen` )
    __device__ __forceinline__ bool outside( int i ) const { return s[ i ] > tol; }

    /// the index, among the new vertices, of the one born on the edge `slot` of the outside vertex `o`
    __device__ __forceinline__ int new_on( int o, int slot ) const {
        const int *nb = N[ cur ] + 3 * o;
        int m = ren[ o ];
        for ( int j = 0; j < slot; ++j )
            m += ! outside( nb[ j ] );
        return m;
    }

    /// the slot of `i` that holds `v` in `a` ( `K` or `N` ), or -1
    __device__ __forceinline__ static int slot_of( const int *a, int i, int v ) {
        return a[ 3 * i ] == v ? 0 : a[ 3 * i + 1 ] == v ? 1 : a[ 3 * i + 2 ] == v ? 2 : -1;
    }

    /// THE PARTNER of the new vertex born on edge `j` of the outside vertex `o`, along face `f` ( one of the two cuts of
    /// that edge ): walk the face from `o` away from the edge, over outside vertices, to the edge where the face comes
    /// back inside -- the new vertex born there. -1 when the topology is broken.
    __device__ __forceinline__ int partner( int o, int j, int f ) const {
        const int *k = K[ cur ], *nb = N[ cur ];
        int i_f = slot_of( k, o, f );
        if ( i_f < 0 || i_f == j )
            return -1;
        int t = 3 - i_f - j, prev = o, at = nb[ 3 * o + t ];
        for ( int step = 0; step <= nv; ++step ) {
            if ( ! outside( at ) )
                return new_on( prev, t );
            const int i1 = slot_of( k, at, f ), e = slot_of( nb, at, prev );
            if ( i1 < 0 || e < 0 || i1 == e )
                return -1;
            t = 3 - i1 - e;
            prev = at;
            at = nb[ 3 * at + t ];
        }
        return -1;
    }

    __device__ __forceinline__ void compact() {
        int *k = K[ cur ];
        for ( int c = lane; c < nc; c += 32 )
            aux[ c ] = 0;
        __syncwarp();
        for ( int i = lane; i < nv; i += 32 )
            for ( int j = 0; j < 3; ++j )
                aux[ k[ 3 * i + j ] ] = 1;
        __syncwarp();
        int run = 0;
        for ( int base = 0; base < nc; base += 32 ) {
            const int c = base + lane;
            const bool live = c < nc && aux[ c ];
            const unsigned bal = __ballot_sync( FULL, live );
            if ( c < nc )
                aux[ c ] = live ? run + __popc( bal & ( ( 1u << lane ) - 1 ) ) : -1;
            run += __popc( bal );
        }
        __syncwarp();
        for ( int i = lane; i < nv; i += 32 )
            for ( int j = 0; j < 3; ++j )
                k[ 3 * i + j ] = aux[ k[ 3 * i + j ] ];
        for ( int base = 0; base < nc; base += 32 ) {    // in place: a cut only moves down, never past the chunk read
            const int c = base + lane;
            const TR v = c < nc ? cid[ c ] : TR( 0 );
            const int to = c < nc ? aux[ c ] : -1;
            __syncwarp();
            if ( to >= 0 )
                cid[ to ] = v;
            __syncwarp();
        }
        nc = run;
    }

    __device__ __forceinline__ int cut( const Plane<TK> &p, TR pid ) {
        const TK *x = X[ cur ], *y = Y[ cur ], *z = Z[ cur ];
        tol = p.tol;
        int no = 0;
        for ( int i = lane; i < nv; i += 32 ) {
            const TK si = p.dx * x[ i ] + p.dy * y[ i ] + p.dz * z[ i ] - p.off;
            s[ i ] = si;
            no += si > p.tol;
        }
        no = warp_sum( no );
        if ( no == 0 )
            return CUT_NONE;
        if ( no == nv ) {
            nv = 0;
            return CUT_EMPTY;
        }
        __syncwarp();
        if ( nc >= capc ) {
            compact();
            if ( nc >= capc )
                return CUT_OVERFLOW;
        }
        const int *k = K[ cur ], *nb = N[ cur ];

        // ---- the new indices: a survivor's rank among the inside vertices, an outside vertex's first new vertex
        int run_in = 0, run_new = 0;
        const unsigned lt = ( 1u << lane ) - 1;
        for ( int base = 0; base < nv; base += 32 ) {
            const int  i = base + lane;
            const bool valid = i < nv;
            const bool o = valid && outside( i ), in = valid && ! o;
            int cn = 0;
            if ( o )
                for ( int j = 0; j < 3; ++j )
                    cn += ! outside( nb[ 3 * i + j ] );
            const unsigned bal = __ballot_sync( FULL, in );
            int incl = cn;
#pragma unroll
            for ( int d = 1; d < 32; d *= 2 ) {
                const int v = __shfl_up_sync( FULL, incl, d );
                if ( lane >= d ) incl += v;
            }
            if ( in ) ren[ i ] = run_in + __popc( bal & lt );
            if ( o )  ren[ i ] = run_new + incl - cn;
            run_in += __popc( bal );
            run_new += __shfl_sync( FULL, incl, 31 );
        }
        const int nb_in = run_in, nm = run_new;
        if ( nb_in + nm > capv )
            return CUT_OVERFLOW;
        __syncwarp();

        const int nx = 1 - cur;
        TK *x2 = X[ nx ], *y2 = Y[ nx ], *z2 = Z[ nx ];
        int *k2 = K[ nx ], *n2 = N[ nx ];
        bool bad = false;
        // ---- the survivors
        for ( int i = lane; i < nv; i += 32 ) {
            if ( outside( i ) )
                continue;
            const int ni = ren[ i ];
            x2[ ni ] = x[ i ]; y2[ ni ] = y[ i ]; z2[ ni ] = z[ i ];
            for ( int j = 0; j < 3; ++j ) {
                k2[ 3 * ni + j ] = k[ 3 * i + j ];
                const int old = nb[ 3 * i + j ];
                int to = ren[ old ];
                if ( outside( old ) ) {
                    const int js = slot_of( nb, old, i );
                    bad |= js < 0;
                    to = nb_in + new_on( old, js < 0 ? 0 : js );
                }
                n2[ 3 * ni + j ] = to;
            }
        }
        // ---- the new vertices: `( f0, f1, new cut )`, their neighbours `( partner along f1, partner along f0, the inside end )`
        for ( int o = lane; o < nv; o += 32 ) {
            if ( ! outside( o ) )
                continue;
            int m = ren[ o ];
            for ( int j = 0; j < 3; ++j ) {
                const int u = nb[ 3 * o + j ];
                if ( outside( u ) )
                    continue;
                const int q = nb_in + m++;
                const TK t = s[ u ] > TK( 0 ) ? TK( 0 ) : s[ u ] / ( s[ u ] - s[ o ] );
                x2[ q ] = x[ u ] + ( x[ o ] - x[ u ] ) * t;
                y2[ q ] = y[ u ] + ( y[ o ] - y[ u ] ) * t;
                z2[ q ] = z[ u ] + ( z[ o ] - z[ u ] ) * t;
                const int f0 = k[ 3 * o + ( j == 0 ? 1 : 0 ) ], f1 = k[ 3 * o + ( j == 2 ? 1 : 2 ) ];
                k2[ 3 * q ] = f0;
                k2[ 3 * q + 1 ] = f1;
                k2[ 3 * q + 2 ] = nc;
                const int p1 = partner( o, j, f1 ), p0 = partner( o, j, f0 );
                bad |= p0 < 0 || p1 < 0;
                n2[ 3 * q ] = nb_in + ( p1 < 0 ? 0 : p1 );
                n2[ 3 * q + 1 ] = nb_in + ( p0 < 0 ? 0 : p0 );
                n2[ 3 * q + 2 ] = ren[ u ];
            }
        }
        if ( __any_sync( FULL, bad ) )
            return CUT_BROKEN;
        if ( lane == 0 )
            cid[ nc ] = pid;
        __syncwarp();
        cur = nx;
        nv = nb_in + nm;
        ++nc;
        return CUT_DONE;
    }

    template<class Pb>
    __device__ __forceinline__ bool cut_leaf( const Pb &pb, const Seed<TK> &f, TR beg, TR end ) {
        for ( TR q0 = beg; q0 < end; q0 += 32 ) {
            Plane<TK> P{};
            if ( q0 + lane < end ) {
                P = bisector<Pb::W>( f, pb.seeds[ q0 + lane ] );
                widen( P, L );
            }
            const int nq = int( min( TR( 32 ), TR( end - q0 ) ) );
            for ( int j = 0; j < nq; ++j ) {
                Plane<TK> pj;
                pj.dx = __shfl_sync( FULL, P.dx, j );
                pj.dy = __shfl_sync( FULL, P.dy, j );
                pj.dz = __shfl_sync( FULL, P.dz, j );
                pj.off = __shfl_sync( FULL, P.off, j );
                pj.tol = __shfl_sync( FULL, P.tol, j );
                const int r = cut( pj, TR( q0 + j ) );
                if ( r == CUT_EMPTY || r == CUT_OVERFLOW || r == CUT_BROKEN ) {
                    state = r;
                    return false;
                }
            }
        }
        return true;
    }
};

// ---- the walk ---------------------------------------------------------------------------------------------------------

/// THE DESCENT ( `Cell2D.cuh::walk`, the whole warp on the same path: every value here is uniform ): depth first, the
/// nearest child first, a node pruned when it is reached, the stack indexed by height
template<class Pb,class Cell>
__device__ __forceinline__ void walk3( const Pb &pb, const typename Pb::SeedT &f, Cell &cell ) {
    using TN   = typename Pb::TN;
    using TR   = typename Pb::TR;
    using Mask = typename Pb::V::Mask;
    const SeedF pf = prune_frame( f );
    TN   stack[ Pb::V::MAX_HEIGHT ];
    Mask pending = 0;
    TN   n = 0;
    int  h = pb.depth;
    while ( true ) {
        const Node<Pb::W,TR> nd = pb.nodes[ n ];
        if ( nd.beg < nd.end && cell.template may_be_cut_by<Pb::W>( centred( nd, pf ) ) ) {
            if ( h <= 1 ) {
                if ( ! cell.cut_leaf( pb, f, nd.beg, nd.end ) )
                    return;
            } else {
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

// ---- the end of a cell ------------------------------------------------------------------------------------------------

/// the seed and the box in double, in the seed's frame: what the re-solve and the finish read
struct Origin { double x, y, z, w, b0[ 3 ], b1[ 3 ]; };

template<class Pb>
__device__ __forceinline__ Origin origin3( const Pb &pb, typename Pb::TR k ) {
    Origin o;
    o.x = double( pb.pos64( k, 0 ) );
    o.y = double( pb.pos64( k, 1 ) );
    o.z = double( pb.pos64( k, 2 ) );
    if constexpr ( Pb::W ) o.w = double( pb.w64( k ) ); else o.w = 0;
    const double p[ 3 ] = { o.x, o.y, o.z };
    for ( int d = 0; d < 3; ++d ) {
        o.b0[ d ] = double( pb.box_min( d ) ) - p[ d ];
        o.b1[ d ] = double( pb.box_max( d ) ) - p[ d ];
    }
    return o;
}

/// THE PLANE OF A CUT, RE-READ in double from its identifier ( seed's frame, OUTWARD normal ): the bisector `n = p_j - p_k`
/// ( `|n|` the distance of the seeds ), or a side of the box ( a unit normal )
template<class Pb>
__device__ __forceinline__ void plane64( const Pb &pb, const Origin &o, typename Pb::TR cid, double *pl ) {
    if ( cid >= 0 ) {
        const double nx = double( pb.pos64( cid, 0 ) ) - o.x, ny = double( pb.pos64( cid, 1 ) ) - o.y, nz = double( pb.pos64( cid, 2 ) ) - o.z;
        double off = 0.5 * ( nx * nx + ny * ny + nz * nz );
        if constexpr ( Pb::W )
            off += 0.5 * ( o.w - double( pb.w64( cid ) ) );
        pl[ 0 ] = nx; pl[ 1 ] = ny; pl[ 2 ] = nz; pl[ 3 ] = off;
        return;
    }
    const int f = int( -1 - cid ), ax = f >> 1, hi = f & 1;
    pl[ 0 ] = ax == 0 ? ( hi ? 1.0 : -1.0 ) : 0.0;
    pl[ 1 ] = ax == 1 ? ( hi ? 1.0 : -1.0 ) : 0.0;
    pl[ 2 ] = ax == 2 ? ( hi ? 1.0 : -1.0 ) : 0.0;
    const double b0 = ax == 0 ? o.b0[ 0 ] : ax == 1 ? o.b0[ 1 ] : o.b0[ 2 ];   // ( selects: no dynamic index into `o` )
    const double b1 = ax == 0 ? o.b1[ 0 ] : ax == 1 ? o.b1[ 1 ] : o.b1[ 2 ];
    pl[ 3 ] = hi ? b1 : -b0;
}

/// THE VERTEX RE-SOLVED from the planes of its three cuts ( `p0`, `p1`, `p2`: `nx, ny, nz, off` ), kept only near the
/// kernel's vertex `( x, y, z )` -- near meaning what a float vertex can be off by, `1e3 eps L / sin`, `L` the farthest
/// corner of the box ( a float vertex is interpolated from what were once the box's corners ): where the float decided a
/// topology the double would not, three planes consecutive in float only can meet anywhere
__device__ __forceinline__ void resolve( const double *p0, const double *p1, const double *p2, double x, double y, double z, double tol2,
                                         double &vx, double &vy, double &vz ) {
    const double c12x = p1[ 1 ] * p2[ 2 ] - p1[ 2 ] * p2[ 1 ], c12y = p1[ 2 ] * p2[ 0 ] - p1[ 0 ] * p2[ 2 ], c12z = p1[ 0 ] * p2[ 1 ] - p1[ 1 ] * p2[ 0 ];
    const double c20x = p2[ 1 ] * p0[ 2 ] - p2[ 2 ] * p0[ 1 ], c20y = p2[ 2 ] * p0[ 0 ] - p2[ 0 ] * p0[ 2 ], c20z = p2[ 0 ] * p0[ 1 ] - p2[ 1 ] * p0[ 0 ];
    const double c01x = p0[ 1 ] * p1[ 2 ] - p0[ 2 ] * p1[ 1 ], c01y = p0[ 2 ] * p1[ 0 ] - p0[ 0 ] * p1[ 2 ], c01z = p0[ 0 ] * p1[ 1 ] - p0[ 1 ] * p1[ 0 ];
    const double det = p0[ 0 ] * c12x + p0[ 1 ] * c12y + p0[ 2 ] * c12z;
    const double e = ( p0[ 0 ] * p0[ 0 ] + p0[ 1 ] * p0[ 1 ] + p0[ 2 ] * p0[ 2 ] ) * ( p1[ 0 ] * p1[ 0 ] + p1[ 1 ] * p1[ 1 ] + p1[ 2 ] * p1[ 2 ] )
                   * ( p2[ 0 ] * p2[ 0 ] + p2[ 1 ] * p2[ 1 ] + p2[ 2 ] * p2[ 2 ] );
    vx = x; vy = y; vz = z;
    const double det2 = det * det;
    if ( ! ( det2 > DET_MIN * DET_MIN * e ) )
        return;
    const double inv = __drcp_rn( det );
    const double rx = ( p0[ 3 ] * c12x + p1[ 3 ] * c20x + p2[ 3 ] * c01x ) * inv;
    const double ry = ( p0[ 3 ] * c12y + p1[ 3 ] * c20y + p2[ 3 ] * c01y ) * inv;
    const double rz = ( p0[ 3 ] * c12z + p1[ 3 ] * c20z + p2[ 3 ] * c01z ) * inv;
    const double dx = rx - x, dy = ry - y, dz = rz - z;
    if ( ( dx * dx + dy * dy + dz * dz ) * det2 > tol2 * e )
        return;
    vx = rx; vy = ry; vz = rz;
}

/// the topology of a finished cell: packed bytes ( the register passes ) ...
struct PackedTopo {
    const unsigned *k, *nn;
    __device__ __forceinline__ int cut( int i, int j ) const { return byte_of( k[ i ], j ); }
    __device__ __forceinline__ int nbr( int i, int j ) const { return byte_of( nn[ i ], j ); }
};
/// ... or `int`s ( the global pass )
struct WideTopo {
    const int *k, *nn;
    __device__ __forceinline__ int cut( int i, int j ) const { return k[ 3 * i + j ]; }
    __device__ __forceinline__ int nbr( int i, int j ) const { return nn[ 3 * i + j ]; }
};

/// A FINISHED CELL as the finish reads it: the vertices in double ( seed's frame ), their topology, per cut its plane
/// ( `pl[ 4 c ]` ), identifier, lowest vertex ( `INT_MAX`: a dead cut ) and number of vertices
template<class TR,class Topo>
struct Fin {
    const double *vx, *vy, *vz;
    Topo          topo;
    const double *pl;
    const TR     *cid;
    const int    *first;                                 ///< ( null: found by a scan of the vertices )
    const int    *count;                                 ///< the vertices of each face
    int           nv, nc;
};

/// the slot of `i` whose cut is `f`, of `i` whose neighbour is `v`
template<class Topo> __device__ __forceinline__ int cut_slot( const Topo &t, int i, int f ) { return t.cut( i, 0 ) == f ? 0 : t.cut( i, 1 ) == f ? 1 : 2; }
template<class Topo> __device__ __forceinline__ int nbr_slot( const Topo &t, int i, int v ) { return t.nbr( i, 0 ) == v ? 0 : t.nbr( i, 1 ) == v ? 1 : 2; }

/// THE END OF A CELL ( every lane of the warp ): each lane walks faces `lane, lane + 32, ...` -- the fan from the face's
/// lowest vertex, the tetrahedra from `g`, the face's sums signed by its own volume -- then the totals by a butterfly
/// ( the same bits on every lane ), written by lane 0. The upper facets go to the COO, room taken by one atomic per warp.
template<class Pb,class Topo>
__device__ __forceinline__ void finish_warp( const Pb &pb, typename Pb::TR k, const Fin<typename Pb::TR,Topo> &c, int lane, const Origin &o ) {
    using TR = typename Pb::TR;
    using TF = std::remove_reference_t<decltype( pb.res( 0 ) )>;
    constexpr unsigned OUT = Pb::OUT;
    constexpr bool FAC = OUT & FACETS, ADJ = OUT & VJP, MOM = OUT & MOMENTS;
    const double rho = pb.density();

    double gx = 0, gy = 0, gz = 0;
    for ( int i = lane; i < c.nv; i += 32 ) { gx += c.vx[ i ]; gy += c.vy[ i ]; gz += c.vz[ i ]; }
    gx = warp_sum( gx ); gy = warp_sum( gy ); gz = warp_sum( gz );
    if ( c.nv > 0 ) {
        const double inv = 1.0 / c.nv;
        gx *= inv; gy *= inv; gz *= inv;
    }
    double gk = 0;
    if constexpr ( ADJ )
        gk = double( pb.g( SI( pb.ids( k ) ) ) );

    // the lowest vertex of face `f` and how many vertices it has: from the tables if the cell has them ( the global pass ),
    // else by a scan of the vertices ( the register passes: their shared memory is what bounds their occupancy )
    auto face_of = [&]( int f, int &v0, int &cnt ) {
        if ( c.first ) {
            v0 = c.first[ f ];
            cnt = c.count[ f ];
            return;
        }
        v0 = INT_MAX;
        cnt = 0;
        for ( int a = c.nv - 1; a >= 0; --a ) {
            const bool on = c.topo.cut( a, 0 ) == f || c.topo.cut( a, 1 ) == f || c.topo.cut( a, 2 ) == f;
            v0 = on ? a : v0;
            cnt += on;
        }
    };

    unsigned long long fbase = 0;
    bool fok = false;
    if constexpr ( FAC ) {
        int nup = 0;
        for ( int f = lane; f < c.nc; f += 32 ) {
            int v0, cnt;
            face_of( f, v0, cnt );
            nup += v0 != INT_MAX && c.cid[ f ] > k;
        }
        int incl = nup;
#pragma unroll
        for ( int d = 1; d < 32; d *= 2 ) {
            const int v = __shfl_up_sync( FULL, incl, d );
            if ( lane >= d ) incl += v;
        }
        const int tot = __shfl_sync( FULL, incl, 31 );
        unsigned long long base = 0;
        if ( lane == 0 && tot )
            base = atomicAdd( &pb.counters->nb_facets, ( unsigned long long ) tot );
        base = __shfl_sync( FULL, base, 0 );
        fbase = base + unsigned( incl - nup );
        fok = base + unsigned( tot ) <= pb.fcap;
    }

    double vol = 0, m1x = 0, m1y = 0, m1z = 0, m2 = 0, gw = 0, gpx = 0, gpy = 0, gpz = 0;
    unsigned t = 0;
    for ( int f = lane; f < c.nc; f += 32 ) {
        int v0, cntf;
        face_of( f, v0, cntf );
        if ( v0 == INT_MAX )
            continue;
        const double nx = c.pl[ 4 * f ], ny = c.pl[ 4 * f + 1 ], nz = c.pl[ 4 * f + 2 ];
        const double ox = c.vx[ v0 ], oy = c.vy[ v0 ], oz = c.vz[ v0 ];
        const double qx = ox - gx, qy = oy - gy, qz = oz - gz;
        double fv = 0, fw = 0, fcx = 0, fcy = 0, fcz = 0, f1x = 0, f1y = 0, f1z = 0, f2 = 0;
        // the triangle ( o, a, b ) of the fan, the tetrahedron ( g, o, a, b ), oriented by the order of `a` and `b`
        auto tri = [&]( int a, int b ) {
            const double ax = c.vx[ a ], ay = c.vy[ a ], az = c.vz[ a ];
            const double bx = c.vx[ b ], by = c.vy[ b ], bz = c.vz[ b ];
            const double e1x = ax - ox, e1y = ay - oy, e1z = az - oz, e2x = bx - ox, e2y = by - oy, e2z = bz - oz;
            const double cx = e1y * e2z - e1z * e2y, cy = e1z * e2x - e1x * e2z, cz = e1x * e2y - e1y * e2x;
            const double det = qx * cx + qy * cy + qz * cz;          // 6 vol( g, o, a, b ), signed
            fv += det;
            if constexpr ( FAC || ADJ ) {
                const double wt = cx * nx + cy * ny + cz * nz;       // 2 |triangle| |n|, signed
                fw += wt;
                if constexpr ( ADJ ) {
                    fcx += wt * ( ox + ax + bx );
                    fcy += wt * ( oy + ay + by );
                    fcz += wt * ( oz + az + bz );
                }
            }
            if constexpr ( MOM ) {
                // int x = V ( g + o + a + b ) / 4, int |x|^2 = V / 10 ( sum of the squares and of the products of the four )
                const double sx = gx + ox + ax + bx, sy = gy + oy + ay + by, sz = gz + oz + az + bz;
                f1x += det * sx; f1y += det * sy; f1z += det * sz;
                const double sq = gx * gx + gy * gy + gz * gz + ox * ox + oy * oy + oz * oz + ax * ax + ay * ay + az * az + bx * bx + by * by + bz * bz;
                f2 += det * ( 0.5 * ( sx * sx + sy * sy + sz * sz + sq ) );
            }
        };
        // THE WALK around the face from its lowest vertex: one cycle, oriented by the sign of its volume
        const int io = cut_slot( c.topo, v0, f );
        int prev = v0, at = c.topo.nbr( v0, io == 0 ? 1 : 0 ), seen = 2;
        for ( int step = 0; step < c.nv; ++step ) {
            const int i1 = cut_slot( c.topo, at, f ), e = nbr_slot( c.topo, at, prev );
            const int nxt = c.topo.nbr( at, 3 - i1 - e );
            if ( nxt == v0 )
                break;
            tri( at, nxt );
            ++seen;
            prev = at;
            at = nxt;
        }
        double sg = fv < 0 ? -1.0 : 1.0;
        if ( seen != cntf ) {
            // SEVERAL CYCLES ( a degenerate input: cuts through vertices that are on them, decided at the rounding, may leave
            // a face in pieces ): every edge of the face, oriented by the planes ( `n_f x n_g` along the boundary of `f`,
            // `g` the edge's other face ), each piece's fan from the same `o` -- the sum is the face's whatever the pieces
            fv = fw = fcx = fcy = fcz = f1x = f1y = f1z = f2 = 0;
            sg = 1;
            for ( int a = 0; a < c.nv; ++a ) {
                const int ia = cut_slot( c.topo, a, f );
                if ( c.topo.cut( a, ia ) != f )
                    continue;
                for ( int j = 0; j < 3; ++j ) {
                    const int b = c.topo.nbr( a, j );
                    if ( j == ia || b < a )
                        continue;
                    const int gf = c.topo.cut( a, 3 - ia - j );
                    const double mx = c.pl[ 4 * gf ], my = c.pl[ 4 * gf + 1 ], mz = c.pl[ 4 * gf + 2 ];
                    const double dx = ny * mz - nz * my, dy = nz * mx - nx * mz, dz = nx * my - ny * mx;
                    const double dir = ( c.vx[ b ] - c.vx[ a ] ) * dx + ( c.vy[ b ] - c.vy[ a ] ) * dy + ( c.vz[ b ] - c.vz[ a ] ) * dz;
                    if ( dir >= 0 ) tri( a, b ); else tri( b, a );
                }
            }
        }
        vol += sg * fv;
        if constexpr ( MOM ) {
            m1x += sg * f1x; m1y += sg * f1y; m1z += sg * f1z;
            m2 += sg * f2;
        }
        if constexpr ( FAC || ADJ ) {
            const TR j = c.cid[ f ];
            if ( j >= 0 ) {
                const double n2 = nx * nx + ny * ny + nz * nz;
                const double coef = n2 > 0 ? rho * fabs( fw ) / ( 4 * n2 ) : 0.0;   // rho |facet| / ( 2 |p_k - p_j| )
                if constexpr ( FAC ) {
                    if ( j > k && fok ) {
                        const unsigned long long q = fbase + t++;
                        pb.fi[ q ] = k;
                        pb.fj[ q ] = j;
                        pb.fc[ q ] = coef;
                    }
                }
                if constexpr ( ADJ ) {
                    // `2 c ( g_k - g_j ) ( x_kj - p_k )`: `|facet| x_kj = s fc / ( 6 |n| )`
                    const double dg = gk - double( pb.g( SI( pb.ids( j ) ) ) );
                    gw += coef * dg;
                    const double h = n2 > 0 ? dg * rho * ( fw < 0 ? -1.0 : 1.0 ) / ( 6 * n2 ) : 0.0;
                    gpx += h * fcx;
                    gpy += h * fcy;
                    gpz += h * fcz;
                }
            }
        }
    }
    vol = warp_sum( vol ) / 6;
    if constexpr ( MOM ) {
        m1x = warp_sum( m1x ) / 24; m1y = warp_sum( m1y ) / 24; m1z = warp_sum( m1z ) / 24;
        m2 = warp_sum( m2 ) / 60;
    }
    if constexpr ( ADJ ) {
        gw = warp_sum( gw );
        gpx = warp_sum( gpx ); gpy = warp_sum( gpy ); gpz = warp_sum( gpz );
    }
    if ( lane != 0 )
        return;
    if constexpr ( bool( OUT & MEASURES ) )
        pb.res( pb.user( k ) ) = TF( rho * vol );
    if constexpr ( ADJ ) {
        if ( pb.grad_pos.p ) {
            pb.grad_pos( SI( k ), 0 ) = TF( gpx );
            pb.grad_pos( SI( k ), 1 ) = TF( gpy );
            pb.grad_pos( SI( k ), 2 ) = TF( gpz );
        }
        if constexpr ( Pb::W )
            if ( pb.grad_w.p )
                pb.grad_w( SI( k ) ) = TF( gw );
    }
    if constexpr ( MOM ) {
        const SI u = pb.user( k );
        pb.bary( u, 0 ) = TF( vol > 0 ? o.x + m1x / vol : o.x );
        pb.bary( u, 1 ) = TF( vol > 0 ? o.y + m1y / vol : o.y );
        pb.bary( u, 2 ) = TF( vol > 0 ? o.z + m1z / vol : o.z );
        pb.cost( u ) = TF( rho * m2 );
    }
}

/// A CELL THAT COULD NOT BE DONE: zeros where it writes ( never a NaN: the error buffer says why )
template<class Pb>
__device__ __forceinline__ void failed3( const Pb &pb, typename Pb::TR k ) {
    using TF = std::remove_reference_t<decltype( pb.res( 0 ) )>;
    constexpr unsigned OUT = Pb::OUT;
    if constexpr ( bool( OUT & MEASURES ) )
        pb.res( pb.user( k ) ) = TF( 0 );
    if constexpr ( bool( OUT & VJP ) ) {
        if ( pb.grad_pos.p ) for ( int d = 0; d < 3; ++d ) pb.grad_pos( SI( k ), d ) = TF( 0 );
        if constexpr ( Pb::W ) if ( pb.grad_w.p ) pb.grad_w( SI( k ) ) = TF( 0 );
    }
    if constexpr ( bool( OUT & MOMENTS ) ) {
        const SI u = pb.user( k );
        for ( int d = 0; d < 3; ++d ) pb.bary( u, d ) = TF( 0 );
        pb.cost( u ) = TF( 0 );
    }
}

/// the tolerance of the re-solve, `( 1e3 eps_float L )^2`, `L` the farthest corner of the box from the seed
__device__ __forceinline__ double resolve_tol2( const Origin &o ) {
    double L2 = 0;
    for ( int d = 0; d < 3; ++d )
        L2 += fmax( o.b0[ d ] * o.b0[ d ], o.b1[ d ] * o.b1[ d ] );
    return 1e-8 * L2;
}

/// the shared memory of a warp of a register pass: the finished cell ( `Fin` ) of `32 S` vertices and cuts
template<class TR,int S>
constexpr int reg_shared_bytes() {
    constexpr int V = 32 * S;
    return ( 8 * ( 3 * V + 4 * V ) + 4 * 2 * V + int( sizeof( TR ) ) * V + 15 ) / 16 * 16;
}

/// THE END OF A REGISTER CELL: its planes and ( re-solved ) vertices to the warp's shared memory `sh`, then `finish_warp`
template<class Pb,int S>
__device__ __forceinline__ void finish_regs( const Pb &pb, typename Pb::TR k, const RegCell<typename Pb::TK,typename Pb::TR,S> &cell, unsigned char *sh ) {
    using TK = typename Pb::TK;
    using TR = typename Pb::TR;
    constexpr int V = 32 * S;
    double   *vx = reinterpret_cast<double *>( sh ), *vy = vx + V, *vz = vy + V, *pl = vz + V;
    unsigned *kk = reinterpret_cast<unsigned *>( pl + 4 * V ), *nn = kk + V;
    TR       *cid = reinterpret_cast<TR *>( nn + V );
    const int lane = cell.lane;
    const Origin o = origin3( pb, k );
    const int nv = cell.state == CUT_EMPTY ? 0 : cell.nv, nc = nv ? cell.nc : 0;
#pragma unroll
    for ( int s = 0; s < S; ++s ) {
        const int c = s * 32 + lane;
        if ( c < nc ) {
            plane64( pb, o, cell.cid[ s ], pl + 4 * c );
            cid[ c ] = cell.cid[ s ];
        }
    }
    __syncwarp();
    const double tol2 = resolve_tol2( o );
#pragma unroll
    for ( int s = 0; s < S; ++s ) {
        const int i = s * 32 + lane;
        if ( i < nv ) {
            const unsigned kw = cell.k[ s ];
            const int a = byte_of( kw, 0 ), b = byte_of( kw, 1 ), q = byte_of( kw, 2 );
            double rx, ry, rz;
            resolve( pl + 4 * a, pl + 4 * b, pl + 4 * q, double( cell.x[ s ] ), double( cell.y[ s ] ), double( cell.z[ s ] ), tol2, rx, ry, rz );
            vx[ i ] = rx; vy[ i ] = ry; vz[ i ] = rz;
            kk[ i ] = kw;
            nn[ i ] = cell.nn[ s ];
        }
    }
    __syncwarp();
    finish_warp( pb, k, Fin<TR,PackedTopo>{ vx, vy, vz, PackedTopo{ kk, nn }, pl, cid, nullptr, nullptr, nv, nc }, lane, o );
    __syncwarp();                                        // ( the shared memory is the next cell's )
}

/// THE END OF A GLOBAL-MEMORY CELL, the same in its slot
template<class Pb>
__device__ __forceinline__ void finish_mem( const Pb &pb, typename Pb::TR k, MemCell<typename Pb::TK,typename Pb::TR> &cell ) {
    using TK = typename Pb::TK;
    using TR = typename Pb::TR;
    const int lane = cell.lane;
    const Origin o = origin3( pb, k );
    const int nv = cell.state == CUT_EMPTY ? 0 : cell.nv, nc = nv ? cell.nc : 0;
    for ( int c = lane; c < nc; c += 32 ) {
        plane64( pb, o, cell.cid[ c ], cell.pl + 4 * c );
        cell.aux[ c ] = INT_MAX;
        cell.cnt[ c ] = 0;
    }
    __syncwarp();
    const double tol2 = resolve_tol2( o );
    const int *K = cell.K[ cell.cur ];
    for ( int i = lane; i < nv; i += 32 ) {
        const int a = K[ 3 * i ], b = K[ 3 * i + 1 ], q = K[ 3 * i + 2 ];
        const double x = cell.X[ cell.cur ][ i ], y = cell.Y[ cell.cur ][ i ], z = cell.Z[ cell.cur ][ i ];
        resolve( cell.pl + 4 * a, cell.pl + 4 * b, cell.pl + 4 * q, x, y, z, tol2, cell.vx[ i ], cell.vy[ i ], cell.vz[ i ] );
        atomicMin( cell.aux + a, i );
        atomicMin( cell.aux + b, i );
        atomicMin( cell.aux + q, i );
        atomicAdd( cell.cnt + a, 1 );
        atomicAdd( cell.cnt + b, 1 );
        atomicAdd( cell.cnt + q, 1 );
    }
    __syncwarp();
    finish_warp( pb, k, Fin<TR,WideTopo>{ cell.vx, cell.vy, cell.vz, WideTopo{ K, cell.N[ cell.cur ] }, cell.pl, cell.cid, cell.aux, cell.cnt, nv, nc }, lane, o );
    __syncwarp();
}

// ---- the kernels --------------------------------------------------------------------------------------------------

/// the box in the seed's frame, in the kernel's float ( `x0, y0, z0, x1, y1, z1` )
template<class Pb>
__device__ __forceinline__ void start3( const Pb &pb, typename Pb::TR k, typename Pb::TK ( &b )[ 6 ] ) {
    using TK = typename Pb::TK;
    for ( int d = 0; d < 3; ++d ) {
        const double p = double( pb.pos64( k, d ) );
        b[ d ] = TK( double( pb.box_min( d ) ) - p );
        b[ 3 + d ] = TK( double( pb.box_max( d ) ) - p );
    }
}

/// a register pass: the cells `list[ i ]` ( or `i` without a list ), `i < count`, one per warp, the warps striding;
/// a cell that overflows `32 S` vertices goes to `ovf`
template<class Pb,int S>
__device__ __forceinline__ void reg_cells( const Pb &pb, const typename Pb::TR *list, unsigned long long count, typename Pb::TR *ovf,
                                           int ovf_counter, unsigned char *sh ) {
    using TK = typename Pb::TK;
    using TR = typename Pb::TR;
    const int lane = threadIdx.x & 31;
    const unsigned long long gwarp = ( SI( blockIdx.x ) * blockDim.x + threadIdx.x ) >> 5, nwarps = ( SI( gridDim.x ) * blockDim.x ) >> 5;
    for ( unsigned long long i = gwarp; i < count; i += nwarps ) {
        const TR k = list ? list[ i ] : TR( i );
        if ( pb.only_global ) {
            if ( lane == 0 )
                ovf[ atomicAdd( &pb.counters->ovf[ ovf_counter ], 1ull ) ] = k;
            continue;
        }
        const typename Pb::SeedT f = pb.seeds[ k ];
        TK b[ 6 ];
        start3( pb, k, b );
        RegCell<TK,TR,S> cell;
        cell.init( lane, b );
        walk3( pb, f, cell );
        if ( cell.state == CUT_OVERFLOW ) {
            if ( lane == 0 )
                ovf[ atomicAdd( &pb.counters->ovf[ ovf_counter ], 1ull ) ] = k;
            continue;
        }
        finish_regs<Pb,S>( pb, k, cell, sh );
    }
}

/// FIRST PASS: every cell, a warp each, rank order ( neighbouring warps walk neighbouring cells )
template<class Pb>
__global__ void __launch_bounds__( BLOCK, first_blocks<typename Pb::TK> ) first_pass( Pb pb, typename Pb::TR *ovf ) {
    extern __shared__ __align__( 16 ) unsigned char shared_bytes_[];
    reg_cells<Pb,S1>( pb, nullptr, ( unsigned long long ) pb.n, ovf, 0,
                      shared_bytes_ + ( threadIdx.x >> 5 ) * reg_shared_bytes<typename Pb::TR,S1>() );
}

/// SECOND PASS: the cells the first one could not hold, `S2` slots; its grid is what the card holds at once and it
/// strides over a count only the card knows
template<class Pb>
__global__ void __launch_bounds__( BLOCK ) second_pass( Pb pb, const typename Pb::TR *list, typename Pb::TR *ovf ) {
    extern __shared__ __align__( 16 ) unsigned char shared_bytes_[];
    reg_cells<Pb,S2>( pb, list, pb.counters->ovf[ 0 ], ovf, 1,
                      shared_bytes_ + ( threadIdx.x >> 5 ) * reg_shared_bytes<typename Pb::TR,S2>() );
}

/// THIRD PASS: the cells past `32 S2` vertices, a warp each in a slot of global memory ( `MemCell` ), the grid exactly the
/// slots: each warp takes the cells `gwarp, gwarp + nb_warps, ...` one after the other -- batches, as many as the list
/// needs. Past the slot ( the per-cell limit ), or a topology the walk cannot follow: a FAILURE.
template<class Pb,class EB>
__global__ void __launch_bounds__( BLOCK ) third_pass( Pb pb, const typename Pb::TR *list, unsigned char *slots, int capv, int capc, EB errors ) {
    using TK = typename Pb::TK;
    using TR = typename Pb::TR;
    const int lane = threadIdx.x & 31;
    const SI gwarp = ( SI( blockIdx.x ) * blockDim.x + threadIdx.x ) >> 5, nwarps = ( SI( gridDim.x ) * blockDim.x ) >> 5;
    unsigned char *mine = slots + gwarp * MemCell<TK,TR>::bytes_for( capv, capc );
    const unsigned long long m = pb.counters->ovf[ 1 ];
    for ( unsigned long long i = gwarp; i < m; i += nwarps ) {
        const TR k = list[ i ];
        const typename Pb::SeedT f = pb.seeds[ k ];
        TK b[ 6 ];
        start3( pb, k, b );
        MemCell<TK,TR> cell;
        cell.attach( mine, capv, capc );
        cell.init( lane, b );
        walk3( pb, f, cell );
        if ( cell.state == CUT_OVERFLOW || cell.state == CUT_BROKEN ) {
            if ( lane == 0 ) {
                failed3( pb, k );
                if ( atomicAdd( &pb.counters->nb_failed, 1ull ) == 0 )
                    errors.record( ERROR_KIND_FAILURE, cell.state == CUT_OVERFLOW ? FAIL_TOO_MANY_VERTICES : FAIL_TOPOLOGY, SI( pb.ids( k ) ) );
            }
            __syncwarp();
            continue;
        }
        finish_mem( pb, k, cell );
    }
}

/// THE KERNEL'S TREE from the tree's tensors ( a thread per node ): the box rounded OUTWARD, the majorant constant UP
template<bool W,class TR,class TB,class TJ>
__global__ void __launch_bounds__( BLOCK ) make_nodes( SI nb_nodes, Strided<TB,3> box, Strided<TB,2> wa, Strided<TB,1> wb,
                                                      Strided<TJ,1> beg, Strided<TJ,1> end, Node<W,TR> *out ) {
    const SI i = SI( blockIdx.x ) * blockDim.x + threadIdx.x;
    if ( i >= nb_nodes )
        return;
    auto down = []( TB v ) -> float { if constexpr ( ! std::is_same_v<TB,float> ) return __double2float_rd( v ); else return v; };
    auto up   = []( TB v ) -> float { if constexpr ( ! std::is_same_v<TB,float> ) return __double2float_ru( v ); else return v; };
    Node<W,TR> nd;
    for ( int d = 0; d < 3; ++d ) {
        nd.lo[ d ] = down( box( i, 0, d ) );
        nd.hi[ d ] = up  ( box( i, 1, d ) );
    }
    if constexpr ( W ) {
        for ( int d = 0; d < 3; ++d )
            nd.a[ d ] = float( wa( i, d ) );
        nd.b = up( wb( i ) );
    }
    nd.beg = TR( beg( i ) );
    nd.end = TR( end( i ) );
    out[ i ] = nd;
}

/// THE KERNEL'S SEEDS: two floats per coordinate and weight for the float kernel, the doubles for the double one
template<class TK,bool W,class TF>
__global__ void __launch_bounds__( BLOCK ) pack_seeds( SI n, Strided<TF,2> pos, Strided<TF,1> w, Seed<TK> *out ) {
    const SI k = SI( blockIdx.x ) * blockDim.x + threadIdx.x;
    if ( k >= n )
        return;
    const double x = double( pos( k, 0 ) ), y = double( pos( k, 1 ) ), z = double( pos( k, 2 ) );
    double v = 0;
    if constexpr ( W )
        v = double( w( k ) );
    if constexpr ( std::is_same_v<TK,float> ) {
        SeedF s;
        s.xh = __double2float_rn( x ); s.xl = __double2float_rn( x - double( s.xh ) );
        s.yh = __double2float_rn( y ); s.yl = __double2float_rn( y - double( s.yh ) );
        s.zh = __double2float_rn( z ); s.zl = __double2float_rn( z - double( s.zh ) );
        s.wh = __double2float_rn( v ); s.wl = __double2float_rn( v - double( s.wh ) );
        out[ k ] = s;
    } else
        out[ k ] = SeedD{ x, y, z, v };
}

/// NEW WEIGHTS INTO THE KERNEL'S SEEDS ( the card's Newton: `w` in ranks, a double per seed ), the positions left alone
template<class TK>
__global__ void __launch_bounds__( BLOCK ) pack_seed_weights( SI n, const double *w, Seed<TK> *out ) {
    const SI k = SI( blockIdx.x ) * blockDim.x + threadIdx.x;
    if ( k >= n )
        return;
    const double v = w[ k ];
    if constexpr ( std::is_same_v<TK,float> ) {
        out[ k ].wh = __double2float_rn( v );
        out[ k ].wl = __double2float_rn( v - double( out[ k ].wh ) );
    } else
        out[ k ].w = v;
}

// ---- the host side ------------------------------------------------------------------------------------------------

/// THE THIRD PASS'S BUDGET: `warps` slots of `capv` vertices in global memory, taken once per call ( Python's choice,
/// `PowerDiagram_Bsp.card_overflow_warps_for` ): `capv = min( max_vertices, 2 n + 8 )` -- a simple polytope with `F` faces
/// has `2 F - 4` vertices, and a cell of `n` seeds in a box has at most `n + 5` faces --, `capc = capv / 2 + 64` cuts
/// ( the live ones, `F`, and room for the dead ones between two compactions ). What does not fit in the slots at once
/// waits for a slot to be free: a matter of speed on rare cells, never a reason to run again.
struct Slots {
    unsigned char *ptr = nullptr;
    SI             bytes = 0;                            ///< what `ptr` holds
    int            capv = 0, capc = 0, warps = 0;

    static Slots sized( SI n, int warps, int max_vertices ) {
        Slots o;
        o.capv  = int( std::max<SI>( 64, std::min<SI>( max_vertices, 2 * n + 8 ) ) );
        o.capc  = o.capv / 2 + 64;
        o.warps = ( std::max( 1, warps ) + BLOCK / 32 - 1 ) / ( BLOCK / 32 ) * ( BLOCK / 32 );
        return o;
    }
    template<class TK,class TR>
    SI bytes_for() const { return MemCell<TK,TR>::bytes_for( capv, capc ) * warps; }

    /// room for the slots of a `< TK, TR >` cell from the pool ( `false`: the pool said no )
    template<class TK,class TR>
    bool take_from( auto &allocator ) {
        bytes = bytes_for<TK,TR>();
        ptr = static_cast<unsigned char *>( take( allocator, bytes ) );
        return ptr != nullptr;
    }
};

/// WHAT A CARD OF THE CALL MAY BORROW FROM ANOTHER ( the card's Newton, `Newton2D.cuh`: the float kernel's card, the double
/// one's, the moments' ): the node records ( written by the majorants, float for every kernel ), the passes' lists, and the
/// packed seeds of the same kernel float. `nullptr`: taken.
template<class TR>
struct CardShare3 {
    const void *nodes = nullptr;
    const void *seeds = nullptr;
    TR         *lists = nullptr;
};

/// THE DIAGRAM ON THE CARD, for one call: the kernel's tree and seeds, the lists and counters of the passes, the slots of
/// the third pass. `prepare` takes it all from the call's allocator and fills the tree and the seeds; `run` launches the
/// passes ( everything `pb` points to as outputs is the caller's ). Nothing is read back ( but with `SDOT_CARD_STATS=1` ).
template<class V,bool W,unsigned OUT,class TF,class TI>
struct Card {
    using Pb    = Problem<V,TF,TI,W,OUT>;
    using TK    = typename V::TK;
    using TR    = typename V::TR;
    using SeedT = typename Pb::SeedT;

    Pb        pb{};
    TR       *lists = nullptr;                           ///< two lists of `n` ranks
    Counters *counters = nullptr;
    Slots     slots{};
    SI        nb_nodes = 0;

    /// `false`: the pool said no ( `allocator.failed` is reported )
    bool prepare( const CudaQueue &queue, const auto &pd, auto &allocator, int max_vertices, int overflow_warps ) {
        return prepare( queue, pd, allocator, Slots::sized( SI( pd.nb_seeds() ), overflow_warps, max_vertices ) );
    }

    /// `slots_`: the third pass's slots, already taken ( shared with another card of the call: they must hold this card's
    /// cells, `bytes_for< TK, TR >` ), or only sized ( `ptr == nullptr` ): taken here. `share`: what is borrowed from another
    /// card of the call instead of taken ( `CardShare3` ). `false`: the pool said no.
    bool prepare( const CudaQueue &queue, const auto &pd, auto &allocator, Slots slots_, const CardShare3<TR> &share = {} ) {
        static_assert( std::decay_t<decltype( pd )>::ct_dim == 3, "the 3D cells of the card" );
        const SI n = SI( pd.nb_seeds() );
        nb_nodes = SI( pd.tree.node_begin.shape( 0 ) );
        int depth = 0;
        for ( SI m = nb_nodes; m; m >>= 1 )
            ++depth;
        if ( depth > V::MAX_HEIGHT || ( std::is_same_v<typename V::TN,int> && nb_nodes > SI( 0x7fffffff ) )
                                   || ( std::is_same_v<TR,int> && n > SI( 0x7fffffff ) - 8 ) )
            throw std::runtime_error( "sdot::gpu3d: the variant chosen does not hold this tree ( see `PowerDiagram_Bsp._card_variant` )" );
        slots = slots_;
        if ( slots.ptr && slots.bytes < slots.template bytes_for<TK,TR>() )
            throw std::runtime_error( "sdot::gpu3d: shared slots too small for this card" );
        if ( ! slots.ptr && ! slots.template take_from<TK,TR>( allocator ) )
            return false;
        auto *nodes = share.nodes ? static_cast<Node<W,TR> *>( const_cast<void *>( share.nodes ) )
                                  : static_cast<Node<W,TR> *>( take( allocator, SI( sizeof( Node<W,TR> ) ) * std::max<SI>( nb_nodes, 1 ) ) );
        auto *seeds = share.seeds ? static_cast<SeedT *>( const_cast<void *>( share.seeds ) )
                                  : static_cast<SeedT *>( take( allocator, SI( sizeof( SeedT ) ) * std::max<SI>( n, 1 ) ) );
        lists    = share.lists ? share.lists : static_cast<TR *>( take( allocator, SI( sizeof( TR ) ) * 2 * std::max<SI>( n, 1 ) ) );
        counters = static_cast<Counters *>( take( allocator, SI( sizeof( Counters ) ) ) );
        if ( ! nodes || ! seeds || ! lists || ! counters )
            return false;
        pb.nodes   = nodes;
        pb.seeds   = seeds;
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
        static const bool only_global = std::getenv( "SDOT_CARD_GLOBAL_ONLY" ) && *std::getenv( "SDOT_CARD_GLOBAL_ONLY" ) != '0';
        pb.only_global = only_global;
        // what is borrowed is the lender's to fill ( and may already hold more recent values than the diagram's tensors )
        refresh( queue, pd, ! share.nodes, ! share.seeds );
        return true;
    }

    /// what another card of the call may borrow from this one ( `CardShare3` ): the nodes, the lists, and with `seeds` the
    /// packed seeds ( a card of the same kernel float )
    CardShare3<TR> lend( bool seeds ) const {
        CardShare3<TR> s;
        s.nodes = pb.nodes;
        s.lists = lists;
        if ( seeds ) s.seeds = pb.seeds;
        return s;
    }

    /// the kernel's tree and seeds from the diagram's tensors
    void refresh( const CudaQueue &queue, const auto &pd, bool nodes = true, bool seeds = true ) {
        using TB = std::remove_const_t<typename std::decay_t<decltype( pd.tree.node_box )>::TF>;
        using TJ = std::remove_const_t<typename std::decay_t<decltype( pd.tree.node_begin )>::TF>;
        Strided<TF,2> wa{};
        Strided<TF,1> wb{};
        if constexpr ( W ) {
            wa = strided( pd.tree.node_wa );
            wb = strided( pd.tree.node_wb );
        }
        if ( nodes )
            launch_kernel( queue, &make_nodes<W,TR,TB,TJ>, blocks_for( nb_nodes ), BLOCK, 0, nb_nodes, strided( pd.tree.node_box ), wa, wb,
                           strided( pd.tree.node_begin ), strided( pd.tree.node_end ), const_cast<Node<W,TR> *>( pb.nodes ) );
        if ( seeds )
            launch_kernel( queue, &pack_seeds<TK,W,TF>, blocks_for( SI( pb.n ) ), BLOCK, 0, SI( pb.n ), pb.pos64, pb.w64,
                           const_cast<SeedT *>( pb.seeds ) );
    }

    /// THE PASSES. `errors`: the kernel form of loom's error buffer
    template<class EB>
    void run( const CudaQueue &queue, const EB &errors ) {
        const SI n = SI( pb.n );
        zero_fill( queue, counters, SI( sizeof( Counters ) ) );
        if ( n == 0 )
            return;
        TR *list1 = lists, *list2 = lists + n;
        constexpr int W1 = reg_shared_bytes<TR,S1>() * ( BLOCK / 32 ), W2 = reg_shared_bytes<TR,S2>() * ( BLOCK / 32 );
        launch_kernel( queue, &first_pass<Pb>, int( ( n + BLOCK / 32 - 1 ) / ( BLOCK / 32 ) ), BLOCK, W1, pb, list1 );
        static const int grid2 = resident_grid( &second_pass<Pb>, BLOCK, W2 );
        launch_kernel( queue, &second_pass<Pb>, grid2, BLOCK, W2, pb, ( const TR * ) list1, list2 );
        launch_kernel( queue, &third_pass<Pb,EB>, slots.warps * 32 / BLOCK, BLOCK, 0, pb, ( const TR * ) list2, slots.ptr, slots.capv, slots.capc, errors );

        static const bool stats = std::getenv( "SDOT_CARD_STATS" ) && *std::getenv( "SDOT_CARD_STATS" ) != '0';
        if ( stats ) {
            Counters c;
            read_back( queue, &c, counters, 1 );
            std::printf( "[card cells 3d] n %lld, over %d vertices %llu ( %.3f %% ), over %d vertices %llu ( %.4f %% ), "
                         "( third pass: %d slots of %d vertices ), failed %llu, facets %llu\n",
                         ( long long ) n, 32 * S1, c.ovf[ 0 ], 100.0 * c.ovf[ 0 ] / n, 32 * S2, c.ovf[ 1 ], 100.0 * c.ovf[ 1 ] / n,
                         slots.warps, slots.capv, c.nb_failed, c.nb_facets );
        }
    }

    /// the upper facets the cells wanted, times `factor`, into a loom ShapeVar ON THE CARD ( past the capacity loom runs
    /// the call again with more room )
    void report_facets( const CudaQueue &queue, const auto &sv, unsigned long long factor ) const {
        launch_kernel( queue, &gpu2d::report_count<std::decay_t<decltype( sdot::kernel_form( queue, MutList(), sv ) )>>, 1, 1, 0,
                       sdot::kernel_form( queue, MutList(), sv ), &counters->nb_facets, factor );
    }
};

// ---- the entry points of the loom calls ( `PowerDiagram_Bsp.py`, the same as `Cell2D.cuh`'s and `Laplacian2D.cuh`'s ) ---

/// THE MEASURES of `pd`; `rho`: the constant density
template<class V>
void measures( const CudaQueue &queue, const auto &pd, auto &&res, const auto &errors, auto &allocator, const auto &rho,
               int max_vertices, int overflow_warps ) {
    using PD = std::decay_t<decltype( pd )>;
    Card<V,PD::has_weights,MEASURES,TFOf<PD>,TIOf<PD>> card;
    if ( ! card.prepare( queue, pd, allocator, max_vertices, overflow_warps ) )
        return;
    gpu2d::set_density( card.pb, rho );
    card.pb.res = strided_out<TFOf<PD>,1>( res );
    card.run( queue, sdot::kernel_form( queue, MutList(), errors ) );
}

/// THE ADJOINT OF THE MEASURES: `grad_res` ( user order ) -> the gradients of the sorted positions and weights
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
        if constexpr ( has_gp ) zero_fill( queue, const_cast<void *>( ( const void * ) grad_pos.data().raw ), SI( sizeof( TF ) ) * 3 * SI( pd.nb_seeds() ) );
        if constexpr ( has_gw ) zero_fill( queue, const_cast<void *>( ( const void * ) grad_w.data().raw ), SI( sizeof( TF ) ) * SI( pd.nb_seeds() ) );
    } else {
        Card<V,W,VJP,TF,TIOf<PD>> card;
        if ( ! card.prepare( queue, pd, allocator, max_vertices, overflow_warps ) )
            return;
        gpu2d::set_density( card.pb, rho );
        card.pb.g        = strided( grad_res );
        card.pb.grad_pos = strided_out<TF,2>( grad_pos );
        if constexpr ( W )
            card.pb.grad_w = strided_out<TF,1>( grad_w );
        card.run( queue, sdot::kernel_form( queue, MutList(), errors ) );
    }
}

/// THE CELLS AND WHAT IS MADE OF THEM, in one call ( `PowerDiagram_Bsp._card_cells` ): the measures, and per `OUT` the
/// laplacian's CSR ( `Laplacian2D.cuh`'s assembly ), the barycentres and the costs
template<class V,unsigned OUT>
void cells( const CudaQueue &queue, const auto &pd, auto &&res, auto &&lap, auto &&mom, const auto &errors, auto &allocator,
            const auto &rho, int max_vertices, int overflow_warps ) {
    using PD = std::decay_t<decltype( pd )>;
    using TF = TFOf<PD>;
    using CardT = Card<V,PD::has_weights,OUT | MEASURES,TF,TIOf<PD>>;
    using TR = typename CardT::TR;
    CardT card;
    if ( ! card.prepare( queue, pd, allocator, max_vertices, overflow_warps ) )
        return;
    gpu2d::set_density( card.pb, rho );
    card.pb.res = strided_out<TF,1>( res );
    if constexpr ( bool( OUT & FACETS ) ) {
        const SI cap = std::max<SI>( SI( lap.nb_nnz.max ) / 2, 1 );
        card.pb.fcap = cap;
        card.pb.fi = static_cast<TR *>( take( allocator, SI( sizeof( TR ) ) * cap ) );
        card.pb.fj = static_cast<TR *>( take( allocator, SI( sizeof( TR ) ) * cap ) );
        card.pb.fc = static_cast<double *>( take( allocator, SI( sizeof( double ) ) * cap ) );
        if ( ! card.pb.fi || ! card.pb.fj || ! card.pb.fc )
            return;
    }
    if constexpr ( bool( OUT & MOMENTS ) ) {
        card.pb.bary = strided_out<TF,2>( mom.bary );
        card.pb.cost = strided_out<TF,1>( mom.cost );
    }
    card.run( queue, sdot::kernel_form( queue, MutList(), errors ) );
    if constexpr ( bool( OUT & FACETS ) ) {
        using TP = std::remove_const_t<typename std::decay_t<decltype( lap.row )>::TF>;
        using TC = std::remove_const_t<typename std::decay_t<decltype( lap.col )>::TF>;
        using TV = std::remove_const_t<typename std::decay_t<decltype( lap.val )>::TF>;
        auto *row = const_cast<TP *>( lap.row.data().raw );
        auto *col = const_cast<TC *>( lap.col.data().raw );
        auto *val = const_cast<TV *>( lap.val.data().raw );
        auto *dia = const_cast<TV *>( lap.dia.data().raw );
        gpu2d::assemble_laplacian( queue, card, allocator, row, col, val, dia );
        card.report_facets( queue, lap.nb_nnz, 2 );
    }
}

} // namespace sdot::gpu3d
