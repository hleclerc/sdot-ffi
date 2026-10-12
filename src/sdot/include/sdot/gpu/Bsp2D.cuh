#pragma once

// =====================================================================================
// THE BSP TREE BUILT ON THE CARD ( `AaBsp.py`, a CUDA device ): the same tree as the CPU's level-by-level kernel
// ( `bsp_build_level.h` ) -- the box of each node, the MEDIAN cut by rank on its longest axis, a node that cannot be
// cut ( a leaf, or all its seeds at one place ) passing its whole slice to its left child, the perfect binary tree of
// `AaBsp.max_depth_for( n, leaf )` levels written in PREORDER -- in ONE ffi call, nothing read back, so that it runs
// under `jax.jit` on traced positions like in eager. Every size is a function of `n` and the leaf size ( Python's ).
//
// = HOW ( presorted lists, the classic k-d tree construction in O( n ) per level )
//
//   1. THE SEEDS SORTED ONCE PER AXIS, by ( coordinate, index ): a stable LSD radix sort of the ordered-integer image of
//      the coordinate ( 8-bit digits; the D axes sorted together, one launch per stage for all of them ). Written here:
//      CUB's headers do not match the pinned `nvcc` ( see `Laplacian2D.cuh` ).
//   2. PER LEVEL, the D lists hold the same slices as the tree ( the slices of a level partition `[ 0, n ) )`, each list
//      sorted along its axis within each slice. Then:
//        * the NODES ( a thread each ): the box is READ off the ends of the slices ( the first and last seed of each
//          sorted list ), the longest axis, the cut `mid = b + ( e - b ) / 2` or none; the node's record is written at
//          its preorder place, and the slices of its two children for the next level;
//        * the SIDE of each seed ( a thread per position, a bit per seed ): the first `mid - b` seeds of the slice in the
//          list of the cut's axis go left;
//        * the other lists STABLY PARTITIONED by that side within their slice ( an exclusive scan of the side flags over
//          the D - 1 lists at once, then a scatter ): a list stays sorted along its axis in each child. The list of the
//          cut's axis is copied as it is ( its first `mid - b` seeds are the left ones ).
//   3. THE LAST LEVEL gives the order: a leaf's seeds in the order of its parent's cut axis ( `seed_indices`: a left leaf
//      ends next to the cut, its sibling starts there -- see `bsp_finish` ), and the inverse map ( `rank_of` ).
//
// = WHAT IT GIVES, AND WHAT DIFFERS FROM THE CPU'S TREE
//
// The same nodes ( slices, boxes, cuts ) as the CPU as soon as the coordinates along a cut are distinct; on TIES at the
// median the CPU's quickselect sends an arbitrary half of the tied seeds left, the card the lowest INDICES -- a valid
// median cut either way ( the same tree shape; the same slice sizes too, but where a tie decides which nodes end up holding
// only equal seeds, which are not cut ). Inside a leaf the order differs ( the
// CPU's is what its selection left ): a rank is a position in the tree's order, contiguous per leaf in both.
//
// DETERMINISTIC: integer scans, a stable sort, counts in shared memory and atomic ORs of the side bits ( order-independent )
// -- the same bits at every run, eager or jitted. Memory from the call's pool: `2 D n` keys ( the scan's room afterwards ),
// `2 D n + n` indices, `n / 8` bytes of sides, and the slices of the last level's nodes ( ~55 bytes per seed in 2D ).
//
// MEASURED ( RTX 2080 Ti, uniform 2D, kernel-only ): 0.92 ms at 1e5, 5.1 ms at 1e6, 43 ms at 1e7 ( the host-driven build
// took 116 ms / 840 ms / 9.1 s; the old campaign's `Bsp2D.cuh`, a radix sort of the whole cloud per level, 44 / 474 ms of
// kernels at 1e6 / 1e7 ). At 1e6 the sort is 1.6 ms, a level ~0.18 ms. See `bench/calibration_lmo_today.md`, GPU step 7.
// =====================================================================================

#include "Laplacian2D.cuh"
#include "Majorant2D.cuh"

namespace sdot::gpu2d {

// ---- the stable LSD radix sort of the D axes ------------------------------------------------------------------------

constexpr int RADIX_BITS  = 8;
constexpr int RADIX_BINS  = 1 << RADIX_BITS;
constexpr int RADIX_BLOCK = 256;                         ///< threads per tile ( = the bins: one counter each )
constexpr int RADIX_WARPS = RADIX_BLOCK / 32;
constexpr int RADIX_ITEMS = 4;                           ///< per thread
constexpr SI  RADIX_TILE  = SI( RADIX_BLOCK ) * RADIX_ITEMS;
static_assert( RADIX_BLOCK == RADIX_BINS, "one thread per bin in the per-tile scans" );

/// the coordinate as an unsigned integer of the same order ( negatives included; `-0` taken as `+0`: they are equal )
__device__ __forceinline__ unsigned long long ord_key( double x ) {
    const unsigned long long u = ( unsigned long long ) __double_as_longlong( x + 0.0 );
    return ( u >> 63 ) ? ~u : ( u | 0x8000000000000000ull );
}
__device__ __forceinline__ unsigned ord_key( float x ) {
    const unsigned u = __float_as_uint( x + 0.0f );
    return ( u >> 31 ) ? ~u : ( u | 0x80000000u );
}

/// the keys of the D axes, end to end ( `keys[ d * n + k ]` ), and the indices they carry
template<class TF,class TK,class TI>
__global__ void __launch_bounds__( BLOCK ) bsp_keys( SI n, int D, Strided<TF,2> pos, TK *keys, TI *vals ) {
    const SI i = SI( blockIdx.x ) * BLOCK + threadIdx.x;
    if ( i >= SI( D ) * n )
        return;
    const SI d = i / n, k = i - d * n;
    keys[ i ] = ord_key( pos( k, d ) );
    vals[ i ] = TI( k );
}

/// a tile's digit counts, in `hist[ ( d * BINS + bin ) * nb_tiles + tile ]`: the exclusive scan of that array is where each
/// ( axis, digit, tile ) starts in the D lists end to end -- the axes in order, the digits in order, the tiles in order
template<class TK,class TI>
__global__ void __launch_bounds__( RADIX_BLOCK ) radix_count( const TK *keys, SI n, int shift, SI nb_tiles, TI *hist ) {
    __shared__ unsigned cnt[ RADIX_BINS ];
    const SI d = SI( blockIdx.x ) / nb_tiles, t = SI( blockIdx.x ) - d * nb_tiles;
    cnt[ threadIdx.x ] = 0;
    __syncthreads();
    const TK *kd = keys + d * n;
#pragma unroll
    for ( int q = 0; q < RADIX_ITEMS; ++q ) {
        const SI i = t * RADIX_TILE + SI( q ) * RADIX_BLOCK + threadIdx.x;
        if ( i < n )
            atomicAdd( &cnt[ unsigned( kd[ i ] >> shift ) & ( RADIX_BINS - 1 ) ], 1u );
    }
    __syncthreads();
    hist[ ( d * RADIX_BINS + threadIdx.x ) * nb_tiles + t ] = TI( cnt[ threadIdx.x ] );
}

/// the STABLE scatter of a tile: warp `w` takes a contiguous chunk of the tile in rounds of 32 seeds, ranks each among
/// the equal digits of its round ( `__match_any_sync` ) after those of the rounds before; the warps' counts scanned per
/// digit and the digits' counts scanned over the tile give each seed its place in the tile SORTED BY DIGIT, where it is
/// staged in shared memory -- then written out in that order, so that consecutive threads write consecutive places of
/// the same digit's run ( a write per seed scattered over 256 runs would not coalesce )
template<class TK,class TI>
__global__ void __launch_bounds__( RADIX_BLOCK ) radix_scatter( const TK *kin, const TI *vin, TK *kout, TI *vout, SI n, int shift,
                                                               SI nb_tiles, const TI *offs, bool write_keys ) {
    __shared__ int wcnt[ RADIX_WARPS ][ RADIX_BINS ];
    __shared__ int start[ RADIX_BINS ];                  // where each digit starts in the staged tile
    __shared__ TI  base[ RADIX_BINS ];                   // ... and in the output
    __shared__ TK  skey[ RADIX_TILE ];
    __shared__ TI  sval[ RADIX_TILE ];
    const SI d = SI( blockIdx.x ) / nb_tiles, t = SI( blockIdx.x ) - d * nb_tiles;
    const int tid = threadIdx.x, w = tid >> 5, lane = tid & 31;
    for ( int i = tid; i < RADIX_WARPS * RADIX_BINS; i += RADIX_BLOCK )
        ( &wcnt[ 0 ][ 0 ] )[ i ] = 0;
    base[ tid ] = offs[ ( d * RADIX_BINS + tid ) * nb_tiles + t ];
    __syncthreads();

    const unsigned lt = ( 1u << lane ) - 1u;
    TK  key[ RADIX_ITEMS ];
    TI  val[ RADIX_ITEMS ];
    int dig[ RADIX_ITEMS ], rnk[ RADIX_ITEMS ];
#pragma unroll
    for ( int q = 0; q < RADIX_ITEMS; ++q ) {
        const SI i = t * RADIX_TILE + SI( w ) * 32 * RADIX_ITEMS + SI( q ) * 32 + lane;
        const bool ok = i < n;
        key[ q ] = ok ? kin[ d * n + i ] : TK( 0 );
        val[ q ] = ok ? vin[ d * n + i ] : TI( 0 );
        dig[ q ] = ok ? int( unsigned( key[ q ] >> shift ) & ( RADIX_BINS - 1 ) ) : RADIX_BINS;
        const unsigned peers = __match_any_sync( 0xffffffffu, dig[ q ] );
        rnk[ q ] = ok ? wcnt[ w ][ dig[ q ] ] + __popc( peers & lt ) : 0;
        __syncwarp();
        if ( ok && lane == __ffs( peers ) - 1 )
            wcnt[ w ][ dig[ q ] ] += __popc( peers );
        __syncwarp();
    }
    __syncthreads();
    {
        int run = 0;
        for ( int v = 0; v < RADIX_WARPS; ++v ) {
            const int c = wcnt[ v ][ tid ];
            wcnt[ v ][ tid ] = run;
            run += c;
        }
        start[ tid ] = run;
    }
    __syncthreads();
    for ( int o = 1; o < RADIX_BINS; o *= 2 ) {          // the digits' counts, scanned ( inclusive, then shifted )
        const int a = tid >= o ? start[ tid - o ] : 0;
        __syncthreads();
        start[ tid ] += a;
        __syncthreads();
    }
    {
        const int excl = tid ? start[ tid - 1 ] : 0;
        __syncthreads();
        start[ tid ] = excl;
    }
    __syncthreads();
#pragma unroll
    for ( int q = 0; q < RADIX_ITEMS; ++q ) {
        if ( dig[ q ] == RADIX_BINS )
            continue;
        const int l = start[ dig[ q ] ] + wcnt[ w ][ dig[ q ] ] + rnk[ q ];
        skey[ l ] = key[ q ];
        sval[ l ] = val[ q ];
    }
    __syncthreads();
    const SI m = n - t * RADIX_TILE < RADIX_TILE ? n - t * RADIX_TILE : RADIX_TILE;
#pragma unroll
    for ( int q = 0; q < RADIX_ITEMS; ++q ) {
        const int l = q * RADIX_BLOCK + tid;
        if ( l >= m )
            break;
        const TK k = skey[ l ];
        const int g = int( unsigned( k >> shift ) & ( RADIX_BINS - 1 ) );
        const TI dst = base[ g ] + TI( l - start[ g ] );
        if ( write_keys )
            kout[ dst ] = k;
        vout[ dst ] = sval[ l ];
    }
}

// ---- the levels -----------------------------------------------------------------------------------------------------

/// what the build writes: the tree's tensors ( `AaBsp`, preorder )
template<class TF,class TO>
struct TreeOut {
    StridedOut<TO,1> beg, end, left, right;
    StridedOut<TF,3> box;
};

/// the nodes of level `level` ( `2^level` of them, a thread each ): box, axis, cut, the record at the preorder place, the
/// slices of the children
template<int D,class TF,class TI,class TO>
__global__ void __launch_bounds__( BLOCK ) bsp_nodes( int depth, int level, SI leaf, SI n, Strided<TF,2> pos, const TI *lists,
                                                     const TI *beg, const TI *end, TI *mid, unsigned char *axs, TI *nbeg, TI *nend,
                                                     TreeOut<TF,TO> out ) {
    const SI j = SI( blockIdx.x ) * BLOCK + threadIdx.x;
    if ( j >= ( SI( 1 ) << level ) )
        return;
    const int h = depth - level;                         // the height of the node ( 1: the last level, leaves only )
    const SI p = node_at<SI>( depth, h, j );
    const TI b = beg[ j ], e = end[ j ];
    TF lo[ D ], hi[ D ];
    int ax = 0;
    TI m = e;
    if ( e > b ) {
#pragma unroll
        for ( int d = 0; d < D; ++d ) {
            lo[ d ] = pos( SI( lists[ SI( d ) * n + SI( b ) ] ), d );
            hi[ d ] = pos( SI( lists[ SI( d ) * n + SI( e ) - 1 ] ), d );
        }
        // the same comparisons as `bsp_build_level`: the first longest axis
#pragma unroll
        for ( int d = 1; d < D; ++d )
            if ( hi[ d ] - lo[ d ] > hi[ ax ] - lo[ ax ] )
                ax = d;
        if ( h > 1 && SI( e - b ) > leaf && hi[ ax ] > lo[ ax ] )
            m = b + ( e - b ) / 2;
    } else {
#pragma unroll
        for ( int d = 0; d < D; ++d )
            lo[ d ] = hi[ d ] = TF( 0 );
    }
#pragma unroll
    for ( int d = 0; d < D; ++d ) {
        out.box( p, 0, d ) = lo[ d ];
        out.box( p, 1, d ) = hi[ d ];
    }
    out.beg( p ) = TO( b );
    out.end( p ) = TO( e );
    out.left( p ) = h > 1 ? TO( p + 1 ) : TO( -1 );
    out.right( p ) = h > 1 ? TO( p + ( SI( 1 ) << ( h - 1 ) ) ) : TO( -1 );
    mid[ j ] = m;
    axs[ j ] = ( unsigned char ) ax;
    if ( h > 1 ) {
        nbeg[ 2 * j ] = b;     nend[ 2 * j ] = m;
        nbeg[ 2 * j + 1 ] = m; nend[ 2 * j + 1 ] = e;
    }
}

/// the SIDES of the seeds, one BIT per seed ( `1`: left ), by seed index: read and written at random, and a bit per seed is
/// what keeps them in L2 ( a byte per seed, 10 MB at 1e7 seeds, went to DRAM: three times slower per level )
__device__ __forceinline__ bool left_of( const unsigned *side, SI id ) { return ( side[ id >> 5 ] >> ( id & 31 ) ) & 1u; }

/// the side of each seed, read in the list of its node's axis ( `side` zeroed before: only the left ones are set, by an
/// atomic OR -- whatever the order, the same bits ); `seg` becomes the node of the next level ( `2 j + right` )
template<class TI>
__global__ void __launch_bounds__( BLOCK ) bsp_side( SI n, TI *seg, const TI *mid, const unsigned char *axs, const TI *lists,
                                                    unsigned *side ) {
    const SI k = SI( blockIdx.x ) * BLOCK + threadIdx.x;
    if ( k >= n )
        return;
    const TI j = seg[ k ], m = mid[ j ];
    const bool left = TI( k ) < m;
    if ( left ) {
        const SI id = SI( lists[ SI( axs[ j ] ) * n + k ] );
        atomicOr( side + ( id >> 5 ), 1u << ( id & 31 ) );
    }
    seg[ k ] = 2 * j + TI( ! left );
}

/// the list `d` that the `r`-th of the `D - 1` lists to partition is, at a position whose node cuts along `ax`: every list but
/// the cut's own ( which the partition leaves as it is: its first `mid - b` seeds are the left ones )
__device__ __forceinline__ int other_list( int r, int ax ) { return r < ax ? r : r + 1; }

/// the scan's tiles on the side flags of the `D - 1` lists to partition, end to end ( the tiles of
/// `Laplacian2D.cuh::scan_tiles`, `SCAN_TILE` entries, the input read through the lists ). The flags are 0 / 1: a warp's
/// ballot gives each lane the count before it, the rounds and the warps are summed by one warp. Entry `i` of a tile is
/// read by round `q = i / BLOCK`, warp `( i % BLOCK ) / 32`: consecutive threads, consecutive entries.
template<int D,class TI>
__global__ void __launch_bounds__( BLOCK ) bsp_scan_sides( SI n, const TI *seg, const unsigned char *axs, const TI *lists,
                                                          const unsigned *side, TI *out, TI *sums, SI tot ) {
    constexpr int WARPS = BLOCK / 32;
    static_assert( SCAN_ITEMS * WARPS <= 32, "the ( round, warp ) counts are scanned by one warp" );
    __shared__ int cnt[ SCAN_ITEMS * WARPS ];
    const int tid = threadIdx.x, w = tid >> 5, lane = tid & 31;
    const unsigned lt = ( 1u << lane ) - 1u;
    const SI b = SI( blockIdx.x ) * SCAN_TILE;
    int pre[ SCAN_ITEMS ];
#pragma unroll
    for ( int q = 0; q < SCAN_ITEMS; ++q ) {
        const SI i = b + SI( q ) * BLOCK + tid;
        bool f = false;
        if ( i < tot ) {
            const SI r = D == 2 ? 0 : i / n, k = i - r * n;
            const int dl = other_list( int( r ), axs[ seg[ k ] >> 1 ] );
            f = left_of( side, SI( lists[ SI( dl ) * n + k ] ) );
        }
        const unsigned bal = __ballot_sync( 0xffffffffu, f );
        pre[ q ] = __popc( bal & lt );
        if ( lane == 0 )
            cnt[ q * WARPS + w ] = __popc( bal );
    }
    __syncthreads();
    if ( w == 0 ) {                                      // the ( round, warp ) counts in the order of the entries, scanned
        const int c = lane < SCAN_ITEMS * WARPS ? cnt[ lane ] : 0;
        int incl = c;
        for ( int o = 1; o < 32; o *= 2 ) {
            const int a = __shfl_up_sync( 0xffffffffu, incl, o );
            if ( lane >= o )
                incl += a;
        }
        if ( lane < SCAN_ITEMS * WARPS )
            cnt[ lane ] = incl - c;
        if ( lane == 31 )
            sums[ blockIdx.x ] = TI( incl );
    }
    __syncthreads();
#pragma unroll
    for ( int q = 0; q < SCAN_ITEMS; ++q ) {
        const SI i = b + SI( q ) * BLOCK + tid;
        if ( i < tot )
            out[ i ] = TI( cnt[ q * WARPS + w ] + pre[ q ] );
    }
}

/// the stable partition of the lists within their slice: the left seeds first, in their order, then the right ones -- the
/// `D - 1` lists other than the cut's by the scan ( `s`: the scan of the tiles, `sums` the tiles' offsets: the scan's last
/// pass, `scan_add`, done here on the two reads ), the cut's own copied as it is
template<int D,class TI>
__global__ void __launch_bounds__( BLOCK ) bsp_partition( SI n, SI tot, const TI *seg, const TI *beg, const TI *mid, const unsigned char *axs,
                                                         const TI *lin, TI *lout, const unsigned *side, const TI *s, const TI *sums ) {
    const SI i = SI( blockIdx.x ) * BLOCK + threadIdx.x;
    if ( i >= tot )
        return;
    const SI r = D == 2 ? 0 : i / n, k = i - r * n;
    const TI j = seg[ k ] >> 1, b = beg[ j ], m = mid[ j ];
    const int ax = axs[ j ];
    if ( r == 0 )
        lout[ SI( ax ) * n + k ] = lin[ SI( ax ) * n + k ];
    const int d = other_list( int( r ), ax );
    const TI id = lin[ SI( d ) * n + k ];
    const SI ib = r * n + SI( b );
    const TI c = ( s[ i ] + sums[ i / SCAN_TILE ] ) - ( s[ ib ] + sums[ ib / SCAN_TILE ] );   // the left seeds before this one in its slice
    const TI dst = left_of( side, SI( id ) ) ? b + c : m + ( TI( k ) - b ) - c;
    lout[ SI( d ) * n + SI( dst ) ] = id;
}

/// the order of the seeds and its inverse: a leaf in the order of its PARENT's cut axis ( `axs`: the axes of the level above
/// the leaves, `seg`: `2 parent + right` ), so that a left leaf ENDS next to the cut and its sibling STARTS there -- the
/// ranks on either side of a leaf boundary are neighbours in space, which the multigrid's packets of consecutive ranks
/// aggregate ( the host's quickselect does the same by accident: it leaves the seeds near the median in the middle )
template<class TI,class TO>
__global__ void __launch_bounds__( BLOCK ) bsp_finish( SI n, const TI *seg, const unsigned char *axs, const TI *lists,
                                                      StridedOut<TO,1> seed, StridedOut<TO,1> rank ) {
    const SI k = SI( blockIdx.x ) * BLOCK + threadIdx.x;
    if ( k >= n )
        return;
    const TI id = lists[ SI( axs[ seg[ k ] >> 1 ] ) * n + k ];
    seed( k ) = TO( id );
    rank( SI( id ) ) = TO( k );
}

template<class TI>
__global__ void bsp_root( TI *beg, TI *end, SI n ) {
    beg[ 0 ] = 0;
    end[ 0 ] = TI( n );
}

/// THE BUILD ( see the header ). `positions`: `[ n, D ]` in the user's order; `leaf`: the leaf size; the outputs are the
/// tree's tensors ( their node count, `2^depth - 1`, gives the depth ). `TI`: the index type of the work ( `int` while
/// `D n` fits in it, Python's choice ). Nothing is read back; a refusal of the pool is reported by `allocator`.
template<int D,class TI>
void build_tree( const CudaQueue &queue, const auto &positions, SI leaf, auto &&seed_indices, auto &&rank_of,
                 auto &&node_begin, auto &&node_end, auto &&node_left, auto &&node_right, auto &&node_box, auto &allocator ) {
    using TF = std::remove_const_t<typename std::decay_t<decltype( positions )>::TF>;
    using TO = std::remove_const_t<typename std::decay_t<decltype( seed_indices )>::TF>;
    using TK = std::conditional_t<sizeof( TF ) == 8, unsigned long long, unsigned>;
    const SI n = SI( positions.shape( 0 ) );
    const SI nb_nodes = SI( node_begin.shape( 0 ) );
    int depth = 0;
    for ( SI m = nb_nodes; m; m >>= 1 )
        ++depth;
    if ( n == 0 || depth == 0 )
        return;
    static_assert( D >= 2, "the build partitions the lists of the other axes" );
    const SI tot = SI( D ) * n, part = SI( D - 1 ) * n, half = SI( 1 ) << ( depth - 1 );
    const SI nb_tiles = ( n + RADIX_TILE - 1 ) / RADIX_TILE, nb_hist = SI( D ) * RADIX_BINS * nb_tiles;

    // ---- the work, from the call's pool
    TK *keys[ 2 ];
    TI *lists[ 2 ], *lb[ 2 ], *le[ 2 ];
    for ( int q = 0; q < 2; ++q ) {
        keys[ q ]  = static_cast<TK *>( take( allocator, SI( sizeof( TK ) ) * tot ) );
        lists[ q ] = static_cast<TI *>( take( allocator, SI( sizeof( TI ) ) * tot ) );
        lb[ q ]    = static_cast<TI *>( take( allocator, SI( sizeof( TI ) ) * half ) );
        le[ q ]    = static_cast<TI *>( take( allocator, SI( sizeof( TI ) ) * half ) );
    }
    TI *seg  = static_cast<TI *>( take( allocator, SI( sizeof( TI ) ) * n ) );
    TI *mid  = static_cast<TI *>( take( allocator, SI( sizeof( TI ) ) * half ) );
    TI *hist = static_cast<TI *>( take( allocator, SI( sizeof( TI ) ) * nb_hist ) );
    TI *sums = static_cast<TI *>( take( allocator, SI( sizeof( TI ) ) * scan_sums_size( std::max( nb_hist, tot ) ) ) );
    unsigned char *axs_of[ 2 ] = { static_cast<unsigned char *>( take( allocator, half ) ), static_cast<unsigned char *>( take( allocator, half ) ) };
    const SI nb_side = ( n + 31 ) / 32;
    auto *side = static_cast<unsigned *>( take( allocator, SI( sizeof( unsigned ) ) * nb_side ) );
    if ( ! keys[ 0 ] || ! keys[ 1 ] || ! lists[ 0 ] || ! lists[ 1 ] || ! lb[ 0 ] || ! lb[ 1 ] || ! le[ 0 ] || ! le[ 1 ] || ! seg || ! mid
         || ! hist || ! sums || ! axs_of[ 0 ] || ! axs_of[ 1 ] || ! side )
        return;
    // the scan of the side flags ( `tot` indices ) in the keys, free once the lists are sorted ( the last pass reads the
    // other buffer and writes no key )
    TI *scan = sizeof( TK ) >= sizeof( TI ) ? reinterpret_cast<TI *>( keys[ 0 ] ) : nullptr;
    if ( ! scan && ! ( scan = static_cast<TI *>( take( allocator, SI( sizeof( TI ) ) * tot ) ) ) )
        return;

    const auto pos = strided( positions );
    TreeOut<TF,TO> out{ strided_out<TO,1>( node_begin ), strided_out<TO,1>( node_end ), strided_out<TO,1>( node_left ),
                        strided_out<TO,1>( node_right ), strided_out<TF,3>( node_box ) };

    // ---- 1. the D lists sorted by ( coordinate, index )
    launch_kernel( queue, &bsp_keys<TF,TK,TI>, blocks_for( tot ), BLOCK, 0, n, D, pos, keys[ 0 ], lists[ 0 ] );
    constexpr int nb_passes = int( sizeof( TK ) ) * 8 / RADIX_BITS;
    static_assert( nb_passes % 2 == 0, "the sorted lists end where they started" );
    for ( int pass = 0; pass < nb_passes; ++pass ) {
        const int src = pass & 1, dst = src ^ 1;
        launch_kernel( queue, &radix_count<TK,TI>, int( D * nb_tiles ), RADIX_BLOCK, 0, ( const TK * ) keys[ src ], n, pass * RADIX_BITS,
                       nb_tiles, hist );
        exclusive_scan( queue, sums, ( const TI * ) hist, hist, nb_hist );
        launch_kernel( queue, &radix_scatter<TK,TI>, int( D * nb_tiles ), RADIX_BLOCK, 0, ( const TK * ) keys[ src ], ( const TI * ) lists[ src ],
                       keys[ dst ], lists[ dst ], n, pass * RADIX_BITS, nb_tiles, ( const TI * ) hist, pass + 1 < nb_passes );
    }

    // ---- 2. the levels
    zero_fill( queue, seg, SI( sizeof( TI ) ) * n );
    launch_kernel( queue, &bsp_root<TI>, 1, 1, 0, lb[ 0 ], le[ 0 ], n );
    int cur = 0, lc = 0;                                 // the lists, the level's slices
    for ( int level = 0; level < depth; ++level ) {
        const SI nb = SI( 1 ) << level;
        unsigned char *axs = axs_of[ level & 1 ];       // ( the level above's stay: the leaves' order reads them )
        launch_kernel( queue, &bsp_nodes<D,TF,TI,TO>, blocks_for( nb ), BLOCK, 0, depth, level, leaf, n, pos, ( const TI * ) lists[ cur ],
                       ( const TI * ) lb[ lc ], ( const TI * ) le[ lc ], mid, axs, lb[ lc ^ 1 ], le[ lc ^ 1 ], out );
        if ( level + 1 == depth )
            break;
        zero_fill( queue, side, SI( sizeof( unsigned ) ) * nb_side );
        launch_kernel( queue, &bsp_side<TI>, blocks_for( n ), BLOCK, 0, n, seg, ( const TI * ) mid, ( const unsigned char * ) axs,
                       ( const TI * ) lists[ cur ], side );
        const SI nb_scan_tiles = ( part + SCAN_TILE - 1 ) / SCAN_TILE;
        launch_kernel( queue, &bsp_scan_sides<D,TI>, int( nb_scan_tiles ), BLOCK, 0, n, ( const TI * ) seg, ( const unsigned char * ) axs,
                       ( const TI * ) lists[ cur ], ( const unsigned * ) side, scan, sums, part );
        launch_kernel( queue, &scan_sums<TI>, 1, BLOCK, 0, sums, nb_scan_tiles );
        launch_kernel( queue, &bsp_partition<D,TI>, blocks_for( part ), BLOCK, 0, n, part, ( const TI * ) seg, ( const TI * ) lb[ lc ],
                       ( const TI * ) mid, ( const unsigned char * ) axs, ( const TI * ) lists[ cur ], lists[ cur ^ 1 ], ( const unsigned * ) side,
                       ( const TI * ) scan, ( const TI * ) sums );
        cur ^= 1;
        lc ^= 1;
    }

    // ---- 3. the order and its inverse ( a single node: its own axis, `seg` being zero )
    launch_kernel( queue, &bsp_finish<TI,TO>, blocks_for( n ), BLOCK, 0, n, ( const TI * ) seg, ( const unsigned char * ) axs_of[ depth > 1 ? ( depth - 2 ) & 1 : 0 ],
                   ( const TI * ) lists[ cur ], strided_out<TO,1>( seed_indices ), strided_out<TO,1>( rank_of ) );
}

// ---- the weight majorants of a tree, from weights in tree order ( `AaBsp.refresh_weight_majorants` on a card ) -----

template<class TF>
__global__ void __launch_bounds__( BLOCK ) bsp_weights_to_double( SI n, Strided<TF,1> src, double *dst ) {
    const SI k = SI( blockIdx.x ) * BLOCK + threadIdx.x;
    if ( k < n )
        dst[ k ] = double( src( k ) );
}

/// `node_wa` / `node_wb` of the tree ( `node_box`, `node_begin`, `node_end` ) for the seeds `sorted_positions` /
/// `sorted_weights` ( tree order ): `Majorant2D.cuh`'s launches, without the card's node records. `TN`: the node index.
template<class TN>
void refresh_majorants( const CudaQueue &queue, const auto &node_box, const auto &node_begin, const auto &node_end,
                        const auto &sorted_positions, const auto &sorted_weights, auto &&node_wa, auto &&node_wb, auto &allocator ) {
    using TW = std::remove_const_t<typename std::decay_t<decltype( node_wa )>::TF>;
    const SI n = SI( sorted_positions.shape( 0 ) );
    Majorants<int,TN> maj;
    double *w = static_cast<double *>( take( allocator, SI( sizeof( double ) ) * std::max<SI>( n, 1 ) ) );
    if ( ! w || ! maj.prepare_for( allocator, SI( node_begin.shape( 0 ) ) ) )
        return;
    using TFW = std::remove_const_t<typename std::decay_t<decltype( sorted_weights )>::TF>;
    launch_kernel( queue, &bsp_weights_to_double<TFW>, blocks_for( n ), BLOCK, 0, n, strided( sorted_weights ), w );
    maj.refresh_from( queue, n, strided( node_box ), strided( node_begin ), strided( node_end ), strided( sorted_positions ), w,
                      ( Node<true,int> * ) nullptr, strided_out<TW,2>( node_wa ), strided_out<TW,1>( node_wb ) );
}

} // namespace sdot::gpu2d
