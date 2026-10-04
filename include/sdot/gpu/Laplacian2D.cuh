#pragma once

// =====================================================================================
// NEWTON'S HESSIAN ON THE CARD: the laplacian of the Laguerre graph, assembled from the facets that the cells of
// `Cell2D.cuh` give ( `Out::FACETS` ), in the layout of `sdotplan/Laplacian.h`:
//
//     c_ij = int_{facet ij} rho / ( 2 |p_i - p_j| ),   L_ii = sum_j c_ij,   L_ij = -c_ij
//
//     row[ n + 1 ], col[ nnz ], val[ nnz ] ( = c_ij > 0, the minus sign is in the matrix ), dia[ n ]
//
// IN RANKS ( the tree's order: neighbours in the tree are neighbours in space, which is what a multigrid wants to
// aggregate -- the old campaign's `rank >> 2` ), the full system ( the gauge is the solver's business ).
//
// = SYMMETRIC TO THE BIT ( the old campaign's `Hess2D.cuh`, and its doc/06 for why it is not a luxury )
//
// Each facet is seen twice, once per cell, and in float the two views can differ by 100 % on a nearly degenerate
// facet -- a conjugate gradient then stops converging. Only the view `i < j` is kept ( the cells only emit it: the
// COO holds each facet once ) and MIRRORED into both rows: `L = L^T` exactly. A facet seen from the higher rank only
// is dropped ( a sliver that one of the two cells does not even see ).
//
//   * COUNT: one thread per COO entry, an atomic for each of its two rows;
//   * SCAN: an exclusive prefix sum gives `row` ( written here: CUB's headers do not match the pinned `nvcc` );
//   * FILL: the same loop, each row's cursor an atomic, THE SAME double on both sides;
//   * SORT each row by column ( insertion, seven entries ): the CSR is then the same to the bit from one run to the next
//     despite the atomics, and so is every product with it;
//   * THE DIAGONAL is summed from the finished row, in its order: `L 1 = 0` to the bit. A row without a neighbour ( an
//     empty cell ) would make the system singular without saying so: its diagonal is set to one, as on the CPU.
//
// Everything is launched on the call's stream, nothing is read back: the number of entries is `2 * nb_facets`, known
// on the card, and the loops read it there.
// =====================================================================================

#include "Cell2D.cuh"

namespace sdot::gpu2d {

template<class T>
__device__ __forceinline__ T atomic_inc( T *p ) {
    if constexpr ( sizeof( T ) == 4 ) return T( atomicAdd( reinterpret_cast<unsigned *>( p ), 1u ) );
    else                              return T( atomicAdd( reinterpret_cast<unsigned long long *>( p ), 1ull ) );
}

// ---- an exclusive prefix sum, in three launches ( tiles, the tiles' sums, the offsets ) ----------------------------

constexpr int SCAN_ITEMS = 8;                            ///< per thread: a tile is `BLOCK * SCAN_ITEMS` entries
constexpr SI  SCAN_TILE  = SI( BLOCK ) * SCAN_ITEMS;

/// the exclusive scan of 128 values, one per thread of the block ( `tmp`: 128 shared slots ); returns the block's total
template<class T>
__device__ __forceinline__ T block_exclusive_scan( T &v, T *tmp ) {
    const int t = threadIdx.x;
    tmp[ t ] = v;
    __syncthreads();
    for ( int o = 1; o < BLOCK; o *= 2 ) {
        const T a = t >= o ? tmp[ t - o ] : T( 0 );
        __syncthreads();
        tmp[ t ] += a;
        __syncthreads();
    }
    const T incl = tmp[ t ], tot = tmp[ BLOCK - 1 ];
    __syncthreads();
    v = incl - v;
    return tot;
}

/// each tile scanned on its own, its total in `sums`
template<class T>
__global__ void __launch_bounds__( BLOCK ) scan_tiles( const T *in, T *out, T *sums, SI n ) {
    __shared__ T tmp[ BLOCK ];
    const SI b = SI( blockIdx.x ) * SCAN_TILE + SI( threadIdx.x ) * SCAN_ITEMS;
    T loc[ SCAN_ITEMS ], s = 0;
#pragma unroll
    for ( int q = 0; q < SCAN_ITEMS; ++q ) {
        loc[ q ] = b + q < n ? in[ b + q ] : T( 0 );
        s += loc[ q ];
    }
    const T tot = block_exclusive_scan( s, tmp );
#pragma unroll
    for ( int q = 0; q < SCAN_ITEMS; ++q ) {
        if ( b + q < n )
            out[ b + q ] = s;
        s += loc[ q ];
    }
    if ( threadIdx.x == 0 )
        sums[ blockIdx.x ] = tot;
}

/// the tiles' totals scanned in place, by one block ( a chunk per thread )
template<class T>
__global__ void __launch_bounds__( BLOCK ) scan_sums( T *sums, SI m ) {
    __shared__ T tmp[ BLOCK ];
    const SI chunk = ( m + BLOCK - 1 ) / BLOCK, b = SI( threadIdx.x ) * chunk, e = b + chunk < m ? b + chunk : m;
    T s = 0;
    for ( SI i = b; i < e; ++i )
        s += sums[ i ];
    block_exclusive_scan( s, tmp );
    for ( SI i = b; i < e; ++i ) {
        const T v = sums[ i ];
        sums[ i ] = s;
        s += v;
    }
}

template<class T>
__global__ void __launch_bounds__( BLOCK ) scan_add( T *out, const T *sums, SI n ) {
    const SI i = SI( blockIdx.x ) * blockDim.x + threadIdx.x;
    if ( i < n )
        out[ i ] += sums[ i / SCAN_TILE ];
}

/// the tiles' totals a scan of `n` values needs
inline SI scan_sums_size( SI n ) { return std::max<SI>( ( n + SCAN_TILE - 1 ) / SCAN_TILE, 1 ); }

/// `out[ i ] = in[ 0 ] + ... + in[ i - 1 ]`, on the call's stream, `sums` holding `scan_sums_size( n )` values
template<class T>
void exclusive_scan( const CudaQueue &queue, T *sums, const T *in, T *out, SI n ) {
    const SI nb_tiles = ( n + SCAN_TILE - 1 ) / SCAN_TILE;
    launch_kernel( queue, &scan_tiles<T>, int( nb_tiles ), BLOCK, 0, in, out, sums, n );
    launch_kernel( queue, &scan_sums<T>, 1, BLOCK, 0, sums, nb_tiles );
    launch_kernel( queue, &scan_add<T>, blocks_for( n ), BLOCK, 0, out, sums, n );
}

/// the same, the tiles' totals taken from the call's pool; `false` if the pool said no
template<class T>
bool exclusive_scan( const CudaQueue &queue, auto &allocator, const T *in, T *out, SI n ) {
    T *sums = static_cast<T *>( take( allocator, SI( sizeof( T ) ) * scan_sums_size( n ) ) );
    if ( ! sums )
        return false;
    exclusive_scan( queue, sums, in, out, n );
    return true;
}

/// how many COO entries there are: what the cells wanted, within what the COO holds
__device__ __forceinline__ unsigned long long coo_size( const Counters *counters, unsigned long long cap ) {
    return min( counters->nb_facets, cap );
}

template<class TR,class TP>
__global__ void __launch_bounds__( BLOCK ) lap_count( const TR *fi, const TR *fj, const Counters *counters, unsigned long long cap, TP *cnt ) {
    const unsigned long long m = coo_size( counters, cap );
    for ( unsigned long long e = SI( blockIdx.x ) * blockDim.x + threadIdx.x; e < m; e += SI( gridDim.x ) * blockDim.x ) {
        atomic_inc( cnt + fi[ e ] );
        atomic_inc( cnt + fj[ e ] );
    }
}

template<class TR,class TP,class TC,class TV>
__global__ void __launch_bounds__( BLOCK ) lap_fill( const TR *fi, const TR *fj, const double *fc, const Counters *counters, unsigned long long cap,
                                                    TP *at, TC *col, TV *val ) {
    const unsigned long long m = coo_size( counters, cap );
    for ( unsigned long long e = SI( blockIdx.x ) * blockDim.x + threadIdx.x; e < m; e += SI( gridDim.x ) * blockDim.x ) {
        const TR i = fi[ e ], j = fj[ e ];
        const TV c = TV( fc[ e ] );
        const TP p = atomic_inc( at + i ), q = atomic_inc( at + j );
        col[ p ] = TC( j ); val[ p ] = c;                // THE SAME double on both sides
        col[ q ] = TC( i ); val[ q ] = c;
    }
}

/// the rows of up to `SORT_R` entries are sorted in REGISTERS ( an odd-even transposition network, every index a
/// compile-time constant ), the others in place ( insertion ): sorting in place in global memory, one read-modify-write
/// per move, cost 2.8 ms at n = 1e6 -- more than the whole rest of the assembly
constexpr int SORT_R = 16;

template<class TP,class TC,class TV>
__global__ void __launch_bounds__( BLOCK ) lap_sort( const TP *row, TC *col, TV *val, TV *dia, SI n ) {
    const SI i = SI( blockIdx.x ) * blockDim.x + threadIdx.x;
    if ( i >= n )
        return;
    const TP b = row[ i ], e = row[ i + 1 ];
    const int len = int( e - b );
    TV d = 0;
    if ( len <= SORT_R ) {
        TC c[ SORT_R ];
        TV v[ SORT_R ];
#pragma unroll
        for ( int q = 0; q < SORT_R; ++q ) {
            c[ q ] = q < len ? col[ b + q ] : TC( -1 ) & ~( TC( 1 ) << ( 8 * sizeof( TC ) - 1 ) );   // the largest TC past the row
            v[ q ] = q < len ? val[ b + q ] : TV( 0 );
        }
#pragma unroll
        for ( int r = 0; r < SORT_R; ++r ) {
#pragma unroll
            for ( int q = r % 2; q + 1 < SORT_R; q += 2 ) {
                const bool sw = c[ q ] > c[ q + 1 ];
                const TC c0 = sw ? c[ q + 1 ] : c[ q ], c1 = sw ? c[ q ] : c[ q + 1 ];
                const TV v0 = sw ? v[ q + 1 ] : v[ q ], v1 = sw ? v[ q ] : v[ q + 1 ];
                c[ q ] = c0; c[ q + 1 ] = c1; v[ q ] = v0; v[ q + 1 ] = v1;
            }
        }
#pragma unroll
        for ( int q = 0; q < SORT_R; ++q ) {
            if ( q < len ) {
                col[ b + q ] = c[ q ];
                val[ b + q ] = v[ q ];
                d += v[ q ];
            }
        }
    } else {
        for ( TP p = b + 1; p < e; ++p ) {
            const TC c = col[ p ];
            const TV v = val[ p ];
            TP q = p;
            for ( ; q > b && col[ q - 1 ] > c; --q ) { col[ q ] = col[ q - 1 ]; val[ q ] = val[ q - 1 ]; }
            col[ q ] = c; val[ q ] = v;
        }
        for ( TP p = b; p < e; ++p )
            d += val[ p ];
    }
    dia[ i ] = d > 0 ? d : TV( 1 );
}

/// what the assembly works in: two arrays of `n + 1` counters and the scan's tiles ( taken once by a solver that
/// assembles at every iteration: the call's pool frees nothing before the call returns )
template<class TP>
struct LapWork {
    TP *cnt = nullptr, *at = nullptr, *sums = nullptr;

    bool take_from( auto &allocator, SI n ) {
        cnt  = static_cast<TP *>( take( allocator, SI( sizeof( TP ) ) * ( n + 1 ) ) );
        at   = static_cast<TP *>( take( allocator, SI( sizeof( TP ) ) * ( n + 1 ) ) );
        sums = static_cast<TP *>( take( allocator, SI( sizeof( TP ) ) * scan_sums_size( n + 1 ) ) );
        return cnt && at && sums;
    }
};

/// THE CSR of the laplacian from the COO of the upper facets that a `Card` with `FACETS` filled ( `card.pb.fi / fj /
/// fc`, `card.counters` ). `row`: `n + 1` entries of `TP`; `col` / `val`: `2 * card.pb.fcap` entries at least.
template<class CardT,class TP,class TC,class TV>
void assemble_laplacian_in( const CudaQueue &queue, const CardT &card, const LapWork<TP> &ws, TP *row, TC *col, TV *val, TV *dia ) {
    const SI n = SI( card.pb.n );
    zero_fill( queue, ws.cnt, SI( sizeof( TP ) ) * ( n + 1 ) );
    static const int grid = resident_grid( &lap_count<typename CardT::TR,TP>, BLOCK );
    launch_kernel( queue, &lap_count<typename CardT::TR,TP>, grid, BLOCK, 0, card.pb.fi, card.pb.fj, card.counters, card.pb.fcap, ws.cnt );
    exclusive_scan( queue, ws.sums, static_cast<const TP *>( ws.cnt ), row, n + 1 );
    cuda_check( cudaMemcpyAsync( ws.at, row, sizeof( TP ) * n, cudaMemcpyDeviceToDevice, queue.stream ), "copy of the row starts" );
    launch_kernel( queue, &lap_fill<typename CardT::TR,TP,TC,TV>, grid, BLOCK, 0, card.pb.fi, card.pb.fj, card.pb.fc, card.counters, card.pb.fcap, ws.at, col, val );
    launch_kernel( queue, &lap_sort<TP,TC,TV>, blocks_for( n ), BLOCK, 0, row, col, val, dia, n );
}

/// the same, its workspace taken from the call's pool; `false` if the pool said no
template<class CardT,class TP,class TC,class TV>
bool assemble_laplacian( const CudaQueue &queue, const CardT &card, auto &allocator, TP *row, TC *col, TV *val, TV *dia ) {
    LapWork<TP> ws;
    if ( ! ws.take_from( allocator, SI( card.pb.n ) ) )
        return false;
    assemble_laplacian_in( queue, card, ws, row, col, val, dia );
    return true;
}

/// THE CELLS AND WHAT IS MADE OF THEM, in one call ( `PowerDiagram_Bsp._card_cells`; what Newton's iteration asks for
/// is `MEASURES | FACETS` ): the measures ( and the status ), and per `OUT` the laplacian's CSR ( `lap.row / col / val /
/// dia`, `lap.nb_nnz` the capacity of `col` and `val` ), the barycentres and the costs of the cells ( `mom.bary`,
/// `mom.cost` ). What is not asked for can be anything ( `0` ).
template<class V,unsigned OUT>
void cells( const CudaQueue &queue, const auto &pd, auto &&res, auto &&work, auto &&lap, auto &&mom, const auto &errors, auto &allocator,
            const auto &rho, int max_vertices ) {
    using PD = std::decay_t<decltype( pd )>;
    using TF = TFOf<PD>;
    using CardT = Card<V,PD::has_weights,OUT | MEASURES,TF,TIOf<PD>>;
    using TR = typename CardT::TR;
    CardT card;
    if ( ! card.prepare( queue, pd, allocator, int( std::min<SI>( work.nb_spill.max, max_vertices ) ), SPILL_WARPS, max_vertices ) )
        return;
    set_density( card.pb, rho );
    card.pb.res    = strided_out<TF,1>( res );
    card.pb.status = reinterpret_cast<int *>( work.status.data().raw );
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
    card.report_spill( queue, work.nb_spill );
    if constexpr ( bool( OUT & FACETS ) ) {
        using TP = std::remove_const_t<typename std::decay_t<decltype( lap.row )>::TF>;
        using TC = std::remove_const_t<typename std::decay_t<decltype( lap.col )>::TF>;
        auto *row = const_cast<TP *>( lap.row.data().raw );
        auto *col = const_cast<TC *>( lap.col.data().raw );
        using TV = std::remove_const_t<typename std::decay_t<decltype( lap.val )>::TF>;
        auto *val = const_cast<TV *>( lap.val.data().raw );
        auto *dia = const_cast<TV *>( lap.dia.data().raw );
        assemble_laplacian( queue, card, allocator, row, col, val, dia );
        card.report_facets( queue, lap.nb_nnz, 2 );      // `nnz = 2 nb_facets`: past the capacity, loom runs again
    }
}

} // namespace sdot::gpu2d
