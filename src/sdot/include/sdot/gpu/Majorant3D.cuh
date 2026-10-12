#pragma once

// =====================================================================================
// THE TREE'S WEIGHT MAJORANTS ON THE CARD, ANY DIMENSION ( used in 3D: `AaBsp.refresh_weight_majorants` on a card ).
// `Majorant2D.cuh`'s three steps with `D` a template parameter -- the rule is `bsp_weight_majorant`'s
// ( `bsp_build_level.h` ): the affine least-squares fit of the node's weights, kept if it clearly tightens the spread of
// the residuals ( beyond what `D + 1` parameters give by chance ) and its slope is well behaved, then
// `b = max( w - a . y )` plus a margin.
//
//   1. UP THE TREE, one launch per height: a leaf sums its seeds, an inner node combines its two children exactly
//      ( the parallel-axis rule ), and solves its candidate slope ( Gauss with partial pivoting );
//   2. THE SPREAD of the residuals under each node's candidate: one thread per seed walks from the root to its leaf, the
//      lanes that share a node reduce together, one atomic min / max per node and warp on ordered integers ( exact );
//   3. THE TENSORS `node_wa`, `node_wb` ( the card's walk builds its float records from them, `Cell3D.cuh::make_nodes` ).
//
// Deterministic: the sums are those of a fixed tree, the extrema exact.
// =====================================================================================

#include "Bsp2D.cuh"

namespace sdot::gpu3d {

using gpu2d::BLOCK;
using gpu2d::MajRed;
using gpu2d::Strided;
using gpu2d::StridedOut;
using gpu2d::strided;
using gpu2d::strided_out;
using gpu2d::take;
using gpu2d::blocks_for;

/// what step 1 leaves per node: count, means, the centred `sum q q^T` ( upper triangle, row by row ) and `sum q dw`,
/// the weight extrema, the candidate slope ( zero: none )
template<int D>
struct MajStatsN {
    static constexpr int NM = D * ( D + 1 ) / 2;
    double m, my[ D ], mw, M[ NM ], r[ D ], wmin, wmax, a[ D ];
};

template<int D>
__host__ __device__ constexpr int sym( int i, int j ) { return i <= j ? i * D - i * ( i - 1 ) / 2 + ( j - i ) : sym<D>( j, i ); }

/// STEP 1, the nodes of height `h` ( `nb` of them )
template<int D,class TB,class TJ,class TF,class TN>
__global__ void __launch_bounds__( BLOCK ) majorant_up( int depth, int h, TN nb, Strided<TB,3> box, Strided<TJ,1> beg, Strided<TJ,1> end,
                                                       Strided<TF,2> pos, const double *w, MajStatsN<D> *st, MajRed *red ) {
    const TN j = TN( blockIdx.x ) * BLOCK + threadIdx.x;
    if ( j >= nb )
        return;
    const TN n = gpu2d::node_at( depth, h, j );
    MajStatsN<D> s;
    s.m = 0; s.mw = 0; s.wmin = s.wmax = 0;
    for ( int d = 0; d < D; ++d ) { s.my[ d ] = 0; s.r[ d ] = 0; s.a[ d ] = 0; }
    for ( int q = 0; q < MajStatsN<D>::NM; ++q ) s.M[ q ] = 0;
    const SI b = SI( beg( n ) ), e = SI( end( n ) );
    if ( e > b ) {
        if ( h == 1 ) {
            double sp[ D ] = {}, sw = 0;
            s.wmin = s.wmax = w[ b ];
            for ( SI k = b; k < e; ++k ) {
                const double v = w[ k ];
                for ( int d = 0; d < D; ++d ) sp[ d ] += double( pos( k, d ) );
                sw += v;
                s.wmin = fmin( s.wmin, v ); s.wmax = fmax( s.wmax, v );
            }
            s.m = double( e - b );
            for ( int d = 0; d < D; ++d ) s.my[ d ] = sp[ d ] / s.m;
            s.mw = sw / s.m;
            for ( SI k = b; k < e; ++k ) {
                double q[ D ];
                for ( int d = 0; d < D; ++d ) q[ d ] = double( pos( k, d ) ) - s.my[ d ];
                const double dw = w[ k ] - s.mw;
                for ( int a = 0; a < D; ++a ) {
                    for ( int c = a; c < D; ++c )
                        s.M[ sym<D>( a, c ) ] += q[ a ] * q[ c ];
                    s.r[ a ] += q[ a ] * dw;
                }
            }
        } else {
            const MajStatsN<D> L = st[ n + 1 ], R = st[ n + ( TN( 1 ) << ( h - 1 ) ) ];
            if ( L.m == 0 || R.m == 0 ) {
                s = L.m == 0 ? R : L;
            } else {
                s.m = L.m + R.m;
                const double f = L.m * R.m / s.m, g = R.m / s.m;
                double dl[ D ];
                for ( int d = 0; d < D; ++d ) dl[ d ] = R.my[ d ] - L.my[ d ];
                const double dw = R.mw - L.mw;
                for ( int d = 0; d < D; ++d ) s.my[ d ] = L.my[ d ] + dl[ d ] * g;
                s.mw = L.mw + dw * g;
                for ( int a = 0; a < D; ++a ) {
                    for ( int c = a; c < D; ++c )
                        s.M[ sym<D>( a, c ) ] = L.M[ sym<D>( a, c ) ] + R.M[ sym<D>( a, c ) ] + dl[ a ] * dl[ c ] * f;
                    s.r[ a ] = L.r[ a ] + R.r[ a ] + dl[ a ] * dw * f;
                }
                s.wmin = fmin( L.wmin, R.wmin ); s.wmax = fmax( L.wmax, R.wmax );
            }
            for ( int d = 0; d < D; ++d ) s.a[ d ] = 0;
        }

        // the candidate slope: the normal equations by Gauss with partial pivoting, then the guards on the slope
        const double spread = s.wmax - s.wmin;
        if ( s.m >= 2 * ( D + 1 ) && spread > 0 ) {
            double A[ D ][ D + 1 ];
            for ( int a = 0; a < D; ++a ) {
                for ( int c = 0; c < D; ++c ) A[ a ][ c ] = s.M[ sym<D>( a, c ) ];
                A[ a ][ D ] = s.r[ a ];
            }
            bool ok = true;
            for ( int c = 0; c < D && ok; ++c ) {
                int p = c;
                for ( int i = c + 1; i < D; ++i )
                    if ( fabs( A[ i ][ c ] ) > fabs( A[ p ][ c ] ) )
                        p = i;
                if ( ! ( fabs( A[ p ][ c ] ) > 0 ) ) {
                    ok = false;
                    break;
                }
                if ( p != c )
                    for ( int q = c; q <= D; ++q ) { const double t = A[ c ][ q ]; A[ c ][ q ] = A[ p ][ q ]; A[ p ][ q ] = t; }
                for ( int i = c + 1; i < D; ++i ) {
                    const double f = A[ i ][ c ] / A[ c ][ c ];
                    for ( int q = c; q <= D; ++q )
                        A[ i ][ q ] -= f * A[ c ][ q ];
                }
            }
            if ( ok ) {
                double fit[ D ];
                for ( int i = D - 1; i >= 0; --i ) {
                    double t = A[ i ][ D ];
                    for ( int q = i + 1; q < D; ++q )
                        t -= A[ i ][ q ] * fit[ q ];
                    fit[ i ] = t / A[ i ][ i ];
                }
                bool well_behaved = true;
                for ( int d = 0; d < D; ++d ) {
                    const double lo = double( box( n, 0, d ) ), hi = double( box( n, 1, d ) );
                    const double reach = fmax( fabs( lo ), fabs( hi ) );
                    if ( ! isfinite( fit[ d ] ) || fabs( fit[ d ] ) * ( hi - lo ) > 8 * spread || fabs( fit[ d ] ) * reach > 100 * spread )
                        well_behaved = false;
                }
                if ( well_behaved )
                    for ( int d = 0; d < D; ++d )
                        s.a[ d ] = fit[ d ];
            }
        }
    }
    st[ n ] = s;
    red[ n ] = MajRed{ gpu2d::ord_of( 1e300 ), gpu2d::ord_of( -1e300 ), gpu2d::ord_of( 0.0 ) };
}

/// STEP 2, one thread per seed ( rank `k` ), walking from the root to its leaf
template<int D,class TJ,class TF,class TN>
__global__ void __launch_bounds__( BLOCK ) majorant_spread( int depth, SI n, Strided<TJ,1> end, Strided<TF,2> pos, const double *w,
                                                           const MajStatsN<D> *st, MajRed *red ) {
    const SI k = SI( blockIdx.x ) * BLOCK + threadIdx.x;
    const bool in = k < n;
    const int lane = threadIdx.x & 31;
    double p[ D ];
    for ( int d = 0; d < D; ++d ) p[ d ] = in ? double( pos( k, d ) ) : 0.0;
    const double wk = in ? w[ k ] : 0.0;
    TN nd = 0;
    for ( int h = depth; h >= 1; --h ) {
        double ay = 0;
        bool act = false;
        if ( in ) {
            for ( int d = 0; d < D; ++d ) {
                const double a = st[ nd ].a[ d ];
                act |= a != 0;
                ay += a * p[ d ];
            }
        }
        double rmin = act ? wk - ay : 1e300, rmax = act ? wk - ay : -1e300, amax = act ? fabs( ay ) : 0.0;
        const long long key = in ? ( long long ) nd : -1 - lane;
        for ( int o = 1; o < 32; o *= 2 ) {
            const double r0 = __shfl_down_sync( 0xffffffffu, rmin, o );
            const double r1 = __shfl_down_sync( 0xffffffffu, rmax, o );
            const double r2 = __shfl_down_sync( 0xffffffffu, amax, o );
            const long long ko = __shfl_down_sync( 0xffffffffu, key, o );
            if ( lane + o < 32 && ko == key ) { rmin = fmin( rmin, r0 ); rmax = fmax( rmax, r1 ); amax = fmax( amax, r2 ); }
        }
        const long long kp = __shfl_up_sync( 0xffffffffu, key, 1 );
        if ( act && ( lane == 0 || kp != key ) ) {
            atomicMin( &red[ nd ].rmin, gpu2d::ord_of( rmin ) );
            atomicMax( &red[ nd ].rmax, gpu2d::ord_of( rmax ) );
            atomicMax( &red[ nd ].amax, gpu2d::ord_of( amax ) );
        }
        if ( h > 1 && in ) {
            const TN lc = nd + 1;
            nd = k < SI( end( lc ) ) ? lc : nd + ( TN( 1 ) << ( h - 1 ) );
        }
    }
}

/// STEP 3's rule for node `i`: the slope kept or dropped, the constant and its margin
template<int D>
__device__ __forceinline__ void majorant_of( const MajStatsN<D> &s, const MajRed &r, double ( &a )[ D ], double &wb ) {
    double bb = s.wmax, amax = 0;
    bool any = false;
    for ( int d = 0; d < D; ++d ) { a[ d ] = s.a[ d ]; any |= a[ d ] != 0; }
    const double spread = s.wmax - s.wmin;
    if ( any ) {
        const double rmin = gpu2d::double_of( r.rmin ), rmax = gpu2d::double_of( r.rmax );
        const double u = 1 - double( D ) / ( s.m - 1 ), by_chance = sqrt( u > 0 ? u : 0.0 );
        if ( rmax - rmin < 0.85 * by_chance * spread ) {
            bb = rmax;
            amax = gpu2d::double_of( r.amax );
        } else
            for ( int d = 0; d < D; ++d ) a[ d ] = 0;
    }
    wb = bb + 1e-6 * ( fabs( bb ) + spread + amax );
    if ( s.m == 0 ) {
        for ( int d = 0; d < D; ++d ) a[ d ] = 0;
        wb = 0;
    }
}

/// STEP 3, one thread per node: the choice of the slope, the constant and its margin
template<int D,class TJ,class TW>
__global__ void __launch_bounds__( BLOCK ) majorant_write( SI nb_nodes, const MajStatsN<D> *st, const MajRed *red,
                                                          StridedOut<TW,2> wa_out, StridedOut<TW,1> wb_out ) {
    const SI i = SI( blockIdx.x ) * BLOCK + threadIdx.x;
    if ( i >= nb_nodes )
        return;
    double a[ D ], wb;
    majorant_of<D>( st[ i ], red[ i ], a, wb );
    for ( int d = 0; d < D; ++d )
        wa_out( i, d ) = TW( a[ d ] );
    wb_out( i ) = std::is_same_v<TW,float> ? TW( __double2float_ru( wb ) ) : TW( wb );   // rounded UP: still a majorant
}

/// STEP 3 for the card's Newton ( `Newton2D.cuh` ): the walk's float RECORDS ( `NodeT`: `Cell3D.cuh::Node< true, TR >`, box
/// rounded outward, constant rounded up ), and the tree's tensors if `wa_out` is given ( what the diagram takes back )
template<int D,class TB,class TJ,class TW,class NodeT>
__global__ void __launch_bounds__( BLOCK ) majorant_write_nodes( SI nb_nodes, Strided<TB,3> box, Strided<TJ,1> beg, Strided<TJ,1> end,
                                                                const MajStatsN<D> *st, const MajRed *red, NodeT *out,
                                                                StridedOut<TW,2> wa_out, StridedOut<TW,1> wb_out ) {
    const SI i = SI( blockIdx.x ) * BLOCK + threadIdx.x;
    if ( i >= nb_nodes )
        return;
    double a[ D ], wb;
    majorant_of<D>( st[ i ], red[ i ], a, wb );
    if ( out ) {
        NodeT nd;
        for ( int d = 0; d < D; ++d ) {
            nd.lo[ d ] = __double2float_rd( double( box( i, 0, d ) ) );
            nd.hi[ d ] = __double2float_ru( double( box( i, 1, d ) ) );
            nd.a[ d ] = float( a[ d ] );
        }
        nd.b = __double2float_ru( wb );
        nd.beg = decltype( nd.beg )( beg( i ) );
        nd.end = decltype( nd.end )( end( i ) );
        out[ i ] = nd;
    }
    if ( wa_out.p ) {
        for ( int d = 0; d < D; ++d )
            wa_out( i, d ) = TW( a[ d ] );
        wb_out( i ) = std::is_same_v<TW,float> ? TW( __double2float_ru( wb ) ) : TW( wb );
    }
}

/// THE REFRESH for a solver that redoes the majorants at every diagram ( `gpu2d::Majorants`' interface, any `D` ): the work
/// buffers taken once, then the launches for one weight vector ( rank order ) into the walk's records
template<int D,class TR,class TN>
struct MajorantsN {
    MajStatsN<D> *st = nullptr;
    MajRed       *red = nullptr;
    SI            nb_nodes = 0;
    int           depth = 0;

    bool prepare( auto &allocator, const auto &pd ) {
        nb_nodes = SI( pd.tree.node_begin.shape( 0 ) );
        depth = 0;
        for ( SI m = nb_nodes; m; m >>= 1 )
            ++depth;
        st  = static_cast<MajStatsN<D> *>( take( allocator, SI( sizeof( MajStatsN<D> ) ) * std::max<SI>( nb_nodes, 1 ) ) );
        red = static_cast<MajRed *>( take( allocator, SI( sizeof( MajRed ) ) * std::max<SI>( nb_nodes, 1 ) ) );
        return st && red;
    }

    template<class PD,class NodeT,class TW = double>
    void refresh( const CudaQueue &queue, const PD &pd, const double *w, NodeT *nodes, StridedOut<TW,2> wa = {}, StridedOut<TW,1> wb = {} ) const {
        const SI n = SI( pd.nb_seeds() );
        const auto box = strided( pd.tree.node_box );
        const auto beg = strided( pd.tree.node_begin ), end = strided( pd.tree.node_end );
        const auto pos = strided( pd.sorted_positions );
        using TB = std::remove_const_t<typename std::decay_t<decltype( pd.tree.node_box )>::TF>;
        using TJ = std::remove_const_t<typename std::decay_t<decltype( pd.tree.node_begin )>::TF>;
        using TF = std::remove_const_t<typename std::decay_t<decltype( pd.sorted_positions )>::TF>;
        for ( int h = 1; h <= depth; ++h ) {
            const TN nb = TN( 1 ) << ( depth - h );
            launch_kernel( queue, &majorant_up<D,TB,TJ,TF,TN>, int( ( SI( nb ) + BLOCK - 1 ) / BLOCK ), BLOCK, 0, depth, h, nb, box, beg, end, pos, w, st, red );
        }
        launch_kernel( queue, &majorant_spread<D,TJ,TF,TN>, blocks_for( n ), BLOCK, 0, depth, n, end, pos, w, ( const MajStatsN<D> * ) st, red );
        launch_kernel( queue, &majorant_write_nodes<D,TB,TJ,TW,NodeT>, blocks_for( nb_nodes ), BLOCK, 0, nb_nodes, box, beg, end,
                       ( const MajStatsN<D> * ) st, ( const MajRed * ) red, nodes, wa, wb );
    }
};

/// `node_wa` / `node_wb` of the tree ( `node_box`, `node_begin`, `node_end`, preorder ) for the seeds `sorted_positions` /
/// `sorted_weights` ( tree order ). `TN`: the node index.
template<int D,class TN>
void refresh_majorants( const CudaQueue &queue, const auto &node_box, const auto &node_begin, const auto &node_end,
                        const auto &sorted_positions, const auto &sorted_weights, auto &&node_wa, auto &&node_wb, auto &allocator ) {
    using TW  = std::remove_const_t<typename std::decay_t<decltype( node_wa )>::TF>;
    using TB  = std::remove_const_t<typename std::decay_t<decltype( node_box )>::TF>;
    using TJ  = std::remove_const_t<typename std::decay_t<decltype( node_begin )>::TF>;
    using TF  = std::remove_const_t<typename std::decay_t<decltype( sorted_positions )>::TF>;
    using TFW = std::remove_const_t<typename std::decay_t<decltype( sorted_weights )>::TF>;
    const SI n = SI( sorted_positions.shape( 0 ) ), nb_nodes = SI( node_begin.shape( 0 ) );
    int depth = 0;
    for ( SI m = nb_nodes; m; m >>= 1 )
        ++depth;
    double *w = static_cast<double *>( take( allocator, SI( sizeof( double ) ) * std::max<SI>( n, 1 ) ) );
    auto *st  = static_cast<MajStatsN<D> *>( take( allocator, SI( sizeof( MajStatsN<D> ) ) * std::max<SI>( nb_nodes, 1 ) ) );
    auto *red = static_cast<MajRed *>( take( allocator, SI( sizeof( MajRed ) ) * std::max<SI>( nb_nodes, 1 ) ) );
    if ( ! w || ! st || ! red )
        return;
    launch_kernel( queue, &gpu2d::bsp_weights_to_double<TFW>, blocks_for( n ), BLOCK, 0, n, strided( sorted_weights ), w );
    const auto box = strided( node_box );
    const auto beg = strided( node_begin ), end = strided( node_end );
    const auto pos = strided( sorted_positions );
    for ( int h = 1; h <= depth; ++h ) {
        const TN nb = TN( 1 ) << ( depth - h );
        launch_kernel( queue, &majorant_up<D,TB,TJ,TF,TN>, int( ( SI( nb ) + BLOCK - 1 ) / BLOCK ), BLOCK, 0, depth, h, nb, box, beg, end, pos,
                       ( const double * ) w, st, red );
    }
    launch_kernel( queue, &majorant_spread<D,TJ,TF,TN>, blocks_for( n ), BLOCK, 0, depth, n, end, pos, ( const double * ) w,
                   ( const MajStatsN<D> * ) st, red );
    launch_kernel( queue, &majorant_write<D,TJ,TW>, blocks_for( nb_nodes ), BLOCK, 0, nb_nodes, ( const MajStatsN<D> * ) st,
                   ( const MajRed * ) red, strided_out<TW,2>( node_wa ), strided_out<TW,1>( node_wb ) );
}

} // namespace sdot::gpu3d
