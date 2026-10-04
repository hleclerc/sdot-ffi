#pragma once

// =====================================================================================
// THE TREE'S WEIGHT MAJORANTS, REDONE ON THE CARD FROM NEW WEIGHTS ( 2D ): what a Newton step changes in the tree,
// and nothing else ( `AaBsp.py`: the shape of the tree depends on the positions only ).
//
// The rule is `bsp_weight_majorant` ( `bsp_build_level.h`, `AaBsp.py::_weight_majorant` ), node by node: the affine
// least-squares fit of the weights of the node's seeds, kept only if it clearly tightens the spread of the residuals
// and its slope is well behaved, then `b = max( w - a . y )` plus a margin. What changes is HOW: the host version runs
// one work item per node over its whole slice -- the root alone reads the million seeds on one thread, 676 ms at 1e6
// on the card. Here, three kinds of launches, every one of them parallel over the seeds or over a level of nodes:
//
//   1. UP THE TREE, one launch per height: a leaf sums its few seeds ( count, means, centred second moments of the
//      positions and of positions x weights, the weight extrema ); an inner node COMBINES its two children ( the
//      parallel-axis rule: `M = M1 + M2 + delta delta^T m1 m2 / m` ), exactly, without reading a seed. The candidate
//      slope of the node is solved there ( the same 2 x 2 Gauss with partial pivoting, the same guards ).
//   2. THE SPREAD OF THE RESIDUALS `w - a . y` under each node's candidate slope -- which no combination of the
//      children gives, the slope being the node's own: one thread per seed walks down from the root to its leaf
//      ( the nodes it meets are its ancestors ), and at each level the lanes of a warp that share a node reduce
//      together ( their ranks are contiguous, so are their nodes' slices ), then ONE atomic min / max per node and
//      warp, on the ordered-integer image of the double ( exact: a min / max does not depend on the order ).
//   3. THE NODE RECORDS of the card's walk ( `Cell2D.cuh::Node`, float, box rounded outward, constant rounded up ),
//      and on request the tree's own tensors ( `node_wa`, `node_wb`: what the diagram takes back after a solve ).
//
// Nothing here depends on the order in which the card runs things: the sums are those of a fixed tree, the extrema
// are exact. The fit may differ from the host's in the last bits ( combination against sums ), which only moves the
// rounding of a pruning bound -- the cells do not depend on the majorant, only the work does.
// =====================================================================================

#include "Reduce.cuh"

namespace sdot::gpu2d {

/// a double as an unsigned integer with the same order, negatives included: what makes `atomicMin` / `atomicMax` exact
__device__ __forceinline__ unsigned long long ord_of( double d ) {
    const unsigned long long u = ( unsigned long long ) __double_as_longlong( d );
    return ( u >> 63 ) ? ~u : ( u | 0x8000000000000000ull );
}
__device__ __forceinline__ double double_of( unsigned long long u ) {
    return __longlong_as_double( ( long long ) ( ( u >> 63 ) ? ( u & 0x7fffffffffffffffull ) : ~u ) );
}

/// what step 1 leaves per node
struct MajStats {
    double m;                                            ///< number of seeds ( 0: an empty slot )
    double my[ 2 ], mw;                                  ///< means
    double M[ 3 ];                                       ///< centred `sum q q^T`: xx, xy, yy
    double r[ 2 ];                                       ///< centred `sum q dw`
    double wmin, wmax;
    double a[ 2 ];                                       ///< the candidate slope ( zero: none )
};

/// what step 2 leaves per node ( ordered integers )
struct MajRed {
    unsigned long long rmin, rmax, amax;
};

/// the node of the `j`-th position, left to right, among the nodes of height `h` ( preorder: left child `n + 1`, right
/// child `n + 2^( height - 1 )` ), in a perfect tree of depth `depth`
template<class TN>
__device__ __forceinline__ TN node_at( int depth, int h, TN j ) {
    TN n = 0;
    int H = depth;
    for ( int b = depth - h - 1; b >= 0; --b ) {
        n += ( ( j >> b ) & 1 ) ? ( TN( 1 ) << ( H - 1 ) ) : TN( 1 );
        --H;
    }
    return n;
}

/// STEP 1, the nodes of height `h` ( `nb` of them )
template<class TB,class TJ,class TF,class TN>
__global__ void __launch_bounds__( BLOCK ) majorant_up( int depth, int h, TN nb, Strided<TB,3> box, Strided<TJ,1> beg, Strided<TJ,1> end,
                                                       Strided<TF,2> pos, const double *w, MajStats *st, MajRed *red ) {
    const TN j = TN( blockIdx.x ) * BLOCK + threadIdx.x;
    if ( j >= nb )
        return;
    const TN n = node_at( depth, h, j );
    MajStats s;
    s.m = 0; s.my[ 0 ] = s.my[ 1 ] = s.mw = 0; s.M[ 0 ] = s.M[ 1 ] = s.M[ 2 ] = 0; s.r[ 0 ] = s.r[ 1 ] = 0;
    s.wmin = s.wmax = 0; s.a[ 0 ] = s.a[ 1 ] = 0;
    const SI b = SI( beg( n ) ), e = SI( end( n ) );
    if ( e > b ) {
        if ( h == 1 ) {
            // a leaf: its seeds, as `bsp_weight_majorant` sums them
            double sx = 0, sy = 0, sw = 0;
            s.wmin = s.wmax = w[ b ];
            for ( SI k = b; k < e; ++k ) {
                const double v = w[ k ];
                sx += double( pos( k, 0 ) ); sy += double( pos( k, 1 ) ); sw += v;
                s.wmin = fmin( s.wmin, v ); s.wmax = fmax( s.wmax, v );
            }
            s.m = double( e - b );
            s.my[ 0 ] = sx / s.m; s.my[ 1 ] = sy / s.m; s.mw = sw / s.m;
            for ( SI k = b; k < e; ++k ) {
                const double q0 = double( pos( k, 0 ) ) - s.my[ 0 ], q1 = double( pos( k, 1 ) ) - s.my[ 1 ], dw = w[ k ] - s.mw;
                s.M[ 0 ] += q0 * q0; s.M[ 1 ] += q0 * q1; s.M[ 2 ] += q1 * q1;
                s.r[ 0 ] += q0 * dw; s.r[ 1 ] += q1 * dw;
            }
        } else {
            // an inner node: its two children, combined ( one of them may be an empty slot )
            const MajStats L = st[ n + 1 ], R = st[ n + ( TN( 1 ) << ( h - 1 ) ) ];
            if ( L.m == 0 || R.m == 0 ) {
                s = L.m == 0 ? R : L;
            } else {
                s.m = L.m + R.m;
                const double f = L.m * R.m / s.m, g = R.m / s.m;
                const double dx = R.my[ 0 ] - L.my[ 0 ], dy = R.my[ 1 ] - L.my[ 1 ], dw = R.mw - L.mw;
                s.my[ 0 ] = L.my[ 0 ] + dx * g; s.my[ 1 ] = L.my[ 1 ] + dy * g; s.mw = L.mw + dw * g;
                s.M[ 0 ] = L.M[ 0 ] + R.M[ 0 ] + dx * dx * f;
                s.M[ 1 ] = L.M[ 1 ] + R.M[ 1 ] + dx * dy * f;
                s.M[ 2 ] = L.M[ 2 ] + R.M[ 2 ] + dy * dy * f;
                s.r[ 0 ] = L.r[ 0 ] + R.r[ 0 ] + dx * dw * f;
                s.r[ 1 ] = L.r[ 1 ] + R.r[ 1 ] + dy * dw * f;
                s.wmin = fmin( L.wmin, R.wmin ); s.wmax = fmax( L.wmax, R.wmax );
            }
            s.a[ 0 ] = s.a[ 1 ] = 0;
        }

        // the candidate slope: the normal equations by Gauss with partial pivoting, then the guards on the slope
        const double spread = s.wmax - s.wmin;
        if ( s.m >= 6 && spread > 0 ) {
            double A[ 2 ][ 3 ] = { { s.M[ 0 ], s.M[ 1 ], s.r[ 0 ] }, { s.M[ 1 ], s.M[ 2 ], s.r[ 1 ] } };
            bool ok = true;
            if ( fabs( A[ 1 ][ 0 ] ) > fabs( A[ 0 ][ 0 ] ) )
                for ( int c = 0; c < 3; ++c ) { const double t = A[ 0 ][ c ]; A[ 0 ][ c ] = A[ 1 ][ c ]; A[ 1 ][ c ] = t; }
            if ( ! ( fabs( A[ 0 ][ 0 ] ) > 0 ) ) ok = false;
            if ( ok ) {
                const double f = A[ 1 ][ 0 ] / A[ 0 ][ 0 ];
                A[ 1 ][ 1 ] -= f * A[ 0 ][ 1 ];
                A[ 1 ][ 2 ] -= f * A[ 0 ][ 2 ];
                if ( ! ( fabs( A[ 1 ][ 1 ] ) > 0 ) ) ok = false;
            }
            if ( ok ) {
                const double f1 = A[ 1 ][ 2 ] / A[ 1 ][ 1 ];
                const double f0 = ( A[ 0 ][ 2 ] - A[ 0 ][ 1 ] * f1 ) / A[ 0 ][ 0 ];
                const double fit[ 2 ] = { f0, f1 };
                bool well_behaved = true;
                for ( int d = 0; d < 2; ++d ) {
                    const double lo = double( box( n, 0, d ) ), hi = double( box( n, 1, d ) );
                    const double reach = fmax( fabs( lo ), fabs( hi ) );
                    if ( fabs( fit[ d ] ) * ( hi - lo ) > 8 * spread || fabs( fit[ d ] ) * reach > 100 * spread )
                        well_behaved = false;
                }
                if ( well_behaved && isfinite( f0 ) && isfinite( f1 ) ) {
                    s.a[ 0 ] = f0;
                    s.a[ 1 ] = f1;
                }
            }
        }
    }
    st[ n ] = s;
    red[ n ] = MajRed{ ord_of( 1e300 ), ord_of( -1e300 ), ord_of( 0.0 ) };
}

/// STEP 2, one thread per seed ( rank `k` ), walking from the root to its leaf
template<class TJ,class TF,class TN>
__global__ void __launch_bounds__( BLOCK ) majorant_spread( int depth, SI n, Strided<TJ,1> end, Strided<TF,2> pos, const double *w,
                                                           const MajStats *st, MajRed *red ) {
    const SI k = SI( blockIdx.x ) * BLOCK + threadIdx.x;
    const bool in = k < n;
    const int lane = threadIdx.x & 31;
    const double x = in ? double( pos( k, 0 ) ) : 0.0, y = in ? double( pos( k, 1 ) ) : 0.0, wk = in ? w[ k ] : 0.0;
    TN nd = 0;
    for ( int h = depth; h >= 1; --h ) {
        double a0 = 0, a1 = 0;
        if ( in ) { a0 = st[ nd ].a[ 0 ]; a1 = st[ nd ].a[ 1 ]; }
        const bool act = in && ( a0 != 0 || a1 != 0 );
        const double ay = a0 * x + a1 * y;
        double rmin = act ? wk - ay : 1e300, rmax = act ? wk - ay : -1e300, amax = act ? fabs( ay ) : 0.0;
        // the lanes that share a node are contiguous: a segmented reduction toward the first lane of each segment
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
            atomicMin( &red[ nd ].rmin, ord_of( rmin ) );
            atomicMax( &red[ nd ].rmax, ord_of( rmax ) );
            atomicMax( &red[ nd ].amax, ord_of( amax ) );
        }
        if ( h > 1 && in ) {
            const TN lc = nd + 1;
            nd = k < SI( end( lc ) ) ? lc : nd + ( TN( 1 ) << ( h - 1 ) );
        }
    }
}

/// STEP 3, one thread per node: the choice of the slope, the constant and its margin, the records
template<class TR,class TB,class TJ>
__global__ void __launch_bounds__( BLOCK ) majorant_write( SI nb_nodes, Strided<TB,3> box, Strided<TJ,1> beg, Strided<TJ,1> end,
                                                          const MajStats *st, const MajRed *red, Node<true,TR> *out,
                                                          StridedOut<double,2> wa_out, StridedOut<double,1> wb_out ) {
    const SI i = SI( blockIdx.x ) * BLOCK + threadIdx.x;
    if ( i >= nb_nodes )
        return;
    const MajStats s = st[ i ];
    double a0 = s.a[ 0 ], a1 = s.a[ 1 ], bb = s.wmax, amax = 0;
    const double spread = s.wmax - s.wmin;
    if ( a0 != 0 || a1 != 0 ) {
        const double rmin = double_of( red[ i ].rmin ), rmax = double_of( red[ i ].rmax );
        const double u = 1 - 2.0 / ( s.m - 1 ), by_chance = sqrt( u > 0 ? u : 0.0 );
        if ( rmax - rmin < 0.85 * by_chance * spread ) {
            bb = rmax;
            amax = double_of( red[ i ].amax );
        } else
            a0 = a1 = 0;
    }
    double wb = bb + 1e-6 * ( fabs( bb ) + spread + amax );
    if ( s.m == 0 ) { a0 = a1 = 0; wb = 0; }

    Node<true,TR> nd;
    for ( int d = 0; d < 2; ++d ) {
        nd.lo[ d ] = __double2float_rd( double( box( i, 0, d ) ) );
        nd.hi[ d ] = __double2float_ru( double( box( i, 1, d ) ) );
    }
    nd.a[ 0 ] = float( a0 );
    nd.a[ 1 ] = float( a1 );
    nd.b = __double2float_ru( wb );
    nd.beg = TR( beg( i ) );
    nd.end = TR( end( i ) );
    out[ i ] = nd;
    if ( wa_out.p ) {
        wa_out( i, 0 ) = a0;
        wa_out( i, 1 ) = a1;
        wb_out( i ) = wb;
    }
}

/// the weights of the seeds in the form the card's kernel reads them ( two floats, or the double )
template<class TK>
__global__ void __launch_bounds__( BLOCK ) pack_weights( SI n, const double *w, typename KernelSeeds<TK>::Wt *out ) {
    const SI k = SI( blockIdx.x ) * BLOCK + threadIdx.x;
    if ( k >= n )
        return;
    const double v = w[ k ];
    if constexpr ( std::is_same_v<TK,float> ) {
        const float vh = __double2float_rn( v );
        out[ k ] = make_float2( vh, __double2float_rn( v - double( vh ) ) );
    } else
        out[ k ] = v;
}

/// THE REFRESH: the work buffers ( taken once ), and the launches for one weight vector
template<class TR,class TN>
struct Majorants {
    MajStats *st = nullptr;
    MajRed   *red = nullptr;
    SI        nb_nodes = 0;
    int       depth = 0;

    bool prepare( auto &allocator, const auto &pd ) {
        nb_nodes = SI( pd.tree.node_begin.shape( 0 ) );
        depth = 0;
        for ( SI m = nb_nodes; m; m >>= 1 )
            ++depth;
        st  = static_cast<MajStats *>( take( allocator, SI( sizeof( MajStats ) ) * std::max<SI>( nb_nodes, 1 ) ) );
        red = static_cast<MajRed *>( take( allocator, SI( sizeof( MajRed ) ) * std::max<SI>( nb_nodes, 1 ) ) );
        return st && red;
    }

    /// the majorants of `w` ( rank order, `n` seeds ) into `nodes` ( the card's records ), and into `wa` / `wb` if given
    template<class PD>
    void refresh( const CudaQueue &queue, const PD &pd, const double *w, Node<true,TR> *nodes,
                  StridedOut<double,2> wa = {}, StridedOut<double,1> wb = {} ) const {
        using TB = std::remove_const_t<typename std::decay_t<decltype( pd.tree.node_box )>::TF>;
        using TJ = std::remove_const_t<typename std::decay_t<decltype( pd.tree.node_begin )>::TF>;
        using TF = std::remove_const_t<typename std::decay_t<decltype( pd.sorted_positions )>::TF>;
        const SI n = SI( pd.nb_seeds() );
        const auto box = strided( pd.tree.node_box );
        const auto beg = strided( pd.tree.node_begin ), end = strided( pd.tree.node_end );
        const auto pos = strided( pd.sorted_positions );
        for ( int h = 1; h <= depth; ++h ) {
            const TN nb = TN( 1 ) << ( depth - h );
            launch_kernel( queue, &majorant_up<TB,TJ,TF,TN>, int( ( SI( nb ) + BLOCK - 1 ) / BLOCK ), BLOCK, 0,
                           depth, h, nb, box, beg, end, pos, w, st, red );
        }
        launch_kernel( queue, &majorant_spread<TJ,TF,TN>, blocks_for( n ), BLOCK, 0, depth, n, end, pos, w, ( const MajStats * ) st, red );
        launch_kernel( queue, &majorant_write<TR,TB,TJ>, blocks_for( nb_nodes ), BLOCK, 0, nb_nodes, box, beg, end,
                       ( const MajStats * ) st, ( const MajRed * ) red, nodes, wa, wb );
    }
};

} // namespace sdot::gpu2d
