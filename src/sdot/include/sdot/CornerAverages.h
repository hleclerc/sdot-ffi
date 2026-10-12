#pragma once

// =====================================================================================
// THE CORNER AVERAGES OF A DG1 DENSITY ( `sdot.corner_averages` ): the spreading of a density given by a value per node of
// each simplex, for the width continuation, on its own mesh.
//
// One iteration is two averages:
//   * `A`: the corners of a simplex take its mean;
//   * `N`: each corner takes the mean, weighted by `|e| / ( D + 1 )`, of the corners of all the simplices touching its node.
// After the first one the field is continuous ( a value per node ) and an iteration is `v <- W v`, a gather over the simplices
// then a sum over the star of each node -- assembled once here as a CSR
// node -> nodes ( ~ 7 non zeros per node in 2D, ~ 15 in 3D, against 2 ( D + 1 ) x 6 or 2 ( D + 1 ) x 24 reads unassembled ). `W` keeps the mass ( `sum m v`, `m` the lumped
// masses ), the positivity, and has its spectrum in [ 0, 1 ].
//
// The state `k` ( in the outputs, `out( s, e, i )` for `k = ks( s )` ):
//   * `k <= 1`: `( 1 - k ) cv + k v1`, `v1 = N A cv` -- `k = 0` is the original;
//   * `method = 0`: `W^t v1`, `t = k - 1`, linear between two integers ( the `ks` are done in increasing order, one pass );
//   * `method = 1`: `exp( -t ( I - W ) ) v1` by its Chebyshev series on [ 0, 1 ]: `exp( -t ( 1 + y ) / 2 ) = sum_j c_j T_j( y )`,
//     `y = I - 2 W`, `c_j = ( 2 - [ j = 0 ] ) ( -1 )^j e^{-z} I_j( z )`, `z = t / 2`, truncated where `e^{-z} I_j( z ) max( v1 )`
//     falls under `eps mean( v1 )` -- about `sqrt( 2 t ln( 1 / eps ) )` products instead of `t`. The negative lobes ( ~ eps ) are
//     clipped and the mass put back.
// The `e^{-z} I_j( z )` come from the backward recurrence ( Miller ), normalized by `I_0 + 2 sum_j I_j = e^z`.
// =====================================================================================

#include <loom/support/common_macros.h>
#include <loom/support/kernels/CpuQueue.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <numeric>
#include <type_traits>
#include <utility>
#include <vector>

namespace sdot {
namespace corner_averages {

/// `f( i )` for `i` in `[ 0, n )`, by ranges over the threads
template<class F>
void par_for( const CpuQueue &queue, SI n, F &&f ) {
    const int nt = int( std::min<SI>( std::max( queue.nb_workers(), 1 ), std::max<SI>( n / 4096, 1 ) ) );
    if ( nt <= 1 ) {
        for ( SI i = 0; i < n; ++i )
            f( i );
        return;
    }
    queue.run_threads( nt, [&]( int t ) {
        const SI b = n * t / nt, e = n * ( t + 1 ) / nt;
        for ( SI i = b; i < e; ++i )
            f( i );
    } );
}

/// `e^{-z} I_j( z )` for `j` in `[ 0, J )`, `J` the first `j` where it falls under `tol` ( at least 1 )
inline std::vector<double> scaled_bessel( double z, double tol ) {
    if ( z <= 0 )
        return { 1.0 };
    const SI top = SI( z + 30 * std::sqrt( z + 1 ) + 60 );
    std::vector<double> r( top + 2, 0.0 );
    r[ top + 1 ] = 0; r[ top ] = 1e-300;
    for ( SI j = top; j > 0; --j ) {
        r[ j - 1 ] = r[ j + 1 ] + ( 2.0 * j / z ) * r[ j ];
        if ( r[ j - 1 ] > 1e250 )                               // ( rescale: the recurrence grows downwards )
            for ( SI i = j - 1; i <= top + 1; ++i )
                r[ i ] *= 1e-250;
    }
    double s = r[ 0 ];
    for ( SI j = 1; j <= top; ++j )
        s += 2 * r[ j ];
    SI J = 1;
    while ( J <= top && r[ J ] / s > tol )
        ++J;
    std::vector<double> out( J );
    for ( SI j = 0; j < J; ++j )
        out[ j ] = r[ j ] / s;
    return out;
}

template<int D,class Nodes,class Simplices>
struct Averager {
    static constexpr int K = D + 1;

    const CpuQueue  &queue;
    const Nodes     &nodes;
    const Simplices &simplices;
    SI               E, N;
    std::vector<double> w;                                      ///< the weight of each corner of a simplex, `|e| / ( D + 1 )`
    std::vector<double> m;                                      ///< the lumped mass of each node
    std::vector<SI>     star_off;                               ///< CSR node -> simplices
    std::vector<std::uint32_t> star;
    std::vector<double> me;                                     ///< the means of the simplices ( scratch )
    std::size_t nb_products = 0;

    Averager( const CpuQueue &queue, const Nodes &nodes, const Simplices &simplices ) : queue( queue ), nodes( nodes ), simplices( simplices ) {
        E = SI( simplices.shape( 0 ) );
        N = SI( nodes.shape( 0 ) );
        double fact = 1;
        for ( int i = 2; i <= D; ++i ) fact *= i;
        w.resize( E );
        par_for( queue, E, [&]( SI e ) {
            double a[ D ][ D ];
            for ( int r = 0; r < D; ++r )
                for ( int c = 0; c < D; ++c )
                    a[ r ][ c ] = nodes( SI( simplices( e, r + 1 ) ), c ) - nodes( SI( simplices( e, 0 ) ), c );
            double det = 1;                                     // ( Gaussian elimination with partial pivoting )
            for ( int c = 0; c < D; ++c ) {
                int p = c;
                for ( int r = c + 1; r < D; ++r )
                    if ( std::fabs( a[ r ][ c ] ) > std::fabs( a[ p ][ c ] ) ) p = r;
                if ( a[ p ][ c ] == 0 ) { det = 0; break; }
                if ( p != c ) { for ( int k = 0; k < D; ++k ) std::swap( a[ p ][ k ], a[ c ][ k ] ); det = -det; }
                det *= a[ c ][ c ];
                for ( int r = c + 1; r < D; ++r ) {
                    const double f = a[ r ][ c ] / a[ c ][ c ];
                    for ( int k = c; k < D; ++k ) a[ r ][ k ] -= f * a[ c ][ k ];
                }
            }
            w[ e ] = std::fabs( det ) / fact / K;
        } );

        // the stars ( sequential: one pass over the corners, cheap next to the products )
        star_off.assign( N + 1, 0 );
        for ( SI e = 0; e < E; ++e )
            for ( int i = 0; i < K; ++i )
                ++star_off[ SI( simplices( e, i ) ) + 1 ];
        std::partial_sum( star_off.begin(), star_off.end(), star_off.begin() );
        star.resize( star_off[ N ] );
        std::vector<SI> pos( star_off.begin(), star_off.end() - 1 );
        for ( SI e = 0; e < E; ++e )
            for ( int i = 0; i < K; ++i )
                star[ pos[ SI( simplices( e, i ) ) ]++ ] = std::uint32_t( e );
        m.assign( N, 0.0 );
        par_for( queue, N, [&]( SI n ) {
            double s = 0;
            for ( SI j = star_off[ n ]; j < star_off[ n + 1 ]; ++j )
                s += w[ star[ j ] ];
            m[ n ] = s;
        } );
        me.resize( E );

        // W, assembled: row n = sum over the star of n of w_e / ( K m_n ) on each corner of e ( duplicates merged )
        W_off.assign( N + 1, 0 );
        par_for( queue, N, [&]( SI n ) {
            SI c = 0;
            for ( SI j = star_off[ n ]; j < star_off[ n + 1 ]; ++j ) c += K;
            W_off[ n + 1 ] = c;                                 // ( an upper bound, compacted below )
        } );
        std::partial_sum( W_off.begin(), W_off.end(), W_off.begin() );
        W_col.resize( W_off[ N ] );
        W_val.resize( W_off[ N ] );
        std::vector<SI> len( N );
        par_for( queue, N, [&]( SI n ) {
            const SI b = W_off[ n ];
            SI l = 0;
            for ( SI j = star_off[ n ]; j < star_off[ n + 1 ]; ++j ) {
                const SI e = star[ j ];
                const double c = m[ n ] > 0 ? w[ e ] / ( K * m[ n ] ) : 0.0;
                for ( int i = 0; i < K; ++i ) {
                    const std::uint32_t col = std::uint32_t( simplices( e, i ) );
                    SI q = 0;
                    while ( q < l && W_col[ b + q ] != col ) ++q;
                    if ( q == l ) { W_col[ b + l ] = col; W_val[ b + l ] = 0; ++l; }
                    W_val[ b + q ] += c;
                }
            }
            len[ n ] = l;
        } );
        SI o = 0;                                               // ( compaction, sequential )
        for ( SI n = 0; n < N; ++n ) {
            const SI b = W_off[ n ];
            for ( SI q = 0; q < len[ n ]; ++q ) { W_col[ o + q ] = W_col[ b + q ]; W_val[ o + q ] = W_val[ b + q ]; }
            W_off[ n ] = o;
            o += len[ n ];
        }
        W_off[ N ] = o;
        W_col.resize( o ); W_val.resize( o );
    }

    std::vector<SI>            W_off;                           ///< W in CSR
    std::vector<std::uint32_t> W_col;
    std::vector<double>        W_val;

    double row( const std::vector<double> &x, SI n ) const {
        double s = 0;
        for ( SI j = W_off[ n ]; j < W_off[ n + 1 ]; ++j )
            s += W_val[ j ] * x[ W_col[ j ] ];
        return s;
    }

    /// `N`: the node values from the means of the simplices ( `me` )
    void to_nodes( std::vector<double> &v ) const {
        v.resize( N );
        par_for( queue, N, [&]( SI n ) {
            double s = 0;
            for ( SI j = star_off[ n ]; j < star_off[ n + 1 ]; ++j )
                s += w[ star[ j ] ] * me[ star[ j ] ];
            v[ n ] = m[ n ] > 0 ? s / m[ n ] : 0.0;
        } );
    }

    /// `dst = W src`
    void product( const std::vector<double> &src, std::vector<double> &dst ) {
        dst.resize( N );
        par_for( queue, N, [&]( SI n ) { dst[ n ] = row( src, n ); } );
        ++nb_products;
    }

    double mass( const std::vector<double> &v ) const {
        double s = 0;
        for ( SI n = 0; n < N; ++n )
            s += m[ n ] * v[ n ];
        return s;
    }
};

/// THE ENTRY POINT: see the header. `params = [ method, eps ]`; `stats( 0 )` the number of products of `W`
template<int D>
void run( const CpuQueue &queue, const auto &nodes, const auto &simplices, const auto &cv, const auto &ks, const auto &params,
          auto &&out, auto &&stats ) {
    using Nd = DECAYED_TYPE_OF( nodes );
    using Sx = DECAYED_TYPE_OF( simplices );
    Averager<D,Nd,Sx> av( queue, nodes, simplices );
    constexpr int K = D + 1;
    const SI E = av.E, N = av.N, S = SI( ks.shape( 0 ) );
    const int method = int( params( 0 ) );
    const double eps = double( params( 1 ) );

    // v1 = N A cv
    par_for( queue, E, [&]( SI e ) {
        double s = 0;
        for ( int i = 0; i < K; ++i )
            s += cv( e, i );
        av.me[ e ] = s / K;
    } );
    std::vector<double> v1;
    av.to_nodes( v1 );
    const double mass = av.mass( v1 );
    double vol = 0, vmax = 0;
    for ( SI e = 0; e < E; ++e ) vol += K * av.w[ e ];
    for ( SI n = 0; n < N; ++n ) vmax = std::max( vmax, std::fabs( v1[ n ] ) );

    auto write = [&]( SI s, double k, const std::vector<double> &v ) {
        const double a = std::min( k, 1.0 );                    // ( below 1: the mix with the original )
        par_for( queue, E, [&]( SI e ) {
            for ( int i = 0; i < K; ++i )
                out( s, e, i ) = k >= 1 ? v[ SI( simplices( e, i ) ) ] : ( 1 - a ) * cv( e, i ) + a * v1[ SI( simplices( e, i ) ) ];
        } );
    };

    std::vector<SI> order( S );
    std::iota( order.begin(), order.end(), SI( 0 ) );
    std::sort( order.begin(), order.end(), [&]( SI a, SI b ) { return ks( a ) < ks( b ); } );

    if ( method == 0 ) {
        // the powers, in one pass over the increasing `ks`
        std::vector<double> cur = v1, nxt( N ), mix( N );
        SI j = 0;
        for ( SI s : order ) {
            const double k = double( ks( s ) );
            if ( k <= 1 ) { write( s, k, v1 ); continue; }
            const double t = k - 1;
            const SI j0 = SI( std::floor( t ) );
            for ( ; j < j0; ++j ) { av.product( cur, nxt ); std::swap( cur, nxt ); }
            const double f = t - double( j0 );
            if ( f < 1e-12 ) { write( s, k, cur ); continue; }
            av.product( cur, nxt );
            for ( SI n = 0; n < N; ++n ) mix[ n ] = ( 1 - f ) * cur[ n ] + f * nxt[ n ];
            write( s, k, mix );
        }
    } else {
        // Chebyshev, each `k` on its own
        const double tol = eps * ( mass / vol ) / std::max( vmax, 1e-300 );
        std::vector<double> t0( N ), t1( N ), t2( N ), acc( N );
        for ( SI s : order ) {
            const double k = double( ks( s ) );
            if ( k <= 1 ) { write( s, k, v1 ); continue; }
            const std::vector<double> c = scaled_bessel( ( k - 1 ) / 2, tol );
            const SI J = SI( c.size() );
            // the recurrence `t2 = 2 Y t1 - t0`, `Y x = x - 2 W x`, and the accumulation: one pass per term
            t0 = v1;
            par_for( queue, N, [&]( SI n ) { acc[ n ] = c[ 0 ] * t0[ n ]; } );
            if ( J > 1 ) {
                par_for( queue, N, [&]( SI n ) { t1[ n ] = t0[ n ] - 2 * av.row( t0, n ); acc[ n ] -= 2 * c[ 1 ] * t1[ n ]; } );
                ++av.nb_products;
            }
            for ( SI j = 2; j < J; ++j ) {
                const double cj = 2 * c[ j ] * ( j % 2 ? -1.0 : 1.0 );
                par_for( queue, N, [&]( SI n ) { t2[ n ] = 2 * ( t1[ n ] - 2 * av.row( t1, n ) ) - t0[ n ]; acc[ n ] += cj * t2[ n ]; } );
                ++av.nb_products;
                std::swap( t0, t1 ); std::swap( t1, t2 );
            }
            par_for( queue, N, [&]( SI n ) { acc[ n ] = std::max( acc[ n ], 0.0 ); } );
            const double ma = av.mass( acc );
            if ( ma > 0 )
                par_for( queue, N, [&]( SI n ) { acc[ n ] *= mass / ma; } );
            write( s, k, acc );
        }
    }
    stats( 0 ) = SI( av.nb_products );
}

#if defined( __CUDACC__ )
/// A TENSOR OF THE CARD on the host: its whole span copied ( the strides in bytes, as the card's views have them ), read and
/// written by indices like the view; `upload` sends it back. The corner averages are a preparation of the density ( a few
/// sparse products on the host's threads ): on a CUDA device the call copies its tensors rather than having a second code
template<class T,int N>
struct HostCopy {
    std::vector<char> buf;
    SI                s[ N ], n[ N ];
    char             *dev = nullptr;

    template<class View>
    HostCopy( cudaStream_t stream, const View &v, bool read = true ) {
        SI span = SI( sizeof( T ) );
        bool empty = false;
        [&]<int... I>( std::integer_sequence<int,I...> ) {
            ( ( s[ I ] = SI( v._strides[ Ct<int,I>() ] ), n[ I ] = SI( v.shape( I ) ) ), ... );
        }( std::make_integer_sequence<int,N>() );
        for ( int d = 0; d < N; ++d ) {
            empty = empty || n[ d ] == 0;
            span += ( n[ d ] > 0 ? n[ d ] - 1 : 0 ) * s[ d ];
        }
        buf.resize( empty ? 0 : span );
        dev = reinterpret_cast<char *>( const_cast<void *>( static_cast<const void *>( v.data().raw ) ) );
        if ( read && ! buf.empty() ) {
            cudaMemcpyAsync( buf.data(), dev, buf.size(), cudaMemcpyDeviceToHost, stream );
            cudaStreamSynchronize( stream );
        }
    }
    SI shape( int d ) const { return n[ d ]; }
    T &operator()( SI i ) { return *reinterpret_cast<T *>( buf.data() + i * s[ 0 ] ); }
    T &operator()( SI i, SI j ) { return *reinterpret_cast<T *>( buf.data() + i * s[ 0 ] + j * s[ 1 ] ); }
    T &operator()( SI i, SI j, SI k ) { return *reinterpret_cast<T *>( buf.data() + i * s[ 0 ] + j * s[ 1 ] + k * s[ 2 ] ); }
    T operator()( SI i ) const { return *reinterpret_cast<const T *>( buf.data() + i * s[ 0 ] ); }
    T operator()( SI i, SI j ) const { return *reinterpret_cast<const T *>( buf.data() + i * s[ 0 ] + j * s[ 1 ] ); }
    T operator()( SI i, SI j, SI k ) const { return *reinterpret_cast<const T *>( buf.data() + i * s[ 0 ] + j * s[ 1 ] + k * s[ 2 ] ); }
    void upload( cudaStream_t stream ) {
        if ( ! buf.empty() ) {
            cudaMemcpyAsync( dev, buf.data(), buf.size(), cudaMemcpyHostToDevice, stream );
            cudaStreamSynchronize( stream );
        }
    }
};

template<class View>
using HostCopyOf = HostCopy<std::remove_const_t<typename std::decay_t<View>::TF>,std::decay_t<View>::ct_rank>;

/// THE ENTRY POINT ON A CUDA DEVICE: the tensors copied to the host, the host's code, the outputs sent back
template<int D,class Queue> requires ( ! std::is_same_v<Queue,CpuQueue> )
void run( const Queue &queue, const auto &nodes, const auto &simplices, const auto &cv, const auto &ks, const auto &params,
          auto &&out, auto &&stats ) {
    const cudaStream_t st = queue.stream;
    const HostCopyOf<decltype( nodes )> h_nodes( st, nodes );
    const HostCopyOf<decltype( simplices )> h_simplices( st, simplices );
    const HostCopyOf<decltype( cv )> h_cv( st, cv );
    const HostCopyOf<decltype( ks )> h_ks( st, ks );
    const HostCopyOf<decltype( params )> h_params( st, params );
    HostCopyOf<decltype( out )> h_out( st, out, false );
    HostCopyOf<decltype( stats )> h_stats( st, stats, false );
    run<D>( CpuQueue(), h_nodes, h_simplices, h_cv, h_ks, h_params, h_out, h_stats );
    h_out.upload( st );
    h_stats.upload( st );
}
#endif

} // namespace corner_averages
} // namespace sdot
