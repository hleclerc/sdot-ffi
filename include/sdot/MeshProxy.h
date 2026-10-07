#pragma once

// =====================================================================================
// THE MOMENT TREE OF A MESH: the groups of simplices that `Mesh.blur_proxy` turns into gaussians, built in C++ for meshes
// of 1e8 simplices.
//
// What it does is the numpy version's ( `Mesh.blur_proxy` ): a k-d bisection of the simplices, along the principal axis of the
// covariance of the group, at its mass median, until a group weighs under `target` and is not elongated
// ( `elongation`, largest / smallest eigenvalue ); a light group cut only for its elongation must gain by it ( `gain` ), or it
// stays a leaf. Each leaf comes out as its mass, its mean and `tr( covariance ) / D`.
//
// What changes is the memory and the time:
//   * NOTHING is stored per simplex but its index ( 32 bits ): the moments ( exact, P1 on a simplex, any dimension ) are
//     recomputed from the nodes when a group needs them -- once per level of the tree, about `log2( leaves )` times;
//   * a LARGE group ( more than `big` simplices ) is split by passes parallel over its simplices: a histogram of the keys gives
//     the mass median, a scatter by blocks the partition, and the moments of the two halves come out of the same pass;
//   * the SMALL groups, once there are enough of them, go to the threads whole, each one split to its leaves by one thread.
//
// The leaves come out ordered by their first simplex, whatever the threads did: the result is deterministic.
// =====================================================================================

#include <loom/support/common_macros.h>
#include <loom/support/kernels/CpuQueue.h>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <vector>

namespace sdot {
namespace mesh_proxy {

/// the moments of a set of simplices: `int rho`, `int x rho`, `int x x^T rho`
template<int D>
struct Moments {
    double m = 0;
    double f[ D ] = {};
    double s[ D ][ D ] = {};

    void add( const Moments &o ) {
        m += o.m;
        for ( int a = 0; a < D; ++a ) {
            f[ a ] += o.f[ a ];
            for ( int b = 0; b < D; ++b )
                s[ a ][ b ] += o.s[ a ][ b ];
        }
    }
};

/// what a group needs to decide: its mass, its mean, `tr( cov ) / D`, the elongation and the principal axis
template<int D>
struct Shape {
    double m, mu[ D ], var, ratio, axis[ D ];
};

/// the eigenvalues ( ascending ) and the unit eigenvector of the largest, of a symmetric `D x D` matrix -- cyclic Jacobi
template<int D>
void principal( double A[ D ][ D ], double &lmin, double &lmax, double *axis ) {
    double V[ D ][ D ];
    for ( int i = 0; i < D; ++i )
        for ( int j = 0; j < D; ++j )
            V[ i ][ j ] = i == j;
    for ( int sweep = 0; sweep < 32; ++sweep ) {
        double off = 0, diag = 0;
        for ( int i = 0; i < D; ++i ) {
            diag += A[ i ][ i ] * A[ i ][ i ];
            for ( int j = i + 1; j < D; ++j )
                off += A[ i ][ j ] * A[ i ][ j ];
        }
        if ( off <= 1e-30 * diag || off == 0 )
            break;
        for ( int p = 0; p < D; ++p )
            for ( int q = p + 1; q < D; ++q ) {
                if ( A[ p ][ q ] == 0 )
                    continue;
                const double th = ( A[ q ][ q ] - A[ p ][ p ] ) / ( 2 * A[ p ][ q ] );
                const double t = ( th >= 0 ? 1.0 : -1.0 ) / ( std::fabs( th ) + std::sqrt( th * th + 1 ) );
                const double c = 1 / std::sqrt( t * t + 1 ), s = t * c;
                for ( int k = 0; k < D; ++k ) {              // A <- A J
                    const double akp = A[ k ][ p ], akq = A[ k ][ q ];
                    A[ k ][ p ] = c * akp - s * akq;
                    A[ k ][ q ] = s * akp + c * akq;
                }
                for ( int k = 0; k < D; ++k ) {              // A <- J^T A
                    const double apk = A[ p ][ k ], aqk = A[ q ][ k ];
                    A[ p ][ k ] = c * apk - s * aqk;
                    A[ q ][ k ] = s * apk + c * aqk;
                }
                for ( int k = 0; k < D; ++k ) {              // V <- V J
                    const double vkp = V[ k ][ p ], vkq = V[ k ][ q ];
                    V[ k ][ p ] = c * vkp - s * vkq;
                    V[ k ][ q ] = s * vkp + c * vkq;
                }
            }
    }
    int imin = 0, imax = 0;
    for ( int i = 1; i < D; ++i ) {
        if ( A[ i ][ i ] < A[ imin ][ imin ] ) imin = i;
        if ( A[ i ][ i ] > A[ imax ][ imax ] ) imax = i;
    }
    lmin = A[ imin ][ imin ];
    lmax = A[ imax ][ imax ];
    for ( int k = 0; k < D; ++k )
        axis[ k ] = V[ k ][ imax ];
}

template<int D>
Shape<D> shape_of( const Moments<D> &g ) {
    Shape<D> r;
    r.m = g.m;
    double cov[ D ][ D ];
    for ( int a = 0; a < D; ++a )
        r.mu[ a ] = g.f[ a ] / g.m;
    double tr = 0;
    for ( int a = 0; a < D; ++a ) {
        for ( int b = 0; b < D; ++b )
            cov[ a ][ b ] = 0.5 * ( g.s[ a ][ b ] + g.s[ b ][ a ] ) / g.m - r.mu[ a ] * r.mu[ b ];
        tr += cov[ a ][ a ];
    }
    r.var = std::max( tr / D, 1e-300 );
    double lmin, lmax;
    principal<D>( cov, lmin, lmax, r.axis );
    r.ratio = lmax > 0 ? lmax / std::max( lmin, 1e-300 ) : 1.0;
    return r;
}

template<int D,class Mesh>
struct Builder {
    using Id = std::uint32_t;

    const CpuQueue &queue;
    const Mesh     &mesh;
    double          target, elongation, gain;
    SI              big = SI( 1 ) << 15;                 ///< a group over this is split by parallel passes
    int             nt;

    std::vector<Id>    ids, tmp;                         ///< the simplices, grouped by ranges; a buffer for the scatter
    std::vector<float> keys;                             ///< the keys of a large group

    struct Leaf { Id first; double m, mu[ D ], var; };
    std::vector<std::vector<Leaf>> leaves_th;

    // the exact integrals of products of barycentric coordinates, divided by the volume
    double w1 = 0, w2[ D + 1 ][ D + 1 ], w3[ D + 1 ][ D + 1 ][ D + 1 ];

    Builder( const CpuQueue &queue, const Mesh &mesh, double target, double elongation, double gain )
        : queue( queue ), mesh( mesh ), target( target ), elongation( elongation ), gain( gain ), nt( std::max( queue.nb_workers(), 1 ) ) {
        auto fact = []( int k ) { double r = 1; for ( int i = 2; i <= k; ++i ) r *= i; return r; };
        const double fd = fact( D );
        w1 = 1.0 / ( D + 1 );
        for ( int i = 0; i <= D; ++i )
            for ( int j = 0; j <= D; ++j ) {
                w2[ i ][ j ] = fd * ( i == j ? 2 : 1 ) / fact( D + 2 );
                for ( int k = 0; k <= D; ++k ) {
                    const int mult = ( i == j && j == k ) ? 6 : ( i == j || j == k || i == k ) ? 2 : 1;
                    w3[ i ][ j ][ k ] = fd * mult / fact( D + 3 );
                }
            }
        leaves_th.resize( nt );
    }

    /// the corners and the values of simplex `e`, and its volume
    double corners( Id e, double P[ D + 1 ][ D ], double v[ D + 1 ] ) const {
        for ( int i = 0; i <= D; ++i ) {
            const SI n = SI( mesh.simplices( SI( e ), i ) );
            for ( int c = 0; c < D; ++c )
                P[ i ][ c ] = double( mesh.nodes( n, c ) );
            v[ i ] = double( mesh.values( SI( e ), i ) );
        }
        double A[ D ][ D ];
        for ( int r = 0; r < D; ++r )
            for ( int c = 0; c < D; ++c )
                A[ r ][ c ] = P[ r + 1 ][ c ] - P[ 0 ][ c ];
        double det = 1;                                  // Gaussian elimination with partial pivoting
        for ( int c = 0; c < D; ++c ) {
            int p = c;
            for ( int r = c + 1; r < D; ++r )
                if ( std::fabs( A[ r ][ c ] ) > std::fabs( A[ p ][ c ] ) ) p = r;
            if ( A[ p ][ c ] == 0 ) return 0;
            if ( p != c ) { for ( int k = 0; k < D; ++k ) std::swap( A[ p ][ k ], A[ c ][ k ] ); det = -det; }
            det *= A[ c ][ c ];
            for ( int r = c + 1; r < D; ++r ) {
                const double f = A[ r ][ c ] / A[ c ][ c ];
                for ( int k = c; k < D; ++k ) A[ r ][ k ] -= f * A[ c ][ k ];
            }
        }
        double fd = 1;
        for ( int i = 2; i <= D; ++i ) fd *= i;
        return std::fabs( det ) / fd;
    }

    /// the mass and the first moment of simplex `e` ( what a key needs )
    double first( Id e, double *f ) const {
        double P[ D + 1 ][ D ], v[ D + 1 ];
        const double vol = corners( e, P, v );
        double m = 0;
        for ( int i = 0; i <= D; ++i ) m += v[ i ];
        for ( int c = 0; c < D; ++c ) f[ c ] = 0;
        for ( int i = 0; i <= D; ++i ) {
            double a = 0;
            for ( int j = 0; j <= D; ++j ) a += v[ j ] * w2[ i ][ j ];
            for ( int c = 0; c < D; ++c ) f[ c ] += vol * a * P[ i ][ c ];
        }
        return vol * m * w1;
    }

    /// all the moments of simplex `e`
    void moments( Id e, Moments<D> &g ) const {
        double P[ D + 1 ][ D ], v[ D + 1 ];
        const double vol = corners( e, P, v );
        double m = 0;
        for ( int i = 0; i <= D; ++i ) m += v[ i ];
        g.m = vol * m * w1;
        for ( int a = 0; a < D; ++a ) {
            g.f[ a ] = 0;
            for ( int b = 0; b < D; ++b ) g.s[ a ][ b ] = 0;
        }
        for ( int i = 0; i <= D; ++i ) {
            double a1 = 0;
            for ( int j = 0; j <= D; ++j ) a1 += v[ j ] * w2[ i ][ j ];
            for ( int c = 0; c < D; ++c ) g.f[ c ] += vol * a1 * P[ i ][ c ];
            for ( int j = 0; j <= D; ++j ) {
                double a2 = 0;
                for ( int k = 0; k <= D; ++k ) a2 += v[ k ] * w3[ i ][ j ][ k ];
                for ( int a = 0; a < D; ++a )
                    for ( int b = 0; b < D; ++b )
                        g.s[ a ][ b ] += vol * a2 * P[ i ][ a ] * P[ j ][ b ];
            }
        }
    }

    double key_of( Id e, const double *axis ) const {
        double f[ D ];
        const double m = first( e, f );
        double k = 0;
        for ( int c = 0; c < D; ++c ) k += f[ c ] * axis[ c ];
        return m > 0 ? k / m : 0;
    }

    void par( SI n, auto &&fn ) const {
        queue.run_threads( nt, [&]( int t ) {
            fn( t, SI( ( long long ) t * n / nt ), SI( ( long long ) ( t + 1 ) * n / nt ) );
        } );
    }

    struct Group { SI b, e; Moments<D> mom; };

    void emit( int t, const Group &g, const Shape<D> &sh ) {
        Leaf l;
        l.first = ids[ g.b ];
        l.m = sh.m;
        for ( int c = 0; c < D; ++c ) l.mu[ c ] = sh.mu[ c ];
        l.var = sh.var;
        leaves_th[ t ].push_back( l );
    }

    /// `true` if `g` is a leaf ( then emitted ): its size, its weight and its elongation decide
    bool is_leaf( const Group &g, const Shape<D> &sh ) const {
        return g.e - g.b == 1 || ( ! ( sh.m > target ) && sh.ratio <= elongation );
    }

    /// the no-gain rule: a light group cut for its elongation only, whose halves are not less elongated
    bool no_gain( const Shape<D> &sh, const Moments<D> &l, const Moments<D> &r ) const {
        if ( sh.m > target )
            return false;
        return std::min( shape_of<D>( l ).ratio, shape_of<D>( r ).ratio ) > gain * sh.ratio;
    }

    /// a SMALL group, to its leaves, on thread `t`
    void split_small( int t, Group root ) {
        struct Item { double key; Id id; Moments<D> mom; };
        std::vector<Item> items;
        std::vector<Group> stack{ root };
        while ( ! stack.empty() ) {
            Group g = stack.back();
            stack.pop_back();
            const Shape<D> sh = shape_of<D>( g.mom );
            if ( is_leaf( g, sh ) ) { emit( t, g, sh ); continue; }
            const SI n = g.e - g.b;
            items.resize( n );
            for ( SI k = 0; k < n; ++k ) {
                Item &it = items[ k ];
                it.id = ids[ g.b + k ];
                moments( it.id, it.mom );
                double kk = 0;
                for ( int c = 0; c < D; ++c ) kk += it.mom.f[ c ] * sh.axis[ c ];
                it.key = it.mom.m > 0 ? kk / it.mom.m : 0;
            }
            std::sort( items.begin(), items.end(), []( const Item &a, const Item &b ) { return a.key < b.key || ( a.key == b.key && a.id < b.id ); } );
            // the mass median: the first item where the cumulated mass reaches half, at least one item on each side
            double cum = 0;
            SI cut = 0;
            for ( ; cut < n - 1; ++cut ) {
                cum += items[ cut ].mom.m;
                if ( cum >= 0.5 * sh.m ) break;
            }
            cut = std::min( cut, n - 2 ) + 1;
            Group l{ g.b, g.b + cut, {} }, r{ g.b + cut, g.e, {} };
            for ( SI k = 0; k < n; ++k ) {
                ids[ g.b + k ] = items[ k ].id;
                ( k < cut ? l : r ).mom.add( items[ k ].mom );
            }
            if ( no_gain( sh, l.mom, r.mom ) ) { emit( t, g, sh ); continue; }
            stack.push_back( r );
            stack.push_back( l );
        }
    }

    /// a LARGE group, split once by parallel passes: its two halves, with their moments
    void split_big( const Group &g, const Shape<D> &sh, Group &l, Group &r ) {
        const SI n = g.e - g.b;
        // the keys, and their range
        std::vector<double> lo_th( nt, 1e300 ), hi_th( nt, -1e300 );
        par( n, [&]( int t, SI b, SI e ) {
            for ( SI k = b; k < e; ++k ) {
                const float key = float( key_of( ids[ g.b + k ], sh.axis ) );
                keys[ g.b + k ] = key;
                lo_th[ t ] = std::min( lo_th[ t ], double( key ) );
                hi_th[ t ] = std::max( hi_th[ t ], double( key ) );
            }
        } );
        const double klo = *std::min_element( lo_th.begin(), lo_th.end() ), khi = *std::max_element( hi_th.begin(), hi_th.end() );
        // the mass median, by a histogram of the masses over the keys
        constexpr int NB = 4096;
        std::vector<std::vector<double>> hist( nt, std::vector<double>( NB, 0.0 ) );
        const double scale = khi > klo ? NB / ( khi - klo ) : 0;
        auto bin = [&]( float key ) { return std::min( NB - 1, std::max( 0, int( ( double( key ) - klo ) * scale ) ) ); };
        par( n, [&]( int t, SI b, SI e ) {
            for ( SI k = b; k < e; ++k ) {
                double f[ D ];
                hist[ t ][ bin( keys[ g.b + k ] ) ] += first( ids[ g.b + k ], f );
            }
        } );
        double cum = 0;
        int split = 0;                                   // the halves: bins `< split` and `>= split`
        for ( ; split < NB; ++split ) {
            double h = 0;
            for ( int t = 0; t < nt; ++t ) h += hist[ t ][ split ];
            if ( cum + h >= 0.5 * sh.m ) break;
            cum += h;
        }
        // the bin that holds the median goes to the lighter side
        double h = 0;
        for ( int t = 0; t < nt; ++t ) h += split < NB ? hist[ t ][ split ] : 0;
        if ( cum + h - 0.5 * sh.m < 0.5 * sh.m - cum ) ++split;
        // the partition, by blocks: count, then scatter ( stable ), with the moments of both sides
        std::vector<SI> nl( nt, 0 );
        par( n, [&]( int t, SI b, SI e ) {
            SI c = 0;
            for ( SI k = b; k < e; ++k ) c += bin( keys[ g.b + k ] ) < split;
            nl[ t ] = c;
        } );
        SI total_l = 0;
        std::vector<SI> off_l( nt ), off_r( nt );
        for ( int t = 0; t < nt; ++t ) { off_l[ t ] = total_l; total_l += nl[ t ]; }
        if ( total_l == 0 || total_l == n ) {
            // every key in one bin ( a degenerate group ): halves by count, in the order they are
            total_l = n / 2;
            l = Group{ g.b, g.b + total_l, {} };
            r = Group{ g.b + total_l, g.e, {} };
        } else {
            SI acc = total_l;
            for ( int t = 0; t < nt; ++t ) {
                off_r[ t ] = acc;
                SI b = SI( ( long long ) t * n / nt ), e = SI( ( long long ) ( t + 1 ) * n / nt );
                acc += ( e - b ) - nl[ t ];
            }
            par( n, [&]( int t, SI b, SI e ) {
                SI pl = off_l[ t ], pr = off_r[ t ];
                for ( SI k = b; k < e; ++k ) {
                    const Id id = ids[ g.b + k ];
                    tmp[ g.b + ( bin( keys[ g.b + k ] ) < split ? pl++ : pr++ ) ] = id;
                }
            } );
            par( n, [&]( int, SI b, SI e ) {
                std::copy( tmp.begin() + g.b + b, tmp.begin() + g.b + e, ids.begin() + g.b + b );
            } );
            l = Group{ g.b, g.b + total_l, {} };
            r = Group{ g.b + total_l, g.e, {} };
        }
        sum_moments( l );
        sum_moments( r );
    }

    void sum_moments( Group &g ) {
        std::vector<Moments<D>> th( nt );
        par( g.e - g.b, [&]( int t, SI b, SI e ) {
            Moments<D> m;
            for ( SI k = b; k < e; ++k ) {
                moments( ids[ g.b + k ], m );
                th[ t ].add( m );
            }
        } );
        g.mom = Moments<D>{};
        for ( auto &m : th ) g.mom.add( m );
    }

    /// THE TREE: `ids` the simplices with a positive mass, then the large groups one at a time, then the small ones on the threads
    void run( SI nb_elems ) {
        // the simplices with a mass ( a zero density brings nothing to a proxy )
        std::vector<std::vector<Id>> keep( nt );
        par( nb_elems, [&]( int t, SI b, SI e ) {
            double f[ D ];
            for ( SI k = b; k < e; ++k )
                if ( first( Id( k ), f ) > 0 )
                    keep[ t ].push_back( Id( k ) );
        } );
        for ( auto &k : keep ) { ids.insert( ids.end(), k.begin(), k.end() ); std::vector<Id>().swap( k ); }
        if ( ids.empty() )
            return;
        Group root{ 0, SI( ids.size() ), {} };
        if ( root.e > big ) {
            tmp.resize( ids.size() );
            keys.resize( ids.size() );
        }
        sum_moments( root );

        std::vector<Group> stack{ root }, small;
        while ( ! stack.empty() ) {
            Group g = stack.back();
            stack.pop_back();
            if ( g.e - g.b <= big ) { small.push_back( g ); continue; }
            const Shape<D> sh = shape_of<D>( g.mom );
            if ( is_leaf( g, sh ) ) { emit( 0, g, sh ); continue; }
            Group l, r;
            split_big( g, sh, l, r );
            if ( no_gain( sh, l.mom, r.mom ) ) { emit( 0, g, sh ); continue; }
            stack.push_back( r );
            stack.push_back( l );
        }
        std::vector<Id>().swap( tmp );
        std::vector<float>().swap( keys );

        std::atomic<SI> next{ 0 };
        queue.run_threads( nt, [&]( int t ) {
            for ( SI k; ( k = next++ ) < SI( small.size() ); )
                split_small( t, small[ k ] );
        } );
    }
};

/// THE ENTRY POINT ( `Mesh.blur_proxy` ): the leaves in `mass [ cap ]`, `mu [ cap, D ]`, `var [ cap ]`, their number in `count( 0 )`
/// -- which can exceed `cap`, in which case nothing is written and the caller retries with more room
template<int D>
void build( const CpuQueue &queue, const auto &mesh, double target, double elongation, double gain,
            auto &&mass, auto &&mu, auto &&var, auto &&count ) {
    using M = DECAYED_TYPE_OF( mesh );
    Builder<D,M> bld( queue, mesh, target, elongation, gain );
    bld.run( SI( mesh.simplices.shape( 0 ) ) );

    std::vector<typename Builder<D,M>::Leaf> all;
    for ( auto &v : bld.leaves_th ) all.insert( all.end(), v.begin(), v.end() );
    std::sort( all.begin(), all.end(), []( const auto &a, const auto &b ) { return a.first < b.first; } );
    count( 0 ) = SI( all.size() );
    const SI cap = SI( mass.shape( 0 ) );
    if ( SI( all.size() ) > cap )
        return;
    for ( SI l = 0; l < SI( all.size() ); ++l ) {
        mass( l ) = all[ l ].m;
        for ( int c = 0; c < D; ++c ) mu( l, c ) = all[ l ].mu[ c ];
        var( l ) = all[ l ].var;
    }
}

} // namespace mesh_proxy
} // namespace sdot
