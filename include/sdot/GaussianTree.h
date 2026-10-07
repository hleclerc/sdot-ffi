#pragma once

// =====================================================================================
// A TREE OF GAUSSIANS: what lets a cell see only the gaussians that weigh on it, whatever their number.
//
// Without it, a cell of a power diagram integrates EVERY gaussian of a `SumOfGaussians` ( O( n G ) per sweep, O( n^2 )
// for a proxy of `4 n` gaussians ). Two remarks bring it down to O( n ):
//
//   * there is NO FAR FIELD. A gaussian is worth nothing a few widths away -- relative to the other terms seen by a
//     cell, not only in absolute terms ( `gather` ) -- so a cell only needs the gaussians near it.
//     A k-d tree on the centres, pruned by a bound of what each node can put on the cell, finds them.
//
//   * at a continuation width `s`, a group of gaussians of radius `r << s` IS one gaussian: the convolved sum differs
//     from the single gaussian of the same mass, mean and ( isotropic ) second moment by `O( ( r / s )^2 )` of its mass.
//     A node of the tree whose radius is under `theta` times its width is therefore emitted as ONE term. The number of
//     terms a cell sees is then about `( k / theta )^d` ( `k` the number of widths `eps` keeps ), whatever `s` and `G`.
//
// The merging is for the stages of a continuation only ( `s > 0` ): at `s = 0` the terms are the gaussians themselves,
// truncated at `eps_exact`.
//
// Host only: `gather` fills a per-thread buffer. The tree is built by `Convolved<SumOfGaussians>` ( the solver ) and
// hung on the density it hands out ( `SumOfGaussians::tree` ).
// =====================================================================================

#include <algorithm>
#include <cmath>
#include <numeric>
#include <vector>

namespace sdot {

/// one term of a sum of gaussians: its centre, its width ( convolution included ), its mass
template<class TF,int D>
struct GaussianTerm {
    TF c[ D ];
    TF s;
    TF w;
};

template<class TF,int D>
struct GaussianTree {
    using Term = GaussianTerm<TF,D>;

    struct Node {
        double lo[ D ], hi[ D ];                         ///< the box of the centres
        double mu[ D ];                                  ///< the mean ( weighted by the masses )
        double mass;
        double var;                                      ///< the isotropic variance of the group: `sum w ( s^2 + |c - mu|^2 / D ) / mass`
        double s_min, s_max;                             ///< the widths of its gaussians
        double radius;                                   ///< the half diagonal of `lo, hi`
        int    begin, end;                               ///< its gaussians, in `items`
        int    left = -1, right = -1;                    ///< its children ( `-1`: a leaf )
    };

    static constexpr int leaf_size = 4;

    double theta      = 0.5;                             ///< a node is ONE term when its radius is under `theta` times its width
    double eps        = 1e-6;                            ///< the truncation, relative to the strongest term of the cell, during the continuation
    double eps_exact  = 1e-13;                           ///< ... and at `s = 0`
    std::vector<Term> items;                             ///< the gaussians, in tree order, with their own widths
    std::vector<Node> nodes;

    GaussianTree() = default;

    /// `get( i, c, s, w )`: the centre coordinate `c`, the width and the mass of gaussian `i`
    void build( int nb, auto &&get ) {
        items.resize( nb );
        for ( int i = 0; i < nb; ++i ) {
            TF c[ D ], s, w;
            get( i, c, s, w );
            for ( int d = 0; d < D; ++d ) items[ i ].c[ d ] = c[ d ];
            items[ i ].s = s;
            items[ i ].w = w;
        }
        nodes.clear();
        nodes.reserve( 2 * ( nb / leaf_size + 1 ) );
        if ( nb )
            build_node( 0, nb );
    }

    int build_node( int b, int e ) {
        const int id = int( nodes.size() );
        nodes.emplace_back();
        Node nd;
        nd.begin = b;
        nd.end = e;
        for ( int d = 0; d < D; ++d ) { nd.lo[ d ] = 1e300; nd.hi[ d ] = -1e300; nd.mu[ d ] = 0; }
        nd.mass = 0;
        nd.s_min = 1e300;
        nd.s_max = 0;
        for ( int i = b; i < e; ++i ) {
            const Term &t = items[ i ];
            const double w = std::max( double( t.w ), 0.0 );
            for ( int d = 0; d < D; ++d ) {
                nd.lo[ d ] = std::min( nd.lo[ d ], double( t.c[ d ] ) );
                nd.hi[ d ] = std::max( nd.hi[ d ], double( t.c[ d ] ) );
                nd.mu[ d ] += w * double( t.c[ d ] );
            }
            nd.mass += w;
            nd.s_min = std::min( nd.s_min, double( t.s ) );
            nd.s_max = std::max( nd.s_max, double( t.s ) );
        }
        for ( int d = 0; d < D; ++d )
            nd.mu[ d ] = nd.mass > 0 ? nd.mu[ d ] / nd.mass : ( nd.lo[ d ] + nd.hi[ d ] ) / 2;
        double v = 0, r2 = 0;
        for ( int i = b; i < e; ++i ) {
            const Term &t = items[ i ];
            double e2 = 0;
            for ( int d = 0; d < D; ++d )
                e2 += ( double( t.c[ d ] ) - nd.mu[ d ] ) * ( double( t.c[ d ] ) - nd.mu[ d ] );
            v += std::max( double( t.w ), 0.0 ) * ( double( t.s ) * double( t.s ) + e2 / D );
        }
        nd.var = nd.mass > 0 ? v / nd.mass : nd.s_max * nd.s_max;
        for ( int d = 0; d < D; ++d )
            r2 += ( nd.hi[ d ] - nd.lo[ d ] ) * ( nd.hi[ d ] - nd.lo[ d ] );
        nd.radius = std::sqrt( r2 ) / 2;

        if ( e - b > leaf_size ) {
            int ax = 0;                                  // the widest axis, cut at the median
            for ( int d = 1; d < D; ++d )
                if ( nd.hi[ d ] - nd.lo[ d ] > nd.hi[ ax ] - nd.lo[ ax ] )
                    ax = d;
            const int m = ( b + e ) / 2;
            std::nth_element( items.begin() + b, items.begin() + m, items.begin() + e,
                              [ax]( const Term &x, const Term &y ) { return x.c[ ax ] < y.c[ ax ]; } );
            nd.left = build_node( b, m );
            nd.right = build_node( m, e );
        }
        nodes[ id ] = nd;
        return id;
    }

    /// THE TERMS that weigh on the box `[ lo, hi ]` at the continuation width `s`, into `out` ( cleared first ).
    ///
    /// The truncation is RELATIVE TO THE CELL: a term is kept when its largest possible value on the box is at least `eps` times
    /// the largest one of all the terms. An absolute cut ( a fixed number of widths ) would empty a cell lying in a desert --
    /// all it receives there are tails, and they are the whole of its density.
    void gather( const double *lo, const double *hi, double s, std::vector<Term> &out ) const {
        out.clear();
        if ( nodes.empty() )
            return;
        const double s2 = s * s;
        const double rel = s > 0 ? eps : eps_exact;
        auto dist2 = [&]( const double *a, const double *b ) {   // between the boxes `[ a, b ]` and `[ lo, hi ]`
            double r = 0;
            for ( int d = 0; d < D; ++d ) {
                const double g = std::max( { a[ d ] - hi[ d ], lo[ d ] - b[ d ], 0.0 } );
                r += g * g;
            }
            return r;
        };
        auto peak = [&]( double v ) {                    // `( 2 pi v ) ^ ( -D/2 )`
            double r = 1;
            for ( int d = 0; d < D; ++d ) r /= std::sqrt( 6.283185307179586 * v );
            return r;
        };
        double best = 0;
        bounds.clear();
        auto emit = [&]( const Term &t, double ub ) {
            if ( ub < rel * best ) return;
            best = std::max( best, ub );
            out.push_back( t );
            bounds.push_back( ub );
        };
        int stack[ 128 ];
        int sp = 0;
        stack[ sp++ ] = 0;
        while ( sp ) {
            const Node &nd = nodes[ stack[ --sp ] ];
            // a bound of what the node can put on the box: its mass, at its nearest point, with its widest and narrowest terms
            const double vmax = std::max( nd.s_max * nd.s_max, nd.var ) + s2, vmin = nd.s_min * nd.s_min + s2;
            const double ub = nd.mass * std::exp( - dist2( nd.lo, nd.hi ) / ( 2 * vmax ) ) * peak( vmin );
            if ( ub < rel * best || ! ( nd.mass > 0 ) )
                continue;
            if ( s > 0 && nd.radius <= theta * std::sqrt( vmin ) ) {
                const double v = nd.var + s2;
                Term t;
                for ( int d = 0; d < D; ++d ) t.c[ d ] = TF( nd.mu[ d ] );
                t.s = TF( std::sqrt( v ) );
                t.w = TF( nd.mass );
                emit( t, nd.mass * std::exp( - dist2( nd.mu, nd.mu ) / ( 2 * v ) ) * peak( v ) );
                continue;
            }
            if ( nd.left < 0 ) {
                for ( int i = nd.begin; i < nd.end; ++i ) {
                    const Term &g = items[ i ];
                    const double v = double( g.s ) * double( g.s ) + s2;
                    double c[ D ];
                    for ( int d = 0; d < D; ++d ) c[ d ] = double( g.c[ d ] );
                    Term t = g;
                    t.s = TF( std::sqrt( v ) );
                    emit( t, std::max( double( g.w ), 0.0 ) * std::exp( - dist2( c, c ) / ( 2 * v ) ) * peak( v ) );
                }
                continue;
            }
            // the nearer child last, so that it is visited first: `best` grows early, and prunes more
            const Node &l = nodes[ nd.left ], &r = nodes[ nd.right ];
            const bool left_nearer = dist2( l.lo, l.hi ) <= dist2( r.lo, r.hi );
            stack[ sp++ ] = left_nearer ? nd.right : nd.left;
            stack[ sp++ ] = left_nearer ? nd.left : nd.right;
        }
        // the terms kept before `best` reached its final value
        size_t k = 0;
        for ( size_t i = 0; i < out.size(); ++i )
            if ( bounds[ i ] >= rel * best )
                out[ k++ ] = out[ i ];
        out.resize( k );
    }

    static inline thread_local std::vector<double> bounds;   ///< the bounds of the terms of `gather`, per thread
};

} // namespace sdot
