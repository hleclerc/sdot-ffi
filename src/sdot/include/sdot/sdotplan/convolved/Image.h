#pragma once

// `Convolved` of an image ( see `sdotplan/Convolved.h` ). Included at the end of `Image.h`.
#include <loom/support/algorithms/CartesianIndices.h>
#include <algorithm>
#include "../Convolved.h"

namespace sdot {
namespace sdotplan {

/// an image: its blurred values, in a buffer of our own
template<class... T>
struct Convolved<Image<T...>> {
    using Dist = Image<T...>;
    using TF = typename Dist::TF;
    static constexpr int D = Dist::ct_dim;
    const Dist &dist;
    std::vector<double> src, buf, tmp;                   ///< the values, flattened ( C order )
    std::vector<TF> out;
    std::optional<Dist> current;
    static constexpr bool possible = true;

    Convolved( const Dist &dist ) : dist( dist ) {
        auto shape = dist.values.shape();
        CartesianIndices<DECAYED_TYPE_OF( shape )> cells{ shape };
        src.resize( PI( cells.size() ) );
        for ( PI flat = 0; flat < PI( cells.size() ); ++flat )
            // `cells[ flat ]` is a `Coords` ( NAMED coordinates ), and a tensor accepts
            // one directly -- no need to unpack it into positional integers any more
            src[ flat ] = double( dist.values( cells[ flat ] ) );
    }

    /// the grid step along axis `a` ( the length of `frame( a )` times the mean spacing of the knots ),
    /// and the smallest of them
    double step( int a ) const {
        return dist.with_defaults( [&]( auto &&img ) {       // `frame` / `knots` may be absent: their defaults
            double l2 = 0;
            for ( int c = 0; c < D; ++c ) { const double f = double( img.frame( a, c ) ); l2 += f * f; }
            const SI nb = SI( img.values.shape( a ) );
            const double extent = double( img.knots( a, nb ) ) - double( img.knots( a, 0 ) );
            return std::sqrt( l2 ) * extent / std::max<SI>( nb, 1 );
        } );
    }
    double min_scale( double ) const {
        double r = step( 0 );
        for ( int a = 1; a < D; ++a ) r = std::min( r, step( a ) );
        return r / 4;
    }

    const Dist &at( double s ) {
        if ( s <= 0 ) return dist;
        // the separable filter: along each axis, a Gaussian of standard deviation `s / step` in pixels,
        // truncated at four standard deviations and renormalized cell by cell ( the border )
        buf = src;
        SI shape[ D ], stride[ D ];
        SI total = 1;
        for ( int a = D - 1; a >= 0; --a ) { shape[ a ] = SI( dist.values.shape( a ) ); stride[ a ] = total; total *= shape[ a ]; }
        for ( int a = 0; a < D; ++a ) {
            const double sp = s / step( a );
            if ( ! ( sp > 1e-3 ) ) continue;
            const SI r = std::min<SI>( SI( std::ceil( 4 * sp ) ), shape[ a ] );
            std::vector<double> ker( 2 * r + 1 );
            for ( SI k = -r; k <= r; ++k ) ker[ k + r ] = std::exp( -0.5 * double( k * k ) / ( sp * sp ) );
            tmp.assign( PI( total ), 0.0 );
            for ( SI flat = 0; flat < total; ++flat ) {
                const SI i = ( flat / stride[ a ] ) % shape[ a ];
                double sum = 0, wsum = 0;
                const SI lo = std::max<SI>( -r, -i ), hi = std::min<SI>( r, shape[ a ] - 1 - i );
                for ( SI k = lo; k <= hi; ++k ) { sum += ker[ k + r ] * buf[ flat + k * stride[ a ] ]; wsum += ker[ k + r ]; }
                tmp[ flat ] = sum / wsum;
            }
            buf.swap( tmp );
        }
        out.assign( buf.begin(), buf.end() );
        using V = DECAYED_TYPE_OF( dist.values );
        auto view = tensor_view<typename V::MemorySpace>( out.data(), dist.values.shape(), typename V::AxisNames{} );
        current.emplace( Dist{ dist.target_mass, dist.nb_dims, dist.shape, view, dist.origin, dist.frame, dist.knots,
                               dist.current_mass, dist.nb_cells_cum, dist.cell_cum_mass } );
        return *current;
    }
};


} // namespace sdotplan
} // namespace sdot
