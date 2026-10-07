#pragma once

// `Convolved` of the Gaussians ( see `sdotplan/Convolved.h` ). Included at the end of `SumOfGaussians.h`.
#include "../Convolved.h"

namespace sdot {
namespace sdotplan {

/// the Gaussians: one copy at `conv_s = s`, carrying a TREE ( `GaussianTree.h` ) once there are enough of them -- each
/// cell then integrates only the terms near it, merged at the width of the stage
template<class... T>
struct Convolved<SumOfGaussians<T...>> {
    using Dist = SumOfGaussians<T...>;
    using TF = typename Dist::TF;
    static constexpr int D = Dist::ct_dim;
    static constexpr SI min_for_tree = 16;               ///< below, every cell sees every gaussian: the tree would cost more
    const Dist &dist;
    std::optional<Dist> current;
    GaussianTree<TF,D> tree;
    bool with_tree;
    static constexpr bool possible = true;

    Convolved( const Dist &dist ) : dist( dist ), with_tree( SI( dist.nb_gaussians ) >= min_for_tree ) {
        if ( with_tree )
            tree.build( int( dist.nb_gaussians ), [&]( int i, TF *c, TF &s, TF &w ) {
                for ( int d = 0; d < D; ++d ) c[ d ] = TF( dist.positions( i, d ) );
                s = TF( dist.sigmas( i ) );
                w = TF( dist.weights( i ) );
            } );
    }

    const Dist &at( double s ) {
        if ( s <= 0 && ! with_tree ) return dist;
        current.emplace( dist.with_convolution( typename Dist::TF( s > 0 ? s : 0 ) ) );
        if ( with_tree )
            current->tree = &tree;
        return *current;
    }
    double min_scale( double ) const { return double( dist.smallest_sigma() ) / 4; }
};


} // namespace sdotplan
} // namespace sdot
