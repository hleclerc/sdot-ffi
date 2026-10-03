#pragma once

// `Convolved` of the Gaussians ( see `sdotplan/Convolved.h` ). Included at the end of `SumOfGaussians.h`.
#include "../Convolved.h"

namespace sdot {
namespace sdotplan {

/// the Gaussians: one copy at `conv_s = s`
template<class... T>
struct Convolved<SumOfGaussians<T...>> {
    using Dist = SumOfGaussians<T...>;
    const Dist &dist;
    std::optional<Dist> current;
    static constexpr bool possible = true;
    Convolved( const Dist &dist ) : dist( dist ) {}
    const Dist &at( double s ) {
        if ( s <= 0 ) return dist;
        current.emplace( dist.with_convolution( typename Dist::TF( s ) ) );
        return *current;
    }
    double min_scale( double ) const { return double( dist.smallest_sigma() ) / 4; }
};


} // namespace sdotplan
} // namespace sdot
