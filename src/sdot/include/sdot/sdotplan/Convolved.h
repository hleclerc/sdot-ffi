#pragma once

// THE CONVOLVED DENSITY, the primary template: `Convolved<Dist>` cannot convolve anything. A distribution
// that knows how to convolve itself specializes this template IN ITS OWN HEADER ( `sdotplan/convolved/<Dist>.h`,
// included at the end of `<Dist>.h` ): the solver thus includes no distribution -- that would require
// the generated header of each one, which only exists if a call has handled one.
#include <optional>
#include <vector>
#include <cmath>

namespace sdot {
namespace sdotplan {

/// the distribution `dist` at width `s`: `at( s )` returns a reference valid until the next `at`
template<class Dist>
struct Convolved {
    const Dist &dist;
    Convolved( const Dist &dist ) : dist( dist ) {}
    static constexpr bool possible = false;              ///< nothing to convolve
    const Dist &at( double ) { return dist; }
    double min_scale( double ) const { return 0; }     ///< below which the convolution no longer changes anything
};

} // namespace sdotplan
} // namespace sdot
