#pragma once

// =====================================================================================
// THE CONTINUATION IN WIDTH: the density convolved by a Gaussian of width `s`, from wide to zero.
//
// What the bench concluded about densities that concentrate ( `solvers_des_familles` README § 9 ):
// direct Newton breaks as soon as some cells have no mass left ( seeds far from any bump:
// no floor, zero Hessian rows ), and first-order methods do not
// fare better. What works is to FIRST SOLVE FOR THE SPREAD-OUT DENSITY --
// convolved by a wide Gaussian, hence positive everywhere -- then to tighten, `s / sqrt( 2 )`
// at each stage, the weights of one stage serving as the start of the next, down to the density
// itself. The ratio `sqrt( 2 )` is the best measured ( 2 and 2^( 1/4 ) make more diagrams ),
// and it is the step through the mass limits ( `Bounds.h` ) that makes the backtracking fall into the hard
// zone, where needles form. The tangent `dw / ds` brings nothing more with it: it
// is not taken up.
//
// What "to convolve" means, distribution by distribution:
//   * a sum of Gaussians: the widths become `sqrt( sigma_i^2 + s^2 )`, nothing else
//     ( `SumOfGaussians::with_convolution` );
//   * an image: its values blurred on its grid ( a separable Gaussian filter, truncated at
//     four standard deviations, renormalized at the edge: positive, mass conserved ), the support unchanged;
//   * a constant density: nothing to do, it is already positive everywhere.
// =====================================================================================

#include "Convolved.h"

namespace sdot {
namespace sdotplan {

/// THE STAGES: `s0, s0 / r, ...` as long as `s >= s_min`, then `0`
inline std::vector<double> continuation_steps( double s0, double ratio, double s_min ) {
    std::vector<double> res;
    if ( s0 > 0 && ratio > 1 )
        for ( double s = s0; s >= s_min * ( 1 - 1e-12 ) && s > 0; s /= ratio )
            res.push_back( s );
    res.push_back( 0 );
    return res;
}

} // namespace sdotplan
} // namespace sdot
