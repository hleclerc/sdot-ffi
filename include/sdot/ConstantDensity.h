#pragma once

#include <loom/support/common_macros.h>

namespace sdot {

// The density of a PIECE on which it is constant -- what the slicing of an image yields,
// and what `UnitDensity` yields for the whole cell.
//
// This is the case where integration is exact and free: `value * measure of the piece`, with no
// quadrature or triangulation (see `PowerDiagram::integrate_into`, which branches on it AT
// COMPILE TIME on `is_constant`).
//
// `sink` is what ties the value to the distribution's PARAMETERS: the integrator knows that
// `d mass / d value` is the volume of the piece, but not where that value is stored -- a slot of
// `values` for an image, nothing at all for the unit density. The closure is therefore built
// where the index is known, and the integrator merely calls it.
template<class TF_,class Sink>
struct ConstantDensity {
    using TF = TF_;
    static constexpr bool is_constant = true;

    TF   value;
    Sink sink;

    HD void add_value_grad( auto &&grad_dist, TF g ) const { sink( grad_dist, g ); }
};

} // namespace sdot
