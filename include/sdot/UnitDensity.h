#pragma once

#include <loom/support/common_macros.h>
#include "ConstantDensity.h"

namespace sdot {

// The distribution that is not one: the Lebesgue measure, density 1 everywhere.
//
// What `PowerDiagram::integrate_into` receives when the caller gave no distribution, so that
// "no distribution" is an ORDINARY case of the same code and not a second implementation --
// exactly the role `EverySeed` plays for the accelerators.
//
// A single piece, the cell itself, and no slicing: no scratch, no copy, not the slightest cut.
// `measures` without a distribution therefore computes exactly what it computed before, down to
// the instruction (the `TF( 1 ) *` folds away at compile time).
struct UnitDensity {
    HD void for_each_piece( const auto &cell, auto &&/*ws*/, auto &&func ) const {
        // constant, and not parameterized: the gradient sink leads nowhere (see
        // `ConstantDensity`). The integrator's `TF( 1 ) *` folds away at compile time.
        using TF = typename DECAYED_TYPE_OF( cell )::TKernel;
        func( cell, ConstantDensity{ TF( 1 ), []( auto &&/*grad_dist*/, auto /*g*/ ) {} } );
    }
};

} // namespace sdot
