#pragma once

// the axes this body names, as declared symbols (autocompletion, standalone compile) instead of
// globals the generated source happens to define around us. Written to the build include tree.
#include <sdot/generated/aggregates/PowerDiagram_Plain.h>
#include <loom/support/common_macros.h>
#include "diagram/Common.h"

namespace sdot {

// THE SEEDS AS THEY CAME ( `PowerDiagram_Plain.py` ): `positions [ n, d ]`, `weights
// [ n ]`, and no acceleration -- each cell is cut by the `n - 1` bisectors, in
// order. The floor against which `PowerDiagram_Bsp` is measured, and what remains when the
// positions are a tracer ( no tree to build on them ).
SDOT_TEMPLATE_DECL_FOR_PowerDiagram_Plain
struct PowerDiagram_Plain {
    SDOT_ATTRIBUTES_OF_PowerDiagram_Plain
    SDOT_DIAGRAM_COMMON( positions, weights )

    HD SI user_id( SI k ) const { return k; }

    template<class TK>
    HD auto provider( SI k0 ) const { return ProviderAll<PowerDiagram_Plain,TK,ct_dim>( *this, k0 ); }

    /// the same diagram, the weights read ELSEWHERE ( see `PowerDiagram_Bsp::with_weights` ); the order
    /// of the members is that of `PowerDiagram.py` + `PowerDiagram_Plain.py`
    HD auto with_weights( auto &&weights_ ) const {
        return ::sdot::PowerDiagram_Plain{ box_min, box_max, bnd_directions, bnd_offsets, start_vertices, start_topo, nb_starts, nb_topo, nb_points, nb_boundaries, nb_dims, positions, weights_ };
    }
};

}
