#pragma once

// the axes this body names, as declared symbols (autocompletion, standalone compile) instead of
// globals the generated source happens to define around us. Written to the build include tree.
#include <sdot/generated/aggregates/PowerDiagram_Bsp.h>
#include <loom/support/common_macros.h>
#include "diagram/Common.h"
#include "AaBsp.h"
#include "engine/Seeds.h"

namespace sdot {

/// the sorted seeds, as `bsp_build_level.h` reads them ( `nb_dims`, `positions`, `weights` )
template<class T_nb_dims,class T_positions,class T_weights>
struct BspCloudView {
    T_nb_dims   nb_dims;
    T_positions positions;
    T_weights   weights;
};

// THE SEEDS IN THE ORDER OF THE TREE ( `PowerDiagram_Bsp.py` ): `sorted_positions` / `sorted_weights`
// are the seeds permuted as `tree.seed_indices` lays them out, so that a leaf is read
// in one piece. The provider is the tree turned inside out ( `cell/Providers.h::ProviderBsp` ),
// and everything here -- ranks, measures, gradients, cut identifiers -- is in that order; it is
// `PowerDiagram_Bsp.py` that translates to the user's order.
SDOT_TEMPLATE_DECL_FOR_PowerDiagram_Bsp
struct PowerDiagram_Bsp {
    SDOT_ATTRIBUTES_OF_PowerDiagram_Bsp
    SDOT_DIAGRAM_COMMON( sorted_positions, sorted_weights )

    HD SI user_id( SI k ) const { return SI( tree.seed_indices( k ) ); }

    /// the memory ( `memo_nbrs / memo_counts`, see `ProviderBsp` ): named or `Unbound`, at compile time
    static constexpr bool has_memo = DECAYED_TYPE_OF( memo_counts )::is_valid;

    template<class TK>
    HD auto provider( SI k0 ) const { return ProviderBsp<PowerDiagram_Bsp,TK,ct_dim,has_weights,has_memo>( *this, k0 ); }

    /// the same seeds and tree without tensor types, for the cell engine compiled as a unit ( `engine/Cells.h` ).
    /// The unit is built for the element types of the tree: floats like the positions, ranks of the integer type of `node_begin`.
    HD auto engine_seeds() const {
        // the ranks are read as the integer of their WIDTH ( `long` and `long long` are both 64 bits, and which one the call's
        // tensors spell depends on the platform: the unit is built for `SI` or `int`, whatever the spelling )
        using TI = std::conditional_t<( sizeof( typename DECAYED_TYPE_OF( tree.node_begin )::TF ) == 8 ), SI, int>;
        static_assert( std::is_same_v<std::remove_cv_t<typename DECAYED_TYPE_OF( tree.node_box )::TF>,TF>, "the tree's boxes are of the float of the positions" );
        using S = engine::BspSeeds<TF,TI,ct_dim,has_weights,has_memo>;
        return S{ engine::view2<TF>( sorted_positions ), engine::opt_view1<TF>( sorted_weights ),
                  typename S::Tree{ engine::view1<TI>( tree.node_begin ), engine::view1<TI>( tree.node_end ), engine::view3<TF>( tree.node_box ),
                                    engine::opt_view2<TF>( tree.node_wa ), engine::opt_view1<TF>( tree.node_wb ) },
                  engine::opt_view2<int>( memo_nbrs ), engine::opt_view1<int>( memo_counts ) };
    }

    /// THE SAME DIAGRAM, the weights and the tree's majorants read ELSEWHERE -- views that
    /// the caller owns and WRITES ( the `SdotPlanNd` solver, which sets weights at each trial ):
    /// the inputs of a call are read-only, its outputs are not. The order of the members is
    /// that of `PowerDiagram_Bsp.py` / `AaBsp.py` ( the same as `kernel_form` above ).
    HD auto with_weights( auto &&sorted_weights_, auto &&node_wa_, auto &&node_wb_ ) const {
        auto tree_ = ::sdot::AaBsp{ tree.seed_indices, tree.node_left, tree.node_right, tree.node_begin, tree.node_end,
                                    tree.node_box, node_wa_, node_wb_, tree.nb_bsp_seeds, tree.nb_bsp_nodes, tree.nb_lohi, tree.nb_dims };
        return ::sdot::PowerDiagram_Bsp{ box_min, box_max, bnd_directions, bnd_offsets, start_vertices, start_topo, nb_starts, nb_topo, nb_points, nb_boundaries, nb_dims,
                                         tree_, sorted_positions, sorted_weights_, memo_nbrs, memo_counts, nb_memo };
    }
    /// ... and the memory also read elsewhere ( the solver writes it at each sweep )
    HD auto with_weights( auto &&sorted_weights_, auto &&node_wa_, auto &&node_wb_, auto &&memo_nbrs_, auto &&memo_counts_ ) const {
        auto tree_ = ::sdot::AaBsp{ tree.seed_indices, tree.node_left, tree.node_right, tree.node_begin, tree.node_end,
                                    tree.node_box, node_wa_, node_wb_, tree.nb_bsp_seeds, tree.nb_bsp_nodes, tree.nb_lohi, tree.nb_dims };
        return ::sdot::PowerDiagram_Bsp{ box_min, box_max, bnd_directions, bnd_offsets, start_vertices, start_topo, nb_starts, nb_topo, nb_points, nb_boundaries, nb_dims,
                                         tree_, sorted_positions, sorted_weights_, memo_nbrs_, memo_counts_, nb_memo };
    }

    /// what `bsp_refresh_majorant` reads: the seeds in the tree's order
    HD auto sorted_cloud() const { return BspCloudView{ nb_dims, sorted_positions, sorted_weights }; }
};


}
