#pragma once

// the axes this body names, as declared symbols (autocompletion, standalone compile) instead of
// globals the generated source happens to define around us. Written to the build include tree.
#include <sdot/generated/aggregates/Cell_1.h>
#include <loom/support/common_macros.h>
#include "cell/Local1.h"
#include "cell/Ops.h"

namespace sdot {

// THE SEGMENT ( `Cell_1.py` ).
//
// The tensors are those of `Cell_1.py` -- the CONVENIENT format, one vertex per row, in the
// caller's float. A kernel never works on them directly: it puts the cell in its local
// form ( `Local`, on the item's scratch, in the kernel's float ), works, and puts it back. The
// operations themselves are in `cell/Ops.h`, shared by the three regimes.
SDOT_TEMPLATE_DECL_FOR_Cell_1
struct Cell_1 {
    SDOT_ATTRIBUTES_OF_Cell_1

    static constexpr int ct_dim = 1;
    using TF = DECAYED_TYPE_OF( vertex_positions )::TF;
    template<class TK> using Local = Local1<TK>;

    HD void init_as_hypercube( auto &&scratch, auto &&origin, auto &&axes, SI cut_id ) {
        cell_ops::init_as_hypercube<Local<KernelType<DECAYED_TYPE_OF( scratch )>>>( *this, scratch, origin, axes, cut_id );
    }
    HD void init_as_unbounded( auto &&scratch ) {
        cell_ops::init_as_unbounded<Local<KernelType<DECAYED_TYPE_OF( scratch )>>>( *this, scratch );
    }
    HD void cut( auto &&res, auto &&scratch, auto &&direction, auto &&offset, SI cut_id ) const {
        cell_ops::cut<Local<KernelType<DECAYED_TYPE_OF( scratch )>>>( *this, res, scratch, direction, offset, cut_id );
    }
    HD void measure( auto &&res, auto &&scratch ) const {
        cell_ops::measure<Local<KernelType<DECAYED_TYPE_OF( scratch )>>>( *this, res, scratch );
    }
    HD void measure_bwd( auto &&res, auto &&grad_res, auto &&grad_vertex_positions, auto &&scratch ) const {
        cell_ops::measure_bwd<Local<KernelType<DECAYED_TYPE_OF( scratch )>>>( *this, res, grad_res, grad_vertex_positions, scratch );
    }
};

}
