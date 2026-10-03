#pragma once

#include <loom/support/common_macros.h>
#include "cell/Ids.h"

namespace sdot {

// What it takes to SPLIT a cell into pieces: ONE spare cell, in the local form, laid on the
// work-item's scratch by the caller ( `PowerDiagram::measures` ). It is the only "scratch"
// that a distribution's contract provides for ( see `distributions/Distribution.py` ).
//
// The SOURCE cell is never touched: `start` copies it into `piece` and cuts, and the following
// cuts cut `piece` in place. That is what allows opening one piece after the other from the
// same cell.
//
// The splitting planes carry `cell_ids::PIECE`: "not a seed", and that is exactly what the
// adjoint reads to know that their share goes nowhere ( `PowerDiagram::scatter_cell_grad` ).
template<class Local>
struct PieceWorkspace {
    using TK = typename Local::TKernel;
    static constexpr int D = Local::ct_dim;

    Local &piece;
    bool  overflow = false;   ///< a cut did not fit: the caller reports it, the result will be discarded

    /// opens a piece: `src` cut by `direction . x <= offset`. Returns `false` if it did not fit.
    HD bool start( const Local &src, const auto &direction, auto offset ) {
        if ( ! piece.copy_from( src ) ) {
            overflow = true;
            return false;
        }
        return cut( direction, offset );
    }

    /// one more cut on the current piece
    HD bool cut( const auto &direction, auto offset ) {
        typename Local::PlaneT p;
        for ( int d = 0; d < D; ++d )
            p.dir[ d ] = TK( direction[ d ] );
        p.off = TK( offset );
        p.id  = cell_ids::PIECE;
        if ( piece.cut( p ) == CutStatus::NO_ROOM ) {
            overflow = true;
            return false;
        }
        return true;
    }

    /// the current piece
    HD void with_current( auto &&func ) const { func( piece ); }

    /// 0 = the piece is empty ( the box does not meet the cell ) -- not an anomaly
    HD SI nb_vertices() const { return piece.nb_vertices(); }
};

} // namespace sdot
