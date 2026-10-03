#pragma once

#include <loom/support/common_macros.h> // HD

#include <climits>

namespace sdot {

// =====================================================================================
// THE IDENTITY OF A CUT, as stored in `cut_ids`.
//
// This is the only thing the geometry keeps of a plane once the cut is made: the plane itself
// is rebuilt when needed -- from the seeds for a bisector, from the vertices that carry it
// for everything else (see `Local2::planes_from_vertices`).
//
//   id >= 0              the bisector with seed `id` (a `cut_id` of `PowerDiagram`)
//   -1 - k               the k-th plane of the DOMAIN; `BOUNDARY == domain_id( 0 )` is what is carried
//                        by a cut that faces no seed when there is nothing more precise
//                        to say -- a `Cell.cut` made from Python, a wall of a box
//   PIECE                a SPLITTING plane added by a distribution (`Image::for_each_piece`)
//   INFINITE             a wall of the replacement simplex of an UNBOUNDED cell: it
//                        does not exist, its offsets are made up and pushed back as cuts come in
//                        (see `Local2::grow_for`), and it disappears the day no vertex
//                        carries it any more
//
// Everything `< 0` is "not a seed": the adjoint of the measure sends nothing there.
// =====================================================================================
namespace cell_ids {
    enum : int {
        INFINITE = INT_MIN,
        PIECE    = INT_MIN + 1,
        BOUNDARY = -1,
    };

    HD constexpr int  domain_id ( int k  ) { return -1 - k; }
    HD constexpr bool is_seed   ( int id ) { return id >= 0; }
    HD constexpr bool is_domain ( int id ) { return id < 0 && id > PIECE; }
    HD constexpr int  domain_num( int id ) { return -1 - id; }
}

/// what a cut did to the cell
namespace CutStatus {
    enum : int {
        UNCHANGED = 0,   ///< the half-space already contained the whole cell: nothing moved
        CUT       = 1,   ///< the cell was cut, in place
        EMPTY     = 2,   ///< the half-space removed everything: `nb == 0`
        NO_ROOM  = 3,   ///< the output does not fit in the capacity; the cell was left INTACT
    };
}

} // namespace sdot
