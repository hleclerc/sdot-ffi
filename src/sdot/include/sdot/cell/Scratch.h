#pragma once

#include <loom/support/common_macros.h> // HD

// =====================================================================================
// THE SCRATCH : a single tensor of words per work-item, carved up by the C++ into what it needs.
//
// An in-memory cell ( `Local1` / `Local2` / `LocalN` ) is made of arrays -- the vertices,
// the identifiers, the cut temporaries -- whose size is ONE CAPACITY, decided by the
// host : exactly what is needed for an operation of `Cell_*.py` ( the number of vertices that a
// cut can produce is bounded ), a guess that loom grows on overflow for a whole
// diagram. These arrays all live in ONE integer tensor per work-item, which `Carver`
// carves up : it is the only scratch a kernel asks for, and it is what carries the batch axis of
// the call ( `Cell.py::CellScratch` ).
//
// THE TWO SIDES MUST AGREE on the size : `Local*::words_for( cap )` here,
// `Cell_*.scratch_words( cap )` on the Python side, with the same formula -- and `attach` checks that it has the
// room, otherwise it sets nothing and says so ( `ShapeVarView::set` ), rather than writing next to it.
// =====================================================================================

#include <loom/support/common_types.h>
#include <cstdint>
#include <cstddef>

namespace sdot {

/// the alignment of each carved array : enough to load eight `float`s at once
static constexpr SI scratch_align = 32;

/// `n` elements of `T` rounded up to the alignment, in 32-bit WORDS
template<class T>
HD constexpr SI words_of( SI n ) {
    const SI bytes = n * SI( sizeof( T ) );
    return ( ( bytes + scratch_align - 1 ) / scratch_align ) * ( scratch_align / 4 );
}

/// carves a zone of `nb_words` words into aligned arrays, in the order of the `take` calls
struct Carver {
    std::int32_t *base;
    SI            nb_words;
    SI            used = 0;
    bool          overflow = false;

    template<class T>
    HD T *take( SI n ) {
        T *res = reinterpret_cast<T *>( base + used );
        used += words_of<T>( n );
        if ( used > nb_words )
            overflow = true;
        return res;
    }
};

} // namespace sdot
