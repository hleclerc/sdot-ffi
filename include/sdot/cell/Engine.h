#pragma once

#include <loom/support/common_macros.h> // HD

// =====================================================================================
// THE ENGINE: the cell leads, the provider answers.
//
// The kernel receives neither a list, nor positions, nor a seed: it receives a PROVIDER and asks it,
// cut after cut, for "the next half-space". The provider sees the state of the cell -- its
// vertices, its `cid`s -- and returns a `Plane`, or `false` when it has nothing left. It therefore decides
// AFTER each cut, looking at the shrinking cell, where a list built in advance
// had to decide before it existed. That is where pruning and the order of the candidates live;
// here there is no policy.
//
//      template<class State> bool next( const State &e, Local &l, Plane<TK,D> &p );
//
// `Local` is its own: if it declares a `Local` type, the engine creates one per CELL in its
// own frame and passes it back at each call -- the stack of a tree descent, for example.
// A stateless provider declares none and pays nothing.
//
// `State` is a template, not a type: the same provider serves the register path ( 2D, CPU:
// `e.nb` is a compile-time constant and `e.vx` a SIMD vector ) and the memory paths
// ( `e.nb` is an integer, `e.vx` a pointer ). Only one policy to write.
//
// = TWO PATHS
//
//   `run_memory`      the bare loop: ask, cut in place, start again. Any dimension,
//                     any device; it is the GPU path, and that of unbounded cells.
//   `run_2d_registers` the register kernel ( `Engine2Reg.h` ): 2D, CPU, BOUNDED cell. Eight
//                     vertices in three vectors, a state machine on their number, and an
//                     EXCURSION into memory when the cell overflows the eight lanes -- from which it
//                     returns as soon as it goes back down.
//
// `run` chooses, at compile time on the dimension and the device, at run time on boundedness.
// =====================================================================================

#include "Plane.h"
#include "State.h"
#include "Ids.h"

namespace sdot {

/// THE BARE LOOP. Returns `0` when the cell is complete ( empty included ), `CutStatus::NO_ROOM`
/// if a cut did not fit -- the cell then stayed at the last valid state, and the caller
/// must report it rather than measure it.
template<class Cell,class Provider>
HD int run_memory( Cell &c, Provider &f, LocalOf<Provider> &loc ) {
    typename Cell::PlaneT p;
    for ( ;; ) {
        if ( c.nb_vertices() == 0 )
            return 0;
        if ( ! f.next( c.state(), loc, p ) )
            return 0;
        const int r = c.cut( p );
        if ( r == CutStatus::EMPTY )
            return 0;
        if ( r == CutStatus::NO_ROOM )
            return CutStatus::NO_ROOM;
    }
}

/// the same loop, which STOPS as soon as the cell is bounded -- the baton is then passed to the
/// register kernel ( see `run` ). Same return convention as `run_memory`.
template<class Cell,class Provider>
HD int run_memory_while_unbounded( Cell &c, Provider &f, LocalOf<Provider> &loc ) {
    typename Cell::PlaneT p;
    while ( ! c.bounded() ) {
        if ( c.nb_vertices() == 0 )
            return 0;
        if ( ! f.next( c.state(), loc, p ) )
            return 0;
        const int r = c.cut( p );
        if ( r == CutStatus::EMPTY )
            return 0;
        if ( r == CutStatus::NO_ROOM )
            return CutStatus::NO_ROOM;
    }
    return 0;
}

} // namespace sdot

#include "Engine2Reg.h"

namespace sdot {

/// THE ENTRY POINT. `ON_CPU` says whether the register kernel is available ( it is written in
/// asimd, which a GPU kernel cannot compile into registers ).
template<bool ON_CPU,class Cell,class Provider>
HD int run( Cell &c, Provider &f ) {
    LocalOf<Provider> loc{};
    if constexpr ( ON_CPU && Cell::ct_dim == 2 ) {
        if ( c.bounded() )
            return run_2d_registers( c, f, loc );
        // unbounded: in memory while walls remain to be pushed back, then in registers
        const int r = run_memory_while_unbounded( c, f, loc );
        if ( r || c.nb_vertices() == 0 || c.unbounded )
            return r;
        return run_2d_registers( c, f, loc );
    } else {
        return run_memory( c, f, loc );
    }
}

} // namespace sdot
