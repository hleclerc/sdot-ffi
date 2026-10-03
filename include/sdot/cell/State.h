#pragma once

// WHAT THE PROVIDER SEES OF THE CELL, and what it can keep there -- the part of the engine's
// contract ( see `Engine.h` ) that depends on no machine, split off so that in-memory cells
// can name it without pulling in the engine.

namespace sdot {

/// for a provider that has nothing to keep from one cut to the next
struct NothingLocal {};

template<class F> struct local_of { using type = NothingLocal; };
template<class F> requires requires { typename F::Local; }
struct local_of<F> { using type = typename F::Local; };
template<class F> using LocalOf = typename local_of<F>::type;

/// WHAT THE PROVIDER SEES when the cell is in memory: `nb` vertices, in arrays.
template<class TK>
struct StateMem {
    int       nb;
    const TK *vx, *vy;
    const int *cid;
    bool      bounded;   ///< false: the vertices are those of a stand-in simplex, deduce nothing from them
};

/// for cells of dimension > 2: `nb` vertices, `D` coordinate arrays
template<class TK,int D>
struct StateMemN {
    int       nb;
    const TK *v[ D ];
    bool      bounded;
};

} // namespace sdot
