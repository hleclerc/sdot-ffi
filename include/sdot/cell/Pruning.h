#pragma once

#include <loom/support/common_macros.h> // HD

// =====================================================================================
// THE PRUNING TEST, ONCE AND FOR ALL.
//
//     can a seed `q` of the box `B`, with weight bounded by `w( q ) <= a . q + b`, still
//     remove anything from the cell of `( p0, w0 )` ?
//
// If `q` cuts the cell, it removes at least one VERTEX from it, and this vertex satisfies
// `|v - q|^2 - w( q ) < |v - p0|^2 - w0`. We therefore reject `B` as soon as ALL the vertices have
//
//     min_{ q in B } ( |v - q|^2 - a . q - b )  >=  |v - p0|^2 - w0
//
// and this is EXACT in the sense that the minimum is computed, not bounded : `|v - q|^2 - a . q` is separable
// per axis, its free minimum is at `q = v + a / 2`, and a per-axis `clamp` gives it. The constant
// majorant is the case `a = 0` and costs no less. This is what separates this test from a
// first version that compared the cell's BOX to the node's : 280 candidates per
// cell where 26 suffice, a box bounding a convex polygon very poorly.
//
// `<= 0` and not `< 0` : a plane that passes exactly through a vertex removes nothing, so admitting it
// costs a useless cut where rejecting it on a rounding would lose a TRUE cut.
//
// THE SIMD IS HERE AND NOT IN THE KERNEL : the vertices belong to it and arrive in registers
// in 2D ( `StateReg` ), the provider reads them as they are -- two `max`, two `fma`, one
// comparison, whatever their number. In memory ( `StateMem*` ) it is a loop.
// =====================================================================================

#include <asimd/asimd.h>
#include "State.h"

namespace sdot {

/// a box of seeds and the affine majorant of their weights, in the kernel float
template<class TK,int D>
struct Box {
    TK lo[ D ], hi[ D ];
    TK a[ D ], b;                                        ///< `w( q ) <= a . q + b`
};

/// the test, for a state in MEMORY ( `nb` vertices, `D` arrays )
template<bool WEIGHTED,class TK,int D>
HD inline bool can_cut_box( int nb, const TK *const *v, const TK *p0, TK w0, const Box<TK,D> &B ) {
    const TK cb = WEIGHTED ? w0 - B.b : TK( 0 );
    for ( int i = 0; i < nb; ++i ) {
        TK s = cb;
        for ( int d = 0; d < D; ++d ) {
            const TK x = v[ d ][ i ];
            TK y = x + ( WEIGHTED ? B.a[ d ] / 2 : TK( 0 ) );
            y = y < B.lo[ d ] ? B.lo[ d ] : ( y > B.hi[ d ] ? B.hi[ d ] : y );
            const TK u = y - x, f = x - p0[ d ];
            s += u * u - f * f;
            if constexpr ( WEIGHTED ) s -= B.a[ d ] * y;
        }
        if ( s <= 0 )
            return true;
    }
    return false;
}

/// the same test, on the eight lanes of an `StateReg` ( 2D, registers )
template<bool WEIGHTED,class TK,class State>
inline bool can_cut_box_reg( const State &e, const TK *p0, TK w0, const Box<TK,2> &B ) {
    using V = asimd::SimdVec<TK,8>;
    // the point of the box closest to the vertex, SHIFTED by half a slope
    V y0 = e.vx, y1 = e.vy;
    if constexpr ( WEIGHTED ) {
        y0 = y0 + V( B.a[ 0 ] / 2 );
        y1 = y1 + V( B.a[ 1 ] / 2 );
    }
    y0 = asimd::min( asimd::max( y0, V( B.lo[ 0 ] ) ), V( B.hi[ 0 ] ) );
    y1 = asimd::min( asimd::max( y1, V( B.lo[ 1 ] ) ), V( B.hi[ 1 ] ) );

    const V g0 = y0 - e.vx, f0 = e.vx - V( p0[ 0 ] );
    const V g1 = y1 - e.vy, f1 = e.vy - V( p0[ 1 ] );
    V s = asimd::fma( g0, g0, g1 * g1 ) - asimd::fma( f0, f0, f1 * f1 );
    if constexpr ( WEIGHTED )
        s = s + V( w0 - B.b ) - asimd::fma( V( B.a[ 0 ] ), y0, V( B.a[ 1 ] ) * y1 );
    const unsigned m = unsigned( asimd::to_bits( asimd::ge( V( TK( 0 ) ), s ) ) );
    return ( m & ( ( 1u << State::nb ) - 1 ) ) != 0;
}

/// THE SINGLE GATE : whatever the state, the same question.
template<bool WEIGHTED,class TK,int D,class State>
HD inline bool can_cut( const State &e, const TK *p0, TK w0, const Box<TK,D> &B ) {
    // an UNBOUNDED cell is a replacement simplex whose corners are made up : nothing
    // to prune against, and the honest answer is "maybe". The accelerator then degenerates into
    // a full sweep, which is the right answer and not a slow path that someone chose.
    if constexpr ( requires { e.bounded; } )
        if ( ! e.bounded )
            return true;
    if constexpr ( requires { e.vx + e.vx; } ) {
        static_assert( D == 2 );
        return can_cut_box_reg<WEIGHTED,TK>( e, p0, w0, B );
    } else if constexpr ( D == 2 ) {
        const TK *v[ 2 ] = { e.vx, e.vy };
        return can_cut_box<WEIGHTED,TK,2>( e.nb, v, p0, w0, B );
    } else {
        return can_cut_box<WEIGHTED,TK,D>( e.nb, e.v, p0, w0, B );
    }
}

} // namespace sdot
