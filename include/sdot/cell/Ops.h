#pragma once

// =====================================================================================
// WHAT THE THREE `Cell_*.h` HAVE IN COMMON: laying a local cell on the scratch of an item, and
// the operations of `Cell_*.py` ( init, cut, measure ) written once over `Local1 / 2 / N`.
//
// THE KERNEL FLOAT is the one the scratch declares ( `kernel_fp_size`, a compile-time
// constant ): `float` by default, `double` by choice. The STORED cell is in the caller's float
// ( `TF`, that of `vertex_positions` ); `load` / `store` convert.
//
// THE CAPACITY is deduced from the scratch: `cap_for` is the largest number of vertices whose
// arrays fit in the words received ( `Local::words_for`, the formula `Cell_*.py` used
// for sizing ). A scratch that is too small is reported on `nb_words` -- which loom
// knows how to grow -- and the cell writes nothing.
// =====================================================================================

#include <loom/support/common_macros.h>
#include "Scratch.h"

#include <type_traits>

namespace sdot {

/// the kernel float, read from the scratch
template<class Scr>
using KernelType = std::conditional_t<DECAYED_TYPE_OF( std::declval<Scr>().kernel_fp_size )::value == 64, double, float>;

/// the largest `cap` such that `words( cap ) <= nb_words` ( `words` increasing )
HD SI cap_for_words( SI nb_words, auto &&words ) {
    SI lo = 0, hi = 1;
    while ( words( hi ) <= nb_words )
        hi *= 2;
    while ( hi - lo > 1 ) {
        const SI mid = ( lo + hi ) / 2;
        if ( words( mid ) <= nb_words ) lo = mid;
        else hi = mid;
    }
    return lo;
}

/// the largest `cap` such that `Local::words_for( cap ) <= nb_words`
template<class Local>
HD SI cap_for( SI nb_words ) {
    return cap_for_words( nb_words, []( SI c ) { return Local::words_for( c ); } );
}

/// a local cell laid on the scratch of an item ( `Local` attached, `cap` deduced )
template<class Local,class Scr>
HD Local local_on( Scr &sc, Carver &cv ) {
    Local c;
    c.attach( cv, cap_for<Local>( cv.nb_words ) );
    return c;
}

/// row `row` of the scratch ( `words` is `[ nb_threads, nb_words ]`: one row per work-item,
/// or a single one when the call is batched over the cells )
template<class Scr>
HD Carver carver_of( Scr &sc, SI row = 0 ) {
    auto w = sc.words( row );
    return Carver{ w.data().raw, SI( w.shape( 0 ) ) };
}

/// the scratch was not enough: we say so ( loom doubles it and retries ), writing nothing
template<class Scr>
HD void ask_more( Scr &sc, const Carver &cv ) {
    sc.nb_words.set( 2 * cv.nb_words + 64 );
}

// ---- the operations of `Cell_*.py`, once and for all -------------------------------------

namespace cell_ops {

template<class Local>
HD void init_as_hypercube( auto &&cell, auto &&scratch, auto &&origin, auto &&axes, SI cut_id ) {
    Carver cv = carver_of( scratch );
    Local c = local_on<Local>( scratch, cv );
    if ( ! c.init_hypercube( origin, axes, int( cut_id ) ) ) { ask_more( scratch, cv ); return; }
    c.tidy();
    c.store( cell );
}

template<class Local>
HD void init_as_unbounded( auto &&cell, auto &&scratch ) {
    Carver cv = carver_of( scratch );
    Local c = local_on<Local>( scratch, cv );
    if ( ! c.init_unbounded() ) { ask_more( scratch, cv ); return; }
    c.store( cell );
}

/// intersects with `direction . x <= offset`, the result going into `res` ( the inputs and
/// the outputs of a call are disjoint ). An overflow is reported on `res.nb_vertices`.
template<class Local>
HD void cut( const auto &cell, auto &&res, auto &&scratch, auto &&direction, auto &&offset, SI cut_id ) {
    using TK = typename Local::TKernel;
    constexpr int D = Local::ct_dim;
    Carver cv = carver_of( scratch );
    Local c = local_on<Local>( scratch, cv );
    if ( ! c.load( cell ) ) { ask_more( scratch, cv ); return; }
    typename Local::PlaneT p;
    for ( int d = 0; d < D; ++d )
        p.dir[ d ] = TK( direction( d ) );
    p.off = TK( offset );
    p.id  = int( cut_id );
    if ( c.cut( p ) == CutStatus::NO_ROOM ) { ask_more( scratch, cv ); return; }
    c.tidy();
    c.store( res );
}

template<class Local>
HD void measure( const auto &cell, auto &&res, auto &&scratch ) {
    using TF = DECAYED_TYPE_OF( res )::TF;
    Carver cv = carver_of( scratch );
    Local c = local_on<Local>( scratch, cv );
    if ( ! c.load( cell ) ) { ask_more( scratch, cv ); return; }
    res = c.template measure<TF>();
}

template<class Local>
HD void measure_bwd( const auto &cell, auto &&res, auto &&grad_res, auto &&grad_vertex_positions, auto &&scratch ) {
    if constexpr ( ! grad_vertex_positions.surely_null ) {
        using TF = DECAYED_TYPE_OF( grad_res )::TF;
        constexpr int D = Local::ct_dim;
        Carver cv = carver_of( scratch );
        Local c = local_on<Local>( scratch, cv );
        if ( ! c.load( cell ) ) { ask_more( scratch, cv ); return; }
        // the cotangent has a value wherever its primal has one: the whole buffer, padding included
        const SI capv = SI( grad_vertex_positions.shape( 0 ) );
        for ( SI i = 0; i < capv; ++i )
            for ( int d = 0; d < D; ++d )
                grad_vertex_positions( i, d ) = 0;
        c.template measure_bwd<TF>( TF( grad_res ), [&]( int i, int d ) -> auto & { return grad_vertex_positions( i, d ).ref(); } );
    }
}

} // namespace cell_ops
} // namespace sdot
