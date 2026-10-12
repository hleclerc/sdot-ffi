#pragma once

// the members + the axes this body names, written to the build include tree by `CallArg_Aggregate`.
#include <sdot/generated/aggregates/Polytope.h>
#include <loom/support/common_macros.h>
#include <loom/support/containers/Vector.h>
#include <loom/support/atomic_add.h>
#include "ConstantDensity.h"

namespace sdot {

// A CONSTANT density on a convex polytope `{ x : directions( c ) . x <= offsets( c ) for every c }`, zero elsewhere.
//
// It is the simplest density that has a support: one piece per cell, the cell cut by the planes it
// crosses. The planes carry `PIECE` ( see `PieceWorkspace.h` ): they face no seed, so the adjoint
// sends their share nowhere, and the only parameter that receives a gradient is `density( 0 )`,
// through the volume of the piece ( `d mass / d density` ).
SDOT_TEMPLATE_DECL_FOR_Polytope
struct Polytope {
    SDOT_ATTRIBUTES_OF_Polytope

    static constexpr int ct_dim = DECAYED_TYPE_OF( nb_dims )::value;
    using TF = DECAYED_TYPE_OF( directions )::TF;

    /// the cell is cut by the planes of the polytope: the integrator reserves a spare cell for it
    static constexpr bool cuts_pieces = true;

    HD void for_each_piece( const auto &cell, auto &&ws, auto &&func ) const {
        const SI nv = SI( cell.nb_vertices() );
        if ( nv == 0 )
            return;

        // A plane that every vertex already satisfies cuts nothing: it is not made, so a cell inside
        // the polytope ( the common case ) is its own piece, without a copy. An UNBOUNDED cell has
        // stop-gap vertices that bound nothing: every plane is then made.
        const bool bounded = cell.bounded();
        const SI nb = SI( directions.shape( 0 ) );
        bool started = false;
        for ( SI c = 0; c < nb; ++c ) {
            const auto n = Vector<TF,ct_dim>::with_func( [&]( PI k ) { return TF( directions( c, k ) ); } );
            const TF off = TF( offsets( c ) );
            if ( bounded ) {
                bool crosses = false;
                for ( SI v = 0; v < nv && ! crosses; ++v ) {
                    TF s = - off;
                    for ( PI k = 0; k < PI( ct_dim ); ++k )
                        s += n[ k ] * TF( cell.coord( int( v ), int( k ) ) );
                    crosses = s > 0;
                }
                if ( ! crosses )
                    continue;
            }
            if ( ! ( started ? ws.cut( n, off ) : ws.start( cell, n, off ) ) )
                return;                                  // no room: recorded by `ws`, the host relaunches with more
            started = true;
            if ( ws.nb_vertices() == 0 )
                return;                                  // the polytope does not meet the cell
        }

        auto dens = ConstantDensity{ TF( density( 0 ) ), [&]( auto &&grad_dist, TF g ) {
            // `d mass / d density` is the volume of the piece; atomic, several work-items share it
            if constexpr ( DECAYED_TYPE_OF( grad_dist.density )::is_valid )
                atomic_add( grad_dist.density( 0 ).ref(), g );
        } };
        if ( started ) ws.with_current( [&]( const auto &piece ) { func( piece, dens ); } );
        else           func( cell, dens );
    }
};

} // namespace sdot
