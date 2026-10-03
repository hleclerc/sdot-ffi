#pragma once

// =====================================================================================
// WHAT A POWER DIAGRAM COMPUTES, whatever the way it stores its seeds.
//
// The cell of seed `i` is where its POWER DISTANCE wins: `|x - d_i|^2 - w_i <=
// |x - d_j|^2 - w_j` for every other `j`. Expanded, the inequality loses its `|x|^2` and becomes a
// half-space: a power diagram costs exactly what a Voronoi costs, one plane per
// rival, the same cut ( `cell/Plane.h::bisector` ).
//
// There is NO diagram here: not a vertex, not a facet. Everything is redone, cell by cell,
// in the work-item's scratch ( `Cell.py::CellScratch` ): a local cell, a second one
// for the pieces of a distribution, and the cotangents of the adjoint. THE CELL LEADS
// ( `cell/Engine.h` ): it asks a provider for its planes, and it is the STORAGE of the
// diagram that says which one -- all the seeds in order for `PowerDiagram_Plain`, the flipped BSP tree
// for `PowerDiagram_Bsp`.
//
// = WHAT A STORAGE PROVIDES
//
//     ct_dim, TF, has_weights
//     SI   nb_seeds() const                   the seeds, IN ITS ORDER
//     auto point( SI k ) const                seed `k` ( Vector<TF,ct_dim> ), weight `weight( k )`
//     SI   user_id( SI k ) const              the index of this seed for the USER
//     auto provider<TK>( SI k0 ) const     what yields the planes of the cell of `k0`
//
// Cells are built in STORAGE ORDER ( `k`, what a provider puts in the cut
// identifiers, and the index of the gradients on the seeds ); what goes out to the user
// -- a measure, a kept cell and its `cut_ids` -- is written at index `user_id( k )`.
// =====================================================================================

#include <loom/support/math.h>
#include <loom/support/common_macros.h>
#include <loom/support/containers/Matrix.h>
#include <loom/support/containers/Vector.h>
#include <loom/support/atomic_add.h>
#include "../cell/Providers.h"
#include "../cell/Engine.h"
#include "../cell/Ops.h"
#include "../PieceWorkspace.h"
#include "../UnitDensity.h"

#include <limits>

namespace sdot {
namespace diagram {

/// one cotangent per vertex, in the scratch
template<class TF>
struct GradVp {
    TF *g;
    SI  cap;
    HD TF &operator()( int i, int d ) { return g[ d * cap + i ]; }
    HD TF  operator()( int i, int d ) const { return g[ d * cap + i ]; }
};

/// a distribution that SPLITS asks for a second cell; `UnitDensity` and smooth
/// densities do not
template<class Dist>
HD constexpr int nb_work_cells() {
    if constexpr ( requires { Dist::cuts_pieces; } ) return Dist::cuts_pieces ? 2 : 1;
    else return 1;
}

/// HOW THE SCRATCH IS CARVED UP for a work-item: `nb_cells` local cells of `cap` vertices, and ( for
/// the adjoint ) one cotangent per vertex -- THE SAME FORMULA as `PowerDiagram._scratch_words`.
template<class Local,class TF>
HD SI words_for( SI cap, int nb_cells, bool with_grad ) {
    return nb_cells * Local::words_for( cap ) + ( with_grad ? words_of<TF>( Local::ct_dim * cap ) : 0 );
}

template<class Local,class TF>
HD SI cap_in( SI nb_words, int nb_cells, bool with_grad ) {
    return cap_for_words( nb_words, [&]( SI c ) { return words_for<Local,TF>( c, nb_cells, with_grad ); } );
}

/// The cell of `k0`, built in `c` from the domain `dom`. Returns `false` if the scratch
/// was not enough -- the cell is then left in its last valid state.
template<class PD,class Local>
HD bool make_cell( const PD &pd, Local &c, SI k0, const auto &dom ) {
    using TK = typename Local::TKernel;
    if ( ! c.load( dom ) )
        return false;
    auto f = pd.template provider<TK>( k0 );
    return run<PD::on_cpu>( c, f ) != CutStatus::NO_ROOM;
}

/// the plane of cut `k` of `cell`, in the positions' float: the bisector REBUILT
/// from the seeds when the cut faces a seed, the plane read back from the geometry otherwise
template<class PD>
HD void plane_of( const PD &pd, const auto &cell, SI k0, int k, auto &dir, typename PD::TF &off ) {
    using TF = typename PD::TF;
    constexpr int D = PD::ct_dim;
    const int id = cell.cid[ k ];
    if ( id >= 0 ) {
        const auto p0 = pd.point( k0 ), p1 = pd.point( id );
        off = ( pd.weight( k0 ) - pd.weight( id ) ) / 2;
        for ( int d = 0; d < D; ++d ) {
            dir[ d ] = p1[ d ] - p0[ d ];
            off += dir[ d ] * ( p0[ d ] + p1[ d ] ) / 2;
        }
    } else {
        TF dk[ D ];
        cell.plane( k, dk, off );
        for ( int d = 0; d < D; ++d )
            dir[ d ] = dk[ d ];
    }
}

// ---- WHAT we integrate against -------------------------------------------------------------------

/// the `D + 1` vertices of simplex `chain`, as points -- what a density receives
template<class TF,int D>
HD auto simplex_points( const auto &cell, const auto &chain ) {
    return Vector<Vector<TF,D>,D+1>( Function(), [&]( PI k ) {
        return Vector<TF,D>::with_func( [&]( PI c ) { return TF( cell.coord( int( chain[ k ] ), int( c ) ) ); } );
    } );
}

/// `res` = the integral of `dist` over `cell`. The distribution SPLITS, we INTEGRATE: on a piece of
/// constant density, `value * measure`; otherwise the piece is split into simplices and it is the density
/// that is integrated over each. Returns `false` if a piece did not fit.
template<class TF,class Local>
HD bool integrate_into( auto &&res, const Local &cell, Local &piece, const auto &dist ) {
    constexpr int D = Local::ct_dim;
    PieceWorkspace<Local> ws{ piece };
    TF sum = 0;
    dist.for_each_piece( cell, ws, [&]( const auto &pc, const auto &dens ) {
        if constexpr ( DECAYED_TYPE_OF( dens )::is_constant ) {
            sum += dens.value * pc.template measure<TF>();
        } else {
            // an unbounded cell has no simplices that mean anything
            if ( ! pc.bounded() ) {
                sum = std::numeric_limits<TF>::max();
                return;
            }
            pc.for_each_simplex( [&]( const auto &chain ) {
                sum += dens.integrate_over_simplex( simplex_points<TF,D>( pc, chain ) );
            } );
        }
    } );
    res = sum;
    return ! ws.overflow;
}

/// the volume of simplex `pts`: `| det( p_i - p_0 ) | / D!`
template<class TF,int D>
HD TF simplex_volume( const auto &pts ) {
    const auto M = Matrix<TF,D>::with_func( [&]( auto r, auto c ) { return pts[ int( c ) + 1 ][ int( r ) ] - pts[ 0 ][ int( r ) ]; } );
    TF det = M.determinant();
    if ( det < 0 ) det = - det;
    for ( int i = 2; i <= D; ++i )
        det /= i;
    return det;
}

/// the MOMENTS of `dist` over `cell`: `mass = int rho`, `first = int x rho`, `second = int |x|^2 rho`
/// -- what a transport cost needs ( `sum_i int_{cell_i} |x - p_i|^2 rho` ) and its
/// barycenters. Each piece is split into simplices: with CONSTANT density ( `Image`, Lebesgue ) their
/// moments are closed forms -- `int_T x = |T| g`, `g` the center, and
/// `int_T |x|^2 = |T| ( sum_i |v_i|^2 + |sum_i v_i|^2 ) / ( ( D + 1 )( D + 2 ) )` -- otherwise it is the
/// quadrature of the density that accumulates them ( `PointwiseDensity::integrate_moments_over_simplex` ).
template<class TF,class Local>
HD bool integrate_moments_into( auto &&mass, auto &&first, auto &&second, const Local &cell, Local &piece, const auto &dist ) {
    constexpr int D = Local::ct_dim;
    PieceWorkspace<Local> ws{ piece };
    TF m = 0, m2 = 0;
    auto mx = Vector<TF,D>::zeros();
    dist.for_each_piece( cell, ws, [&]( const auto &pc, const auto &dens ) {
        if ( ! pc.bounded() ) {
            m = std::numeric_limits<TF>::max();
            return;
        }
        if constexpr ( ! DECAYED_TYPE_OF( dens )::is_constant ) {
            pc.for_each_simplex( [&]( const auto &chain ) {
                dens.integrate_moments_over_simplex( simplex_points<TF,D>( pc, chain ), m, mx, m2 );
            } );
        } else {
            const TF rho = TF( dens.value );
            pc.for_each_simplex( [&]( const auto &chain ) {
                const auto pts = simplex_points<TF,D>( pc, chain );
                const TF w = rho * simplex_volume<TF,D>( pts );
                auto sum = Vector<TF,D>::zeros();
                TF sq = 0;
                for ( int k = 0; k <= D; ++k ) {
                    for ( int d = 0; d < D; ++d )
                        sum[ d ] += pts[ k ][ d ];
                    sq += norm_2_p2( pts[ k ] );
                }
                m += w;
                for ( int d = 0; d < D; ++d )
                    mx[ d ] += w * sum[ d ] / ( D + 1 );
                m2 += w * ( sq + norm_2_p2( sum ) ) / ( ( D + 1 ) * ( D + 2 ) );
            } );
        }
    } );
    mass = m;
    for ( int d = 0; d < D; ++d )
        first( d ) = mx[ d ];
    second = m2;
    return ! ws.overflow;
}

/// `res( k )` = the measure of cell `k`, for the seeds of this work-item -- a strided loop:
/// `nb_threads` work-items share the cells, in storage order ( two consecutive
/// seeds are neighbors in space there ). `scratch` is the work-item's working tensor --
/// and where we say it was too small.
/// THE MEMORY of a cell ( `ProviderBsp`, `MEMO` ): its neighbors, as sorted ranks, written into
/// `memo_nbrs( k, . )` / `memo_counts( k )` -- or nothing ( `0`: zero count ) if they exceed the
/// capacity. `memo_*` are `0` when the call does not want them.
template<class Local>
HD void memorize( Local &c, SI k, auto &&memo_nbrs, auto &&memo_counts ) {
    if constexpr ( requires { memo_nbrs( k, 0 ); } ) {
        c.tidy();
        const int nc = c.nb_cuts(), cap = int( memo_nbrs.shape( 1 ) );
        int m = 0;
        for ( int q = 0; q < nc && m <= cap; ++q ) {
            const int id = c.cid[ q ];
            if ( id < 0 ) continue;                      // the domain
            if ( m == cap ) { m = cap + 1; break; }      // too many for the capacity: no memory
            int r = m++;                                 // insertion, sorted ascending
            while ( r > 0 && int( memo_nbrs( k, r - 1 ) ) > id ) { memo_nbrs( k, r ) = memo_nbrs( k, r - 1 ); --r; }
            memo_nbrs( k, r ) = id;
        }
        memo_counts( k ) = m > cap ? 0 : m;
    }
}

template<class PD>
HD void measures( const PD &pd, auto &&res, const auto &dom, auto &&scratch, const auto &dist,
               auto &&memo_nbrs, auto &&memo_counts, SI thread_index, SI nb_threads ) {
    using TF    = typename PD::TF;
    using TK    = KernelType<DECAYED_TYPE_OF( scratch )>;
    using Local = typename DECAYED_TYPE_OF( dom )::template Local<TK>;
    constexpr int nbc = nb_work_cells<DECAYED_TYPE_OF( dist )>();
    Carver cv = carver_of( scratch );
    const SI cap = cap_in<Local,TF>( cv.nb_words, nbc, false );
    Local c, piece;
    c.attach( cv, cap );
    if constexpr ( nbc > 1 ) piece.attach( cv, cap );
    else                     piece = c;                  // never touched: the density does not split

    const SI n = pd.nb_seeds();
    for ( SI k = thread_index; k < n; k += nb_threads ) {
        if ( ! make_cell( pd, c, k, dom ) || ! integrate_into<TF>( res( pd.user_id( k ) ), c, piece, dist ) ) {
            ask_more( scratch, cv );
            return;
        }
        memorize( c, k, memo_nbrs, memo_counts );
    }
}

/// the moments of each cell ( see `integrate_moments_into` ), same sweep as `measures`
template<class PD>
HD void moments( const PD &pd, auto &&mass, auto &&first, auto &&second, const auto &dom, auto &&scratch, const auto &dist,
              SI thread_index, SI nb_threads ) {
    using TF    = typename PD::TF;
    using TK    = KernelType<DECAYED_TYPE_OF( scratch )>;
    using Local = typename DECAYED_TYPE_OF( dom )::template Local<TK>;
    constexpr int nbc = nb_work_cells<DECAYED_TYPE_OF( dist )>();
    Carver cv = carver_of( scratch );
    const SI cap = cap_in<Local,TF>( cv.nb_words, nbc, false );
    Local c, piece;
    c.attach( cv, cap );
    if constexpr ( nbc > 1 ) piece.attach( cv, cap );
    else                     piece = c;

    const SI n = pd.nb_seeds();
    for ( SI k = thread_index; k < n; k += nb_threads ) {
        const SI u = pd.user_id( k );
        if ( ! make_cell( pd, c, k, dom ) || ! integrate_moments_into<TF>( mass( u ), first( u ), second( u ), c, piece, dist ) ) {
            ask_more( scratch, cv );
            return;
        }
    }
}

// ---- the adjoint ---------------------------------------------------------------------------------
// The chain is `m_k <- vertices <- planes <- seeds`, and each arrow is a closed form:
// `measure_bwd` answers the first; a vertex is the CORNER of its `D` cuts, so it solves
// `A x = b` with the cut directions as rows, and a small solve per vertex sends
// its cotangent back onto its planes ( `scatter_cell_grad` ); a plane is the weighted bisector of
// two seeds, which differentiates in two lines. Nothing of the forward is kept: the cell is REBUILT.

/// `grad_vp` ( one cotangent per vertex of `cell` ) -> the seeds. Every seed other than the cell
/// touches receives an atomic add; the share of `k0`, to which EVERY vertex contributes, is
/// summed in a register and added once.
template<class PD>
HD void scatter_cell_grad( const PD &pd, SI k0, const auto &cell, const auto &grad_vp, auto &&grad_positions, auto &&grad_weights ) {
    using TF = typename PD::TF;
    constexpr int D = PD::ct_dim;
    if constexpr ( grad_positions.surely_null && grad_weights.surely_null ) {
        return;
    } else {
        if ( ! cell.bounded() || cell.nb_vertices() == 0 )
            return;

        auto atomic_add_to = []( auto &&dst, TF v ) {
            if constexpr ( ! dst.surely_null )
                atomic_add( dst.ref(), v );
        };

        const auto p0 = pd.point( k0 );
        auto acc_p0 = Vector<TF,D>::zeros();
        TF acc_w0 = 0;

        const int nv = cell.nb_vertices();
        for ( int v = 0; v < nv; ++v ) {
            const auto q = Vector<TF,D>::with_func( [&]( PI d ) { return grad_vp( v, int( d ) ); } );
            const auto x = Vector<TF,D>::with_func( [&]( PI d ) { return TF( cell.coord( v, int( d ) ) ); } );

            // `A x = b`, rows = the directions of the vertex's cuts. A cotangent `q` on `x`
            // reaches the planes through `u` with `A^T u = q`: `d off_r -> u[ r ]`, `d dir_r -> - u[ r ] * x`.
            // Rows that do not face a seed are read back from the geometry: their scale
            // changes nothing for the `u` of the rows that matter.
            Matrix<TF,D> At;
            int cuts[ D ];
            for ( int r = 0; r < D; ++r ) {
                cuts[ r ] = cell.vertex_cut( v, r );
                Vector<TF,D> dir;
                TF off;
                plane_of( pd, cell, k0, cuts[ r ], dir, off );
                for ( int c = 0; c < D; ++c )
                    At( c, r ) = dir[ c ];
            }
            const auto u = Matrix<TF,D>::solve_ge( At, q );

            for ( int r = 0; r < D; ++r ) {
                const int k1 = cell.cid[ cuts[ r ] ];
                if ( k1 < 0 )                            // the domain, a piece, or a fake wall
                    continue;
                // the plane of the pair `( k0, k1 )`: `dir = p1 - p0`,
                // `off = ( |p1|^2 - |p0|^2 ) / 2 + ( w0 - w1 ) / 2`. Both differentiate in place.
                const TF g_off = u[ r ];
                const auto p1 = pd.point( k1 );
                for ( int d = 0; d < D; ++d ) {
                    const TF g_dir = - g_off * x[ d ];
                    atomic_add_to( grad_positions( k1, d ), g_dir + g_off * p1[ d ] );
                    acc_p0[ d ] -= g_dir + g_off * p0[ d ];
                }
                atomic_add_to( grad_weights( k1 ), - g_off / 2 );
                acc_w0 += g_off / 2;
            }
        }

        for ( int d = 0; d < D; ++d )
            atomic_add_to( grad_positions( k0, d ), acc_p0[ d ] );
        atomic_add_to( grad_weights( k0 ), acc_w0 );
    }
}

template<class PD,class Local>
HD bool integrate_bwd_into( const PD &pd, SI k0, auto &&grad_res, const Local &cell, Local &piece,
                         GradVp<typename PD::TF> &grad_vp, auto &&grad_positions, auto &&grad_weights,
                         auto &&grad_dist, const auto &dist ) {
    using TF = typename PD::TF;
    constexpr int D = PD::ct_dim;
    PieceWorkspace<Local> ws{ piece };
    const TF g = grad_res;
    dist.for_each_piece( cell, ws, [&]( const auto &pc, const auto &dens ) {
        if constexpr ( DECAYED_TYPE_OF( dens )::is_constant ) {
            // the DENSITY's share: mass is linear in it, so the derivative with respect to
            // the value carried by this piece IS its volume ( an infinite piece has none )
            if ( pc.bounded() )
                dens.add_value_grad( grad_dist, g * pc.template measure<TF>() );
            // ... and the GEOMETRY's share, through the usual chain
            pc.template measure_bwd<TF>( g * dens.value, grad_vp );
            scatter_cell_grad( pd, k0, pc, grad_vp, grad_positions, grad_weights );
        } else {
            if ( ! pc.bounded() )
                return;
            const int nv = pc.nb_vertices();
            for ( int v = 0; v < nv; ++v )
                for ( int c = 0; c < D; ++c )
                    grad_vp( v, c ) = 0;
            pc.for_each_simplex( [&]( const auto &chain ) {
                auto grad_pts = Vector<Vector<TF,D>,D+1>( Function(), [&]( PI ) { return Vector<TF,D>::zeros(); } );
                dens.integrate_over_simplex_bwd( simplex_points<TF,D>( pc, chain ), g, grad_pts, grad_dist );
                for ( int k = 0; k <= D; ++k )
                    for ( int c = 0; c < D; ++c )
                        grad_vp( int( chain[ k ] ), c ) += grad_pts[ k ][ c ];
            } );
            scatter_cell_grad( pd, k0, pc, grad_vp, grad_positions, grad_weights );
        }
    } );
    return ! ws.overflow;
}

template<class PD>
HD void measures_bwd( const PD &pd, auto &&res, const auto &dom, auto &&grad_res, auto &&grad_positions, auto &&grad_weights,
                   auto &&scratch, const auto &dist, auto &&grad_dist, SI thread_index, SI nb_threads ) {
    using TF    = typename PD::TF;
    using TK    = KernelType<DECAYED_TYPE_OF( scratch )>;
    using Local = typename DECAYED_TYPE_OF( dom )::template Local<TK>;
    constexpr int nbc = nb_work_cells<DECAYED_TYPE_OF( dist )>();
    Carver cv = carver_of( scratch );
    const SI cap = cap_in<Local,TF>( cv.nb_words, nbc, true );
    Local c, piece;
    c.attach( cv, cap );
    if constexpr ( nbc > 1 ) piece.attach( cv, cap );
    else                     piece = c;
    GradVp<TF> grad_vp{ cv.take<TF>( PD::ct_dim * cap ), cap };

    const SI n = pd.nb_seeds();
    for ( SI k = thread_index; k < n; k += nb_threads ) {
        if ( ! make_cell( pd, c, k, dom )
          || ! integrate_bwd_into( pd, k, grad_res( pd.user_id( k ) ), c, piece, grad_vp, grad_positions, grad_weights, grad_dist, dist ) ) {
            ask_more( scratch, cv );
            return;
        }
    }
}

/// The cell of seed `k`, KEPT: built like any other, then put into `res` -- with its cut
/// identifiers translated for the user. The only query whose memory is a function
/// of the number of seeds: which is what a DISPLAY is.
template<class PD>
HD void build_cell( const PD &pd, SI k, const auto &dom, auto &&res, auto &&scratch, SI thread_index ) {
    using TK    = KernelType<DECAYED_TYPE_OF( scratch )>;
    using Local = typename DECAYED_TYPE_OF( dom )::template Local<TK>;
    Carver cv = carver_of( scratch, thread_index );
    Local c = local_on<Local>( scratch, cv );
    if ( ! make_cell( pd, c, k, dom ) ) {
        ask_more( scratch, cv );
        return;
    }
    c.tidy();
    for ( int q = 0; q < c.nb_cuts(); ++q )
        if ( c.cid[ q ] >= 0 )
            c.cid[ q ] = int( pd.user_id( c.cid[ q ] ) );
    c.store( res );
}

/// ONE ROW OF THE transport HESSIAN, `d m_k / d w_j` for the neighbors `j` of cell `k`:
/// the bisector `( k, j )` slides by `dw / ( 2 | p_k - p_j | )` when `w_j` goes up by `dw`, and what
/// it sweeps is the density integrated over the common FACET. So
/// `d m_k / d w_j = - int_{facet} rho / ( 2 | p_k - p_j | )`, and `d m_k / d w_k` is the summed
/// opposite ( a row sums to zero -- which the caller recomposes ). The domain ( negative ids ) does not
/// move. `res`: `ids( r )` ( user identifiers ) and `vals( r )` ( the values, POSITIVE ),
/// `nb_nbrs` the count -- a capacity that loom doubles if it is missing.
///
/// For piecewise-constant density only ( `Image`, Lebesgue ): the facet of a piece is
/// flat and the density is a number there.
template<class PD>
HD void hessian_row( const PD &pd, SI k, const auto &dom, auto &&res, auto &&scratch, SI thread_index, const auto &dist ) {
    using TF    = typename PD::TF;
    using TK    = KernelType<DECAYED_TYPE_OF( scratch )>;
    using Local = typename DECAYED_TYPE_OF( dom )::template Local<TK>;
    constexpr int D = PD::ct_dim;
    constexpr int nbc = nb_work_cells<DECAYED_TYPE_OF( dist )>();
    Carver cv = carver_of( scratch, thread_index );
    const SI cap = cap_in<Local,TF>( cv.nb_words, nbc, false );
    Local c, piece;
    c.attach( cv, cap );
    if constexpr ( nbc > 1 ) piece.attach( cv, cap );
    else                     piece = c;
    if ( ! make_cell( pd, c, k, dom ) ) {
        ask_more( scratch, cv );
        return;
    }
    c.tidy();

    // the neighbors of the cell are its live cuts: one slot per cut, accumulated piece
    // by piece ( a piece carries the same identifiers, plus those of its tile )
    const int nc = c.nb_cuts();
    if ( ! res.nb_nbrs.set( nc ) )
        return;                                          // too many neighbors: loom doubles and calls again
    for ( int q = 0; q < nc; ++q ) {
        res.ids( q ) = c.cid[ q ] >= 0 ? int( pd.user_id( c.cid[ q ] ) ) : int( c.cid[ q ] );
        res.vals( q ) = 0;
    }
    const auto p0 = pd.point( k );

    PieceWorkspace<Local> ws{ piece };
    dist.for_each_piece( c, ws, [&]( const auto &pc, const auto &dens ) {
        static_assert( DECAYED_TYPE_OF( dens )::is_constant, "hessian: piecewise-constant density only" );
        const TF rho = TF( dens.value );
        pc.template for_each_facet<TF>( [&]( int cut, TF mes ) {
            const int id = pc.cid[ cut ];
            if ( id < 0 )
                return;                                  // the domain, or a tile edge: immobile
            const auto pj = pd.point( id );
            TF d2 = 0;
            for ( int d = 0; d < D; ++d )
                d2 += ( pj[ d ] - p0[ d ] ) * ( pj[ d ] - p0[ d ] );
            const TF val = rho * mes / ( 2 * sdot::sqrt( d2 ) );
            for ( int q = 0; q < nc; ++q )
                if ( c.cid[ q ] == id ) {
                    res.vals( q ) += val;
                    break;
                }
        } );
    } );
    if ( ws.overflow )
        ask_more( scratch, cv );
}

} // namespace diagram
} // namespace sdot
