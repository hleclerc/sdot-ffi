#pragma once

#include <loom/support/common_macros.h> // HD

// =====================================================================================
// THE 2D CELL IN MEMORY: the intermediate representation, the one the kernels manipulate.
//
// The vertices in CYCLIC ORDER are all the geometry: the area is read by the shoelace formula, and
// the invariant "CUT `i` CARRIES THE EDGE `[ v_i, v_i+1 ]`" gives the connectivity without searching
// for anything -- vertex `i` is the corner of cuts `i-1` and `i`, and `nb_cuts == nb_vertices`. Everything
// fits in two coordinate arrays and one array of cut identifiers, carved out of the
// work-item's scratch ( `Scratch.h` ) at a capacity `cap` that the host has decided.
//
// This is NOT what a user sees (`Cell.py` derives the "convenient" tensors from it): it is
// what the already-written algorithms run on -- the in-place scalar cut, the excursion of the
// register kernel (`Engine2Reg.h`), the computation of the measure and its adjoint.
//
// = THE PLANES ARE NOT STORED, unless they have to be
//
// A BOUNDED cell does not need its planes: they are read back from its edges
// (`planes_from_vertices`), and neither the cut nor the measure reads them. An UNBOUNDED cell,
// on the other hand, is a replacement simplex whose `INFINITE` walls carry invented
// offsets that must be PUSHED BACK before each cut (`grow_for`) -- and for that the planes are needed.
// They are thus kept (`has_planes`) only in that regime, which is the rare one.
//
// = THE SCALAR CUT, IN PLACE
//
// The outside of a convex set cut by a half-plane is a contiguous CYCLIC range: we count the
// vertices outside, find the start of the range, and the output has EXACTLY
// `nb - nb_out + 2` vertices. The two intersections are anchored on the INSIDE vertex
// (`v_in + ( v_out - v_in ) * t` is exactly `v_in` at `t == 0`, which the symmetric form
// does not give -- and without which the range stops being contiguous).
//
// The scan is in TWO passes, and this is the opposite of what one would think: counting and looking
// for the start of the range in the same loop makes fewer instructions and runs slower, the
// carried dependency preventing the vectorizer from taking the dot product. Measured on the bench:
// 17.1 ns per cut in one pass, 13.2 in two.
// =====================================================================================

#include <loom/support/math.h>
#include <loom/support/common_types.h>
#include <loom/support/containers/Vector.h>
#include "Scratch.h"
#include "Plane.h"
#include "State.h"
#include "Ids.h"

#include <type_traits>
#include <limits>
#include <cmath>

namespace sdot {

template<class TK>
struct Local2 {
    static constexpr int ct_dim = 2;
    using TKernel = TK;
    using PlaneT  = Plane<TK,2>;

    int  nb         = 0;      ///< vertices ( = cuts ); 0 = empty
    int  cap        = 0;      ///< the capacity of the arrays
    bool unbounded  = false;  ///< `INFINITE` walls remain
    bool has_planes = false;  ///< `pdx / pdy / po` are up to date

    TK  *vx = nullptr, *vy = nullptr;
    int *cid = nullptr;
    TK  *pdx = nullptr, *pdy = nullptr, *po = nullptr;   ///< the plane of cut `i` ( if `has_planes` )
    TK  *s = nullptr;                                    ///< the dot products of the current cut
    TK  *rx = nullptr, *ry = nullptr;                    ///< the push velocities ( unbounded )

    // ---- the scratch -------------------------------------------------------------------------

    /// the number of words needed for `cap` vertices -- THE SAME FORMULA as `Cell_2.scratch_words`
    HD static constexpr SI words_for( SI cap ) { return 8 * words_of<TK>( cap ) + words_of<int>( cap ); }

    /// lays the arrays out in `c`. `false` if there is no room ( nothing is then usable )
    HD bool attach( Carver &c, SI capacity ) {
        cap = int( capacity );
        vx = c.take<TK>( cap ); vy = c.take<TK>( cap ); cid = c.take<int>( cap );
        pdx = c.take<TK>( cap ); pdy = c.take<TK>( cap ); po = c.take<TK>( cap );
        s = c.take<TK>( cap ); rx = c.take<TK>( cap ); ry = c.take<TK>( cap );
        nb = 0; unbounded = false; has_planes = false;
        return ! c.overflow;
    }

    /// copies `o` ( geometry and state ): `cap` must be enough
    HD bool copy_from( const Local2 &o ) {
        if ( o.nb > cap )
            return false;
        nb = o.nb; unbounded = o.unbounded; has_planes = o.has_planes;
        for ( int i = 0; i < nb; ++i ) {
            vx[ i ] = o.vx[ i ]; vy[ i ] = o.vy[ i ]; cid[ i ] = o.cid[ i ];
            if ( has_planes ) { pdx[ i ] = o.pdx[ i ]; pdy[ i ] = o.pdy[ i ]; po[ i ] = o.po[ i ]; }
        }
        return true;
    }

    // ---- what everyone reads -----------------------------------------------------------------
    HD int  nb_vertices() const { return nb; }
    HD int  nb_cuts () const { return nb; }
    HD bool bounded () const { return ! unbounded; }
    HD TK   coord   ( int i, int d ) const { return d ? vy[ i ] : vx[ i ]; }

    /// the two cuts of vertex `i`, in order: `r == 0` -> cut `i-1`, `r == 1` -> `i`
    HD int  vertex_cut ( int i, int r ) const { return r ? i : ( i ? i - 1 : nb - 1 ); }

    /// what the supplier sees ( see `Engine.h` )
    HD StateMem<TK> state() const { return { nb, vx, vy, cid, ! unbounded }; }

    // ---- the starting states -----------------------------------------------------------------

    /// the parallelogram `origin + s * axes( 0 ) + t * axes( 1 )`, `s, t` in `[ 0, 1 ]`, in direct
    /// cyclic order if `axes` is. All the cuts carry `cut_id`.
    HD bool init_hypercube( const auto &origin, const auto &axes, int cut_id ) {
        if ( cap < 4 )
            return false;
        const TK ox = TK( origin[ 0 ] ), oy = TK( origin[ 1 ] );
        const TK ax = TK( axes( 0, 0 ) ), ay = TK( axes( 0, 1 ) );
        const TK bx = TK( axes( 1, 0 ) ), by = TK( axes( 1, 1 ) );
        vx[ 0 ] = ox;           vy[ 0 ] = oy;
        vx[ 1 ] = ox + ax;      vy[ 1 ] = oy + ay;
        vx[ 2 ] = ox + ax + bx; vy[ 2 ] = oy + ay + by;
        vx[ 3 ] = ox + bx;      vy[ 3 ] = oy + by;
        for ( int i = 0; i < 4; ++i )
            cid[ i ] = cut_id;
        nb         = 4;
        unbounded  = cut_id == cell_ids::INFINITE;
        has_planes = false;
        if ( unbounded )
            planes_from_vertices();
        return true;
    }

    /// "THE WHOLE PLANE": the triangle `( 0, 0 ), ( 1, 0 ), ( 0, 1 )` whose three sides are
    /// marked `INFINITE`. Its offsets are invented; `grow_for` pushes them back cut after cut.
    HD bool init_unbounded() {
        if ( cap < 3 )
            return false;
        vx[ 0 ] = 0; vy[ 0 ] = 0;
        vx[ 1 ] = 1; vy[ 1 ] = 0;
        vx[ 2 ] = 0; vy[ 2 ] = 1;
        for ( int i = 0; i < 3; ++i )
            cid[ i ] = cell_ids::INFINITE;
        nb        = 3;
        unbounded = true;
        planes_from_vertices();
        return true;
    }

    HD void make_empty() { nb = 0; unbounded = false; has_planes = false; }
    HD void tidy() {}                                 ///< nothing to tidy: no dead cut in 2D

    // ---- the planes, read back from the geometry ---------------------------------------------

    /// the plane of edge `i`: the OUTWARD normal of `[ v_i, v_i+1 ]` for a direct polygon,
    /// and the offset read on `v_i`. Exact for a real plane as well as for a pushed wall -- both
    /// ends of an edge are on its plane.
    HD void plane_of_edge( int i, TK &dx, TK &dy, TK &off ) const {
        const int j = i + 1 < nb ? i + 1 : 0;
        dx  = vy[ j ] - vy[ i ];
        dy  = vx[ i ] - vx[ j ];
        off = dx * vx[ i ] + dy * vy[ i ];
    }

    HD void planes_from_vertices() {
        for ( int i = 0; i < nb; ++i )
            plane_of_edge( i, pdx[ i ], pdy[ i ], po[ i ] );
        has_planes = true;
    }

    /// the plane of cut `i`, in the caller's float `T` ( read back from the vertices, or
    /// taken from the table when it is kept )
    template<class T>
    HD void plane( int i, T *dir, T &off ) const {
        if ( has_planes ) {
            dir[ 0 ] = T( pdx[ i ] ); dir[ 1 ] = T( pdy[ i ] ); off = T( po[ i ] );
        } else {
            TK dx, dy, o;
            plane_of_edge( i, dx, dy, o );
            dir[ 0 ] = T( dx ); dir[ 1 ] = T( dy ); off = T( o );
        }
    }

    // ---- the cut -----------------------------------------------------------------------------

    /// Cuts by `p`, IN PLACE. Returns a `CutStatus`; on `NO_ROOM` the cell is left intact.
    HD int cut( const PlaneT &p ) {
        if ( unbounded ) {
            grow_for( p );
            return cut_impl<true>( p );
        }
        return has_planes ? cut_impl<true>( p ) : cut_impl<false>( p );
    }

    /// How many vertices the half-space leaves OUTSIDE -- the "nothing to remove" test, separate and
    /// small: the predicate `s > 0` is only written here and in `cut_impl`, so the two cannot
    /// answer differently on a vertex at the epsilon of the plane.
    HD int nb_outside( const PlaneT &p ) const {
        int res = 0;
        for ( int i = 0; i < nb; ++i )
            res += ( p.dir[ 0 ] * vx[ i ] + p.dir[ 1 ] * vy[ i ] - p.off ) > 0;
        return res;
    }

    template<bool PL>
    HD int cut_impl( const PlaneT &p ) {
        int nb_out = 0;
        for ( int i = 0; i < nb; ++i ) {                 // pure reduction: the vectorizer takes it
            s[ i ] = p.dir[ 0 ] * vx[ i ] + p.dir[ 1 ] * vy[ i ] - p.off;
            nb_out += s[ i ] > 0;
        }
        if ( nb_out == 0 )
            return CutStatus::UNCHANGED;
        if ( nb_out == nb ) {
            nb = 0;
            unbounded = false;
            return CutStatus::EMPTY;
        }

        int i1 = 0;                                      // unique: the outside of a convex set
        for ( int i = 0; i < nb; ++i ) {                 // cut is in one piece
            const int q = i ? i - 1 : nb - 1;
            if ( s[ i ] > 0 && ! ( s[ q ] > 0 ) ) { i1 = i; break; }
        }

        const int nb_in  = nb - nb_out;
        const int new_nb = nb_in + 2;
        if ( new_nb > cap )
            return CutStatus::NO_ROOM;                  // the cell stays INTACT

        const int j0 = ( i1 + nb - 1 ) % nb;             // last INSIDE before the range
        const int j2 = ( i1 + nb_out - 1 ) % nb;         // last OUTSIDE
        const int j3 = ( j2 + 1 ) % nb;                  // first INSIDE after

        const TK s0 = s[ j0 ], s1 = s[ i1 ], s2 = s[ j2 ], s3 = s[ j3 ];
        const TK ta  = s0 / ( s0 - s1 );
        const TK pax = vx[ j0 ] + ( vx[ i1 ] - vx[ j0 ] ) * ta;
        const TK pay = vy[ j0 ] + ( vy[ i1 ] - vy[ j0 ] ) * ta;
        const TK tb  = s3 / ( s3 - s2 );
        const TK pbx = vx[ j3 ] + ( vx[ j2 ] - vx[ j3 ] ) * tb;
        const TK pby = vy[ j3 ] + ( vy[ j2 ] - vy[ j3 ] ) * tb;
        // cut `j2` carries the edge `[ v_j2, v_j3 ]`, which survives truncated: it is read back
        // NOW, `j2` is about to be overwritten.
        const int bid = cid[ j2 ];
        TK bdx = 0, bdy = 0, bo = 0;
        if constexpr ( PL ) { bdx = pdx[ j2 ]; bdy = pdy[ j2 ]; bo = po[ j2 ]; }

        auto move = [ & ]( int d, int t ) {
            vx[ d ] = vx[ t ]; vy[ d ] = vy[ t ]; cid[ d ] = cid[ t ];
            if constexpr ( PL ) { pdx[ d ] = pdx[ t ]; pdy[ d ] = pdy[ t ]; po[ d ] = po[ t ]; }
        };
        auto put = [ & ]( int d, TK x, TK y, int id, TK dx, TK dy, TK o ) {
            vx[ d ] = x; vy[ d ] = y; cid[ d ] = id;
            if constexpr ( PL ) { pdx[ d ] = dx; pdy[ d ] = dy; po[ d ] = o; }
        };

        if ( i1 <= j2 ) {
            if ( nb_out == 1 ) {                         // one notch more: the tail goes RIGHT
                for ( int i = nb; i > i1 + 1; --i ) move( i, i - 1 );
            } else if ( nb_out > 2 ) {                   // too much room: the tail comes back LEFT
                const int gap = nb_out - 2;
                for ( int i = j2 + 1; i < nb; ++i ) move( i - gap, i );
            }                                            // `nb_out == 2`: nothing to shift
            put( i1,     pax, pay, p.id, p.dir[ 0 ], p.dir[ 1 ], p.off );
            put( i1 + 1, pbx, pby, bid,  bdx, bdy, bo );
        } else {
            // the range WRAPS AROUND, so the inside is contiguous: `[ j3, j3 + nb_in )`.
            if ( j3 >= 2 ) for ( int o = 0; o < nb_in; ++o ) move( 2 + o, j3 + o );
            else           for ( int o = nb_in - 1; o >= 0; --o ) move( 2 + o, j3 + o );
            put( 0, pax, pay, p.id, p.dir[ 0 ], p.dir[ 1 ], p.off );
            put( 1, pbx, pby, bid,  bdx, bdy, bo );
        }
        nb = new_nb;

        if ( unbounded ) {
            unbounded = false;
            for ( int i = 0; i < nb; ++i )
                unbounded |= cid[ i ] == cell_ids::INFINITE;
            if ( ! unbounded )
                has_planes = false;                      // nothing left to push back: we no longer keep them
        }
        return CutStatus::CUT;
    }

    // ---- the replacement simplex ---------------------------------------------------------------

    /// The VELOCITY of vertex `i` when the `INFINITE` walls of `g` are pushed back: it is the corner of
    /// its two cuts, so it solves the same 2x2 with the `INFINITE` indicators as right-hand
    /// side. Zero for a real vertex, which does not move.
    HD void growth_rate( int i, TK &rx, TK &ry ) const {
        const int c0 = vertex_cut( i, 0 ), c1 = vertex_cut( i, 1 );
        const TK f0 = cid[ c0 ] == cell_ids::INFINITE, f1 = cid[ c1 ] == cell_ids::INFINITE;
        if ( f0 == 0 && f1 == 0 ) { rx = 0; ry = 0; return; }
        const TK a1 = pdx[ c0 ], b1 = pdy[ c0 ], a2 = pdx[ c1 ], b2 = pdy[ c1 ];
        const TK det = a1 * b2 - b1 * a2;
        if ( det == 0 ) { rx = 0; ry = 0; return; }
        rx = ( f0 * b2 - b1 * f1 ) / det;
        ry = ( a1 * f1 - f0 * a2 ) / det;
    }

    /// PUSHES BACK the `INFINITE` walls until the ranking of the vertices by `p` is the one
    /// it has at infinity. Each vertex travels in a straight line, so its signed distance to the plane is
    /// AFFINE in the push and "when would it change side?" is a division; we push
    /// beyond the farthest one, and a bit more, so that no vertex stays ON the plane.
    HD void grow_for( const PlaneT &p ) {
        if ( ! has_planes )
            planes_from_vertices();

        static constexpr int max_rounds = 4;   // 2 suffice in exact arithmetic
        // THE MARGIN IS NOT A MACHINE EPSILON: a vertex pushed by one epsilon falls back to the
        // precision of the dot product, and two vertices coincident to 1e-15 then rank
        // each on its own side of the plane -- which breaks the combinatorics of the cut ( seen in 4D ).
        // It is small compared to the geometry, large compared to the rounding: where the FAKE
        // vertices land has no geometric meaning anyway.
        const TK margin = std::is_same_v<TK,float> ? TK( 1e-5 ) : TK( 1e-6 );

        for ( int i = 0; i < nb; ++i )
            growth_rate( i, rx[ i ], ry[ i ] );

        // "the velocity does not change the distance to the plane" is judged AT A TOLERANCE, not at zero:
        // the planes are READ BACK from the geometry, so an exactly axial normal comes out with
        // components at 1e-17, and `root = - s / rate` would then make a push of 1e17. A ray
        // parallel to the plane to within 1e-9 counts as parallel.
        const TK tol = TK( 1e-9 ) * std::sqrt( p.dir[ 0 ] * p.dir[ 0 ] + p.dir[ 1 ] * p.dir[ 1 ] );

        TK g = 0;
        for ( int round = 0; round < max_rounds; ++round ) {
            bool push = false;
            TK grow = 0;
            for ( int i = 0; i < nb; ++i ) {
                const TK rate = p.dir[ 0 ] * rx[ i ] + p.dir[ 1 ] * ry[ i ];
                const TK nr = std::sqrt( rx[ i ] * rx[ i ] + ry[ i ] * ry[ i ] );
                if ( nr == 0 || ( rate < 0 ? - rate : rate ) <= tol * nr )
                    continue;
                const TK s = p.dir[ 0 ] * ( vx[ i ] + g * rx[ i ] ) + p.dir[ 1 ] * ( vy[ i ] + g * ry[ i ] ) - p.off;
                // `>= 0`, not `> 0`: a vertex exactly ON the plane is not yet on the side where
                // it would be at infinity -- it is precisely a case to push.
                const TK root = - s / rate;
                if ( root >= 0 ) {
                    push = true;
                    if ( root > grow )
                        grow = root;
                }
            }
            if ( ! push )
                break;
            g += grow + ( grow + 1 ) * margin;
        }
        if ( g == 0 )
            return;

        for ( int i = 0; i < nb; ++i ) {
            if ( cid[ i ] == cell_ids::INFINITE )
                po[ i ] += g;
            vx[ i ] += g * rx[ i ];
            vy[ i ] += g * ry[ i ];
        }
    }

    // ---- the measure, in the caller's float `TF` -----------------------------------------------

    template<class TF>
    HD TF measure() const {
        if ( unbounded )
            return std::numeric_limits<TF>::max();
        TF sum = 0;
        for ( int i = 0; i < nb; ++i ) {
            const int j = i + 1 < nb ? i + 1 : 0;
            sum += TF( vx[ i ] ) * TF( vy[ j ] ) - TF( vx[ j ] ) * TF( vy[ i ] );
        }
        return sum / 2;
    }

    /// the adjoint of the shoelace formula: `grad_vp( i, d )` receives the cotangent of vertex `i` ( WRITTEN, not
    /// accumulated ). Nothing for an unbounded cell, whose measure is a constant.
    template<class TF>
    HD void measure_bwd( TF grad_res, auto &&grad_vp ) const {
        if ( unbounded )
            return;
        for ( int i = 0; i < nb; ++i ) {
            const int p = i ? i - 1 : nb - 1;
            const int n = i + 1 < nb ? i + 1 : 0;
            grad_vp( i, 0 ) = grad_res * ( TF( vy[ n ] ) - TF( vy[ p ] ) ) / 2;
            grad_vp( i, 1 ) = grad_res * ( TF( vx[ p ] ) - TF( vx[ n ] ) ) / 2;
        }
    }

    /// a fan from vertex 0: the triangles `( 0, i, i+1 )` TILE the convex set.
    /// `func( chain )` receives three vertex indices.
    /// `func( c, measure )` for each cut `c` that carries an edge: its LENGTH ( cut `i`
    /// carries the edge `[ v_i, v_i+1 ]` ) -- what the Hessian of a transport reads ( `diagram::hessian_row` )
    template<class TF>
    HD void for_each_facet( auto &&func ) const {
        for ( int i = 0; i < nb; ++i ) {
            const int j = i + 1 < nb ? i + 1 : 0;
            const TF dx = TF( vx[ j ] ) - TF( vx[ i ] ), dy = TF( vy[ j ] ) - TF( vy[ i ] );
            func( i, sdot::sqrt( dx * dx + dy * dy ) );
        }
    }

    HD void for_each_simplex( auto &&func ) const {
        Vector<SI,3> chain;
        chain[ 0 ] = 0;
        for ( int i = 1; i + 1 < nb; ++i ) {
            chain[ 1 ] = i;
            chain[ 2 ] = i + 1;
            func( chain );
        }
    }

    HD void bbox( TK *lo, TK *hi ) const {
        lo[ 0 ] = hi[ 0 ] = nb ? vx[ 0 ] : TK( 0 );
        lo[ 1 ] = hi[ 1 ] = nb ? vy[ 0 ] : TK( 0 );
        for ( int i = 1; i < nb; ++i ) {
            lo[ 0 ] = vx[ i ] < lo[ 0 ] ? vx[ i ] : lo[ 0 ]; hi[ 0 ] = vx[ i ] > hi[ 0 ] ? vx[ i ] : hi[ 0 ];
            lo[ 1 ] = vy[ i ] < lo[ 1 ] ? vy[ i ] : lo[ 1 ]; hi[ 1 ] = vy[ i ] > hi[ 1 ] ? vy[ i ] : hi[ 1 ];
        }
    }

    // ---- the tensors ( see `Cell.py` for their shape ) ----------------------------------------

    /// from a `Cell_2` view ( an already indexed item ): `vertex_positions [ nv, 2 ]`, `cut_ids`.
    /// `false` if the cell does not fit in `cap`.
    HD bool load( const auto &c ) {
        const int n = int( SI( c.nb_vertices ) );
        if ( n > cap )
            return false;
        nb = n;
        unbounded = false;
        for ( int i = 0; i < nb; ++i ) {
            vx [ i ] = TK( c.vertex_positions( i, 0 ) );
            vy [ i ] = TK( c.vertex_positions( i, 1 ) );
            cid[ i ] = int( SI( c.cut_ids( i ) ) );
            unbounded |= cid[ i ] == cell_ids::INFINITE;
        }
        has_planes = false;
        return true;
    }

    /// to a `Cell_2` view. Returns `false` if it is too small: the wanted count is then
    /// recorded ( `ShapeVarView::set` ) and NOTHING is written -- the host reserves more and retries.
    HD bool store( auto &&c ) const {
        if ( ! c.nb_vertices.set( nb ) )
            return false;
        c.nb_cuts.set( nb );
        for ( int i = 0; i < nb; ++i ) {
            c.vertex_positions( i, 0 ) = vx[ i ];
            c.vertex_positions( i, 1 ) = vy[ i ];
            c.cut_ids( i ) = cid[ i ];
        }
        return true;
    }
};

} // namespace sdot
