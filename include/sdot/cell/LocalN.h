#pragma once

#include <loom/support/common_macros.h> // HD

// =====================================================================================
// THE CELL IN DIMENSION `D >= 3`: a SIMPLE polytope that cuts itself.
//
// Nothing of 2D survives here: there is no global cyclic order any more, and a cut carries a
// FACE. Connectivity is therefore carried by the VERTICES, each one naming its `D` cuts and its `D`
// neighbors:
//
//     vk[ r ][ i ]   the `r`-th cut of vertex `i`, as an INDEX into `cid` -- sorted ascending
//     vn[ r ][ i ]   the neighbor of `i` ON THE OTHER SIDE of the edge carried by the `D - 1` cuts
//                    other than `vk[ r ][ i ]`: neighbor `r` is "opposite" cut `r`
//
// A vertex of a simple polytope has `D` cuts, hence `D` bundles of `D - 1` cuts, hence `D`
// edges, and this matching is a bijection. It makes FREE what an edge list made us
// search for: the faces that carry edge `r` are the cuts of the vertex minus the
// `r`-th one, without traversing anything.
//
// THE CUT LIST IS LOCAL: `cid[ 0 .. nc )` carries the global identifiers, everything else
// handles only indices into it. It only ever ADDS during the life of the cell -- a
// cut that loses all its vertices leaves a dead entry there -- and `compact()` removes them only
// when it is full.
//
// THE ASSUMPTION: each vertex is on EXACTLY `D` planes ( general position ). This is what makes
// the cut purely combinatorial: two new vertices of the created face are neighbors exactly
// when they share `D - 2` OLD cuts. A plane that goes exactly through a vertex does not
// remove it ( strict `s > 0` ), which is the only concession made to degenerate
// configurations.
//
// THE CUT FILLS THE HOLES: vertices that are outside leave free slots, the new vertices
// settle into them, and the survivors keep their index -- so their adjacency stays valid
// as is, and there is no renumbering table. The commit is `O( nm )`, not
// `O( nv )`; the only case where a kept vertex moves is that of a cut that removes more
// vertices than it creates, and then only the difference moves.
//
// This is the bench's `Cellule3D.h`, with the dimension as a parameter.
// =====================================================================================

#include <loom/support/math.h>
#include <loom/support/common_types.h>
#include <loom/support/containers/Matrix.h>
#include <loom/support/containers/Vector.h>
#include <asimd/asimd.h>
#include "Scratch.h"
#include "Plane.h"
#include "State.h"
#include "Ids.h"

#include <type_traits>
#include <limits>
#include <cmath>

namespace sdot {

template<class TK,int D>
struct LocalN {
    static_assert( D >= 3, "below 3D, it is `Local2`" );
    static constexpr int ct_dim = D;
    using TKernel = TK;
    using PlaneT  = Plane<TK,D>;

    int  nv = 0, nc = 0;
    int  cap = 0;                                        ///< the capacity: vertices AND cuts
    bool unbounded  = false;
    bool has_planes = false;

    TK  *v [ D ];
    int *vk[ D ];
    int *vn[ D ];
    int *cid;
    TK  *pd[ D ], *po;                                   ///< the plane of cut `k` ( if `has_planes` )

    // the temporaries, in the scratch too
    TK  *s, *nx[ D ], *rate[ D ], *s3[ 3 ];
    int *hole, *nk[ D - 1 ], *rec_v, *rec_f, *dest, *nn_[ D ], *fresh, *src, *dst, *m, *apex[ D ], *v0, *on_face;

    // ---- the scratch --------------------------------------------------------------------------

    /// how many words are needed for `cap` vertices ( and as many cuts ) -- THE SAME FORMULA as
    /// `Cell_N.scratch_words`
    HD static constexpr SI words_for( SI cap ) {
        return ( 4 * D + 5 ) * words_of<TK>( cap ) + ( 5 * D + 10 ) * words_of<int>( cap );
    }

    HD bool attach( Carver &c, SI capacity ) {
        cap = int( capacity );
        for ( int d = 0; d < D; ++d ) { v[ d ] = c.take<TK>( cap ); vk[ d ] = c.take<int>( cap ); vn[ d ] = c.take<int>( cap ); }
        cid = c.take<int>( cap );
        for ( int d = 0; d < D; ++d ) pd[ d ] = c.take<TK>( cap );
        po = c.take<TK>( cap );
        s = c.take<TK>( cap );
        for ( int d = 0; d < D; ++d ) { nx[ d ] = c.take<TK>( cap ); rate[ d ] = c.take<TK>( cap ); }
        for ( int d = 0; d < 3; ++d ) s3[ d ] = c.take<TK>( cap );
        hole = c.take<int>( cap );
        for ( int d = 0; d + 1 < D; ++d ) nk[ d ] = c.take<int>( cap );
        rec_v = c.take<int>( cap ); rec_f = c.take<int>( cap ); dest = c.take<int>( cap );
        for ( int d = 0; d < D; ++d ) nn_[ d ] = c.take<int>( cap );
        fresh = c.take<int>( cap ); src = c.take<int>( cap ); dst = c.take<int>( cap ); m = c.take<int>( cap );
        for ( int d = 0; d < D; ++d ) apex[ d ] = c.take<int>( cap );
        v0 = c.take<int>( cap ); on_face = c.take<int>( cap );
        nv = 0; nc = 0; unbounded = false; has_planes = false;
        return ! c.overflow;
    }

    HD bool copy_from( const LocalN &o ) {
        if ( o.nv > cap || o.nc > cap )
            return false;
        nv = o.nv; nc = o.nc; unbounded = o.unbounded; has_planes = o.has_planes;
        for ( int i = 0; i < nv; ++i )
            for ( int d = 0; d < D; ++d ) { v[ d ][ i ] = o.v[ d ][ i ]; vk[ d ][ i ] = o.vk[ d ][ i ]; vn[ d ][ i ] = o.vn[ d ][ i ]; }
        for ( int k = 0; k < nc; ++k ) {
            cid[ k ] = o.cid[ k ];
            if ( has_planes ) { for ( int d = 0; d < D; ++d ) pd[ d ][ k ] = o.pd[ d ][ k ]; po[ k ] = o.po[ k ]; }
        }
        return true;
    }

    // ---- what everybody reads ------------------------------------------------------------
    HD int  nb_vertices() const { return nv; }
    HD int  nb_cuts () const { return nc; }
    HD bool bounded () const { return ! unbounded; }
    HD TK   coord   ( int i, int d ) const { return v[ d ][ i ]; }
    HD int  vertex_cut ( int i, int r ) const { return vk[ r ][ i ]; }

    HD StateMemN<TK,D> state() const {
        StateMemN<TK,D> e;
        e.nb = nv;
        for ( int d = 0; d < D; ++d )
            e.v[ d ] = v[ d ];
        e.bounded = ! unbounded;
        return e;
    }

    // ---- the starting states -----------------------------------------------------------------

    /// the parallelotope `origin + sum_j t_j axes( j )`, `t` in `[ 0, 1 ]^D`. Vertex `b` has the bits
    /// of `b` as coordinates; cut `2 j + bit_j( b )` carries it, and its neighbor opposite
    /// that cut is `b ^ ( 1 << j )`. The cuts are in axis order, hence sorted.
    HD bool init_hypercube( const auto &origin, const auto &axes, int cut_id ) {
        if ( ( 1 << D ) > cap )
            return false;
        nv = 1 << D;
        nc = 2 * D;
        for ( int b = 0; b < nv; ++b ) {
            for ( int d = 0; d < D; ++d ) {
                TK x = TK( origin[ d ] );
                for ( int j = 0; j < D; ++j )
                    if ( b & ( 1 << j ) )
                        x += TK( axes( j, d ) );
                v[ d ][ b ] = x;
            }
            for ( int j = 0; j < D; ++j ) {
                vk[ j ][ b ] = 2 * j + ( ( b >> j ) & 1 );
                vn[ j ][ b ] = b ^ ( 1 << j );
            }
        }
        for ( int k = 0; k < nc; ++k )
            cid[ k ] = cut_id;
        unbounded  = cut_id == cell_ids::INFINITE;
        has_planes = false;
        if ( unbounded )
            planes_from_vertices();
        return true;
    }

    /// "THE WHOLE SPACE": the unit simplex, whose `D + 1` walls are marked `INFINITE`. Vertex
    /// 0 is the origin, on cuts `0 .. D-1` ( `x_c >= 0` ); vertex `n >= 1` is
    /// `e_{n-1}`, on the same ones minus `n-1`, plus cut `D` ( `sum x <= 1` ).
    HD bool init_unbounded() {
        if ( D + 1 > cap )
            return false;
        nv = D + 1;
        nc = D + 1;
        for ( int n = 0; n <= D; ++n )
            for ( int d = 0; d < D; ++d )
                v[ d ][ n ] = TK( d + 1 == n );
        for ( int r = 0; r < D; ++r ) {
            vk[ r ][ 0 ] = r;
            vn[ r ][ 0 ] = r + 1;                        // opposite `x_r >= 0`: along `e_r`
        }
        for ( int n = 1; n <= D; ++n ) {
            for ( int r = 0; r + 1 < D; ++r ) {
                const int c = r + ( r >= n - 1 );
                vk[ r ][ n ] = c;
                vn[ r ][ n ] = c + 1;                    // the vertex that also misses cut `c`
            }
            vk[ D - 1 ][ n ] = D;
            vn[ D - 1 ][ n ] = 0;                        // opposite the closing cut: the origin
        }
        for ( int k = 0; k <= D; ++k )
            cid[ k ] = cell_ids::INFINITE;
        for ( int c = 0; c < D; ++c ) {
            for ( int d = 0; d < D; ++d )
                pd[ d ][ c ] = - TK( d == c );
            po[ c ] = 0;
        }
        for ( int d = 0; d < D; ++d )
            pd[ d ][ D ] = 1;
        po[ D ] = 1;
        unbounded  = true;
        has_planes = true;
        return true;
    }

    HD void make_empty() { nv = 0; nc = 0; unbounded = false; has_planes = false; }

    // ---- the planes, read back from the geometry ---------------------------------------------------

    /// the OUTWARD normal of face `k` and its offset, read back from its vertices. `false` if the face
    /// does not span a hyperplane ( dead, or degenerate ).
    ///
    /// NOT "the first `D` vertices found": on a facet of dimension `D - 1 >= 3`, `D`
    /// vertices may well lie in the same 2-face, and two vertices coinciding to 1e-15
    /// ( a cut passed through a vertex ) are the common case. So we build an
    /// ORTHONORMAL basis of the space spanned by `p - p0`, by Gram-Schmidt, keeping only a point
    /// whose residual is clear; the normal is the generalized cross product of that basis.
    HD bool plane_of_cut( int k, TK *dir, TK &off ) const {
        int p0 = -1;
        TK  base[ D - 1 ][ D ];
        TK  scale = 0;
        int nb = 0;
        for ( int i = 0; i < nv; ++i ) {
            bool on = false;
            for ( int r = 0; r < D; ++r )
                on |= vk[ r ][ i ] == k;
            on_face[ i ] = on;
            if ( ! on )
                continue;
            if ( p0 < 0 ) { p0 = i; continue; }
            if ( nb == D - 1 )
                continue;
            TK u[ D ];
            TK n2 = 0;
            for ( int d = 0; d < D; ++d ) { u[ d ] = v[ d ][ i ] - v[ d ][ p0 ]; n2 += u[ d ] * u[ d ]; }
            if ( n2 > scale ) scale = n2;
            for ( int b = 0; b < nb; ++b ) {
                TK dot = 0;
                for ( int d = 0; d < D; ++d ) dot += u[ d ] * base[ b ][ d ];
                for ( int d = 0; d < D; ++d ) u[ d ] -= dot * base[ b ][ d ];
            }
            TK r2 = 0;
            for ( int d = 0; d < D; ++d ) r2 += u[ d ] * u[ d ];
            if ( r2 <= scale * TK( 1e-12 ) )
                continue;                                // in the space already spanned, or coincident
            const TK inv = 1 / std::sqrt( r2 );
            for ( int d = 0; d < D; ++d ) base[ nb ][ d ] = u[ d ] * inv;
            ++nb;
        }
        if ( nb < D - 1 )
            return false;

        // the generalized cross product of the `D - 1` basis vectors: `n_i` is the minor without
        // column `i`, with sign `( -1 )^i`
        for ( int i = 0; i < D; ++i ) {
            const auto M = Matrix<TK,D-1>::with_func( [&]( auto r, auto c ) {
                const int col = int( c ) + ( int( c ) >= i );
                return base[ int( r ) ][ col ];
            } );
            const TK m = M.determinant();
            dir[ i ] = ( i % 2 ) ? - m : m;
        }
        off = 0;
        for ( int d = 0; d < D; ++d )
            off += dir[ d ] * v[ d ][ p0 ];

        // outward: the vertex FARTHEST from the plane is inside ( not the first one that comes, which may be
        // a vertex coinciding with the face, 1e-16 off on either side )
        TK far = 0;
        for ( int i = 0; i < nv; ++i ) {
            if ( on_face[ i ] )
                continue;
            TK s = - off;
            for ( int d = 0; d < D; ++d )
                s += dir[ d ] * v[ d ][ i ];
            if ( ( s < 0 ? - s : s ) > ( far < 0 ? - far : far ) )
                far = s;
        }
        if ( far > 0 ) {
            for ( int d = 0; d < D; ++d )
                dir[ d ] = - dir[ d ];
            off = - off;
        }
        return true;
    }

    HD void planes_from_vertices() {
        for ( int k = 0; k < nc; ++k ) {
            TK dir[ D ], off;
            if ( ! plane_of_cut( k, dir, off ) ) {
                for ( int d = 0; d < D; ++d ) dir[ d ] = 0;
                off = 0;
            }
            for ( int d = 0; d < D; ++d )
                pd[ d ][ k ] = dir[ d ];
            po[ k ] = off;
        }
        has_planes = true;
    }

    template<class T>
    HD void plane( int k, T *dir, T &off ) const {
        if ( has_planes ) {
            for ( int d = 0; d < D; ++d ) dir[ d ] = T( pd[ d ][ k ] );
            off = T( po[ k ] );
        } else {
            TK dk[ D ], o = 0;
            if ( ! plane_of_cut( k, dk, o ) )
                for ( int d = 0; d < D; ++d ) dk[ d ] = 0;
            for ( int d = 0; d < D; ++d ) dir[ d ] = T( dk[ d ] );
            off = T( o );
        }
    }

    // ---- the cut ----------------------------------------------------------------------------

    HD int cut( const PlaneT &p ) {
        if ( unbounded )
            grow_for( p );
        return cut_impl( p );
    }

    /// THE FIRST PASS: `si[ i ] = dir . v_i - off` for every vertex, written to `out` ( if not null ),
    /// and the number of vertices strictly outside. Eight vertices per step, in registers: the
    /// carved arrays are runtime pointers, which a compiler cannot prove disjoint from `out`, so it
    /// does not vectorize the scalar loop. The tail is loaded PARTIALLY ( no byte beyond `nv`, and
    /// its dead lanes do not count ).
    HD int signed_distances( const PlaneT &p, TK *out ) const {
#ifdef __CUDACC__
        int nb_out = 0;                                  // the GPU has no register kernel: the plain loop
        for ( int i = 0; i < nv; ++i ) {
            TK si = - p.off;
            for ( int d = 0; d < D; ++d )
                si += p.dir[ d ] * v[ d ][ i ];
            if ( out )
                out[ i ] = si;
            nb_out += si > 0;
        }
        return nb_out;
#else
        using V = asimd::SimdVec<TK,8>;
        const V zero( TK( 0 ) ), moff( - p.off );
        V dir[ D ];
        for ( int d = 0; d < D; ++d )
            dir[ d ] = V( p.dir[ d ] );
        auto dist = [&]( auto load ) {
            V sv = moff;
            for ( int d = 0; d < D; ++d )
                sv = asimd::fma( dir[ d ], load( d ), sv );
            return sv;
        };
        int nb_out = 0;
        const int full = nv & ~7;
        int i = 0;
        for ( ; i < full; i += 8 ) {
            const V sv = dist( [&]( int d ) { return V::load_unaligned( v[ d ] + i ); } );
            if ( out )
                sv.store_unaligned( out + i );
            nb_out += __builtin_popcountll( asimd::to_bits( sv > zero ) );
        }
        if ( i < nv ) {
            const auto q = asimd::LaneRange<0>( nv - i );
            const V sv = dist( [&]( int d ) { return V::load_partial( v[ d ] + i, q ); } );
            if ( out )
                sv.store_partial( out + i, q );
            nb_out += __builtin_popcountll( asimd::to_bits( sv > zero, q ) );
        }
        return nb_out;
#endif
    }

    HD int nb_outside( const PlaneT &p ) const { return signed_distances( p, nullptr ); }

    HD int cut_impl( const PlaneT &p ) {
        const int nb_out = signed_distances( p, s );
        if ( nb_out == 0 )
            return CutStatus::UNCHANGED;
        if ( nb_out == nv ) {
            nv = 0; nc = 0;
            unbounded = false;
            return CutStatus::EMPTY;
        }

        if ( nc >= cap ) {
            compact();
            if ( nc >= cap )
                return CutStatus::NO_ROOM;
        }
        const int knew = nc;

        // ---- A SINGLE PASS OVER THE VERTICES: the holes left by the outside vertices and, for
        // each of them, its crossing edges -- from which the new vertices are born. `nk`: the
        // INHERITED cuts, sorted; `rec_v` / `rec_f`: the INSIDE vertex to reattach, and its slot.
        int nt = 0, nm = 0;

        for ( int o = 0; o < nv; ++o ) {
            if ( ! ( s[ o ] > 0 ) )
                continue;
            hole[ nt++ ] = o;
            for ( int j = 0; j < D; ++j ) {
                const int u = vn[ j ][ o ];
                if ( s[ u ] > 0 )
                    continue;                            // edge entirely outside: it dies
                if ( nm >= cap )
                    return CutStatus::NO_ROOM;

                // ANCHORED ON THE INSIDE VERTEX: with `s_u == 0` the symmetric form does not return `v_u`
                // in floating point, and the vertex would end up on the other side of the plane.
                const TK t = s[ u ] / ( s[ u ] - s[ o ] );
                for ( int d = 0; d < D; ++d )
                    nx[ d ][ nm ] = v[ d ][ u ] + ( v[ d ][ o ] - v[ d ][ u ] ) * t;
                for ( int r = 0, q = 0; r < D; ++r )
                    if ( r != j )
                        nk[ q++ ][ nm ] = vk[ r ][ o ];

                rec_v[ nm ] = u;
                int f = 0;
                for ( int r = 0; r < D; ++r )
                    if ( vn[ r ][ u ] == o )
                        f = r;
                rec_f[ nm ] = f;
                ++nm;
            }
        }

        const int nn = nv - nt;
        const int new_nv = nn + nm;
        if ( new_nv > cap )
            return CutStatus::NO_ROOM;

        // where each new vertex goes: into a hole while any remain, then appended
        for ( int j = 0; j < nm; ++j )
            dest[ j ] = j < nt ? hole[ j ] : nv + ( j - nt );

        // ---- THE NEIGHBORS OF THE NEW VERTICES. Opposite `knew` ( slot `D - 1` ): the inside end
        // it comes from. The others are its neighbors ON THE NEW FACE: two new vertices are
        // neighbors exactly when they share `D - 2` old cuts, and the edge joining them
        // is then opposite the inherited cut that they do NOT share.
        for ( int i = 0; i < nm; ++i ) {
            for ( int r = 0; r + 1 < D; ++r )
                nn_[ r ][ i ] = -1;
            nn_[ D - 1 ][ i ] = rec_v[ i ];
        }
        for ( int i = 0; i < nm; ++i ) {
            int ki[ D - 1 ];
            for ( int a = 0; a < D - 1; ++a )
                ki[ a ] = nk[ a ][ i ];
            for ( int j = i + 1; j < nm; ++j ) {
                // both lists are sorted and strictly increasing: the number of common cuts is the number of
                // entries of `i` found in `j`, and the entry of each that is NOT shared ( the edge's
                // opposite cut ) falls out of the same comparisons. Branch-free: a merge with a
                // data-dependent branch per step mispredicts about as often as it decides.
                int kj[ D - 1 ];
                for ( int b = 0; b < D - 1; ++b )
                    kj[ b ] = nk[ b ][ j ];
                int common = 0, ai = 0, bj = 0;
                for ( int a = 0; a < D - 1; ++a ) {
                    bool in = false;
                    for ( int b = 0; b < D - 1; ++b )
                        in |= ki[ a ] == kj[ b ];
                    common += in;
                    ai = in ? ai : a;
                }
                if ( common != D - 2 )
                    continue;
                for ( int b = 0; b < D - 1; ++b ) {
                    bool in = false;
                    for ( int a = 0; a < D - 1; ++a )
                        in |= kj[ b ] == ki[ a ];
                    bj = in ? bj : b;
                }
                nn_[ ai ][ i ] = dest[ j ];
                nn_[ bj ][ j ] = dest[ i ];
            }
        }

        // ---- COMMIT. Nothing has moved so far.
        for ( int j = 0; j < nm; ++j ) {
            const int m = dest[ j ];
            for ( int d = 0; d < D; ++d )
                v[ d ][ m ] = nx[ d ][ j ];
            for ( int r = 0; r + 1 < D; ++r )
                vk[ r ][ m ] = nk[ r ][ j ];             // < `knew`, hence sorted
            vk[ D - 1 ][ m ] = knew;
            for ( int r = 0; r < D; ++r )
                vn[ r ][ m ] = nn_[ r ][ j ];
        }
        for ( int i = 0; i < nm; ++i )                   // the reattachment, on the INSIDE vertex side
            vn[ rec_f[ i ] ][ rec_v[ i ] ] = dest[ i ];

        // ---- THE HOLES THAT REMAIN, when the cut removes more vertices than it creates
        if ( nm < nt ) {
            int th = nt;
            while ( th > nm && hole[ th - 1 ] >= new_nv ) --th;

            int nmv = 0;
            int ct = th, cd = nm;
            for ( int i = new_nv; i < nv; ++i ) {
                if ( ct < nt && hole[ ct ] == i ) { ++ct; continue; }   // this slot IS a hole
                src[ nmv ] = i;
                dst[ nmv ] = hole[ cd++ ];
                fresh[ i - new_nv ] = dst[ nmv ];
                ++nmv;
            }
            for ( int t = 0; t < nmv; ++t ) {
                const int a = src[ t ], b = dst[ t ];
                for ( int d = 0; d < D; ++d ) v[ d ][ b ] = v[ d ][ a ];
                for ( int r = 0; r < D; ++r ) { vk[ r ][ b ] = vk[ r ][ a ]; vn[ r ][ b ] = vn[ r ][ a ]; }
            }
            // and the neighbors that pointed to them; a neighbor may itself have moved
            for ( int t = 0; t < nmv; ++t ) {
                const int a = src[ t ], b = dst[ t ];
                for ( int j = 0; j < D; ++j ) {
                    const int w = vn[ j ][ b ];
                    const int q = w >= new_nv ? fresh[ w - new_nv ] : w;
                    for ( int r = 0; r < D; ++r )
                        if ( vn[ r ][ q ] == a ) { vn[ r ][ q ] = b; break; }
                }
            }
        }

        cid[ knew ] = p.id;
        if ( has_planes ) {
            for ( int d = 0; d < D; ++d )
                pd[ d ][ knew ] = p.dir[ d ];
            po[ knew ] = p.off;
        }
        nc = knew + 1;
        nv = new_nv;

        if ( unbounded ) {
            unbounded = false;
            for ( int i = 0; i < nv; ++i )
                for ( int r = 0; r < D; ++r )
                    unbounded |= cid[ vk[ r ][ i ] ] == cell_ids::INFINITE;
            if ( ! unbounded )
                has_planes = false;
        }
        return CutStatus::CUT;
    }

    /// before putting the cell into memory: dead cuts do not leave from here
    HD void tidy() { compact(); }

    /// REMOVE THE DEAD CUTS. The renumbering is MONOTONE, so the lists stay sorted.
    HD void compact() {
        for ( int k = 0; k < nc; ++k ) m[ k ] = -1;
        for ( int i = 0; i < nv; ++i )
            for ( int r = 0; r < D; ++r )
                m[ vk[ r ][ i ] ] = 0;
        int q = 0;
        for ( int k = 0; k < nc; ++k ) {
            if ( m[ k ] != 0 )
                continue;
            cid[ q ] = cid[ k ];
            if ( has_planes ) {
                for ( int d = 0; d < D; ++d ) pd[ d ][ q ] = pd[ d ][ k ];
                po[ q ] = po[ k ];
            }
            m[ k ] = q++;
        }
        for ( int i = 0; i < nv; ++i )
            for ( int r = 0; r < D; ++r )
                vk[ r ][ i ] = m[ vk[ r ][ i ] ];
        nc = q;
    }

    // ---- the replacement simplex -----------------------------------------------------------

    /// the velocity of vertex `i` when the `INFINITE` walls are pushed back: it solves the `D x D` of
    /// its cuts with the `INFINITE` indicators as right-hand side
    HD void growth_rate( int i, TK *rate ) const {
        bool any = false;
        for ( int r = 0; r < D; ++r )
            any |= cid[ vk[ r ][ i ] ] == cell_ids::INFINITE;
        if ( ! any ) {
            for ( int d = 0; d < D; ++d ) rate[ d ] = 0;
            return;
        }
        const auto A = Matrix<TK,D>::with_func( [&]( auto r, auto c ) { return pd[ int( c ) ][ vk[ int( r ) ][ i ] ]; } );
        const auto b = Vector<TK,D>::with_func( [&]( PI r ) { return TK( cid[ vk[ r ][ i ] ] == cell_ids::INFINITE ); } );
        const auto x = Matrix<TK,D>::solve_ge( A, b );
        for ( int d = 0; d < D; ++d )
            rate[ d ] = x[ d ];
    }

    HD void grow_for( const PlaneT &p ) {
        if ( ! has_planes )
            planes_from_vertices();

        static constexpr int max_rounds = 4;
        // THE MARGIN IS NOT A MACHINE EPSILON: a vertex pushed back by an epsilon falls back to the
        // precision of the dot product, and two vertices coinciding to 1e-15 then get classified
        // each on its own side of the plane -- which breaks the combinatorics of the cut ( seen in 4D ).
        // It is small compared to the geometry, large compared to rounding: where the FAKE
        // vertices land has no geometric meaning anyway.
        const TK margin = std::is_same_v<TK,float> ? TK( 1e-5 ) : TK( 1e-6 );

        for ( int i = 0; i < nv; ++i ) {
            TK r[ D ];
            growth_rate( i, r );
            for ( int d = 0; d < D; ++d )
                rate[ d ][ i ] = r[ d ];
        }

        // "the velocity does not change the distance to the plane" is judged AT A TOLERANCE, not at zero:
        // the planes are READ BACK from the geometry, so an exactly axial normal comes out with
        // components at 1e-17, and `root = - s / rate` would then make a push of 1e17 -- and 1e38 on the
        // next round. A ray parallel to the plane to within 1e-9 counts as parallel.
        TK nd = 0;
        for ( int d = 0; d < D; ++d )
            nd += p.dir[ d ] * p.dir[ d ];
        const TK tol = TK( 1e-9 ) * std::sqrt( nd );

        TK g = 0;
        for ( int round = 0; round < max_rounds; ++round ) {
            bool push = false;
            TK grow = 0;
            for ( int i = 0; i < nv; ++i ) {
                TK dr = 0, s = - p.off, nr = 0;
                for ( int d = 0; d < D; ++d ) {
                    dr += p.dir[ d ] * rate[ d ][ i ];
                    s  += p.dir[ d ] * ( v[ d ][ i ] + g * rate[ d ][ i ] );
                    nr += rate[ d ][ i ] * rate[ d ][ i ];
                }
                if ( nr == 0 || ( dr < 0 ? - dr : dr ) <= tol * std::sqrt( nr ) )
                    continue;
                const TK root = - s / dr;
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

        for ( int k = 0; k < nc; ++k )
            if ( cid[ k ] == cell_ids::INFINITE )
                po[ k ] += g;
        for ( int i = 0; i < nv; ++i )
            for ( int d = 0; d < D; ++d )
                v[ d ][ i ] += g * rate[ d ][ i ];
    }

    // ---- the measure: a fan of simplices over the face lattice ----------------------

    HD bool has_cut( int i, int k ) const {
        for ( int r = 0; r < D; ++r )
            if ( vk[ r ][ i ] == k )
                return true;
        return false;
    }

    /// `func( chain )` for each simplex of the standard triangulation: a vertex of the cell,
    /// coned over each facet that does not contain it, each triangulated the same way one dimension
    /// lower. For each face encountered we need ONE of its vertices ( "the apex" ): `apex` keeps
    /// one row per depth and one slot per cut, filled in a single pass over the vertices
    /// of the face -- `D * nc` words, where indexing the faces by their SET of cuts costs
    /// `nc^( D - 1 )`.
    HD void for_each_simplex( auto &&func ) const {
        if ( nv == 0 )
            return;
        Vector<SI,D+1> chain;
        Vector<SI,D> face_cuts;
        chain[ 0 ] = 0;
        for_each_simplex_rec( chain, face_cuts, func, Ct<int,D>() );
    }

    template<int K>
    HD void for_each_simplex_rec( Vector<SI,D+1> &chain, Vector<SI,D> &face_cuts, auto &&func, Ct<int,K> ) const {
        constexpr int depth = D - K;                     // the number of cuts that define the face
        if constexpr ( K == 0 ) {
            func( chain );
        } else {
            const SI p = chain[ depth ];                 // the apex of the face, fixed for this subtree
            for ( int c = 0; c < nc; ++c )
                apex[ depth ][ c ] = -1;
            for ( int i = 0; i < nv; ++i ) {
                bool on = true;
                for ( int m = 0; m < depth; ++m )
                    on &= has_cut( i, int( face_cuts[ m ] ) );
                if ( ! on )
                    continue;
                for ( int r = 0; r < D; ++r ) {
                    const int c = vk[ r ][ i ];
                    bool in = false;
                    for ( int m = 0; m < depth; ++m )
                        in |= int( face_cuts[ m ] ) == c;
                    if ( ! in && apex[ depth ][ c ] < 0 )
                        apex[ depth ][ c ] = i;
                }
            }
            // ... then cone `p` over the facets that do NOT contain it ( the others would give
            // flat simplices )
            for ( int c = 0; c < nc; ++c ) {
                const int a = apex[ depth ][ c ];
                if ( a < 0 || has_cut( int( p ), c ) )
                    continue;
                face_cuts[ depth ] = c;
                chain[ depth + 1 ] = a;
                for_each_simplex_rec( chain, face_cuts, func, Ct<int,K-1>() );
            }
        }
    }

    /// IN 3D: ACCUMULATE THE FACES RATHER THAN ORDER THEM. Neither the volume nor the area of a face
    /// needs the ORDER of the vertices: it is enough, per face, to have ONE of its vertices `v_f` and the sum
    /// `S_f` of the cross products of its edges seen from it -- the face is planar and convex,
    /// so the triangles `( v_f, edge )` tile it and their cross products are parallel.
    /// The volume is `sum_f | ( v_f - g ) . S_f | / 6` for any interior `g`. Two
    /// passes without ever searching: `O( V + E )` instead of `O( F ( V + E ) )` for the cycles --
    /// measured on the bench: -15 % on the whole 3D diagram.
    /// IN 3D: per cut `f`, ONE vertex `v0[ f ]` and the sum `s3[ . ][ f ]` of the cross products
    /// of its edges seen from it -- twice the area vector of the face. What `measure_3d` and
    /// `for_each_facet` both read ( see `measure_3d` for why to accumulate rather than order ).
    HD void accumulate_faces_3d() const {
        static_assert( D == 3 );
        TK *const *s = s3;
        for ( int k = 0; k < nc; ++k ) { v0[ k ] = -1; s[ 0 ][ k ] = s[ 1 ][ k ] = s[ 2 ][ k ] = 0; }
        for ( int i = nv - 1; i >= 0; --i )
            for ( int r = 0; r < 3; ++r )
                v0[ vk[ r ][ i ] ] = i;

        for ( int a = 0; a < nv; ++a ) {
            for ( int j = 0; j < 3; ++j ) {
                const int b = vn[ j ][ a ];
                if ( b <= a )
                    continue;                            // each edge seen only once
                // the two faces that carry edge `j`: the cuts of the vertex minus the `j`-th
                for ( int r = 0; r < 3; ++r ) {
                    if ( r == j )
                        continue;
                    const int f = vk[ r ][ a ];
                    const int o = v0[ f ];
                    const TK ax = v[0][a] - v[0][o], ay = v[1][a] - v[1][o], az = v[2][a] - v[2][o];
                    const TK bx = v[0][b] - v[0][o], by = v[1][b] - v[1][o], bz = v[2][b] - v[2][o];
                    TK cx = ay * bz - az * by, cy = az * bx - ax * bz, cz = ax * by - ay * bx;
                    if ( s[0][f] * cx + s[1][f] * cy + s[2][f] * cz < 0 ) { cx = -cx; cy = -cy; cz = -cz; }
                    s[0][f] += cx; s[1][f] += cy; s[2][f] += cz;
                }
            }
        }
    }

    /// `func( c, measure )` for each cut `c` that carries a face: its AREA ( 3D only )
    template<class TF>
    HD void for_each_facet( auto &&func ) const {
        static_assert( D == 3, "for_each_facet: 3D only beyond the plane" );
        if ( nv < 4 )
            return;
        accumulate_faces_3d();
        TK *const *s = s3;
        for ( int k = 0; k < nc; ++k ) {
            if ( v0[ k ] < 0 )
                continue;
            const TF sx = TF( s[0][k] ), sy = TF( s[1][k] ), sz = TF( s[2][k] );
            func( k, sdot::sqrt( sx * sx + sy * sy + sz * sz ) / 2 );
        }
    }

    template<class TF>
    HD TF measure_3d() const {
        static_assert( D == 3 );
        if ( nv < 4 )
            return 0;
        // the accumulation is done in the KERNEL's float ( the `s3` arrays are `TK` );
        // the final sum is in `TF`
        accumulate_faces_3d();
        TK *const *s = s3;

        TF g[ 3 ] = { 0, 0, 0 };
        for ( int i = 0; i < nv; ++i )
            for ( int d = 0; d < 3; ++d )
                g[ d ] += TF( v[ d ][ i ] );
        for ( int d = 0; d < 3; ++d )
            g[ d ] /= nv;

        TF vol = 0;
        for ( int k = 0; k < nc; ++k ) {
            const int o = v0[ k ];
            if ( o < 0 )
                continue;                                // dead cut, not yet compacted
            const TF t = ( TF( v[0][o] ) - g[0] ) * TF( s[0][k] ) + ( TF( v[1][o] ) - g[1] ) * TF( s[1][k] ) + ( TF( v[2][o] ) - g[2] ) * TF( s[2][k] );
            vol += t < 0 ? -t : t;
        }
        return vol / 6;
    }

    template<class TF>
    HD TF measure() const {
        if ( unbounded )
            return std::numeric_limits<TF>::max();
        if constexpr ( D == 3 )
            return measure_3d<TF>();
        TF sum = 0;
        for_each_simplex( [&]( const auto &chain ) {
            const auto M = Matrix<TF,D>::with_func( [&]( auto r, auto c ) {
                return TF( v[ int( r ) ][ chain[ int( c ) + 1 ] ] ) - TF( v[ int( r ) ][ chain[ 0 ] ] );
            } );
            const TF det = M.determinant();
            sum += det < 0 ? - det : det;
        } );
        TF fact = 1;
        for ( int i = 2; i <= D; ++i )
            fact *= i;
        return sum / fact;
    }

    /// the adjoint: `grad_vp( i, d )` ACCUMULATES ( a vertex is in several simplices ), so it
    /// is zeroed first, over the `nv` vertices.
    template<class TF>
    HD void measure_bwd( TF grad_res, auto &&grad_vp ) const {
        for ( int i = 0; i < nv; ++i )
            for ( int d = 0; d < D; ++d )
                grad_vp( i, d ) = 0;
        if ( unbounded )
            return;

        TF fact = 1;
        for ( int i = 2; i <= D; ++i )
            fact *= i;
        const TF g = grad_res / fact;

        // `d|det|/dM = sign( det ) * cofactor( M )`, and each column of `M` is an apex minus the
        // first, which therefore receives MINUS the sum of the columns
        for_each_simplex( [&]( const auto &chain ) {
            const auto M = Matrix<TF,D>::with_func( [&]( auto r, auto c ) {
                return TF( v[ int( r ) ][ chain[ int( c ) + 1 ] ] ) - TF( v[ int( r ) ][ chain[ 0 ] ] );
            } );
            const TF det = M.determinant();
            const TF sg = det < 0 ? - g : g;
            for ( int r = 0; r < D; ++r ) {
                TF row_sum = 0;
                for ( int c = 0; c < D; ++c ) {
                    const TF minor = M.without_row_and_col( r, c ).determinant();
                    const TF cof = ( ( r + c ) % 2 ? - minor : minor ) * sg;
                    grad_vp( chain[ c + 1 ], r ) += cof;
                    row_sum += cof;
                }
                grad_vp( chain[ 0 ], r ) -= row_sum;
            }
        } );
    }

    HD void bbox( TK *lo, TK *hi ) const {
        for ( int d = 0; d < D; ++d )
            lo[ d ] = hi[ d ] = nv ? v[ d ][ 0 ] : TK( 0 );
        for ( int i = 1; i < nv; ++i )
            for ( int d = 0; d < D; ++d ) {
                lo[ d ] = v[ d ][ i ] < lo[ d ] ? v[ d ][ i ] : lo[ d ];
                hi[ d ] = v[ d ][ i ] > hi[ d ] ? v[ d ][ i ] : hi[ d ];
            }
    }

    // ---- the tensors ------------------------------------------------------------------------

    /// from a `Cell_N` view: `vertex_positions [ nv, D ]`, `vertex_cuts` / `vertex_nbrs`
    /// `[ nv, D ]`, `cut_ids [ nc ]`. `false` if it does not fit in `cap`.
    HD bool load( const auto &c ) {
        const int n = int( SI( c.nb_vertices ) ), k = int( SI( c.nb_cuts ) );
        if ( n > cap || k > cap )
            return false;
        nv = n; nc = k;
        for ( int i = 0; i < nv; ++i )
            for ( int d = 0; d < D; ++d ) {
                v [ d ][ i ] = TK( c.vertex_positions( i, d ) );
                vk[ d ][ i ] = int( SI( c.vertex_cuts( i, d ) ) );
                vn[ d ][ i ] = int( SI( c.vertex_nbrs( i, d ) ) );
            }
        for ( int q = 0; q < nc; ++q )
            cid[ q ] = int( SI( c.cut_ids( q ) ) );
        // over the vertices and not over the list: a DEAD cut ( with no vertex ) may linger there
        // until the next compaction, and a dead wall bounds nothing
        unbounded = false;
        for ( int i = 0; i < nv; ++i )
            for ( int r = 0; r < D; ++r )
                unbounded |= cid[ vk[ r ][ i ] ] == cell_ids::INFINITE;
        has_planes = false;
        return true;
    }

    HD bool store( auto &&c ) const {
        if ( ! c.nb_vertices.set( nv ) )
            return false;
        if ( ! c.nb_cuts.set( nc ) )
            return false;
        for ( int i = 0; i < nv; ++i )
            for ( int d = 0; d < D; ++d ) {
                c.vertex_positions( i, d ) = v [ d ][ i ];
                c.vertex_cuts     ( i, d ) = vk[ d ][ i ];
                c.vertex_nbrs     ( i, d ) = vn[ d ][ i ];
            }
        for ( int k = 0; k < nc; ++k )
            c.cut_ids( k ) = cid[ k ];
        return true;
    }
};

} // namespace sdot
