#pragma once

#include <loom/support/common_macros.h> // HD

// =====================================================================================
// THE PROVIDERS OF A POWER DIAGRAM -- "which half-space to cut now?"
//
// The engine ( `Engine.h` ) knows neither seed nor tree: it draws planes. A provider is an
// object with one method, `next( state, local, plan )`, that fills the plane and returns `true`, or
// returns `false` when it has nothing left. Two here:
//
//   `ProviderAll`   all the other seeds, in order. No acceleration -- this is what
//                       `make_cell` does when it is not given an accelerator, and it is the
//                       FLOOR against which the others are measured.
//   `ProviderBsp`    the BSP tree TURNED INSIDE OUT: a descent that would PUSH its candidates becomes
//                       a SUSPENDED traversal whose explicit stack lives in the `Local` that the
//                       engine houses per cell. Each `next` resumes where the previous one
//                       stopped: preorder, the child closest to the seed first, and a
//                       pruning that sees the cell AS IT HAS BECOME, cut after cut.
//
// THE BISECTOR is computed in the positions' float and returned in the kernel's
// ( `bisector`, `Plane.h` ): this is the only place where the two precisions meet.
// =====================================================================================

#include <loom/support/common_types.h>
#include "Pruning.h"
#include "Plane.h"
#include "State.h"

namespace sdot {

/// ALL THE OTHER SEEDS, in storage order.
template<class PD,class TK,int D>
struct ProviderAll {
    using TF = typename PD::TF;

    const PD &pd;
    SI  n, k0, k = 0;
    TF  p0[ D ], w0;

    HD ProviderAll( const PD &pd, SI k0 ) : pd( pd ), n( pd.nb_seeds() ), k0( k0 ) {
        const auto p = pd.point( k0 );
        for ( int d = 0; d < D; ++d )
            p0[ d ] = p[ d ];
        w0 = pd.weight( k0 );
    }

    template<class State>
    HD bool next( const State &, NothingLocal &, Plane<TK,D> &p ) {
        if ( k == k0 ) ++k;                              // we do not cut ourselves
        if ( k >= n ) return false;
        const SI j = k++;
        const auto pj = pd.point( j );
        TF p1[ D ];
        for ( int d = 0; d < D; ++d )
            p1[ d ] = pj[ d ];
        p = bisector<TK,D>( p0, w0, p1, pd.weight( j ), int( j ) );
        return true;
    }
};

/// THE BSP TREE, TRAVERSED ON DEMAND. `pd.tree` is the tree ( `AaBsp.py` ), and the storage's seeds
/// are IN ITS ORDER: the leaf `[ begin, end )` is read in one piece through
/// `pd.point( k )`, and the identifier of a cut is that rank `k`. `WEIGHTED`: Laguerre diagram
/// ( the affine majorant of the weights enters the pruning ) -- a compile-time constant, so that
/// the Euclidean case pays for neither the slopes nor the `a . y` terms.
///
/// `MEMO`: THE MEMORY ( `PowerDiagram_Bsp.memo_nbrs / memo_counts` ). The storage remembers,
/// per seed, the RANKS of the neighbors of its cell at the last `measures` ( sorted increasing ); they
/// are proposed FIRST, before any descent, then the ordinary traversal SKIPS them ( no
/// plane twice: `cut` is not idempotent ). What this spares is not the boxes --
/// a box that contains a true neighbor passes the pruning whatever happens -- it is the
/// TRANSIENT cuts, those that a nearby seed makes before a true neighbor supplants it: half
/// of the effective cuts in 3D, each one a polytope update. Measured on the bench
/// ( `solvers_des_familles`, README § 11 ): -25 to -42 % of the 3D diagram, and stale
/// memories -- those of Voronoi on a Laguerre diagram -- still yield -18 %: a false
/// memory only costs one first pass. A seed without memory ( `memo_counts == 0` ) takes
/// the ordinary path, up to the comparison.
template<class PD,class TK,int D,bool WEIGHTED,bool MEMO = false>
struct ProviderBsp {
    using TF = typename PD::TF;

    /// THE STATE OF THE TRAVERSAL, one per cell, housed in the engine's frame. The stack is bounded by the
    /// DEPTH of the tree, not by its size: at each level one node is popped and two are
    /// pushed. 64 levels are worth 2^64 seeds.
    struct Local {
        SI   stack[ 64 ];
        int  top = 0;
        SI   k = 0, stop = 0;                             ///< the slice of the open leaf
        bool started = false;
        int  ipre = 0;                                   ///< MEMO: where the memories pre-pass stands
        int  iskip = 0;                                  ///< MEMO: the skip cursor in the open leaf
    };

    const PD &pd;
    SI  k0;
    SI  depth;                                           ///< `nb_nodes == 2^depth - 1`
    int npre = 0;                                        ///< MEMO: how many memories for `k0`
    TF  p0[ D ], w0;
    TK  q0[ D ], v0;                                     ///< the same, for the pruning

    HD ProviderBsp( const PD &pd, SI k0 ) : pd( pd ), k0( k0 ) {
        const auto p = pd.point( k0 );
        for ( int d = 0; d < D; ++d ) {
            p0[ d ] = p[ d ];
            q0[ d ] = TK( p[ d ] );
        }
        w0 = pd.weight( k0 );
        v0 = TK( w0 );

        depth = 0;
        for ( SI m = SI( pd.tree.node_begin.shape( 0 ) ); m; m >>= 1 )
            ++depth;

        if constexpr ( MEMO )
            npre = int( pd.memo_counts( k0 ) );
    }

    /// MEMO: the `q`-th memory of `k0`, a rank
    HD SI remembered( int q ) const {
        if constexpr ( MEMO ) return SI( pd.memo_nbrs( k0, q ) );
        else return 0;
    }

    HD void plane_of_rank( SI k, Plane<TK,D> &p ) const {
        const auto pj = pd.point( k );
        TF p1[ D ];
        for ( int d = 0; d < D; ++d )
            p1[ d ] = pj[ d ];
        p = bisector<TK,D>( p0, w0, p1, pd.weight( k ), int( k ) );
    }

    /// the node's box, and the majorant of its weights, in the kernel's float
    HD Box<TK,D> box( SI n ) const {
        Box<TK,D> B;
        for ( int d = 0; d < D; ++d ) {
            B.lo[ d ] = TK( pd.tree.node_box( n, 0, d ) );
            B.hi[ d ] = TK( pd.tree.node_box( n, 1, d ) );
            B.a[ d ] = 0;
        }
        B.b = 0;
        if constexpr ( WEIGHTED ) {
            for ( int d = 0; d < D; ++d )
                B.a[ d ] = TK( pd.tree.node_wa( n, d ) );
            B.b = TK( pd.tree.node_wb( n ) );
        }
        return B;
    }

    /// the squared distance from the seed to the box of node `n` -- an ORDERING KEY only, to
    /// visit the closest child first, hence the cuts that bite most first.
    HD TF proximity( SI n ) const {
        TF res = 0;
        for ( int d = 0; d < D; ++d ) {
            const TF lo = TF( pd.tree.node_box( n, 0, d ) ), hi = TF( pd.tree.node_box( n, 1, d ) );
            // branch-free ( it is the unpredictable one of the descent ): the distance to `[ lo, hi ]`
            TF e = lo - p0[ d ];
            const TF f = p0[ d ] - hi;
            e = f > e ? f : e;
            e = e > 0 ? e : TF( 0 );
            res += e * e;
        }
        return res;
    }

    template<class State>
    HD bool next( const State &e, Local &l, Plane<TK,D> &p ) {
        if ( ! l.started ) {                              // the root: index 0, height `depth`
            l.stack[ l.top++ ] = depth;
            l.started = true;
        }

        if constexpr ( MEMO ) {                          // the pre-pass: yesterday's neighbors, without descending
            if ( l.ipre < npre ) {
                plane_of_rank( remembered( l.ipre++ ), p );
                return true;
            }
        }

        for ( ;; ) {
            // ---- a leaf is open: we return the next seed of its slice
            while ( l.k < l.stop ) {
                const SI k = l.k++;
                if ( k == k0 )
                    continue;
                if constexpr ( MEMO ) {                  // already proposed? the cursor advances with `k`
                    while ( l.iskip < npre && remembered( l.iskip ) < k ) ++l.iskip;
                    if ( l.iskip < npre && remembered( l.iskip ) == k ) { ++l.iskip; continue; }
                }
                plane_of_rank( k, p );
                return true;
            }

            if ( l.top == 0 )
                return false;                            // the tree is exhausted

            // `h` travels on the stack, packed with the index ( six bits suffice )
            const SI en = l.stack[ --l.top ];
            const SI n  = en >> 6;
            const SI h  = en & 63;

            // an EMPTY slot: the right child of a node that had nothing left to share
            const SI beg = SI( pd.tree.node_begin( n ) );
            const SI end = SI( pd.tree.node_end( n ) );
            if ( beg >= end )
                continue;

            // the whole point of the tree: a subtree that can no longer reach the cell is
            // not descended. Tested at POP time, hence against the cell as it is now.
            if ( ! can_cut<WEIGHTED,TK,D>( e, q0, v0, box( n ) ) )
                continue;

            if ( h <= 1 ) {                              // a leaf
                l.k = beg;
                l.stop = end;
                if constexpr ( MEMO ) {                  // the skip cursor: the first memory >= beg
                    int lo = 0, hi = npre;
                    while ( lo < hi ) { const int m = ( lo + hi ) / 2; if ( remembered( m ) < beg ) lo = m + 1; else hi = m; }
                    l.iskip = lo;
                }
                continue;
            }

            // the children are DEDUCED: the tree is perfect binary in preorder, the left one is right
            // next to it and the right one is `2^( h - 1 )` nodes away. The closest is pushed LAST.
            const SI lc = n + 1, rc = n + ( SI( 1 ) << ( h - 1 ) ), hc = h - 1;
            if ( proximity( lc ) <= proximity( rc ) ) {
                l.stack[ l.top++ ] = ( rc << 6 ) | hc;
                l.stack[ l.top++ ] = ( lc  << 6 ) | hc;
            } else {
                l.stack[ l.top++ ] = ( lc  << 6 ) | hc;
                l.stack[ l.top++ ] = ( rc << 6 ) | hc;
            }
        }
    }
};

} // namespace sdot
