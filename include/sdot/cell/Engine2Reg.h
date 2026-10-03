#pragma once


// =====================================================================================
// THE REGISTER KERNEL -- 2D, bounded cell, CPU.
//
// The cell there is three vectors of eight lanes:
//
//      vx[ 0 .. 7 ]    the vertices, in cyclic order
//      vy[ 0 .. 7 ]
//      cid[ 0 .. 7 ]   the identity of the cut that carries the edge `[ v_i, v_i+1 ]`
//
// and `NB`, the number of vertices, is a COMPILE-TIME CONSTANT: it is what makes the valid-lane
// mask immediate, the rotations immediate, and the loop absent. Eight lanes because
// that is the window the statistics point to: 98.4 % of the intermediate states of a
// Voronoi diagram have eight vertices or fewer.
//
// WHAT MAKES THE CUT LOOP-FREE. The outside of a convex set cut by a half-plane is a contiguous
// CYCLIC range. The sign mask is thus, up to a rotation, a block of ones: its two
// ends are read with two `ctz` on rotations of the mask, and with `NB` known these rotations
// are immediates. There is nothing left to walk through.
//
// THE STATE MACHINE. `nb` changes at almost every effective cut ( `nb - nb_out + 2` ), so
// each size is a function `step<NB>` ( `step` = step ) that cuts as long as the size does not change and RETURNS the
// new size otherwise; `run_2d_registers` redispatches through a `switch`. `step` is
// `always_inline`, and this is not negotiable: not inlined, it would take the cell by
// reference across a real call, hence through MEMORY, and the three vectors would cease
// to be registers. ( The bench's `musttail` form was measured equivalent, and it is not
// portable. )
//
// OVERFLOW IS AN EXCURSION. From `NB == 8`, a cut that removes only one vertex would
// need a ninth. The cell is then put down in `Local2` -- which IS the workshop -- and we
// keep cutting it in scalar, in place, until it drops back to eight vertices: it
// is then reloaded into the registers and we start again. The kernel thus always returns a
// complete cell, and `NO_ROOM` only remains if the scratch capacity is too small.
//
// THE ASIMD DICTIONARY is enough: `fma`, `permute`, `select`, `to_bits`, `mask_from_bits`,
// `bcast_lane`. The width is a parameter of the type, so `SimdVec<double,8>` is the same code
// on two registers -- which makes the `double` kernel come without one more line.
// =====================================================================================

#include <asimd/asimd.h>
#include "Local2.h"
#include "State.h"

namespace sdot {

/// WHAT THE SUPPLIER SEES when the cell fits in the registers. `nb` is a compile-time
/// constant; `x / y / id` extract ONE lane by going through the stack -- this is expensive, and
/// intended: a supplier that does not look at the cell pays nothing ( the `store` is dead, the
/// compiler removes it ), a supplier that looks at it pays what it really costs.
template<class TK,int NB>
struct StateReg {
    using V  = asimd::SimdVec<TK,8>;
    using VI = asimd::SimdVec<asimd::SI32,8>;
    static constexpr int nb = NB;

    V  vx, vy;
    VI cid;

    TK  x ( int i ) const { alignas( 64 ) TK t[ 8 ]; vx.store_aligned( t ); return t[ i ]; }
    TK  y ( int i ) const { alignas( 64 ) TK t[ 8 ]; vy.store_aligned( t ); return t[ i ]; }
    int id( int i ) const { alignas( 64 ) asimd::SI32 t[ 8 ]; cid.store_aligned( t ); return t[ i ]; }
};

namespace engine2 {

enum : int {
    FINISHED    = -2,   ///< the supplier has nothing left: the cell is finished, at `NB` vertices
    OVERFLOWED = -1,   ///< beyond eight: `Local2` carries the eight vertices, and `pending` the cut
    EMPTY    =  0,   ///< a half-plane took everything away
};

/// "lane i", by a BIT pattern: one `kmovb` on a machine with mask registers, whereas
/// `eq( iota, i )` needs a `vpbroadcastd` and a `vpcmpeqd`. Three lanes are designated at
/// every cut.
inline auto lane( int i ) { return asimd::mask_from_bits<8>( 1u << i ); }

/// ONE STEP: cuts at fixed `NB` as long as the size does not change, returns the new size otherwise.
template<int NB,class TK,class Provider>
[[gnu::always_inline]] inline int step( asimd::SimdVec<TK,8> &vx, asimd::SimdVec<TK,8> &vy,
                                         asimd::SimdVec<asimd::SI32,8> &cid,
                                         Provider &f, Local2<TK> &a, LocalOf<Provider> &loc,
                                         Plane<TK,2> &pending ) {
    using V  = asimd::SimdVec<TK,8>;
    using VI = asimd::SimdVec<asimd::SI32,8>;
    constexpr unsigned valid = ( 1u << NB ) - 1;
    const VI IOTA = VI::iota( 0 );

    for ( ;; ) {
        Plane<TK,2> p;
        if ( ! f.next( StateReg<TK,NB>{ vx, vy, cid }, loc, p ) )
            return FINISHED;

        // ---- THE TEST, WHICH IS ALREADY THE CUT: `s > 0` is outside.
        const V s = asimd::fma( V( p.dir[ 0 ] ), vx, asimd::fma( V( p.dir[ 1 ] ), vy, V( - p.off ) ) );
        const unsigned m = unsigned( asimd::to_bits( s > V( TK( 0 ) ) ) ) & valid;

        if ( ! m )
            continue;
        if ( m == valid )
            return EMPTY;

        // ---- THE TWO ENDS OF THE OUTSIDE RANGE.
        const unsigned prev = ( ( m << 1 ) | ( m >> ( NB - 1 ) ) ) & valid;
        const unsigned next = ( ( m >> 1 ) | ( m << ( NB - 1 ) ) ) & valid;
        const int i1 = __builtin_ctz( m & ~prev );          // first OUTSIDE of the range
        const int j2 = __builtin_ctz( m & ~next );          // last OUTSIDE
        const int j0 = i1 ? i1 - 1 : NB - 1;                // last INSIDE before
        const int j3 = j2 + 1 < NB ? j2 + 1 : 0;            // first INSIDE after
        const int nb_in = NB - __builtin_popcount( m );
        const int nn = nb_in + 2;

        // ---- THE TWO INTERSECTIONS, IN A SINGLE DIVISION. A in lane 0, B in lane 1: the two
        // anchors ( the INSIDE vertices ) are in distinct lanes, hence no collision when
        // `j0 == j3`.
        const auto v1 = lane( 1 );
        const VI anc = asimd::select( v1, VI( j3 ), VI( j0 ) );
        const VI oth = asimd::select( v1, VI( j2 ), VI( i1 ) );
        const V vax = asimd::permute( vx, anc ), vox = asimd::permute( vx, oth );
        const V vay = asimd::permute( vy, anc ), voy = asimd::permute( vy, oth );
        const V sa  = asimd::permute( s,  anc ), so  = asimd::permute( s,  oth );
        const V t   = sa / ( sa - so );
        const V pcx = asimd::fma( vox - vax, t, vax );
        const V pcy = asimd::fma( voy - vay, t, vay );

        // ---- THE REASSEMBLY: `[ v_j3, ..., v_j0, A, B ]`, `A` on the new cut, `B` on what
        // remains of the cut `j2`.
        VI og = VI( j3 ) + IOTA;
        og = asimd::select( asimd::ge( og, VI( NB ) ), og - VI( NB ), og );
        og = og & VI( 7 );

        const auto mA = lane( nb_in ), mB = lane( nb_in + 1 );

        V nvx = asimd::permute( vx, og );
        nvx = asimd::select( mA, asimd::bcast_lane<0>( pcx ), nvx );
        nvx = asimd::select( mB, asimd::bcast_lane<1>( pcx ), nvx );
        V nvy = asimd::permute( vy, og );
        nvy = asimd::select( mA, asimd::bcast_lane<0>( pcy ), nvy );
        nvy = asimd::select( mB, asimd::bcast_lane<1>( pcy ), nvy );
        VI nid = asimd::permute( cid, og );
        nid = asimd::select( mA, VI( p.id ), nid );
        nid = asimd::select( mB, asimd::permute( cid, VI( j2 ) ), nid );

        // ---- THE NEW STATE.
        if constexpr ( NB == 8 ) if ( nn > 8 ) {
            a.nb = 8;                                    // the excursion will start from there
            pending = p;                                 // ... and will replay this cut
            vx.store_unaligned( a.vx );
            vy.store_unaligned( a.vy );
            cid.store_unaligned( reinterpret_cast<asimd::SI32 *>( a.cid ) );
            return OVERFLOWED;
        }
        vx = nvx; vy = nvy; cid = nid;
        if ( nn != NB )
            return nn;
    }
}

/// THE EXCURSION, in place in `Local2`, until the cell drops back to eight vertices.
/// Returns the new size ( and reloads the registers ), `EMPTY`, `FINISHED` -- or `NO_ROOM` of
/// `CutStatus` ( > 8 ) if the capacity is not enough.
template<class TK,class Provider>
inline int excursion( asimd::SimdVec<TK,8> &vx, asimd::SimdVec<TK,8> &vy,
                      asimd::SimdVec<asimd::SI32,8> &cid,
                      Provider &f, Local2<TK> &a, LocalOf<Provider> &loc, Plane<TK,2> p ) {
    using V  = asimd::SimdVec<TK,8>;
    using VI = asimd::SimdVec<asimd::SI32,8>;
    for ( ;; ) {
        const int r = a.cut( p );
        if ( r == CutStatus::EMPTY )
            return EMPTY;
        if ( r == CutStatus::NO_ROOM )
            return CutStatus::NO_ROOM + 8;              // outside `3..8`, and outside the codes < 0
        if ( a.nb <= 8 ) {
            vx  = V::load_unaligned( a.vx );
            vy  = V::load_unaligned( a.vy );
            cid = VI::load_unaligned( reinterpret_cast<const asimd::SI32 *>( a.cid ) );
            return a.nb;
        }
        if ( ! f.next( a.state(), loc, p ) )
            return FINISHED;                                 // finished, with more than eight sides
    }
}

} // namespace engine2

/// THE KERNEL, restarting from the cell `c` as it is. Returns `0` ( complete cell, empty
/// included ) or `CutStatus::NO_ROOM`.
template<class TK,class Provider>
int run_2d_registers( Local2<TK> &c, Provider &f, LocalOf<Provider> &loc ) {
    using namespace engine2;
    using V  = asimd::SimdVec<TK,8>;
    using VI = asimd::SimdVec<asimd::SI32,8>;

    int nb = c.nb;
    if ( nb <= 0 )
        return 0;
    if ( c.cap < 8 )
        return CutStatus::NO_ROOM;                      // the workshop must receive the eight lanes

    V  vx = V( TK( 0 ) ), vy = V( TK( 0 ) );
    VI cid = VI( 0 );
    Plane<TK,2> pending;
    bool in_excursion = nb > 8;
    if ( ! in_excursion ) {
        // the eight lanes are loaded in full: beyond `nb` they are the scratch's temporaries,
        // never read by a valid lane
        vx  = V::load_unaligned( c.vx );
        vy  = V::load_unaligned( c.vy );
        cid = VI::load_unaligned( reinterpret_cast<const asimd::SI32 *>( c.cid ) );
    } else {
        // too big for the registers from the start: we ask for a first cut in memory
        if ( ! f.next( c.state(), loc, pending ) )
            return 0;
    }

    for ( ;; ) {
        int r;
        if ( in_excursion ) {
            r = excursion( vx, vy, cid, f, c, loc, pending );
            in_excursion = false;
        } else {
            switch ( nb ) {
                case 3:  r = step<3>( vx, vy, cid, f, c, loc, pending ); break;
                case 4:  r = step<4>( vx, vy, cid, f, c, loc, pending ); break;
                case 5:  r = step<5>( vx, vy, cid, f, c, loc, pending ); break;
                case 6:  r = step<6>( vx, vy, cid, f, c, loc, pending ); break;
                case 7:  r = step<7>( vx, vy, cid, f, c, loc, pending ); break;
                default: r = step<8>( vx, vy, cid, f, c, loc, pending ); break;
            }
        }

        if ( r == FINISHED ) {
            if ( c.nb <= 8 || nb <= 8 ) {                // the cell is in the registers
                c.nb = nb;
                vx.store_unaligned( c.vx );
                vy.store_unaligned( c.vy );
                cid.store_unaligned( reinterpret_cast<asimd::SI32 *>( c.cid ) );
            }                                            // otherwise it is already in `c`
            return 0;
        }
        if ( r == EMPTY ) {
            c.nb = 0;
            return 0;
        }
        if ( r == OVERFLOWED ) {
            in_excursion = true;
            nb = 9;
            continue;
        }
        if ( r > 8 )
            return CutStatus::NO_ROOM;
        nb = r;
    }
}

} // namespace sdot
