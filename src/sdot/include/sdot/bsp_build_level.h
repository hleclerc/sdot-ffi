#pragma once

#include <loom/support/math.h>
#include <loom/support/common_macros.h>
#include <loom/support/containers/Vector.h>

namespace sdot {

// ONE LEVEL of the BSP construction, for ONE node -- see `AaBsp.py` for what a node
// carries, and why the shape of the tree does not depend on the data.
//
// = Why a level and not the whole tree
//
// A level cannot start before the previous one is finished (it reads the slices it
// produces), and there is no GLOBAL barrier in a kernel -- only within a
// work-group. The barrier is therefore the END OF THE LAUNCH: the host chains `depth` calls, which
// is the usual pattern on GPU and costs only `depth` launches (about fifteen at 1e6 seeds).
//
// None of this has a capacity to guess: `depth` is `max_depth_for( n, leaf_size )`, a
// function of `n` alone, so the loop unrolls under a `jit` like anywhere else.
//
// = What a work-item does, and what makes the writes disjoint
//
// One work-item per node of the level. The slices `[ begin, end )` of a level PARTITION
// `[ 0, n )` -- this is the reason for the propagation described below (`mid = end`) -- so
// two work-items never write the same cell, neither in the output cloud nor in the permutation
// scratch. No atomics, no barrier.
//
// The cloud is DOUBLE-BUFFERED (`pos_in` -> `pos_out`) because the inputs and outputs of a
// call are disjoint (see `driver.call`), and it carries the PERMUTED positions and weights
// next to the indices: a node then reads its points in one piece, where an indirection through
// `seed_indices` would make it a sparse gather -- which matters all the more since the top
// levels are handled by very few work-items.


// Quickselect (Hoare, median-of-three pivot): rearranges `perm[ b .. e )` so that the rank
// `t - b` is at index `t`, everything before it <= and everything after it >=.
//
// Written by hand rather than `std::nth_element`: nothing from libstdc++ is used in the
// kernels here, and introselect would make device compilation depend on it. The key is read
// through an indirection (`pos( perm( k ), ax )`), but `perm` starts as the identity and the two
// Hoare sweeps are linear, so the accesses remain nearly sequential within the node's slice.
HD void bsp_select( auto &&perm, const auto &pos, SI b, SI e, SI t, int ax ) {
    using TF = typename DECAYED_TYPE_OF( pos )::TF;

    auto key = [&]( SI k ) { return TF( pos( SI( perm( k ) ), ax ) ); };
    auto swp = [&]( SI i, SI j ) { const SI x = SI( perm( i ) ); perm( i ) = SI( perm( j ) ); perm( j ) = x; };

    SI lo = b, hi = e;
    while ( hi - lo > 2 ) {
        const SI c = lo + ( hi - lo ) / 2;
        const TF k0 = key( lo ), k1 = key( c ), k2 = key( hi - 1 );
        // the pivot is ALWAYS a value present in the slice: this is what guarantees that
        // the two sweeps below stop without a bounds test.
        const TF pivot = k0 < k1 ? ( k1 < k2 ? k1 : ( k0 < k2 ? k2 : k0 ) )
                                 : ( k0 < k2 ? k0 : ( k1 < k2 ? k2 : k1 ) );

        SI i = lo - 1, j = hi;
        while ( true ) {
            do { ++i; } while ( key( i ) < pivot );
            do { --j; } while ( key( j ) > pivot );
            if ( i >= j )
                break;
            swp( i, j );
        }

        // `[ lo, j ]` and `[ j + 1, hi )`, both non-empty (Hoare with a present pivot splits
        // in the middle even when all values are equal, so the recursion converges).
        if ( t <= j )
            hi = j + 1;
        else
            lo = j + 1;
    }

    if ( hi - lo == 2 && key( lo ) > key( lo + 1 ) )
        swp( lo, lo + 1 );
}


// `( a, b )` such that `w_k <= a . y_k + b` for every seed of the slice -- see
// `AaBsp.py::_weight_majorant` for WHY the majorant is affine and how the candidate is
// retained. Same rule, same threshold; only the fit differs (see below).
template<int ct_dim>
HD void bsp_weight_majorant( const auto &pos, const auto &w, SI b, SI e, auto &&wa_out, auto &&wb_out ) {
    using TF = typename DECAYED_TYPE_OF( pos )::TF;

    const SI m = e - b;

    TF wmin = TF( w( b ) ), wmax = wmin, wsum = 0;
    auto psum = Vector<TF,ct_dim>::zeros();
    auto plo = Vector<TF,ct_dim>::with_func( [&]( PI d ) { return TF( pos( b, d ) ); } ), phi = plo;
    for ( SI k = b; k < e; ++k ) {
        const TF v = TF( w( k ) );
        wmin = v < wmin ? v : wmin;
        wmax = v > wmax ? v : wmax;
        wsum += v;
        for ( int d = 0; d < ct_dim; ++d ) {
            const TF y = TF( pos( k, d ) );
            psum[ d ] += y;
            plo[ d ] = y < plo[ d ] ? y : plo[ d ];
            phi[ d ] = y > phi[ d ] ? y : phi[ d ];
        }
    }
    const TF spread = wmax - wmin;

    auto a = Vector<TF,ct_dim>::zeros();
    if ( m >= 2 * ( ct_dim + 1 ) && spread > 0 ) {
        const TF inv = TF( 1 ) / TF( m );
        const auto pm = Vector<TF,ct_dim>::with_func( [&]( PI d ) { return psum[ d ] * inv; } );
        const TF wm = wsum * inv;

        // the NORMAL EQUATIONS of the centered least squares, `[ q^T q | q^T dw ]`, solved by Gauss
        // with partial pivoting. The host goes through an SVD (`lstsq`) instead, better conditioned -- and that
        // need not be the case here: whatever `a` comes out, the `b` computed below is RAISED
        // until it majorizes, so a mediocre fit can only prune less, never lie.
        TF A[ ct_dim ][ ct_dim + 1 ];
        for ( int i = 0; i < ct_dim; ++i )
            for ( int j = 0; j <= ct_dim; ++j )
                A[ i ][ j ] = 0;
        for ( SI k = b; k < e; ++k ) {
            const TF dw = TF( w( k ) ) - wm;
            for ( int i = 0; i < ct_dim; ++i ) {
                const TF qi = TF( pos( k, i ) ) - pm[ i ];
                for ( int j = 0; j < ct_dim; ++j )
                    A[ i ][ j ] += qi * ( TF( pos( k, j ) ) - pm[ j ] );
                A[ i ][ ct_dim ] += qi * dw;
            }
        }

        bool ok = true;
        for ( int c = 0; c < ct_dim && ok; ++c ) {
            int p = c;
            for ( int i = c + 1; i < ct_dim; ++i )
                if ( sdot::fabs( A[ i ][ c ] ) > sdot::fabs( A[ p ][ c ] ) )
                    p = i;
            if ( ! ( sdot::fabs( A[ p ][ c ] ) > 0 ) ) {     // zero column -> no fit
                ok = false;
                break;
            }
            if ( p != c )
                for ( int j = c; j <= ct_dim; ++j ) {
                    const TF t = A[ c ][ j ]; A[ c ][ j ] = A[ p ][ j ]; A[ p ][ j ] = t;
                }
            for ( int i = c + 1; i < ct_dim; ++i ) {
                const TF f = A[ i ][ c ] / A[ c ][ c ];
                for ( int j = c; j <= ct_dim; ++j )
                    A[ i ][ j ] -= f * A[ c ][ j ];
            }
        }

        if ( ok ) {
            auto fit = Vector<TF,ct_dim>::zeros();
            for ( int i = ct_dim - 1; i >= 0; --i ) {
                TF s = A[ i ][ ct_dim ];
                for ( int j = i + 1; j < ct_dim; ++j )
                    s -= A[ i ][ j ] * fit[ j ];
                fit[ i ] = s / A[ i ][ i ];
            }

            TF rmin = 0, rmax = 0;
            for ( SI k = b; k < e; ++k ) {
                TF r = TF( w( k ) );
                for ( int d = 0; d < ct_dim; ++d )
                    r -= fit[ d ] * TF( pos( k, d ) );
                if ( k == b ) { rmin = r; rmax = r; }
                else { rmin = r < rmin ? r : rmin; rmax = r > rmax ? r : rmax; }
            }

            // the tightening that CHANCE already gives to `d + 1` parameters on `m` points: without
            // this correction a node of purely random weights would retain the affine fit one time
            // out of three. See `AaBsp.py::_weight_majorant`.
            const TF u = TF( 1 ) - TF( ct_dim ) / TF( m - 1 );
            const TF by_chance = sdot::sqrt( u > 0 ? u : TF( 0 ) );
            // and a slope that, over the node's extent, far exceeds the spread of the weights
            // is an artifact of conditioning ( seeds aligned to within 1e-8 ), not a
            // fit: it would make a `b` at 1e9 that no longer majorizes anything useful. See
            // `AaBsp.py::_weight_majorant`.
            bool well_behaved = true;
            for ( int d = 0; d < ct_dim; ++d ) {
                const TF reach = sdot::fabs( plo[ d ] ) > sdot::fabs( phi[ d ] ) ? sdot::fabs( plo[ d ] ) : sdot::fabs( phi[ d ] );
                if ( sdot::fabs( fit[ d ] ) * ( phi[ d ] - plo[ d ] ) > 8 * spread || sdot::fabs( fit[ d ] ) * reach > TF( 100 ) * spread )
                    well_behaved = false;                        // the margin on `b`, relative to `|a . y|`, must stay negligible
            }
            if ( well_behaved && rmax - rmin < TF( 0.85 ) * by_chance * spread )
                a = fit;
        }
    }

    TF bb = 0, amax = 0;
    for ( SI k = b; k < e; ++k ) {
        TF ay = 0;
        for ( int d = 0; d < ct_dim; ++d )
            ay += a[ d ] * TF( pos( k, d ) );
        const TF v = TF( w( k ) ) - ay;
        if ( k == b ) bb = v; else bb = v > bb ? v : bb;
        amax = sdot::fabs( ay ) > amax ? sdot::fabs( ay ) : amax;
    }

    for ( int d = 0; d < ct_dim; ++d )
        wa_out( d ) = a[ d ];

    // a rounding MARGIN on the constant, and on it alone -- see `_weight_majorant`: `b` is
    // the only term that the host and the kernel would compute differently, and a `b` rounded
    // down would cease to majorize.
    wb_out = bb + TF( 1e-6 ) * ( sdot::fabs( bb ) + spread + amax );
}

/// the majorant of ONE node, redone on fresh weights ( `AaBsp.refresh_weight_majorants` ): the
/// slice `[ b, e )` of the cloud `src` -- the seeds IN THE TREE'S ORDER. Takes only what it
/// needs, and above all NOT the whole tree: its current majorants are what is being replaced.
HD void bsp_refresh_majorant( const auto &src, const auto &beg, const auto &end, auto &&wa_out, auto &&wb_out ) {
    constexpr int ct_dim = CT_VALUE( src.nb_dims );
    const SI b = SI( beg ), e = SI( end );
    if ( e <= b ) {
        for ( int d = 0; d < ct_dim; ++d )
            wa_out( d ) = 0;
        wb_out = 0;
        return;
    }
    bsp_weight_majorant<ct_dim>( src.positions, src.weights, b, e, wa_out, wb_out );
}


// The body of the level, for the node whose slice is `[ beg_in, end_in )`.
//
// `mid_out` says where to cut: the left child receives `[ beg, mid )` and the right one `[ mid, end )`. A
// node that has nothing left to cut returns `mid = end`, hence passes everything to the left -- this is the
// PROPAGATION described in `AaBsp.py`, which keeps the partition of `[ 0, n )` from one level to the
// next, hence the disjoint writes.
HD void bsp_build_level( const auto &src, auto &&dst, auto &&perm,
                      const auto &beg_in, const auto &end_in,
                      auto &&box_out, auto &&wa_out, auto &&wb_out, auto &&mid_out,
                      SI leaf_size ) {
    // the dimension is a COMPILE-TIME count (`nb_dims : CtShapeVar`), and it is THAT which carries it:
    // the shape of a tensor, for its part, crosses as runtime integers. This is why
    // this function takes whole clouds and not their members.
    constexpr int ct_dim = CT_VALUE( src.nb_dims );
    using TF = typename DECAYED_TYPE_OF( src.positions )::TF;

    const auto &pos_in = src.positions;
    const auto &w_in   = src.weights;
    const auto &ord_in = src.order;
    auto &&pos_out = dst.positions;
    auto &&w_out   = dst.weights;
    auto &&ord_out = dst.order;

    const SI b = SI( beg_in );
    const SI e = SI( end_in );

    // an EMPTY slot: the right child of a node that passed everything to the left. It has nothing to read
    // or write in the cloud, but its per-node outputs are its own and nobody else will
    // write them -- an output buffer is not zeroed.
    mid_out = e;
    if ( e <= b ) {
        for ( int d = 0; d < ct_dim; ++d ) {
            box_out( 0, d ) = 0;
            box_out( 1, d ) = 0;
        }
        if constexpr ( DECAYED_TYPE_OF( wa_out )::is_valid ) {
            for ( int d = 0; d < ct_dim; ++d )
                wa_out( d ) = 0;
            wb_out = 0;
        }
        return;
    }

    // ---- the subtree's box
    auto lo = Vector<TF,ct_dim>::with_func( [&]( PI d ) { return TF( pos_in( b, d ) ); } );
    auto hi = lo;
    for ( SI k = b + 1; k < e; ++k )
        for ( int d = 0; d < ct_dim; ++d ) {
            const TF v = TF( pos_in( k, d ) );
            lo[ d ] = v < lo[ d ] ? v : lo[ d ];
            hi[ d ] = v > hi[ d ] ? v : hi[ d ];
        }
    // `lo` THEN `hi`, in the same array: one box = one contiguous read on the walk side.
    for ( int d = 0; d < ct_dim; ++d ) {
        box_out( 0, d ) = lo[ d ];
        box_out( 1, d ) = hi[ d ];
    }

    // ---- the weights majorant. No weights -> both tensors are `NoneTensor` and this whole
    // block disappears at COMPILE time, as in `AaBsp.cxx`.
    if constexpr ( DECAYED_TYPE_OF( wa_out )::is_valid )
        bsp_weight_majorant<ct_dim>( pos_in, w_in, b, e, wa_out, wb_out );

    // ---- cut, or propagate
    int ax = 0;
    for ( int d = 1; d < ct_dim; ++d )
        if ( hi[ d ] - lo[ d ] > hi[ ax ] - lo[ ax ] )
            ax = d;

    for ( SI k = b; k < e; ++k )
        perm( k ) = k;

    SI mid = e;
    // `hi[ ax ] <= lo[ ax ]`: all the seeds at the same place, no cut would separate them.
    if ( e - b > leaf_size && hi[ ax ] > lo[ ax ] ) {
        // the MEDIAN, not the middle of the box: this is what bounds the depth by
        // `log2( n / leaf_size )` whatever the distribution.
        mid = b + ( e - b ) / 2;
        bsp_select( perm, pos_in, b, e, mid, ax );
    }
    mid_out = mid;

    // ---- the permuted cloud. A leaf (or a node that propagates) copies its slice as
    // is: `perm` has remained the identity there, so it is the same code.
    for ( SI j = b; j < e; ++j ) {
        const SI s = SI( perm( j ) );
        ord_out( j ) = ord_in( s );
        for ( int d = 0; d < ct_dim; ++d )
            pos_out( j, d ) = pos_in( s, d );
        if constexpr ( DECAYED_TYPE_OF( w_in )::is_valid )
            w_out( j ) = w_in( s );
    }
}

} // namespace sdot
