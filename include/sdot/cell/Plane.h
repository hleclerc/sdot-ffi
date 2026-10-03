#pragma once

#include <loom/support/common_macros.h> // HD

#include <loom/support/common_types.h>

namespace sdot {

/// THE HALF-SPACE `dir . x <= off`, as a provider returns it to the kernel : the geometry in the
/// KERNEL float (`TK`, `float` by default), and the identity that will go into `cut_ids` -- it is what
/// carries the connectivity, and the geometry of the plane does not depend on it (see `cell/Ids.h`).
///
/// `dir` is not normalized : `off` is the dot product it is compared against as
/// is, so `( 2n, 2o )` designates the same half-space as `( n, o )`.
template<class TK,int D>
struct Plane {
    TK  dir[ D ];
    TK  off;
    int id;

    /// `dir . x - off` : positive OUTSIDE
    HD TK dist( const TK *x ) const {
        TK s = - off;
        for ( int d = 0; d < D; ++d )
            s += dir[ d ] * x[ d ];
        return s;
    }
};

/// THE POWER BISECTOR of `( p0, w0 )` and `( p1, w1 )`, on the `p0` side. Computed in the
/// positions float (`TF`) and returned in the kernel's : the plane is rounded ONCE, on
/// a result, and not term by term.
///
/// `|x - p0|^2 - w0 <= |x - p1|^2 - w1` loses its `|x|^2` on both sides and becomes
/// `( p1 - p0 ) . x <= ( p1 - p0 ) . ( p0 + p1 ) / 2 + ( w0 - w1 ) / 2` : the Euclidean perpendicular bisector
/// SHIFTED along its normal by the weight gap. Written unnormalized, which is also
/// why the weight term is divided by two and not by `|p1 - p0|`.
template<class TK,int D,class TF>
HD Plane<TK,D> bisector( const TF *p0, TF w0, const TF *p1, TF w1, int id ) {
    Plane<TK,D> res;
    TF off = ( w0 - w1 ) / 2;
    for ( int d = 0; d < D; ++d ) {
        const TF dd = p1[ d ] - p0[ d ];
        off += dd * ( p0[ d ] + p1[ d ] ) / 2;
        res.dir[ d ] = TK( dd );
    }
    res.off = TK( off );
    res.id  = id;
    return res;
}

} // namespace sdot
