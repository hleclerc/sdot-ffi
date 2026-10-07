#pragma once

#include <loom/support/common_macros.h>
#include <loom/support/containers/Matrix.h>
#include <loom/support/containers/Vector.h>
#include <loom/support/math.h>

// `sdot::` and not `std::` for the mathematics -- see `SumOfGaussians.cxx` and
// `loom/support/math.h`: it is the device that picks the implementation.

namespace sdot {

// "I do not know how to integrate myself, but I know how to EVALUATE myself at a point" -- the
// black-box density, wrapped in an adaptive quadrature.
//
// It is an IMPLEMENTATION of the piece contract (see `distributions/Distribution.py`), not a
// regime of the integrator: `PowerDiagram::integrate_into` knows no quadrature, it asks the
// density for the integral over a simplex and hands it the cotangent of the vertices. A density
// that KNOWS how to integrate itself -- a closed formula, a reduction to a special function, like
// `SumOfGaussians` in 2D -- answers directly and never comes through here. One that does not
// wraps itself in this, in one line:
//
//     void for_each_piece( const auto &cell, auto &&, auto &&func ) const {
//         func( cell, PointwiseDensity{ *this } );
//     }
//
// What it asks of `D`: `value_at( x )`, `gradient_at( x )`, `add_value_grad_at( gd, x, g )`.
//
// = The rule, and why it is not enough on its own
//
// `d + 1` nodes of equal weight `1 / ( d + 1 )`, each carrying `alpha` on one vertex and `beta` on
// the other `d`: the symmetric rule exact up to degree 2. `alpha` comes from exactness on the
// `lambda_i^2`: `alpha^2 + d beta^2 = 2 / ( d + 2 )` with `alpha + d beta = 1`, hence
// `( d + 1 ) alpha^2 - 2 alpha + ( 2 - d ) / ( d + 2 ) = 0`. In 2D we recover `( 2/3, 1/6 )`, in 3D
// `( 0.585410, 0.138197 )`: the classical rules of the triangle and the tetrahedron.
//
// Its error goes as `( simplex size / density scale ) ^ 4` -- unacceptable as soon as a cell is
// large compared to a narrow gaussian. Hence the SUBDIVISION: the simplex is bisected along its
// longest edge as long as the rule and its two halves disagree. It is the only generic way to
// catch up with a scale we do not know -- we precisely do not ask the density to declare it, it
// is a black box.
//
// A bisection POPS ONE element and PUSHES TWO, so the stack is bounded by the DEPTH and not by
// the number of leaves: `max_depth + 2` entries, each a barycentric matrix `( d + 1 ) ^ 2`. That
// is what makes it usable in a kernel -- no allocation, a bound written in the code. It is not
// free for all that (in 3D, ~1 KB per work-item) and that budget is NOT counted by
// `PowerDiagram.measures`, which does not know the density.
//
// = The adjoint
//
// That of WHAT IS COMPUTED, not of the ideal integral -- the only way to be consistent with the
// forward, and that is why the backward REDOES the same subdivision (the criterion is
// deterministic, so it lands on the same leaves). On a leaf, two terms: the volume moves with its
// vertices (cofactor of the determinant), and a node is a FIXED barycentric combination of those
// vertices, so the cotangent of `gradient_at` is spread over them with those weights. The vertices
// of a leaf being themselves barycentric combinations of the ORIGINAL vertices, a final
// multiplication brings everything back where the integrator expects it.
template<class D>
struct PointwiseDensity {
    using TF = typename D::TF;
    static constexpr int ct_dim = D::ct_dim;

    /// not constant over a piece: the integrator will go through the simplex decomposition.
    static constexpr bool is_constant = false;

    /// Bisect at least down to here before being allowed to stop: the rule only sees
    /// `d + 1` points, and a narrow density can fall exactly between them -- in which case the simplex
    /// and its halves would agree on zero. It is the classical safeguard of an adaptive
    /// quadrature, and it costs four evaluations.
    static constexpr int min_depth = 2;
    static constexpr int max_depth = 8;     ///< bounds the stack AND the worst-case cost
    /// Relative gap tolerated between a simplex and its two halves -- what bounds the
    /// integration error, and that is its only role.
    ///
    /// It ALSO bounds, unintentionally, the jump the value makes when a sub-simplex flips
    /// from "refine" to "accept": a value-driven adaptive scheme is only smooth to within `rtol`,
    /// by construction. It is not a setting to tighten -- a finite difference divides this
    /// jump by its step, so NO reasonable `rtol` makes the scheme checkable at a tight step,
    /// and tightening it only pays bisection levels for nothing. It is up to the CHECKER
    /// to take a step matched to the scheme (see the tests), and to an optimizer to know that it
    /// works on a piecewise smooth function.
    static constexpr TF  rtol      = 1e-5;

    D dens;

    /// A sub-simplex: its `d + 1` vertices in BARYCENTRIC coordinates of the original simplex.
    /// It is this representation, and not the points, that lets the adjoint get back to the
    /// original vertices without solving anything.
    using Bary = Vector<Vector<TF,ct_dim+1>,ct_dim+1>;

    HD static Bary whole() {
        return Bary( Function(), []( PI k ) {
            return Vector<TF,ct_dim+1>( Function(), [&]( PI j ) { return TF( j == k ); } ); } );
    }

    HD static auto points_of( const Bary &b, const auto &pts ) {
        return Vector<Vector<TF,ct_dim>,ct_dim+1>( Function(), [&]( PI k ) {
            return Vector<TF,ct_dim>( Function(), [&]( PI c ) {
                TF s = 0;
                for ( SI j = 0; j <= ct_dim; ++j )
                    s += b[ k ][ j ] * pts[ j ][ c ];
                return s;
            } );
        } );
    }

    /// MAUBACH bisection: we ALWAYS cut the edge `( vertex 0, vertex d )`, and the children
    /// put the new vertex back in second position, the others shifted by one.
    ///
    /// The rule is COMBINATORIAL -- it looks at no length -- and that is what matters most
    /// here. Cutting "the longest edge" seems better, but it is a DISCONTINUOUS choice:
    /// two almost equal edges, and an infinitesimal move of a seed flips the whole
    /// pattern, so the computed value jumps by the quadrature error -- and the adjoint can no
    /// longer match a finite difference. With an index rule, the pattern being fixed, each
    /// sub-simplex is an AFFINE function of the original vertices: the value is smooth, and the
    /// only remaining discontinuity is the decision to refine, bounded by `rtol`.
    ///
    /// The vertex rotation is what bounds the degradation of shapes (Maubach 1995). Measured
    /// over 8 levels, the minimal quality holds: 0.32 in 2D, 0.14 in 3D, 0.13 in 4D -- against
    /// 0.50 / 0.22 / 0.18 for the longest edge, and 0.13 / 0.09 / 0.08 for a naive cyclic rule.
    HD static void bisect( const Bary &b, const auto &/*pts*/, Bary &lo, Bary &hi ) {
        Vector<TF,ct_dim+1> mid;
        for ( SI j = 0; j <= ct_dim; ++j )
            mid[ j ] = ( b[ 0 ][ j ] + b[ ct_dim ][ j ] ) / 2;

        lo[ 0 ] = b[ 0 ];
        hi[ 0 ] = b[ ct_dim ];
        lo[ 1 ] = mid;
        hi[ 1 ] = mid;
        for ( SI k = 1; k < ct_dim; ++k ) {
            lo[ k + 1 ] = b[ k ];
            hi[ k + 1 ] = b[ k ];
        }
    }

    /// The leaves of the subdivision, with the value of the rule on each. The forward sums them,
    /// the backward differentiates them again -- same traversal, hence same leaves.
    HD void for_each_leaf( const auto &pts, auto &&func ) const {
        Vector<Bary,max_depth+2> stack;
        Vector<int,max_depth+2>  depth;
        Vector<TF,max_depth+2>   value;

        SI top = 0;
        stack[ 0 ] = whole();
        depth[ 0 ] = 0;
        value[ 0 ] = rule( points_of( stack[ 0 ], pts ) );

        while ( top >= 0 ) {
            const Bary b = stack[ top ];
            const int dp = depth[ top ];
            const TF  cv = value[ top ];
            --top;

            if ( dp >= max_depth ) {
                func( b, cv );
                continue;
            }

            Bary lo, hi;
            bisect( b, pts, lo, hi );
            const TF v0 = rule( points_of( lo, pts ) );
            const TF v1 = rule( points_of( hi, pts ) );
            const TF fine = v0 + v1;

            const TF diff = fine > cv ? fine - cv : cv - fine;
            const TF mag  = fine < 0 ? -fine : fine;
            if ( dp >= min_depth && diff <= rtol * mag ) {
                // we keep the TWO HALVES as leaves, not the parent: it is `fine` that the
                // forward adds up, so it is `fine` that the backward must differentiate.
                func( lo, v0 );
                func( hi, v1 );
                continue;
            }

            ++top; stack[ top ] = lo; depth[ top ] = dp + 1; value[ top ] = v0;
            ++top; stack[ top ] = hi; depth[ top ] = dp + 1; value[ top ] = v1;
        }
    }

    HD TF integrate_over_simplex( const auto &pts ) const {
        TF res = 0;
        for_each_leaf( pts, [&]( const Bary &, TF v ) { res += v; } );
        return res;
    }

    /// the mass of a facet, when the wrapped density knows it ( `SumOfGaussians` in 3D: it does not know its simplices, but its
    /// facets reduce to the 2D case ) -- what the laplacian of a transport reads ( `sdotplan/Sweep.h` )
    HD TF facet_mass( const auto &pc, int cut ) const requires requires { dens.facet_mass( pc, cut ); } {
        return dens.facet_mass( pc, cut );
    }

    /// the moments of order 0, 1, 2 of the density over the simplex, ACCUMULATED into `m` / `mx` / `m2`
    /// -- the same rule, on the same leaves, each node weighing `vol / ( d + 1 ) * rho( x )`.
    HD void integrate_moments_over_simplex( const auto &pts, TF &m, auto &mx, TF &m2 ) const {
        for_each_leaf( pts, [&]( const Bary &b, TF ) {
            const auto P = points_of( b, pts );
            const TF det = edge_matrix( P ).determinant();
            const TF w = ( det < 0 ? -det : det ) / factorial() / ( ct_dim + 1 );
            for ( SI q = 0; q <= ct_dim; ++q ) {
                const auto x = node( P, q );
                const TF r = w * dens.value_at( x );
                m += r;
                mx += r * x;
                m2 += r * norm_2_p2( x );
            }
        } );
    }

    HD void integrate_over_simplex_bwd( const auto &pts, TF g, auto &&grad_pts, auto &&grad_dist ) const {
        for_each_leaf( pts, [&]( const Bary &b, TF /*v*/ ) {
            auto gl = Vector<Vector<TF,ct_dim>,ct_dim+1>( Function(), []( PI ) {
                return Vector<TF,ct_dim>::zeros(); } );

            rule_bwd( points_of( b, pts ), g, gl, grad_dist );

            // vertex `k` of the leaf is `sum_j b[k][j] * pts[j]`: its cotangent is
            // redistributed with exactly those weights.
            for ( SI k = 0; k <= ct_dim; ++k )
                for ( SI j = 0; j <= ct_dim; ++j )
                    for ( PI c = 0; c < ct_dim; ++c )
                        grad_pts[ j ][ c ] += b[ k ][ j ] * gl[ k ][ c ];
        } );
    }

    // ---- the rule itself, on ONE simplex given by its points ---------------------------------

    HD static auto barycentric() {
        const TF d = ct_dim;
        const TF alpha = ( 1 + sdot::sqrt( 1 - ( d + 1 ) * ( 2 - d ) / ( d + 2 ) ) ) / ( d + 1 );
        return Vector<TF,2>( Values(), alpha, ( 1 - alpha ) / d );
    }

    HD static TF factorial() {
        TF res = 1;
        for ( int i = 2; i <= ct_dim; ++i )
            res *= i;
        return res;
    }

    /// `M` = the `d` edges issued from `P[ 0 ]`, as columns: its determinant gives the volume, and
    /// its cofactors the derivative of that volume.
    HD static auto edge_matrix( const auto &P ) {
        return Matrix<TF,ct_dim>::with_func( [&]( auto r, auto c ) { return P[ c + 1 ][ r ] - P[ 0 ][ r ]; } );
    }

    /// node `q`: `beta` everywhere, `alpha` on vertex `q`.
    HD static auto node( const auto &P, SI q ) {
        const auto ab = barycentric();
        auto tot = Vector<TF,ct_dim>::zeros();
        for ( SI k = 0; k <= ct_dim; ++k )
            tot = tot + P[ k ];
        return ab[ 1 ] * tot + ( ab[ 0 ] - ab[ 1 ] ) * P[ q ];
    }

    HD TF rule( const auto &P ) const {
        const TF det = edge_matrix( P ).determinant();
        const TF vol = ( det < 0 ? -det : det ) / factorial();

        TF s = 0;
        for ( SI q = 0; q <= ct_dim; ++q )
            s += dens.value_at( node( P, q ) );
        return vol * s / ( ct_dim + 1 );
    }

    HD void rule_bwd( const auto &P, TF g, auto &&gP, auto &&grad_dist ) const {
        const auto ab = barycentric();
        const auto M = edge_matrix( P );
        const TF det = M.determinant();
        const TF vol = ( det < 0 ? -det : det ) / factorial();

        TF s = 0;
        for ( SI q = 0; q <= ct_dim; ++q )
            s += dens.value_at( node( P, q ) );
        s /= ( ct_dim + 1 );

        // ---- the VOLUME part: `d|det|/dM = sign( det ) * cofactor( M )`, and each column of
        // `M` is a vertex minus the first, so the first picks up MINUS the sum of the columns.
        const TF gv = ( det < 0 ? -g : g ) * s / factorial();
        for ( PI r = 0; r < ct_dim; ++r ) {
            TF row_sum = 0;
            for ( PI c = 0; c < ct_dim; ++c ) {
                const TF minor = M.without_row_and_col( r, c ).determinant();
                const TF cof = ( ( r + c ) % 2 ? -minor : minor ) * gv;
                gP[ c + 1 ][ r ] += cof;
                row_sum += cof;
            }
            gP[ 0 ][ r ] -= row_sum;
        }

        // ---- the NODES part, and at the same point that of the density PARAMETERS
        const TF gn = g * vol / ( ct_dim + 1 );
        for ( SI q = 0; q <= ct_dim; ++q ) {
            const auto x = node( P, q );
            const auto gr = dens.gradient_at( x );
            for ( SI k = 0; k <= ct_dim; ++k ) {
                const TF w = ab[ 1 ] + ( q == k ? ab[ 0 ] - ab[ 1 ] : TF( 0 ) );
                for ( PI c = 0; c < ct_dim; ++c )
                    gP[ k ][ c ] += gn * gr[ c ] * w;
            }
            dens.add_value_grad_at( grad_dist, x, gn );
        }
    }
};

} // namespace sdot
