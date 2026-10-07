#pragma once

// the members + the axes this body names, written to the build include tree by `CallArg_Aggregate`.
#include <sdot/generated/aggregates/SumOfGaussians.h>
#include <loom/support/common_macros.h>
#include <loom/support/containers/Vector.h>
#include <loom/support/atomic_add.h>
#include "PointwiseDensity.h"
#include "GaussianTree.h"

namespace sdot {

// A sum of ISOTROPIC gaussians, seen as a density to integrate over a cell.
//
//   rho( x ) = somme_i  w_i * exp( - r_i^2 / ( 2 s_i^2 ) ) / ( 2 pi s_i^2 ) ^ ( d / 2 ),
//   r_i = | x - c_i |
//
// `w_i` is the MASS of gaussian `i`, not its height -- the total mass is therefore the sum of the
// weights, and normalizing is a division (see `distributions/SumOfGaussians.py`).
//
// It is the first density that is not piecewise constant, and all it has to say fits in three
// POINTWISE answers: the value, the gradient, and where to put `d rho / d parameters`.
// It CUTS nothing (its piece is the whole cell) and knows nothing about cells; it is
// `PowerDiagram::integrate_into` that, seeing `is_constant == false`, cuts them into simplices and
// does its quadrature there. The split is there, and it has no third side.
SDOT_TEMPLATE_DECL_FOR_SumOfGaussians
struct SumOfGaussians {
    SDOT_ATTRIBUTES_OF_SumOfGaussians

    static constexpr int ct_dim = DECAYED_TYPE_OF( nb_dims )::value;
    using TF = DECAYED_TYPE_OF( positions )::TF;

    /// THE CONVOLUTION by a gaussian of width `conv_s` ( `with_convolution` ): on a sum of
    /// gaussians it only changes the widths, `sigma_i' = sqrt( sigma_i^2 + conv_s^2 )`, nothing
    /// else -- what the width continuation of `SdotPlanNd` walks through ( `sdotplan/Continuation.h` ).
    /// A member APART from the generated attributes: `0` by default, hence absent from any ordinary call.
    TF conv_s = 0;

    /// THE LOCAL TERMS ( `GaussianTree.h` ). With a `tree` ( hung by `Convolved<SumOfGaussians>`, the solver ), each cell
    /// sees only the terms that weigh on it -- the gaussians near it, or groups of them merged at the continuation
    /// width -- and the sums below run over `terms` instead of the generated attributes. FORWARD ONLY: the adjoint
    /// ( `add_value_grad_at`, `integrate_over_simplex_bwd` ) always runs over the gaussians themselves, and a density
    /// with a tree is never differentiated ( the solver does not ).
    using Term = GaussianTerm<TF,ct_dim>;
    const GaussianTree<TF,ct_dim> *tree = nullptr;
    const Term *terms = nullptr;
    SI nb_local = 0;

    HD SI nb_terms () const { return terms ? nb_local : SI( nb_gaussians ); }
    HD TF center_of( SI i, PI c ) const { return terms ? terms[ i ].c[ c ] : TF( positions( i, c ) ); }
    HD TF weight_of( SI i ) const { return terms ? terms[ i ].w : TF( weights( i ) ); }

    HD TF sigma_of( SI i ) const {
        if ( terms ) return terms[ i ].s;                 // the convolution is already in
        const TF s = TF( sigmas( i ) );
        return conv_s > 0 ? sdot::sqrt( s * s + conv_s * conv_s ) : s;
    }
    HD SumOfGaussians with_convolution( TF s ) const { SumOfGaussians r( *this ); r.conv_s = s; return r; }
    /// the smallest width ( before convolution ): the scale below which the continuation stops
    HD TF smallest_sigma() const { TF r = TF( sigmas( 0 ) ); for ( SI i = 1; i < SI( sigmas.shape( 0 ) ); ++i ) r = sdot::fmin( r, TF( sigmas( i ) ) ); return r; }

    /// A single piece, the cell itself, and not a cut: nothing to cut when the density
    /// is defined everywhere by the same formula. The cutting scratch is therefore not touched (and
    /// `extra_cuts_per_piece` returns 0, so it is not even allocated).
    ///
    /// HOW we integrate depends on the dimension, and it is a choice made here, by
    /// composition -- the integrator only sees the contract:
    ///   * in 2D we KNOW how (see `wedge_measure`), so we pass ourselves;
    ///   * beyond that we do not, so we declare ourselves a BLACK BOX by wrapping in
    ///     `PointwiseDensity`, which only needs `value_at` / `gradient_at`.
    ///
    /// With a `tree`, the piece is integrated by a copy of `self` that carries the terms of this cell.
    HD void for_each_piece( const auto &cell, auto &&/*ws*/, auto &&func ) const {
#if ! defined( __CUDA_ARCH__ )
        if ( tree && ! terms ) {
            double lo[ ct_dim ], hi[ ct_dim ];
            for ( int d = 0; d < ct_dim; ++d ) { lo[ d ] = 1e300; hi[ d ] = -1e300; }
            for ( SI v = 0; v < SI( cell.nb_vertices() ); ++v )
                for ( int d = 0; d < ct_dim; ++d ) {
                    const double x = double( cell.coord( int( v ), d ) );
                    lo[ d ] = x < lo[ d ] ? x : lo[ d ];
                    hi[ d ] = x > hi[ d ] ? x : hi[ d ];
                }
            thread_local std::vector<Term> buf;
            tree->gather( lo, hi, double( conv_s ), buf );
            SumOfGaussians local( *this );
            local.terms = buf.data();
            local.nb_local = SI( buf.size() );
            if constexpr ( ct_dim == 2 )
                func( cell, local );
            else
                func( cell, PointwiseDensity{ local } );
            return;
        }
#endif
        if constexpr ( ct_dim == 2 )
            func( cell, *this );
        else
            func( cell, PointwiseDensity{ *this } );
    }

    /// a term WIDE against the triangle `pts` ( during a continuation ) is integrated by a 7 point rule instead of the
    /// exact reduction: the error goes as `( diameter / width )^6`, and it costs 7 exponentials instead of ~100
    static constexpr TF quadrature_ratio = 1.5;
    HD TF quadrature_term( SI i, const auto &pts ) const;

    // ---- EXACT integration in 2D ------------------------------------------------------------------
    // An isotropic gaussian over a triangle has no elementary closed form -- it is Owen's T
    // function -- but it REDUCES to a smooth, bounded 1D integral, whose accuracy no longer depends
    // on the shape of the cell. That is the whole difference with a quadrature rule on the
    // triangle, whose error goes as `( cell size / sigma ) ^ 4`.
    //
    // The reduction: after translation/scaling, the triangle is decomposed into three signed
    // corners `( 0, P, Q )`, and a corner is integrated in polar coordinates --
    //
    //     W( P, Q ) = sign( p ) / 2pi * Int_{t_P}^{t_Q} [ 1 - exp( -( p^2 + t^2 ) / 2 ) ] p dt / ( p^2 + t^2 )
    //
    // where `p` is the signed distance from the origin to the line `PQ` and `t` the abscissa along it.
    // Written THIS WAY -- the difference taken under the integral, not between two arctangents -- it has
    // no catastrophic cancellation, even when the origin grazes the line.
    //
    // Beyond `|t| = tail_cut`, the exponential is worth nothing and only the
    // Lorentzian remains, whose antiderivative is `arctan( t / p )`: the tails are therefore EXACT and
    // free, and the quadrature only works on a bounded interval where the integrand has a
    // scale `>= 1`. Measured: `8 x 4 = 32` evaluations per edge give 4e-14 absolute error
    // on configurations chosen to be nasty (seed on the center, huge triangle, grazing
    // sliver).
    static constexpr bool is_constant = false;

    /// the moments ( `diagram::integrate_moments_into` ): the exact reduction does not give them, it is
    /// the adaptive quadrature of `PointwiseDensity` that accumulates them, in 2D as elsewhere.
    HD void integrate_moments_over_simplex( const auto &pts, TF &m, auto &mx, TF &m2 ) const {
        PointwiseDensity{ *this }.integrate_moments_over_simplex( pts, m, mx, m2 );
    }

    static constexpr TF  tail_cut   = 8;    ///< `exp( -t^2/2 ) < 1e-14` beyond: the tail is exact
    static constexpr int nb_panels  = 4;    ///< Gauss-Legendre panels over the core

    /// The SIGNED standard normal measure of the triangle `( 0, P, Q )` -- the corner. `P` and `Q` are 2D whatever
    /// `ct_dim`: the in-plane frame of a 3D facet ( `facet_mass` ) goes through here too.
    HD TF wedge_measure( const auto &P, const auto &Q ) const;

    /// The standard normal measure of the triangle `ys` (positive, any orientation).
    HD TF std_triangle_measure( const auto &ys ) const;

    /// `Int_T rho`, exact. `pts`: the 3 vertices.
    HD TF integrate_over_simplex( const auto &pts ) const;

    /// The ELEMENTARY adjoint -- that is the remarkable point: the value requires a special
    /// function, its derivatives do not. Everything reduces to BOUNDARY integrals, which for a gaussian
    /// along a segment are `erf`s:
    ///   * a vertex: moving it sweeps its two edges, hence `Int rho * lambda` on each;
    ///   * the center: `d/dc = -Int grad rho = -Contour rho n ds` (divergence) -- equal, and it is a
    ///     good check, to MINUS the sum of the per-vertex derivatives;
    ///   * sigma: `d rho / d sigma = sigma * laplacian( rho )` (heat identity, `t = sigma^2/2`),
    ///     hence again a boundary flux, and `y . n` is CONSTANT there along an edge;
    ///   * the weight: the measure itself, already computed.
    HD void integrate_over_simplex_bwd( const auto &pts, TF g, auto &&grad_pts, auto &&grad_dist ) const;

    /// For an edge `A -> B` of the triangle `A, B, C`, in the standard frame: the OUTWARD normal, the
    /// signed distance `y . n` (constant along the edge), `Int phi ds`, and `Int phi lambda_A ds`.
    struct EdgeInfo { Vector<TF,2> n; TF p; TF j0; TF j1a; };
    HD EdgeInfo edge_info( const auto &A, const auto &B, const auto &C ) const;

    /// The NORMALIZED kernel of gaussian `i` at `x` (mass 1), and the squared distance --
    /// the two quantities from which everything else follows, computed once.
    HD auto kernel_at( SI i, const auto &x ) const;

    /// `Int_{facet} rho ds` over the facet `cut` of the cell `pc` -- what the laplacian of a transport reads
    /// ( `sdotplan/Sweep.h` ). In 2D, the edge `[ v_cut, v_cut+1 ]`: a gaussian along a segment is an `erf`, the
    /// distance to the segment being constant. In 3D, a planar polygon: an isotropic gaussian SPLITS into a 1D factor
    /// in the distance to the plane and a 2D gaussian in the plane, whose measure over the polygon is the exact 2D
    /// reduction ( `wedge_measure` ) on a fan of triangles. The proxy of a 3D mesh needs it ( `SdotPlanNd._solve_by_proxy` ).
    HD TF   facet_mass    ( const auto &pc, int cut ) const requires ( ct_dim == 2 || ct_dim == 3 );
    HD TF   facet_mass_2d ( const auto &pc, int cut ) const;
    HD TF   facet_mass_3d ( const auto &pc, int cut ) const;   ///< `pc.for_each_facet_triangle` ( `LocalN` )

    HD TF   value_at      ( const auto &x ) const;   ///< rho( x )
    HD auto gradient_at   ( const auto &x ) const;   ///< grad rho( x ), a `Vector<TF,ct_dim>`

    /// Accumulates `g * d rho( x ) / d parameter` into the cotangent of each parameter.
    ///
    /// It is the "distribution" half of the adjoint, and it only knows `x`: the integrator
    /// passes it the quadrature weight of the node (`g`), without knowing what is behind it. The
    /// additions are ATOMIC -- a wide gaussian is seen by the work-items of many
    /// cells at once -- and each is guarded by the validity of its target, a non-differentiated
    /// parameter arriving as a `NoneTensor`.
    HD void add_value_grad_at( auto &&grad_dist, const auto &x, TF g ) const;
};

}

#include "SumOfGaussians.cxx"

// its convolved density, for the transport solver ( `sdotplan/Convolved.h` )
#include "sdotplan/convolved/SumOfGaussians.h"
