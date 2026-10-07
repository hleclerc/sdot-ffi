#pragma once

// =====================================================================================
// A MOMENT OF A DENSITY AS A DENSITY: `x_k rho( x )`, `|x|^2 rho( x )` or `rho( x )` itself, so that what the
// diagram knows how to do with a density -- integrate it over a cell, and send the cotangent back onto the
// seeds, the weights and the density's own parameters ( `measures`, `measures_bwd` ) -- it does for a MOMENT, without
// a line of adjoint written for it.
//
//     sdot::diagram::moment_dist<K>( dist )        K < D : int x_K rho     K == D : int |x|^2 rho     K == D + 1 : int rho
//
// The weight is a polynomial, so `rho( x ) p( x )` has the value, the gradient and the parameters' derivative that the
// product rule says, and the integral over a piece is the ADAPTIVE QUADRATURE of `PointwiseDensity` -- on every density, constant
// pieces ( `Image` ) and affine ones ( `Mesh` ) included: their closed forms are not used here, the quadrature is exact up
// to degree 2 and refines beyond ( a few 1e-6 of relative gap measured against `moments` ). What is differentiated is what is
// computed, as everywhere in `PointwiseDensity`.
//
// What it is for: the transport COST and the BARYCENTERS, differentiated by `SdotPlanNd` ( `_attach_derivatives` ).
// =====================================================================================

#include "../PointwiseDensity.h"

namespace sdot {
namespace diagram {

/// the density of one piece, multiplied by the polynomial `K` ( see above ), in dimension `D`
template<class Dens,int D,int K>
struct MomentDensity {
    using TF = typename Dens::TF;
    static constexpr int ct_dim = D;
    static constexpr bool is_constant = false;

    Dens dens;

    /// the density itself: `PointwiseDensity` wraps the one that evaluates
    HD decltype( auto ) inner() const {
        if constexpr ( requires { dens.dens; } ) return ( dens.dens );
        else                                      return ( dens );
    }

    HD static TF poly( const auto &x ) {
        if constexpr ( K < D )       return TF( x[ K ] );
        else if constexpr ( K == D ) return norm_2_p2( x );
        else                         return TF( 1 );
    }

    HD static auto poly_grad( const auto &x ) {
        return Vector<TF,D>::with_func( [&]( PI c ) {
            if constexpr ( K < D )       return TF( c == PI( K ) );
            else if constexpr ( K == D ) return TF( 2 ) * TF( x[ c ] );
            else                         return TF( 0 );
        } );
    }

    HD TF rho_at( const auto &x ) const {
        if constexpr ( Dens::is_constant ) return TF( dens.value );
        else                               return TF( inner().value_at( x ) );
    }

    HD TF value_at( const auto &x ) const { return poly( x ) * rho_at( x ); }

    HD auto gradient_at( const auto &x ) const {
        const TF r = rho_at( x ), p = poly( x );
        const auto pg = poly_grad( x );
        if constexpr ( Dens::is_constant ) {
            return Vector<TF,D>::with_func( [&]( PI c ) { return r * pg[ c ]; } );
        } else {
            const auto rg = inner().gradient_at( x );
            return Vector<TF,D>::with_func( [&]( PI c ) { return p * TF( rg[ c ] ) + r * pg[ c ]; } );
        }
    }

    HD void add_value_grad_at( auto &&grad_dist, const auto &x, TF g ) const {
        if constexpr ( Dens::is_constant ) dens.add_value_grad( grad_dist, g * poly( x ) );
        else                               inner().add_value_grad_at( grad_dist, x, g * poly( x ) );
    }

    HD TF integrate_over_simplex( const auto &pts ) const {
        return PointwiseDensity<MomentDensity>{ *this }.integrate_over_simplex( pts );
    }

    HD void integrate_over_simplex_bwd( const auto &pts, TF g, auto &&grad_pts, auto &&grad_dist ) const {
        PointwiseDensity<MomentDensity>{ *this }.integrate_over_simplex_bwd( pts, g, grad_pts, grad_dist );
    }
};

/// a distribution whose pieces carry the moment `K` of their density
template<class Dist,int K>
struct MomentDist {
    static constexpr bool cuts_pieces = [] { if constexpr ( requires { Dist::cuts_pieces; } ) return bool( Dist::cuts_pieces ); else return false; }();

    Dist dist;

    HD void for_each_piece( const auto &cell, auto &&ws, auto &&func ) const {
        dist.for_each_piece( cell, ws, [&]( const auto &pc, const auto &dens ) {
            func( pc, MomentDensity<std::decay_t<decltype( dens )>, std::decay_t<decltype( pc )>::ct_dim, K>{ dens } );
        } );
    }
};

template<int K>
HD auto moment_dist( const auto &dist ) {
    return MomentDist<std::decay_t<decltype( dist )>, K>{ dist };
}

} // namespace diagram
} // namespace sdot
