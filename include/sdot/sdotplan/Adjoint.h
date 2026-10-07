#pragma once

// =====================================================================================
// THE LINEAR ALGEBRA OF THE ADJOINT: what `d weights / d ( seeds, masses, density )` needs of the solver, and nothing else.
//
// At the fitted weights `w`, the residual `F( w, p, nu, rho ) = m( w, p, rho ) - nu` vanishes and its jacobian
// with respect to `w` is the laplacian of the Laguerre graph ( `Laplacian.h` ) -- the very matrix of the last Newton
// step. By the implicit function theorem
//
//     d w = - L^-1 ( d_p F dp + d_rho F drho - d nu )
//
// so the derivative of the solve is ONE solve with that matrix, on the diagram that measured the residual. `laplacian_solve`
// is that solve: `x = L^+ rhs`, the gauge `x_0 = 0` being the linear solver's responsibility. `L` is symmetric: the
// ADJOINT of the map `rhs -> x` is the same map, so the forward and the backward of the call run this one function.
//
// `pd` carries the weights ALREADY ( a diagram built at `w`, its majorants included ): nothing is written on it.
// =====================================================================================

#include "Linear.h"
#include "Sweep.h"

#include <cmath>
#include <limits>

namespace sdot {
namespace sdotplan {

/// `x = L^+ rhs` ( both in user order ), `L` the laplacian of the diagram `pd_in` for the density `dist`. `method`: a `Lin`.
/// A linear solver that fails leaves NaNs in `x`.
template<class TK>
void laplacian_solve( const CpuQueue &queue, const auto &pd_in, const auto &dom, const auto &dist, const auto &rhs, int method, SI cap0,
                      auto &&x ) {
    auto pd = pd_in;                                     // a copy of the VIEWS: `Sweep` wants a writable diagram it will not write
    using PD = DECAYED_TYPE_OF( pd );
    using Dist = DECAYED_TYPE_OF( dist );
    constexpr int D = PD::ct_dim;
    const SI n = pd.nb_seeds();

    Sweep<PD,DECAYED_TYPE_OF( dom ),Dist,TK> bal( queue, pd, pd_in, dom, dist, cap0 );
    std::vector<double> a;
    std::vector<Facet> fa;
    bal.measures( a, &fa );

    Laplacian L;
    L.assemble( n, fa );
    auto lin = linear_solver( Lin( method ), n, D );
    lin->order( bal.rank_of );

    std::vector<double> b( n ), d( n, 0.0 );
    for ( SI i = 0; i < n; ++i )
        b[ i ] = double( rhs( i ) );
    if ( ! lin->solves( L, b, d ) )
        std::fill( d.begin(), d.end(), std::numeric_limits<double>::quiet_NaN() );
    for ( SI i = 0; i < n; ++i )
        x( i ) = d[ i ];
}

} // namespace sdotplan
} // namespace sdot
