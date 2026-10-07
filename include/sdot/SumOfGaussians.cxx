#pragma once

#include <loom/support/common_macros.h>
#include <loom/support/containers/Vector.h>
#include "SumOfGaussians.h"
#include <loom/support/math.h>

// Mathematics goes through `sdot::` (`loom/support/math.h`), NEVER through `std::`.
//
// It is not a style preference: `std::exp` / `std::atan` / `std::erf` on a `float`
// do not exist in CUDA device code (they lower to intrinsics that the device compiler
// cannot resolve), whereas the toolkit's overloads do exist. `sdot::` designates
// one or the other depending on the target, in a single place.
//
// `sqrt` is included even though it works everywhere (it is a native instruction): a rule that
// tolerates an exception is not a rule that gets followed.

#define UTP SDOT_TEMPLATE_DECL_FOR_SumOfGaussians
#define DTP SumOfGaussians<SDOT_TEMPLATE_ARGS_FOR_SumOfGaussians>

namespace sdot {

UTP HD auto DTP::kernel_at( SI i, const auto &x ) const {
    struct Kernel { TF phi; TF r2; TF s; };

    const TF s = sigma_of( i );
    TF r2 = 0;
    for ( PI c = 0; c < ct_dim; ++c ) {
        const TF e = TF( x[ c ] ) - center_of( i, c );
        r2 += e * e;
    }

    // `( 2 pi s^2 ) ^ ( -d/2 )`, written as an integer power of `1 / ( s sqrt( 2 pi ) )`: d
    // multiplications instead of a `pow`, and `ct_dim` being known at compile time the loop
    // disappears.
    const TF two_pi = TF( 6.283185307179586476925286766559 );
    const TF inv = TF( 1 ) / ( s * sdot::sqrt( two_pi ) );
    TF norm = 1;
    for ( int k = 0; k < ct_dim; ++k )
        norm *= inv;

    return Kernel{ norm * sdot::exp( - r2 / ( 2 * s * s ) ), r2, s };
}

UTP HD typename DTP::TF DTP::value_at( const auto &x ) const {
    const SI n = nb_terms();
    TF res = 0;
    for ( SI i = 0; i < n; ++i )
        res += weight_of( i ) * kernel_at( i, x ).phi;
    return res;
}

UTP HD auto DTP::gradient_at( const auto &x ) const {
    // `d/dx exp( -r^2 / 2s^2 ) = - ( x - c ) / s^2 * ...`: the gradient of a gaussian points towards
    // its center, with the factor `1 / s^2`.
    const SI n = nb_terms();
    auto res = Vector<TF,ct_dim>::zeros();
    for ( SI i = 0; i < n; ++i ) {
        const auto k = kernel_at( i, x );
        const TF f = weight_of( i ) * k.phi / ( k.s * k.s );
        for ( PI c = 0; c < ct_dim; ++c )
            res[ c ] -= f * ( TF( x[ c ] ) - center_of( i, c ) );
    }
    return res;
}

UTP HD void DTP::add_value_grad_at( auto &&grad_dist, const auto &x, TF g ) const {
    auto add_to = []( auto &&dst, TF v ) {
        if constexpr ( ! DECAYED_TYPE_OF( dst )::surely_null )
            atomic_add( dst.ref(), v );
    };

    // nothing requested: not a read, not an exponential. The test is at COMPILE time, so
    // a pure forward does not pay for the existence of this block.
    if constexpr ( DECAYED_TYPE_OF( grad_dist.weights )::surely_null
                && DECAYED_TYPE_OF( grad_dist.positions )::surely_null
                && DECAYED_TYPE_OF( grad_dist.sigmas )::surely_null ) {
        return;
    } else {
        const SI n = nb_gaussians;
        for ( SI i = 0; i < n; ++i ) {
            const auto k = kernel_at( i, x );
            const TF w = TF( weights( i ) );

            // d rho / d w_i = the normalized kernel
            add_to( grad_dist.weights( i ), g * k.phi );

            // d rho / d c_i = w_i * phi * ( x - c_i ) / s^2   ( the gradient AT x, sign flipped )
            if constexpr ( ! DECAYED_TYPE_OF( grad_dist.positions )::surely_null ) {
                const TF f = g * w * k.phi / ( k.s * k.s );
                for ( PI c = 0; c < ct_dim; ++c )
                    add_to( grad_dist.positions( i, c ), f * ( TF( x[ c ] ) - TF( positions( i, c ) ) ) );
            }

            // d rho / d s_i = w_i * phi * ( r^2 / s^3 - d / s ): the first term comes from
            // the exponential, the second from the normalization constant `s^-d`.
            add_to( grad_dist.sigmas( i ),
                    g * w * k.phi * ( k.r2 / ( k.s * k.s * k.s ) - TF( ct_dim ) / k.s ) * ( TF( sigmas( i ) ) / k.s ) );
        }
    }
}

// ---- exact integration in 2D ------------------------------------------------------------------
// See `SumOfGaussians.h` for the reduction. Here, the 8 Gauss-Legendre nodes (symmetric,
// given as a half-table) and the four methods.

namespace detail {
    // 8-point Gauss-Legendre on [ -1, 1 ], positive half
    LOOM_CONSTANT( double gl8_x[ 4 ] ) = { 0.1834346424956498, 0.5255324099163290,
                                           0.7966664774136267, 0.9602898564975363 };
    LOOM_CONSTANT( double gl8_w[ 4 ] ) = { 0.3626837833783620, 0.3137066458778873,
                                           0.2223810344533745, 0.1012285362903763 };

    // `Phi`, the standard normal cumulative distribution function
    template<class TF> HD TF std_normal_cdf( TF u ) {
        return TF( 0.5 ) * ( 1 + sdot::erf( u * TF( 0.70710678118654752440 ) ) );
    }
}

UTP HD typename DTP::TF DTP::facet_mass( const auto &pc, int cut ) const requires ( ct_dim == 2 || ct_dim == 3 ) {
    if constexpr ( ct_dim == 2 )
        return facet_mass_2d( pc, cut );
    else
        return facet_mass_3d( pc, cut );
}

UTP HD typename DTP::TF DTP::facet_mass_2d( const auto &pc, int cut ) const {
    const int nb = pc.nb_vertices();
    const int j = cut + 1 < nb ? cut + 1 : 0;
    const TF ax = TF( pc.coord( cut, 0 ) ), ay = TF( pc.coord( cut, 1 ) );
    const TF ex = TF( pc.coord( j, 0 ) ) - ax, ey = TF( pc.coord( j, 1 ) ) - ay;
    const TF L = sdot::sqrt( ex * ex + ey * ey );
    if ( ! ( L > 0 ) )
        return 0;
    const TF ux = ex / L, uy = ey / L;                    // the tangent, and a normal
    const TF nx = -uy, ny = ux;
    TF res = 0;
    const SI ng = nb_terms();
    for ( SI i = 0; i < ng; ++i ) {
        const TF s = sigma_of( i ), m = weight_of( i );
        const TF px = ax - center_of( i, 0 ), py = ay - center_of( i, 1 );
        const TF d = px * nx + py * ny;                   // the signed distance from the center to the line
        const TF t0 = px * ux + py * uy, t1 = t0 + L;
        const TF q = d * d / ( 2 * s * s );
        if ( q > TF( 700 ) )
            continue;                                    // nothing, up to rounding
        const TF is2 = TF( 0.70710678118654752440 ) / s;
        const TF E = sdot::erf( t1 * is2 ) - sdot::erf( t0 * is2 );
        res += m * sdot::exp( -q ) * E / ( 2 * s * TF( 2.50662827463100050242 ) );   // `sqrt( 2 pi )`
    }
    return res;
}

UTP HD typename DTP::TF DTP::facet_mass_3d( const auto &pc, int cut ) const {
    // the unit normal of the face: the sum of the cross products of its fan, all of the same orientation ( a convex face )
    const auto pt = [&]( int i ) { return Vector<TF,3>( Function(), [&]( PI c ) { return TF( pc.coord( i, int( c ) ) ); } ); };
    TF N[ 3 ] = { 0, 0, 0 };
    int o0 = -1;                                         // a vertex of the face: the distance to its plane is read there
    pc.for_each_facet_triangle( cut, [&]( int o, int a, int b ) {
        o0 = o;
        const auto O = pt( o ), A = pt( a ), B = pt( b );
        const TF ax = A[ 0 ] - O[ 0 ], ay = A[ 1 ] - O[ 1 ], az = A[ 2 ] - O[ 2 ];
        const TF bx = B[ 0 ] - O[ 0 ], by = B[ 1 ] - O[ 1 ], bz = B[ 2 ] - O[ 2 ];
        TF cx = ay * bz - az * by, cy = az * bx - ax * bz, cz = ax * by - ay * bx;
        if ( N[ 0 ] * cx + N[ 1 ] * cy + N[ 2 ] * cz < 0 ) { cx = -cx; cy = -cy; cz = -cz; }
        N[ 0 ] += cx; N[ 1 ] += cy; N[ 2 ] += cz;
    } );
    const TF nn = sdot::sqrt( N[ 0 ] * N[ 0 ] + N[ 1 ] * N[ 1 ] + N[ 2 ] * N[ 2 ] );
    if ( ! ( nn > 0 ) )
        return 0;
    const TF n[ 3 ] = { N[ 0 ] / nn, N[ 1 ] / nn, N[ 2 ] / nn };

    // an orthonormal frame `( u, w )` of the plane: `u` from the axis `n` is the least aligned with
    const TF an[ 3 ] = { sdot::fabs( n[ 0 ] ), sdot::fabs( n[ 1 ] ), sdot::fabs( n[ 2 ] ) };
    const int k = an[ 0 ] < an[ 1 ] ? ( an[ 0 ] < an[ 2 ] ? 0 : 2 ) : ( an[ 1 ] < an[ 2 ] ? 1 : 2 );
    TF u[ 3 ] = { -n[ k ] * n[ 0 ], -n[ k ] * n[ 1 ], -n[ k ] * n[ 2 ] };
    u[ k ] += 1;
    const TF nu = sdot::sqrt( u[ 0 ] * u[ 0 ] + u[ 1 ] * u[ 1 ] + u[ 2 ] * u[ 2 ] );
    for ( int c = 0; c < 3; ++c )
        u[ c ] /= nu;
    const TF w[ 3 ] = { n[ 1 ] * u[ 2 ] - n[ 2 ] * u[ 1 ], n[ 2 ] * u[ 0 ] - n[ 0 ] * u[ 2 ], n[ 0 ] * u[ 1 ] - n[ 1 ] * u[ 0 ] };

    // `rho = w_i * phi_1( d / s ) / s * phi_2( y / s ) / s^2`: the 1D factor across the plane, times the 2D standard measure of
    // the face in the scaled in-plane coordinates -- the sum over the fan, each triangle positive
    const auto P0 = pt( o0 );
    TF res = 0;
    const SI ng = nb_terms();
    for ( SI i = 0; i < ng; ++i ) {
        const TF s = sigma_of( i );
        const TF c[ 3 ] = { center_of( i, 0 ), center_of( i, 1 ), center_of( i, 2 ) };
        const TF d = ( P0[ 0 ] - c[ 0 ] ) * n[ 0 ] + ( P0[ 1 ] - c[ 1 ] ) * n[ 1 ] + ( P0[ 2 ] - c[ 2 ] ) * n[ 2 ];
        const TF q = d * d / ( 2 * s * s );
        if ( q > TF( 700 ) )
            continue;                                    // nothing, up to rounding
        const auto proj = [&]( int v ) {
            const auto P = pt( v );
            const TF e[ 3 ] = { P[ 0 ] - c[ 0 ], P[ 1 ] - c[ 1 ], P[ 2 ] - c[ 2 ] };
            return Vector<TF,2>( Values(), ( e[ 0 ] * u[ 0 ] + e[ 1 ] * u[ 1 ] + e[ 2 ] * u[ 2 ] ) / s,
                                           ( e[ 0 ] * w[ 0 ] + e[ 1 ] * w[ 1 ] + e[ 2 ] * w[ 2 ] ) / s );
        };
        TF acc = 0;
        pc.for_each_facet_triangle( cut, [&]( int o, int a, int b ) {
            acc += std_triangle_measure( Vector<Vector<TF,2>,3>( Values(), proj( o ), proj( a ), proj( b ) ) );
        } );
        res += weight_of( i ) * sdot::exp( -q ) * acc / ( s * TF( 2.50662827463100050242 ) );   // `sqrt( 2 pi )`
    }
    return res;
}

UTP HD typename DTP::TF DTP::wedge_measure( const auto &P, const auto &Q ) const {
    const TF two_pi = TF( 6.283185307179586476925286766559 );

    const TF dx = Q[ 0 ] - P[ 0 ], dy = Q[ 1 ] - P[ 1 ];
    const TF L = sdot::sqrt( dx * dx + dy * dy );
    if ( ! ( L > 0 ) )
        return 0;

    // `( n, u )` DIRECT, so that `cross( P, Q ) = p * L`: the sign of `p` is that of the area
    // of the corner, and `t` grows from `P` towards `Q`.
    const TF ux = dx / L, uy = dy / L;
    const TF nx = uy, ny = -ux;
    const TF p = nx * P[ 0 ] + ny * P[ 1 ];
    const TF ap = p < 0 ? -p : p;
    if ( ! ( ap > 0 ) )                     // the origin IS on the line: flat corner
        return 0;

    const TF t0 = ux * P[ 0 ] + uy * P[ 1 ];
    const TF t1 = ux * Q[ 0 ] + uy * Q[ 1 ];

    TF acc = 0;
    if ( ap >= tail_cut ) {
        // the gaussian is worth nothing along the whole line: only the Lorentzian remains
        acc = sdot::atan( t1 / ap ) - sdot::atan( t0 / ap );
    } else {
        // The TAILS first, each bounded by the segment itself: a segment entirely
        // beyond `tail_cut` on one side only has no core at all, and its tail goes from `t0` to
        // `t1`, not from `tail_cut` to `t1`. Clipping "symmetrically" would count `[ tail_cut, t0 ]`
        // too much -- a clearly visible share of angle when the edge is long and grazes the origin.
        if ( t0 < -tail_cut ) {
            const TF e = t1 < -tail_cut ? t1 : -tail_cut;
            acc += sdot::atan( e / ap ) - sdot::atan( t0 / ap );
        }
        if ( t1 > tail_cut ) {
            const TF b = t0 > tail_cut ? t0 : tail_cut;
            acc += sdot::atan( t1 / ap ) - sdot::atan( b / ap );
        }

        const TF c0 = t0 > -tail_cut ? t0 : -tail_cut;
        const TF c1 = t1 <  tail_cut ? t1 :  tail_cut;

        // the core, by composite Gauss-Legendre. The integrand has a scale `>= 1` there (the factor
        // `1 - exp` cancels the Lorentzian peak when `p` is small), so a few panels
        // suffice whatever the configuration.
        if ( c1 > c0 ) {
            const TF h = ( c1 - c0 ) / ( 2 * nb_panels );
            for ( int k = 0; k < nb_panels; ++k ) {
                const TF m = c0 + ( 2 * k + 1 ) * h;
                for ( int j = 0; j < 4; ++j ) {
                    for ( int sg = -1; sg <= 1; sg += 2 ) {
                        const TF t = m + sg * h * TF( detail::gl8_x[ j ] );
                        const TF r2 = ap * ap + t * t;
                        acc += h * TF( detail::gl8_w[ j ] ) * ( 1 - sdot::exp( - r2 / 2 ) ) * ap / r2;
                    }
                }
            }
        }
    }

    return ( p < 0 ? -acc : acc ) / two_pi;
}

UTP HD typename DTP::TF DTP::std_triangle_measure( const auto &ys ) const {
    const TF s = wedge_measure( ys[ 0 ], ys[ 1 ] )
               + wedge_measure( ys[ 1 ], ys[ 2 ] )
               + wedge_measure( ys[ 2 ], ys[ 0 ] );
    // the signed sum carries the ORIENTATION of the triangle; the measure has none.
    return s < 0 ? -s : s;
}

UTP HD typename DTP::EdgeInfo DTP::edge_info( const auto &A, const auto &B, const auto &C ) const {
    const TF two_pi = TF( 6.283185307179586476925286766559 );
    const TF sq_2pi = TF( 2.5066282746310005024157652848110 );

    EdgeInfo res{ Vector<TF,2>::zeros(), 0, 0, 0 };

    const TF dx = B[ 0 ] - A[ 0 ], dy = B[ 1 ] - A[ 1 ];
    const TF L2 = dx * dx + dy * dy;
    if ( ! ( L2 > 0 ) )
        return res;
    const TF L = sdot::sqrt( L2 );

    // the OUTWARD normal: the one that points away from the third vertex
    TF nx = dy / L, ny = -dx / L;
    if ( nx * ( C[ 0 ] - A[ 0 ] ) + ny * ( C[ 1 ] - A[ 1 ] ) > 0 ) { nx = -nx; ny = -ny; }
    res.n[ 0 ] = nx;
    res.n[ 1 ] = ny;
    res.p = nx * A[ 0 ] + ny * A[ 1 ];      // constant along the edge

    // `| A + s ( B - A ) |^2 = L^2 ( s - s0 )^2 + p^2`: the foot of the perpendicular, and the
    // distance to the line. That is what pulls out an `exp( -p^2/2 )` as a factor and leaves a
    // 1D gaussian, hence `erf`s.
    const TF s0 = - ( A[ 0 ] * dx + A[ 1 ] * dy ) / L2;
    TF p2 = A[ 0 ] * A[ 0 ] + A[ 1 ] * A[ 1 ] - L2 * s0 * s0;
    if ( p2 < 0 ) p2 = 0;

    const TF u0 = - L * s0, u1 = L * ( 1 - s0 );
    const TF d_phi = detail::std_normal_cdf( u1 ) - detail::std_normal_cdf( u0 );
    const TF e = sdot::exp( - p2 / 2 );

    res.j0  = e * d_phi / sq_2pi;
    res.j1a = ( e / two_pi ) * ( ( 1 - s0 ) * sq_2pi * d_phi
                               - ( sdot::exp( - u0 * u0 / 2 ) - sdot::exp( - u1 * u1 / 2 ) ) / L );
    return res;
}

UTP HD typename DTP::TF DTP::integrate_over_simplex( const auto &pts ) const {
    static_assert( ct_dim == 2, "the exact path is 2D; beyond that we go through PointwiseDensity" );

    // the diameter of the triangle, for the choice of the rule ( during a continuation only: at `s = 0` the integral is exact )
    TF diam2 = 0;
    if ( terms && conv_s > 0 )
        for ( int a = 0; a < 3; ++a )
            for ( int b = a + 1; b < 3; ++b ) {
                const TF dx = pts[ a ][ 0 ] - pts[ b ][ 0 ], dy = pts[ a ][ 1 ] - pts[ b ][ 1 ];
                diam2 = sdot::fmax( diam2, dx * dx + dy * dy );
            }

    const SI n = nb_terms();
    TF res = 0;
    for ( SI i = 0; i < n; ++i ) {
        const TF s = sigma_of( i );
        if ( diam2 > 0 && s * s > quadrature_ratio * quadrature_ratio * diam2 ) {
            res += quadrature_term( i, pts );
            continue;
        }
        const auto ys = Vector<Vector<TF,2>,3>( Function(), [&]( PI k ) {
            return Vector<TF,2>( Function(), [&]( PI c ) { return ( pts[ k ][ c ] - center_of( i, c ) ) / s; } );
        } );
        res += weight_of( i ) * std_triangle_measure( ys );
    }
    return res;
}

UTP HD typename DTP::TF DTP::quadrature_term( SI i, const auto &pts ) const {
    // Dunavant's degree 5 rule: the centroid, and two orbits of three points ( barycentric `( a, b, b )` )
    const TF ax = pts[ 1 ][ 0 ] - pts[ 0 ][ 0 ], ay = pts[ 1 ][ 1 ] - pts[ 0 ][ 1 ];
    const TF bx = pts[ 2 ][ 0 ] - pts[ 0 ][ 0 ], by = pts[ 2 ][ 1 ] - pts[ 0 ][ 1 ];
    TF area = ( ax * by - ay * bx ) / 2;
    area = area < 0 ? -area : area;
    const TF s = sigma_of( i );
    const TF cx = center_of( i, 0 ), cy = center_of( i, 1 );
    auto f = [&]( TF l0, TF l1, TF l2 ) {
        const TF x = l0 * pts[ 0 ][ 0 ] + l1 * pts[ 1 ][ 0 ] + l2 * pts[ 2 ][ 0 ] - cx;
        const TF y = l0 * pts[ 0 ][ 1 ] + l1 * pts[ 1 ][ 1 ] + l2 * pts[ 2 ][ 1 ] - cy;
        return sdot::exp( - ( x * x + y * y ) / ( 2 * s * s ) );
    };
    const TF a1 = TF( 0.059715871789770 ), b1 = TF( 0.470142064105115 );
    const TF a2 = TF( 0.797426985353087 ), b2 = TF( 0.101286507323456 );
    const TF acc = TF( 0.225 ) * f( TF( 1 ) / 3, TF( 1 ) / 3, TF( 1 ) / 3 )
                 + TF( 0.132394152788506 ) * ( f( a1, b1, b1 ) + f( b1, a1, b1 ) + f( b1, b1, a1 ) )
                 + TF( 0.125939180544827 ) * ( f( a2, b2, b2 ) + f( b2, a2, b2 ) + f( b2, b2, a2 ) );
    return weight_of( i ) * area * acc / ( TF( 6.283185307179586476925286766559 ) * s * s );
}

UTP HD void DTP::integrate_over_simplex_bwd( const auto &pts, TF g, auto &&grad_pts, auto &&grad_dist ) const {
    static_assert( ct_dim == 2, "the exact path is 2D; beyond that we go through PointwiseDensity" );

    auto add_to = []( auto &&dst, TF v ) {
        if constexpr ( ! DECAYED_TYPE_OF( dst )::surely_null )
            atomic_add( dst.ref(), v );
    };

    const SI n = nb_gaussians;
    for ( SI i = 0; i < n; ++i ) {
        const TF s = sigma_of( i );
        const TF w = TF( weights( i ) );
        const auto ys = Vector<Vector<TF,2>,3>( Function(), [&]( PI k ) {
            return Vector<TF,2>( Function(), [&]( PI c ) { return ( pts[ k ][ c ] - TF( positions( i, c ) ) ) / s; } );
        } );

        // one pass over the three edges: each one pays into its TWO vertices (with the
        // barycentric weight that is 1 at one and 0 at the other), and into sigma via `y . n`.
        auto dm = Vector<Vector<TF,2>,3>( Function(), []( PI ) { return Vector<TF,2>::zeros(); } );
        TF dsig = 0;
        for ( SI k = 0; k < 3; ++k ) {
            const SI k1 = ( k + 1 ) % 3, k2 = ( k + 2 ) % 3;
            const auto ei = edge_info( ys[ k ], ys[ k1 ], ys[ k2 ] );
            for ( PI c = 0; c < 2; ++c ) {
                dm[ k  ][ c ] += ei.j1a * ei.n[ c ];
                dm[ k1 ][ c ] += ( ei.j0 - ei.j1a ) * ei.n[ c ];
            }
            dsig += ei.p * ei.j0;
        }

        // the vertices, then the center -- which is MINUS their sum (translating the triangle and the
        // gaussian together changes nothing), which avoids a second differentiation.
        const TF f = g * w / s;
        auto dc = Vector<TF,2>::zeros();
        for ( SI k = 0; k < 3; ++k ) {
            for ( PI c = 0; c < 2; ++c ) {
                grad_pts[ k ][ c ] += f * dm[ k ][ c ];
                dc[ c ] -= f * dm[ k ][ c ];
            }
        }

        if constexpr ( ! DECAYED_TYPE_OF( grad_dist.positions )::surely_null )
            for ( PI c = 0; c < 2; ++c )
                add_to( grad_dist.positions( i, c ), dc[ c ] );

        add_to( grad_dist.sigmas( i ), - f * dsig * ( TF( sigmas( i ) ) / s ) );   // `d sigma' / d sigma`
        add_to( grad_dist.weights( i ), g * std_triangle_measure( ys ) );
    }
}

#undef UTP
#undef DTP

}
