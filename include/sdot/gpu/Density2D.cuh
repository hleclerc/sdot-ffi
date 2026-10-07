#pragma once

// =====================================================================================
// THE DENSITIES THE CARD'S CELLS INTEGRATE ( 2D ): what `Cell2D.cuh::finish_cell` and the step of `Newton2D.cuh` call, EDGE
// BY EDGE, on the polygon of a cell -- the mass, the integral along each facet ( the laplacian's `c_ij` ) and the moments.
//
//   * `DensConst`  a constant ( `Problem::rho` ): the cells' own closed forms, the code of before -- nothing here;
//   * `DensImage`  a piecewise-constant image on a regular grid ( `Image` with a diagonal frame and uniform knots );
//   * `DensGauss`  a sum of isotropic gaussians ( `SumOfGaussians`, 2D ), possibly convolved ( the width continuation );
//   * `DensMesh`   a continuous piecewise linear density on a triangle mesh ( `Mesh`, 2D ).
//
// Everything is a CIRCULATION on the boundary of the cell, so a cell is never cut ( the old campaigns: `gpu_des_familles`
// doc/08, `solvers_des_familles` § 9 and § 12 ). The sums are accumulated as if the polygon were counterclockwise; the
// caller multiplies by the sign of its area ( `DensSums` ).
//
// = The image: Green on the rows ( `solvers_des_familles/src/solver/Image.h::arete` )
//
//     mass( P ) = int_P rho = circ_{dP} F0( x, y ) dy,   F0( x, y ) = int_{x0}^{x} rho( t, y ) dt
//
// On row `j`, `rho` does not depend on `y`: `F0 = s0[ j ][ i ] + v[ j ][ i ] ( x - i hx )` with `s0` the prefix sum of the row
// ( made once per density, `DensityHost2D.cuh` ). On a piece of an edge that stays in one pixel, `F0` is affine in `x`: its
// integral in `y` is exact at the middle. The walk is an Amanatides-Woo in the grid, and the same piece gives `int rho ds`
// ( the facet ). A constant subtracted from `F0` changes nothing on a closed polygon: the value at the first vertex is, which
// keeps the terms at the scale of the cell's mass ( three digits otherwise lost ). The moments are the same circulations of
// `F1 = int t rho dt`, `F2 = int t^2 rho dt` ( prefix sums `s1`, `s2` ) and of `y F0`, `y^2 F0`, cubic at most on a piece:
// two Gauss points are exact.
//
// = The gaussians: the polar corner and the boundary fluxes ( `SumOfGaussians.cxx` )
//
// The mass of a polygon is the sum over its edges of the signed standard-normal measure of the corner ( centre, a, b ) --
// `SumOfGaussians::wedge_measure`, the same code; the facet is an `erf` ( `facet_mass` ); the moments are boundary fluxes
// too: `int ( x - c ) phi = -s^2 circ phi n ds` and `int |x - c|^2 phi = 2 s^2 m - s^2 circ ( ( x - c ) . n ) phi ds`, each
// edge's `int phi ds` being the facet's `erf` -- closed forms, where the CPU uses an adaptive quadrature ( the moments only:
// barycentres and cost agree with the CPU to its quadrature, ~1e-4 relative on a narrow gaussian ).
// =====================================================================================

#include <loom/support/common_types.h>
#include <cuda_runtime.h>
#include <cmath>

namespace sdot::gpu2d {

/// the constant density: the cells' own closed forms ( `Problem::rho` / `rho_dev` )
struct DensConst {};

/// a piecewise-constant image: pixel `( i, j )` is `[ x0 + i hx, x0 + ( i + 1 ) hx ] x [ y0 + j hy, ... ]`
struct DensImage {
    double        x0, y0, hx, hy;
    SI            nx, ny;
    const double *v;                                     ///< `v[ j * nx + i ]`
    const double *s0;                                    ///< `s0[ j * ( nx + 1 ) + i ] = hx sum_{k < i} v`
    const double *s1, *s2;                               ///< the same of `X`, `X^2` ( `X = x - x0` ): the moments ( may be null )
};

/// a sum of isotropic gaussians, `w_g` the mass of gaussian `g`, `s_g` its width ( convolved: `sqrt( sigma^2 + s^2 )` )
struct DensGauss {
    SI            nb;
    const double *cx, *cy, *s, *w;
};

/// a piecewise linear density on a triangle mesh ( `Mesh`, DG1 ): the nodes, the values at the corners, the triangles ( node indices ),
/// the gradients of their barycentric coordinates, and the bounding volume hierarchy that finds the triangles a region meets
/// ( `distributions/Mesh.py` ). Everything in doubles.
struct DensMesh {
    SI            nt = 0, nb = 0;                        ///< triangles, hierarchy nodes
    const double *px = nullptr, *py = nullptr;           ///< the nodes
    const double *val = nullptr;                         ///< `val[ 3 e + i ]`: the value of triangle `e` at its corner `i`
    const int    *tri = nullptr;                         ///< `tri[ 3 e + i ]`: the node of corner `i` of triangle `e`
    const double *grad = nullptr;                        ///< `grad[ 6 e + 2 i + c ]`: component `c` of the gradient of `lambda_i`
    const double *blo = nullptr, *bhi = nullptr;         ///< `[ nb, 2 ]`: the boxes of the hierarchy
    const int    *links = nullptr;                       ///< `[ nb, 4 ]`: left, right, begin, end ( a leaf: left < 0 )
};

/// what a cell accumulates, counterclockwise: the mass, and for the moments `int ( x - p ) rho`, `int |x - p|^2 rho` ( `p`
/// the seed ) -- or, for the image, about the image's corner ( `finish` brings them to the seed )
struct DensSums {
    double m = 0, mx = 0, my = 0, m2 = 0;
};

// ---- the image ------------------------------------------------------------------------------------------------------

/// where the references of a cell are read: the pixel of `( X, Y )` ( image frame ), clamped
__device__ __forceinline__ SI image_col( const DensImage &im, double X ) { const double u = floor( X / im.hx ); return u < 0 ? 0 : ( u >= double( im.nx ) ? im.nx - 1 : SI( u ) ); }
__device__ __forceinline__ SI image_row( const DensImage &im, double Y ) { const double u = floor( Y / im.hy ); return u < 0 ? 0 : ( u >= double( im.ny ) ? im.ny - 1 : SI( u ) ); }

/// the constants subtracted from `F0`, `F1`, `F2` for a cell: their values at the pixel of its first vertex
struct ImageRefs {
    double r0 = 0, r1 = 0, r2 = 0;
    bool   set = false;
};

/// ONE EDGE `( X0, Y0 ) -> ( X1, Y1 )` in the image's frame ( `X = x - x0` ): its share of the circulations into `acc`, and
/// `int rho ds` along it ( returned ). `MOM`: the moments too ( about the image's corner ).
template<bool MOM>
__device__ __forceinline__ double image_edge( const DensImage &im, ImageRefs &ref, double X0, double Y0, double X1, double Y1, DensSums &acc ) {
    if ( ! ref.set ) {
        const SI i = image_col( im, X0 ), j = image_row( im, Y0 ), o = j * ( im.nx + 1 ) + i;
        ref.r0 = im.s0[ o ];
        if constexpr ( MOM ) { ref.r1 = im.s1[ o ]; ref.r2 = im.s2[ o ]; }
        ref.set = true;
    }
    const double dX = X1 - X0, dY = Y1 - Y0;
    const double L = sqrt( dX * dX + dY * dY );
    if ( ! ( L > 0 ) )
        return 0;
    const double INF = 1e300;
    const double u0 = X0 / im.hx, w0 = Y0 / im.hy, du = dX / im.hx, dw = dY / im.hy;
    SI i = image_col( im, X0 ), j = image_row( im, Y0 );
    const SI si = du > 0 ? 1 : -1, sj = dw > 0 ? 1 : -1;
    double tx = du == 0 ? INF : ( double( du > 0 ? i + 1 : i ) - u0 ) / du;
    double ty = dw == 0 ? INF : ( double( dw > 0 ? j + 1 : j ) - w0 ) / dw;
    const double ax = du == 0 ? INF : fabs( 1 / du ), ay = dw == 0 ? INF : fabs( 1 / dw );
    tx = tx < 0 ? 0 : tx;
    ty = ty < 0 ? 0 : ty;
    double tp = 0, Xp = X0, line = 0;
    const SI guard = im.nx + im.ny + 4;
    for ( SI g = 0; g < guard; ++g ) {
        const bool by_x = tx < ty;
        double tn = by_x ? tx : ty;
        const bool last = ! ( tn < 1 );
        if ( last ) tn = 1;
        const double Xc = X0 + dX * tn;
        const SI o = j * ( im.nx + 1 ) + i;
        const double v = im.v[ j * im.nx + i ], dt = tn - tp, bx = double( i ) * im.hx;
        // `( xp + xc ) / 2 - i hx` as two differences: the cancellation happens term by term ( the old `arete` )
        acc.m += dt * dY * ( im.s0[ o ] - ref.r0 + v * 0.5 * ( ( Xp - bx ) + ( Xc - bx ) ) );
        line += dt * L * v;
        if constexpr ( MOM ) {
            // two Gauss points on the piece: `F1` is quadratic, `F2` and `y^2 F0` cubic in the parameter
            const double g1 = 0.21132486540518711775, g2 = 0.78867513459481288225;
            const double dYp = dt * dY;
            for ( int q = 0; q < 2; ++q ) {
                const double tau = tp + dt * ( q ? g2 : g1 );
                const double X = X0 + dX * tau, Y = Y0 + dY * tau, xi = X - bx;
                const double F0 = im.s0[ o ] - ref.r0 + v * xi;
                const double F1 = im.s1[ o ] - ref.r1 + v * xi * 0.5 * ( X + bx );
                const double F2 = im.s2[ o ] - ref.r2 + v * xi * ( X * X + X * bx + bx * bx ) / 3;
                acc.mx += 0.5 * dYp * F1;
                acc.my += 0.5 * dYp * Y * F0;
                acc.m2 += 0.5 * dYp * ( F2 + Y * Y * F0 );
            }
        }
        if ( last )
            break;
        if ( by_x ) { i += si; i = i < 0 ? 0 : ( i >= im.nx ? im.nx - 1 : i ); tx += ax; }
        else        { j += sj; j = j < 0 ? 0 : ( j >= im.ny ? im.ny - 1 : j ); ty += ay; }
        tp = tn;
        Xp = Xc;
    }
    return line;
}

// ---- the gaussians ( `SumOfGaussians.cxx`, the same formulas ) ------------------------------------------------------------

/// the signed standard normal measure of the triangle `( 0, P, Q )` ( `SumOfGaussians::wedge_measure` )
__device__ __forceinline__ double gauss_wedge( double Px, double Py, double Qx, double Qy ) {
    const double two_pi = 6.283185307179586476925286766559, tail_cut = 8;
    const double dx = Qx - Px, dy = Qy - Py;
    const double L = sqrt( dx * dx + dy * dy );
    if ( ! ( L > 0 ) )
        return 0;
    const double ux = dx / L, uy = dy / L, nx = uy, ny = -ux;
    const double p = nx * Px + ny * Py, ap = fabs( p );
    if ( ! ( ap > 0 ) )
        return 0;
    const double t0 = ux * Px + uy * Py, t1 = ux * Qx + uy * Qy;
    double acc = 0;
    if ( ap >= tail_cut ) {
        acc = atan( t1 / ap ) - atan( t0 / ap );
    } else {
        if ( t0 < -tail_cut ) {
            const double e = t1 < -tail_cut ? t1 : -tail_cut;
            acc += atan( e / ap ) - atan( t0 / ap );
        }
        if ( t1 > tail_cut ) {
            const double b = t0 > tail_cut ? t0 : tail_cut;
            acc += atan( t1 / ap ) - atan( b / ap );
        }
        const double c0 = t0 > -tail_cut ? t0 : -tail_cut, c1 = t1 < tail_cut ? t1 : tail_cut;
        if ( c1 > c0 ) {
            // composite Gauss-Legendre, 8 points per panel. The CPU takes 4 panels on the core whatever its length; here
            // panels of at most half a ( normalized ) unit, up to the CPU's 4: the same panels on the long edges, ONE ( 8
            // exponentials instead of 32 ) on the short edges of a fine diagram -- whose core is short, so a narrower panel
            // than the CPU's ( measured: one panel of up to 4 units lost 1e-8 of a cell's mass )
            const double gx[ 4 ] = { 0.1834346424956498, 0.5255324099163290, 0.7966664774136267, 0.9602898564975363 };
            const double gw[ 4 ] = { 0.3626837833783620, 0.3137066458778873, 0.2223810344533745, 0.1012285362903763 };
            const int np = min( 4, int( ceil( ( c1 - c0 ) * 2 ) ) );   // 1 .. 4
            const double h = ( c1 - c0 ) / ( 2 * np );           // the half-width of a panel
            for ( int k = 0; k < np; ++k ) {
                const double m = c0 + ( 2 * k + 1 ) * h;
#pragma unroll
                for ( int q = 0; q < 4; ++q ) {
#pragma unroll
                    for ( int sg = -1; sg <= 1; sg += 2 ) {
                        const double t = m + sg * h * gx[ q ];
                        const double r2 = ap * ap + t * t;
                        acc += h * gw[ q ] * ( 1 - exp( -r2 / 2 ) ) * ap / r2;
                    }
                }
            }
        }
    }
    return ( p < 0 ? -acc : acc ) / two_pi;
}

/// the corners of an edge `a -> b` ( seed's frame, the seed at `( ox, oy )` ): `sum_g w_g wedge_g`, SIGNED ( counterclockwise
/// positive ) -- what the frozen cells of the step sum ( `Newton2D.cuh::frozen_mass` )
__device__ __forceinline__ double gauss_corners( const DensGauss &g, double ox, double oy, double ax, double ay, double bx, double by ) {
    double res = 0;
    for ( SI q = 0; q < g.nb; ++q ) {
        const double s = g.s[ q ], cx = g.cx[ q ] - ox, cy = g.cy[ q ] - oy;
        res += g.w[ q ] * gauss_wedge( ( ax - cx ) / s, ( ay - cy ) / s, ( bx - cx ) / s, ( by - cy ) / s );
    }
    return res;
}

/// `int rho ds` along the edge `a -> b` ( `SumOfGaussians::facet_mass` ), and for `MOM` the boundary fluxes of the moments into
/// `acc` ( `mx`, `my`, `m2`; the corners' share is the caller's, per gaussian: `gauss_cell` )
template<bool MOM>
__device__ __forceinline__ double gauss_lines( const DensGauss &g, double ox, double oy, double ax, double ay, double bx, double by, DensSums &acc ) {
    const double ex = bx - ax, ey = by - ay;
    const double L = sqrt( ex * ex + ey * ey );
    if ( ! ( L > 0 ) )
        return 0;
    const double ux = ex / L, uy = ey / L;
    const double nx = uy, ny = -ux;                      // outward, the polygon counterclockwise
    double line = 0;
    for ( SI q = 0; q < g.nb; ++q ) {
        const double s = g.s[ q ], w = g.w[ q ];
        const double cx = g.cx[ q ] - ox, cy = g.cy[ q ] - oy;   // the centre in the seed's frame
        const double Ax = ax - cx, Ay = ay - cy;          // the edge from the centre
        const double d = Ax * ( -uy ) + Ay * ux;           // the signed distance from the centre to the line
        const double t0 = Ax * ux + Ay * uy, t1 = t0 + L;
        const double qq = d * d / ( 2 * s * s );
        if ( qq > 700 )
            continue;                                    // nothing, up to rounding
        const double is2 = 0.70710678118654752440 / s;
        const double u0 = t0 * is2, u1 = t1 * is2;
        if ( u0 > 6.5 || u1 < -6.5 )
            continue;                                    // `erf` is +-1 to the last bit on both ends: `J = 0` exactly
        const double J = w * exp( -qq ) * ( erf( u1 ) - erf( u0 ) ) / ( 2 * s * 2.50662827463100050242 );
        line += J;
        if constexpr ( MOM ) {
            // `int ( x - c ) phi = -s^2 circ phi n ds`, `int |x - c|^2 phi = 2 s^2 m - s^2 circ ( ( x - c ) . n ) phi ds`, brought
            // to the seed ( `c - p` the centre in its frame ): the fluxes here, the `m` terms in `gauss_cell`
            const double s2 = s * s, pe = Ax * nx + Ay * ny;
            acc.mx -= s2 * nx * J;
            acc.my -= s2 * ny * J;
            acc.m2 -= s2 * J * ( pe + 2 * ( cx * nx + cy * ny ) );
        }
    }
    return line;
}

/// THE CORNERS OF A WHOLE CELL, gaussian by gaussian ( `edges( f )` visits its edges `f( ax, ay, bx, by )` in order, seed's
/// frame ): the mass of gaussian `g` is `w_g | sum_edges wedge_g |` -- never negative, as the CPU's `| sum |` per triangle of
/// the fan ( the cells far from a narrow bump weigh `1e-22`: their noise must not be a negative mass ). Adds the masses and,
/// for `MOM`, their share of the moments about the seed to `acc` ( sign of the cell's orientation `sg` for the fluxes already
/// there ).
template<bool MOM,class Edges>
__device__ __forceinline__ void gauss_cell( const DensGauss &g, double ox, double oy, const Edges &edges, DensSums &acc ) {
    for ( SI q = 0; q < g.nb; ++q ) {
        const double s = g.s[ q ], cx = g.cx[ q ] - ox, cy = g.cy[ q ] - oy;
        double W = 0;
        edges( [&]( double ax, double ay, double bx, double by ) {
            W += gauss_wedge( ( ax - cx ) / s, ( ay - cy ) / s, ( bx - cx ) / s, ( by - cy ) / s );
        } );
        const double m = g.w[ q ] * fabs( W );
        acc.m += m;
        if constexpr ( MOM ) {
            acc.mx += cx * m;
            acc.my += cy * m;
            acc.m2 += ( 2 * s * s + cx * cx + cy * cy ) * m;
        }
    }
}

// ---- the mesh ( `Mesh.h`, the same formulas ) ------------------------------------------------------------------------------

/// `lambda_i( x, y )` of triangle `e`
__device__ __forceinline__ double mesh_lambda( const DensMesh &m, SI e, int i, double x, double y ) {
    const int n0 = m.tri[ 3 * e ];
    return ( i == 0 ? 1.0 : 0.0 ) + m.grad[ 6 * e + 2 * i ] * ( x - m.px[ n0 ] ) + m.grad[ 6 * e + 2 * i + 1 ] * ( y - m.py[ n0 ] );
}

/// the pieces `T INTERSECT triangle e` of the polygon `( qx, qy )` ( `nq` vertices, counterclockwise, GLOBAL coordinates ) and
/// their integrals, with the moments about the seed `( ox, oy )` for `MOM`: each convex piece is cut by the three half-planes
/// of the triangle, then split in a fan whose triangles integrate the affine density in closed form
template<bool MOM>
__device__ __forceinline__ void mesh_element( const DensMesh &m, SI e, const double *qx0, const double *qy0, int nq0, double ox, double oy, double sgn, DensSums &acc ) {
    double ax[ 8 ], ay[ 8 ], bx[ 8 ], by[ 8 ];
    int nq = nq0;
    for ( int j = 0; j < nq; ++j ) { ax[ j ] = qx0[ j ]; ay[ j ] = qy0[ j ]; }
    for ( int i = 0; i < 3 && nq >= 3; ++i ) {
        int nn = 0;
        double sj = mesh_lambda( m, e, i, ax[ 0 ], ay[ 0 ] );
        for ( int j = 0; j < nq; ++j ) {
            const int k = j + 1 < nq ? j + 1 : 0;
            const double sk = mesh_lambda( m, e, i, ax[ k ], ay[ k ] );
            if ( sj >= 0 ) { bx[ nn ] = ax[ j ]; by[ nn ] = ay[ j ]; ++nn; }
            if ( ( sj >= 0 ) != ( sk >= 0 ) ) {
                const double t = sj / ( sj - sk );
                bx[ nn ] = ax[ j ] + t * ( ax[ k ] - ax[ j ] );
                by[ nn ] = ay[ j ] + t * ( ay[ k ] - ay[ j ] );
                ++nn;
            }
            sj = sk;
        }
        nq = nn;
        for ( int j = 0; j < nq; ++j ) { ax[ j ] = bx[ j ]; ay[ j ] = by[ j ]; }
    }
    if ( nq < 3 )
        return;

    // the density: `f( x ) = v_0 + grad f . ( x - x_0 )`, `grad f = sum_i v_i G_i`
    const int *t = m.tri + 3 * e;
    const double x0 = m.px[ t[ 0 ] ], y0 = m.py[ t[ 0 ] ], v0 = m.val[ 3 * e ];
    double gfx = 0, gfy = 0;
    for ( int i = 0; i < 3; ++i ) {
        gfx += m.val[ 3 * e + i ] * m.grad[ 6 * e + 2 * i ];
        gfy += m.val[ 3 * e + i ] * m.grad[ 6 * e + 2 * i + 1 ];
    }
    for ( int j = 1; j + 1 < nq; ++j ) {
        const int id[ 3 ] = { 0, j, j + 1 };
        double X[ 3 ], Y[ 3 ], f[ 3 ];
        for ( int c = 0; c < 3; ++c ) {
            X[ c ] = ax[ id[ c ] ] - ox;
            Y[ c ] = ay[ id[ c ] ] - oy;
            f[ c ] = v0 + gfx * ( ax[ id[ c ] ] - x0 ) + gfy * ( ay[ id[ c ] ] - y0 );
        }
        const double T = sgn * 0.5 * ( ( X[ 1 ] - X[ 0 ] ) * ( Y[ 2 ] - Y[ 0 ] ) - ( X[ 2 ] - X[ 0 ] ) * ( Y[ 1 ] - Y[ 0 ] ) );
        const double sf = f[ 0 ] + f[ 1 ] + f[ 2 ];
        acc.m += T * sf / 3;
        if constexpr ( MOM ) {
            // `int lambda_i lambda_j = ( 1 + [ i == j ] ) |T| / 12`, `int lambda_i lambda_j lambda_k = ( 6, 2, 1 ) |T| / 60`
            for ( int a = 0; a < 3; ++a ) {
                acc.mx += T / 12 * X[ a ] * ( sf + f[ a ] );
                acc.my += T / 12 * Y[ a ] * ( sf + f[ a ] );
            }
            double s2 = 0;
            for ( int a = 0; a < 3; ++a )
                for ( int b = 0; b < 3; ++b ) {
                    const double ab = X[ a ] * X[ b ] + Y[ a ] * Y[ b ];
                    for ( int c = 0; c < 3; ++c ) {
                        const int same = ( a == b ) + ( b == c ) + ( a == c );
                        s2 += ab * f[ c ] * ( same == 3 ? 6.0 : same == 1 ? 2.0 : 1.0 );
                    }
                }
            acc.m2 += T / 60 * s2;
        }
    }
}

/// the hierarchy walked with the box `[ lo, hi ]`: `body( e )` for each triangle in a leaf it meets
template<class F>
__device__ __forceinline__ void mesh_walk( const DensMesh &m, double lox, double loy, double hix, double hiy, F &&body ) {
    if ( m.nb == 0 )
        return;
    // the box a little WIDER: a triangle that only touches the region -- along a side it may own ( `mesh_owns` ) -- must not be
    // lost to a rounding of the region's coordinates
    const double mx = 1e-9 * ( hix - lox ) + 1e-12 * fmax( fabs( lox ), fabs( hix ) ), my = 1e-9 * ( hiy - loy ) + 1e-12 * fmax( fabs( loy ), fabs( hiy ) );
    lox -= mx; hix += mx; loy -= my; hiy += my;
    SI stack[ 64 ];
    int top = 0;
    stack[ top++ ] = 0;
    while ( top ) {
        const SI n = stack[ --top ];
        if ( m.bhi[ 2 * n ] < lox || m.blo[ 2 * n ] > hix || m.bhi[ 2 * n + 1 ] < loy || m.blo[ 2 * n + 1 ] > hiy )
            continue;
        const int left = m.links[ 4 * n ];
        if ( left >= 0 ) {
            stack[ top++ ] = m.links[ 4 * n + 1 ];
            stack[ top++ ] = left;
            continue;
        }
        for ( SI e = m.links[ 4 * n + 2 ]; e < m.links[ 4 * n + 3 ]; ++e )
            body( e );
    }
}

/// THE TRIANGLE `( a, b, c )` ( the seed's frame, the seed at `( ox, oy )` ) against the mesh: its SIGNED share of the mass
/// ( the sign of its orientation ) and, for `MOM`, of the moments about the seed, into `acc`
template<bool MOM>
__device__ __forceinline__ void mesh_triangle( const DensMesh &m, double ox, double oy, double Xa, double Ya, double Xb, double Yb, double Xc, double Yc, DensSums &acc ) {
    double sgn = 1;
    const double area2 = ( Xb - Xa ) * ( Yc - Ya ) - ( Xc - Xa ) * ( Yb - Ya );
    if ( ! ( area2 != 0 ) )
        return;
    if ( area2 < 0 ) {
        const double tx = Xb, ty = Yb;
        Xb = Xc; Yb = Yc; Xc = tx; Yc = ty;
        sgn = -1;
    }
    const double qx[ 3 ] = { ox + Xa, ox + Xb, ox + Xc }, qy[ 3 ] = { oy + Ya, oy + Yb, oy + Yc };
    mesh_walk( m, fmin( qx[ 0 ], fmin( qx[ 1 ], qx[ 2 ] ) ), fmin( qy[ 0 ], fmin( qy[ 1 ], qy[ 2 ] ) ),
                  fmax( qx[ 0 ], fmax( qx[ 1 ], qx[ 2 ] ) ), fmax( qy[ 0 ], fmax( qy[ 1 ], qy[ 2 ] ) ),
               [&]( SI e ) { mesh_element<MOM>( m, e, qx, qy, 3, ox, oy, sgn, acc ); } );
}

/// WHO COUNTS AN EDGE THAT LIES ON A SIDE OF TRIANGLE `e` ( `Mesh.h::owns_facet`, the same rule ): the edge is held by the triangle
/// across that side as well, and is counted once, decided on the NODES with one strict inequality -- a node is outside the edge's line
/// when `n . x - off > 0`, inside otherwise, the normal in a canonical orientation. The triangle counts the edge iff the line separates
/// its nodes: 1 or 0, never one half
__device__ __forceinline__ bool mesh_owns( const DensMesh &m, SI e, double Ax, double Ay, double Bx, double By ) {
    double nx = By - Ay, ny = Ax - Bx;                   // the normal of the line
    const double nn = sqrt( nx * nx + ny * ny );
    const double sg = fabs( nx ) > 1e-9 * nn ? ( nx < 0 ? -1.0 : 1.0 ) : ( ny < 0 ? -1.0 : 1.0 );
    const double off = nx * Ax + ny * Ay;
    double s[ 3 ], scale = fabs( off );
    const int *t = m.tri + 3 * e;
    for ( int i = 0; i < 3; ++i ) {
        const double dot = nx * m.px[ t[ i ] ] + ny * m.py[ t[ i ] ];
        s[ i ] = sg * ( dot - off );
        scale = fmax( scale, fabs( dot ) );
    }
    const double tol = 1e-11 * scale;
    bool out = false, in = false;
    for ( int i = 0; i < 3; ++i ) {
        out = out || s[ i ] > tol;
        in  = in  || ! ( s[ i ] > tol );
    }
    return out && in;
}

/// `int rho ds` along the edge `a -> b` ( the seed's frame ): the segment cut by each triangle, `f` affine on each piece. An
/// edge ON the boundary of two triangles ( on a mesh line ) is in both, and counted by one of them ( `mesh_owns` )
__device__ __forceinline__ double mesh_line( const DensMesh &m, double ox, double oy, double ax, double ay, double bx, double by ) {
    const double Ax = ox + ax, Ay = oy + ay, Bx = ox + bx, By = oy + by;
    const double L = sqrt( ( bx - ax ) * ( bx - ax ) + ( by - ay ) * ( by - ay ) );
    if ( ! ( L > 0 ) )
        return 0;
    const double eps = 1e-10;
    double res = 0;
    mesh_walk( m, fmin( Ax, Bx ), fmin( Ay, By ), fmax( Ax, Bx ), fmax( Ay, By ), [&]( SI e ) {
        double t0 = 0, t1 = 1, w = 1;
        for ( int i = 0; i < 3 && t0 < t1; ++i ) {
            const double sa = mesh_lambda( m, e, i, Ax, Ay ), sb = mesh_lambda( m, e, i, Bx, By );
            if ( sa < 0 && sb < 0 ) { t1 = t0; break; }
            if ( fabs( sa ) < eps && fabs( sb ) < eps ) w = mesh_owns( m, e, Ax, Ay, Bx, By ) ? 1.0 : 0.0;   // the segment is on this side
            else if ( sa < 0 )  t0 = fmax( t0, sa / ( sa - sb ) );
            else if ( sb < 0 )  t1 = fmin( t1, sa / ( sa - sb ) );
        }
        if ( ! ( t1 > t0 ) )
            return;
        const int *t = m.tri + 3 * e;
        const double x0 = m.px[ t[ 0 ] ], y0 = m.py[ t[ 0 ] ];
        double gfx = 0, gfy = 0;
        for ( int i = 0; i < 3; ++i ) {
            gfx += m.val[ 3 * e + i ] * m.grad[ 6 * e + 2 * i ];
            gfy += m.val[ 3 * e + i ] * m.grad[ 6 * e + 2 * i + 1 ];
        }
        const double fa = m.val[ 3 * e ] + gfx * ( Ax + t0 * ( Bx - Ax ) - x0 ) + gfy * ( Ay + t0 * ( By - Ay ) - y0 );
        const double fb = m.val[ 3 * e ] + gfx * ( Ax + t1 * ( Bx - Ax ) - x0 ) + gfy * ( Ay + t1 * ( By - Ay ) - y0 );
        res += w * L * ( t1 - t0 ) * 0.5 * ( fa + fb );
    } );
    return res;
}

// ---- the dispatch: one edge of a cell, whatever the density ------------------------------------------------------------

/// what a cell keeps from one edge to the next ( the image's references )
template<class D> struct DensState {};
template<> struct DensState<DensImage> { ImageRefs ref; };
/// the mesh: the first vertex of the cell, the apex of the fan of triangles that its mass is made of
template<> struct DensState<DensMesh> { double x0 = 0, y0 = 0; bool set = false; };

/// ONE EDGE of a FROZEN cell ( the step ) of the seed `( ox, oy )`, `a -> b` in the seed's frame: its SIGNED share of the mass
/// ( counterclockwise positive: a cell that turns inside out weighs less than nothing )
template<class D>
__device__ __forceinline__ void density_edge_mass( const D &dens, DensState<D> &st, double ox, double oy, double ax, double ay, double bx, double by,
                                                   DensSums &acc ) {
    if constexpr ( std::is_same_v<D,DensImage> ) {
        const double Ox = ox - dens.x0, Oy = oy - dens.y0;
        image_edge<false>( dens, st.ref, Ox + ax, Oy + ay, Ox + bx, Oy + by, acc );
    } else if constexpr ( std::is_same_v<D,DensMesh> ) {
        if ( ! st.set ) { st.x0 = ax; st.y0 = ay; st.set = true; }
        mesh_triangle<false>( dens, ox, oy, st.x0, st.y0, ax, ay, bx, by, acc );
    } else
        acc.m += gauss_corners( dens, ox, oy, ax, ay, bx, by );
}

/// THE IMAGE'S sums of a counterclockwise walk turned into the cell's: `sg` the sign of its area; the moments about the seed
/// ( `mx`, `my`, `m2` = the cost of the cell )
__device__ __forceinline__ DensSums image_finish( const DensImage &dens, DensSums acc, double sg, double ox, double oy ) {
    DensSums r;
    r.m = sg * acc.m;
    // about the image's corner -> about the seed `P`
    const double Px = ox - dens.x0, Py = oy - dens.y0;
    const double mx = sg * acc.mx, my = sg * acc.my, m2 = sg * acc.m2;
    r.mx = mx - Px * r.m;
    r.my = my - Py * r.m;
    r.m2 = m2 - 2 * ( Px * mx + Py * my ) + ( Px * Px + Py * Py ) * r.m;
    return r;
}

} // namespace sdot::gpu2d
