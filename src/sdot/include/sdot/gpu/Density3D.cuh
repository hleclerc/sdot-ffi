#pragma once

// =====================================================================================
// THE TETRAHEDRAL MESH DENSITY OF THE CARD'S 3D CELLS ( `Mesh`, DG1: a value per corner of each tetrahedron, affine on it ):
// what `Cell3D.cuh::finish_warp` integrates on a cell -- the mass, the integral on each facet ( the laplacian's `c_ij` ) and
// the moments. The 2D sibling is `Density2D.cuh::DensMesh`.
//
// = A CELL AGAINST THE MESH: CONES ON THE BOUNDARY
//
// `int_{ C INTERSECT T } f`, for the cell `C` and an element `T` ( `f` the affine function of `T`, over all the space ), is the sum
// over the faces of `C INTERSECT T` of the signed cones from ONE apex, `g` the mean of the cell's vertices -- and these faces are
//
//   * the faces of `T` clipped by the planes of `C` that cut it ( one to three: the element is classified first );
//   * on each of these planes, the SECTION of `T` ( a triangle or a quadrilateral ) clipped by the others -- a piece of the face
//     of `C`, whose integral is also the facet's share ( `mesh3_cell` ).
//
// The elements are those the box of the cell meets, shared between the lanes of the warp, each done ONCE; only convex polygons
// of a few vertices are clipped, never a polyhedron, and nothing is walked per face of the cell. ( Measured on 1e4 seeds and
// 1.3e4 elements: cutting the tetrahedra of the fans one by one, ~ 175 times the constant density's time; the faces of `C`
// clipped by the elements, each lane walking the hierarchy for its faces, ~ 60 times. )
//
// TWO FACES IN THE SAME PLANE ( a face of `C` on a face of `T` ): an element with no node strictly inside a plane that cuts it
// meets the cell on a flat piece, and is skipped -- of the two elements on both sides of the face, the one inside counts the
// facet. A face of `T` on a plane of `C` is dropped when their outward normals agree ( the section is that face already ).
//
// Every integral is in the SEED's frame ( the cell's ), the mesh's nodes brought to it at the element: the moments are about
// the seed, as the cells write them. `int lambda_a lambda_b lambda_c = ( 6, 2, 1 ) V / 120`, `int lambda_a lambda_b =
// ( 1 + [ a = b ] ) V / 20`.
// =====================================================================================

#include "Density2D.cuh"

namespace sdot::gpu3d {

/// a DG1 density on a tetrahedral mesh ( `Mesh`, 3D ): the nodes, the values at the corners, the tetrahedra ( node indices ),
/// the gradients of their barycentric coordinates, the bounding volume hierarchy ( `distributions/Mesh.py` ). In doubles.
struct DensMesh3 {
    SI            nt = 0, nb = 0;                        ///< tetrahedra, hierarchy nodes
    const double *pn = nullptr;                          ///< `pn[ 3 q + c ]`: the nodes
    const double *val = nullptr;                         ///< `val[ 4 e + i ]`: the value of tetrahedron `e` at its corner `i`
    const int    *tet = nullptr;                         ///< `tet[ 4 e + i ]`: the node of corner `i`
    const double *grad = nullptr;                        ///< `grad[ 12 e + 3 i + c ]`: component `c` of the gradient of `lambda_i`
    const double *blo = nullptr, *bhi = nullptr;         ///< `[ nb, 3 ]`: the boxes of the hierarchy
    const int    *links = nullptr;                       ///< `[ nb, 4 ]`: left, right, begin, end ( a leaf: left < 0 )
};

/// what a cell accumulates: the mass, `int ( x - p ) rho`, `int |x - p|^2 rho`
struct DensSums3 {
    double m = 0, mx = 0, my = 0, mz = 0, m2 = 0;
};


/// an affine function `c + G . ( x - x0 )` ( the seed's frame ): a barycentric coordinate, the density of an element, the
/// signed distance to a plane of the cell
struct Affine3 {
    double c, gx, gy, gz, x0, y0, z0;
    __device__ __forceinline__ double operator()( double x, double y, double z ) const { return c + gx * ( x - x0 ) + gy * ( y - y0 ) + gz * ( z - z0 ); }
};

/// a convex polygon ( 3D, the seed's frame ), at most `CAP` vertices ( past it, the vertices are dropped: a face of more than
/// `CAP - 4` vertices is cut in triangles by the caller )
template<int _CAP>
struct Poly3 {
    static constexpr int CAP = _CAP;
    int    n = 0;
    double x[ CAP ], y[ CAP ], z[ CAP ];
    __device__ __forceinline__ void push( double a, double b, double c ) { if ( n < CAP ) { x[ n ] = a; y[ n ] = b; z[ n ] = c; ++n; } }
};

/// how a polygon left a clip: cut ( or whole, or empty ), or LYING on the plane
enum { CLIP_DONE = 0, CLIP_ON = 1 };

/// `p` clipped by `h >= 0`, in place. A polygon whose vertices are all within `tol` of the plane is left as it is ( `CLIP_ON`:
/// the caller decides ); a vertex within `tol` is on the plane
template<class P>
__device__ __forceinline__ int clip3( P &p, const Affine3 &h, double tol ) {
    constexpr int CAP = P::CAP;
    double s[ CAP ];
    bool on = true, all_in = true;
    for ( int j = 0; j < p.n; ++j ) {
        const double v = h( p.x[ j ], p.y[ j ], p.z[ j ] );
        s[ j ] = fabs( v ) <= tol ? 0.0 : v;
        on = on && s[ j ] == 0;
        all_in = all_in && s[ j ] >= 0;
    }
    if ( on )
        return CLIP_ON;
    if ( all_in )
        return CLIP_DONE;
    // in place: the kept vertices and the crossings, written behind the reads ( a copy of the first vertex closes the loop )
    const double fx = p.x[ 0 ], fy = p.y[ 0 ], fz = p.z[ 0 ], fs = s[ 0 ];
    double ox[ CAP ], oy[ CAP ], oz[ CAP ];
    int m = 0;
    for ( int j = 0; j < p.n; ++j ) {
        const bool last = j + 1 == p.n;
        const double jx = p.x[ j ], jy = p.y[ j ], jz = p.z[ j ], sj = s[ j ];
        const double kx = last ? fx : p.x[ j + 1 ], ky = last ? fy : p.y[ j + 1 ], kz = last ? fz : p.z[ j + 1 ], sk = last ? fs : s[ j + 1 ];
        if ( sj >= 0 && m < CAP ) { ox[ m ] = jx; oy[ m ] = jy; oz[ m ] = jz; ++m; }
        if ( ( ( sj > 0 && sk < 0 ) || ( sj < 0 && sk > 0 ) ) && m < CAP ) {
            const double t = sj / ( sj - sk );
            ox[ m ] = jx + t * ( kx - jx ); oy[ m ] = jy + t * ( ky - jy ); oz[ m ] = jz + t * ( kz - jz );
            ++m;
        }
    }
    for ( int j = 0; j < m; ++j ) { p.x[ j ] = ox[ j ]; p.y[ j ] = oy[ j ]; p.z[ j ] = oz[ j ]; }
    p.n = m;
    return CLIP_DONE;
}

/// THE TETRAHEDRON `( X, Y, Z )[ 0 .. 3 ]` of signed volume `V`, the affine density `F` at its corners: its share of the mass
/// and of the moments into `acc`
template<bool MOM>
__device__ __forceinline__ void tet3( const double ( &X )[ 4 ], const double ( &Y )[ 4 ], const double ( &Z )[ 4 ], const double ( &F )[ 4 ], double V, DensSums3 &acc ) {
    const double sf = F[ 0 ] + F[ 1 ] + F[ 2 ] + F[ 3 ];
    acc.m += V * sf / 4;
    if constexpr ( MOM ) {
        for ( int a = 0; a < 4; ++a ) {
            acc.mx += V / 20 * X[ a ] * ( sf + F[ a ] );
            acc.my += V / 20 * Y[ a ] * ( sf + F[ a ] );
            acc.mz += V / 20 * Z[ a ] * ( sf + F[ a ] );
        }
        double s2 = 0;
        for ( int a = 0; a < 4; ++a )
            for ( int b = 0; b < 4; ++b ) {
                const double ab = X[ a ] * X[ b ] + Y[ a ] * Y[ b ] + Z[ a ] * Z[ b ];
                for ( int c = 0; c < 4; ++c ) {
                    const int same = ( a == b ) + ( b == c ) + ( a == c );
                    s2 += ab * F[ c ] * ( same == 3 ? 6.0 : same == 1 ? 2.0 : 1.0 );
                }
            }
        acc.m2 += V / 120 * s2;
    }
}

/// the signed cones from `( gx, gy, gz )` on the fan of `p` ( its orientation by the order of its vertices ) of the affine `f`,
/// into `acc`
template<bool MOM,class P>
__device__ __forceinline__ void cones3( const P &p, double gx, double gy, double gz, const Affine3 &f, DensSums3 &acc ) {
    if ( p.n < 3 )
        return;
    const double fg = f( gx, gy, gz );
    const double f0 = f( p.x[ 0 ], p.y[ 0 ], p.z[ 0 ] );
    for ( int j = 1; j + 1 < p.n; ++j ) {
        const double X[ 4 ] = { gx, p.x[ 0 ], p.x[ j ], p.x[ j + 1 ] };
        const double Y[ 4 ] = { gy, p.y[ 0 ], p.y[ j ], p.y[ j + 1 ] };
        const double Z[ 4 ] = { gz, p.z[ 0 ], p.z[ j ], p.z[ j + 1 ] };
        const double ax = X[ 1 ] - gx, ay = Y[ 1 ] - gy, az = Z[ 1 ] - gz;
        const double bx = X[ 2 ] - gx, by = Y[ 2 ] - gy, bz = Z[ 2 ] - gz;
        const double cx = X[ 3 ] - gx, cy = Y[ 3 ] - gy, cz = Z[ 3 ] - gz;
        const double V = ( ax * ( by * cz - bz * cy ) + ay * ( bz * cx - bx * cz ) + az * ( bx * cy - by * cx ) ) / 6;
        const double F[ 4 ] = { fg, f0, f( X[ 2 ], Y[ 2 ], Z[ 2 ] ), f( X[ 3 ], Y[ 3 ], Z[ 3 ] ) };
        tet3<MOM>( X, Y, Z, F, V, acc );
    }
}

/// `int f dA` on `p`, its area counted along `( nx, ny, nz )` ( signed: the orientation of `p` against it )
template<class P>
__device__ __forceinline__ double surface3( const P &p, const Affine3 &f, double nx, double ny, double nz ) {
    if ( p.n < 3 )
        return 0;
    const double nn = sqrt( nx * nx + ny * ny + nz * nz );
    if ( ! ( nn > 0 ) )
        return 0;
    const double f0 = f( p.x[ 0 ], p.y[ 0 ], p.z[ 0 ] );
    double res = 0;
    for ( int j = 1; j + 1 < p.n; ++j ) {
        const double ax = p.x[ j ] - p.x[ 0 ], ay = p.y[ j ] - p.y[ 0 ], az = p.z[ j ] - p.z[ 0 ];
        const double bx = p.x[ j + 1 ] - p.x[ 0 ], by = p.y[ j + 1 ] - p.y[ 0 ], bz = p.z[ j + 1 ] - p.z[ 0 ];
        const double A = ( ( ay * bz - az * by ) * nx + ( az * bx - ax * bz ) * ny + ( ax * by - ay * bx ) * nz ) / ( 2 * nn );
        res += A * ( f0 + f( p.x[ j ], p.y[ j ], p.z[ j ] ) + f( p.x[ j + 1 ], p.y[ j + 1 ], p.z[ j + 1 ] ) ) / 3;
    }
    return res;
}

/// element `e` in the seed's frame ( the seed at `O` ): its nodes, its barycentric coordinates, its density
struct Elem3 {
    double  N[ 4 ][ 3 ];
    Affine3 lam[ 4 ], f;
};

__device__ __forceinline__ void load_elem( const DensMesh3 &m, SI e, const double ( &O )[ 3 ], Elem3 &el ) {
    const int *q = m.tet + 4 * e;
    for ( int i = 0; i < 4; ++i )
        for ( int c = 0; c < 3; ++c )
            el.N[ i ][ c ] = m.pn[ 3 * q[ i ] + c ] - O[ c ];
    for ( int i = 0; i < 4; ++i )
        el.lam[ i ] = Affine3{ i == 0 ? 1.0 : 0.0, m.grad[ 12 * e + 3 * i ], m.grad[ 12 * e + 3 * i + 1 ], m.grad[ 12 * e + 3 * i + 2 ], el.N[ 0 ][ 0 ], el.N[ 0 ][ 1 ], el.N[ 0 ][ 2 ] };
    el.f = Affine3{ m.val[ 4 * e ], 0, 0, 0, el.N[ 0 ][ 0 ], el.N[ 0 ][ 1 ], el.N[ 0 ][ 2 ] };
    for ( int i = 0; i < 4; ++i ) {
        el.f.gx += m.val[ 4 * e + i ] * el.lam[ i ].gx;
        el.f.gy += m.val[ 4 * e + i ] * el.lam[ i ].gy;
        el.f.gz += m.val[ 4 * e + i ] * el.lam[ i ].gz;
    }
}

/// the tolerance on the barycentric coordinates ( a point on a plane of an element )
constexpr double MESH3_TOL = 1e-10;

/// the hierarchy walked with the box `[ lo, hi ]` ( global coordinates ): `body( e )` for each element of a leaf it meets
template<class F>
__device__ __forceinline__ void mesh3_walk( const DensMesh3 &m, const double ( &lo )[ 3 ], const double ( &hi )[ 3 ], F &&body ) {
    if ( m.nb == 0 )
        return;
    double l[ 3 ], h[ 3 ];
    for ( int d = 0; d < 3; ++d ) {
        // a little WIDER: an element that only touches the region must not be lost to a rounding ( `Density2D.cuh::mesh_walk` )
        const double mg = 1e-9 * ( hi[ d ] - lo[ d ] ) + 1e-12 * fmax( fabs( lo[ d ] ), fabs( hi[ d ] ) );
        l[ d ] = lo[ d ] - mg;
        h[ d ] = hi[ d ] + mg;
    }
    SI stack[ 64 ];
    int top = 0;
    stack[ top++ ] = 0;
    while ( top ) {
        const SI n = stack[ --top ];
        if ( m.bhi[ 3 * n ] < l[ 0 ] || m.blo[ 3 * n ] > h[ 0 ] || m.bhi[ 3 * n + 1 ] < l[ 1 ] || m.blo[ 3 * n + 1 ] > h[ 1 ] ||
             m.bhi[ 3 * n + 2 ] < l[ 2 ] || m.blo[ 3 * n + 2 ] > h[ 2 ] )
            continue;
        const int left = m.links[ 4 * n ];
        if ( left >= 0 ) {
            if ( top + 2 <= 64 ) {
                stack[ top++ ] = m.links[ 4 * n + 1 ];
                stack[ top++ ] = left;
            }
            continue;
        }
        for ( SI e = m.links[ 4 * n + 2 ]; e < m.links[ 4 * n + 3 ]; ++e )
            body( e );
    }
}

/// THE WALK IN TWO TIMES: the elements the box meets that `keep( e )` takes are gathered ( up to `B` per lane ), then `body( e )`
/// runs on them. A walk calls its body when ITS lane reaches a leaf, the other lanes of the warp waiting: the clipping, which is
/// the cost, would run one lane at a time. Gathered, the lanes clip together ( a full list is emptied during the walk )
template<int B = 48,class K,class F>
__device__ __forceinline__ void mesh3_gather( const DensMesh3 &m, const double ( &lo )[ 3 ], const double ( &hi )[ 3 ], K &&keep, F &&body ) {
    SI ids[ B ];
    int nb = 0;
    mesh3_walk( m, lo, hi, [&]( SI e ) {
        if ( ! keep( e ) )
            return;
        if ( nb == B ) {
            for ( int q = 0; q < B; ++q ) body( ids[ q ] );
            nb = 0;
        }
        ids[ nb++ ] = e;
    } );
    for ( int q = 0; q < nb; ++q )
        body( ids[ q ] );
}

/// THE SECTION of the element by a plane ( `h` its four nodes' signed values, positive inside, `tol` on that scale ): the nodes on
/// it and the crossings of the edges, as a convex polygon turning counterclockwise about `( nx, ny, nz )` ( at most 4 vertices )
template<class P>
__device__ __forceinline__ void section3( const Elem3 &el, const double ( &h )[ 4 ], double tol, double nx, double ny, double nz, P &cap ) {
    cap.n = 0;
    for ( int i = 0; i < 4; ++i )
        if ( fabs( h[ i ] ) <= tol )
            cap.push( el.N[ i ][ 0 ], el.N[ i ][ 1 ], el.N[ i ][ 2 ] );
    for ( int i = 0; i < 4; ++i )
        for ( int k = i + 1; k < 4; ++k )
            if ( ( h[ i ] > tol && h[ k ] < -tol ) || ( h[ i ] < -tol && h[ k ] > tol ) ) {
                const double t = h[ i ] / ( h[ i ] - h[ k ] );
                cap.push( el.N[ i ][ 0 ] + t * ( el.N[ k ][ 0 ] - el.N[ i ][ 0 ] ), el.N[ i ][ 1 ] + t * ( el.N[ k ][ 1 ] - el.N[ i ][ 1 ] ),
                          el.N[ i ][ 2 ] + t * ( el.N[ k ][ 2 ] - el.N[ i ][ 2 ] ) );
            }
    if ( cap.n < 3 ) {
        cap.n = 0;
        return;
    }
    // the orientation of ( 0, a, b ) about `n`
    auto turn = [&]( int a, int b ) {
        const double ax = cap.x[ a ] - cap.x[ 0 ], ay = cap.y[ a ] - cap.y[ 0 ], az = cap.z[ a ] - cap.z[ 0 ];
        const double bx = cap.x[ b ] - cap.x[ 0 ], by = cap.y[ b ] - cap.y[ 0 ], bz = cap.z[ b ] - cap.z[ 0 ];
        return ( ay * bz - az * by ) * nx + ( az * bx - ax * bz ) * ny + ( ax * by - ay * bx ) * nz;
    };
    auto swap = [&]( int a, int b ) {
        const double tx = cap.x[ a ], ty = cap.y[ a ], tz = cap.z[ a ];
        cap.x[ a ] = cap.x[ b ]; cap.y[ a ] = cap.y[ b ]; cap.z[ a ] = cap.z[ b ];
        cap.x[ b ] = tx; cap.y[ b ] = ty; cap.z[ b ] = tz;
    };
    if ( cap.n == 4 ) {
        // a quadrilateral: vertex 2 is the one opposite vertex 0 ( the two others on both sides of the diagonal )
        if ( turn( 1, 2 ) * turn( 3, 2 ) > 0 ) {
            if ( turn( 2, 1 ) * turn( 3, 1 ) < 0 ) swap( 1, 2 );   // 1 is opposite 0
            else                                   swap( 3, 2 );   // 3 is
        }
        if ( turn( 1, 2 ) < 0 ) swap( 1, 3 );
    } else if ( turn( 1, 2 ) < 0 )
        swap( 1, 2 );
}

/// A CELL AGAINST THE MESH: the planes `pl[ 4 j .. 4 j + 3 ]` ( `nc` of them, the seed's frame, OUTWARD normals: `n . x <= off`
/// inside ), the box `[ lo, hi ]` of the cell ( seed's frame ), `L` its size. The elements the box meets are shared between the
/// lanes ( the `lane`-th of every 32 ), each one CLASSIFIED against the planes:
///
///   * on the outer side of a plane, or with no node strictly inside one of those that cut it ( a flat intersection ): nothing;
///   * strictly inside all the planes: its own integral ( closed form );
///   * else, cut by the `act` planes ( those it has a node not strictly inside ): `C INTERSECT T` is bounded by the faces of `T`
///     clipped by them, and on each of them by the SECTION of `T` clipped by the others -- the cones from `g` of both into `acc`,
///     and the sections are the pieces of the cell's facets: `int rho dA` on plane `j` into `fl[ j ]` ( `FAC`, `j < FL` ).
///
/// An element cut by MANY planes ( larger than the cell, or about its size ) would clip each of its sections by all the others,
/// a cost in their number squared: from `SECT_MAX` planes on, the piece on plane `j` is the cell's face `j` ( `face( j, P )`, the
/// caller's: outward, `false` when it can not ) clipped by the four half-spaces of the element instead.
///
/// Only the planes that carry a face of the cell count ( `has_face( j )` ): the others are cuts made redundant by later ones,
/// which an element larger than the cell would see as cutting it, for nothing.
template<bool MOM,bool FAC,int FL,class Face,class HasFace>
__device__ __forceinline__ void mesh3_cell( const DensMesh3 &m, const double ( &O )[ 3 ], const double ( &g )[ 3 ], const double *pl, int nc,
                                            const double ( &lo )[ 3 ], const double ( &hi )[ 3 ], double L, int lane, DensSums3 &acc, double *fl,
                                            const Face &face, const HasFace &has_face ) {
    constexpr int SECT_MAX = 3;
    double glo[ 3 ], ghi[ 3 ];
    for ( int c = 0; c < 3; ++c ) { glo[ c ] = O[ c ] + lo[ c ]; ghi[ c ] = O[ c ] + hi[ c ]; }
    const double tol = 1e-10 * L;
    int count = 0;
    mesh3_gather( m, glo, ghi, [&]( SI ) { return ( count++ & 31 ) == lane; }, [&]( SI e ) {
        Elem3 el;
        load_elem( m, e, O, el );
        // the planes that cut the element ( `act` ), their values at the nodes ( `|n|_1` times the distance inside: no square root )
        // ( an element larger than the cell is cut by most of its planes: up to `ACT` of them )
        // The FLOAT decides the planes the element is far from ( a margin of `1e-4 L`, far above its rounding ), the double the others
        constexpr int ACT = 64;
        int act[ ACT ], na = 0;
        float fN[ 4 ][ 3 ];
        for ( int i = 0; i < 4; ++i ) for ( int c = 0; c < 3; ++c ) fN[ i ][ c ] = float( el.N[ i ][ c ] );
        for ( int j = 0; j < nc; ++j ) {
            if ( ! has_face( j ) )
                continue;
            const double nx = pl[ 4 * j ], ny = pl[ 4 * j + 1 ], nz = pl[ 4 * j + 2 ], off = pl[ 4 * j + 3 ];
            const float fnx = float( nx ), fny = float( ny ), fnz = float( nz ), foff = float( off );
            const float fm = 1e-4f * float( L ) * ( fabsf( fnx ) + fabsf( fny ) + fabsf( fnz ) );
            int fin = 0, fout = 0;
            for ( int i = 0; i < 4; ++i ) {
                const float hf = foff - ( fnx * fN[ i ][ 0 ] + fny * fN[ i ][ 1 ] + fnz * fN[ i ][ 2 ] );
                fin += hf > fm;
                fout += hf < -fm;
            }
            if ( fout == 4 )
                return;                                  // outside
            if ( fin == 4 )
                continue;                                // inside
            const double tj = tol * ( fabs( nx ) + fabs( ny ) + fabs( nz ) );
            double h[ 4 ];
            int nout = 0, nin = 0;
            for ( int i = 0; i < 4; ++i ) {
                h[ i ] = off - ( nx * el.N[ i ][ 0 ] + ny * el.N[ i ][ 1 ] + nz * el.N[ i ][ 2 ] );
                nout += h[ i ] < -tj;
                nin += h[ i ] > tj;
            }
            if ( nout == 4 || ( nin == 0 && tj > 0 ) )
                return;                                  // outside, or flat against the plane
            if ( nin < 4 ) {
                if ( na == ACT )
                    return;                              // ( a cell of more than 64 planes, all cutting one element )
                act[ na++ ] = j;
            }
        }
        if ( na == 0 ) {
            // strictly inside the cell: the element itself
            const double X[ 4 ] = { el.N[ 0 ][ 0 ], el.N[ 1 ][ 0 ], el.N[ 2 ][ 0 ], el.N[ 3 ][ 0 ] };
            const double Y[ 4 ] = { el.N[ 0 ][ 1 ], el.N[ 1 ][ 1 ], el.N[ 2 ][ 1 ], el.N[ 3 ][ 1 ] };
            const double Z[ 4 ] = { el.N[ 0 ][ 2 ], el.N[ 1 ][ 2 ], el.N[ 2 ][ 2 ], el.N[ 3 ][ 2 ] };
            const double ax = X[ 1 ] - X[ 0 ], ay = Y[ 1 ] - Y[ 0 ], az = Z[ 1 ] - Z[ 0 ];
            const double bx = X[ 2 ] - X[ 0 ], by = Y[ 2 ] - Y[ 0 ], bz = Z[ 2 ] - Z[ 0 ];
            const double cx = X[ 3 ] - X[ 0 ], cy = Y[ 3 ] - Y[ 0 ], cz = Z[ 3 ] - Z[ 0 ];
            const double V = fabs( ax * ( by * cz - bz * cy ) + ay * ( bz * cx - bx * cz ) + az * ( bx * cy - by * cx ) ) / 6;
            const double F[ 4 ] = { el.f( X[ 0 ], Y[ 0 ], Z[ 0 ] ), el.f( X[ 1 ], Y[ 1 ], Z[ 1 ] ), el.f( X[ 2 ], Y[ 2 ], Z[ 2 ] ), el.f( X[ 3 ], Y[ 3 ], Z[ 3 ] ) };
            tet3<MOM>( X, Y, Z, F, V, acc );
            return;
        }
        auto plane = [&]( int q ) {
            const int j = act[ q ];
            return Affine3{ pl[ 4 * j + 3 ], -pl[ 4 * j ], -pl[ 4 * j + 1 ], -pl[ 4 * j + 2 ], 0, 0, 0 };
        };
        auto tol_of = [&]( int q ) {
            const int j = act[ q ];
            return tol * ( fabs( pl[ 4 * j ] ) + fabs( pl[ 4 * j + 1 ] ) + fabs( pl[ 4 * j + 2 ] ) );
        };
        // the polygons in local arrays of `CAP` vertices: 8 for an element cut by one or two planes ( a section of 4 vertices clipped
        // once, a triangle twice ), 32 else ( the faces of the cell ) -- a smaller frame, half the time of the clipping
        auto cut = [&]<int CAP>() {
            // the faces of the element clipped by the planes that cut it ( dropped on one with the same outward normal )
            for ( int i = 0; i < 4; ++i ) {
                Poly3<CAP> r;
                for ( int j = 0; j < 4; ++j )
                    if ( j != i ) r.push( el.N[ j ][ 0 ], el.N[ j ][ 1 ], el.N[ j ][ 2 ] );
                // outward: the normal of `( r0, r1, r2 )` against the gradient of `lambda_i` ( which points inward )
                const double ux = r.x[ 1 ] - r.x[ 0 ], uy = r.y[ 1 ] - r.y[ 0 ], uz = r.z[ 1 ] - r.z[ 0 ];
                const double vx = r.x[ 2 ] - r.x[ 0 ], vy = r.y[ 2 ] - r.y[ 0 ], vz = r.z[ 2 ] - r.z[ 0 ];
                const double nr = ( uy * vz - uz * vy ) * el.lam[ i ].gx + ( uz * vx - ux * vz ) * el.lam[ i ].gy + ( ux * vy - uy * vx ) * el.lam[ i ].gz;
                if ( nr > 0 ) {
                    const double tx = r.x[ 1 ], ty = r.y[ 1 ], tz = r.z[ 1 ];
                    r.x[ 1 ] = r.x[ 2 ]; r.y[ 1 ] = r.y[ 2 ]; r.z[ 1 ] = r.z[ 2 ];
                    r.x[ 2 ] = tx; r.y[ 2 ] = ty; r.z[ 2 ] = tz;
                }
                bool keep = true;
                for ( int q = 0; q < na && keep && r.n >= 3; ++q ) {
                    const int j = act[ q ];
                    if ( clip3( r, plane( q ), tol_of( q ) ) == CLIP_ON )
                        keep = el.lam[ i ].gx * pl[ 4 * j ] + el.lam[ i ].gy * pl[ 4 * j + 1 ] + el.lam[ i ].gz * pl[ 4 * j + 2 ] > 0;   // ( normals opposite )
                }
                if ( keep )
                    cones3<MOM>( r, g[ 0 ], g[ 1 ], g[ 2 ], el.f, acc );
            }
            // the sections: the pieces of the cell's faces in the element
            for ( int q = 0; q < na; ++q ) {
                const int j = act[ q ];
                const Affine3 hq = plane( q );
                const double hn[ 4 ] = { hq( el.N[ 0 ][ 0 ], el.N[ 0 ][ 1 ], el.N[ 0 ][ 2 ] ), hq( el.N[ 1 ][ 0 ], el.N[ 1 ][ 1 ], el.N[ 1 ][ 2 ] ),
                                         hq( el.N[ 2 ][ 0 ], el.N[ 2 ][ 1 ], el.N[ 2 ][ 2 ] ), hq( el.N[ 3 ][ 0 ], el.N[ 3 ][ 1 ], el.N[ 3 ][ 2 ] ) };
                Poly3<CAP> cap;
                bool by_face = false;
                if constexpr ( CAP >= 32 )
                    by_face = face( j, cap );
                if ( by_face ) {
                    for ( int i = 0; i < 4 && cap.n >= 3; ++i )
                        clip3( cap, el.lam[ i ], MESH3_TOL );   // ( on a face of the element: kept whole )
                } else {
                    section3( el, hn, tol_of( q ), pl[ 4 * j ], pl[ 4 * j + 1 ], pl[ 4 * j + 2 ], cap );
                    for ( int k = 0; k < na && cap.n >= 3; ++k )
                        if ( k != q )
                            clip3( cap, plane( k ), tol_of( k ) );
                }
                if ( cap.n < 3 )
                    continue;
                cones3<MOM>( cap, g[ 0 ], g[ 1 ], g[ 2 ], el.f, acc );
                if constexpr ( FAC )
                    if ( j < FL )
                        fl[ j ] += surface3( cap, el.f, pl[ 4 * j ], pl[ 4 * j + 1 ], pl[ 4 * j + 2 ] );
            }
        };
        if ( na < SECT_MAX )
            cut.template operator()<8>();
        else
            cut.template operator()<32>();
    } );
}

} // namespace sdot::gpu3d
