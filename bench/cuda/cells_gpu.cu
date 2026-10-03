// THE GEOMETRIC FLOOR ON GPU, in FP32.
//
// = What this bench measures, and what it does not
//
// Exactly the same protocol as `--cells` of the CGAL bench, so as to be comparable with it:
// the FINAL LIST of neighbors is given, and only the CUT and the MEASURE are timed. This is the
// time of a cell if an oracle gave the connectivity without ever being wrong. So there is
// no tree here, no search, no rejected seed -- nothing of what makes up half of our CPU time.
// The figure to compare with is `cut + measure` of the CPU bench: 7078 ns/seed in 3D, 216 in 2D, measured
// on one thread on this machine.
//
// The list comes from the file written by `power_{2,3}d_light --dump`, which also carries the
// REFERENCE volume computed in FP64 by `Cell3T<128>`. This bench therefore does not talk to CGAL: it replays, and
// it is judged cell by cell.
//
// = The two placements, and why both are needed
//
//   --map thread   one cell per THREAD, the cell in local memory.
//   --map warp     one cell per WARP, the cell in SHARED, the 32 lanes sweeping its vertices.
//
// The first is the direct port of the CPU code, and it is the naive form. Its problem is not
// VRAM -- a `Cell3` in FP32 is ~2.3 KB, hence 160 MB for the 70 000 resident threads of a
// 2080 Ti, out of 11 GB -- but the CACHE: the hot footprint of an average 3D cell is ~1 KB, and
// 1024 threads per SM ask for 1 MB where the L1 is 64 KB. We thrash by a factor of sixteen.
//
// The second exists for that. It costs a rewrite (the prefixes become `__ballot`s and
// `__popc`s) but it places the cell where it fits, and it makes the vertex reads
// contiguous -- it is precisely the SoA form of `Cell3T` that allows it, `vx/vy/vz` separate: the
// 32 lanes read 32 consecutive floats.
//
// = FP32, and what had to change for it to hold
//
// Our predicate is `d . v - off`, of DEGREE 1 in the vertex coordinates. This is the whole
// difference with `insphere`, of degree 5, which forces CGAL into its interval filter: here there
// is no filter to double nor exact fallback, hence no arithmetic divergence in the warp.
//
// But the naive form of `off` does not work in FP32. Written
//
//     off = ( w0 - w1 ) / 2 + d . ( p0 + p1 ) / 2
//
// it adds up terms of `O(1)` to return an `O(|d|)` -- with `|d| ~ 0.017` on 2e5 seeds,
// that is a cancellation of two decimals, and the absolute error of 1e-7 on `off` moves the plane by
// `1e-7 / |d| = 6e-6`, i.e. 3e-4 of the cell size. So we work in the SEED's frame: setting
// `x' = x - p0`, the same quantity becomes
//
//     off' = ( w0 - w1 ) / 2 + |d|^2 / 2
//
// which has no cancellation left. The cost is nil (it is the same translation for the whole cell,
// and the volume does not see it), the gain is two decimals given back. This is the kind of thing that is not
// visible in FP64 and that decides everything in FP32.
//
// = The measure: a FACE-FREE formulation
//
// `Cell3T::measure` recovers the faces through a hash table, then fans them. On GPU this is
// the wrong computation: the table is 1 KB of local memory to reset to -1 per cell, and insertion
// is a loop with unpredictable exit. Yet we can do without it entirely.
//
// By the divergence theorem, `V = (1/3) sum_f x_f . A_f`, where `x_f` is ONE point of the plane of the
// face and `A_f` its outward area vector. And the area vector of a closed planar polygon is
// `(1/2) sum_edges p x q`, the edges taken in the direct sense seen from outside. Hence
//
//     V = (1/6) sum_faces sum_{edges of the face} x_f . ( p x q )
//
// -- and there is no face left to form: each incidence ( edge, face ) contributes on its own. An
// edge is on exactly TWO faces, those of its two common cuts `c0` and `c1`, and the direct sense around `c0` is
// that of `n0 x n1` (checked on the cube: the +x and +y faces give
// +z, which is indeed the direct sense seen from +x). The edge's two contributions combine:
//
//     V = (1/6) sum_edges  s_e  ( x_{c0} - x_{c1} ) . ( a x b ),   s_e = sign( (b-a) . (n0 x n1) )
//
// One cross product and one dot product per edge, no table, no branch, no grouping,
// `O(E)` instead of `O(F(V+E))`. `--measure faces` keeps the other path so that the comparison
// exists, and so that the two check each other.
//
// = What to read in the output
//
// `max deviation` is the worst RELATIVE deviation from the FP64 reference, cell by cell -- the only figure
// that says whether FP32 gets the geometry right. The sum of the volumes is not enough: wrong cells
// compensate each other, this bench has already shown it once.

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <cmath>
#include <string>
#include <vector>
#include <algorithm>

#define CK( x ) do { cudaError_t e_ = ( x ); if ( e_ != cudaSuccess ) {                    \
        std::fprintf( stderr, "CUDA %s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString( e_ ) ); \
        std::exit( 1 ); } } while ( 0 )

// ------------------------------------------------------------------ the connectivity file

struct Csr {
    int dim = 0;
    long long n = 0, nv = 0;
    std::vector<double> own, nbrs, ref;         ///< AoS on disk: x,y[,z],w
    std::vector<long long> off;
};

static bool read_csr( const char *path, Csr &c ) {
    std::FILE *f = std::fopen( path, "rb" );
    if ( ! f ) { std::fprintf( stderr, "unreadable: %s\n", path ); return false; }
    std::int32_t magic = 0, d = 0;
    if ( std::fread( &magic, 4, 1, f ) != 1 || magic != 0x31434450 ) {
        std::fprintf( stderr, "not a PDC1 dump: %s\n", path ); std::fclose( f ); return false;
    }
    auto rd = [ & ]( void *p, std::size_t sz, std::size_t k ) { if ( std::fread( p, sz, k, f ) != k ) std::fprintf( stderr, "truncated dump\n" ); };
    rd( &d, 4, 1 );
    std::int64_t n = 0, nv = 0;
    rd( &n, 8, 1 );
    rd( &nv, 8, 1 );
    c.dim = d; c.n = n; c.nv = nv;
    c.own.resize( std::size_t( n ) * ( d + 1 ) );
    c.off.resize( std::size_t( n ) + 1 );
    c.nbrs.resize( std::size_t( nv ) * ( d + 1 ) );
    c.ref.resize( std::size_t( n ) );
    rd( c.own.data(), 8, c.own.size() );
    rd( c.off.data(), 8, c.off.size() );
    rd( c.nbrs.data(), 8, c.nbrs.size() );
    rd( c.ref.data(), 8, c.ref.size() );
    std::fclose( f );
    return true;
}

// ------------------------------------------------------------------ small warp tools

__device__ __forceinline__ unsigned lt_mask() { return ( 1u << ( threadIdx.x & 31 ) ) - 1u; }

template<class T>
__device__ __forceinline__ T warp_sum( T v ) {
    for ( int o = 16; o; o >>= 1 ) v += __shfl_xor_sync( 0xffffffffu, v, o );
    return v;
}

__device__ __forceinline__ void sort3i( int *a ) {
    int t;
    if ( a[ 0 ] > a[ 1 ] ) { t = a[ 0 ]; a[ 0 ] = a[ 1 ]; a[ 1 ] = t; }
    if ( a[ 1 ] > a[ 2 ] ) { t = a[ 1 ]; a[ 1 ] = a[ 2 ]; a[ 2 ] = t; }
    if ( a[ 0 ] > a[ 1 ] ) { t = a[ 0 ]; a[ 0 ] = a[ 1 ]; a[ 1 ] = t; }
}

/// the two cuts common to two neighboring vertices: an edge of a simple polytope is
/// the intersection of two planes, so there are exactly two.
__device__ __forceinline__ void shared2( const int *a, const int *b, int &c0, int &c1 ) {
    c0 = c1 = 0;
    int k = 0;
    #pragma unroll
    for ( int i = 0; i < 3; ++i )
        #pragma unroll
        for ( int j = 0; j < 3; ++j )
            if ( a[ i ] == b[ j ] ) { if ( k++ == 0 ) c0 = a[ i ]; else c1 = a[ i ]; break; }
}

__device__ __forceinline__ bool share_one_but( const int *a, const int *b, int skip ) {
    #pragma unroll
    for ( int i = 0; i < 3; ++i ) {
        if ( a[ i ] == skip ) continue;
        #pragma unroll
        for ( int j = 0; j < 3; ++j )
            if ( a[ i ] == b[ j ] ) return true;
    }
    return false;
}

enum { CUT_UNCHANGED = 0, CUT_EMPTY = 1, CUT_OVERFLOW = 2, CUT_DONE = 3 };

/// THE PLANE of a cut, IN THE SEED'S FRAME. See the header: this is where FP32 is won or lost.
/// `c >= 0` designates the `c`-th neighbor of the cell; `c < 0` a face of the cube, translated
/// too (`x >= 0` becomes `-x' <= x0`).
template<class TF>
__device__ __forceinline__ void plane3( int c, const TF *px, const TF *py, const TF *pz, const TF *pw,
                                        long long base, TF x0, TF y0, TF z0, TF w0,
                                        TF &dx, TF &dy, TF &dz, TF &off ) {
    if ( c >= 0 ) {
        dx = px[ base + c ] - x0;
        dy = py[ base + c ] - y0;
        dz = pz[ base + c ] - z0;
        off = TF( 0.5 ) * ( w0 - pw[ base + c ] + dx * dx + dy * dy + dz * dz );
    } else {
        dx = dy = dz = 0;
        switch ( c ) {
            case -1: dx = -1; off = x0;         break;
            case -2: dx =  1; off = TF( 1 ) - x0; break;
            case -3: dy = -1; off = y0;         break;
            case -4: dy =  1; off = TF( 1 ) - y0; break;
            case -5: dz = -1; off = z0;         break;
            default: dz =  1; off = TF( 1 ) - z0; break;
        }
    }
}

// ================================================================== 3D, one cell per THREAD

/// The direct port of `pd::Cell3T`, in local buffers. Same algorithm, same order, same
/// safeguards: on overflow the cell stays INTACT and the caller counts.
template<class TF, int MaxNv>
struct Cell3G {
    static constexpr int MaxNe = 3 * MaxNv / 2 + 2;
    static constexpr int MaxNf = MaxNv / 2 + 2;

    int nb, ne;
    TF vx[ MaxNv ], vy[ MaxNv ], vz[ MaxNv ];
    int vc[ MaxNv ][ 3 ];
    int ea[ MaxNe ], eb[ MaxNe ];

    /// the unit cube SEEN FROM THE SEED. The six faces keep the indices `-1 .. -6`.
    __device__ void init_as_box( TF x0, TF y0, TF z0 ) {
        nb = 8; ne = 12;
        const TF lo[ 3 ] = { -x0, -y0, -z0 }, hi[ 3 ] = { TF( 1 ) - x0, TF( 1 ) - y0, TF( 1 ) - z0 };
        for ( int b = 0; b < 8; ++b ) {
            const int i = b & 1, j = ( b >> 1 ) & 1, k = ( b >> 2 ) & 1;
            vx[ b ] = i ? hi[ 0 ] : lo[ 0 ];
            vy[ b ] = j ? hi[ 1 ] : lo[ 1 ];
            vz[ b ] = k ? hi[ 2 ] : lo[ 2 ];
            vc[ b ][ 0 ] = i ? -2 : -1;
            vc[ b ][ 1 ] = j ? -4 : -3;
            vc[ b ][ 2 ] = k ? -6 : -5;
            sort3i( vc[ b ] );
        }
        int e = 0;
        for ( int b = 0; b < 8; ++b )
            for ( int d = 0; d < 3; ++d ) {
                const int o = b ^ ( 1 << d );
                if ( o > b ) { ea[ e ] = b; eb[ e ] = o; ++e; }
            }
    }

    __device__ int cut( TF dx, TF dy, TF dz, TF off, int cut_id ) {
        TF s[ MaxNv ];
        int nb_out = 0;
        for ( int i = 0; i < nb; ++i ) {
            s[ i ] = dx * vx[ i ] + dy * vy[ i ] + dz * vz[ i ] - off;
            nb_out += s[ i ] > 0;
        }
        if ( nb_out == 0 ) return CUT_UNCHANGED;
        if ( nb_out == nb ) { nb = 0; ne = 0; return CUT_EMPTY; }

        int map[ MaxNv ], nn = 0;
        for ( int i = 0; i < nb; ++i ) map[ i ] = s[ i ] > 0 ? -1 : nn++;

        TF nx[ MaxNv ], ny[ MaxNv ], nz[ MaxNv ];
        int ncut[ MaxNv ][ 3 ], from[ MaxNe ], which[ MaxNe ], nm = 0;
        for ( int e = 0; e < ne; ++e ) {
            const int a = ea[ e ], b = eb[ e ];
            const bool oa = s[ a ] > 0, ob = s[ b ] > 0;
            which[ e ] = -1;
            if ( oa == ob ) continue;
            if ( nn + nm >= MaxNv ) return CUT_OVERFLOW;
            const int in = oa ? b : a, out = oa ? a : b;
            // anchor on the vertex that is INSIDE: with `s_in == 0` the symmetric form does not return `v_in`
            // in floating point, and the vertex ends up ON THE OTHER SIDE of the plane.
            const TF t = s[ in ] / ( s[ in ] - s[ out ] );
            nx[ nm ] = vx[ in ] + ( vx[ out ] - vx[ in ] ) * t;
            ny[ nm ] = vy[ in ] + ( vy[ out ] - vy[ in ] ) * t;
            nz[ nm ] = vz[ in ] + ( vz[ out ] - vz[ in ] ) * t;
            int c0, c1;
            shared2( vc[ a ], vc[ b ], c0, c1 );
            ncut[ nm ][ 0 ] = c0; ncut[ nm ][ 1 ] = c1; ncut[ nm ][ 2 ] = cut_id;
            sort3i( ncut[ nm ] );
            from[ e ] = in; which[ e ] = nm; ++nm;
        }
        const int new_nb = nn + nm;
        if ( new_nb > MaxNv ) return CUT_OVERFLOW;

        int na = 0, ta[ MaxNe ], tb[ MaxNe ];
        for ( int e = 0; e < ne; ++e ) {
            const int a = ea[ e ], b = eb[ e ];
            int A = -1, B = -1;
            if ( which[ e ] >= 0 )      { A = map[ from[ e ] ]; B = nn + which[ e ]; }
            else if ( map[ a ] >= 0 )   { A = map[ a ];         B = map[ b ]; }
            if ( A < 0 ) continue;
            if ( na < MaxNe ) { ta[ na ] = A; tb[ na ] = B; }
            ++na;
        }
        // the sides of the NEW FACE: two new vertices are neighbors exactly when they are on
        // a same OLD cut.
        for ( int i = 0; i < nm; ++i )
            for ( int j = i + 1; j < nm; ++j )
                if ( share_one_but( ncut[ i ], ncut[ j ], cut_id ) ) {
                    if ( na < MaxNe ) { ta[ na ] = nn + i; tb[ na ] = nn + j; }
                    ++na;
                }
        if ( na > MaxNe ) return CUT_OVERFLOW;

        // ---- COMMIT. GOING UP: `map[ i ] <= i`, so the write stays behind the read.
        for ( int i = 0; i < nb; ++i ) {
            const int m = map[ i ];
            if ( m < 0 ) continue;
            vx[ m ] = vx[ i ]; vy[ m ] = vy[ i ]; vz[ m ] = vz[ i ];
            vc[ m ][ 0 ] = vc[ i ][ 0 ]; vc[ m ][ 1 ] = vc[ i ][ 1 ]; vc[ m ][ 2 ] = vc[ i ][ 2 ];
        }
        for ( int i = 0; i < nm; ++i ) {
            vx[ nn + i ] = nx[ i ]; vy[ nn + i ] = ny[ i ]; vz[ nn + i ] = nz[ i ];
            vc[ nn + i ][ 0 ] = ncut[ i ][ 0 ]; vc[ nn + i ][ 1 ] = ncut[ i ][ 1 ];
            vc[ nn + i ][ 2 ] = ncut[ i ][ 2 ];
        }
        for ( int e = 0; e < na; ++e ) { ea[ e ] = ta[ e ]; eb[ e ] = tb[ e ]; }
        nb = new_nb; ne = na;
        return CUT_DONE;
    }

    /// THE MEASURE BY EDGES: the face-free formulation of the header.
    __device__ TF measure_edges( const TF *px, const TF *py, const TF *pz, const TF *pw,
                                 long long base, TF x0, TF y0, TF z0, TF w0 ) const {
        if ( nb == 0 ) return 0;
        TF gx = 0, gy = 0, gz = 0;
        for ( int i = 0; i < nb; ++i ) { gx += vx[ i ]; gy += vy[ i ]; gz += vz[ i ]; }
        gx /= nb; gy /= nb; gz /= nb;

        TF vol = 0;
        for ( int e = 0; e < ne; ++e ) {
            const int a = ea[ e ], b = eb[ e ];
            int c0, c1;
            shared2( vc[ a ], vc[ b ], c0, c1 );
            TF n0x, n0y, n0z, o0, n1x, n1y, n1z, o1;
            plane3( c0, px, py, pz, pw, base, x0, y0, z0, w0, n0x, n0y, n0z, o0 );
            plane3( c1, px, py, pz, pw, base, x0, y0, z0, w0, n1x, n1y, n1z, o1 );
            // the foot of the perpendicular DROPPED FROM g: the point of the plane closest to the center
            // of the cell, hence the one that best conditions the final dot product.
            const TF k0 = ( o0 - ( n0x * gx + n0y * gy + n0z * gz ) )
                        / ( n0x * n0x + n0y * n0y + n0z * n0z );
            const TF k1 = ( o1 - ( n1x * gx + n1y * gy + n1z * gz ) )
                        / ( n1x * n1x + n1y * n1y + n1z * n1z );
            const TF wx = n0x * k0 - n1x * k1, wy = n0y * k0 - n1y * k1, wz = n0z * k0 - n1z * k1;
            const TF dx = n0y * n1z - n0z * n1y, dy = n0z * n1x - n0x * n1z, dz = n0x * n1y - n0y * n1x;
            const TF ax = vx[ a ] - gx, ay = vy[ a ] - gy, az = vz[ a ] - gz;
            const TF bx = vx[ b ] - gx, by = vy[ b ] - gy, bz = vz[ b ] - gz;
            const TF sg = ( bx - ax ) * dx + ( by - ay ) * dy + ( bz - az ) * dz > 0 ? TF( 1 ) : TF( -1 );
            const TF cx = ay * bz - az * by, cy = az * bx - ax * bz, cz = ax * by - ay * bx;
            vol += sg * ( wx * cx + wy * cy + wz * cz );
        }
        return vol * TF( 1.0 / 6.0 );
    }

    /// THE MEASURE BY FACES: the faithful port of `gather_faces`, kept for the comparison.
    /// The table is brought down from 256 to 64 slots -- `MaxNf` is 34, and 1 KB of local memory to
    /// reset to -1 per cell is not paid for the same way here as on an L1 stack.
    __device__ TF measure_faces() const {
        if ( nb == 0 ) return 0;
        constexpr unsigned HT = 64;
        int slot[ MaxNv ][ 3 ], ht[ HT ];
        int fcut[ MaxNf ], fv0[ MaxNf ];
        TF fsx[ MaxNf ], fsy[ MaxNf ], fsz[ MaxNf ];
        for ( unsigned h = 0; h < HT; ++h ) ht[ h ] = -1;
        int nf = 0;
        for ( int i = 0; i < nb; ++i )
            for ( int r = 0; r < 3; ++r ) {
                const int c = vc[ i ][ r ];
                unsigned h = ( unsigned( c ) * 2654435761u ) & ( HT - 1 );
                while ( ht[ h ] >= 0 && fcut[ ht[ h ] ] != c ) h = ( h + 1 ) & ( HT - 1 );
                int k = ht[ h ] >= 0 ? ht[ h ] : nf;
                if ( k == nf ) {
                    if ( k >= MaxNf ) continue;
                    ht[ h ] = k; ++nf;
                    fcut[ k ] = c; fv0[ k ] = i;
                    fsx[ k ] = fsy[ k ] = fsz[ k ] = 0;
                }
                slot[ i ][ r ] = k;
            }
        TF gx = 0, gy = 0, gz = 0;
        for ( int i = 0; i < nb; ++i ) { gx += vx[ i ]; gy += vy[ i ]; gz += vz[ i ]; }
        gx /= nb; gy /= nb; gz /= nb;
        for ( int e = 0; e < ne; ++e ) {
            const int a = ea[ e ], b = eb[ e ];
            for ( int r = 0; r < 3; ++r ) {
                const int c = vc[ a ][ r ];
                if ( c != vc[ b ][ 0 ] && c != vc[ b ][ 1 ] && c != vc[ b ][ 2 ] ) continue;
                const int k = slot[ a ][ r ], o = fv0[ k ];
                if ( o == a || o == b ) continue;
                const TF ux = vx[ a ] - vx[ o ], uy = vy[ a ] - vy[ o ], uz = vz[ a ] - vz[ o ];
                const TF wx = vx[ b ] - vx[ o ], wy = vy[ b ] - vy[ o ], wz = vz[ b ] - vz[ o ];
                TF cx = uy * wz - uz * wy, cy = uz * wx - ux * wz, cz = ux * wy - uy * wx;
                if ( fsx[ k ] * cx + fsy[ k ] * cy + fsz[ k ] * cz < 0 ) { cx = -cx; cy = -cy; cz = -cz; }
                fsx[ k ] += cx; fsy[ k ] += cy; fsz[ k ] += cz;
            }
        }
        TF vol = 0;
        for ( int k = 0; k < nf; ++k ) {
            const int o = fv0[ k ];
            const TF d = ( vx[ o ] - gx ) * fsx[ k ] + ( vy[ o ] - gy ) * fsy[ k ] + ( vz[ o ] - gz ) * fsz[ k ];
            vol += d < 0 ? -d : d;
        }
        return vol * TF( 1.0 / 6.0 );
    }
};

template<class TF, int MaxNv, bool FACES>
__global__ void k3_thread( long long n, const long long *off,
                           const TF *mx, const TF *my, const TF *mz, const TF *mw,
                           const TF *px, const TF *py, const TF *pz, const TF *pw,
                           TF *out, int *ovf ) {
    const long long i = blockIdx.x * ( long long ) blockDim.x + threadIdx.x;
    if ( i >= n ) return;
    const TF x0 = mx[ i ], y0 = my[ i ], z0 = mz[ i ], w0 = mw[ i ];
    Cell3G<TF, MaxNv> c;
    c.init_as_box( x0, y0, z0 );
    const long long b = off[ i ], e = off[ i + 1 ];
    int bad = 0, nu = 0;
    for ( long long j = b; j < e; ++j ) {
        const TF dx = px[ j ] - x0, dy = py[ j ] - y0, dz = pz[ j ] - z0;
        const TF o = TF( 0.5 ) * ( w0 - pw[ j ] + dx * dx + dy * dy + dz * dz );
        const int r = c.cut( dx, dy, dz, o, int( j - b ) );
        bad |= r == CUT_OVERFLOW;
        nu += r == CUT_UNCHANGED;
    }
    out[ i ] = FACES ? c.measure_faces() : c.measure_edges( px, py, pz, pw, b, x0, y0, z0, w0 );
    if ( bad ) atomicAdd( ovf, 1 );
    atomicAdd( ovf + 1, nu );
}

// ================================================================== 3D, one cell per WARP

/// WHAT IS IN SHARED, and nothing more. Everything that can live in registers does: the
/// temporaries of `cut` (the new vertices, the outgoing edges) are spread over the lanes,
/// a handful per lane, and do not cost a byte of shared. This is what brings the cell
/// from ~6 KB down to ~2.9 KB, hence the occupancy from 10 to 22 warps per SM.
template<class TF, int MaxNv>
struct WCell {
    static constexpr int MaxNe = 3 * MaxNv / 2 + 2;
    TF vx[ MaxNv ], vy[ MaxNv ], vz[ MaxNv ];
    int vc[ MaxNv ][ 3 ];
    int ea[ MaxNe ], eb[ MaxNe ];
    TF s[ MaxNv ];
    int map[ MaxNv ];
};

template<class TF, int MaxNv>
__device__ int warp_cut( WCell<TF, MaxNv> &C, int &nb, int &ne,
                         TF dx, TF dy, TF dz, TF off, int cut_id ) {
    constexpr int MaxNe = WCell<TF, MaxNv>::MaxNe;
    constexpr int RV = ( MaxNv + 31 ) / 32;             // rounds over the vertices
    constexpr int RE = ( MaxNe + 31 ) / 32;             // rounds over the edges
    const int lane = threadIdx.x & 31;
    const unsigned full = 0xffffffffu, lt = ( 1u << lane ) - 1u;

    int nb_out = 0;
    for ( int i = lane; i < nb; i += 32 ) {
        const TF si = dx * C.vx[ i ] + dy * C.vy[ i ] + dz * C.vz[ i ] - off;
        C.s[ i ] = si;
        nb_out += si > 0;
    }
    __syncwarp();
    nb_out = warp_sum( nb_out );
    if ( nb_out == 0 ) return CUT_UNCHANGED;
    if ( nb_out == nb ) { nb = 0; ne = 0; return CUT_EMPTY; }

    // ---- `map`, exclusive prefix of the KEPT vertices. `nn` stays identical on the 32 lanes: the
    // `__ballot`s are uniform, so are the `__popc`s.
    int nn = 0;
    for ( int base = 0; base < nb; base += 32 ) {
        const int i = base + lane;
        const bool keep = i < nb && ! ( C.s[ i ] > 0 );
        const unsigned m = __ballot_sync( full, keep );
        if ( i < nb ) C.map[ i ] = keep ? nn + __popc( m & lt ) : -1;
        nn += __popc( m );
    }
    __syncwarp();

    // ---- the edges, READ INTO REGISTERS before anything moves. This is what allows
    // writing `ea/eb` in place below without one lane overwriting what another has not read.
    int ra[ RE ], rb[ RE ], rw[ RE ], rf[ RE ], rc0[ RE ], rc1[ RE ];
    TF rx[ RE ], ry[ RE ], rz[ RE ];
    int nm = 0;
    for ( int r = 0; r < RE; ++r ) {
        const int e = r * 32 + lane;
        const bool live = e < ne;
        const int a = live ? C.ea[ e ] : 0, b = live ? C.eb[ e ] : 0;
        ra[ r ] = a; rb[ r ] = b; rw[ r ] = -1; rf[ r ] = 0;
        const bool oa = live && C.s[ a ] > 0, ob = live && C.s[ b ] > 0;
        const bool cross = live && ( oa != ob );
        const unsigned m = __ballot_sync( full, cross );
        const int idx = nm + __popc( m & lt );
        nm += __popc( m );
        if ( cross ) {
            const int in = oa ? b : a, out = oa ? a : b;
            const TF t = C.s[ in ] / ( C.s[ in ] - C.s[ out ] );
            rx[ r ] = C.vx[ in ] + ( C.vx[ out ] - C.vx[ in ] ) * t;
            ry[ r ] = C.vy[ in ] + ( C.vy[ out ] - C.vy[ in ] ) * t;
            rz[ r ] = C.vz[ in ] + ( C.vz[ out ] - C.vz[ in ] ) * t;
            shared2( C.vc[ a ], C.vc[ b ], rc0[ r ], rc1[ r ] );
            rf[ r ] = in; rw[ r ] = idx;
        }
    }
    if ( nn + nm > MaxNv ) return CUT_OVERFLOW;         // uniform: the cell stays intact
    __syncwarp();

    // ---- COMPACTION. Same precaution: we read everything, synchronize, then write.
    TF cx[ RV ], cy[ RV ], cz[ RV ];
    int c0_[ RV ], c1_[ RV ], c2_[ RV ], cm[ RV ];
    for ( int r = 0; r < RV; ++r ) {
        const int i = r * 32 + lane;
        cm[ r ] = -1;
        if ( i < nb ) {
            cm[ r ] = C.map[ i ];
            cx[ r ] = C.vx[ i ]; cy[ r ] = C.vy[ i ]; cz[ r ] = C.vz[ i ];
            c0_[ r ] = C.vc[ i ][ 0 ]; c1_[ r ] = C.vc[ i ][ 1 ]; c2_[ r ] = C.vc[ i ][ 2 ];
        }
    }
    __syncwarp();
    for ( int r = 0; r < RV; ++r )
        if ( cm[ r ] >= 0 ) {
            const int m = cm[ r ];
            C.vx[ m ] = cx[ r ]; C.vy[ m ] = cy[ r ]; C.vz[ m ] = cz[ r ];
            C.vc[ m ][ 0 ] = c0_[ r ]; C.vc[ m ][ 1 ] = c1_[ r ]; C.vc[ m ][ 2 ] = c2_[ r ];
        }
    // the NEW vertices go to `>= nn`, the compaction to `< nn`: the two do not cross.
    for ( int r = 0; r < RE; ++r )
        if ( rw[ r ] >= 0 ) {
            const int k = nn + rw[ r ];
            C.vx[ k ] = rx[ r ]; C.vy[ k ] = ry[ r ]; C.vz[ k ] = rz[ r ];
            int t3[ 3 ] = { rc0[ r ], rc1[ r ], cut_id };
            sort3i( t3 );
            C.vc[ k ][ 0 ] = t3[ 0 ]; C.vc[ k ][ 1 ] = t3[ 1 ]; C.vc[ k ][ 2 ] = t3[ 2 ];
        }
    __syncwarp();

    // ---- THE OUTGOING EDGES. `map` has not been touched, and `ea/eb` has been read in full.
    int na = 0;
    for ( int r = 0; r < RE; ++r ) {
        const int e = r * 32 + lane;
        int A = -1, B = -1;
        if ( e < ne ) {
            if ( rw[ r ] >= 0 )                 { A = C.map[ rf[ r ] ]; B = nn + rw[ r ]; }
            else if ( C.map[ ra[ r ] ] >= 0 )   { A = C.map[ ra[ r ] ]; B = C.map[ rb[ r ] ]; }
        }
        const bool push = A >= 0;
        const unsigned m = __ballot_sync( full, push );
        const int idx = na + __popc( m & lt );
        na += __popc( m );
        if ( push && idx < MaxNe ) { C.ea[ idx ] = A; C.eb[ idx ] = B; }
    }
    __syncwarp();
    // the sides of the NEW FACE. The outer loop is uniform (`nm` is), so the inner `__ballot`s
    // do see all 32 lanes.
    for ( int i = 0; i < nm; ++i )
        for ( int jb = i + 1; jb < nm; jb += 32 ) {
            const int j = jb + lane;
            const bool push = j < nm && share_one_but( C.vc[ nn + i ], C.vc[ nn + j ], cut_id );
            const unsigned m = __ballot_sync( full, push );
            const int idx = na + __popc( m & lt );
            na += __popc( m );
            if ( push && idx < MaxNe ) { C.ea[ idx ] = nn + i; C.eb[ idx ] = nn + j; }
        }
    __syncwarp();
    if ( na > MaxNe ) return CUT_OVERFLOW;              // impossible if the vertices fit (Euler)
    nb = nn + nm; ne = na;
    return CUT_DONE;
}

template<class TF, int MaxNv>
__device__ TF warp_measure( const WCell<TF, MaxNv> &C, int nb, int ne,
                            const TF *px, const TF *py, const TF *pz, const TF *pw,
                            long long base, TF x0, TF y0, TF z0, TF w0 ) {
    const int lane = threadIdx.x & 31;
    if ( nb == 0 ) return 0;
    TF gx = 0, gy = 0, gz = 0;
    for ( int i = lane; i < nb; i += 32 ) { gx += C.vx[ i ]; gy += C.vy[ i ]; gz += C.vz[ i ]; }
    gx = warp_sum( gx ) / nb; gy = warp_sum( gy ) / nb; gz = warp_sum( gz ) / nb;

    TF vol = 0;
    for ( int e = lane; e < ne; e += 32 ) {
        const int a = C.ea[ e ], b = C.eb[ e ];
        int c0, c1;
        shared2( C.vc[ a ], C.vc[ b ], c0, c1 );
        TF n0x, n0y, n0z, o0, n1x, n1y, n1z, o1;
        plane3( c0, px, py, pz, pw, base, x0, y0, z0, w0, n0x, n0y, n0z, o0 );
        plane3( c1, px, py, pz, pw, base, x0, y0, z0, w0, n1x, n1y, n1z, o1 );
        const TF k0 = ( o0 - ( n0x * gx + n0y * gy + n0z * gz ) )
                    / ( n0x * n0x + n0y * n0y + n0z * n0z );
        const TF k1 = ( o1 - ( n1x * gx + n1y * gy + n1z * gz ) )
                    / ( n1x * n1x + n1y * n1y + n1z * n1z );
        const TF wx = n0x * k0 - n1x * k1, wy = n0y * k0 - n1y * k1, wz = n0z * k0 - n1z * k1;
        const TF dx = n0y * n1z - n0z * n1y, dy = n0z * n1x - n0x * n1z, dz = n0x * n1y - n0y * n1x;
        const TF ax = C.vx[ a ] - gx, ay = C.vy[ a ] - gy, az = C.vz[ a ] - gz;
        const TF bx = C.vx[ b ] - gx, by = C.vy[ b ] - gy, bz = C.vz[ b ] - gz;
        const TF sg = ( bx - ax ) * dx + ( by - ay ) * dy + ( bz - az ) * dz > 0 ? TF( 1 ) : TF( -1 );
        const TF cx = ay * bz - az * by, cy = az * bx - ax * bz, cz = ax * by - ay * bx;
        vol += sg * ( wx * cx + wy * cy + wz * cz );
    }
    return warp_sum( vol ) * TF( 1.0 / 6.0 );
}

extern __shared__ char smem_raw[];

template<class TF, int MaxNv>
__global__ void k3_warp( long long n, const long long *off,
                         const TF *mx, const TF *my, const TF *mz, const TF *mw,
                         const TF *px, const TF *py, const TF *pz, const TF *pw,
                         TF *out, int *ovf ) {
    auto *cells = reinterpret_cast<WCell<TF, MaxNv> *>( smem_raw );
    const int lane = threadIdx.x & 31, w = threadIdx.x >> 5;
    const long long i = blockIdx.x * ( long long ) ( blockDim.x >> 5 ) + w;
    if ( i >= n ) return;
    WCell<TF, MaxNv> &C = cells[ w ];

    const TF x0 = mx[ i ], y0 = my[ i ], z0 = mz[ i ], w0 = mw[ i ];
    // the cube, written by the lanes concerned. Twelve edges, eight vertices: a single pass.
    if ( lane < 8 ) {
        const int lo = lane, ix = lo & 1, iy = ( lo >> 1 ) & 1, iz = ( lo >> 2 ) & 1;
        C.vx[ lo ] = ix ? TF( 1 ) - x0 : -x0;
        C.vy[ lo ] = iy ? TF( 1 ) - y0 : -y0;
        C.vz[ lo ] = iz ? TF( 1 ) - z0 : -z0;
        int t3[ 3 ] = { ix ? -2 : -1, iy ? -4 : -3, iz ? -6 : -5 };
        sort3i( t3 );
        C.vc[ lo ][ 0 ] = t3[ 0 ]; C.vc[ lo ][ 1 ] = t3[ 1 ]; C.vc[ lo ][ 2 ] = t3[ 2 ];
    }
    if ( lane == 0 ) {
        int e = 0;
        for ( int b = 0; b < 8; ++b )
            for ( int d = 0; d < 3; ++d ) {
                const int o = b ^ ( 1 << d );
                if ( o > b ) { C.ea[ e ] = b; C.eb[ e ] = o; ++e; }
            }
    }
    __syncwarp();

    int nb = 8, ne = 12, bad = 0;
    const long long b0 = off[ i ], e0 = off[ i + 1 ];
    for ( long long j = b0; j < e0; ++j ) {
        const TF dx = px[ j ] - x0, dy = py[ j ] - y0, dz = pz[ j ] - z0;
        const TF o = TF( 0.5 ) * ( w0 - pw[ j ] + dx * dx + dy * dy + dz * dz );
        bad |= warp_cut( C, nb, ne, dx, dy, dz, o, int( j - b0 ) ) == CUT_OVERFLOW;
    }
    const TF v = warp_measure( C, nb, ne, px, py, pz, pw, b0, x0, y0, z0, w0 );
    if ( lane == 0 ) { out[ i ] = v; if ( bad ) atomicAdd( ovf, 1 ); }
}

// ================================================================== 2D, one cell per THREAD

/// The direct port of `pd::CellSoAT`. No warp version: a 2D cell has six vertices on
/// average, a 32-lane warp would waste five out of six. The right grain there would be a QUARTER warp,
/// and that is not what we are trying to find out here.
template<class TF, int MaxNv>
struct Cell2G {
    int nb;
    TF vx[ MaxNv ], vy[ MaxNv ];
    TF cdx[ MaxNv ], cdy[ MaxNv ], co[ MaxNv ];
    int cid[ MaxNv ];

    __device__ void init_as_box( TF x0, TF y0 ) {
        nb = 4;
        const TF lx = -x0, hx = TF( 1 ) - x0, ly = -y0, hy = TF( 1 ) - y0;
        vx[ 0 ] = lx; vy[ 0 ] = ly;
        vx[ 1 ] = hx; vy[ 1 ] = ly;
        vx[ 2 ] = hx; vy[ 2 ] = hy;
        vx[ 3 ] = lx; vy[ 3 ] = hy;
        const TF dx[ 4 ] = { 0, 1, 0, -1 }, dy[ 4 ] = { -1, 0, 1, 0 };
        const TF of[ 4 ] = { y0, hx, hy, x0 };
        for ( int i = 0; i < 4; ++i ) { cdx[ i ] = dx[ i ]; cdy[ i ] = dy[ i ]; co[ i ] = of[ i ]; cid[ i ] = -1; }
    }

    /// `nrun` returns the number of outer RUNS. All the rest of the code assumes it is ONE --
    /// the outside of a convex set cut by a half-space is in one piece -- and it is that
    /// hypothesis that decides the direction of the shifts. If a sign is wrong, the run breaks into
    /// two and the cell is rewritten any which way: this counter says whether that is what happens.
    __device__ int cut( TF dx, TF dy, TF off, int cut_id, int &nrun ) {
        TF s[ MaxNv ];
        for ( int i = 0; i < nb; ++i ) s[ i ] = dx * vx[ i ] + dy * vy[ i ] - off;
        int nb_out = 0, i1 = 0;
        nrun = 0;
        bool prev_out = s[ nb - 1 ] > 0;
        for ( int i = 0; i < nb; ++i ) {
            const bool out = s[ i ] > 0;
            if ( out ) { ++nb_out; if ( ! prev_out ) { i1 = i; ++nrun; } }
            prev_out = out;
        }
        if ( nb_out == 0 ) return CUT_UNCHANGED;
        if ( nb_out == nb ) { nb = 0; return CUT_EMPTY; }
        const int nb_in = nb - nb_out, new_nb = nb_in + 2;
        if ( new_nb > MaxNv ) return CUT_OVERFLOW;

        const int j0 = ( i1 + nb - 1 ) % nb, j2 = ( i1 + nb_out - 1 ) % nb, j3 = ( j2 + 1 ) % nb;
        const TF s0 = s[ j0 ], s1 = s[ i1 ], s2 = s[ j2 ], s3 = s[ j3 ];
        const TF ta = s0 / ( s0 - s1 );
        const TF pax = vx[ j0 ] + ( vx[ i1 ] - vx[ j0 ] ) * ta;
        const TF pay = vy[ j0 ] + ( vy[ i1 ] - vy[ j0 ] ) * ta;
        const TF tb = s3 / ( s3 - s2 );
        const TF pbx = vx[ j3 ] + ( vx[ j2 ] - vx[ j3 ] ) * tb;
        const TF pby = vy[ j3 ] + ( vy[ j2 ] - vy[ j3 ] ) * tb;
        const TF bdx = cdx[ j2 ], bdy = cdy[ j2 ], bof = co[ j2 ];
        const int bid = cid[ j2 ];

        #define MOVE( dst, src ) do { const int d_ = ( dst ), s_ = ( src );                     \
            vx[ d_ ] = vx[ s_ ]; vy[ d_ ] = vy[ s_ ]; cdx[ d_ ] = cdx[ s_ ];                    \
            cdy[ d_ ] = cdy[ s_ ]; co[ d_ ] = co[ s_ ]; cid[ d_ ] = cid[ s_ ]; } while ( 0 )
        #define PUT( k, x, y, ddx, ddy, dof, did ) do { const int k_ = ( k );                   \
            vx[ k_ ] = ( x ); vy[ k_ ] = ( y ); cdx[ k_ ] = ( ddx ); cdy[ k_ ] = ( ddy );       \
            co[ k_ ] = ( dof ); cid[ k_ ] = ( did ); } while ( 0 )

        if ( i1 <= j2 ) {
            if ( nb_out == 1 )       for ( int i = nb; i > i1 + 1; --i ) MOVE( i, i - 1 );
            else if ( nb_out > 2 ) { const int gap = nb_out - 2;
                                     for ( int i = j2 + 1; i < nb; ++i ) MOVE( i - gap, i ); }
            PUT( i1 + 0, pax, pay, dx, dy, off, cut_id );
            PUT( i1 + 1, pbx, pby, bdx, bdy, bof, bid );
        } else {
            if ( j3 >= 2 ) for ( int o = 0; o < nb_in; ++o )  MOVE( 2 + o, j3 + o );
            else           for ( int o = nb_in - 1; o >= 0; --o ) MOVE( 2 + o, j3 + o );
            PUT( 0, pax, pay, dx, dy, off, cut_id );
            PUT( 1, pbx, pby, bdx, bdy, bof, bid );
        }
        #undef MOVE
        #undef PUT
        nb = new_nb;
        return CUT_DONE;
    }

    __device__ TF measure() const {
        TF a = 0;
        for ( int i = 0, j = nb - 1; i < nb; j = i++ ) a += vx[ j ] * vy[ i ] - vx[ i ] * vy[ j ];
        return TF( 0.5 ) * ( a < 0 ? -a : a );
    }

    /// THE SAME SHOELACE, accumulated in FP64 over vertices that remain in FP32. It serves only to settle one
    /// question: when FP32 misses the geometry, is it the CUT that placed the vertices
    /// wrong, or only the shoelace that cancels? On a very elongated cell the terms
    /// `vx * vy` are `L^2` for an area `L * w` -- the cancellation is `L / w`, and this is
    /// exactly what this path removes, without changing the vertices.
    __device__ double measure_hi() const {
        double a = 0;
        for ( int i = 0, j = nb - 1; i < nb; j = i++ )
            a += double( vx[ j ] ) * double( vy[ i ] ) - double( vx[ i ] ) * double( vy[ j ] );
        return 0.5 * ( a < 0 ? -a : a );
    }
};

template<class TF, int MaxNv, bool HI>
__global__ void k2_thread( long long n, const long long *off,
                           const TF *mx, const TF *my, const TF *mw,
                           const TF *px, const TF *py, const TF *pw,
                           TF *out, int *ovf ) {
    const long long i = blockIdx.x * ( long long ) blockDim.x + threadIdx.x;
    if ( i >= n ) return;
    const TF x0 = mx[ i ], y0 = my[ i ], w0 = mw[ i ];
    Cell2G<TF, MaxNv> c;
    c.init_as_box( x0, y0 );
    const long long b = off[ i ], e = off[ i + 1 ];
    int bad = 0, broken = 0;
    for ( long long j = b; j < e; ++j ) {
        const TF dx = px[ j ] - x0, dy = py[ j ] - y0;
        const TF o = TF( 0.5 ) * ( w0 - pw[ j ] + dx * dx + dy * dy );
        int nrun = 0;
        const int r = c.cut( dx, dy, o, int( j - b ), nrun );
        bad |= r == CUT_OVERFLOW;
        broken += r == CUT_UNCHANGED;
    }
    out[ i ] = HI ? TF( c.measure_hi() ) : c.measure();
    if ( bad ) atomicAdd( ovf, 1 );
    atomicAdd( ovf + 1, broken );
}

// ------------------------------------------------------------------ the driver

template<class TF>
struct Dev {
    TF *mx = nullptr, *my = nullptr, *mz = nullptr, *mw = nullptr;
    TF *px = nullptr, *py = nullptr, *pz = nullptr, *pw = nullptr;
    TF *out = nullptr;
    long long *off = nullptr;
    int *ovf = nullptr;
};

/// From the AoS on disk to the SoA on the GPU: separate `mx[i], my[i], ...`, because that is what
/// coalescing asks for -- 32 neighboring threads read 32 consecutive floats.
template<class TF>
static void upload( const Csr &c, Dev<TF> &d, double &ms_h2d ) {
    const int dim = c.dim, k = dim + 1;
    const std::size_t n = std::size_t( c.n ), nv = std::size_t( c.nv );
    std::vector<TF> h( std::max( n, nv ) * 4 );
    auto put = [ & ]( TF **dst, const std::vector<double> &src, std::size_t cnt, int comp ) {
        for ( std::size_t i = 0; i < cnt; ++i ) h[ i ] = TF( src[ i * k + comp ] );
        CK( cudaMalloc( dst, cnt * sizeof( TF ) ) );
        CK( cudaMemcpy( *dst, h.data(), cnt * sizeof( TF ), cudaMemcpyHostToDevice ) );
    };
    cudaEvent_t a, b; CK( cudaEventCreate( &a ) ); CK( cudaEventCreate( &b ) );
    CK( cudaEventRecord( a ) );
    put( &d.mx, c.own, n, 0 ); put( &d.my, c.own, n, 1 );
    if ( dim == 3 ) put( &d.mz, c.own, n, 2 );
    put( &d.mw, c.own, n, dim );
    put( &d.px, c.nbrs, nv, 0 ); put( &d.py, c.nbrs, nv, 1 );
    if ( dim == 3 ) put( &d.pz, c.nbrs, nv, 2 );
    put( &d.pw, c.nbrs, nv, dim );
    CK( cudaMalloc( &d.off, ( n + 1 ) * sizeof( long long ) ) );
    CK( cudaMemcpy( d.off, c.off.data(), ( n + 1 ) * sizeof( long long ), cudaMemcpyHostToDevice ) );
    CK( cudaMalloc( &d.out, n * sizeof( TF ) ) );
    CK( cudaMalloc( &d.ovf, 2 * sizeof( int ) ) );
    CK( cudaEventRecord( b ) ); CK( cudaEventSynchronize( b ) );
    float f = 0; CK( cudaEventElapsedTime( &f, a, b ) ); ms_h2d = f;
}

struct Opt {
    std::string file, map = "warp", prec = "f32", measure = "edges";
    int reps = 5, maxnv = 64, wpb = 4, tpb = 128;
};

template<class TF>
static void run( const Csr &c, const Opt &o ) {
    Dev<TF> d;
    double ms_h2d = 0;
    upload( c, d, ms_h2d );
    const long long n = c.n;
    const bool warp = c.dim == 3 && o.map == "warp";

    size_t shb = 0;
    dim3 grid, blk;
    if ( warp ) {
        const size_t cell = o.maxnv == 32 ? sizeof( WCell<TF, 32> )
                          : o.maxnv == 128 ? sizeof( WCell<TF, 128> ) : sizeof( WCell<TF, 64> );
        shb = cell * o.wpb;
        blk = dim3( 32 * o.wpb );
        grid = dim3( ( unsigned ) ( ( n + o.wpb - 1 ) / o.wpb ) );
        if      ( o.maxnv == 32  ) CK( cudaFuncSetAttribute( k3_warp<TF, 32 >, cudaFuncAttributeMaxDynamicSharedMemorySize, 65536 ) );
        else if ( o.maxnv == 128 ) CK( cudaFuncSetAttribute( k3_warp<TF, 128>, cudaFuncAttributeMaxDynamicSharedMemorySize, 65536 ) );
        else                       CK( cudaFuncSetAttribute( k3_warp<TF, 64 >, cudaFuncAttributeMaxDynamicSharedMemorySize, 65536 ) );
    } else {
        blk = dim3( o.tpb );
        grid = dim3( ( unsigned ) ( ( n + o.tpb - 1 ) / o.tpb ) );
    }

    auto launch = [ & ]() {
        CK( cudaMemset( d.ovf, 0, 2 * sizeof( int ) ) );
        if ( c.dim == 2 ) {
            if ( o.measure == "hi" ) k2_thread<TF, 64, true ><<< grid, blk >>>( n, d.off, d.mx, d.my, d.mw, d.px, d.py, d.pw, d.out, d.ovf );
            else                     k2_thread<TF, 64, false><<< grid, blk >>>( n, d.off, d.mx, d.my, d.mw, d.px, d.py, d.pw, d.out, d.ovf );
        } else if ( warp ) {
            if      ( o.maxnv == 32  ) k3_warp<TF, 32 ><<< grid, blk, shb >>>( n, d.off, d.mx, d.my, d.mz, d.mw, d.px, d.py, d.pz, d.pw, d.out, d.ovf );
            else if ( o.maxnv == 128 ) k3_warp<TF, 128><<< grid, blk, shb >>>( n, d.off, d.mx, d.my, d.mz, d.mw, d.px, d.py, d.pz, d.pw, d.out, d.ovf );
            else                       k3_warp<TF, 64 ><<< grid, blk, shb >>>( n, d.off, d.mx, d.my, d.mz, d.mw, d.px, d.py, d.pz, d.pw, d.out, d.ovf );
        } else {
            if ( o.measure == "faces" ) k3_thread<TF, 64, true ><<< grid, blk >>>( n, d.off, d.mx, d.my, d.mz, d.mw, d.px, d.py, d.pz, d.pw, d.out, d.ovf );
            else                        k3_thread<TF, 64, false><<< grid, blk >>>( n, d.off, d.mx, d.my, d.mz, d.mw, d.px, d.py, d.pz, d.pw, d.out, d.ovf );
        }
    };

    launch();                                            // warm-up: JIT, hidden, clocks
    CK( cudaDeviceSynchronize() );
    CK( cudaGetLastError() );

    cudaEvent_t a, b; CK( cudaEventCreate( &a ) ); CK( cudaEventCreate( &b ) );
    double best = 1e30;
    for ( int r = 0; r < o.reps; ++r ) {
        CK( cudaEventRecord( a ) );
        launch();
        CK( cudaEventRecord( b ) );
        CK( cudaEventSynchronize( b ) );
        float f = 0; CK( cudaEventElapsedTime( &f, a, b ) );
        best = std::min( best, double( f ) );
    }
    CK( cudaGetLastError() );

    std::vector<TF> out( static_cast<std::size_t>( n ) );
    CK( cudaMemcpy( out.data(), d.out, out.size() * sizeof( TF ), cudaMemcpyDeviceToHost ) );
    int ovf2[ 2 ] = { 0, 0 };
    CK( cudaMemcpy( ovf2, d.ovf, 2 * sizeof( int ), cudaMemcpyDeviceToHost ) );
    const int ovf = ovf2[ 0 ], broken = ovf2[ 1 ];

    // THE JUDGE: the sum is not enough, we want the worst RELATIVE deviation cell by cell.
    double sum = 0, emax = 0;
    long long iworst = -1, n6 = 0, n3 = 0;
    for ( long long i = 0; i < n; ++i ) {
        sum += double( out[ i ] );
        const double r = c.ref[ i ];
        if ( r > 0 ) {
            const double e = std::fabs( double( out[ i ] ) - r ) / r;
            if ( e > emax ) { emax = e; iworst = i; }
            n6 += e > 1e-6;
            n3 += e > 1e-3;
        }
    }

    const double ns = 1e6 * best / double( n );
    char occ[ 32 ];
    if ( warp ) std::snprintf( occ, sizeof occ, "wpb %-3d", o.wpb );
    else        std::snprintf( occ, sizeof occ, "tpb %-3d", o.tpb );
    std::printf( " %-6s %-3s %-5s maxnv %-3d %s : %8.3f ms %8.1f ns/seed %7.1f Mcell/s"
                 " sum %.9f max dev %.2e >1e-6 %lld >1e-3 %lld overflow %d [H2D %.1f ms]\n",
                 warp ? "warp" : "thread", o.prec.c_str(), o.measure.c_str(),
                 warp ? o.maxnv : 64, occ,
                 best, ns, double( n ) / best / 1e3, sum, emax, n6, n3, ovf, ms_h2d );
    ( void ) iworst; ( void ) broken;

    cudaFree( d.mx ); cudaFree( d.my ); cudaFree( d.mz ); cudaFree( d.mw );
    cudaFree( d.px ); cudaFree( d.py ); cudaFree( d.pz ); cudaFree( d.pw );
    cudaFree( d.off ); cudaFree( d.out ); cudaFree( d.ovf );
}

int main( int argc, char **argv ) {
    Opt o;
    for ( int i = 1; i < argc; ++i ) {
        const std::string s = argv[ i ];
        if      ( s == "--map"     && i + 1 < argc ) o.map     = argv[ ++i ];
        else if ( s == "--prec"    && i + 1 < argc ) o.prec    = argv[ ++i ];
        else if ( s == "--measure" && i + 1 < argc ) o.measure = argv[ ++i ];
        else if ( s == "--reps"    && i + 1 < argc ) o.reps    = std::atoi( argv[ ++i ] );
        else if ( s == "--maxnv"   && i + 1 < argc ) o.maxnv   = std::atoi( argv[ ++i ] );
        else if ( s == "--wpb"     && i + 1 < argc ) o.wpb     = std::atoi( argv[ ++i ] );
        else if ( s == "--tpb"     && i + 1 < argc ) o.tpb     = std::atoi( argv[ ++i ] );
        else if ( s == "--help" ) {
            std::printf( "usage: cells_gpu DUMP [--map thread|warp] [--prec f32|f64]\n"
                         "                      [--measure edges|faces] [--maxnv 32|64]\n"
                         "                      [--wpb W] [--tpb T] [--reps R]\n" );
            return 0;
        } else o.file = s;
    }
    if ( o.file.empty() ) { std::fprintf( stderr, "a --dump file is required\n" ); return 1; }

    Csr c;
    if ( ! read_csr( o.file.c_str(), c ) ) return 1;
    double refs = 0;
    for ( double v : c.ref ) refs += v;
    std::printf( "%s dim %d n %lld neighbors %.2f reference sum %.9f\n",
                 o.file.c_str(), c.dim, c.n, double( c.nv ) / double( c.n ), refs );

    if ( o.prec == "f64" ) run<double>( c, o );
    else                   run<float>( c, o );
    return 0;
}
