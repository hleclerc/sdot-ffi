#pragma once

// =====================================================================================
// THE DENSITY OF A CARD SOLVE, ON THE HOST SIDE: what `Newton2D.cuh::solve` holds for the density of its cells
// ( `Density2D.cuh` ), and how it changes along the WIDTH CONTINUATION ( `sdotplan/Continuation.h`: the density convolved by a
// gaussian of width `s`, from wide to zero ). One class per kind, the same four things:
//
//   prepare( queue, allocator, in )   the buffers, taken once from the call's pool, and the density as given ( `s = 0` )
//   at( queue, s, moments )           the density at width `s` ( `moments`: with what the moments read too )
//   dev                               what the cells read ( `Problem::dens` )
//   min_scale / possible              `Convolved< Dist >::min_scale` / `possible`, for the stages
//
// THE SAME CONVOLUTION AS THE CPU's ( `sdotplan/convolved/*.h` ), so that the same `Tuning` gives the same stages:
//
//   * an image: its values blurred on its grid, a separable gaussian of `s / step` pixels per axis, truncated at four
//     standard deviations and renormalized at the edge ( axis 0 then axis 1, each output pixel the same sum, in the same
//     order, as `Convolved< Image >::at` ), then the prefix sums of the rows ( `DensImage` );
//   * gaussians: their widths `sqrt( sigma^2 + s^2 )`, nothing else;
//   * a constant: nothing ( the stages, if any, solve the same problem: an `Image` of equal values on the CPU convolves to
//     itself ).
// =====================================================================================

#include "Reduce.cuh"
#include "Density3D.cuh"
#include <algorithm>
#include <cmath>
#include <vector>

namespace sdot::gpu2d {

// ---- what the call hands over ( the generated handler builds one of these around its tensors ) ----------------------------

/// an image: its values flattened `j * nx + i` ( `i` along x ), its geometry in the solver's options
template<class Vals>
struct ImageIn { const Vals &values; };
template<class Vals>
ImageIn<Vals> image_in( const Vals &values ) { return { values }; }

/// gaussians: centres `[ nb, 2 ]`, widths `[ nb ]`, masses `[ nb ]`
template<class P,class S,class M>
struct GaussIn { const P &pos; const S &sigma; const M &mass; };
template<class P,class S,class M>
GaussIn<P,S,M> gauss_in( const P &pos, const S &sigma, const M &mass ) { return { pos, sigma, mass }; }

/// a mesh: nodes `[ nn, 2 ]`, values `[ ne, 3 ]` ( DG1, per corner ), triangles `[ ne, 3 ]`, gradients `[ ne, 6 ]`, the hierarchy `[ nb, 2 ]` twice and `[ nb, 4 ]`
template<class N,class V,class T,class G,class L,class H,class K>
struct MeshIn { const N &nodes; const V &values; const T &tri; const G &grad; const L &lo; const H &hi; const K &links; };
template<class N,class V,class T,class G,class L,class H,class K>
MeshIn<N,V,T,G,L,H,K> mesh_in( const N &nodes, const V &values, const T &tri, const G &grad, const L &lo, const H &hi, const K &links ) {
    return { nodes, values, tri, grad, lo, hi, links };
}

/// a tetrahedral mesh ( 3D, `Density3D.cuh::DensMesh3` ): the same tensors, `[ ne, 4 ]`, gradients `[ ne, 12 ]`, boxes `[ nb, 3 ]`
template<class N,class V,class T,class G,class L,class H,class K>
struct Mesh3In { const N &nodes; const V &values; const T &tet; const G &grad; const L &lo; const H &hi; const K &links; };
template<class N,class V,class T,class G,class L,class H,class K>
Mesh3In<N,V,T,G,L,H,K> mesh3_in( const N &nodes, const V &values, const T &tet, const G &grad, const L &lo, const H &hi, const K &links ) {
    return { nodes, values, tet, grad, lo, hi, links };
}

template<class T> struct IsImageIn : std::false_type {};
template<class V> struct IsImageIn<ImageIn<V>> : std::true_type {};
template<class T> struct IsGaussIn : std::false_type {};
template<class P,class S,class M> struct IsGaussIn<GaussIn<P,S,M>> : std::true_type {};

// ---- the kernels ----------------------------------------------------------------------------------------------------------

template<class TF>
__global__ void __launch_bounds__( BLOCK ) gather_doubles( SI n, Strided<TF,1> src, double *dst ) {
    const SI k = SI( blockIdx.x ) * BLOCK + threadIdx.x;
    if ( k < n ) dst[ k ] = double( src( k ) );
}

/// any strided matrix `[ rows, cols ]` as a flat buffer of `T`, row by row
template<class TS,class T>
__global__ void __launch_bounds__( BLOCK ) gather_matrix( SI rows, SI cols, Strided<TS,2> src, T *dst ) {
    const SI k = SI( blockIdx.x ) * BLOCK + threadIdx.x;
    if ( k < rows * cols ) dst[ k ] = T( src( k / cols, k % cols ) );
}

/// ... and the two columns of a matrix `[ rows, 2 ]` apart
template<class TS>
__global__ void __launch_bounds__( BLOCK ) gather_columns2( SI rows, Strided<TS,2> src, double *c0, double *c1 ) {
    const SI k = SI( blockIdx.x ) * BLOCK + threadIdx.x;
    if ( k < rows ) { c0[ k ] = double( src( k, 0 ) ); c1[ k ] = double( src( k, 1 ) ); }
}

template<class TF>
__global__ void __launch_bounds__( BLOCK ) gather_centres( SI n, Strided<TF,2> src, double *cx, double *cy ) {
    const SI k = SI( blockIdx.x ) * BLOCK + threadIdx.x;
    if ( k < n ) { cx[ k ] = double( src( k, 0 ) ); cy[ k ] = double( src( k, 1 ) ); }
}

/// `s_eff = sqrt( sigma^2 + s^2 )` ( `SumOfGaussians::sigma_of` ), `sigma` at `s = 0`
__global__ void __launch_bounds__( BLOCK ) gauss_widths( SI n, const double *sigma, double s, double *out ) {
    const SI k = SI( blockIdx.x ) * BLOCK + threadIdx.x;
    if ( k < n ) out[ k ] = s > 0 ? sqrt( sigma[ k ] * sigma[ k ] + s * s ) : sigma[ k ];
}

/// the filter's taps `exp( -k^2 / 2 sp^2 )`, `k` in `[ -r, r ]` ( `Convolved< Image >::at` )
__global__ void __launch_bounds__( BLOCK ) blur_taps( SI r, double sp, double *ker ) {
    const SI q = SI( blockIdx.x ) * BLOCK + threadIdx.x;
    if ( q <= 2 * r ) {
        const double k = double( q - r );
        ker[ q ] = exp( -0.5 * ( k * k ) / ( sp * sp ) );
    }
}

/// the sum of the taps that fall inside the axis, per position along it ( the renormalization at the edge )
__global__ void __launch_bounds__( BLOCK ) blur_norms( SI len, SI r, const double *ker, double *wsum ) {
    const SI p = SI( blockIdx.x ) * BLOCK + threadIdx.x;
    if ( p >= len )
        return;
    const SI lo = -r > -p ? -r : -p, hi = r < len - 1 - p ? r : len - 1 - p;
    double s = 0;
    for ( SI k = lo; k <= hi; ++k ) s += ker[ k + r ];
    wsum[ p ] = s;
}

/// a flat buffer of doubles, read like the input ( `Strided` )
struct FlatDoubles {
    const double *p;
    __device__ __forceinline__ double operator()( SI k ) const { return p[ k ]; }
};

/// one axis of the separable filter: `dst = sum_k ker src( . + k ) / wsum`, in the order of the CPU's loop ( `src`: the
/// input as given, or the other axis's output )
template<class Src>
__global__ void __launch_bounds__( BLOCK ) blur_axis( SI nx, SI ny, int axis, SI r, const double *ker, const double *wsum, Src src, double *dst ) {
    const SI flat = SI( blockIdx.x ) * BLOCK + threadIdx.x;
    if ( flat >= nx * ny )
        return;
    const SI i = flat % nx, j = flat / nx;
    const SI p = axis ? j : i, len = axis ? ny : nx, stride = axis ? nx : 1;
    const SI lo = -r > -p ? -r : -p, hi = r < len - 1 - p ? r : len - 1 - p;
    double s = 0;
    for ( SI k = lo; k <= hi; ++k ) s += ker[ k + r ] * double( src( flat + k * stride ) );
    dst[ flat ] = s / wsum[ p ];
}

/// the prefix sums of the rows ( one thread per row ): `s0` always, `s1` / `s2` for the moments ( `DensImage` )
__global__ void __launch_bounds__( BLOCK ) image_prefix( SI nx, SI ny, double hx, const double *v, double *s0, double *s1, double *s2 ) {
    const SI j = SI( blockIdx.x ) * BLOCK + threadIdx.x;
    if ( j >= ny )
        return;
    double a0 = 0, a1 = 0, a2 = 0;
    const double h2 = hx * hx, h3 = h2 * hx;
    for ( SI i = 0; i < nx; ++i ) {
        const SI o = j * ( nx + 1 ) + i;
        const double vv = v[ j * nx + i ], di = double( i );
        s0[ o ] = a0;
        a0 += vv * hx;
        if ( s1 ) {
            s1[ o ] = a1;
            s2[ o ] = a2;
            a1 += vv * h2 * ( 2 * di + 1 ) * 0.5;
            a2 += vv * h3 * ( 3 * di * di + 3 * di + 1 ) / 3;
        }
    }
    s0[ j * ( nx + 1 ) + nx ] = a0;
    if ( s1 ) { s1[ j * ( nx + 1 ) + nx ] = a1; s2[ j * ( nx + 1 ) + nx ] = a2; }
}

// ---- the three kinds ----------------------------------------------------------------------------------------------------

/// a constant: nothing changes with `s`. `possible` and `min_scale` are the CPU's for the distribution it came from ( an
/// `Image` of equal values convolves, no distribution does not: the solver's options say which )
struct ConstDensityHost {
    using Dev = DensConst;
    Dev    dev{};
    bool   possible = false;
    double scale = 0;
    double min_scale( const CudaQueue & ) const { return scale; }
    void   at( const CudaQueue &, double, bool ) {}
};

/// an image ( see the header ): `geom` = `x0, y0, hx, hy, nx, ny`. Four doubles per pixel: the density at the current width
/// `v` and the three prefix sums ( `s1` is the blur's scratch between the two axes: the prefix sums come after ). The input
/// is read where it is, at each width ( the continuation goes from wide to sharp: each stage starts again from it ). `v` is
/// kept rather than taken from the differences of `s0`: a zero pixel stays an exact zero ( the laplacian's facets ).
template<class TFV>
struct ImageDensityHost {
    using Dev = DensImage;
    Dev     dev{};
    static constexpr bool possible = true;
    SI      nx = 0, ny = 0;
    double  x0 = 0, y0 = 0, hx = 1, hy = 1;
    Strided<TFV,1> src{};
    double *v = nullptr, *s0 = nullptr, *s1 = nullptr, *s2 = nullptr, *ker = nullptr, *wsum = nullptr;
    double  cur = -1;                                    ///< the width of what `v` holds ( -1: nothing yet )
    bool    cur_moments = false;

    template<class In>
    bool prepare( const CudaQueue &, auto &allocator, const In &in, const double *geom ) {
        x0 = geom[ 0 ]; y0 = geom[ 1 ]; hx = geom[ 2 ]; hy = geom[ 3 ];
        nx = SI( geom[ 4 ] ); ny = SI( geom[ 5 ] );
        const SI np = nx * ny, nt = ( nx + 1 ) * ny;
        auto vec = [&]( SI m ) { return static_cast<double *>( take( allocator, SI( sizeof( double ) ) * std::max<SI>( m, 1 ) ) ); };
        v = vec( np );
        s0 = vec( nt ); s1 = vec( nt ); s2 = vec( nt );
        const SI kmax = 2 * std::max( nx, ny ) + 1;
        ker = vec( kmax ); wsum = vec( std::max( nx, ny ) );
        src = strided( in.values );
        return v && s0 && s1 && s2 && ker && wsum;
    }

    /// `Convolved< Image >::step( a ) / 4`, the smallest
    double min_scale( const CudaQueue & ) const { return std::min( hx, hy ) / 4; }

    void at( const CudaQueue &queue, double s, bool moments ) {
        bool tables = false;
        if ( s != cur ) {
            const SI np = nx * ny;
            // the axes that blur ( axis 0 first, as the CPU )
            int axes[ 2 ], na = 0;
            double sps[ 2 ];
            for ( int a = 0; a < 2; ++a ) {
                const double sp = s > 0 ? s / ( a ? hy : hx ) : 0;
                if ( sp > 1e-3 ) { sps[ na ] = sp; axes[ na++ ] = a; }
            }
            if ( na == 0 )
                launch_kernel( queue, &gather_doubles<TFV>, blocks_for( np ), BLOCK, 0, np, src, v );
            for ( int q = 0; q < na; ++q ) {
                const int a = axes[ q ];
                const SI len = a ? ny : nx;
                const SI r = std::min<SI>( SI( std::ceil( 4 * sps[ q ] ) ), len );
                double *dst = q + 1 == na ? v : s1;
                launch_kernel( queue, &blur_taps, blocks_for( 2 * r + 1 ), BLOCK, 0, r, sps[ q ], ker );
                launch_kernel( queue, &blur_norms, blocks_for( len ), BLOCK, 0, len, r, ( const double * ) ker, wsum );
                if ( q == 0 )
                    launch_kernel( queue, &blur_axis<Strided<TFV,1>>, blocks_for( np ), BLOCK, 0, nx, ny, a, r, ( const double * ) ker,
                                   ( const double * ) wsum, src, dst );
                else
                    launch_kernel( queue, &blur_axis<FlatDoubles>, blocks_for( np ), BLOCK, 0, nx, ny, a, r, ( const double * ) ker,
                                   ( const double * ) wsum, FlatDoubles{ s1 }, dst );
            }
            cur = s;
            tables = true;
        }
        if ( tables || ( moments && ! cur_moments ) ) {
            launch_kernel( queue, &image_prefix, blocks_for( ny ), BLOCK, 0, nx, ny, hx, ( const double * ) v, s0, moments ? s1 : nullptr,
                           moments ? s2 : nullptr );
            cur_moments = moments;
        }
        dev = DensImage{ x0, y0, hx, hy, nx, ny, v, s0, moments ? s1 : nullptr, moments ? s2 : nullptr };
    }
};

/// gaussians ( see the header )
struct GaussDensityHost {
    using Dev = DensGauss;
    Dev     dev{};
    static constexpr bool possible = true;
    SI      nb = 0;
    double *cx = nullptr, *cy = nullptr, *sigma = nullptr, *s = nullptr, *w = nullptr;
    double  cur = -1;

    template<class In>
    bool prepare( const CudaQueue &queue, auto &allocator, const In &in, const double * ) {
        nb = SI( in.sigma.shape( 0 ) );
        auto vec = [&]( SI m ) { return static_cast<double *>( take( allocator, SI( sizeof( double ) ) * std::max<SI>( m, 1 ) ) ); };
        cx = vec( nb ); cy = vec( nb ); sigma = vec( nb ); s = vec( nb ); w = vec( nb );
        if ( ! cx || ! cy || ! sigma || ! s || ! w )
            return false;
        using TP = std::remove_const_t<typename std::decay_t<decltype( in.pos )>::TF>;
        using TS = std::remove_const_t<typename std::decay_t<decltype( in.sigma )>::TF>;
        using TM = std::remove_const_t<typename std::decay_t<decltype( in.mass )>::TF>;
        if ( nb > 0 ) {
            launch_kernel( queue, &gather_centres<TP>, blocks_for( nb ), BLOCK, 0, nb, strided( in.pos ), cx, cy );
            launch_kernel( queue, &gather_doubles<TS>, blocks_for( nb ), BLOCK, 0, nb, strided( in.sigma ), sigma );
            launch_kernel( queue, &gather_doubles<TM>, blocks_for( nb ), BLOCK, 0, nb, strided( in.mass ), w );
        }
        return true;
    }

    /// `smallest_sigma / 4` ( one read back of the widths )
    double min_scale( const CudaQueue &queue ) const {
        std::vector<double> h( std::max<SI>( nb, 1 ), 0.0 );
        if ( nb > 0 )
            read_back( queue, h.data(), ( const double * ) sigma, nb );
        double r = h[ 0 ];
        for ( SI i = 1; i < nb; ++i ) r = std::min( r, h[ i ] );
        return r / 4;
    }

    void at( const CudaQueue &queue, double sc, bool ) {
        if ( sc != cur && nb > 0 )
            launch_kernel( queue, &gauss_widths, blocks_for( nb ), BLOCK, 0, nb, ( const double * ) sigma, sc, s );
        cur = sc;
        dev = DensGauss{ nb, cx, cy, s, w };
    }
};

/// a mesh ( see the header ): the call's tensors gathered as doubles and ints. No continuation ( `possible` is false: a mesh is
/// already as smooth as a piecewise linear density gets, and the CPU does not convolve it either )
struct MeshDensityHost {
    using Dev = DensMesh;
    Dev     dev{};
    static constexpr bool possible = false;

    template<class In>
    bool prepare( const CudaQueue &queue, auto &allocator, const In &in, const double * ) {
        const SI nn = SI( in.nodes.shape( 0 ) ), ne = SI( in.tri.shape( 0 ) ), nb = SI( in.lo.shape( 0 ) );
        auto vec = [&]( SI m ) { return static_cast<double *>( take( allocator, SI( sizeof( double ) ) * std::max<SI>( m, 1 ) ) ); };
        auto ivec = [&]( SI m ) { return static_cast<int *>( take( allocator, SI( sizeof( int ) ) * std::max<SI>( m, 1 ) ) ); };
        double *px = vec( nn ), *py = vec( nn ), *val = vec( 3 * ne ), *grad = vec( 6 * ne ), *blo = vec( 2 * nb ), *bhi = vec( 2 * nb );
        int *tri = ivec( 3 * ne ), *links = ivec( 4 * nb );
        if ( ! px || ! py || ! val || ! grad || ! blo || ! bhi || ! tri || ! links )
            return false;
        using TN = std::remove_const_t<typename std::decay_t<decltype( in.nodes )>::TF>;
        using TV = std::remove_const_t<typename std::decay_t<decltype( in.values )>::TF>;
        using TT = std::remove_const_t<typename std::decay_t<decltype( in.tri )>::TF>;
        using TG = std::remove_const_t<typename std::decay_t<decltype( in.grad )>::TF>;
        using TL = std::remove_const_t<typename std::decay_t<decltype( in.lo )>::TF>;
        using TH = std::remove_const_t<typename std::decay_t<decltype( in.hi )>::TF>;
        using TK = std::remove_const_t<typename std::decay_t<decltype( in.links )>::TF>;
        launch_kernel( queue, &gather_columns2<TN>, blocks_for( nn ), BLOCK, 0, nn, strided( in.nodes ), px, py );
        launch_kernel( queue, &gather_matrix<TV,double>, blocks_for( 3 * ne ), BLOCK, 0, ne, SI( 3 ), strided( in.values ), val );
        launch_kernel( queue, &gather_matrix<TT,int>, blocks_for( 3 * ne ), BLOCK, 0, ne, SI( 3 ), strided( in.tri ), tri );
        launch_kernel( queue, &gather_matrix<TG,double>, blocks_for( 6 * ne ), BLOCK, 0, ne, SI( 6 ), strided( in.grad ), grad );
        launch_kernel( queue, &gather_matrix<TL,double>, blocks_for( 2 * nb ), BLOCK, 0, nb, SI( 2 ), strided( in.lo ), blo );
        launch_kernel( queue, &gather_matrix<TH,double>, blocks_for( 2 * nb ), BLOCK, 0, nb, SI( 2 ), strided( in.hi ), bhi );
        launch_kernel( queue, &gather_matrix<TK,int>, blocks_for( 4 * nb ), BLOCK, 0, nb, SI( 4 ), strided( in.links ), links );
        dev = DensMesh{ ne, nb, px, py, val, tri, grad, blo, bhi, links };
        return true;
    }

    double min_scale( const CudaQueue & ) const { return 0; }
    void   at( const CudaQueue &, double, bool ) {}
};

/// a tetrahedral mesh: the call's tensors gathered as doubles and ints, flat ( no continuation, as `MeshDensityHost` )
struct Mesh3DensityHost {
    using Dev = gpu3d::DensMesh3;
    Dev     dev{};
    static constexpr bool possible = false;

    template<class In>
    bool prepare( const CudaQueue &queue, auto &allocator, const In &in, const double * ) {
        const SI nn = SI( in.nodes.shape( 0 ) ), ne = SI( in.tet.shape( 0 ) ), nb = SI( in.lo.shape( 0 ) );
        auto vec = [&]( SI m ) { return static_cast<double *>( take( allocator, SI( sizeof( double ) ) * std::max<SI>( m, 1 ) ) ); };
        auto ivec = [&]( SI m ) { return static_cast<int *>( take( allocator, SI( sizeof( int ) ) * std::max<SI>( m, 1 ) ) ); };
        double *pn = vec( 3 * nn ), *val = vec( 4 * ne ), *grad = vec( 12 * ne ), *blo = vec( 3 * nb ), *bhi = vec( 3 * nb );
        int *tet = ivec( 4 * ne ), *links = ivec( 4 * nb );
        if ( ! pn || ! val || ! grad || ! blo || ! bhi || ! tet || ! links )
            return false;
        using TN = std::remove_const_t<typename std::decay_t<decltype( in.nodes )>::TF>;
        using TV = std::remove_const_t<typename std::decay_t<decltype( in.values )>::TF>;
        using TT = std::remove_const_t<typename std::decay_t<decltype( in.tet )>::TF>;
        using TG = std::remove_const_t<typename std::decay_t<decltype( in.grad )>::TF>;
        using TL = std::remove_const_t<typename std::decay_t<decltype( in.lo )>::TF>;
        using TH = std::remove_const_t<typename std::decay_t<decltype( in.hi )>::TF>;
        using TK = std::remove_const_t<typename std::decay_t<decltype( in.links )>::TF>;
        launch_kernel( queue, &gather_matrix<TN,double>, blocks_for( 3 * nn ), BLOCK, 0, nn, SI( 3 ), strided( in.nodes ), pn );
        launch_kernel( queue, &gather_matrix<TV,double>, blocks_for( 4 * ne ), BLOCK, 0, ne, SI( 4 ), strided( in.values ), val );
        launch_kernel( queue, &gather_matrix<TT,int>, blocks_for( 4 * ne ), BLOCK, 0, ne, SI( 4 ), strided( in.tet ), tet );
        launch_kernel( queue, &gather_matrix<TG,double>, blocks_for( 12 * ne ), BLOCK, 0, ne, SI( 12 ), strided( in.grad ), grad );
        launch_kernel( queue, &gather_matrix<TL,double>, blocks_for( 3 * nb ), BLOCK, 0, nb, SI( 3 ), strided( in.lo ), blo );
        launch_kernel( queue, &gather_matrix<TH,double>, blocks_for( 3 * nb ), BLOCK, 0, nb, SI( 3 ), strided( in.hi ), bhi );
        launch_kernel( queue, &gather_matrix<TK,int>, blocks_for( 4 * nb ), BLOCK, 0, nb, SI( 4 ), strided( in.links ), links );
        dev = Dev{ ne, nb, pn, val, tet, grad, blo, bhi, links };
        return true;
    }

    double min_scale( const CudaQueue & ) const { return 0; }
    void   at( const CudaQueue &, double, bool ) {}
};

/// the host class of what the call hands over: a 0-d tensor ( or a number ) is a constant
template<class In> struct DensityHostOf { using type = ConstDensityHost; };
template<class V> struct DensityHostOf<ImageIn<V>> { using type = ImageDensityHost<std::remove_const_t<typename V::TF>>; };
template<class P,class S,class M> struct DensityHostOf<GaussIn<P,S,M>> { using type = GaussDensityHost; };
template<class N,class V,class T,class G,class L,class H,class K> struct DensityHostOf<MeshIn<N,V,T,G,L,H,K>> { using type = MeshDensityHost; };
template<class N,class V,class T,class G,class L,class H,class K> struct DensityHostOf<Mesh3In<N,V,T,G,L,H,K>> { using type = Mesh3DensityHost; };

} // namespace sdot::gpu2d
