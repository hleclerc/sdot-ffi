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

/// one axis of the separable filter: `dst = sum_k ker src( . + k ) / wsum`, in the order of the CPU's loop
__global__ void __launch_bounds__( BLOCK ) blur_axis( SI nx, SI ny, int axis, SI r, const double *ker, const double *wsum, const double *src, double *dst ) {
    const SI flat = SI( blockIdx.x ) * BLOCK + threadIdx.x;
    if ( flat >= nx * ny )
        return;
    const SI i = flat % nx, j = flat / nx;
    const SI p = axis ? j : i, len = axis ? ny : nx, stride = axis ? nx : 1;
    const SI lo = -r > -p ? -r : -p, hi = r < len - 1 - p ? r : len - 1 - p;
    double s = 0;
    for ( SI k = lo; k <= hi; ++k ) s += ker[ k + r ] * src[ flat + k * stride ];
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

/// an image ( see the header ): `geom` = `x0, y0, hx, hy, nx, ny`
struct ImageDensityHost {
    using Dev = DensImage;
    Dev     dev{};
    static constexpr bool possible = true;
    SI      nx = 0, ny = 0;
    double  x0 = 0, y0 = 0, hx = 1, hy = 1;
    double *v0 = nullptr, *v = nullptr, *tmp = nullptr, *s0 = nullptr, *s1 = nullptr, *s2 = nullptr, *ker = nullptr, *wsum = nullptr;
    double  cur = -1;                                    ///< the width of what `v` holds ( -1: nothing yet )
    bool    cur_moments = false;

    template<class In>
    bool prepare( const CudaQueue &queue, auto &allocator, const In &in, const double *geom ) {
        x0 = geom[ 0 ]; y0 = geom[ 1 ]; hx = geom[ 2 ]; hy = geom[ 3 ];
        nx = SI( geom[ 4 ] ); ny = SI( geom[ 5 ] );
        const SI np = nx * ny, nt = ( nx + 1 ) * ny;
        auto vec = [&]( SI m ) { return static_cast<double *>( take( allocator, SI( sizeof( double ) ) * std::max<SI>( m, 1 ) ) ); };
        v0 = vec( np ); v = vec( np ); tmp = vec( np );
        s0 = vec( nt ); s1 = vec( nt ); s2 = vec( nt );
        const SI kmax = 2 * std::max( nx, ny ) + 1;
        ker = vec( kmax ); wsum = vec( std::max( nx, ny ) );
        if ( ! v0 || ! v || ! tmp || ! s0 || ! s1 || ! s2 || ! ker || ! wsum )
            return false;
        using TFV = std::remove_const_t<typename std::decay_t<decltype( in.values )>::TF>;
        launch_kernel( queue, &gather_doubles<TFV>, blocks_for( np ), BLOCK, 0, np, strided( in.values ), v0 );
        return true;
    }

    /// `Convolved< Image >::step( a ) / 4`, the smallest
    double min_scale( const CudaQueue & ) const { return std::min( hx, hy ) / 4; }

    void at( const CudaQueue &queue, double s, bool moments ) {
        bool tables = false;
        if ( s != cur ) {
            const SI np = nx * ny;
            cuda_check( cudaMemcpyAsync( v, v0, sizeof( double ) * np, cudaMemcpyDeviceToDevice, queue.stream ), "copy of the image" );
            if ( s > 0 ) {
                for ( int a = 0; a < 2; ++a ) {
                    const SI len = a ? ny : nx;
                    const double sp = s / ( a ? hy : hx );
                    if ( ! ( sp > 1e-3 ) )
                        continue;
                    const SI r = std::min<SI>( SI( std::ceil( 4 * sp ) ), len );
                    launch_kernel( queue, &blur_taps, blocks_for( 2 * r + 1 ), BLOCK, 0, r, sp, ker );
                    launch_kernel( queue, &blur_norms, blocks_for( len ), BLOCK, 0, len, r, ( const double * ) ker, wsum );
                    launch_kernel( queue, &blur_axis, blocks_for( np ), BLOCK, 0, nx, ny, a, r, ( const double * ) ker, ( const double * ) wsum,
                                   ( const double * ) v, tmp );
                    std::swap( v, tmp );
                }
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

/// the host class of what the call hands over: a 0-d tensor ( or a number ) is a constant
template<class In> struct DensityHostOf { using type = ConstDensityHost; };
template<class V> struct DensityHostOf<ImageIn<V>> { using type = ImageDensityHost; };
template<class P,class S,class M> struct DensityHostOf<GaussIn<P,S,M>> { using type = GaussDensityHost; };

} // namespace sdot::gpu2d
