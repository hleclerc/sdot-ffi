#pragma once

// =====================================================================================
// REDUCTIONS ON THE CARD THAT GIVE THE SAME NUMBER AT EVERY RUN ( what the Newton of `Newton2D.cuh` and the
// linear solvers of `Linear2D.cuh` decide on ).
//
// A floating-point sum made of atomics depends on the order in which the blocks arrive: two runs of the same solve
// would then differ in the last bits of a merit, and a line search that compares `n2r <= ( 1 - t / 2 ) nr` may take
// another branch -- another iteration count, another diagram count, a solve that cannot be compared with itself.
// So every reduction here is a TREE OF FIXED SHAPE:
//
//   * a first launch of `RED_GRID` blocks ( a fixed number, not the card's ), each thread striding over the items
//     with the same stride at every run, each block combining its threads in shared memory by a fixed pairing;
//   * a second launch of ONE block combining the partials of the blocks in the same way.
//
// The accumulator is a plain struct with `identity()` and `combine( other )` ( sums, minima, maxima, counts ); the item
// function is a functor `f( i, acc )` ( a struct and not a lambda: nvcc's extended lambdas do not like every
// enclosing scope that loom generates ). Nothing is read back: the result stays on the card, where the next kernel
// reads it -- the caller reads it back only when the HOST has to decide something.
// =====================================================================================

#include "Cell2D.cuh"

namespace sdot::gpu2d {

constexpr int RED_GRID = 240;                            ///< blocks of the first pass ( a fixed number: the same tree at every run )

/// the threads of a block combined in shared memory, always in the same order; the result is that of thread 0
template<class Acc>
__device__ __forceinline__ Acc block_combine( Acc v ) {
    __shared__ __align__( 16 ) unsigned char raw[ BLOCK * sizeof( Acc ) ];
    Acc *tmp = reinterpret_cast<Acc *>( raw );
    const int t = threadIdx.x;
    tmp[ t ] = v;
    __syncthreads();
    for ( int s = BLOCK / 2; s > 0; s /= 2 ) {
        if ( t < s ) {
            Acc a = tmp[ t ];
            a.combine( tmp[ t + s ] );
            tmp[ t ] = a;
        }
        __syncthreads();
    }
    Acc res = tmp[ 0 ];
    __syncthreads();                                     // `raw` may be reused by a second call in the same kernel
    return res;
}

template<class Acc,class F>
__global__ void __launch_bounds__( BLOCK ) reduce_pass( SI n, F f, Acc *partials ) {
    Acc acc = Acc::identity();
    for ( SI i = SI( blockIdx.x ) * BLOCK + threadIdx.x; i < n; i += SI( gridDim.x ) * BLOCK )
        f( i, acc );
    acc = block_combine( acc );
    if ( threadIdx.x == 0 )
        partials[ blockIdx.x ] = acc;
}

template<class Acc>
__global__ void __launch_bounds__( BLOCK ) reduce_last( int nb, const Acc *partials, Acc *out ) {
    Acc acc = Acc::identity();
    for ( int i = threadIdx.x; i < nb; i += BLOCK )
        acc.combine( partials[ i ] );
    acc = block_combine( acc );
    if ( threadIdx.x == 0 )
        *out = acc;
}

/// `*out = combine_i f( i )` over `[ 0, n )`, on the call's stream, nothing read back. `partials`: `RED_GRID` slots.
template<class Acc,class F>
void reduce( const CudaQueue &queue, SI n, const F &f, Acc *partials, Acc *out ) {
    const int grid = int( std::min<SI>( RED_GRID, std::max<SI>( 1, ( n + BLOCK - 1 ) / BLOCK ) ) );
    launch_kernel( queue, &reduce_pass<Acc,F>, grid, BLOCK, 0, n, f, partials );
    launch_kernel( queue, &reduce_last<Acc>, 1, BLOCK, 0, grid, partials, out );
}

// ---- the accumulators that come back everywhere -------------------------------------------------------------------

/// one sum
struct Sum1 {
    double s;
    __host__ __device__ static Sum1 identity() { return { 0.0 }; }
    __device__ void combine( const Sum1 &o ) { s += o.s; }
};

/// two sums ( a fused kernel that wants two dot products )
struct Sum2 {
    double s0, s1;
    __host__ __device__ static Sum2 identity() { return { 0.0, 0.0 }; }
    __device__ void combine( const Sum2 &o ) { s0 += o.s0; s1 += o.s1; }
};

/// three sums ( the second step of the K-cycle )
struct Sum3 {
    double s0, s1, s2;
    __host__ __device__ static Sum3 identity() { return { 0.0, 0.0, 0.0 }; }
    __device__ void combine( const Sum3 &o ) { s0 += o.s0; s1 += o.s1; s2 += o.s2; }
};

/// a maximum
struct Max1 {
    double m;
    __host__ __device__ static Max1 identity() { return { 0.0 }; }
    __device__ void combine( const Max1 &o ) { m = fmax( m, o.m ); }
};

/// a minimum
struct Min1 {
    double m;
    __host__ __device__ static Min1 identity() { return { 1e300 }; }
    __device__ void combine( const Min1 &o ) { m = fmin( m, o.m ); }
};

/// a minimum and a count
struct MinCount {
    double m;
    unsigned long long c;
    __host__ __device__ static MinCount identity() { return { 1e300, 0ull }; }
    __device__ void combine( const MinCount &o ) { m = fmin( m, o.m ); c += o.c; }
};

/// the extent of a cloud ( the similarity start )
struct Extent2 {
    double lo[ 2 ], hi[ 2 ];
    __host__ __device__ static Extent2 identity() { return { { 1e300, 1e300 }, { -1e300, -1e300 } }; }
    __device__ void combine( const Extent2 &o ) {
        for ( int d = 0; d < 2; ++d ) { lo[ d ] = fmin( lo[ d ], o.lo[ d ] ); hi[ d ] = fmax( hi[ d ], o.hi[ d ] ); }
    }
};

/// the same in any dimension
template<int D>
struct ExtentN {
    double lo[ D ], hi[ D ];
    __host__ __device__ static ExtentN identity() {
        ExtentN e;
        for ( int d = 0; d < D; ++d ) { e.lo[ d ] = 1e300; e.hi[ d ] = -1e300; }
        return e;
    }
    __device__ void combine( const ExtentN &o ) {
        for ( int d = 0; d < D; ++d ) { lo[ d ] = fmin( lo[ d ], o.lo[ d ] ); hi[ d ] = fmax( hi[ d ], o.hi[ d ] ); }
    }
};

// ---- elementwise helpers ----------------------------------------------------------------------------------------

__global__ void __launch_bounds__( BLOCK ) fill_value( double *x, double v, SI n ) {
    const SI i = SI( blockIdx.x ) * BLOCK + threadIdx.x;
    if ( i < n ) x[ i ] = v;
}

__global__ void __launch_bounds__( BLOCK ) copy_values( double *dst, const double *src, SI n ) {
    const SI i = SI( blockIdx.x ) * BLOCK + threadIdx.x;
    if ( i < n ) dst[ i ] = src[ i ];
}

/// `x -= *s * scale` ( a mean, a gauge value: read on the card )
__global__ void __launch_bounds__( BLOCK ) subtract_scalar( double *x, const double *s, double scale, SI n ) {
    const SI i = SI( blockIdx.x ) * BLOCK + threadIdx.x;
    if ( i < n ) x[ i ] -= *s * scale;
}

/// the sum of a vector, into a device scalar
struct SumOf {
    const double *x;
    __device__ void operator()( SI i, Sum1 &acc ) const { acc.s += x[ i ]; }
};

/// `u . v`
struct DotOf {
    const double *u, *v;
    __device__ void operator()( SI i, Sum1 &acc ) const { acc.s += u[ i ] * v[ i ]; }
};

/// the slots of a reduction: the partials of the first pass and the result, taken once from the call's pool
template<class Acc>
struct RedSlot {
    Acc *partials = nullptr, *out = nullptr;
    bool take_from( auto &allocator ) {
        partials = static_cast<Acc *>( take( allocator, SI( sizeof( Acc ) ) * RED_GRID ) );
        out      = static_cast<Acc *>( take( allocator, SI( sizeof( Acc ) ) ) );
        return partials && out;
    }
};

} // namespace sdot::gpu2d
