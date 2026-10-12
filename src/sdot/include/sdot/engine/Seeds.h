#pragma once

#include <loom/support/common_macros.h> // HD
#include <loom/support/common_types.h>
#include <loom/support/containers/Vector.h>
#include <loom/support/Ct.h>
#include "../cell/Providers.h"

// (the types that cross the boundary are `visibility( "default" )`: a function template is as visible as the least
// visible of its arguments, and kernels are compiled `-fvisibility=hidden`)
// =====================================================================================
// THE SEEDS OF A DIAGRAM AS THE CELL ENGINE SEES THEM, with no tensor type in sight.
//
// The storages ( `PowerDiagram_Plain`, `PowerDiagram_Bsp` ) are templates on the views of the call that
// instantiates them: every layout, every axis name, every kernel has its own type. The engine -- the loop that
// cuts a cell with the planes a provider yields -- only needs to READ a few arrays, and the `sdot_engine` module
// compiles it ONCE, in a shared library, for these canonical storages: raw pointers and BYTE strides read at run
// time, the element types and the structural choices ( weights ? memory ? ) as template parameters.
//
// They answer what the providers ask of a storage ( `cell/Providers.h` ): `point`, `weight`, `nb_seeds`, and for the
// tree `tree.node_*`, `memo_*`.
// =====================================================================================

namespace sdot {
namespace engine {

template<class T> struct __attribute__(( visibility( "default" ) )) V1 {
    const char *p; SI s0, n0;
    HD T  operator()( SI i ) const { return *reinterpret_cast<const T *>( p + i * s0 ); }
    HD SI shape( int ) const { return n0; }
};
template<class T> struct __attribute__(( visibility( "default" ) )) V2 {
    const char *p; SI s0, s1, n0, n1;
    HD T  operator()( SI i, SI j ) const { return *reinterpret_cast<const T *>( p + i * s0 + j * s1 ); }
    HD SI shape( int d ) const { return d == 0 ? n0 : n1; }
};
template<class T> struct __attribute__(( visibility( "default" ) )) V3 {
    const char *p; SI s0, s1, s2, n0, n1, n2;
    HD T  operator()( SI i, SI j, SI k ) const { return *reinterpret_cast<const T *>( p + i * s0 + j * s1 + k * s2 ); }
    HD SI shape( int d ) const { return d == 0 ? n0 : d == 1 ? n1 : n2; }
};

// the views of a tensor of the call, by raw address and byte strides
template<class T> HD V1<T> view1( const auto &t ) {
    return { t.data().template as<const char>(), SI( t.strides()[ Ct<int,0>() ] ), SI( t.shape( Ct<int,0>() ) ) };
}
template<class T> HD V2<T> view2( const auto &t ) {
    return { t.data().template as<const char>(), SI( t.strides()[ Ct<int,0>() ] ), SI( t.strides()[ Ct<int,1>() ] ), SI( t.shape( Ct<int,0>() ) ), SI( t.shape( Ct<int,1>() ) ) };
}
template<class T> HD V3<T> view3( const auto &t ) {
    return { t.data().template as<const char>(), SI( t.strides()[ Ct<int,0>() ] ), SI( t.strides()[ Ct<int,1>() ] ), SI( t.strides()[ Ct<int,2>() ] ),
             SI( t.shape( Ct<int,0>() ) ), SI( t.shape( Ct<int,1>() ) ), SI( t.shape( Ct<int,2>() ) ) };
}

// the same for a member that may be unbound ( a `NoneTensor`: no weights, no memory ): an empty view
template<class T> HD V1<T> opt_view1( const auto &t ) { if constexpr ( DECAYED_TYPE_OF( t )::is_valid ) return view1<T>( t ); else return {}; }
template<class T> HD V2<T> opt_view2( const auto &t ) { if constexpr ( DECAYED_TYPE_OF( t )::is_valid ) return view2<T>( t ); else return {}; }

/// the seeds in their order, cut by all the others ( `PowerDiagram_Plain` )
template<class TR,int D,bool WEIGHTED>
struct __attribute__(( visibility( "default" ) )) PlainSeeds {
    static constexpr int  ct_dim      = D;
    static constexpr bool on_cpu      = true;
    static constexpr bool has_weights = WEIGHTED;
    using TF = TR;

    V2<TR> positions;
    V1<TR> weights;

    HD SI   nb_seeds() const { return positions.n0; }
    HD auto point( SI k ) const { return Vector<TF,D>::with_func( [&]( PI d ) { return TF( positions( k, d ) ); } ); }
    HD TF   weight( SI k ) const { if constexpr ( WEIGHTED ) return weights( k ); else return TF( 0 ); }

    template<class TK>
    HD auto provider( SI k0 ) const { return ProviderAll<PlainSeeds,TK,D>( *this, k0 ); }
};

/// the seeds in the order of the tree, and the tree ( `PowerDiagram_Bsp` )
template<class TR,class TI,int D,bool WEIGHTED,bool MEMO>
struct __attribute__(( visibility( "default" ) )) BspSeeds {
    static constexpr int  ct_dim      = D;
    static constexpr bool on_cpu      = true;
    static constexpr bool has_weights = WEIGHTED;
    using TF = TR;

    struct Tree {
        V1<TI> node_begin, node_end;
        V3<TR> node_box;
        V2<TR> node_wa;
        V1<TR> node_wb;
    };
    V2<TR>  positions;
    V1<TR>  weights;
    Tree    tree;
    V2<int> memo_nbrs;
    V1<int> memo_counts;

    HD SI   nb_seeds() const { return positions.n0; }
    HD auto point( SI k ) const { return Vector<TF,D>::with_func( [&]( PI d ) { return TF( positions( k, d ) ); } ); }
    HD TF   weight( SI k ) const { if constexpr ( WEIGHTED ) return weights( k ); else return TF( 0 ); }

    template<class TK>
    HD auto provider( SI k0 ) const { return ProviderBsp<BspSeeds,TK,D,WEIGHTED,MEMO>( *this, k0 ); }
};

} // namespace engine
} // namespace sdot
