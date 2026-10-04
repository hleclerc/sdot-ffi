// THE CPU MULTIGRID ( `sdotplan/Multigrid.h` ) ON THE DUMPED SYSTEMS ( see `linbench.cu` ): the iteration counts of its variants
// ( packets, smoothing degree, truncation, ... ) on the very systems of the card's Newton solves -- what told how many
// iterations the smoothed aggregation would save before it was written for the card.
//
//   g++ -std=c++20 -O3 -march=native -fopenmp -I<sdot-ffi>/include -I<loom>/include -I<eigen> -o cpumg cpumg.cpp
//   OMP_NUM_THREADS=8 ./cpumg <prefix> <first> <count> [pack= nu= truncate= omega= stop= rebuild= recycle= cheb= tol= levels=1]
#ifdef _OPENMP
#  include <omp.h>
#endif
#define SDOT_EIGEN 1
#include <Eigen/SparseCholesky>
#include <Eigen/SparseCore>
#define private public                                   // ( the levels, for `levels=1`: a bench tool )
#include <sdot/sdotplan/Multigrid.h>
#undef private
#include <cstdio>
#include <cstring>
#include <map>
#include <string>

using namespace sdot; using namespace sdot::sdotplan;

static bool load( const std::string &name, Laplacian &L, std::vector<double> &b ) {
    FILE *f = std::fopen( name.c_str(), "rb" );
    if ( ! f ) return false;
    SI head[ 3 ];
    if ( std::fread( head, sizeof( SI ), 3, f ) != 3 || head[ 2 ] != 4 ) { std::fclose( f ); return false; }
    const SI n = head[ 0 ], nnz = head[ 1 ];
    L.n = n;
    L.row.resize( n + 1 ); L.col.resize( nnz ); L.c.resize( nnz ); L.dia.resize( n ); b.resize( n );
    std::vector<std::uint32_t> col( nnz );
    bool ok = std::fread( L.row.data(), sizeof( SI ), n + 1, f ) == size_t( n + 1 ) && std::fread( col.data(), 4, nnz, f ) == size_t( nnz )
           && std::fread( L.c.data(), 8, nnz, f ) == size_t( nnz ) && std::fread( L.dia.data(), 8, n, f ) == size_t( n )
           && std::fread( b.data(), 8, n, f ) == size_t( n );
    for ( SI e = 0; e < nnz; ++e ) L.col[ e ] = col[ e ];
    std::fclose( f );
    return ok;
}

int main( int argc, char **argv ) {
    std::string prefix = argv[ 1 ];
    int first = std::atoi( argv[ 2 ] ), count = std::atoi( argv[ 3 ] );
    std::map<std::string,double> kv;
    for ( int a = 4; a < argc; ++a ) {
        const char *eq = std::strchr( argv[ a ], '=' );
        if ( eq ) kv[ std::string( argv[ a ], size_t( eq - argv[ a ] ) ) ] = std::atof( eq + 1 );
    }
    auto get = [&]( const char *k, double d ) { auto it = kv.find( k ); return it == kv.end() ? d : it->second; };
    Mg mg;
    mg.pack = int( get( "pack", 8 ) );
    mg.nu = int( get( "nu", 3 ) );
    mg.truncate = get( "truncate", 0.2 );
    mg.omega_p = get( "omega", 0.7 );
    mg.stop = int( get( "stop", 1000 ) );
    mg.rebuild = int( get( "rebuild", 4 ) );
    mg.recycle = int( get( "recycle", 2 ) );
    mg.cheb = get( "cheb", 10 );
    mg.tol = get( "tol", 1e-6 );
    int tot = 0;
    for ( int k = first; k < first + count; ++k ) {
        Laplacian L;
        std::vector<double> b, d;
        if ( ! load( prefix + "_" + std::to_string( k ) + ".bin", L, b ) ) { std::printf( "cannot load %d\n", k ); return 1; }
        if ( k == first ) {
            std::vector<SI> id( L.n );
            for ( SI i = 0; i < L.n; ++i ) id[ i ] = i;
            mg.order( id );
        }
        const int it0 = mg.st.nb_iter;
        const double t0 = mg_now();
        mg.solves( L, b, d );
        std::printf( " %d", mg.st.nb_iter - it0 );
        std::fflush( stdout );
        tot += mg.st.nb_iter - it0;
        (void) t0;
    }
    if ( get( "levels", 0 ) ) {
        std::printf( "\n" );
        for ( size_t l = 0; l < mg.lev.size(); ++l )
            std::printf( "  level %zu: n %lld nnz %lld ( %.1f / row )  P nnz %lld\n", l, mg.lev[ l ].n, mg.lev[ l ].nnz, double( mg.lev[ l ].nnz ) / mg.lev[ l ].n,
                         l < mg.prol.size() ? SI( mg.prol[ l ].col.size() ) : SI( 0 ) );
    }
    std::printf( "  | TOTAL %d ( %.1f / solve ), t_hier %.2f s t_res %.2f s\n", tot, double( tot ) / count, mg.st.t_hierarchy, mg.st.t_res );
}
