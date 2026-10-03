// THE 2D COUNTERPART OF `power_3d_light.cpp` : what a lightened version of CGAL would cost in 2D.
//
// The question asked is that of the KERNEL. `Exact_predicates_inexact_constructions_kernel` (Epick)
// evaluates each orientation / in_power_circle predicate first in INTERVAL ARITHMETIC ; if
// the interval contains zero -- that is, if the sign is not certain -- it replays the predicate
// in exact arithmetic. `-DLIGHT_KERNEL` replaces all of it with `Simple_cartesian<double>` : the predicate becomes
// a bare FP64 determinant, with no interval and no fallback. This is EXACTLY what our cut does.
//
// The difference between the two binaries is therefore the price of robustness, and nothing else : same
// algorithm, same structure, same insertion order (Hilbert, bulk insertion).
//
//   `--cells` chains OUR 2D cell (`CellSoAT`) and OUR measure on the neighbours returned by the
//   triangulation. This is the "connectivity replay" : only the FINAL cuts are attempted.
//
// Warning : without a filter, a degeneracy can break the triangulation. The checks are the
// number of vertices and the sum of the areas (which must equal 1).

#ifdef LIGHT_KERNEL
#   include <CGAL/Simple_cartesian.h>
using K = CGAL::Simple_cartesian<double>;
static constexpr const char *kernel_name = "Simple_cartesian<double> (bare FP64 predicates)";
#else
#   include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
using K = CGAL::Exact_predicates_inexact_constructions_kernel;
static constexpr const char *kernel_name = "Epick (exact filtered predicates)";
#endif

#include <CGAL/Regular_triangulation_2.h>

#include "geometry/Cell.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <string>
#include <vector>
#include "dump_csr.h"

using Rt   = CGAL::Regular_triangulation_2<K>;
using Wp   = Rt::Weighted_point;
using Bare = Rt::Bare_point;

namespace {

std::size_t vertex_count( const Rt &rt ) { return rt.number_of_vertices(); }

double now() {
    using namespace std::chrono;
    return duration<double>( steady_clock::now().time_since_epoch() ).count();
}

bool load_cloud( const std::string &path, std::vector<Wp> &pts ) {
    std::FILE *f = std::fopen( path.c_str(), "rb" );
    if ( ! f ) { std::printf( "cannot open '%s'\n", path.c_str() ); return false; }
    std::fseek( f, 0, SEEK_END );
    const long sz = std::ftell( f );
    std::fseek( f, 0, SEEK_SET );
    std::string buf( std::size_t( sz ) + 1, '\0' );
    const std::size_t rd = std::fread( buf.data(), 1, std::size_t( sz ), f );
    std::fclose( f );
    buf[ rd ] = 0;
    const char *p = buf.data(), *e = p + rd;
    auto num = [ & ]( double &out ) {
        for ( ;; ) {
            while ( p < e && ( *p == ' ' || *p == '\t' || *p == '\n' || *p == '\r' ) ) ++p;
            if ( p < e && *p == '#' ) { while ( p < e && *p != '\n' ) ++p; continue; }
            break;
        }
        if ( p >= e ) return false;
        char *q = nullptr;
        out = std::strtod( p, &q );
        if ( q == p ) return false;
        p = q;
        return true;
    };
    double v;
    if ( ! num( v ) ) return false;
    const int n = int( v );
    pts.clear();
    pts.reserve( n );
    for ( int i = 0; i < n; ++i ) {
        double x, y, w;
        if ( ! num( x ) || ! num( y ) || ! num( w ) ) return false;
        pts.emplace_back( Bare( x, y ), w );
    }
    return true;
}

} // namespace

int main( int argc, char **argv ) {
    int n = 1000000, reps = 3;
    int decoy = 0;
    bool cells = false;
    std::string load, dump;
    for ( int i = 1; i < argc; ++i ) {
        const std::string s = argv[ i ];
        if ( s == "--load" && i + 1 < argc ) load = argv[ ++i ];
        else if ( s == "--dump" && i + 1 < argc ) dump = argv[ ++i ];
        else if ( s == "--reps" && i + 1 < argc ) reps = std::atoi( argv[ ++i ] );
        else if ( s == "--cells" ) cells = true;
        else if ( s == "--decoy" && i + 1 < argc ) decoy = std::atoi( argv[ ++i ] );
        else if ( s == "--help" ) {
            std::printf( "usage: power_2d_light [n] [--load F] [--reps R] [--cells]\n" );
            return 0;
        } else n = std::atoi( argv[ i ] );
    }

    std::vector<Wp> pts;
    if ( ! load.empty() ) {
        if ( ! load_cloud( load, pts ) ) return 1;
        n = int( pts.size() );
    } else {
        std::mt19937_64 gen( 0 );
        std::uniform_real_distribution<double> u( 0.01, 0.99 );
        pts.reserve( n );
        for ( int i = 0; i < n; ++i ) pts.emplace_back( Bare( u( gen ), u( gen ) ), 0.0 );
    }

    double tt = 1e30, ta = 1e30, tc = 1e30;
    double neighbors = 0, sum = 0;
    std::size_t vertices = 0, hidden = 0;

    for ( int r = 0; r < reps; ++r ) {
        const double t0 = now();
        Rt rt;
        rt.insert( pts.begin(), pts.end() );
        const double t1 = now();

        // THE ADJACENCY. In 2D it is ALREADY light : `incident_vertices` is a circulator that follows
        // the faces around the vertex, with no temporary associative container -- unlike the
        // 3D version. There is therefore no "home-made" variant to oppose here.
        std::vector<Rt::Vertex_handle> adj;
        double vs = 0;
        for ( auto v = rt.finite_vertices_begin(); v != rt.finite_vertices_end(); ++v ) {
            adj.clear();
            Rt::Vertex_circulator c = rt.incident_vertices( v ), done( c );
            if ( c != nullptr ) do {
                if ( ! rt.is_infinite( c ) ) adj.push_back( c );
            } while ( ++c != done );
            vs += double( adj.size() );
        }
        const double t2 = now();

        // THE CONNECTIVITY REPLAY. In two steps, and that is the point : the final LIST of
        // neighbours is built first, OUTSIDE the timer (this is the work that our accelerator
        // replaces, not the one we want to measure) ; then only the cut and the
        // measure are timed, on this list and on it alone. What this gives is therefore the geometric
        // floor : the time of ONE cell if one knew in advance, without ever being wrong,
        // which seeds cut it.
        double sv = 0;
        if ( cells ) {
            std::vector<double> own, nbrs;                  // `own` : x,y,w of the seed. `nbrs` : same
            std::vector<std::size_t> off( 1, 0 );           // split of `nbrs` per seed
            own.reserve( 3 * vertex_count( rt ) );
            for ( auto v = rt.finite_vertices_begin(); v != rt.finite_vertices_end(); ++v ) {
                const auto &p0 = v->point();
                own.push_back( p0.x() ); own.push_back( p0.y() ); own.push_back( p0.weight() );
                Rt::Vertex_circulator c = rt.incident_vertices( v ), done( c );
                if ( c != nullptr ) do {
                    if ( rt.is_infinite( c ) ) continue;
                    const auto &p1 = c->point();
                    nbrs.push_back( p1.x() ); nbrs.push_back( p1.y() ); nbrs.push_back( p1.weight() );
                } while ( ++c != done );
                off.push_back( nbrs.size() / 3 );
            }

            const double u0 = now();
            for ( std::size_t i = 0; i + 1 < off.size(); ++i ) {
                const double x0 = own[ 3 * i ], y0 = own[ 3 * i + 1 ], w0 = own[ 3 * i + 2 ];
                pd::CellSoAT<64> cl;
                cl.init_as_unit_square();
                pd::SI id = 0;
                for ( std::size_t j = off[ i ]; j < off[ i + 1 ]; ++j ) {
                    const double x1 = nbrs[ 3 * j ], y1 = nbrs[ 3 * j + 1 ], w1 = nbrs[ 3 * j + 2 ];
                    const double dx = x1 - x0, dy = y1 - y0;
                    const double o = dx * ( x0 + x1 ) / 2 + dy * ( y0 + y1 ) / 2 + ( w0 - w1 ) / 2;
                    cl.cut( dx, dy, o, id++ );
                }
                // THE DECOYS : `--decoy K` adds K planes TAKEN FROM REAL SEEDS, but far enough
                // away that none of them cuts. The slope in K is the price of ONE USELESS CUT --
                // the kind our accelerator attempts and that yields nothing.
                for ( int d = 0; d < decoy; ++d ) {
                    const std::size_t j = ( i + 50 + std::size_t( d ) * 7 ) % ( off.size() - 1 );
                    const double x1 = own[ 3 * j ], y1 = own[ 3 * j + 1 ], w1 = own[ 3 * j + 2 ];
                    const double dx = x1 - x0, dy = y1 - y0;
                    const double o = dx * ( x0 + x1 ) / 2 + dy * ( y0 + y1 ) / 2 + ( w0 - w1 ) / 2;
                    cl.cut( dx, dy, o, id++ );
                }
                sv += cl.measure();
            }
            tc = std::min( tc, now() - u0 );
            sum = sv;

            // THE DUMP, only once and OUTSIDE the timer (see the 3D version).
            if ( ! dump.empty() && r == 0 ) {
                std::vector<double> ref( off.size() - 1 );
                for ( std::size_t i = 0; i + 1 < off.size(); ++i ) {
                    const double x0 = own[ 3 * i ], y0 = own[ 3 * i + 1 ], w0 = own[ 3 * i + 2 ];
                    pd::CellSoAT<64> cl;
                    cl.init_as_unit_square();
                    pd::SI id = 0;
                    for ( std::size_t j = off[ i ]; j < off[ i + 1 ]; ++j ) {
                        const double x1 = nbrs[ 3 * j ], y1 = nbrs[ 3 * j + 1 ], w1 = nbrs[ 3 * j + 2 ];
                        const double dx = x1 - x0, dy = y1 - y0;
                        const double o = dx * ( x0 + x1 ) / 2 + dy * ( y0 + y1 ) / 2 + ( w0 - w1 ) / 2;
                        cl.cut( dx, dy, o, id++ );
                    }
                    ref[ i ] = cl.measure();
                }
                dump_csr( dump, 2, own, off, nbrs, ref );
            }
        }
        tt = std::min( tt, t1 - t0 );
        ta = std::min( ta, t2 - t1 );
        vertices = rt.number_of_vertices();
        hidden = std::size_t( n ) - vertices;
        neighbors = vs / double( vertices );
    }

    std::printf( "%-46s n=%d   kernel %s\n", load.empty() ? "uniforme" : load.c_str(), n, kernel_name );
    std::printf( "   triangulation  %7.3f s (%7.0f ns/seed)\n", tt, 1e9 * tt / n );
    std::printf( "   adjacency      %7.3f s (%7.0f ns/seed)\n", ta, 1e9 * ta / n );
    if ( cells ) {
        std::printf( "   cut + measure  %7.3f s (%7.0f ns/seed)   decoys %d\n", tc, 1e9 * tc / n, decoy );
        std::printf( "   FULL CHAIN     %7.3f s (%7.0f ns/seed)   neighbours %.2f   hidden %zu   sum %.9f\n",
                     tt + ta + tc, 1e9 * ( tt + ta + tc ) / n, neighbors, hidden, sum );
    } else
        std::printf( "   LOWER BOUND    %7.3f s (%7.0f ns/seed)   neighbours %.2f   hidden %zu\n",
                     tt + ta, 1e9 * ( tt + ta ) / n, neighbors, hidden );
    return 0;
}
