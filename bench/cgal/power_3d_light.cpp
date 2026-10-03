// WHAT A LIGHTENED VERSION OF CGAL'S ALGORITHM WOULD COST.
//
// `power_3d.cpp` measures CGAL as it is normally used. This one breaks the figure down into three
// items that can be switched on separately, to know what, in CGAL's nanoseconds, is due to the
// ALGORITHM (incremental insertion in a regular triangulation, which would have to be written
// anyway) and what is due to the GENERICITY of the library (filtered predicates, generic
// adjacency queries) -- that is, the ceiling of what a home-made rewrite could gain.
//
//   1. THE KERNEL.  `-DLIGHT_KERNEL` replaces `Exact_predicates_inexact_constructions_kernel` by
//      `Simple_cartesian<double>`: no more interval filter, no more exact fallback, the
//      predicates become FP64 determinants -- exactly what our cut already does.
//      The difference between the two binaries IS the price of robustness.
//
//      Beware: without a filter, a degeneracy can make the triangulation loop or break.
//      The uniform cloud has none; the check is the number of vertices and the sum of the volumes.
//
//   2. THE ADJACENCY.  `--adj fast` replaces `finite_adjacent_vertices` -- which goes through a
//      temporary associative container at each vertex -- by a walk over the incident cells with
//      O(1) deduplication on a STAMP stored in the vertex (`with_info`). This is what we would
//      write ourselves, and it changes nothing about the algorithm: the same structure is read.
//
//   3. THE CHAIN.  `--cells` chains OUR cut and OUR measure on the returned neighbors, as
//      in `power_3d.cpp`, so that the total is comparable with the bench.

#ifdef LIGHT_KERNEL
#   include <CGAL/Simple_cartesian.h>
using K = CGAL::Simple_cartesian<double>;
static constexpr const char *kernel_name = "Simple_cartesian<double> (bare FP64 predicates)";
#else
#   include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
using K = CGAL::Exact_predicates_inexact_constructions_kernel;
static constexpr const char *kernel_name = "Epick (exact filtered predicates)";
#endif

#include <CGAL/Regular_triangulation_3.h>
#include <CGAL/Regular_triangulation_vertex_base_3.h>
#include <CGAL/Regular_triangulation_cell_base_3.h>
#include <CGAL/Triangulation_vertex_base_with_info_3.h>

#include "geometry/Cell3.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <string>
#include <vector>
#include "dump_csr.h"

using Vbb = CGAL::Regular_triangulation_vertex_base_3<K>;
using Vb  = CGAL::Triangulation_vertex_base_with_info_3<int, K, Vbb>;
using Cb  = CGAL::Regular_triangulation_cell_base_3<K>;
using Tds = CGAL::Triangulation_data_structure_3<Vb, Cb>;
using Rt   = CGAL::Regular_triangulation_3<K, Tds>;
using Wp   = Rt::Weighted_point;
using Bare = Rt::Bare_point;
using Vh   = Rt::Vertex_handle;

namespace {

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
        double x, y, z, w;
        if ( ! num( x ) || ! num( y ) || ! num( z ) || ! num( w ) ) return false;
        pts.emplace_back( Bare( x, y, z ), w );
    }
    return true;
}

} // namespace

int main( int argc, char **argv ) {
    int n = 200000, reps = 3;
    int decoy = 0;
    bool cells = false, fast = false;
    std::string load, dump;
    for ( int i = 1; i < argc; ++i ) {
        const std::string s = argv[ i ];
        if ( s == "--load" && i + 1 < argc ) load = argv[ ++i ];
        else if ( s == "--dump" && i + 1 < argc ) dump = argv[ ++i ];
        else if ( s == "--reps" && i + 1 < argc ) reps = std::atoi( argv[ ++i ] );
        else if ( s == "--cells" ) cells = true;
        else if ( s == "--decoy" && i + 1 < argc ) decoy = std::atoi( argv[ ++i ] );
        else if ( s == "--adj" && i + 1 < argc ) fast = std::string( argv[ ++i ] ) == "fast";
        else if ( s == "--help" ) {
            std::printf( "usage: power_3d_light [n] [--load F] [--reps R] [--cells] [--adj fast]\n" );
            return 0;
        } else n = std::atoi( argv[ i ] );
    }

    std::vector<Wp> pts;
    if ( ! load.empty() ) {
        if ( ! load_cloud( load, pts ) ) return 1;
        n = int( pts.size() );
    } else {
        std::mt19937_64 gen( 0 );
        std::uniform_real_distribution<double> u( 0.0, 1.0 );
        pts.reserve( n );
        for ( int i = 0; i < n; ++i ) pts.emplace_back( Bare( u( gen ), u( gen ), u( gen ) ), 0.0 );
    }

    double tt = 1e30, ta = 1e30, tc = 1e30;
    double neighbors = 0, sum = 0;
    std::size_t vertices = 0, hidden = 0;

    for ( int r = 0; r < reps; ++r ) {
        const double t0 = now();
        Rt rt( pts.begin(), pts.end() );
        const double t1 = now();

        // THE STAMP: `info` carries the number of the vertex whose neighborhood is being walked. Two
        // incident cells share an edge, hence propose the same neighbor -- this duplicate is what CGAL's
        // associative container eliminates, and what an integer eliminates too.
        int stamp = 0;
        for ( auto v = rt.finite_vertices_begin(); v != rt.finite_vertices_end(); ++v )
            v->info() = -1;

        double vs = 0;
        std::vector<Vh> adj;
        std::vector<Rt::Cell_handle> inc;
        for ( auto v = rt.finite_vertices_begin(); v != rt.finite_vertices_end(); ++v ) {
            adj.clear();
            if ( fast ) {
                inc.clear();
                rt.incident_cells( v, std::back_inserter( inc ) );
                ++stamp;
                for ( auto c : inc )
                    for ( int i = 0; i < 4; ++i ) {
                        const Vh u = c->vertex( i );
                        if ( u == Vh( v ) || rt.is_infinite( u ) || u->info() == stamp ) continue;
                        u->info() = stamp;
                        adj.push_back( u );
                    }
            } else
                rt.finite_adjacent_vertices( v, std::back_inserter( adj ) );
            vs += double( adj.size() );
        }
        const double t2 = now();

        // THE CONNECTIVITY REPLAY. In two steps, and that is the point: the final LIST of
        // neighbors is built first, OUTSIDE the timer (it is the work that our accelerator
        // replaces, not the one we want to measure); then only the cut and the
        // measure are timed, on that list and on it alone. What this gives is thus the geometric
        // floor: the time of ONE cell if we knew in advance, without ever being wrong,
        // which seeds cut it.
        double sv = 0;
        if ( cells ) {
            std::vector<double> own, nbrs;                  // x,y,z,w of the seed / of each neighbor
            std::vector<std::size_t> off( 1, 0 );
            stamp = 0;
            for ( auto v = rt.finite_vertices_begin(); v != rt.finite_vertices_end(); ++v )
                v->info() = -1;
            for ( auto v = rt.finite_vertices_begin(); v != rt.finite_vertices_end(); ++v ) {
                adj.clear();
                if ( fast ) {
                    inc.clear();
                    rt.incident_cells( v, std::back_inserter( inc ) );
                    ++stamp;
                    for ( auto c : inc )
                        for ( int i = 0; i < 4; ++i ) {
                            const Vh u = c->vertex( i );
                            if ( u == Vh( v ) || rt.is_infinite( u ) || u->info() == stamp ) continue;
                            u->info() = stamp;
                            adj.push_back( u );
                        }
                } else
                    rt.finite_adjacent_vertices( v, std::back_inserter( adj ) );

                const auto &wp0 = v->point();
                own.push_back( wp0.x() ); own.push_back( wp0.y() );
                own.push_back( wp0.z() ); own.push_back( wp0.weight() );
                for ( auto u : adj ) {
                    const auto &wp1 = u->point();
                    nbrs.push_back( wp1.x() ); nbrs.push_back( wp1.y() );
                    nbrs.push_back( wp1.z() ); nbrs.push_back( wp1.weight() );
                }
                off.push_back( nbrs.size() / 4 );
            }

            const double u0 = now();
            for ( std::size_t i = 0; i + 1 < off.size(); ++i ) {
                const double x0 = own[ 4 * i ], y0 = own[ 4 * i + 1 ],
                             z0 = own[ 4 * i + 2 ], w0 = own[ 4 * i + 3 ];
                pd::Cell3T<128> c;
                c.init_as_unit_cube();
                pd::SI id = 0;
                for ( std::size_t j = off[ i ]; j < off[ i + 1 ]; ++j ) {
                    const double x1 = nbrs[ 4 * j ], y1 = nbrs[ 4 * j + 1 ],
                                 z1 = nbrs[ 4 * j + 2 ], w1 = nbrs[ 4 * j + 3 ];
                    const double dx = x1 - x0, dy = y1 - y0, dz = z1 - z0;
                    const double o = ( w0 - w1 ) / 2 + dx * ( x0 + x1 ) / 2
                                   + dy * ( y0 + y1 ) / 2 + dz * ( z0 + z1 ) / 2;
                    c.cut( pd::Vec<3>{ dx, dy, dz }, o, id++ );
                }
                // THE DECOYS: `--decoy K` adds K planes TAKEN FROM REAL SEEDS, but far enough
                // away that none cuts. The slope in K is the price of ONE USELESS CUT --
                // the one that our accelerator tries and that brings nothing.
                for ( int d = 0; d < decoy; ++d ) {
                    const std::size_t j = ( i + 50 + std::size_t( d ) * 7 ) % ( off.size() - 1 );
                    const double x1 = own[ 4 * j ], y1 = own[ 4 * j + 1 ],
                                 z1 = own[ 4 * j + 2 ], w1 = own[ 4 * j + 3 ];
                    const double dx = x1 - x0, dy = y1 - y0, dz = z1 - z0;
                    const double o = ( w0 - w1 ) / 2 + dx * ( x0 + x1 ) / 2
                                   + dy * ( y0 + y1 ) / 2 + dz * ( z0 + z1 ) / 2;
                    c.cut( pd::Vec<3>{ dx, dy, dz }, o, id++ );
                }
                sv += c.measure();
            }
            tc = std::min( tc, now() - u0 );
            sum = sv;

            // THE DUMP, once only and OUTSIDE the timer: the timed loop above must
            // not carry one more `if`, and the reference volume is recomputed here at
            // no measured cost.
            if ( ! dump.empty() && r == 0 ) {
                std::vector<double> ref( off.size() - 1 );
                for ( std::size_t i = 0; i + 1 < off.size(); ++i ) {
                    const double x0 = own[ 4 * i ], y0 = own[ 4 * i + 1 ],
                                 z0 = own[ 4 * i + 2 ], w0 = own[ 4 * i + 3 ];
                    pd::Cell3T<128> c;
                    c.init_as_unit_cube();
                    pd::SI id = 0;
                    for ( std::size_t j = off[ i ]; j < off[ i + 1 ]; ++j ) {
                        const double x1 = nbrs[ 4 * j ], y1 = nbrs[ 4 * j + 1 ],
                                     z1 = nbrs[ 4 * j + 2 ], w1 = nbrs[ 4 * j + 3 ];
                        const double dx = x1 - x0, dy = y1 - y0, dz = z1 - z0;
                        const double o = ( w0 - w1 ) / 2 + dx * ( x0 + x1 ) / 2
                                       + dy * ( y0 + y1 ) / 2 + dz * ( z0 + z1 ) / 2;
                        c.cut( pd::Vec<3>{ dx, dy, dz }, o, id++ );
                    }
                    ref[ i ] = c.measure();
                }
                dump_csr( dump, 3, own, off, nbrs, ref );
            }
        }
        tt = std::min( tt, t1 - t0 );
        ta = std::min( ta, t2 - t1 );
        vertices = rt.number_of_vertices();
        hidden = std::size_t( n ) - vertices;
        neighbors = vs / double( vertices );
    }

    std::printf( "%-46s n=%d   kernel %s   adjacency %s\n",
                 load.empty() ? "uniform" : load.c_str(), n, kernel_name, fast ? "HOME-MADE" : "CGAL" );
    std::printf( "   triangulation  %7.3f s (%7.0f ns/seed)\n", tt, 1e9 * tt / n );
    std::printf( "   adjacency      %7.3f s (%7.0f ns/seed)\n", ta, 1e9 * ta / n );
    if ( cells ) {
        std::printf( "   cut + measure  %7.3f s (%7.0f ns/seed)   decoys %d\n", tc, 1e9 * tc / n, decoy );
        std::printf( "   WHOLE CHAIN    %7.3f s (%7.0f ns/seed)   neighbors %.2f   hidden %zu   sum %.9f\n",
                     tt + ta + tc, 1e9 * ( tt + ta + tc ) / n, neighbors, hidden, sum );
    } else
        std::printf( "   LOWER BOUND    %7.3f s (%7.0f ns/seed)   neighbors %.2f   hidden %zu\n",
                     tt + ta, 1e9 * ( tt + ta ) / n, neighbors, hidden );
    return 0;
}
