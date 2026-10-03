// THE 3D BENCHMARK, counterpart of `power_2d.cpp`: what it costs CGAL to build the REGULAR
// TRIANGULATION of a weighted 3D cloud, and to extract the adjacency.
//
// = What it measures, and above all what it does NOT measure
//
// It times two things:
//
//   1. `Regular_triangulation_3` on `n` weighted points -- range insertion, hence with the
//      spatial sort that CGAL does on its own;
//   2. the tour of each vertex (`finite_adjacent_vertices`), which returns the LIST OF NEIGHBORS.
//
// It builds NO cell: neither the dual vertices, nor the faces, nor the clip on the cube, nor
// the volumes. The figure it yields is therefore a LOWER BOUND on the price CGAL would pay to
// do the same work as us. That is what we want to know first: if this bound already exceeds
// our full time, the question is settled without writing the extraction.
//
// The sanity check is the average number of neighbors: a Poissonian 3D Laguerre cell
// has about 15.5 of them, and that is what our bench counts on its side (15.2 on this cloud).
//
// = The kernel
//
// `Exact_predicates_inexact_constructions_kernel`, as in 2D: exact predicates -- which makes the
// triangulation robust, and that is precisely what our cut does not need to be -- and
// constructions in `double`, to compare with our FP64.

#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Regular_triangulation_3.h>

// OUR headers, taken as is in the bench. They have no dependency -- that is the whole point of
// `2d_des_familles` -- so including them here brings in neither CGAL nor gmp: the dependency goes
// in that direction only. `--cells` then measures the COMPLETE chain: CGAL triangulation, adjacency,
// then OUR cut and OUR measure, with the 15.5 true neighbors instead of the 87.8 proposed.
#include "geometry/Cell3.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <string>
#include <vector>

using K    = CGAL::Exact_predicates_inexact_constructions_kernel;
using Rt   = CGAL::Regular_triangulation_3<K>;
using Wp   = Rt::Weighted_point;
using Bare = Rt::Bare_point;

namespace {

double now() {
    using namespace std::chrono;
    return duration<double>( steady_clock::now().time_since_epoch() ).count();
}

/// A cloud from `2d_des_familles/cases/` in 3D: `#` lines, then `n`, then `n` times "x y z w".
/// Same weight convention as ours -- CGAL minimizes `|x - p|^2 - w` -- so nothing to convert.
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
    bool cells = false, tri = false;
    std::string load;
    for ( int i = 1; i < argc; ++i ) {
        const std::string s = argv[ i ];
        if ( s == "--load" && i + 1 < argc ) load = argv[ ++i ];
        else if ( s == "--reps" && i + 1 < argc ) reps = std::atoi( argv[ ++i ] );
        else if ( s == "--cells" ) cells = true;
        else if ( s == "--sort" ) { cells = true; tri = true; }
        else if ( s == "--help" ) {
            std::printf( "usage: power_3d [n] [--load FILE] [--reps R] [--cells]\n"
                         "  --cells  also builds the cells with OUR kernel, and sums\n" );
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
    std::size_t hidden = 0, vertices = 0;
    for ( int r = 0; r < reps; ++r ) {
        const double t0 = now();
        Rt rt( pts.begin(), pts.end() );
        const double t1 = now();

        double vs = 0;
        std::vector<Rt::Vertex_handle> adj;
        for ( auto v = rt.finite_vertices_begin(); v != rt.finite_vertices_end(); ++v ) {
            adj.clear();
            rt.finite_adjacent_vertices( v, std::back_inserter( adj ) );
            vs += double( adj.size() );
        }
        const double t2 = now();

        double sv = 0;
        if ( cells ) {
            std::vector<Rt::Vertex_handle> ad;
            for ( auto v = rt.finite_vertices_begin(); v != rt.finite_vertices_end(); ++v ) {
                ad.clear();
                rt.finite_adjacent_vertices( v, std::back_inserter( ad ) );
                const auto &wp0 = v->point();
                const double x0 = wp0.x(), y0 = wp0.y(), z0 = wp0.z(), w0 = wp0.weight();
                pd::Cell3T<128> c;
                c.init_as_unit_cube();
                pd::SI id = 0;
                if ( tri ) {
                    // FROM NEAREST TO FARTHEST, in the power sense. Here the 15.5 planes are
                    // ALL faces of the final cell: the order therefore does not change the result,
                    // only the size of the INTERMEDIATE polyhedra -- and the cut sweeps all
                    // the vertices every time.
                    std::sort( ad.begin(), ad.end(), [ & ]( Rt::Vertex_handle a, Rt::Vertex_handle b ) {
                        const auto &pa = a->point(); const auto &pb = b->point();
                        const double da = ( pa.x() - x0 ) * ( pa.x() - x0 ) + ( pa.y() - y0 ) * ( pa.y() - y0 )
                                        + ( pa.z() - z0 ) * ( pa.z() - z0 ) - pa.weight();
                        const double db = ( pb.x() - x0 ) * ( pb.x() - x0 ) + ( pb.y() - y0 ) * ( pb.y() - y0 )
                                        + ( pb.z() - z0 ) * ( pb.z() - z0 ) - pb.weight();
                        return da < db;
                    } );
                }
                for ( auto u : ad ) {
                    const auto &wp1 = u->point();
                    const double dx = wp1.x() - x0, dy = wp1.y() - y0, dz = wp1.z() - z0;
                    const double off = ( w0 - wp1.weight() ) / 2
                                     + dx * ( x0 + wp1.x() ) / 2
                                     + dy * ( y0 + wp1.y() ) / 2
                                     + dz * ( z0 + wp1.z() ) / 2;
                    c.cut( pd::Vec<3>{ dx, dy, dz }, off, id++ );
                }
                sv += c.measure();
            }
        }
        const double t3 = now();

        tt = std::min( tt, t1 - t0 );
        ta = std::min( ta, t2 - t1 );
        if ( cells ) { tc = std::min( tc, t3 - t2 ); sum = sv; }
        vertices = rt.number_of_vertices();
        hidden = std::size_t( n ) - vertices;
        neighbors = vs / double( vertices );
    }

    std::printf( "%-46s n=%d\n", load.empty() ? "uniform" : load.c_str(), n );
    std::printf( "   triangulation  %7.3f s (%7.0f ns/seed)\n", tt, 1e9 * tt / n );
    std::printf( "   adjacency      %7.3f s (%7.0f ns/seed)\n", ta, 1e9 * ta / n );
    if ( cells ) {
        std::printf( "   cut + measure  %7.3f s (%7.0f ns/seed)\n", tc, 1e9 * tc / n );
        std::printf( "   WHOLE CHAIN    %7.3f s (%7.0f ns/seed)   neighbors %.2f   hidden %zu"
                     "   sum %.9f\n",
                     tt + ta + tc, 1e9 * ( tt + ta + tc ) / n, neighbors, hidden, sum );
    } else
        std::printf( "   LOWER BOUND    %7.3f s (%7.0f ns/seed)   neighbors %.2f   hidden %zu\n",
                     tt + ta, 1e9 * ( tt + ta ) / n, neighbors, hidden );
    return 0;
}
