// The SAME computation as `test_PowerDiagram::pd accelerated`, but with CGAL : a 2D power
// diagram on `n` seeds of the unit square, the cells clipped to this square, and their AREAS.
//
// = Why a separate program, and not a test
//
// It is a BENCHMARK REFERENCE, not a verification : what is asked of it is a time, under the same
// conditions as ours (same machine, same n, same FP64), and the sum of the areas as the only
// check -- it must equal 1 up to rounding, otherwise the figure measures nothing.
//
// = What CGAL does, and what it does not do
//
// `Regular_triangulation_2` is the regular triangulation (the dual of the power diagram) :
// it returns the ADJACENCY, not the cells. The cell of a seed is rebuilt by turning around
// it (`incident_faces`) and joining the POWER CENTERS of the incident faces -- this is the
// dual polygon. Two things come out of it that do not exist on our side :
//
//   * a "hidden" seed (`hidden`) has no cell at all : this is the case a low enough weight
//     produces, and CGAL removes it from the triangulation instead of leaving it an empty cell.
//   * a BORDER cell is infinite. CGAL does not know our domain, so the clip to the unit
//     square is done here, by hand (Sutherland-Hodgman), afterwards.
//
// The clip is therefore ON OUR SIDE in this comparison, whereas it is in the kernel for ours.
// This is the only place where the two do not do exactly the same work, and it plays in favor
// of CGAL (our box is cut BEFORE the bisectors, its own after) -- to keep in mind when
// reading the gap.
//
// = The kernel
//
// `Exact_predicates_inexact_constructions_kernel` : exact predicates (this is what makes the
// triangulation robust), constructions in `double`. It is the normal choice for a measurement, and
// the one that compares to our FP64. A kernel with exact constructions would be another benchmark.

#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Regular_triangulation_2.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

// The KERNEL serves directly as traits : `CGAL::Regular_triangulation_traits_2`, the adapter
// of old, no longer exists in CGAL 6 -- a kernel now carries its own `Weighted_point_2`.
using K    = CGAL::Exact_predicates_inexact_constructions_kernel;
using Rt   = CGAL::Regular_triangulation_2<K>;
using Wp   = Rt::Weighted_point;
using Bare = Rt::Bare_point;

namespace {

struct P2 { double x, y; };

// Sutherland-Hodgman against ONE half-plane `a.x + b.y <= c`, in place in `poly`.
void clip_half( std::vector<P2> &poly, double a, double b, double c, std::vector<P2> &tmp ) {
    tmp.clear();
    const std::size_t n = poly.size();
    for ( std::size_t i = 0; i < n; ++i ) {
        const P2 &p = poly[ i ];
        const P2 &q = poly[ ( i + 1 ) % n ];
        const double sp = a * p.x + b * p.y - c;
        const double sq = a * q.x + b * q.y - c;
        if ( sp <= 0 )
            tmp.push_back( p );
        if ( ( sp < 0 && sq > 0 ) || ( sp > 0 && sq < 0 ) ) {
            const double t = sp / ( sp - sq );
            tmp.push_back( { p.x + t * ( q.x - p.x ), p.y + t * ( q.y - p.y ) } );
        }
    }
    poly.swap( tmp );
}

double area( const std::vector<P2> &poly ) {
    double a = 0;
    const std::size_t n = poly.size();
    for ( std::size_t i = 0; i < n; ++i ) {
        const P2 &p = poly[ i ];
        const P2 &q = poly[ ( i + 1 ) % n ];
        a += p.x * q.y - q.x * p.y;
    }
    return 0.5 * ( a < 0 ? -a : a );
}

/// A cloud from `2d_des_familles/cases/` : `#` lines, then `n`, then `n` times "x y w".
///
/// The weight convention is the SAME on both sides -- CGAL minimizes `|x - p|^2 - w` like us --
/// so the files are read as is, without conversion, and both really compute the same
/// diagram. This is what makes it possible to compare times on HARD clouds, where the uniform one stops
/// being representative.
bool load_cloud( const std::string &path, std::vector<Wp> &pts ) {
    std::FILE *f = std::fopen( path.c_str(), "rb" );
    if ( ! f ) {
        std::printf( "cannot open '%s'\n", path.c_str() );
        return false;
    }
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
        if ( ! num( x ) || ! num( y ) || ! num( w ) )
            return false;
        pts.emplace_back( Bare( x, y ), w );
    }
    return true;
}

} // namespace

int main( int argc, char **argv ) {
    int n = 1000000, reps = 3;
    std::string load;
    for ( int i = 1; i < argc; ++i ) {
        const std::string s = argv[ i ];
        if ( s == "--load" && i + 1 < argc )       load = argv[ ++i ];
        else if ( s == "--reps" && i + 1 < argc )  reps = std::atoi( argv[ ++i ] );
        else                                       n = std::atoi( s.c_str() );
    }

    std::vector<Wp> pts;
    if ( ! load.empty() ) {
        if ( ! load_cloud( load, pts ) )
            return 1;
        n = int( pts.size() );
        std::printf( "  cloud '%s' : %d seeds\n", load.c_str(), n );
    } else {
        // the SAME cloud as the python benchmark, in spirit : uniform in [ 0.01, 0.99 ]^2. Not the
        // same draws (two different generators), which does not matter for a timing.
        std::mt19937_64 rng( 0 );
        std::uniform_real_distribution<double> uni( 0.01, 0.99 );
        pts.reserve( n );
        for ( int i = 0; i < n; ++i )
            pts.emplace_back( Bare( uni( rng ), uni( rng ) ), 0.0 );
    }

    double best = 1e300, last_sum = 0;
    for ( int r = 0; r < reps; ++r ) {
        const auto t0 = std::chrono::steady_clock::now();

        // BULK insertion : CGAL then sorts the points itself (Hilbert) and inserts with
        // spatial localization, which is by far the fastest path -- inserting one by
        // one would mostly measure the cost of localization.
        Rt rt;
        rt.insert( pts.begin(), pts.end() );

        // the cells, one per finite vertex : the dual polygon, then the clip to the unit square.
        double sum = 0;
        std::vector<P2> poly, tmp;
        for ( auto v = rt.finite_vertices_begin(); v != rt.finite_vertices_end(); ++v ) {
            poly.clear();
            poly.push_back( { 0, 0 } ); poly.push_back( { 1, 0 } );
            poly.push_back( { 1, 1 } ); poly.push_back( { 0, 1 } );

            // cut by the power bisector with each neighbour. Going through the NEIGHBOURS
            // rather than the power centers of the faces avoids having to handle the infinite
            // faces separately : a half-plane is a half-plane, bounded or not.
            const auto &p0 = v->point();
            Rt::Vertex_circulator c = rt.incident_vertices( v ), done( c );
            if ( c != nullptr ) {
                do {
                    if ( rt.is_infinite( c ) )
                        continue;
                    const auto &p1 = c->point();
                    const double dx = p1.x() - p0.x(), dy = p1.y() - p0.y();
                    const double off = dx * ( p0.x() + p1.x() ) / 2 + dy * ( p0.y() + p1.y() ) / 2
                                     + ( p0.weight() - p1.weight() ) / 2;
                    clip_half( poly, dx, dy, off, tmp );
                    if ( poly.size() < 3 )
                        break;
                } while ( ++c != done );
            }
            sum += poly.size() >= 3 ? area( poly ) : 0.0;
        }

        const double dt = std::chrono::duration<double>( std::chrono::steady_clock::now() - t0 ).count();
        best = std::min( best, dt );
        last_sum = sum;
    }

    std::printf( "  %d seeds in 2D : %.3f s (%.0f ns/seed), sum of measures %.6f\n",
                 n, best, best / n * 1e9, last_sum );
    return 0;
}
