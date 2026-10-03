#pragma once

// =====================================================================================
// THE SOLVER'S DIAGRAMS : what Newton asks of a power diagram, and nothing else.
//
//     set_weights( w )              set the weights ( user order ), rebuild the majorants
//     measures( a, &fa )             ONE sweep : the measure of each cell AND its facets
//                                   `c_ij = int_{facet} rho / ( 2 |p_i - p_j| )` ( `Laplacian.h` )
//
// Everything runs here on the CPU queue ( `CpuQueue::run_threads` : contiguous slices of cells,
// in storage order, where two consecutive seeds are neighbours in space ). The scratch
// of the cells is MANAGED HERE, not by loom : a solver chains a hundred diagrams in a single
// call, and an overflow must only restart the sweep in progress, not the whole call. One
// row of words per thread, doubled as long as a cell does not fit.
//
// The cell of a seed is built as everywhere ( `diagram::make_cell` : the domain, then the
// planes that the storage provider proposes ), its mass integrated as everywhere
// ( `diagram::integrate_into` ), and its facets read as `diagram::hessian_row` reads them -- this
// file adds no geometry, it chains.
//
// A NON-CONSTANT DENSITY on a facet : the distribution must know how to integrate its density on
// a facet ( `facet_mass( piece, cut )` -- 2D gaussians do it in closed form ) ; failing
// that the laplacian cannot be assembled and the compilation says so.
// =====================================================================================

#include <loom/support/kernels/CpuQueue.h>
#include "../diagram/Ops.h"
#include "../bsp_build_level.h"
#include "Laplacian.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <vector>

namespace sdot {
namespace sdotplan {

inline double now() {
    using namespace std::chrono;
    return duration<double>( steady_clock::now().time_since_epoch() ).count();
}

/// the mass of a facet for the density of a piece : `rho * measure` for a constant density, what
/// the distribution can say about it otherwise
template<class Dens,class Pc>
double density_facet_mass( const Dens &dens, const Pc &pc, int cut, double mes ) {
    if constexpr ( Dens::is_constant )
        return double( dens.value ) * mes;
    else if constexpr ( requires { dens.facet_mass( pc, cut ); } )
        return double( dens.facet_mass( pc, cut ) );
    else
        static_assert( Dens::is_constant, "sdotplan : this distribution does not know how to integrate its density on a facet ( `facet_mass` )" );
    return 0;
}

/// ONE CELL of the seed of rank `k` : its mass in `mass`, and `facet( j, c_kj )` for each neighbour
/// `j` ( in ranks ). Returns `false` if the scratch was not enough.
template<class PD,class Local>
bool measure_and_facets( const PD &pd, SI k, Local &c, Local &piece, const auto &dom, const auto &dist,
                         double &mass, auto &&facet, bool with_facets ) {
    using TF = typename PD::TF;
    constexpr int D = PD::ct_dim;
    if ( ! diagram::make_cell( pd, c, k, dom ) )
        return false;
    TF m = 0;
    if ( ! diagram::integrate_into<TF>( m, c, piece, dist ) )
        return false;
    mass = double( m );
    if ( ! with_facets )
        return true;

    // the neighbours of the cell are its live cuts : one slot per cut, accumulated piece
    // by piece ( a piece carries the same identifiers, plus those of its box )
    c.tidy();
    const int nc = c.nb_cuts();
    double vals[ 512 ];
    if ( nc > 512 )
        return false;
    for ( int q = 0; q < nc; ++q )
        vals[ q ] = 0;
    PieceWorkspace<Local> ws{ piece };
    dist.for_each_piece( c, ws, [&]( const auto &pc, const auto &dens ) {
        pc.template for_each_facet<TF>( [&]( int cut, TF mes ) {
            const int id = pc.cid[ cut ];
            if ( id < 0 )
                return;                                  // the domain, or a box edge : immobile
            const double val = density_facet_mass( dens, pc, cut, double( mes ) );
            for ( int q = 0; q < nc; ++q )
                if ( c.cid[ q ] == id ) {
                    vals[ q ] += val;
                    break;
                }
        } );
    } );
    if ( ws.overflow )
        return false;

    const auto p0 = pd.point( k );
    for ( int q = 0; q < nc; ++q ) {
        const int id = c.cid[ q ];
        if ( id < 0 )
            continue;
        const auto pj = pd.point( id );
        double d2 = 0;
        for ( int d = 0; d < D; ++d )
            d2 += double( pj[ d ] - p0[ d ] ) * double( pj[ d ] - p0[ d ] );
        if ( d2 > 0 )
            facet( id, vals[ q ] / ( 2 * std::sqrt( d2 ) ) );
    }
    return true;
}

/// THE DIAGRAM AS SEEN BY NEWTON. `pd` is the storage ( `PowerDiagram_Bsp` / `_Plain` ) whose
/// weights -- and tree majorants -- are WRITABLE views ( `with_weights` ) : this is
/// where they are written.
template<class PD,class Dom,class Dist,class TK>
struct Sweep {
    using TF = typename PD::TF;
    using Local = typename Dom::template Local<TK>;
    static constexpr int D = PD::ct_dim;
    static constexpr int nbc = diagram::nb_work_cells<Dist>();

    const CpuQueue &queue;
    PD             &pd;                                  ///< the weights are written there ( output views )
    const Dom      &dom;
    const Dist     *dist;                                ///< the current density ( it changes from one step to the next )
    int             nt;                                  ///< virtual threads ( contiguous slices )
    SI              cap;                                 ///< vertices per local cell
    SI              words = 0;
    std::vector<std::vector<std::int32_t>> scratch;      ///< one row per thread
    std::vector<std::vector<Facet>>      fa_th;        ///< the facets, per thread
    std::vector<SI> rank_of;                             ///< the rank of seed `i` ( user order )

    SI     nb_overflowed = 0;                               ///< scratch doublings, in total
    int    nb_diag    = 0;
    double t_majorant = 0, t_diag = 0;

    /// `pd_in` : the storage as it comes in ( its memories, `memo_*`, are copied into those of `pd` )
    Sweep( const CpuQueue &queue, PD &pd, const auto &pd_in, const Dom &dom, const Dist &dist, SI cap0 )
        : queue( queue ), pd( pd ), dom( dom ), dist( &dist ), nt( std::max( queue.nb_workers(), 1 ) ), cap( std::max<SI>( cap0, 8 ) ) {
        const SI n = pd.nb_seeds();
        scratch.resize( nt );
        fa_th.resize( nt );
        rank_of.resize( n );
        for ( SI k = 0; k < n; ++k )
            rank_of[ pd.user_id( k ) ] = k;
        resize_scratch();
        if constexpr ( requires { PD::has_memo; } ) {
            if constexpr ( PD::has_memo ) {              // the memory : the previous memories, or nothing
                const SI K = SI( pd.memo_nbrs.shape( 1 ) );
                for ( SI k = 0; k < n; ++k ) {
                    const int c = int( pd_in.memo_counts( k ) );
                    pd.memo_counts( k ) = c;
                    for ( SI q = 0; q < c && q < K; ++q )
                        pd.memo_nbrs( k, q ) = int( pd_in.memo_nbrs( k, q ) );
                }
            }
        }
    }

    SI n() const { return pd.nb_seeds(); }

    void resize_scratch() {
        words = diagram::words_for<Local,TF>( cap, nbc, false );
        for ( auto &s : scratch )
            s.assign( size_t( words ) + 16, 0 );
    }

    /// one contiguous slice of `[ 0, n )` per thread
    static void thread_range( SI n, int t, int nt, SI &b, SI &e ) {
        b = SI( ( long long ) t * n / nt );
        e = SI( ( long long ) ( t + 1 ) * n / nt );
    }

    /// THE WEIGHTS `W` ( user order ) set on the storage, and the tree majorants rebuilt
    void set_weights( const std::vector<double> &W ) {
        const double t0 = now();
        const SI n = this->n();
        if constexpr ( requires { pd.sorted_weights; } ) {
            queue.run_threads( nt, [&]( int t ) {
                SI b, e;
                thread_range( n, t, nt, b, e );
                for ( SI k = b; k < e; ++k )
                    pd.sorted_weights( k ) = TF( W[ pd.user_id( k ) ] );
            } );
            const SI nb_nodes = SI( pd.tree.node_begin.shape( 0 ) );
            queue.run_threads( nt, [&]( int t ) {
                SI b, e;
                thread_range( nb_nodes, t, nt, b, e );
                for ( SI m = b; m < e; ++m )
                    bsp_refresh_majorant( pd.sorted_cloud(), pd.tree.node_begin( m ), pd.tree.node_end( m ),
                                          pd.tree.node_wa( m ), pd.tree.node_wb( m ) );
            } );
        } else {
            queue.run_threads( nt, [&]( int t ) {
                SI b, e;
                thread_range( n, t, nt, b, e );
                for ( SI k = b; k < e; ++k )
                    pd.weights( k ) = TF( W[ k ] );
            } );
        }
        t_majorant += now() - t0;
    }

    /// THE MEASURES `a` ( user order ) at the current weights, and if `fa` is not null the facets
    /// `c_ij` ( user identifiers ). One sweep, restarted on scratch overflow.
    void measures( std::vector<double> &a, std::vector<Facet> *fa = nullptr ) {
        const double t0 = now();
        const SI n = this->n();
        a.resize( n );
        for ( ;; ) {
            std::atomic<bool> overflowed{ false };
            queue.run_threads( nt, [&]( int t ) {
                Carver cv{ scratch[ t ].data(), words };
                Local c, piece;
                c.attach( cv, cap );
                if constexpr ( nbc > 1 ) piece.attach( cv, cap );
                else                     piece = c;
                auto &fv = fa_th[ t ];
                fv.clear();
                SI b, e;
                thread_range( n, t, nt, b, e );
                for ( SI k = b; k < e; ++k ) {
                    const SI i = pd.user_id( k );
                    if ( ! measure_and_facets( pd, k, c, piece, dom, *dist, a[ i ],
                                               [&]( SI j, double cij ) { fv.push_back( Facet{ i, pd.user_id( j ), cij } ); },
                                               fa != nullptr ) ) {
                        overflowed = true;
                        return;
                    }
                    // the memory ( 3D ) : the neighbours of this cell, proposed first to the next sweep --
                    // even from a rejected trial, a memory stays exact ( it only orders the cuts )
                    if constexpr ( requires { PD::has_memo; } ) {
                        if constexpr ( PD::has_memo )
                            diagram::memorize( c, k, pd.memo_nbrs, pd.memo_counts );
                    }
                }
            } );
            if ( ! overflowed )
                break;
            cap *= 2;
            resize_scratch();
            ++nb_overflowed;
        }
        if ( fa ) {
            fa->clear();
            for ( auto &v : fa_th )
                fa->insert( fa->end(), v.begin(), v.end() );
        }
        t_diag += now() - t0;
        ++nb_diag;
    }

    /// THE MOMENTS at the current weights : the BARYCENTER of each cell ( its seed if it is empty )
    /// and the transport cost `sum_i int_{cell_i} |x - p_i|^2 rho`, already reduced.
    ///
    /// One more sweep, on the SAME scratch and the same per-thread split as `measures` -- this is
    /// what spares the caller from rebuilding a diagram from Python to obtain a cost
    /// ( `integrate_moments_into` returns `int rho`, `int x rho` and `int |x|^2 rho`, in absolute
    /// coordinates, and the cost of a cell follows in one line ).
    ///
    /// THE MASS DOES NOT COME OUT OF IT, and this is deliberate : for a density that is not constant per
    /// piece, `integrate_moments_into` uses quadrature where `integrate_into` has a closed form ( the
    /// exact reduction of `SumOfGaussians` ) -- measured 7e-4 relative gap on a gaussian. The
    /// mass of a cell is the one Newton measured ( `newton.a` ), not this one.
    void moments( std::vector<double> &bary, double &cost ) {
        const double t0 = now();
        const SI n = this->n();
        std::vector<double> mass( n, 0.0 );
        bary.assign( size_t( n ) * D, 0.0 );
        std::vector<double> cost_th( nt, 0.0 );
        for ( ;; ) {
            std::atomic<bool> overflowed{ false };
            queue.run_threads( nt, [&]( int t ) {
                Carver cv{ scratch[ t ].data(), words };
                Local c, piece;
                c.attach( cv, cap );
                if constexpr ( nbc > 1 ) piece.attach( cv, cap );
                else                     piece = c;
                double acc = 0;
                SI b, e;
                thread_range( n, t, nt, b, e );
                for ( SI k = b; k < e; ++k ) {
                    const SI i = pd.user_id( k );
                    if ( ! diagram::make_cell( pd, c, k, dom ) ) {
                        overflowed = true;
                        return;
                    }
                    TF m = 0, m2 = 0;
                    TF mx[ D ];
                    auto first = [&]( int d ) -> TF & { return mx[ d ]; };
                    if ( ! diagram::integrate_moments_into<TF>( m, first, m2, c, piece, *dist ) ) {
                        overflowed = true;
                        return;
                    }
                    const auto p = pd.point( k );
                    double pp = 0, px = 0;
                    for ( int d = 0; d < D; ++d ) {
                        pp += double( p[ d ] ) * double( p[ d ] );
                        px += double( p[ d ] ) * double( mx[ d ] );
                    }
                    mass[ i ] = double( m );
                    acc += double( m2 ) - 2 * px + double( m ) * pp;
                    // an empty cell keeps its seed as barycenter -- there is nothing else to say,
                    // and dividing by zero downstream would be worse
                    for ( int d = 0; d < D; ++d )
                        bary[ size_t( i ) * D + d ] = double( m ) > 0 ? double( mx[ d ] ) / double( m ) : double( p[ d ] );
                }
                cost_th[ t ] = acc;
            } );
            if ( ! overflowed )
                break;
            cap *= 2;
            resize_scratch();
            ++nb_overflowed;
        }
        cost = 0;
        for ( double v : cost_th )
            cost += v;
        t_diag += now() - t0;
        ++nb_diag;
    }
};

} // namespace sdotplan
} // namespace sdot
