#pragma once

// =====================================================================================
// SQUASHING CELLS ALONG A DIRECTION: how far can we go? ( 2D )
//
// The port of `solvers_des_familles/src/solver/Ecrasement.h` and `src/cell/FournisseurAlpha2D.h`
// ( README § 7, § 9.7 ) onto sdot's cell and providers.
//
// Newton proposes `d`; the damping tries `w + t d` and refuses as long as a cell falls below the
// floor. Each trial is a diagram. Here, for only the cells that the trial found
// below the floor, we compute the LIMIT along `d` -- the first `alpha` where the cell falls
// below `eps` -- with EXACT cells computed one by one, warm-started:
//
//   * the POLYNOMIAL ( Lebesgue measure ): as long as the cell keeps the same edges, each vertex
//     is AFFINE in `alpha` and the area is a polynomial of degree 2; we predict its root, we check
//     with an exact cell at `0.99 x` the prediction ( same edges = exact polynomial = limit
//     confirmed ), otherwise the computed cell carries the new combinatorics and we start over;
//   * the mass BISECTION ( any other density ): the mass along `w + alpha d` is not
//     a polynomial; a bisection on `alpha` between 0 and the horizon, one exact cell per round.
//
// = The cell at `w + alpha d`, without refreshing the tree ( `PdAlpha` )
//
// The affine majorant is linear in the weights: if a node carries `w( y ) <= a_w . y + b_w` and
// `d( y ) <= a_d . y + b_d`, then for any `alpha >= 0`
//
//     w( y ) + alpha d( y ) <= ( a_w + alpha a_d ) . y + ( b_w + alpha b_d ).
//
// This is exact, so pruning stays exact, and `alpha` can CHANGE FROM ONE CELL TO ANOTHER. The
// majorant of `d` is computed once per direction, outside the tree. `PdAlpha` is the
// storage seen this way: the same points, the same boxes, shifted weights and majorants --
// and it is the ORDINARY `ProviderBsp` that traverses it.
//
// = The warm start
//
// Before the traversal, we propose the planes of the cell's neighbors at `alpha = 0` ( Newton's
// Laplacian already carries them ): this is the provider's MEMORY ( `MEMO` ), which proposes them
// first and skips them afterwards. Six cuts without a traversal, and the cell is already small when
// the traversal starts: pruning, being exact, discards almost everything.
// =====================================================================================

#include "Sweep.h"
#include "../UnitDensity.h"
#include <atomic>
#include <limits>

namespace sdot {
namespace sdotplan {

constexpr double INFINITE = std::numeric_limits<double>::infinity();

// ---- the storage at `w + alpha d` -------------------------------------------------------------------

/// the majorant of a node, shifted: `wa + alpha da`
template<class View>
struct MajorantAlphaA {
    const View   &wa;
    const double *da;
    double        alpha;
    double operator()( SI n, int d ) const { return double( wa( n, d ) ) + alpha * da[ 2 * n + d ]; }
};
template<class View>
struct MajorantAlphaB {
    const View   &wb;
    const double *db;
    double        alpha;
    double operator()( SI n ) const { return double( wb( n ) ) + alpha * db[ n ]; }
};

/// the tree as seen by `ProviderBsp`: boxes and slices as they are, majorants shifted
template<class Tree>
struct AlphaTree {
    const DECAYED_TYPE_OF( std::declval<Tree>().node_box )   &node_box;
    const DECAYED_TYPE_OF( std::declval<Tree>().node_begin ) &node_begin;
    const DECAYED_TYPE_OF( std::declval<Tree>().node_end )   &node_end;
    MajorantAlphaA<DECAYED_TYPE_OF( std::declval<Tree>().node_wa )> node_wa;
    MajorantAlphaB<DECAYED_TYPE_OF( std::declval<Tree>().node_wb )> node_wb;
};

/// THE BSP STORAGE at `w + alpha d`, the hot list as memory ( RANKS, sorted ascending )
template<class PD>
struct PdAlphaBsp {
    using TF = typename PD::TF;
    static constexpr int  ct_dim = PD::ct_dim;
    static constexpr bool on_cpu = true;
    static constexpr bool has_weights = true;

    const PD     &pd;
    const double *w, *d;                                 ///< user order
    double        alpha;
    AlphaTree<DECAYED_TYPE_OF( std::declval<PD>().tree )> tree;
    const int    *hot;
    int           nb_hot;

    SI   nb_seeds() const { return pd.nb_seeds(); }
    auto point( SI k ) const { return pd.point( k ); }
    SI   user_id( SI k ) const { return pd.user_id( k ); }
    TF   weight( SI k ) const { const SI i = pd.user_id( k ); return TF( w[ i ] + alpha * d[ i ] ); }
    int  memo_counts( SI ) const { return nb_hot; }
    int  memo_nbrs( SI, int q ) const { return hot[ q ]; }

    template<class TK>
    auto provider( SI k0 ) const { return ProviderBsp<PdAlphaBsp,TK,ct_dim,true,true>( *this, k0 ); }
};

/// THE FLAT STORAGE at `w + alpha d`: all the seeds, no warm start
template<class PD>
struct PdAlphaPlain {
    using TF = typename PD::TF;
    static constexpr int  ct_dim = PD::ct_dim;
    static constexpr bool on_cpu = true;
    static constexpr bool has_weights = true;

    const PD     &pd;
    const double *w, *d;
    double        alpha;

    SI   nb_seeds() const { return pd.nb_seeds(); }
    auto point( SI k ) const { return pd.point( k ); }
    SI   user_id( SI k ) const { return k; }
    TF   weight( SI k ) const { return TF( w[ k ] + alpha * d[ k ] ); }

    template<class TK>
    auto provider( SI k0 ) const { return ProviderAll<PdAlphaPlain,TK,ct_dim>( *this, k0 ); }
};

/// ONLY THE PLANES OF A LIST ( of ranks ): to rebuild a cell whose neighbors are known
template<class PDA,class TK>
struct ProviderList {
    const PDA &pd;
    SI  k0;
    const int *ranks;
    int nb, q = 0;
    typename PDA::TF p0[ 2 ], w0;

    ProviderList( const PDA &pd, SI k0, const int *ranks, int nb ) : pd( pd ), k0( k0 ), ranks( ranks ), nb( nb ) {
        const auto p = pd.point( k0 );
        p0[ 0 ] = p[ 0 ]; p0[ 1 ] = p[ 1 ];
        w0 = pd.weight( k0 );
    }
    template<class State>
    bool next( const State &, NothingLocal &, Plane<TK,2> &p ) {
        if ( q >= nb ) return false;
        const SI k = ranks[ q++ ];
        const auto pj = pd.point( k );
        typename PDA::TF p1[ 2 ] = { pj[ 0 ], pj[ 1 ] };
        p = bisector<TK,2>( p0, w0, p1, pd.weight( k ), int( k ) );
        return true;
    }
};

// ---- the polynomial of a cell -----------------------------------------------------------------------

/// THE POLYNOMIAL OF A CELL: `q( alpha ) = a0 + a1 alpha + a2 alpha^2`, and what we get from it.
struct CellPolynomial {
    enum State : int { OK = 0, EMPTY_AT_START, OVERFLOWED, DEGENERATE };

    double a0 = 0, a1 = 0, a2 = 0;
    double alpha_edge = INFINITE;   ///< first edge that vanishes: the combinatorics change
    int    nb_edges = 0;
    int    state = OK;

    double operator()( double alpha ) const { return a0 + alpha * ( a1 + alpha * a2 ); }

    /// the real roots of `q( alpha ) == level`, sorted; returns their number ( 0, 1 or 2 ).
    int roots( double level, double &r1, double &r2 ) const {
        const double c = a0 - level;
        if ( std::fabs( a2 ) <= 1e-300 ) {
            if ( a1 == 0 ) return 0;
            r1 = -c / a1;
            return 1;
        }
        const double disc = a1 * a1 - 4 * a2 * c;
        if ( disc < 0 ) return 0;
        const double s = std::sqrt( disc );
        const double q = -0.5 * ( a1 + ( a1 >= 0 ? s : -s ) );
        double x1 = q / a2, x2 = ( q != 0 ) ? c / q : x1;
        if ( x1 > x2 ) std::swap( x1, x2 );
        r1 = x1; r2 = x2;
        return 2;
    }

    /// the first POSITIVE root of `q == level` ( `INFINITE` if there is none ): the predicted limit
    double first_root( double level ) const {
        double r1, r2;
        const int nr = roots( level, r1, r2 );
        if ( nr >= 1 && r1 > 0 ) return r1;
        if ( nr >= 2 && r2 > 0 ) return r2;
        return INFINITE;
    }
};

/// A line `n . x <= c + alpha delta`, with a fixed normal.
struct Line2 { double nx, ny, c, delta; };

/// THE POLYNOMIAL OF A CELL `cel` of the seed of rank `k0`, computed at weights `w + alpha0 d`, along
/// `d`: `q( beta )` is the area at `w + ( alpha0 + beta ) d`, combinatorics frozen. `pda` gives the
/// points by rank and `w`, `d` by identifier.
template<class Local,class PDA>
CellPolynomial cell_polynomial( const Local &cel, SI k0, const PDA &pda, double alpha0, std::vector<Line2> &lines,
                              std::vector<double> &v0x, std::vector<double> &v0y, std::vector<double> &v1x, std::vector<double> &v1y ) {
    CellPolynomial q;
    if ( cel.nb == 0 ) { q.state = CellPolynomial::EMPTY_AT_START; return q; }
    const int nb = cel.nb;
    q.nb_edges = nb;
    lines.resize( nb ); v0x.resize( nb ); v0y.resize( nb ); v1x.resize( nb ); v1y.resize( nb );

    // ---- the lines: edge `j` goes from `v_j` to `v_j+1`, carried by cut `cid[ j ]`
    const SI i0 = pda.user_id( k0 );
    const auto pi = pda.point( k0 );
    const double xi = double( pi[ 0 ] ), yi = double( pi[ 1 ] ), wi = pda.w[ i0 ] + alpha0 * pda.d[ i0 ], di = pda.d[ i0 ];
    for ( int j = 0; j < nb; ++j ) {
        const int id = cel.cid[ j ];
        if ( id >= 0 ) {
            const SI ij = pda.user_id( id );
            const auto pj = pda.point( id );
            const double xj = double( pj[ 0 ] ), yj = double( pj[ 1 ] ), wj = pda.w[ ij ] + alpha0 * pda.d[ ij ];
            const double nx = xj - xi, ny = yj - yi;
            lines[ j ] = { nx, ny, 0.5 * ( nx * ( xj + xi ) + ny * ( yj + yi ) + wi - wj ), 0.5 * ( di - pda.d[ ij ] ) };
        } else {                                         // the domain: the line read off the edge, immobile
            typename Local::TKernel dx, dy, off;
            cel.plane_of_edge( j, dx, dy, off );
            lines[ j ] = { double( dx ), double( dy ), double( off ), 0 };
        }
    }

    // ---- the vertices, affine: `v_j` is the intersection of edges `j-1` and `j`
    for ( int j = 0; j < nb; ++j ) {
        const Line2 &a = lines[ j ? j - 1 : nb - 1 ], &b = lines[ j ];
        const double det = a.nx * b.ny - a.ny * b.nx;
        if ( ! ( std::fabs( det ) > 0 ) ) { q.state = CellPolynomial::DEGENERATE; return q; }
        v0x[ j ] = ( a.c * b.ny - b.c * a.ny ) / det;
        v0y[ j ] = ( a.nx * b.c - b.nx * a.c ) / det;
        v1x[ j ] = ( a.delta * b.ny - b.delta * a.ny ) / det;
        v1y[ j ] = ( a.nx * b.delta - b.nx * a.delta ) / det;
    }

    // ---- the signed area, and the signed length of each edge
    double a0 = 0, a1 = 0, a2 = 0;
    for ( int j = 0; j < nb; ++j ) {
        const int l = j + 1 < nb ? j + 1 : 0;
        a0 += v0x[ j ] * v0y[ l ] - v0x[ l ] * v0y[ j ];
        a1 += v0x[ j ] * v1y[ l ] - v0x[ l ] * v1y[ j ] + v1x[ j ] * v0y[ l ] - v1x[ l ] * v0y[ j ];
        a2 += v1x[ j ] * v1y[ l ] - v1x[ l ] * v1y[ j ];
    }
    const double sg = a0 < 0 ? -0.5 : 0.5;
    q.a0 = sg * a0; q.a1 = sg * a1; q.a2 = sg * a2;

    for ( int j = 0; j < nb; ++j ) {
        const int l = j + 1 < nb ? j + 1 : 0;
        const double tx = -lines[ j ].ny, ty = lines[ j ].nx; // along edge `j`
        double l0 = ( v0x[ l ] - v0x[ j ] ) * tx + ( v0y[ l ] - v0y[ j ] ) * ty;
        double l1 = ( v1x[ l ] - v1x[ j ] ) * tx + ( v1y[ l ] - v1y[ j ] ) * ty;
        if ( l0 < 0 ) { l0 = -l0; l1 = -l1; }
        if ( l1 < 0 )
            q.alpha_edge = std::min( q.alpha_edge, -l0 / l1 );
    }
    return q;
}

// ---- the pass -----------------------------------------------------------------------------------------

/// WHAT WE KNOW OF A CELL at the end: its limit, and how we obtained it.
struct CellLimit {
    enum State : int { CONFIRMED = 0, CORRECTED, HORIZON, EMPTY_AT_START, FAILED };
    double alpha      = INFINITE;    ///< the first `alpha` where the cell falls below `level`
    double alpha_poly = INFINITE;    ///< what the polynomial at 0 predicted
    int    rounds      = 0;         ///< cells computed in addition to the one at 0
    int    state       = FAILED;
};

struct BoundsOptions {
    double coeff     = 0.99;       ///< we check at `a_ok + coeff * ( predicted - a_ok )`
    double tol       = 1e-2;       ///< relative precision requested on the limit
    int    max_rounds = 12;
};

/// THE LIMITS, on a 2D `Sweep`: the "predict, check, correct" pass ( polynomial, Lebesgue
/// measure ) or the mass bisection ( any other density ), for only the requested cells.
template<class Bal>
struct Bounds2D {
    using PD    = DECAYED_TYPE_OF( std::declval<Bal>().pd );
    using Local = typename Bal::Local;
    using TK    = typename Local::TKernel;
    using TF    = typename PD::TF;
    using Dist  = DECAYED_TYPE_OF( *std::declval<Bal>().dist );
    static constexpr bool bsp = requires( const PD &p ) { p.tree; };
    static constexpr bool polynomial = std::is_same_v<Dist,UnitDensity>;
    using PDA = std::conditional_t<bsp,PdAlphaBsp<PD>,PdAlphaPlain<PD>>;

    Bal            &bal;
    BoundsOptions  o;
    std::vector<double> dt, da, db;                      ///< `d` in tree order, its majorant per node
    std::vector<std::vector<std::int32_t>> scratch;      ///< the scratch of THIS pass, per thread ( it grows by itself )
    std::vector<SI> caps;
    std::vector<CellLimit> lim;                      ///< per identifier

    struct ThreadState {                                         ///< what a thread keeps from one cell to the next
        std::vector<Line2> lines;
        std::vector<double> v0x, v0y, v1x, v1y;
        std::vector<int> hot;                         ///< the neighbors of the last good cell ( sorted ranks )
        std::vector<int> cids_ok, c2;                    ///< ALL its cuts, domain included ( the combinatorics )
    };

    Bounds2D( Bal &bal ) : bal( bal ) {
        scratch.resize( bal.nt );
        caps.assign( bal.nt, bal.cap );
        for ( int t = 0; t < bal.nt; ++t )
            scratch[ t ].assign( size_t( diagram::words_for<Local,TF>( caps[ t ], Bal::nbc, false ) ) + 16, 0 );
    }

    /// the storage at `w + alpha d`, the hot list `hot` ( sorted ranks )
    PDA pd_alpha( const double *w, const double *d, double alpha, const int *hot, int nb ) const {
        if constexpr ( bsp )
            return PDA{ bal.pd, w, d, alpha,
                        { bal.pd.tree.node_box, bal.pd.tree.node_begin, bal.pd.tree.node_end,
                          { bal.pd.tree.node_wa, da.data(), alpha }, { bal.pd.tree.node_wb, db.data(), alpha } },
                        hot, nb };
        else
            return PDA{ bal.pd, w, d, alpha };
    }

    /// ONCE PER DIRECTION: `d` in tree order, and its majorant per node
    void prepare( const double *d ) {
        if constexpr ( bsp ) {
            const SI n = bal.n();
            dt.resize( n );
            for ( SI k = 0; k < n; ++k ) dt[ k ] = d[ bal.pd.user_id( k ) ];
            const SI nb_nodes = SI( bal.pd.tree.node_begin.shape( 0 ) );
            da.assign( 2 * nb_nodes, 0.0 );
            db.assign( nb_nodes, 0.0 );
            struct WeightsView { const double *v; double operator()( SI k ) const { return v[ k ]; } };
            struct Wa { double *v; double &operator()( int d ) const { return v[ d ]; } };
            bal.queue.run_threads( bal.nt, [&]( int t ) {
                SI b, e;
                Bal::thread_range( nb_nodes, t, bal.nt, b, e );
                for ( SI m = b; m < e; ++m ) {
                    const SI nb = SI( bal.pd.tree.node_begin( m ) ), ne = SI( bal.pd.tree.node_end( m ) );
                    if ( ne <= nb ) continue;
                    bsp_weight_majorant<2>( bal.pd.sorted_positions, WeightsView{ dt.data() }, nb, ne, Wa{ da.data() + 2 * m }, db[ m ] );
                }
            } );
        }
    }

    /// THE CELL of rank `k` at `w + alpha d`, warm-started from `hot` ( `full_walk = false`: those planes
    /// only ). Returns `false` if the thread's scratch was not enough even after growing ( never, in practice ).
    bool compute_cell( int t, const PDA &pda, SI k, Local &c, Local &piece, bool full_walk, const int *ranks = nullptr, int nb_ranks = 0 ) {
        for ( int trial = 0; trial < 8; ++trial ) {
            Carver cv{ scratch[ t ].data(), SI( scratch[ t ].size() ) - 16 };
            c.attach( cv, caps[ t ] );
            if constexpr ( Bal::nbc > 1 ) piece.attach( cv, caps[ t ] );
            else                          piece = c;
            bool ok;
            if ( full_walk ) {
                ok = diagram::make_cell( pda, c, k, bal.dom );
            } else {
                ok = c.load( bal.dom );
                if ( ok ) {
                    ProviderList<PDA,TK> f( pda, k, ranks, nb_ranks );
                    ok = run<true>( c, f ) != CutStatus::NO_ROOM;
                }
            }
            if ( ok ) return true;
            caps[ t ] *= 2;
            scratch[ t ].assign( size_t( diagram::words_for<Local,TF>( caps[ t ], Bal::nbc, false ) ) + 16, 0 );
        }
        return false;
    }

    /// the mass of `c` against the sweep's distribution
    double mass( Local &c, Local &piece ) const {
        if ( c.nb == 0 ) return 0;
        TF m = 0;
        if ( ! diagram::integrate_into<TF>( m, c, piece, *bal.dist ) ) return 0;
        return double( m );
    }

    /// THE LIMITS of the `bad_cells` cells ( identifiers ), along `d` from `w`, under `horizon`,
    /// at level `eps`; `L` carries the neighbors at `alpha = 0`. Returns `min_i alpha_i`; `nb_cells`
    /// counts the computed cells. `lim[ i ]` is filled for each `i` of `bad_cells`.
    double alpha_min( const std::vector<double> &w, const std::vector<double> &d, const std::vector<SI> &bad_cells,
                      double horizon, double eps, const Laplacian &L, SI &nb_cells ) {
        const SI n = bal.n();
        if ( SI( lim.size() ) != n ) lim.assign( n, CellLimit{} );
        prepare( d.data() );
        std::atomic<double> current{ INFINITE };
        auto lower = [&]( double a ) {
            double c = current.load();
            while ( a < c && ! current.compare_exchange_weak( c, a ) ) {}
        };
        std::atomic<SI> nb_cel{ 0 };
        std::vector<ThreadState> thread_states( bal.nt );
        const SI nm = SI( bad_cells.size() );

        bal.queue.run_threads( bal.nt, [&]( int t ) {
            ThreadState &f = thread_states[ t ];
            Local c, piece;
            SI b, e;
            Bal::thread_range( nm, t, bal.nt, b, e );
            for ( SI j = b; j < e; ++j ) {
                const SI i = bad_cells[ j ], k = bal.rank_of[ i ];
                CellLimit &Li = lim[ i ];
                Li = CellLimit{};
                // the neighbors at 0, as sorted ranks: the warm start, and the cell at 0 without a traversal
                f.hot.clear();
                for ( SI q = L.row[ i ]; q < L.row[ i + 1 ]; ++q )
                    f.hot.push_back( int( bal.rank_of[ L.col[ q ] ] ) );
                std::sort( f.hot.begin(), f.hot.end() );
                auto keep_neighbors = [&]() {             // the current cell is good: we restart from it
                    f.hot.clear();
                    f.cids_ok.assign( c.cid, c.cid + c.nb );
                    std::sort( f.cids_ok.begin(), f.cids_ok.end() );
                    for ( int q = 0; q < c.nb; ++q ) if ( c.cid[ q ] >= 0 ) f.hot.push_back( c.cid[ q ] );
                    std::sort( f.hot.begin(), f.hot.end() );
                };
                auto finish = [&]( double a, int state ) { Li.alpha = a; Li.state = state; if ( state != CellLimit::HORIZON ) lower( a ); };

                if constexpr ( polynomial ) {
                    // ---- predict, check, correct
                    PDA p0 = pd_alpha( w.data(), d.data(), 0, f.hot.data(), int( f.hot.size() ) );
                    compute_cell( t, p0, k, c, piece, false, f.hot.data(), int( f.hot.size() ) );
                    CellPolynomial q = cell_polynomial( c, k, p0, 0, f.lines, f.v0x, f.v0y, f.v1x, f.v1y );
                    if ( q.state != CellPolynomial::OK ) { Li.state = CellLimit::EMPTY_AT_START; Li.alpha = 0; lower( 0 ); continue; }
                    Li.alpha_poly = q.first_root( eps );
                    double a_ok = 0, a_bad = INFINITE, pred = Li.alpha_poly, target = pred;
                    keep_neighbors();
                    bool done = false;
                    for ( ; Li.rounds < o.max_rounds && ! done; ) {
                        bool on_pred = target == pred;
                        double a_test = std::min( target, horizon );
                        if ( ! ( a_test < a_bad ) ) { a_test = 0.5 * ( a_ok + a_bad ); on_pred = false; }
                        if ( on_pred && a_test < horizon ) a_test = a_ok + o.coeff * ( a_test - a_ok );
                        if ( ! ( a_test > a_ok ) ) { finish( a_ok, CellLimit::CORRECTED ); done = true; break; }

                        PDA pa = pd_alpha( w.data(), d.data(), a_test, f.hot.data(), int( f.hot.size() ) );
                        compute_cell( t, pa, k, c, piece, true );
                        ++Li.rounds;
                        ++nb_cel;

                        // ---- same edges ( domain included ): the polynomial was exact from `a_ok` to `a_test`
                        f.c2.assign( c.cid, c.cid + c.nb );
                        std::sort( f.c2.begin(), f.c2.end() );
                        const bool same = c.nb > 0 && f.c2 == f.cids_ok;
                        if ( same ) {
                            if ( a_test >= horizon ) { finish( pred < INFINITE && pred < horizon ? pred : horizon, CellLimit::HORIZON ); done = true; break; }
                            if ( on_pred ) { finish( pred, Li.rounds == 1 ? CellLimit::CONFIRMED : CellLimit::CORRECTED ); done = true; break; }
                            a_ok = a_test;
                            target = pred;
                            if ( a_bad < INFINITE && a_bad - a_ok <= o.tol * a_bad ) { finish( a_ok, CellLimit::CORRECTED ); done = true; break; }
                            continue;
                        }

                        // ---- the combinatorics changed: the computed cell carries the new polynomial
                        const CellPolynomial q2 = cell_polynomial( c, k, pa, a_test, f.lines, f.v0x, f.v0y, f.v1x, f.v1y );
                        const double m = q2.state == CellPolynomial::OK ? q2.a0 : 0.0;
                        if ( m >= eps ) {                // good: we restart from here
                            a_ok = a_test;
                            keep_neighbors();
                            const double beta = q2.first_root( eps );
                            pred = target = a_test + beta;
                            if ( a_test >= horizon ) { finish( horizon, CellLimit::HORIZON ); done = true; break; }
                            if ( ! ( pred < a_bad ) ) target = 0.5 * ( a_ok + a_bad );
                            if ( beta <= o.tol * a_test ) { finish( a_test, CellLimit::CORRECTED ); done = true; break; }
                        } else {                         // bad: the limit is before
                            a_bad = a_test;
                            double r1, r2, beta = -INFINITE;
                            if ( q2.state == CellPolynomial::OK ) {
                                const int nr = q2.roots( eps, r1, r2 );
                                if ( nr >= 1 && r1 < 0 ) beta = r1;
                                if ( nr >= 2 && r2 < 0 ) beta = r2;
                            }
                            target = a_test + beta;
                            if ( ! ( target > a_ok && target < a_bad ) ) target = 0.5 * ( a_ok + a_bad );
                            if ( target == pred ) target = std::nextafter( target, a_ok );
                        }
                        if ( a_bad < INFINITE && a_bad - a_ok <= o.tol * a_bad ) { finish( a_ok, CellLimit::CORRECTED ); done = true; break; }
                    }
                    if ( ! done ) finish( a_ok, CellLimit::FAILED );   // the conservative one, for lack of better
                } else {
                    // ---- the mass bisection, between 0 ( admissible ) and the horizon ( the trial saw it below )
                    double a_ok = 0, a_bad = horizon;
                    for ( ; Li.rounds < o.max_rounds && a_bad - a_ok > o.tol * a_bad; ) {
                        const double mid = 0.5 * ( a_ok + a_bad );
                        PDA pa = pd_alpha( w.data(), d.data(), mid, f.hot.data(), int( f.hot.size() ) );
                        compute_cell( t, pa, k, c, piece, true );
                        ++Li.rounds;
                        ++nb_cel;
                        if ( mass( c, piece ) >= eps ) { a_ok = mid; keep_neighbors(); }
                        else a_bad = mid;
                    }
                    Li.alpha = a_ok;
                    Li.state = CellLimit::CORRECTED;
                    lower( a_ok );
                }
            }
        } );
        nb_cells += nb_cel.load();
        double res = current.load();
        return res < INFINITE ? res : horizon;
    }
};

} // namespace sdotplan
} // namespace sdot
