#pragma once

// =====================================================================================
// THE SOLVER'S ENTRY POINT ( what `SdotPlanNd.py` calls, in ONE `driver.call` ): the starting point, the
// width continuation if one is needed, Newton at each step, and what comes out.
//
// = The starting point
//
// Newton ( KMT ) requires an ADMISSIBLE start: no empty cell. We start from the given weights
// ( `w0`: the weights of a neighboring fit, which a reconstruction lives on ) or from zero -- the
// Voronoi, none of whose cells is empty as long as every seed is in the domain. A warm start
// that EMPTIES a cell ( weights inherited from other positions ) is worse than the Voronoi: the
// theory starts from a strictly positive floor, and an empty cell crawls back to it ( its
// gradient is constant, its hessian row zero ). We then restart from zero if that is better.
// ( two ways of doing better than the Voronoi were tried, and rejected: only "reviving" the empty
// cells, and a multiscale start -- both cascade, see `solvers_des_familles` README § 8. )
//
// And the Voronoi itself can leave empty cells ( seeds OUTSIDE the domain ): then the
// weights of a SIMILITUDE that brings the cloud back into the domain -- the Voronoi of the translated and
// contracted cloud is written as a power diagram of the original cloud, `w_i = |p_i|^2 - |a p_i +
// b|^2 / a`, and its cells are all fed.
//
// = The width continuation ( `Continuation.h` )
//
// There remains the case where cells have no mass because the DENSITY has none where they are
// ( narrow bumps, deserts ): neither a geometric start nor the damping can do anything about it,
// and direct Newton stagnates ( README § 9.1 ). We then first solve for the density convolved
// by a wide gaussian ( positive everywhere ), then narrower and narrower, each step starting from
// the weights of the previous one. `AUTO` triggers it when the best start still leaves a
// cell under `continuation_threshold` times the smallest target mass.
//
// = The target mass
//
// The cells PARTITION the domain: their masses sum to the mass of the density in the
// domain, whatever the weights. A density that the domain truncates ( gaussians in
// a box ) does not weigh 1 there: the target masses `nu` are rescaled to it, at each step,
// without which the residual can never vanish. What comes out is the transport towards the density
// RESTRICTED to the domain, normalized -- which is what is meant when a domain is given.
// =====================================================================================

#include "Continuation.h"
#include "Bounds.h"
#include "Newton.h"

namespace sdot {
namespace sdotplan {

/// what the caller reads in `stats( . )` -- same list on the python side ( `SdotPlanNd._STATS` )
enum Stat : int {
    STATUS = 0, RESIDUAL, RESIDUAL0, NB_ITER, NB_DIAG, NB_BACKTRACKS, T_MAJORANT, T_DIAG, T_ASM, T_LIN, T_LIM, EPS,
    DOMAIN_MASS, NB_OVERFLOWED, NB_CELL_LIM, NB_LIMIT_ROUNDS, LIN_NB_HIERARCHIES, LIN_NB_ITER, LIN_WORST, START, T_TOTAL,
    NB_CONTINUATION_STEPS, MIN_START_MASS, IT_SWITCH,
    NB_STATS
};
enum Start : int { START_GIVEN = 0, START_VORONOI = 1, START_SIMILARITY = 2 };

/// what each row of the history carries -- same list on the python side ( `SdotPlanNd._HISTORY` )
enum Hist : int { H_STEP = 0, H_T, H_RESIDUAL_L2, H_MIN_MASS, H_MAX_RESIDUAL, H_NB_DIAG, H_NB_EVALS, H_S, NB_HIST };

struct SolverOptions {
    NewtonOptions newton;
    Lin    lin = Lin::AUTO;
    LinearOptions lin_options;       ///< the tolerance, the AMG variant ( defaults: those of the solver )
    SI     cap0 = 64;                ///< vertices per local cell, at the start
    enum Continuation : int { NEVER = 0, AUTO = 1, ALWAYS = 2 };
    int    continuation = AUTO;
    double continuation_threshold = 1e-2; ///< AUTO: a cell under this factor of the smallest target triggers it
    double conv_s0 = 0;              ///< the first width ( 0: half the diameter of the domain )
    double conv_ratio = 1.4142135623730951;
    double conv_min = 0;             ///< the last width before 0 ( 0: the scale of the distribution )
};

/// the weights of the Voronoi of a SIMILITUDE of the cloud that fits it in the box `[ lo, hi ]`: the box
/// of the cloud is contracted ( never dilated ) and translated into the box reduced by a margin
template<int D>
void similarity( const auto &pd, const double *lo, const double *hi, std::vector<double> &w, double margin = 0.1 ) {
    const SI n = pd.nb_seeds();
    double p_lo[ D ], p_hi[ D ];
    for ( int d = 0; d < D; ++d ) { p_lo[ d ] = 1e300; p_hi[ d ] = -1e300; }
    for ( SI k = 0; k < n; ++k ) {
        const auto p = pd.point( k );
        for ( int d = 0; d < D; ++d ) { p_lo[ d ] = std::min( p_lo[ d ], double( p[ d ] ) ); p_hi[ d ] = std::max( p_hi[ d ], double( p[ d ] ) ); }
    }
    double a = 1;
    for ( int d = 0; d < D; ++d ) {
        const double span_dom = ( hi[ d ] - lo[ d ] ) * ( 1 - 2 * margin );
        const double span_pts = std::max( p_hi[ d ] - p_lo[ d ], 1e-300 );
        a = std::min( a, span_dom / span_pts );
    }
    double b[ D ];
    for ( int d = 0; d < D; ++d )                        // the center of the contracted cloud on the center of the domain
        b[ d ] = ( lo[ d ] + hi[ d ] ) / 2 - a * ( p_lo[ d ] + p_hi[ d ] ) / 2;
    w.assign( n, 0.0 );
    for ( SI k = 0; k < n; ++k ) {
        const auto p = pd.point( k );
        double pp = 0, qq = 0;
        for ( int d = 0; d < D; ++d ) {
            const double q = a * double( p[ d ] ) + b[ d ];
            pp += double( p[ d ] ) * double( p[ d ] );
            qq += q * q;
        }
        w[ pd.user_id( k ) ] = pp - qq / a;
    }
}

inline double minimum( const std::vector<double> &v ) {
    double m = v.empty() ? 0 : v[ 0 ];
    for ( double x : v ) m = std::min( m, x );
    return m;
}

/// THE SOLVER. `pd` carries WRITABLE weights and majorants ( `with_weights` ); `nu` and `w0`
/// are in user order. `weights` ( user order ), `hist` ( `nb_steps`, `rows [ step,
/// NB_HIST ]`, `weights [ step, n ]` optional ), `stats`, `masses [ n ]`, `bary [ n, D ]` and
/// `cost` ( cost ) are the outputs.
template<class TK>
void solve( const CpuQueue &queue, auto &pd, const auto &pd_in, const auto &dom, const auto &dist, const auto &nu_in, const auto &w0_in,
               const SolverOptions &o, auto &&weights, auto &&hist, auto &&stats, auto &&masses, auto &&bary, auto &&cost ) {
    using PD = DECAYED_TYPE_OF( pd );
    using Dist = DECAYED_TYPE_OF( dist );
    constexpr int D = PD::ct_dim;
    const SI n = pd.nb_seeds();
    const double t_begin = now();

    Convolved<Dist> conv( dist );
    Sweep<PD,DECAYED_TYPE_OF( dom ),Dist,TK> bal( queue, pd, pd_in, dom, dist, o.cap0 );
    auto lin = linear_solver( o.lin, n, D, o.lin_options );
    lin->order( bal.rank_of );                           // the tree order, for the aggregation of the multigrid
    Newton<decltype( bal )> newton( bal, *lin, o.newton );

    // ---- the history, one row per accepted step
    SI nb_steps = 0;
    int nb_steps_before = 0;                              // the steps of the previous stages
    double s_current = 0;
    const SI cap_steps = SI( hist.rows.shape( 0 ) );
    newton.o.after_step = [&]( int it, double t, int nb_evals ) {
        if ( nb_steps >= cap_steps ) return;
        const auto &A = newton.a;
        double mn = A.empty() ? 0 : A[ 0 ], mx = 0, l2 = 0;
        for ( SI i = 0; i < n; ++i ) {
            mn = std::min( mn, A[ i ] );
            mx = std::max( mx, std::fabs( A[ i ] - newton.nu[ i ] ) );
            l2 += ( A[ i ] - newton.nu[ i ] ) * ( A[ i ] - newton.nu[ i ] );
        }
        hist.rows( nb_steps, int( H_STEP ) ) = double( nb_steps_before + it );
        hist.rows( nb_steps, int( H_T ) ) = t;
        hist.rows( nb_steps, int( H_RESIDUAL_L2 ) ) = std::sqrt( l2 );
        hist.rows( nb_steps, int( H_MIN_MASS ) ) = mn;
        hist.rows( nb_steps, int( H_MAX_RESIDUAL ) ) = mx;
        hist.rows( nb_steps, int( H_NB_DIAG ) ) = double( bal.nb_diag );
        hist.rows( nb_steps, int( H_NB_EVALS ) ) = double( nb_evals );
        hist.rows( nb_steps, int( H_S ) ) = s_current;
        if constexpr ( DECAYED_TYPE_OF( hist.weights )::is_valid )
            for ( SI i = 0; i < n; ++i )
                hist.weights( nb_steps, i ) = newton.w[ i ];
        ++nb_steps;
    };

    // ---- the target, and the start ( on the widest density if the continuation is forced )
    std::vector<double> nu( n ), w( n, 0.0 );
    for ( SI i = 0; i < n; ++i ) nu[ i ] = double( nu_in( i ) );
    bool given = false;
    if constexpr ( DECAYED_TYPE_OF( w0_in )::is_valid ) {
        for ( SI i = 0; i < n; ++i ) { w[ i ] = double( w0_in( i ) ); given |= w[ i ] != 0; }
    }
    int start = given ? START_GIVEN : START_VORONOI;
    const double gauge = w[ 0 ];
    for ( SI i = 0; i < n; ++i ) w[ i ] -= gauge;

    // the stages of the continuation: the list of widths, `0` last
    double s0 = o.conv_s0;
    if ( s0 <= 0 ) {                                     // half the diameter of the domain, or of the cloud
        double lo[ D ], hi[ D ];
        for ( int d = 0; d < D; ++d ) { lo[ d ] = 1e300; hi[ d ] = -1e300; }
        if constexpr ( DECAYED_TYPE_OF( pd.box_min )::is_valid ) {
            for ( int d = 0; d < D; ++d ) { lo[ d ] = double( pd.box_min( d ) ); hi[ d ] = double( pd.box_max( d ) ); }
        } else {
            for ( SI k = 0; k < n; ++k ) {
                const auto p = pd.point( k );
                for ( int d = 0; d < D; ++d ) { lo[ d ] = std::min( lo[ d ], double( p[ d ] ) ); hi[ d ] = std::max( hi[ d ], double( p[ d ] ) ); }
            }
        }
        double diam2 = 0;
        for ( int d = 0; d < D; ++d ) diam2 += ( hi[ d ] - lo[ d ] ) * ( hi[ d ] - lo[ d ] );
        s0 = 0.5 * std::sqrt( diam2 );
    }
    const double s_min = o.conv_min > 0 ? o.conv_min : conv.min_scale( s0 );
    std::vector<double> scales = ( o.continuation == SolverOptions::ALWAYS && Convolved<Dist>::possible ) ? continuation_steps( s0, o.conv_ratio, s_min )
                                                                                                          : std::vector<double>{ 0.0 };

    std::vector<double> a;
    std::vector<Facet> fa;
    s_current = scales[ 0 ];
    bal.dist = &conv.at( s_current );
    newton.measures_and_facets( w, a, fa );
    const double nu_min = minimum( nu );
    if ( given && minimum( a ) < 1e-3 * nu_min ) {       // a warm start that empties a cell: the Voronoi, if it does better
        std::vector<double> w0( n, 0.0 ), a0;
        std::vector<Facet> fa0;
        newton.measures_and_facets( w0, a0, fa0 );
        if ( minimum( a0 ) > minimum( a ) ) { w.swap( w0 ); a.swap( a0 ); fa.swap( fa0 ); start = START_VORONOI; }
        else newton.bal.set_weights( w );
    }
    if constexpr ( DECAYED_TYPE_OF( pd.box_min )::is_valid ) {
        if ( minimum( a ) <= 0 ) {                       // seeds outside the domain: the similarity
            double lo[ D ], hi[ D ];
            for ( int d = 0; d < D; ++d ) { lo[ d ] = double( pd.box_min( d ) ); hi[ d ] = double( pd.box_max( d ) ); }
            std::vector<double> w1, a1;
            std::vector<Facet> fa1;
            similarity<D>( pd, lo, hi, w1 );
            newton.measures_and_facets( w1, a1, fa1 );
            if ( minimum( a1 ) > minimum( a ) ) { w.swap( w1 ); a.swap( a1 ); fa.swap( fa1 ); start = START_SIMILARITY; }
            else newton.bal.set_weights( w );
        }
    }
    const double min_start_mass = minimum( a );
    // AUTO: the density is missing where cells are -> the continuation, from the same start
    if ( o.continuation == SolverOptions::AUTO && Convolved<Dist>::possible && scales.size() == 1
      && ( min_start_mass < o.continuation_threshold * nu_min ) ) {
        scales = continuation_steps( s0, o.conv_ratio, s_min );
        s_current = scales[ 0 ];
        bal.dist = &conv.at( s_current );
        newton.measures_and_facets( w, a, fa );
    }

    // ---- the stages
    NewtonStats total;
    double domain_mass = 0;
    for ( PI step = 0; step < scales.size(); ++step ) {
        s_current = scales[ step ];
        if ( step > 0 ) {                               // the next density: the measures of the start have to be redone
            bal.dist = &conv.at( s_current );
            newton.measures_and_facets( w, a, fa );
        }
        // the target mass, at the scale of what the domain contains of THIS density
        std::vector<double> nu_s = nu;
        double nu_mass = 0;
        domain_mass = 0;
        for ( SI i = 0; i < n; ++i ) { domain_mass += a[ i ]; nu_mass += nu[ i ]; }
        if ( domain_mass > 0 && nu_mass > 0 && domain_mass != nu_mass )
            for ( SI i = 0; i < n; ++i ) nu_s[ i ] *= domain_mass / nu_mass;
        newton.nu = nu_s;
        newton.a = a;
        newton.fa = fa;
        if ( o.newton.trace )
            std::printf( "  stage %d / %d : s = %.4e, domain mass %.6f, smallest mass %.3e\n",
                         int( step + 1 ), int( scales.size() ), s_current, domain_mass, minimum( a ) );

        // the step by the limits ( 2D ): the pass is plugged into Newton when it is requested
        if constexpr ( D == 2 ) {
            Bounds2D<decltype( bal )> lim( bal );
            LocalBounds ll;
            ll.alpha_min = [&]( const std::vector<double> &W, const std::vector<double> &Dd, const std::vector<SI> &bad_cells,
                                double horizon, double eps, const Laplacian &L, SI &nb_cells ) {
                return lim.alpha_min( W, Dd, bad_cells, horizon, eps, L, nb_cells );
            };
            if ( o.newton.step == NewtonOptions::LIMITS )
                newton.bounds = &ll;
            newton.solves( w, true );
            newton.bounds = nullptr;
        } else
            newton.solves( w, true );

        // what the stage leaves: its weights, its measures ( for the next stage ), its counters
        w = newton.w;
        a = newton.a;
        fa = newton.fa;
        nb_steps_before = nb_steps > 0 ? int( double( hist.rows( nb_steps - 1, int( H_STEP ) ) ) ) + 1 : 0;
        total.nb_iter += newton.st.nb_iter;
        total.nb_backtracks += newton.st.nb_backtracks;
        total.nb_cell_lim += newton.st.nb_cell_lim;
        total.nb_limit_rounds += newton.st.nb_limit_rounds;
        total.t_asm += newton.st.t_asm;
        total.t_lim += newton.st.t_lim;
        if ( step == 0 ) { total.residual0 = newton.st.residual0; total.eps = newton.st.eps; total.it_switch = newton.st.it_switch; }
        total.status = newton.st.status;
        total.residual = newton.st.residual;
        if ( newton.st.status == NewtonStats::LINEAR_FAILURE )
            break;
        newton.st = NewtonStats{};
    }
    hist.nb_steps.set( nb_steps );

    for ( SI i = 0; i < n; ++i )
        weights( i ) = w[ i ];

    // the moments sweep below is not a RESIDUAL EVALUATION: the diagram count
    // the caller reads must remain that of the descent, otherwise two versions of the code
    // can no longer be compared
    const int nb_diag_descent = bal.nb_diag;

    // ---- THE MOMENTS, on the TRUE density and at the fitted weights
    // One more sweep, and that is all that the transport cost, the barycenters and the cell
    // masses require: Python no longer has to rebuild a diagram to get them. On `conv.at(
    // 0 )` explicitly -- a solve that stopped midway through the continuation would otherwise leave
    // moments of a CONVOLVED density, which are not the ones that were asked for.
    // the MASS of a cell is the one Newton measured -- not that of the moments sweep,
    // which uses quadrature where the measure has a closed form ( see `Sweep::moments` ).
    for ( SI i = 0; i < n; ++i )
        masses( i ) = a[ i ];
    {
        bal.dist = &conv.at( 0 );
        bal.set_weights( w );
        std::vector<double> bc;
        double c = 0;
        bal.moments( bc, c );
        for ( SI i = 0; i < n; ++i )
            for ( int d = 0; d < D; ++d )
                bary( i, d ) = bc[ size_t( i ) * D + d ];
        cost = c;
    }

    auto put = [&]( int i, double v ) { stats( i ) = v; };
    put( STATUS, double( total.status ) );
    put( RESIDUAL, total.residual );
    put( RESIDUAL0, total.residual0 );
    put( NB_ITER, double( total.nb_iter ) );
    put( NB_DIAG, double( nb_diag_descent ) );
    put( NB_BACKTRACKS, double( total.nb_backtracks ) );
    put( T_MAJORANT, bal.t_majorant );
    put( T_DIAG, bal.t_diag );
    put( T_ASM, total.t_asm );
    put( T_LIN, lin->st.total() );
    put( T_LIM, total.t_lim );
    put( EPS, total.eps );
    put( DOMAIN_MASS, domain_mass );
    put( NB_OVERFLOWED, double( bal.nb_overflowed ) );
    put( NB_CELL_LIM, double( total.nb_cell_lim ) );
    put( NB_LIMIT_ROUNDS, double( total.nb_limit_rounds ) );
    put( LIN_NB_HIERARCHIES, double( lin->st.nb_hierarchies ) );
    put( LIN_NB_ITER, double( lin->st.nb_iter ) );
    put( LIN_WORST, lin->st.worst );
    put( START, double( start ) );
    put( T_TOTAL, now() - t_begin );
    put( NB_CONTINUATION_STEPS, double( scales.size() ) );
    put( MIN_START_MASS, min_start_mass );
    put( IT_SWITCH, double( total.it_switch ) );
}

} // namespace sdotplan
} // namespace sdot
