#pragma once

// =====================================================================================
// THE SEMI-DISCRETE TRANSPORT, SOLVED: find `w` such that `mass( Lag_i( w ) ) = nu_i` for every `i`.
// The damped Newton of `solvers_des_familles` ( `src/solver/Newton.h`, README § 3, § 7, § 10 ), which
// won against everything that was tried -- L-BFGS and conjugate gradient on the dual ( 2 to 4x more
// diagrams ), series continuation, multiscale, the barrier -- taken up here on the
// diagrams of `Sweep.h`.
//
// = The dual, and why Newton
//
//     Phi( w ) = integral min_i ( |x - p_i|^2 - w_i ) rho( x ) dx + sum_i w_i nu_i
//
// is CONCAVE, with gradient `nu_i - mass( Lag_i( w ) )`, and its hessian is, up to sign, the
// laplacian of the Laguerre graph ( `Laplacian.h` ). Its kernel is the constants: we fix `w_0 = 0`.
//
// = The damping ( Kitagawa-Merigot-Thibert ), and what it protects
//
// The hessian is only defined as long as no cell is empty. The trial step must therefore
// keep every mass above a floor `eps` fixed at the start ( half the smallest
// mass, initial or target ), and decrease the residual by at least `1 - t / 2`. The start must
// be admissible ( no empty cell ): that is the caller's business ( `Solve.h` ).
//
// The decrease is required STRICT ( `n2 < nr` ): without that, a step that tends to zero passes
// the test by equality as soon as `t` is negligible, and Newton spins in place forever
// ( measured: 54 diagrams per iteration at constant residual ). We then exit in STAGNATION -- the
// numerical floor, not a failure, and the difference is read on `residual`.
//
// = The trial step
//
// TRIALS: `t` restarts from `mult_ok` times the last accepted step ( capped at 1 ), and is halved
// as long as the step is refused. Restarting from 1 at each iteration cost ten diagrams per
// step in the linear phase ( `t ~ 1e-3`, nearly empty cells ) only to fall back to the same `t`.
//
// LIMITS ( 2D ): the trial `t = beta` first; if some cells fall below `eps` there, their
// LIMITS along `d` ( `Bounds.h`: one exact cell per round, warm, the polynomial or the
// mass bisection ), the step brought back under the smallest, and we start again. The best step
// measured ( -37 % of diagrams on the lines, the backtracks disappear on the densities ); it
// requires a limits pass that only 2D knows how to do today.
//
// = What an iteration costs
//
// ONE diagram per trial step, and nothing more: the accepted step delivers both the measures ( the
// residual ) and the facets ( the next hessian ). Time is counted per item -- majorants,
// diagrams, assembly, solve -- because it is the BREAKDOWN we want to read.
// =====================================================================================

#include "Sweep.h"
#include "Linear.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <functional>
#include <vector>

namespace sdot {
namespace sdotplan {

struct NewtonOptions {
    double tol_abs    = 1e-8;    ///< stop: `max_i |a_i - nu_i| <= tol_abs` ...
    double tol_rel    = 0;       ///< ... or `max_i |a_i - nu_i| / nu_i <= tol_rel` ( 0: never )
    int    maxit      = 100;
    int    max_backtracks = 60; ///< halvings of the step, at most, per iteration
    double t_min      = 1e-10;   ///< below this, we declare STAGNATION
    double mult_ok    = 4;       ///< TRIALS: the next trial starts from `mult_ok * t` ( capped at 1 )
    bool   trace      = false;
    enum Step : int { TRIALS = 0, LIMITS = 1 };
    int    step       = TRIALS;
    double factor     = 0.9;     ///< LIMITS: `t = factor * alpha*`
    double beta0      = 0.25;    ///< LIMITS: the very first trial
    double mult_lim   = 2;       ///< LIMITS: after a trial that passed DIRECTLY, `beta *= mult_lim`
    double confidence = 0;       ///< LIMITS: after a CORRECTED step, the next trial is at least `confidence * t`
    /// called after each ACCEPTED step ( and at the start, `it = 0` ): `w` and `a` are those of the step
    std::function<void( int it, double t, int nb_evals )> after_step;
};

struct NewtonStats {
    int    status = 0;              ///< why the loop stopped ( `Status` )
    enum Status : int { RUNNING = 0, CONVERGED = 1, MAX_ITERATIONS = 2, STAGNATION = 3, LINEAR_FAILURE = 4 };
    double residual  = 0;           ///< the `max_i |a_i - nu_i|` reached
    double residual0 = 0;           ///< the same AT THE START
    double eps    = 0;           ///< the mass floor of the damping
    int    nb_iter = 0, nb_backtracks = 0;
    SI     nb_cell_lim = 0;      ///< cells computed by the limits passes, in all
    int    nb_limit_rounds = 0;   ///< LIMITS: trials corrected by local limits
    double t_asm = 0, t_lin = 0, t_lim = 0;
    static const char *text( int status ) {
        switch ( status ) {
            case CONVERGED:      return "CONVERGED";
            case MAX_ITERATIONS: return "MAX ITERATIONS";
            case STAGNATION:     return "STAGNATION";
            case LINEAR_FAILURE: return "LINEAR SOLVER FAILURE";
            default:             return "?";
        }
    }
};

/// WHAT A LIMITS PASS returns to Newton ( `Bounds.h`, 2D ): `alpha` per requested cell.
/// A `Sweep` that has none ( `bounds == nullptr` ) takes the step by TRIALS.
struct LocalBounds {
    /// the limits of the `bad_cells` ( bad ) cells ( identifiers ) along `d` from `w`, under
    /// `horizon`, at level `eps`; returns `min_i alpha_i` and the number of cells computed
    std::function<double( const std::vector<double> &w, const std::vector<double> &d, const std::vector<SI> &bad_cells,
                          double horizon, double eps, const Laplacian &L, SI &nb_cells )> alpha_min;
};

template<class Bal>
struct Newton {
    Bal                &bal;
    LinearSolver    &lin;
    NewtonOptions       o;
    const LocalBounds *bounds = nullptr;

    std::vector<double> nu;      ///< the target mass, per seed
    std::vector<double> w;       ///< the current weights, `w[ 0 ] == 0`
    std::vector<double> a;       ///< the current masses
    std::vector<double> d;       ///< the last Newton direction ( `d[ 0 ] == 0` )
    std::vector<Facet> fa;     ///< the facets of the current diagram ( that of `w` )
    NewtonStats         st;
    double              t_last = 1;   ///< the last accepted step
    int                 nb_evals_last = 0;
    double              beta;            ///< LIMITS: the next trial -- KEPT from one `solves` to the next ( the
                                         ///< steps of a continuation: a close start accepts `t = 1` right away )

    Newton( Bal &bal, LinearSolver &lin, NewtonOptions o = {} ) : bal( bal ), lin( lin ), o( o ), beta( o.beta0 ) {}

    static double norm2( const std::vector<double> &v ) {
        double s = 0;
        for ( double x : v ) s += x * x;
        return std::sqrt( s );
    }

    /// `| a - nu |_2`, the merit of the damping
    double merit( const std::vector<double> &A ) const {
        double s = 0;
        for ( SI i = 0; i < SI( A.size() ); ++i ) s += ( nu[ i ] - A[ i ] ) * ( nu[ i ] - A[ i ] );
        return std::sqrt( s );
    }

    /// THE MEASURES AND THE FACETS for the weights `W`
    void measures_and_facets( const std::vector<double> &W, std::vector<double> &res, std::vector<Facet> &f ) {
        bal.set_weights( W );
        bal.measures( res, &f );
    }

    /// THE LOOP, from `w_init` ( `a` and `fa` ALREADY computed for `w_init` if `already_measured` ).
    /// Returns `true` if the stopping criterion is reached. The diagram carries the ACCEPTED weights on output.
    bool solves( const std::vector<double> &w_init, bool already_measured = false ) {
        const SI n = bal.n();
        std::vector<double> a2, b, w2;
        std::vector<Facet> fa2;
        Laplacian L;

        w = w_init;
        const double gauge = w[ 0 ];
        for ( SI i = 0; i < n; ++i )                     // the gauge, imposed here and maintained by
            w[ i ] -= gauge;                             // `d[ 0 ] = 0` afterwards
        if ( ! already_measured )
            measures_and_facets( w, a, fa );
        t_last = 1;
        nb_evals_last = 1;
        if ( o.after_step ) o.after_step( 0, 0, 1 );

        double eps = 0;
        for ( int it = 0; it < o.maxit; ++it ) {
            double worst = 0, worst_rel = 0;
            SI nb_empty = 0;
            b.assign( n, 0.0 );
            for ( SI i = 0; i < n; ++i ) {
                nb_empty += ! ( a[ i ] > 0 );
                worst = std::max( worst, std::fabs( nu[ i ] - a[ i ] ) );
                worst_rel = std::max( worst_rel, std::fabs( nu[ i ] - a[ i ] ) / nu[ i ] );
                b[ i ] = nu[ i ] - a[ i ];               // `-r`, the right-hand side of Newton
            }
            if ( it == 0 ) {                             // the mass floor of the damping
                double am = a[ 0 ], nm = nu[ 0 ];
                for ( SI i = 0; i < n; ++i ) { am = std::min( am, a[ i ] ); nm = std::min( nm, nu[ i ] ); }
                eps = 0.5 * std::min( nm, am );
                st.eps = eps;
                st.residual0 = worst;
            }
            const double nr = merit( a );
            st.residual = worst;

            if ( worst <= o.tol_abs || ( o.tol_rel > 0 && worst_rel <= o.tol_rel ) ) {
                if ( o.trace )
                    std::printf( "    it %2d  |r|_2 %.3e  max|a-nu| %.3e  CONVERGED\n", it, nr, worst );
                st.status = NewtonStats::CONVERGED;
                return true;
            }
            ++st.nb_iter;
            const int g0 = bal.nb_diag;

            double t0 = now();
            L.assemble( n, fa );
            st.t_asm += now() - t0;
            t0 = now();
            const bool solved = lin.solves( L, b, d );
            st.t_lin += now() - t0;
            if ( ! solved ) {
                st.status = NewtonStats::LINEAR_FAILURE;
                return false;
            }

            // ---- THE TRIAL THEN THE LOCAL LIMITS: the diagram of the step first, and if some cells
            // fall below `eps` there, their limits ( on their own ), the step brought back under the
            // smallest, and we start again -- non-monotonicity may reveal others
            double t = std::min( 1.0, o.mult_ok * t_last );
            bool already = false;                           // the diagram at `t` is already done
            double alpha_lim = -1;
            int nb_evals = 0;
            if ( o.step == NewtonOptions::LIMITS && bounds ) {
                t = beta;
                std::vector<SI> bad_cells;
                w2.resize( n );
                double t_done = -1;                      // the step whose diagram is in `a2`
                for ( int limit_round = 0; limit_round < 8; ++limit_round ) {
                    for ( SI i = 0; i < n; ++i ) w2[ i ] = w[ i ] + t * d[ i ];
                    w2[ 0 ] = 0;
                    measures_and_facets( w2, a2, fa2 );
                    ++nb_evals;
                    t_done = t;
                    bad_cells.clear();
                    for ( SI i = 0; i < n; ++i ) if ( a2[ i ] < eps ) bad_cells.push_back( i );
                    if ( bad_cells.empty() ) break;
                    ++st.nb_limit_rounds;
                    t0 = now();
                    bal.set_weights( w );
                    SI nb_cel = 0;
                    const double al = std::min( t, bounds->alpha_min( w, d, bad_cells, t, eps, L, nb_cel ) );
                    st.nb_cell_lim += nb_cel;
                    st.t_lim += now() - t0;
                    if ( o.trace )
                        std::printf( "      trial t %.3e : %d cells below eps, local limit %.3e ( %lld cells computed )\n",
                                     t, int( bad_cells.size() ), al, ( long long ) nb_cel );
                    t = o.factor * al;
                    if ( t < o.t_min ) break;
                }
                // a zero limit is no reason to stagnate: we hand back to the trials,
                // from half of the last computed step
                if ( t < o.t_min ) t = t_done / 2;
                already = t == t_done;
                alpha_lim = t;
                const bool direct = t >= beta;
                beta = std::min( 1.0, std::max( direct ? o.mult_lim * beta : beta, o.confidence * t ) );
            }

            // ---- THE DAMPING
            bool taken = false;
            const double t_lim0 = t;
            w2.resize( n );
            for ( int trial = 0; trial < o.max_backtracks; ++trial ) {
                if ( ! ( trial == 0 && already ) ) {        // otherwise, already done at `t`
                    for ( SI i = 0; i < n; ++i ) w2[ i ] = w[ i ] + t * d[ i ];
                    w2[ 0 ] = 0;                         // the gauge, imposed and not hoped for
                    measures_and_facets( w2, a2, fa2 );
                    ++nb_evals;
                }
                double m2 = a2[ 0 ];                     // the `eps` floor is an ABSOLUTE mass
                for ( SI i = 0; i < n; ++i ) m2 = std::min( m2, a2[ i ] );
                const double n2r = merit( a2 );
                if ( m2 >= eps && std::isfinite( n2r ) && n2r <= ( 1 - t / 2 ) * nr && n2r < nr ) { taken = true; break; }
                t /= 2;
                ++st.nb_backtracks;
                if ( t < o.t_min )
                    break;
            }
            if ( o.trace ) {
                std::printf( "    it %2d  |r|_2 %.3e  max|a-nu| %.3e  %lld empty  step %.2e  %d diag  [majorant %.2f  diag %.2f  asm %.2f  lin %.2f]",
                             it, nr, worst, ( long long ) nb_empty, t, bal.nb_diag - g0, bal.t_majorant, bal.t_diag, st.t_asm, lin.st.total() );
                if ( alpha_lim >= 0 )
                    std::printf( "  alpha* %.2e%s", alpha_lim, t < t_lim0 ? " REFUSED" : "" );
                std::printf( "\n" );
                std::fflush( stdout );
            }
            if ( ! taken ) {
                bal.set_weights( w );                    // the diagram takes the accepted weights back
                st.status = NewtonStats::STAGNATION;        // the numerical floor, not a failure
                return false;
            }
            t_last = t;
            nb_evals_last = nb_evals;
            w.swap( w2 );
            a.swap( a2 );
            fa.swap( fa2 );
            if ( o.after_step ) o.after_step( it + 1, t, nb_evals );
        }
        // the last point: what it is worth
        double worst = 0;
        for ( SI i = 0; i < n; ++i ) worst = std::max( worst, std::fabs( nu[ i ] - a[ i ] ) );
        st.residual = worst;
        st.status = NewtonStats::MAX_ITERATIONS;
        return false;
    }
};

} // namespace sdotplan
} // namespace sdot
