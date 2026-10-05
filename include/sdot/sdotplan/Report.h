#pragma once

// WHAT A SOLVE REPORTS, the same for the CPU solver ( `Solve.h` ) and the card's ( `gpu/Newton2D.cuh` ): the layout of
// `stats` and of the rows of `history`, which `SdotPlanNd._STATS` / `_HISTORY` read by position.

namespace sdot {
namespace sdotplan {

/// what the caller reads in `stats( . )` -- same list on the python side ( `SdotPlanNd._STATS` ). `IT_DOUBLE`: the card's mixed
/// precision, the iteration where the double kernel took over ( -1: never; 0 on the CPU and for a single kernel ). `SCRATCH_BYTES`:
/// what the card's solve took from XLA's pool ( 0 on the CPU ). THE AGGREGATION ( `Aggregation.h` ): `NB_CLUSTERS` clusters
/// holding `NB_AGGREGATED` seeds, `NB_DUPLICATES` of them exact duplicates; `RESIDUAL` is the residual the status speaks of
/// ( the aggregated problem's when there are clusters ), `RESIDUAL_FULL` the full problem's ( `max |a_i - nu_i|`, the exact
/// duplicates answered for by their representative ); `NB_POLISH`: the diagrams of the re-splitting ( counted in `NB_DIAG` )
enum Stat : int {
    STATUS = 0, RESIDUAL, RESIDUAL0, NB_ITER, NB_DIAG, NB_BACKTRACKS, T_MAJORANT, T_DIAG, T_ASM, T_LIN, T_LIM, EPS,
    DOMAIN_MASS, NB_OVERFLOWED, NB_CELL_LIM, NB_LIMIT_ROUNDS, LIN_NB_HIERARCHIES, LIN_NB_ITER, LIN_WORST, START, T_TOTAL,
    NB_CONTINUATION_STEPS, MIN_START_MASS, IT_SWITCH, IT_DOUBLE, SCRATCH_BYTES,
    NB_CLUSTERS, NB_AGGREGATED, NB_DUPLICATES, RESIDUAL_FULL, NB_POLISH,
    NB_STATS
};
enum Start : int { START_GIVEN = 0, START_VORONOI = 1, START_SIMILARITY = 2 };

/// what each row of the history carries -- same list on the python side ( `SdotPlanNd._HISTORY` )
enum Hist : int { H_STEP = 0, H_T, H_RESIDUAL_L2, H_MIN_MASS, H_MAX_RESIDUAL, H_NB_DIAG, H_NB_EVALS, H_S, NB_HIST };

/// why a solve stopped ( `NewtonStats::Status` on the CPU, the same numbers ); `CAPACITY` is the card's: a capacity
/// of the call was too small, loom runs the call again with more ( eagerly ) or raises ( under a trace ).
/// `CONVERGED_AGGREGATED`: the aggregated problem passed the test, the full one did not ( its floor: `Aggregation.h` )
enum SolveStatus : int { S_RUNNING = 0, S_CONVERGED = 1, S_MAX_ITERATIONS = 2, S_STAGNATION = 3, S_LINEAR_FAILURE = 4, S_CAPACITY = 5, S_FAILURE = 6,
                         S_CONVERGED_AGGREGATED = 7 };

/// THE DAMPING THAT CAN NO LONGER PASS, the same for both solvers. A trial at `t` is taken when the merit drops by `t / 2`
/// of itself, i.e. when the secant slope `( nr - n2r ) / t` reaches `nr / 2` -- which, as `t` shrinks, tends to the
/// derivative along `d` ( `-nr` for an exact Newton direction ). Two refused trials in a row whose secant slopes agree
/// ( the merit is linear in `t` there ) and whose extrapolation to `t = 0` is under `nr / 4` say that no shorter trial
/// will pass: the residual left is not in the range of the step. That is the floor of the weights' double ( `lines_equal`:
/// two seeds 1e-8 apart, one ulp of their weight moves 5e-6 of their mass; the merit then falls by 2.6 % of itself
/// along `d`, and halving down to `t_min` cost 30 diagrams of a 53-diagram solve ), and we stop there at once, in
/// STAGNATION as before. A trial below the mass floor or not finite says nothing on the slope and resets it.
struct HopelessDamping {
    double t_prev = 0, s_prev = 0;
    bool   has_prev = false;

    /// after a REFUSED trial at `t` of merit `n2r` ( `ok`: above the floor and finite ): `true` to stop halving
    bool refused( double t, double n2r, double nr, bool ok ) {
        if ( ! ok || ! ( nr > 0 ) ) { has_prev = false; return false; }
        const double s = ( nr - n2r ) / t;
        const bool hopeless = has_prev && t_prev == 2 * t && s < nr / 4 && s_prev < nr / 4 &&
                              s - s_prev <= 0.25 * ( s_prev > 0 ? s_prev : - s_prev ) && s_prev - s <= 0.25 * ( s_prev > 0 ? s_prev : - s_prev ) &&
                              2 * s - s_prev < nr / 4;
        t_prev = t;
        s_prev = s;
        has_prev = true;
        return hopeless;
    }
};

} // namespace sdotplan
} // namespace sdot
