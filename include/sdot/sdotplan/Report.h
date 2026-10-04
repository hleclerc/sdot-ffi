#pragma once

// WHAT A SOLVE REPORTS, the same for the CPU solver ( `Solve.h` ) and the card's ( `gpu/Newton2D.cuh` ): the layout of
// `stats` and of the rows of `history`, which `SdotPlanNd._STATS` / `_HISTORY` read by position.

namespace sdot {
namespace sdotplan {

/// what the caller reads in `stats( . )` -- same list on the python side ( `SdotPlanNd._STATS` ). `IT_DOUBLE`: the card's mixed
/// precision, the iteration where the double kernel took over ( -1: never; 0 on the CPU and for a single kernel )
enum Stat : int {
    STATUS = 0, RESIDUAL, RESIDUAL0, NB_ITER, NB_DIAG, NB_BACKTRACKS, T_MAJORANT, T_DIAG, T_ASM, T_LIN, T_LIM, EPS,
    DOMAIN_MASS, NB_OVERFLOWED, NB_CELL_LIM, NB_LIMIT_ROUNDS, LIN_NB_HIERARCHIES, LIN_NB_ITER, LIN_WORST, START, T_TOTAL,
    NB_CONTINUATION_STEPS, MIN_START_MASS, IT_SWITCH, IT_DOUBLE,
    NB_STATS
};
enum Start : int { START_GIVEN = 0, START_VORONOI = 1, START_SIMILARITY = 2 };

/// what each row of the history carries -- same list on the python side ( `SdotPlanNd._HISTORY` )
enum Hist : int { H_STEP = 0, H_T, H_RESIDUAL_L2, H_MIN_MASS, H_MAX_RESIDUAL, H_NB_DIAG, H_NB_EVALS, H_S, NB_HIST };

/// why a solve stopped ( `NewtonStats::Status` on the CPU, the same numbers ); `CAPACITY` is the card's: a capacity
/// of the call was too small, loom runs the call again with more ( eagerly ) or raises ( under a trace )
enum SolveStatus : int { S_RUNNING = 0, S_CONVERGED = 1, S_MAX_ITERATIONS = 2, S_STAGNATION = 3, S_LINEAR_FAILURE = 4, S_CAPACITY = 5, S_FAILURE = 6 };

} // namespace sdotplan
} // namespace sdot
