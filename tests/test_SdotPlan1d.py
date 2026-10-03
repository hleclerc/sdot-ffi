import numpy

from loom import driver
from sdot import Image, SdotPlan1d, SumOfDiracs1d
from loom.devices import Cpu
from errand import test
from loom.testing import need
from loom.util import info
from loom.testing import check_grad

# driver.ftype = "FP64"


if test( "basic" ):
    src = SumOfDiracs1d( positions = [ 0, 1 ] )
    dst = Image( values = [ 1, 0, 1 ] )

    otp = SdotPlan1d( src, dst, with_barycenters = True )

    info( otp.cost )
    info( otp.barycenters )

if test( "cost_uniform" ):
    # A single dirac (mass 1) against a uniform density on [0,1] (mass 1) : the cost is
    # exactly Integral_0^1 (x - 0.5)^2 dx = 1/12, and the barycenter of the target slice is 0.5.
    # This case also crosses the `udp_cont` bounds guard (a single cell, loop not entered).
    otp = SdotPlan1d( SumOfDiracs1d( positions = [ 0.5 ] ), Image( values = [ 1 ] ), with_barycenters = True )

    assert abs( float( otp.cost ) - 1 / 12 ) < 1e-6
    assert abs( float( otp.barycenters.sum() ) - 0.5 ) < 1e-6

if test( "cost_two_cells" ):
    # Uniform density 0.5 on [0,2] (mass 1), a dirac at 1.0 : the draw crosses TWO
    # cells (the `while` loop of `udp_cont` runs once), expected cost 1/3.
    otp = SdotPlan1d( SumOfDiracs1d( positions = [ 1.0 ] ), Image( values = [ 1, 1 ] ) )

    assert abs( float( otp.cost ) - 1 / 3 ) < 1e-6

if test( "grad_cost" ):
    need( "grad" )
    # Derivative of `cost` with respect to the dirac positions (via `update_outputs_bwd`). Positions
    # well separated so that the `check_grad` perturbation changes neither the sorted order nor
    # the slice assignment ; the cost is then smooth and the adjoint 2 w_i ( p_i - b_i ) exact.
    positions = driver.array( [ 0.2, 0.5, 0.9 ] )

    check_grad( lambda p: SdotPlan1d( SumOfDiracs1d( positions = p ), Image( values = [ 1, 0, 1 ] ) ).cost, positions )

if test( "grad_values" ):
    need( "grad" )
    # Derivative of `cost` with respect to the image values : direct term Integral (x-p)^2 + boundary
    # term -Phi_k. Strictly positive values and slice boundaries (W = 1/3, 2/3) interior to
    # the central cell -> CDF C1 at these points, hence an exact adjoint.
    values = driver.array( [ 1.0, 3.0, 1.0 ] )
    info( values )

    check_grad( lambda v: SdotPlan1d( SumOfDiracs1d( positions = [ 0.2, 0.5, 0.9 ] ), Image( values = v ) ).cost, values )

if test( "group_size_cooperative" ):
    need( "grad" )
    # Force `local_size > 1` on CPU (never the perf path there -- see `Cpu.group_size`'s docstring --
    # but the only place to cheaply exercise the cooperative code against a `local_size == 1` case it
    # would trivially degenerate around). The SORT stays bit-identical regardless of `local_size` (each
    # pass is stable w.r.t. its own input order, chunk-by-chunk, ascending index -- ties still resolve
    # by original index, exactly like the sequential algorithm). The SWEEP no longer does, by design:
    # `Image::udp_at`'s jump-start reads a `cell_cum_mass`/chunk-weight-prefix built via chunked
    # (order-dependent) floating-point reduction, so results match only within numeric tolerance, not
    # bit-for-bit -- see [[group-cooperative-sort]].
    positions = [ 0.9, 0.1, 0.5, 0.3, 0.7, 0.05, 0.95, 0.42, 0.63, 0.18 ]
    values = [ 1, 2, 0, 3, 1 ]

    otp_ref = SdotPlan1d( SumOfDiracs1d( positions = positions ), Image( values = values ), with_barycenters = True )
    cost_ref = float( otp_ref.cost )
    bary_ref = numpy.asarray( otp_ref.barycenters )

    orig_group_size = Cpu.group_size
    Cpu.group_size = lambda self, **_: 4
    try:
        otp = SdotPlan1d( SumOfDiracs1d( positions = positions ), Image( values = values ), with_barycenters = True )
        assert abs( float( otp.cost ) - cost_ref ) < 1e-9 * max( 1.0, abs( cost_ref ) )
        assert numpy.allclose( numpy.asarray( otp.barycenters ), bary_ref, rtol = 1e-9, atol = 1e-9 )

        weights = driver.array( [ 1.0 ] * len( positions ) )
        check_grad( lambda w: SdotPlan1d( SumOfDiracs1d( positions = positions, weights = w ), Image( values = values ) ).cost, weights )
    finally:
        Cpu.group_size = orig_group_size

if test( "zero_density_cells" ):
    need( "grad" )
    # Zero-density cells at the head AND at the tail -- directly targets the non-trivial case of
    # `Image::udp_at` : a limit falling EXACTLY on a zero cumulative mass must land on the
    # FIRST cell that shares this value (smallest-`c` rule), not the last -- otherwise
    # `udp_start()`'s behavior (never an early advance onto a zero head cell) would not be
    # reproduced, and a zero tail cell would never receive its piece (losing its direct
    # contribution to `grad_values`, nonzero even at zero density -- see `second_moment_about`).
    positions = [ 0.1, 0.3, 0.5, 0.7, 0.9 ]

    for values in ( [ 0, 0, 1, 2, 1 ], [ 1, 2, 1, 0, 0 ] ):
        otp_ref = SdotPlan1d( SumOfDiracs1d( positions = positions ), Image( values = values ) )
        cost_ref = float( otp_ref.cost )

        orig_group_size = Cpu.group_size
        Cpu.group_size = lambda self, **_: 4
        try:
            otp = SdotPlan1d( SumOfDiracs1d( positions = positions ), Image( values = values ) )
            assert abs( float( otp.cost ) - cost_ref ) < 1e-9 * max( 1.0, abs( cost_ref ) )

            weights = driver.array( [ 1.0 ] * len( positions ) )
            check_grad( lambda w: SdotPlan1d( SumOfDiracs1d( positions = positions, weights = w ), Image( values = values ) ).cost, weights )
        finally:
            Cpu.group_size = orig_group_size

if test( "single_cell_target" ):
    # `nb_cells (1) < group_size (4)` -- exercise `build_cell_cum_mass`'s cell-chunking when there are
    # FEWER cells than cooperating work-items (most work-items own an empty cell sub-range).
    positions = [ 0.2, 0.5, 0.7, 0.9 ]
    values = [ 1 ]

    otp_ref = SdotPlan1d( SumOfDiracs1d( positions = positions ), Image( values = values ) )
    cost_ref = float( otp_ref.cost )

    orig_group_size = Cpu.group_size
    Cpu.group_size = lambda self, **_: 4
    try:
        otp = SdotPlan1d( SumOfDiracs1d( positions = positions ), Image( values = values ) )
        assert abs( float( otp.cost ) - cost_ref ) < 1e-9 * max( 1.0, abs( cost_ref ) )
    finally:
        Cpu.group_size = orig_group_size

if test( "more_threads_than_diracs" ):
    need( "grad" )
    # `local_size (8) > nb (2)` -- most work-items own an EMPTY dirac chunk (`lo == hi`): exercises
    # `chunked_weight_prefix`/the sweep's per-work-item loop bounds when several chunks never execute.
    positions = [ 0.3, 0.7 ]
    values = [ 1, 2, 1 ]

    otp_ref = SdotPlan1d( SumOfDiracs1d( positions = positions ), Image( values = values ) )
    cost_ref = float( otp_ref.cost )

    orig_group_size = Cpu.group_size
    Cpu.group_size = lambda self, **_: 8
    try:
        otp = SdotPlan1d( SumOfDiracs1d( positions = positions ), Image( values = values ) )
        assert abs( float( otp.cost ) - cost_ref ) < 1e-9 * max( 1.0, abs( cost_ref ) )

        weights = driver.array( [ 1.0, 1.0 ] )
        check_grad( lambda w: SdotPlan1d( SumOfDiracs1d( positions = positions, weights = w ), Image( values = values ) ).cost, weights )
    finally:
        Cpu.group_size = orig_group_size

if test( "boundary_straddling_cell_grad" ):
    need( "grad" )
    # 6 uniformly spread diracs, uniform weights -> at `group_size = 3` (chunks of 2), the chunk
    # limits fall at target mass 1/3 and 2/3 -- STRICTLY INSIDE the two cells of equal
    # mass (cumulative edges 0, 1/2, 1), not on an edge. Directly targets the ATOMIC add of
    # `update_outputs_bwd` on `grad_values` : without it, the cell split between two concurrent
    # work-items would lose one of the two contributions (unprotected write). Note : the CPU
    # work-items run cooperatively (not preemptively) -- this test validates
    # the ARITHMETIC of the phase-1/phase-2/atomic_add path (via finite differences), not the detection of
    # the race itself (which would require a thread sanitizer or a GPU run under stress).
    positions = [ 0.1, 0.2, 0.3, 0.7, 0.8, 0.9 ]
    values = driver.array( [ 1.0, 1.0 ] )

    check_grad( lambda v: SdotPlan1d( SumOfDiracs1d( positions = positions ), Image( values = v ) ).cost, values )

    orig_group_size = Cpu.group_size
    Cpu.group_size = lambda self, **_: 3
    try:
        check_grad( lambda v: SdotPlan1d( SumOfDiracs1d( positions = positions ), Image( values = v ) ).cost, values )
    finally:
        Cpu.group_size = orig_group_size

if test( "grad_weights" ):
    need( "grad" )
    # Derivative of `cost` with respect to the dirac weights : suffix sum of the potential jumps Phi.
    # Positive weights ; the edges stay interior to a cell (smooth cost). The normalization
    # (Python, differentiated by the framework) is traversed end to end by `check_grad`.
    weights = driver.array( [ 1.0, 1.0, 2.0 ] )

    check_grad( lambda w: SdotPlan1d( SumOfDiracs1d( positions = [ 0.2, 0.5, 0.9 ], weights = w ), Image( values = [ 1, 3, 1 ] ) ).cost, weights )

if test( "joint_position_and_weight_value_grad" ):
    need( "grad" )
    # Differentiates positions AND weights AND image values AT THE SAME TIME (barycenters NOT stored,
    # the default) -- the only case where `update_outputs_bwd`'s position-grad block (recompute-b_i path)
    # AND its weights/values-grad block both run in the SAME call, thus exercising
    # the hoisted sort (hoisted `sort_diracs`) shared between the two.
    positions = driver.array( [ 0.2, 0.5, 0.9 ] )
    weights   = driver.array( [ 1.0, 1.0, 2.0 ] )
    values    = driver.array( [ 1.0, 3.0, 1.0 ] )

    check_grad( lambda p, w, v: SdotPlan1d( SumOfDiracs1d( positions = p, weights = w ), Image( values = v ) ).cost,
                positions, weights, values )

    orig_group_size = Cpu.group_size
    Cpu.group_size = lambda self, **_: 4
    try:
        check_grad( lambda p, w, v: SdotPlan1d( SumOfDiracs1d( positions = p, weights = w ), Image( values = v ) ).cost,
                    positions, weights, values )
    finally:
        Cpu.group_size = orig_group_size
