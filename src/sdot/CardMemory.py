"""WHAT THE CARD'S SOLVE TAKES FROM THE CARD, and the check made before it starts.

The card's `SdotPlanNd` ( `gpu/Newton2D.cuh` ) takes its buffers DURING its ffi call, from XLA's pool ( loom's `Scratch`,
`ffi::ScratchAllocator` ). When the pool cannot give them, XLA's BFC allocator does not say no at once: it waits ~10 s for
memory that another stream might free ( `AllocatorRetry` ), and only then refuses. loom makes the first refusal of a call
final ( the pool is not asked again: one wait per call, then a `RESOURCE_EXHAUSTED` error that says how much was refused
and what the call had taken ) -- but the clear error should come at once, with what to do. Hence this model: what the
solve takes, per buffer, from the sizes `Newton2D.cuh` asks for ( `card_solve_bytes` ), checked against what the pool
has left before anything is launched ( `check_card_memory` ).

The model follows the C++ take by take ( the comments name the C++ objects ); the only part that depends on the data is
the coarse matrices of the multigrid ( taken while it runs, `CardLinear::csr_room` ), counted per fine unknown as measured
on the uniform cloud. `stats[ "scratch_bytes" ]` is what a solve really took: `test_SdotPlanNd` holds the model to it.
"""
import math
import os

#: the cells of a diagram the float kernel's finish store holds ( `Cell2D.cuh::DEFER_CAP`, `SDOT_CARD_DEFER_CAP` ): past
#: it, the first pass runs by chunks
DEFER_CAP = int( os.environ.get( "SDOT_CARD_DEFER_CAP", "0" ) or 0 ) or 1 << 20
#: vertices per cell kept by the cells for the step ( `Cell2D.cuh::EDGE_CAP` )
EDGE_CAP = 16
#: vertex slots per cell of the finish store ( `Cell2D.cuh::RD` )
RD = 12
#: bytes of a node record ( `Cell2D.cuh::Node< true, TR >`, 16-aligned ) and of the majorants' work per node
#: ( `Majorant2D.cuh::MajStats`, `MajRed` )
NODE_BYTES, MAJ_BYTES = 48, 104 + 24
#: the same in 3D ( `Cell3D.cuh::Node< true, TR >`, `Majorant3D.cuh::MajStatsN< 3 >`, `MajRed` )
NODE_BYTES_3D, MAJ_BYTES_3D = 48, 152 + 24
#: the coarse matrices of the card multigrid ( the smoothed level's `P`, `P^T`, the Galerkin products, with the 25 % room
#: of `csr_room` ): entries per FINE unknown, measured on the uniform cloud ( 1e6 seeds: 86 MB of float entries )
MG_COARSE_ENTRIES = 11.0
#: ... and in 3D ( a denser graph, 15.5 neighbours per seed, but packets of 8: `SdotPlanNd._CARD_MG_PACK` ), measured on the
#: uniform cloud ( 2e5 seeds, float and double levels: 11.2 )
MG_COARSE_ENTRIES_3D = 11.0


def _levels( n, shift, stop ):
    """the sizes of the multigrid's levels ( `CardLinear::prepare` )"""
    sizes, m = [ n ], n
    while m > stop and len( sizes ) < 24:
        m = ( m + ( 1 << shift ) - 1 ) >> shift
        sizes.append( m )
    return sizes


def card_solve_bytes( n, nb_nodes = None, precision = "mixed", linear = "mg", mg_float = True, recycle = 2, shift = 2, smoothed = 1,
                      stop = 64, overflow_bytes = 0, max_iter = 100, keep_weights = False, jitted = False, dim = 2 ):
    """the bytes the card's solve of `n` seeds takes, PER PART: a dict `{ name: bytes }`. `nb_nodes`: the tree's ( `None`: the
    one `AaBsp` builds for `n` ); `precision`: the kernels ( `fp32`, `fp64`, `mixed` -- `auto` is `mixed` on the card );
    `linear`: `mg`, `cg` or `host`; `overflow_bytes`: the fourth pass's slots ( `Cell2D.cuh::Overflow`, given by the caller
    who chose them ). The parts `tree` and `outputs` are the XLA buffers of the calls, the others the solve's scratch.

    `jitted`: the check runs while TRACING, so `memory_stats()` cannot see the program's own buffers, which XLA takes when the
    program starts: its arguments and constants ( the positions and the masses, traced or not ), the masses normalized, the
    weights it returns. The part `jitted program` counts them from the shapes, 40 bytes per seed: measured on the card at
    1e7 seeds ( `memory_analysis()` of the jitted solve: 1355 MB of XLA buffers where `tree` + `outputs` say 1000 ).

    `dim = 3`: the 3D solve ( `Cell3D.cuh`'s cards, no finish store nor edges, denser facets )."""
    from .AaBsp import AaBsp
    from .SdotPlanNd import card_facet_capacity
    n = int( n )
    N = int( nb_nodes ) if nb_nodes is not None else AaBsp.max_nb_nodes_for( n )
    tr = 4 if n <= 2 ** 31 - 9 else 8
    fcap = card_facet_capacity( n, dim )
    float_first = precision in ( "fp32", "mixed", "auto" )
    p = {}
    p[ "overflow slots" ] = int( overflow_bytes )
    if int( dim ) == 3:
        return _card_solve_bytes_3d( p, n, N, tr, fcap, precision, linear, mg_float, recycle, shift, smoothed, stop, max_iter, keep_weights, jitted )
    # the card ( `Card::prepare` ): nodes, packed seeds ( float4 + float2 or double2 + double ), two lists, the finish store
    cells = NODE_BYTES * N + 24 * n + 2 * tr * n
    if float_first:
        m = min( n, DEFER_CAP )
        cells += ( 8 + tr ) * RD * m + 4 * m
    if precision in ( "mixed", "auto" ):
        cells += 24 * n                                  # the double kernel's seeds ( it borrows the rest )
    p[ "cells" ] = cells
    p[ "facets" ] = ( 2 * tr + 8 ) * fcap                # one COO for both slots
    p[ "diagram slots" ] = 2 * ( 8 + EDGE_CAP * tr + 4 ) * n   # measures, edges, their counts
    p[ "majorants" ] = MAJ_BYTES * N
    p[ "newton vectors" ] = 5 * 8 * n                    # nu, w, w2, d, b
    p[ "laplacian" ] = 8 * ( n + 1 ) + 2 * fcap * ( tr + 8 ) + 8 * n + 2 * 8 * ( n + 1 )   # row, col, val, dia; `LapWork`
    # the linear solver ( `CardLinear::prepare` ): the vectors that live within a solve ( the flexible CG's r, z, p, q; the
    # fine level's input, output and diagonal in float ) are carved from the COO, which is free during a solve
    lin = 0
    if linear not in ( "cg", "host" ):
        tv = 4 if mg_float else 8
        lin = 16 * int( recycle ) * n                    # the recycled solutions and their products
        if mg_float:
            lin += 2 * fcap * 4                          # the fine level's values in float
        sizes = _levels( n, int( shift ), int( stop ) )
        lin += 7 * tv * sizes[ 0 ] + sum( ( 14 * tv + 8 ) * m for m in sizes[ 1: ] )
        lin += sum( 8 * ( sizes[ l ] + 1 ) + 8 * ( sizes[ l + 1 ] + 1 ) for l in range( min( int( smoothed ), len( sizes ) - 1 ) ) )
        lin += int( MG_COARSE_ENTRIES * ( 4 + tv ) * n )
    p[ "linear solver" ] = lin
    # the XLA buffers: the tree's tensors ( `AaBsp` ) and the solve's outputs ( and its inputs: the target, the start )
    p[ "tree" ] = 24 * n + ( 32 + 16 + 24 ) * N
    p[ "outputs" ] = ( 8 + 8 + 16 + 8 + 8 + 8 ) * n + 24 * N + ( ( int( max_iter ) + 1 ) * 8 * n if keep_weights else 0 )
    if jitted:
        p[ "jitted program" ] = ( 16 + 8 + 8 + 8 ) * n          # positions, masses, normalized masses, the weights returned
    return p


def _linear_bytes( n, fcap, linear, mg_float, recycle, shift, smoothed, stop, coarse_entries ):
    """the linear solver's part ( `CardLinear::prepare` ), see `card_solve_bytes`"""
    if linear in ( "cg", "host" ):
        return 0
    tv = 4 if mg_float else 8
    lin = 16 * int( recycle ) * n
    if mg_float:
        lin += 2 * fcap * 4
    sizes = _levels( n, int( shift ), int( stop ) )
    lin += 7 * tv * sizes[ 0 ] + sum( ( 14 * tv + 8 ) * m for m in sizes[ 1: ] )
    lin += sum( 8 * ( sizes[ l ] + 1 ) + 8 * ( sizes[ l + 1 ] + 1 ) for l in range( min( int( smoothed ), len( sizes ) - 1 ) ) )
    lin += int( coarse_entries * ( 4 + tv ) * n )
    return lin


def _card_solve_bytes_3d( p, n, N, tr, fcap, precision, linear, mg_float, recycle, shift, smoothed, stop, max_iter, keep_weights, jitted ):
    """`card_solve_bytes` in 3D: the same takes as `Newton2D.cuh::solve` with `Cell3D.cuh`'s cards"""
    # the cards ( `gpu3d::Card::prepare` ): node records, the packed seeds ( 32 bytes, both kernels ), two lists; mixed: the
    # double kernel's seeds ( it borrows the rest ); the moments' card borrows everything
    cells = NODE_BYTES_3D * N + 32 * n + 2 * tr * n
    if precision in ( "mixed", "auto" ):
        cells += 32 * n
    p[ "cells" ] = cells
    p[ "facets" ] = ( 2 * tr + 8 ) * fcap
    p[ "diagram slots" ] = 2 * 8 * n                     # the measures of the two slots ( no edges in 3D )
    p[ "majorants" ] = MAJ_BYTES_3D * N
    p[ "newton vectors" ] = 5 * 8 * n
    p[ "laplacian" ] = 8 * ( n + 1 ) + 2 * fcap * ( tr + 8 ) + 8 * n + 2 * 8 * ( n + 1 )
    p[ "linear solver" ] = _linear_bytes( n, fcap, linear, mg_float, recycle, shift, smoothed, stop, MG_COARSE_ENTRIES_3D )
    p[ "tree" ] = 32 * n + ( 48 + 24 + 8 ) * N
    p[ "outputs" ] = ( 8 + 8 + 24 + 8 + 8 + 8 ) * n + 32 * N + ( ( int( max_iter ) + 1 ) * 8 * n if keep_weights else 0 )
    if jitted:
        p[ "jitted program" ] = ( 24 + 8 + 8 + 8 ) * n
    return p


def card_pool():
    """`( limit, in_use )`: the bytes of XLA's pool on the current card, and those in use ( `None` if they cannot be read: not
    jax, or a pool without statistics )"""
    try:
        import loom
        import jax
        dev_id = int( getattr( loom.resolved_device(), "device_id", 0 ) or 0 )
        devs = [ d for d in jax.devices() if d.platform == "gpu" ]
        dev = next( ( d for d in devs if d.id == dev_id ), devs[ 0 ] if devs else None )
        st = dev.memory_stats() if dev is not None else None
        if not st or "bytes_limit" not in st:
            return None
        return int( st[ "bytes_limit" ] ), int( st.get( "bytes_in_use", 0 ) )
    except Exception:
        return None


def check_card_memory( n, **kw ):
    """raises a `MemoryError` when the card's solve of `n` seeds ( `card_solve_bytes( n, **kw )` ) does not fit in what XLA's
    pool has left -- at once, rather than XLA's ten seconds of waiting followed by a refusal from inside the call. Returns
    the parts ( `None` when the pool cannot be read: then only the call can tell ). `SDOT_CARD_MEMORY_CHECK=0` skips it."""
    if os.environ.get( "SDOT_CARD_MEMORY_CHECK", "1" ) == "0":
        return None
    pool = card_pool()
    if pool is None:
        return None
    limit, in_use = pool
    parts = card_solve_bytes( n, **kw )
    need, free = sum( parts.values() ), limit - in_use
    if need <= free:
        return parts
    # how many seeds would fit ( the model is affine in `n` but for the tree's steps: bisection )
    lo, hi = 0, int( n )
    while hi - lo > max( 1, lo // 100 ):
        mid = ( lo + hi ) // 2
        if sum( card_solve_bytes( mid, **{ **kw, "nb_nodes": None } ).values() ) <= free: lo = mid
        else: hi = mid
    gb = lambda b: f"{ b / 1e9 :.2f} GB"
    top = ", ".join( f"{ k } { v / n :.0f}" for k, v in sorted( parts.items(), key = lambda kv: -kv[ 1 ] )[ :5 ] )
    hints = [ f"solve fewer seeds ( about { lo :.3g} fit )" ]
    if kw.get( "precision", "mixed" ) in ( "mixed", "auto" ):
        fp32 = sum( card_solve_bytes( n, **{ **kw, "precision": "fp32" } ).values() )
        hints.append( f"use precision = 'fp32' ( { gb( fp32 ) }, but the float kernel alone may stagnate on degenerate clouds )" )
    hints += [ f"give XLA more of the card ( XLA_PYTHON_CLIENT_MEM_FRACTION, before jax starts: 0.75 by default; the pool is "
               f"{ gb( limit ) } )", "free the arrays the program keeps on the card", "or solve on the CPU ( LOOM_DEVICE=cpu )" ]
    raise MemoryError( f"SdotPlanNd on the card: the solve of { n } seeds needs about { gb( need ) } of the card ( { need / n :.0f} "
                       f"bytes per seed; the largest parts, bytes per seed: { top } ) and XLA's pool has { gb( free ) } left "
                       f"( { gb( limit ) }, { gb( in_use ) } in use ). " + "; ".join( hints ) +
                       ". ( The estimate: `sdot.CardMemory.card_solve_bytes`; SDOT_CARD_MEMORY_CHECK=0 skips this check, the call "
                       "then fails by itself after XLA's ten seconds of waiting. )" )
