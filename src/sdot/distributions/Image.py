import numpy

import loom
from loom.compilation.FfiCode import FfiCode
import loom
from loom.tensor import Axis, AxisList, CtShapeVar, RealTensor, ShapeVar
from loom.util import ComputedAttribute

from .Distribution import Distribution


class Image( Distribution ):
    """
        Piecewise constant function on a grid.

        Each square/cube/hypercube is defined by `origin` and `frame( dir ) * knots( ... )`

        By default, knots is equal to 0, 1, ... for each dim.
    """

    nb_dims          : CtShapeVar
    shape            : ShapeVar[ "dim" ]

    num_knot         : Axis[ "shape + 1" ]
    img_pos          : AxisList[ "dim", "shape" ]
    dim              : Axis[ "nb_dims" ]
    dir              : Axis[ "nb_dims" ]

    values           : RealTensor[ "img_pos..." ]

    origin           : RealTensor[ "dim" ]
    frame            : RealTensor[ "dir", "dim" ]
    knots            : RealTensor[ "dim", "num_knot" ]

    current_mass     : ComputedAttribute[ RealTensor, ( "values", "frame", "knots" ) ]

    # `nb_cells_cum`/`num_cell_cum`: an INDEPENDENT ShapeVar + Axis pair (not derived from the
    # per-dim `shape`, unlike `num_knot`) -- `cell_cum_mass` is a FLAT array over ALL cells
    # (`nb_pieces + 1`), not expressible as an affine function of the per-dim `shape` the way
    # `num_knot` is (that one stays ragged over `dim`, fine for `knots` but wrong-shaped here). Same
    # two-field shape as `SumOfDiracs`'s `nb_diracs`/`num_dirac`. Prescribed once in
    # `_update_cell_cum_mass`, exactly like `SdotPlan1d`'s own `nb_diracs`/`nb_dims` are prescribed
    # from elsewhere. A DECLARED axis (as opposed to a bare `Tensor`, fine for the truly-scalar
    # `current_mass`) is required so the generated C++ struct's `cell_cum_mass( c )` accepts an
    # index at all -- a bare `Tensor` field only ever gets a RANK-0 call operator.
    nb_cells_cum     : ShapeVar
    num_cell_cum     : Axis[ "nb_cells_cum" ]

    # exclusive prefix sum of each cell's mass ([nb_pieces+1], see `nb_pieces`) -- lets `SdotPlan1d`
    # jump straight into the middle of its sequential walk (`Image::udp_at`) instead of following it
    # there step by step. Depends only on `values`/`frame`/`knots`, so it is CACHED like `current_mass`
    # (`ensure_cell_cum_mass` below) instead of being rebuilt on every `SdotPlan1d` forward/backward
    # call -- previously it was, twice per call, see [[otplan1d-kernel-profile]].
    cell_cum_mass    : ComputedAttribute[ RealTensor[ "num_cell_cum" ], ( "values", "frame", "knots" ) ]

    def __init__( self, values, **kwargs ) -> None:
        self.__base_init__( values = values, target_mass = 1.0, **kwargs )

    def _grid_geometry( self ):
        """`( d, frame, origin, lo, hi )` of the grid, on the HOST side -- or `None` if it is not
        readable here (a tracer under `jit`). Bounding is an optimization: we do without it rather than
        make the call fail."""
        try:
            d = int( self.nb_dims.value )
            shape = numpy.asarray( self.shape.value, dtype = int ).reshape( -1 )
            frame = numpy.asarray( self.frame, dtype = float ).reshape( d, d ) if self.frame.is_defined else numpy.eye( d )
            origin = numpy.asarray( self.origin, dtype = float ).reshape( -1 ) if self.origin.is_defined else numpy.zeros( d )
            if self.knots.is_defined:
                knots = numpy.asarray( self.knots, dtype = float ).reshape( d, -1 )
                lo = knots[ :, 0 ]
                hi = numpy.array( [ knots[ a, shape[ a ] ] for a in range( d ) ] )
            else:
                lo, hi = numpy.zeros( d ), shape.astype( float )
        except ( TypeError, ValueError ):
            return None
        return d, frame, origin, lo, hi

    def bounding_half_spaces( self ):
        # see `Distribution.bounding_half_spaces`. The support is the block of the grid, written in
        # PHYSICAL coordinates: `t_a = n_a . ( x - origin )` with `n_a` the column `a` of `F^-1`
        # (convention `x = origin + F^T t`, that of `Image::measure`), and the useful band goes from
        # `knots( a, 0 )` to `knots( a, shape_a )`.
        g = self._grid_geometry()
        if g is None:
            return None
        d, frame, origin, lo, hi = g
        nrm = numpy.linalg.inv( frame ).T                        # row `a` = the normal of axis `a`

        sh = nrm @ origin
        return ( numpy.concatenate( [ nrm, -nrm ] ),
                 numpy.concatenate( [ hi + sh, -( lo + sh ) ] ) )

    # ---- the display of the cells where the image has mass ( `PowerDiagram.support_pieces` ) --------------------------------

    def display_blocks( self, threshold = 0.0 ):
        """`( dirs, offs, ids )`: the tiles that have mass -- a value `> threshold * max( values )` -- gathered into BOXES of the
        grid ( see `Distribution.display_blocks` ). On each wall of a box, the tiles behind it are ALL with mass ( the wall is a
        `SEAM` ) or all without ( a `SUPPORT`, the grid's border included ): a wall is never a seam in one place and a border in
        another. Built from the tiles by merging, along one axis at a time, the boxes that follow each other with the same
        section and the same walls on the other axes ( `_merge_along` ) -- which is exactly what keeps the walls uniform -- until
        nothing merges any more. Vectorized: one sort per pass."""
        return self._display( threshold )[ "blocks" ]

    def blocks_of_cells( self, cells, threshold = 0.0 ):
        """`( owners, blocks )` for the cells of `cells` ( see `Distribution.blocks_of_cells` ). On the box of each cell, in grid
        coordinates: a cell whose box only meets tiles with mass is kept whole ( block `-1` ), one that meets none is dropped,
        and the others are cut to each block their box meets ( a block that the box meets and the cell does not gives an
        empty piece, which draws nothing )."""
        from ._display import cell_vertices, boxes_of
        g = self._display( threshold )
        d, shape, knots, label = g[ "d" ], g[ "shape" ], g[ "knots" ], g[ "label" ]

        vp, live, nv = cell_vertices( cells )
        t_lo, t_hi = boxes_of( ( vp - g[ "origin" ] ) @ g[ "nrm" ].T, live )      # in grid coordinates

        # the tiles of the box: `knot_index` of `Image.cxx`, `[ k0, k1 )`
        n = len( nv )
        k0 = numpy.zeros( ( n, d ), int )
        k1 = numpy.zeros( ( n, d ), int )
        for a in range( d ):
            k0[ :, a ] = numpy.clip( numpy.searchsorted( knots[ a ], t_lo[ :, a ], side = "right" ) - 1, 0, shape[ a ] - 1 )
            k1[ :, a ] = numpy.clip( numpy.searchsorted( knots[ a ], t_hi[ :, a ], side = "right" ) - 1, 0, shape[ a ] - 1 ) + 1

        count = _box_count( g[ "sat" ], k0, k1 )
        volume = numpy.prod( k1 - k0, axis = 1 )
        whole = ( nv > 0 ) & ( count == volume )
        mixed = numpy.nonzero( ( nv > 0 ) & ( count > 0 ) & ( count < volume ) )[ 0 ]

        owners = [ numpy.nonzero( whole )[ 0 ] ]
        blocks = [ numpy.full( len( owners[ 0 ] ), -1 ) ]
        for i in mixed:
            box = tuple( slice( k0[ i, a ], k1[ i, a ] ) for a in range( d ) )
            bs = numpy.unique( label[ box ] )
            bs = bs[ bs >= 0 ]
            owners.append( numpy.full( len( bs ), i ) )
            blocks.append( bs )
        return numpy.concatenate( owners ), numpy.concatenate( blocks )

    def _display( self, threshold ):
        """the grid seen by the display, and its blocks -- computed once per `threshold`"""
        cache = self.__dict__.setdefault( "_display_cache", {} )
        key = float( threshold )
        if key in cache:
            return cache[ key ]
        from ..Cell import SEAM, SUPPORT
        from ._display import pack_blocks

        d = int( self.nb_dims.value )
        values = numpy.asarray( self.values, dtype = float )
        shape = values.shape[ -d: ]
        values = values.reshape( shape )
        full = values > key * float( values.max( initial = 0 ) )

        _, frame, origin, _, _ = self._grid_geometry()
        if self.knots.is_defined:
            kn = numpy.asarray( self.knots, dtype = float ).reshape( d, -1 )
            knots = [ kn[ a, : shape[ a ] + 1 ] for a in range( d ) ]
        else:
            knots = [ numpy.arange( shape[ a ] + 1, dtype = float ) for a in range( d ) ]
        nrm = numpy.linalg.inv( frame ).T                        # row `a` = the normal of axis `a`
        shift = nrm @ origin

        sat = numpy.zeros( tuple( s + 1 for s in shape ), numpy.int64 )
        sat[ ( slice( 1, None ), ) * d ] = full
        for a in range( d ):
            sat = numpy.cumsum( sat, axis = a )

        # the boxes: the tiles with mass, then merged along each axis while that changes something
        lo = numpy.argwhere( full )
        hi = lo + 1
        pad = numpy.pad( full, 1 )
        ids = numpy.empty( ( len( lo ), d, 2 ), numpy.int32 )                # [ :, a, 0 ]: the low wall, [ :, a, 1 ]: the high one
        for a in range( d ):
            for side, step in ( ( 0, -1 ), ( 1, 1 ) ):
                nb = lo + 1
                nb[ :, a ] += step
                ids[ :, a, side ] = numpy.where( pad[ tuple( nb.T ) ], SEAM, SUPPORT )
        while True:
            m = len( lo )
            for a in range( d ):
                lo, hi, ids = _merge_along( lo, hi, ids, a )
            if len( lo ) == m:
                break
        label = numpy.full( shape, -1, numpy.int64 )
        for b in range( len( lo ) ):
            label[ tuple( slice( lo[ b, a ], hi[ b, a ] ) for a in range( d ) ) ] = b

        # the walls, in PHYSICAL coordinates: `t_a = nrm_a . x - shift_a` is the grid coordinate along axis `a`
        B = len( lo )
        dirs = numpy.zeros( ( B, 2 * d, d ) )
        offs = numpy.zeros( ( B, 2 * d ) )
        for a in range( d ):
            dirs[ :, 2 * a ], offs[ :, 2 * a ] = -nrm[ a ], -( knots[ a ][ lo[ :, a ] ] + shift[ a ] )
            dirs[ :, 2 * a + 1 ], offs[ :, 2 * a + 1 ] = nrm[ a ], knots[ a ][ hi[ :, a ] ] + shift[ a ]
        blocks = pack_blocks( dirs, offs, ids.reshape( B, 2 * d ) )

        res = cache[ key ] = dict( d = d, shape = shape, knots = knots, nrm = nrm, origin = origin, sat = sat, label = label,
                                   blocks = blocks, box_lo = lo, box_hi = hi )
        return res

    def extra_cuts_per_piece( self, nb_dims ):
        # a piece is `cell INTERSECT grid block`, and a block is the intersection of 2d
        # half-spaces (see `Image::_for_each_piece`). The cuts of the cell itself are already
        # counted by the capacity of the cell.
        return 2 * nb_dims

    @property
    def nb_pieces( self ):
        """Total flat cell count for the 1D case `SdotPlan1d` consumes (a single `dim`) -- used to
        size `cell_cum_mass` (`nb_pieces + 1`, see `_update_cell_cum_mass`)."""
        return self.shape.static_count()

    def normalized_version( self, nb_dims = None ):
        # update mass
        mass = self.mass

        # normalize
        if self.target_mass.is_defined:
            return Image(
                nb_dims = self.nb_dims.value,
                shape = self.shape.value,

                values = self.target_mass / mass * self.values,

                origin = self.origin,
                frame = self.frame,
                knots = self.knots,

                current_mass = self.target_mass,
                batch_axes = self.batch_axes,
            )

        return self

    def _update_current_mass( self ):
        # res = RealTensor[ tuple( self.batch_axes ) ]()
        loom.ffi_call(
            "mass",
            FfiCode.per_item( code = "outputs.image.current_mass( batch_index ) = outputs.image( batch_index ).measure();",
                ),
            FfiCode.per_item( "outputs.image( batch_index ).measure_bwd( grad_of_outputs.image( batch_index ).values, "
                           "grad_of_outputs.image( batch_index ).current_mass );" ),
            image = loom.out( self, writes = ( "current_mass", ) ),
            has_dynamic_capacity = False,
        )

    def batch_slice( self, index ):
        # see `Distribution.batch_slice`. Mirrors `Sinogram.image( k )`'s existing unbatched
        # slice: `origin`/`frame`/`knots` are shared geometry (common to every angle), only
        # `values` varies -- `Image.__init__` re-infers `shape`/`nb_dims` from the sliced
        # buffer, nothing else needs restating.
        if not self.batch_axes:
            return None
        kwargs = {}
        for name in ( "origin", "frame", "knots" ):
            attr = getattr( self, name )
            if attr.is_defined:
                kwargs[ name ] = attr.value
        return Image( values = self.values.value[ index ], **kwargs )

    def ensure_cell_cum_mass( self ):
        """Materializes `cell_cum_mass` (a lazy `ComputedAttribute`, mirrors `mass`/`current_mass`)
        if not already cached. Called once by `SdotPlan1d` before it reads `dst_dist.cell_cum_mass` --
        a plain method (not a same-named property) because the FIELD itself must keep the name
        `cell_cum_mass` for the C++ struct it crosses the FFI as."""
        if self.cell_cum_mass.is_undefined:
            self._update_cell_cum_mass()

    def _update_cell_cum_mass( self ):
        # `values`/`frame`/`knots` carry no real gradient THROUGH `cell_cum_mass`: it is a routing
        # helper for `SdotPlan1d`'s walk, and `d cost/d values` is already computed there directly (a
        # closed form, `Phi_k`/`second_moment_about`), never via `cell_cum_mass`. `stop_gradient`
        # them going INTO this call so it needs no `backward` at all, and so `cell_cum_mass` itself
        # never carries a gradient trace back to `values` wherever it is read afterwards (once, here
        # -- not at each read site, see `driver.stop_gradient`'s docstring).
        # `origin`/`frame`/`knots` may be unset (`NoneTensor`, defaulted later by `with_defaults`
        # inside `fill_cell_cum_mass`): omitted entirely rather than passed as `None`, so `Image`
        # leaves them unbound instead of trying to `.set( None )` them.
        detached_kwargs = {}
        for name in ( "origin", "frame", "knots" ):
            attr = getattr( self, name )
            if attr.is_defined:
                detached_kwargs[ name ] = loom.ops().stop_gradient( attr.value )
        detached = Image(
            nb_dims = self.nb_dims.value,
            shape = self.shape.value,
            values = loom.ops().stop_gradient( self.values.value ),
            batch_axes = self.batch_axes,
            **detached_kwargs,
        )

        # prescribe `nb_cells_cum` (the ShapeVar behind `cell_cum_mass`'s declared axis, see its
        # docstring) BEFORE binding the buffer below -- `set_raw` does not observe sizes from the
        # buffer it is handed, it trusts the ShapeVar to already carry the right count.
        self.nb_cells_cum = self.nb_pieces + 1

        # a plain OUTPUT tensor (like `Cell.measure`'s `res`), built from its OWN local axis (a
        # free-standing `RealTensor[...]` cannot resolve a string axis name outside an aggregate's
        # scope) -- not a nested `image.cell_cum_mass` write, so `detached` never needs to carry the
        # result back out itself.
        cum_axis = Axis( ShapeVar( self.nb_pieces + 1 ), name = "num_cell_cum" )
        cell_cum_mass = RealTensor[ *self.batch_axes, cum_axis ]()

        loom.ffi_call(
            "cell_cum_mass",
            FfiCode.per_item( code = "inputs.image( batch_index ).fill_cell_cum_mass( outputs.cell_cum_mass( batch_index ) );",
            ),
            image = detached,
            cell_cum_mass = loom.out( cell_cum_mass ),
            has_dynamic_capacity = False,
        )
        self.cell_cum_mass.set_raw( cell_cum_mass.raw )

    def try_update_sdotplan1d( self, plan ):
        """Closed-form, pure-JAX fast path for `SdotPlan1d.update_outputs` when `self` is its
        1D piecewise-constant target: bypasses `loom.ffi_call`/the C++ kernel entirely (ordinary
        JAX autodiff differentiates straight through `_pure_jax_cost1d.cost_1d_ot`), evaluated
        ~1.2x-6x faster than the C++ kernel from n=1e6 through 1e8 diracs (see
        [[pure-jax-otplan1d]]). Declines (returns False, caller falls back to loom.ffi_call)
        when: the backend is not JAX (this path uses `jax.lax.map`/`jnp.argsort` directly, not
        the backend-agnostic `Tensor` algebra), `self` is not 1D, more than one batch axis is
        in play (matches today's ONE `num_angle` axis in production; untested beyond that),
        `plan` wants `barycenters` (not implemented here), or `plan.src_dist` cannot supply
        plain positions/weights (`raw_1d_diracs`, e.g. an unsupported dirac-source type).

        Batched over angles with `jax.lax.map` (sequential), NOT `jax.vmap`: `vmap` batches
        `jnp.argsort`/the fancy-index gathers by materializing a `[ nb_angles * n, ... ]`
        transpose/fusion -- fine at moderate scale, but a genuine OOM at n=1e8 x 5 angles
        (needing >13GiB). `lax.map` processes one angle at a time (no such batched-sort
        buffer) and, measured, is never slower -- often faster -- than `vmap` at every scale
        tried (1e6-1e8), so there is no size threshold to pick between them.
        """
        if plan._with_barycenters:
            return False
        if int( self.nb_dims.value ) != 1:
            return False
        if len( self.batch_axes ) > 1:
            return False
        if loom.resolved_framework().module_name != "jax":
            return False

        diracs = plan.src_dist.raw_1d_diracs()
        if diracs is None:
            return False
        weights, batched_extra, project_fn = diracs

        import jax
        from ._pure_jax_cost1d import cost_1d_ot

        values = self.values.value
        s_min = self.origin.value[ 0 ] if self.origin.is_defined else 0.0
        dw = self.frame.value[ 0, 0 ] if self.frame.is_defined else 1.0

        if self.batch_axes:
            # only the genuinely PER-ANGLE leaves are mapped (`lax.map` requires every leaf of
            # its pytree argument to share the same leading/mapped axis) -- shared (unbatched)
            # `weights`, and `batched_extra`'s own projection inputs, are closed over instead
            # where they are not batched, so nothing is ever materialized for more than one
            # angle at a time (see `raw_1d_diracs`'s docstring on why this matters).
            mapped = { "values": values, **batched_extra }
            if weights.ndim > 1:
                mapped[ "weights" ] = weights

            # `jax.checkpoint`: without it, the backward through `lax.map` still retains every
            # angle's forward residuals at once (confirmed: OOMs at n=1e8 x 5 angles needing
            # >13GiB even though the forward itself, thanks to the lazy `project_fn` above,
            # never materializes more than one angle's data) -- recomputing the forward per
            # angle during the backward instead is what actually bounds peak memory to ONE
            # angle's worth, letting this scale to n=1e8 (measured ~920ms/angle, matched
            # against a single-angle-isolated baseline -- i.e. no residual per-angle overhead).
            @jax.checkpoint
            def body( leaves ):
                w = leaves[ "weights" ] if "weights" in leaves else weights
                extra = { k: leaves[ k ] for k in batched_extra }
                p = project_fn( extra )
                return cost_1d_ot( p, w, leaves[ "values" ], s_min, dw )

            cost = jax.lax.map( body, mapped )
        else:
            cost = cost_1d_ot( project_fn( {} ), weights, values, s_min, dw )

        plan.cost = cost
        return True


def _box_count( sat, lo, hi ):
    """the number of tiles with mass in each box `[ lo, hi )` ( `[ m, d ]` grid indices ), read on the 2^d corners of a
    summed-area table"""
    d = lo.shape[ 1 ]
    res = numpy.zeros( len( lo ), numpy.int64 )
    for corner in range( 2 ** d ):
        up = [ ( corner >> a ) & 1 for a in range( d ) ]
        idx = tuple( numpy.where( up[ a ], hi[ :, a ], lo[ :, a ] ) for a in range( d ) )
        res += ( -1 ) ** ( d - sum( up ) ) * sat[ idx ]
    return res


def _merge_along( lo, hi, ids, a ):
    """the boxes `[ lo, hi )` ( walls `ids [ m, d, 2 ]` ) merged along axis `a`: those with the same extent and the same walls on
    the other axes, that follow each other along `a`, become one -- the walls of the result on the other axes are those
    common walls, so they stay uniform, and along `a` they are the first box's low wall and the last one's high wall"""
    m, d = lo.shape
    if m < 2:
        return lo, hi, ids
    oth = [ b for b in range( d ) if b != a ]
    key = numpy.concatenate( [ lo[ :, oth ], hi[ :, oth ], ids[ :, oth, : ].reshape( m, -1 ) ], axis = 1 )
    order = numpy.lexsort( [ lo[ :, a ] ] + [ key[ :, c ] for c in reversed( range( key.shape[ 1 ] ) ) ] )
    lo, hi, ids, key = lo[ order ], hi[ order ], ids[ order ], key[ order ]
    start = numpy.ones( m, bool )
    start[ 1: ] = ( key[ 1: ] != key[ :-1 ] ).any( axis = 1 ) | ( hi[ :-1, a ] != lo[ 1:, a ] )
    first = numpy.nonzero( start )[ 0 ]
    last = numpy.append( first[ 1: ], m ) - 1
    nlo, nhi, nids = lo[ first ].copy(), hi[ first ].copy(), ids[ first ].copy()
    nhi[ :, a ] = hi[ last, a ]
    nids[ :, a, 1 ] = ids[ last, a, 1 ]
    return nlo, nhi, nids
