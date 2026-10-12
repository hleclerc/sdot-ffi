"""What PRODUCES the catalogue of sdot:  python -m sdot.catalogue

The standard usage of sdot (`provoke`), with the loops to time (`loom.bench_loop`) where the architecture of a kernel is a
choice; the rest -- the loops on the hardware, the collection of the kernels, the timings, the choices, what to leave out of
the wheel -- is `loom.catalogue.build`.

A kernel that is not triggered here is not in the catalogue: it will be compiled on the user's machine (`auto` mode), or
will be missing (`LOOM_KERNELS=catalogue`). The list is therefore a DECISION: what a standard usage does, in 2D and 3D, in
Voronoi and Laguerre, with both kernel floats, the uniform density and an image, and their derivatives.
"""
import numpy as np


def provoke():
    import loom
    from loom.benchmark import synchronise as wait
    from sdot import PowerDiagram, Cell, box_half_spaces, Image, SumOfGaussians

    rng = np.random.default_rng( 0 )
    own = _own_arrays()
    for d in ( 2, 3 ):
        n = 200
        pos = own( rng.uniform( 0.05, 0.95, size = ( n, d ) ) )
        w = own( rng.uniform( -0.3, 0.3, n ) * n ** ( -2.0 / d ) )
        box = box_half_spaces( [ 0 ] * d, [ 1 ] * d )
        for kernel in ( "FP32", "FP64" ):
            for weights in ( None, w ):
                for acc in ( "plain", None ):
                    pd = PowerDiagram( pos, weights = weights, boundaries = box, accelerator = acc, kernel_dtype = kernel )
                    wait( pd.measures.value )
                    pd.moments
                    pd.hessian_rows()
                    pd.cells
                    # derivatives of the measures: with respect to the positions, then the weights
                    _vjp( lambda p: PowerDiagram( p, weights = weights, boundaries = box, accelerator = acc, kernel_dtype = kernel ).measures, pos )
                    if weights is not None:
                        _vjp( lambda q: PowerDiagram( pos, weights = q, boundaries = box, accelerator = acc, kernel_dtype = kernel ).measures, w )
        # an image density, and a sum of Gaussians
        img = Image( values = own( rng.uniform( 0.5, 1.5, size = ( 16, ) * d ) ) )
        pd = PowerDiagram( pos * 16, boundaries = box_half_spaces( [ 0 ] * d, [ 16 ] * d ), distribution = img )
        wait( pd.measures.value )
        pd.moments
        if d == 2:
            sog = SumOfGaussians( positions = pos[ :4 ], sigmas = [ 0.1 ] * 4, weights = [ 1.0 ] * 4 )
            pd = PowerDiagram( pos, boundaries = box, distribution = sog )
            wait( pd.measures.value )
            _vjp( lambda p: PowerDiagram( p, boundaries = box, distribution = sog ).measures, pos )

    # THE TRANSPORT ( `OtProblem.solve` ): one kernel per ( dimension, storage, density ) -- on CPU only,
    # where the solver lives ( `sdot/sdotplan/` ); an image and Gaussians in 2D, an image in 3D
    from sdot import Iterative, OtProblem, SumOfDiracs, Tuning
    if loom.resolved_device().is_cpu:
        for d in ( 2, 3 ):
            pos = own( rng.uniform( 0.05, 0.95, size = ( 100, d ) ) )
            img = Image( values = own( rng.uniform( 0.5, 1.5, size = ( 8, ) * d ) ), origin = [ 0.0 ] * d, frame = ( np.eye( d ) / 8 ).tolist() )
            for acc in ( None, "plain" ):
                plan = OtProblem( SumOfDiracs( pos ), img ).solve( Iterative( max_iter = 20, tuning = Tuning( accelerator = acc ) ) )
                plan.cost_and_position_grad()
            if d == 2:
                sog = SumOfGaussians( positions = pos[ :4 ], sigmas = [ 0.1 ] * 4, weights = [ 1.0 ] * 4 )
                OtProblem( SumOfDiracs( pos ), sog ).solve( Iterative( max_iter = 20 ) )

    # the cell alone, what `Cell` exposes
    for d in ( 2, 3 ):
        c = Cell.make_hypercube( d, [ 0 ] * d, np.eye( d ).tolist() )
        c.cut( [ 1.0 ] * d, 1.0 )
        wait( c.measure.value )

    timed()


def timed():
    """The kernels whose architecture is a CHOICE: run at a size where they dominate, timed on every architecture the
    machine can behave as (`loom.bench_loop`; a plain loop that runs once when nothing is being built)."""
    from sdot import PowerDiagram, Image, SdotPlan1d, SumOfDiracs1d, box_half_spaces
    import loom
    from loom.benchmark import synchronise as wait
    rng = np.random.default_rng( 0 )
    n = 200000
    own = _own_arrays()
    pos = own( rng.uniform( 0.05, 0.95, size = ( n, 3 ) ) )
    box = box_half_spaces( [ 0 ] * 3, [ 1 ] * 3 )
    for _ in loom.bench_loop( "power_diagram_measures" ):
        wait( PowerDiagram( pos, boundaries = box, accelerator = None ).measures.value )
    for _ in loom.bench_loop( "power_diagram_moments" ):
        PowerDiagram( pos, boundaries = box, accelerator = None ).moments

    m = 1000000
    src = SumOfDiracs1d( positions = own( np.sort( rng.uniform( 0, 1, m ) ) ) )
    img = Image( values = own( rng.uniform( 0.5, 1.5, 20000 ) ) )
    for _ in loom.bench_loop( "update_outputs_SdotPlan1d" ):
        float( SdotPlan1d( src, img ).cost )


def _own_arrays():
    """The arrays of the framework in force (Torch does not mix its tensors with numpy's; Jax does)."""
    from loom.drivers import framework_defaults
    import loom
    return ( lambda a: a ) if framework_defaults.framework_name() == "jax" else loom.array


def _vjp( f, x ):
    """A backward pass, to trigger the `bwd` kernel of `f` (same path as `check_grad`). A framework without autodiff (numpy, cupy)
    has none, hence no such kernel."""
    import loom
    probe = f( x )
    dense_shape = tuple( probe.shape )
    crop = lambda t: t.raw[ tuple( slice( 0, s ) for s in dense_shape ) ]
    try:
        out, pullback = loom.vjp( lambda v: crop( f( v ) ), x )
    except NotImplementedError:
        return
    pullback( loom.random( out.shape, seed = 0 ) )


if __name__ == "__main__":
    import os
    import loom
    if os.environ.get( "LOOM_FRAMEWORK" ) == "torch":
        # torch on the card is a known failure ( the pointer FFI on a GPU ): its kernels are collected for the CPU only
        from loom.devices import hardware
        every = hardware.runs
        hardware.runs = lambda: [ r for r in every() if r.device == "cpu" ]
    loom.catalogue.build( "sdot", provoke )
