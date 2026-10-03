"""What TRIGGERS the catalogue's compilations: the standard usage of sdot, exercised once.

    SDOT_CATALOGUE_RECORD=dir python -m sdot.catalogue [--device cpu|cuda]

Each call below goes through `compile_and_register`, which drops the generated source into `dir`
(see `loom.compilation.catalogue`); `scripts/build_catalogue.py` then compiles this record per
variant. A kernel that is not triggered here is not in the catalogue -- it will be compiled on
the user's machine (`auto` mode), or will be missing (`SDOT_KERNELS=catalogue`). The list is
therefore a DECISION: what a standard usage does, in 2D and 3D, in Voronoi and Laguerre, with
both kernel floats, the uniform density and an image, and their derivatives.

Small and fast (a few seconds excluding compilation): it is the SOURCES that matter, not the
sizes.
"""
import argparse

import numpy as np


def provoke( device = None ):
    from loom import driver
    if device:
        driver.device = device
    from sdot import PowerDiagram, Cell, box_half_spaces, Image, SumOfGaussians

    rng = np.random.default_rng( 0 )
    for d in ( 2, 3 ):
        n = 200
        pos = rng.uniform( 0.05, 0.95, size = ( n, d ) )
        w = rng.uniform( -0.3, 0.3, n ) * n ** ( -2.0 / d )
        box = box_half_spaces( [ 0 ] * d, [ 1 ] * d )
        for kernel in ( "FP32", "FP64" ):
            for weights in ( None, w ):
                for acc in ( "plain", None ):
                    pd = PowerDiagram( pos, weights = weights, boundaries = box, accelerator = acc, kernel_dtype = kernel )
                    np.asarray( pd.measures.value )
                    pd.moments
                    pd.hessian_rows()
                    pd.cells
                    # derivatives of the measures: with respect to the positions, then the weights
                    _vjp( lambda p: PowerDiagram( p, weights = weights, boundaries = box, accelerator = acc, kernel_dtype = kernel ).measures, pos )
                    if weights is not None:
                        _vjp( lambda q: PowerDiagram( pos, weights = q, boundaries = box, accelerator = acc, kernel_dtype = kernel ).measures, w )
        # an image density, and a sum of Gaussians
        img = Image( values = rng.uniform( 0.5, 1.5, size = ( 16, ) * d ) )
        pd = PowerDiagram( pos * 16, boundaries = box_half_spaces( [ 0 ] * d, [ 16 ] * d ), distribution = img )
        np.asarray( pd.measures.value )
        pd.moments
        if d == 2:
            sog = SumOfGaussians( positions = pos[ :4 ], sigmas = [ 0.1 ] * 4, weights = [ 1.0 ] * 4 )
            pd = PowerDiagram( pos, boundaries = box, distribution = sog )
            np.asarray( pd.measures.value )
            _vjp( lambda p: PowerDiagram( p, boundaries = box, distribution = sog ).measures, pos )

    # THE TRANSPORT ( `OtProblem.solve` ): one kernel per ( dimension, storage, density ) -- on CPU only,
    # where the solver lives ( `sdot/sdotplan/` ); an image and Gaussians in 2D, an image in 3D
    from sdot import Iterative, OtProblem, SumOfDiracs, Tuning
    if driver.device.is_cpu:
        for d in ( 2, 3 ):
            pos = rng.uniform( 0.05, 0.95, size = ( 100, d ) )
            img = Image( values = rng.uniform( 0.5, 1.5, size = ( 8, ) * d ), origin = [ 0.0 ] * d, frame = ( np.eye( d ) / 8 ).tolist() )
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
        np.asarray( c.measure.value )


def _vjp( f, x ):
    """A backward pass, to trigger the `bwd` kernel of `f` (same path as `check_grad`)."""
    from loom import driver
    probe = f( x )
    dense_shape = tuple( probe.shape )
    crop = lambda t: t.raw[ tuple( slice( 0, s ) for s in dense_shape ) ]
    out, pullback = driver.vjp( lambda v: crop( f( v ) ), x )
    pullback( driver.random( out.shape, seed = 0 ) )


def main( argv = None ):
    p = argparse.ArgumentParser( description = __doc__.splitlines()[ 0 ] )
    p.add_argument( "--device", default = None )
    a = p.parse_args( argv )
    provoke( a.device )
    print( "catalogue: triggered" )


if __name__ == "__main__":
    main()
