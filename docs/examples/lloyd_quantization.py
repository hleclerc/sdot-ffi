"""The outputs of `lloyd-quantization.md`: one animated page per density, the diracs at each Lloyd
iteration ( `docs/public/examples/lloyd-*.html` ).

    python docs/examples/lloyd_quantization.py [ pentagon gaussians-2d gaussians-3d potato ]
"""
import sys
from pathlib import Path

import numpy as np
import sdot

OUT = Path( __file__ ).resolve().parents[ 1 ] / "public" / "examples"
NB_FRAMES = 40


def quantize( name, target, n, background = None, color = None ):
    """Runs `sdot.quantization` with `n` diracs, records the positions at each iteration, and writes a
    page whose frames are ( at most `NB_FRAMES` of ) these iterations."""
    history = []
    x = sdot.quantization( target, n, seed = 0, callback = lambda it, pos, plan: history.append( np.asarray( pos ) ) )
    history.append( np.asarray( x ) )
    print( f"{ name }: { len( history ) - 1 } iterations" )

    # Lloyd moves a lot at first, then less and less: frames evenly spaced on a log scale
    kept = np.unique( np.geomspace( 1, len( history ), NB_FRAMES ).astype( int ) - 1 )
    viz = sdot.Visualizer( title = f"Lloyd quantization: { name }", background = background, frame_axis = "iteration" )
    for k, it in enumerate( kept ):
        if k:
            viz.new_frame( int( it ) )
        viz.add_points( history[ it ], color = color )
    viz.write_html( OUT / f"lloyd-{ name }.html" )
    return x


def pentagon( n = 1000 ):
    angles = np.pi / 2 + 2 * np.pi * np.arange( 5 ) / 5
    target = sdot.Polygon( np.stack( [ np.cos( angles ), np.sin( angles ) ], axis = 1 ) )
    return quantize( "pentagon", target, n )


def three_gaussians( d, n ):
    centers = np.array( [ [ -0.5, -0.3, -0.2 ], [ 0.5, -0.2, 0.1 ], [ 0.0, 0.5, 0.3 ] ] )[ :, :d ]
    target = sdot.SumOfGaussians( centers, sigmas = np.array( [ 0.25, 0.15, 0.2 ] ), weights = np.array( [ 1.0, 0.6, 0.8 ] ),
                                  support_sigmas = 3 )
    return quantize( f"gaussians-{ d }d", target, n )


def potato( n = 4000 ):
    # the density: the brightness of the photo ( the background is dense, the shadow almost empty )
    from PIL import Image
    img = Image.open( OUT / "img" / "potato.png" ).convert( "RGBA" )
    img = Image.alpha_composite( Image.new( "RGBA", img.size, "white" ), img ).convert( "L" )
    img = img.resize( ( img.width // 4, img.height // 4 ), Image.Resampling.LANCZOS )
    values = np.ascontiguousarray( np.asarray( img, dtype = float ).T[ :, ::-1 ] ) + 1.0     # [ x, y ], y upwards; never empty
    return quantize( "potato", sdot.Image( values ), n, background = "#000000", color = "#ffffff" )


CASES = {
    "pentagon"    : pentagon,
    "gaussians-2d": lambda: three_gaussians( 2, 2000 ),
    "gaussians-3d": lambda: three_gaussians( 3, 4000 ),
    "potato"      : potato,
}

if __name__ == "__main__":
    for name in sys.argv[ 1: ] or CASES:
        CASES[ name ]()
