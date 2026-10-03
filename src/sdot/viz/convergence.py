"""Writes a standalone HTML page (inline SVG) showing one or more convergence curves
(loss, residual, ...) over the iterations of a solver -- LBFGS today, Newton tomorrow,
the same function for both: it knows nothing about who calls it, only SEQUENCES of
values.

A single page says a lot about a solver: the SLOPE on a log scale tells whether the convergence is
linear (LBFGS -- a straight line) or quadratic (Newton -- a straight line that starts to dive).
Hence the log scale by default on the y axis. SVG rather than the canvas+WebGL of `Visualizer`:
a handful of points per curve (the number of iterations), not millions -- no need for its
base64 machinery.
"""
import numpy as np


#: same tints as `Visualizer.scale_color` (no cross dependency: just the same
#: constant), so that a curve and the diagram it summarizes, if they appear in the
#: same experiment, do not contradict each other on what a color means.
_GOLDEN_STRIDE = 0.6180339887498949


def _color( index ):
    import colorsys
    h = ( index * _GOLDEN_STRIDE ) % 1.0
    r, g, b = colorsys.hsv_to_rgb( h, 0.65, 0.75 )
    return f"rgb({ int( r * 255 ) },{ int( g * 255 ) },{ int( b * 255 ) })"


def _nice_log_ticks( lo, hi ):
    """The powers of 10 covering `[ lo, hi ]` (`lo > 0`), at least two."""
    import math
    a, b = math.floor( math.log10( lo ) ), math.ceil( math.log10( hi ) )
    if a == b:
        a, b = a - 1, b + 1
    return list( range( a, b + 1 ) )


def _nice_linear_ticks( lo, hi, count = 6 ):
    if hi <= lo:
        return [ lo ]
    step = ( hi - lo ) / max( count - 1, 1 )
    return [ lo + k * step for k in range( count ) ]


def write_convergence_html( series, out_path, title = "convergence", xlabel = "iteration",
                            ylabel = "residual", log_y = True ):
    """`series`: `{ name: [ y0, y1, ... ] }` (an implicit abscissa `0, 1, 2, ...`), or
    `{ name: [ ( x0, y0 ), ( x1, y1 ), ... ] }` for an explicit abscissa (non-consecutive
    steps, for instance). Returns the path written.

    `log_y`: log scale on the y axis (the default -- a convergence curve says nothing on a
    linear scale, it is squashed against the axis from the first two steps on). Values `<= 0`
    then have no point: `0` IS the goal, but cannot be placed on a log axis -- only its
    approach can be read, in the slope.
    """
    curves = {}
    for name, ys in series.items():
        pts = [ tuple( p ) if isinstance( p, ( tuple, list ) ) else ( i, p )
               for i, p in enumerate( ys ) ]
        if log_y:
            pts = [ ( x, y ) for x, y in pts if y > 0 ]
        if pts:
            curves[ name ] = pts

    W, H = 760, 420
    ml, mr, mt, mb = 64, 16, 34, 44           # margins: room for the ticks and the title

    all_x = [ x for pts in curves.values() for x, _ in pts ]
    all_y = [ y for pts in curves.values() for _, y in pts ]
    x_lo, x_hi = ( min( all_x ), max( all_x ) ) if all_x else ( 0, 1 )
    y_lo, y_hi = ( min( all_y ), max( all_y ) ) if all_y else ( 1e-12, 1.0 )
    if x_hi <= x_lo:
        x_hi = x_lo + 1
    if log_y:
        y_ticks = _nice_log_ticks( y_lo, y_hi )
        y_lo_l, y_hi_l = y_ticks[ 0 ], y_ticks[ -1 ]
        def to_y( y ):
            return H - mb - ( np.log10( y ) - y_lo_l ) / ( y_hi_l - y_lo_l ) * ( H - mt - mb )
    else:
        if y_hi <= y_lo:
            y_hi = y_lo + 1
        pad = 0.05 * ( y_hi - y_lo )
        y_lo, y_hi = y_lo - pad, y_hi + pad
        y_ticks = _nice_linear_ticks( y_lo, y_hi )
        def to_y( y ):
            return H - mb - ( y - y_lo ) / ( y_hi - y_lo ) * ( H - mt - mb )

    x_ticks = _nice_linear_ticks( x_lo, x_hi )
    def to_x( x ):
        return ml + ( x - x_lo ) / ( x_hi - x_lo ) * ( W - ml - mr )

    svg = [ f'<svg viewBox="0 0 { W } { H }" xmlns="http://www.w3.org/2000/svg" '
           f'font-family="sans-serif" font-size="11">' ]
    svg.append( f'<rect x="0" y="0" width="{ W }" height="{ H }" fill="#ffffff"/>' )
    svg.append( f'<text x="{ W / 2 }" y="18" text-anchor="middle" font-size="14">{ title }</text>' )

    # grid + ticks. The values are formatted WITHOUT a space before `}` -- everything that
    # follows `:` in an f-string is taken LITERALLY as a format specifier, a space there
    # thus becomes PART of the format (`ValueError: Invalid format specifier '.1f '`), unlike
    # a plain name (`{ ml }`) where the space is normal Python.
    for yt in y_ticks:
        y = to_y( 10 ** yt if log_y else yt )
        y_s = f"{ y:.1f}"
        svg.append( f'<line x1="{ ml }" y1="{ y_s }" x2="{ W - mr }" y2="{ y_s }" '
                   f'stroke="#e5e5e5" stroke-width="1"/>' )
        label = f"1e{ yt }" if log_y else f"{ yt:.3g}"
        svg.append( f'<text x="{ ml - 6 }" y="{ y + 3:.1f}" text-anchor="end">{ label }</text>' )
    for xt in x_ticks:
        x_s = f"{ to_x( xt ):.1f}"
        svg.append( f'<line x1="{ x_s }" y1="{ mt }" x2="{ x_s }" y2="{ H - mb }" '
                   f'stroke="#f2f2f2" stroke-width="1"/>' )
        svg.append( f'<text x="{ x_s }" y="{ H - mb + 16 }" text-anchor="middle">{ xt:.3g}</text>' )
    svg.append( f'<text x="{ ( ml + W - mr ) / 2 }" y="{ H - 6 }" text-anchor="middle">{ xlabel }</text>' )
    svg.append( f'<text x="14" y="{ ( mt + H - mb ) / 2 }" text-anchor="middle" '
               f'transform="rotate(-90 14 { ( mt + H - mb ) / 2 })">{ ylabel }</text>' )
    svg.append( f'<rect x="{ ml }" y="{ mt }" width="{ W - ml - mr }" height="{ H - mt - mb }" '
               f'fill="none" stroke="#999"/>' )

    # curves + legend
    legend_y = mt + 4
    for i, ( name, pts ) in enumerate( curves.items() ):
        color = _color( i )
        path = " ".join( f'{ "M" if k == 0 else "L" }{ to_x( x ):.1f},{ to_y( y ):.1f}'
                        for k, ( x, y ) in enumerate( pts ) )
        svg.append( f'<path d="{ path }" fill="none" stroke="{ color }" stroke-width="2"/>' )
        for x, y in pts:
            svg.append( f'<circle cx="{ to_x( x ):.1f}" cy="{ to_y( y ):.1f}" r="2" fill="{ color }"/>' )
        lx = W - mr - 10
        svg.append( f'<circle cx="{ lx - 90 }" cy="{ legend_y }" r="4" fill="{ color }"/>' )
        svg.append( f'<text x="{ lx - 82 }" y="{ legend_y + 4 }">{ name }</text>' )
        legend_y += 16

    svg.append( "</svg>" )

    html = ( f'<!DOCTYPE html><html><head><meta charset="utf-8"><title>{ title }</title></head>'
           f'<body style="margin:0;display:flex;justify-content:center;padding:24px 0;">'
           + "".join( svg ) + "</body></html>" )

    from pathlib import Path
    path = Path( out_path )
    path.parent.mkdir( parents = True, exist_ok = True )
    path.write_text( html )
    print( f"OUTPUT: file://{ path.absolute() }" )
    return path
