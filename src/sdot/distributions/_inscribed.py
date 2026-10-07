import numpy


def largest_cube( dirs, offs ):
    """The largest axis-aligned CUBE inside the convex polytope `dirs . x <= offs`: `( lo, hi, side )`, or `None` if the
    polytope has no interior. A linear program -- `max r` under `a_i . c + r |a_i|_1 <= b_i` for the centre `c` and the half-side `r`,
    the support of a cube in the direction `a_i` being `a_i . c + r |a_i|_1`."""
    from scipy.optimize import linprog
    A = numpy.asarray( dirs, dtype = float ).reshape( len( offs ), -1 )
    b = numpy.asarray( offs, dtype = float ).reshape( -1 )
    d = A.shape[ 1 ]
    res = linprog( c = numpy.r_[ numpy.zeros( d ), -1.0 ], A_ub = numpy.c_[ A, numpy.abs( A ).sum( axis = 1 ) ], b_ub = b,
                   bounds = [ ( None, None ) ] * d + [ ( 0, None ) ], method = "highs" )
    if not res.success or res.x[ d ] <= 0:
        return None
    centre, r = res.x[ :d ], res.x[ d ]
    return centre - r, centre + r, 2 * r
