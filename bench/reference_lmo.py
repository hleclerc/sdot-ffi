"""The headline numbers of the old C++ campaign ( `nsdot/solvers_des_familles` ), to compare the python
`sdot` with.

Conditions of ALL of them ( README § 3 ): the machine `lmo` ( Xeon W-2145 ), 8 PINNED threads, kernel
`double`, the MINIMUM over the repetitions, the machine for the bench alone. The README is
`nsdot/solvers_des_familles/README.md`; `source` says which line(s) carry the number ( they were all
re-read when this file was written -- see the NOTES at the bottom for what differs from the summary the
campaign gives of itself ).

This file declares no work ( it does not import errand ).

    diagram_ref( "uniform2d", 10**6 )                 -> { "seconds": 0.207, "ns_per_seed": 207, "source": ..., ... }
    newton_ref( "lines_voronoi", "best" )             -> { "iterations": 12, "diagrams": 19, "seconds": 4.25, ... }

Units: seconds; `ns_per_seed` = nanoseconds per seed, for ONE diagram ( the measure of a power diagram
at fixed weights, the tree already built ).
"""

CONDITIONS = "lmo ( Xeon W-2145 ), 8 pinned threads, kernel double, min over reps, machine alone"


# -- ONE DIAGRAM -------------------------------------------------------------------------------------
# key: ( case, n, memory ) -- `memory`: "none" = the plain tree walk, "memo" = with the neighbour memory
# ( README § 11 ), the form the library has now ( `memory = None` of `PowerDiagram` is the default of the dimension ).
# Several rows for the same key: the campaign improved the same number over time; `diagram_ref` returns
# the LAST one ( the most recent state of the C++ ), `diagram_history` all of them.

DIAGRAM = [
    # § 3, `xmake run diagramme --threads 8` : THE FIRST numbers
    dict( case = "uniform2d",      n = 1_000_000, memory = "none", seconds = 0.207, ns_per_seed = 207,  stage = "first numbers ( § 3 )", source = "README l.128" ),
    dict( case = "lines_voronoi",  n = 100_000,   memory = "none", seconds = 0.021, ns_per_seed = 211,  stage = "first numbers ( § 3 )", source = "README l.129" ),
    dict( case = "lines_equal",    n = 100_000,   memory = "none", seconds = 0.073, ns_per_seed = 733,  stage = "first numbers ( § 3 )", source = "README l.130" ),
    dict( case = "uniform3d",      n = 1_000_000, memory = "none", seconds = 2.768, ns_per_seed = 2768, stage = "first numbers ( § 3 )", source = "README l.131" ),
    dict( case = "planes_voronoi", n = 100_000,   memory = "none", seconds = 0.260, ns_per_seed = 2602, stage = "first numbers ( § 3 )", source = "README l.132" ),
    dict( case = "planes_equal",   n = 100_000,   memory = "none", seconds = 0.460, ns_per_seed = 4603, stage = "first numbers ( § 3 )", source = "README l.133" ),

    # § 11.1, 3D, the neighbour memory ( min of 10 reps, the machine alone )
    dict( case = "uniform3d",      n = 100_000,   memory = "none", seconds = 0.174, ns_per_seed = 1740, stage = "§ 11.1", source = "README l.1626-1627" ),
    dict( case = "uniform3d",      n = 100_000,   memory = "memo", seconds = 0.112, ns_per_seed = 1120, stage = "§ 11.1 ( memory = ranks )", source = "README l.1628" ),
    dict( case = "uniform3d",      n = 100_000,   memory = "memo", seconds = 0.109, ns_per_seed = 1090, stage = "§ 11.4 ( memory = frontier, no stack )", source = "README l.1712" ),
    dict( case = "planes_voronoi", n = 100_000,   memory = "none", seconds = 0.181, ns_per_seed = 1810, stage = "§ 11.1", source = "README l.1630" ),
    dict( case = "planes_voronoi", n = 100_000,   memory = "memo", seconds = 0.115, ns_per_seed = 1150, stage = "§ 11.1", source = "README l.1631" ),
    dict( case = "planes_equal",   n = 100_000,   memory = "none", seconds = 0.324, ns_per_seed = 3240, stage = "§ 11.1", source = "README l.1632" ),
    dict( case = "planes_equal",   n = 100_000,   memory = "memo", seconds = 0.266, ns_per_seed = 2660, stage = "§ 11.1", source = "README l.1633" ),
    dict( case = "uniform3d",      n = 1_000_000, memory = "none", seconds = 1.88,  ns_per_seed = 1880, stage = "§ 11.1", source = "README l.1634" ),
    dict( case = "uniform3d",      n = 1_000_000, memory = "memo", seconds = 1.17,  ns_per_seed = 1170, stage = "§ 11.1", source = "README l.1635" ),

    # § 19.10, after § 19.11 ( vertices solved from their planes ), kernel double, n = 1e6, uniform, isolated diagram
    dict( case = "uniform2d",      n = 1_000_000, memory = "memo", seconds = 0.146, ns_per_seed = 146,  stage = "after § 19.11 ( Voronoi, weights 0 )", source = "README l.4344" ),
    dict( case = "uniform2d",      n = 1_000_000, memory = "memo", seconds = 0.167, ns_per_seed = 167,  stage = "after § 19.11 ( Laguerre, weights ~ h^2 )", source = "README l.4345", weights = "laguerre" ),
    dict( case = "uniform3d",      n = 1_000_000, memory = "memo", seconds = 1.880, ns_per_seed = 1880, stage = "after § 19.11 ( Voronoi )", source = "README l.4346" ),
    dict( case = "uniform3d",      n = 1_000_000, memory = "memo", seconds = 1.921, ns_per_seed = 1921, stage = "after § 19.11 ( Laguerre )", source = "README l.4347", weights = "laguerre" ),
]

#: the same table of § 19.5, `float` and `double`, before / after the `fp32` repairs ( n = 1e6, uniform, ns / seed ):
#: 2D double Voronoi 137 / 145, Laguerre 164 / 163; 3D double Voronoi 1960 / 1946, Laguerre 1964 / 1982
#: ( README l.4132-4133 ); the `float` kernel is 142-168 ns in 2D ( l.4132-4133 ), 160 / 179 after § 19.11 ( l.4344-4345 )
DIAGRAM_FLOAT_NOTE = "kernel float: 2D 160 ( Voronoi ) / 179 ( Laguerre ) ns/seed, 3D 1912 / 1887, after § 19.11 ( README l.4344-4347 )"


def diagram_history( case, n, memory = None, weights = None ):
    """every row of `DIAGRAM` for this case ( `memory`: `"none"` | `"memo"` | `None` for any; `weights`: `"laguerre"` | `None` )"""
    # n within 1 %: the deduplicated lines cloud has 99 944 seeds and the campaign calls it 1e5
    return [ r for r in DIAGRAM
             if r[ "case" ] == case and abs( r[ "n" ] - n ) <= 0.01 * r[ "n" ]
             and ( memory is None or r[ "memory" ] == memory )
             and r.get( "weights" ) == weights ]


def diagram_ref( case, n, memory = None, weights = None ):
    """the most recent row for this case at THIS n ( `None` if the campaign has none: the ratio is then not printed )"""
    rows = diagram_history( case, n, memory, weights )
    return rows[ -1 ] if rows else None


# -- THE NEWTON ---------------------------------------------------------------------------------------
# n = 1e5 ( the file's n ), 8 pinned threads, double; tolerance `max|a - nu| / nu <= 1e-6` ( `--newton-tol`,
# README l.781 of `main_newton.cpp` ), the linear solver of the time ( Cholesky in 2D at this size ).
#
# Variants ( README § 24 ):
#   "kmt"     the damping of Kitagawa-Merigot-Thibert: `t = 1, 1/2, 1/4 ...`, mass floor, decrease of the L2
#             residual by `1 - t / 2` ( `--pas essais --residu lin` )               = `step = "trials"` here
#   "kmt_log" KMT with the `log` residual and the switch to `lin` ( the DEFAULT of the library after § 24.5 )
#   "best"    the limits step + `log` + switch ( `--pas essai-limites --facteur 0.9 --residu log` ), 2D only
#                                                                                      = `step = "limits"` here
#   "model"   the span model, K = 2 ( 2D only, option, not in the library )

NEWTON = [
    # 2D uniform
    dict( case = "uniform2d", variant = "kmt_log", iterations = 6, diagrams = 7, seconds = 1.10, source = "README l.6489 ( `job -b` table )" ),
    dict( case = "uniform2d", variant = "best",    iterations = 6, diagrams = 7, seconds = 1.14, source = "README l.6490 ( 'reference ( limites )' )" ),
    dict( case = "uniform2d", variant = "kmt",     iterations = None, diagrams = 8, seconds = None, source = "README l.5853 ( § 24.5: 8 diagrams before `log`, 7 after )" ),

    # 2D lines / Voronoi ( the DEDUPLICATED cloud, 99 944 seeds )
    dict( case = "lines_voronoi", variant = "kmt",     iterations = 23, diagrams = 78, seconds = 10.94, residual = 2.00e-07, status = "converged", source = "README l.5790" ),
    dict( case = "lines_voronoi", variant = "kmt_log", iterations = 14, diagrams = 39, seconds = 6.44,  residual = 8.23e-07, status = "converged", source = "README l.5792 ( l.6492: 6.40 s )" ),
    dict( case = "lines_voronoi", variant = "best",    iterations = 12, diagrams = 19, seconds = 4.25,  residual = 2.69e-08, status = "converged", source = "README l.5793 ( l.6493: 4.30 s )" ),
    dict( case = "lines_voronoi", variant = "model",   iterations = 12, diagrams = 13, seconds = 6.57,  residual = 8.08e-07, status = "converged", source = "README l.5794" ),

    # 2D lines / equal areas ( the DEGENERATE cloud: 56 coincident pairs, stagnates at 2.35e-6 )
    dict( case = "lines_equal", variant = "kmt",     iterations = 24, diagrams = 113, seconds = 14.21, residual = 2.35e-06, status = "stagnation", source = "README l.5798" ),
    dict( case = "lines_equal", variant = "kmt_log", iterations = 16, diagrams = 74,  seconds = 10.01, residual = 2.35e-06, status = "stagnation", source = "README l.5799" ),
    dict( case = "lines_equal", variant = "best",    iterations = 13, diagrams = 53,  seconds = 7.50,  residual = 2.35e-06, status = "stagnation", source = "README l.5800 ( l.6495: 7.55 s )" ),
    dict( case = "lines_equal", variant = "model",   iterations = 12, diagrams = 13,  seconds = 7.40,  residual = 9.65e-07, status = "converged",   source = "README l.5801" ),

    # 3D uniform
    dict( case = "uniform3d", variant = "kmt",      iterations = 6, diagrams = 9, seconds = 5.2,  source = "README l.145 ( § 3, the FIRST numbers: Cholesky 2D / AMG 3D of the time; l.5810: 3.05 s in 9 diagrams with the later solver )" ),
    dict( case = "uniform3d", variant = "kmt_p025", iterations = None, diagrams = 5, seconds = 1.93, source = "README l.5810 ( the exponent residual `p = 0.25`, not in the library )" ),

    # 3D planes / Voronoi
    dict( case = "planes_voronoi", variant = "kmt",     iterations = 13, diagrams = 27, seconds = 10.48, residual = 1.16e-10, status = "converged", source = "README l.5805" ),
    dict( case = "planes_voronoi", variant = "kmt_p025",iterations = 9,  diagrams = 17, seconds = 7.53,  residual = 1.63e-12, status = "converged", source = "README l.5806" ),
    dict( case = "planes_voronoi", variant = "kmt_log", iterations = 9,  diagrams = 17, seconds = 7.50,  residual = 1.96e-09, status = "converged", source = "README l.5807" ),
    dict( case = "planes_voronoi", variant = "first",   iterations = 13, diagrams = 27, seconds = 16.3, source = "README l.146 ( § 3, the FIRST numbers, before the faster linear solver )" ),
]
# 3D planes / equal volumes: the campaign has no Newton row in its tables ( the file was made by the bench's own Newton ).


def newton_ref( case, variant ):
    """the row of `NEWTON` for this case and variant, or `None`"""
    for r in NEWTON:
        if r[ "case" ] == case and r[ "variant" ] == variant:
            return r
    return None


def newton_variants( case ):
    return [ r for r in NEWTON if r[ "case" ] == case ]


#: the variant the new `step` option answers to: the closest thing in the old campaign ( see `bench/README.md` )
STEP_TO_VARIANT = { "trials": "kmt", "limits": "best" }


def variant_of( step, residual ):
    """the row an ( step, residual ) pair answers to: trials + lin = `kmt`, trials + log / power = `kmt_log`
    ( 3D: `kmt_p025` has the same counts ), limits + log = `best`, limits + lin: no old row ( `kmt` )"""
    if step == "limits":
        return "best" if residual != "lin" else "kmt"
    return "kmt" if residual == "lin" else "kmt_log"


# -- THE DENSITIES ( § 9 gaussians, § 12 image ) ------------------------------------------------------------
# The old bench's own density paths, which are NOT the library's: § 9 is the width continuation of the library ( sqrt( 2 )
# from s = 0.5, the gaussians widened -- the same path ), § 12 a box-blur ( three running means, a B-spline ) continuation
# with an ADAPTIVE scale or a hand-tuned geometric one, the library a gaussian blur and the fixed sqrt( 2 ) ratio. `path`
# says which. `variant`: `kmt` = trials ( `--pas essais` ), `limits` = `--pas essai-limites` ( the mass bisection, § 9.7 ).
# n = 1e5 for § 9 ( `--threads 8`, Cholesky ), 2e4 for § 12 ( 8 threads ); the seeds uniform; tolerance `1e-6` relative.

DENSITY = [
    # § 9, 4 gaussians ( `cases.gauss4` ), the domain the unit square, continuation sqrt( 2 ) from s = 0.5 down to sigma / 4, then 0
    dict( case = "gauss4", sigma = 0.05, n = 100_000, variant = "kmt",    iterations = 72,  diagrams = 123, backtracks = 39,  seconds = 39,
          path = "width continuation sqrt2 from 0.5, previous weights", source = "README l.1254 ( § 9.2 ), l.1448 ( § 9.7: 123 (39) )" ),
    dict( case = "gauss4", sigma = 0.05, n = 100_000, variant = "limits", iterations = None, diagrams = 97, backtracks = 1, seconds = None,
          path = "width continuation sqrt2 from 0.5, mass limits", source = "README l.1448 ( § 9.7: 97 (1) )" ),
    dict( case = "gauss4", sigma = 0.02, n = 100_000, variant = "kmt",    iterations = 139, diagrams = 400, backtracks = 246, seconds = 124,
          path = "width continuation sqrt2 from 0.5, previous weights", source = "README l.1544 ( § 10: 400 (139), 124 s ); l.1266 ( § 9.2: 93 s ), l.1449" ),
    dict( case = "gauss4", sigma = 0.02, n = 100_000, variant = "limits", iterations = 130, diagrams = 204, backtracks = 2,   seconds = 108,
          path = "width continuation sqrt2 from 0.5, mass limits", source = "README l.1545 ( § 10: 204 (130), 108 s ); l.1449 ( § 9.7: 204 (2) )" ),
    dict( case = "gauss4", sigma = 0.01, n = 100_000, variant = "kmt",    iterations = 192, diagrams = 645, backtracks = 436, seconds = 153,
          path = "width continuation sqrt2 from 0.5, previous weights", source = "README l.1267 ( § 9.2 ), l.1450" ),
    dict( case = "gauss4", sigma = 0.01, n = 100_000, variant = "limits", iterations = None, diagrams = 288, backtracks = 3,  seconds = None,
          path = "width continuation sqrt2 from 0.5, mass limits", source = "README l.1450 ( § 9.7: 288 (3) )" ),
    # § 12, the synthetic image 512^2 ( `cases.synthetic_image` ), n = 2e4, `--pas essai-limites`
    dict( case = "image_hole", size = 512, n = 20_000, variant = "limits", iterations = None, diagrams = 100, seconds = 3.9,
          path = "box-blur continuation, hand-tuned geometric scale", source = "README l.1964 ( § 12.6, conv, with hole )" ),
    dict( case = "image_hole", size = 512, n = 20_000, variant = "limits", iterations = None, diagrams = 124, seconds = 4.9,
          path = "box-blur continuation, adaptive ( both criteria )", source = "README l.1966 ( § 12.6, conv, with hole )" ),
    dict( case = "image_hole", size = 512, n = 20_000, variant = "limits", iterations = None, diagrams = 113, seconds = 4.36,
          path = "box-blur continuation, step chooser target 800 ( the best with hole )", source = "README l.2083 ( § 12.6.2 )" ),
    dict( case = "image_hole", size = 512, n = 20_000, variant = "limits_direct", iterations = None, diagrams = 33, seconds = None,
          path = "no continuation: STAGNATION", source = "README l.1963 ( § 12.6: x 33 )" ),
    dict( case = "image", size = 512, n = 20_000, variant = "limits", iterations = None, diagrams = 90, seconds = 3.6,
          path = "box-blur continuation, hand-tuned geometric scale", source = "README l.1964 ( § 12.6, conv, without hole )" ),
    dict( case = "image", size = 512, n = 20_000, variant = "limits", iterations = None, diagrams = 95, seconds = 3.7,
          path = "box-blur continuation, adaptive ( both criteria ): the brief's `--chemin conv --adaptatif`", source = "README l.1966 ( § 12.6 )" ),
    dict( case = "image", size = 512, n = 20_000, variant = "limits", iterations = None, diagrams = 83, seconds = 3.31,
          path = "box-blur continuation, step chooser target 3200 ( the best of the bench )", source = "README l.2085 ( § 12.6.2 )" ),
    dict( case = "image", size = 512, n = 20_000, variant = "limits_direct", iterations = 226, diagrams = 448, backtracks = 2, seconds = 16.5,
          path = "no continuation ( contrast 60:1 ), mass limits", source = "README l.1863 ( § 12.4 )" ),
    dict( case = "image", size = 512, n = 20_000, variant = "kmt_direct", iterations = 100, diagrams = 1041, backtracks = 940, seconds = 12.1,
          path = "no continuation, trials from t = 1: NOT CONVERGED ( residual 18.2 -> 15.5 )", source = "README l.1862 ( § 12.4 )" ),
    # n = 1e5: only the FLOOR path ( `( 1 - t ) + t rho`, 14 geometric steps ), not comparable step by step
    dict( case = "image", size = 512, n = 100_000, variant = "limits", iterations = 397, diagrams = 792, seconds = 188,
          path = "floor continuation ( melange, 14 geometric steps ), Cholesky", source = "README l.3227 / l.3240 ( § 15.17 )" ),
    dict( case = "image", size = 512, n = 100_000, variant = "limits", iterations = None, diagrams = 796, seconds = 90.72,
          path = "floor continuation, AMG ( spai0, 1e-6 )", source = "README l.3457 ( § 17.3 )" ),
]


def density_rows( case, n, sigma = None, size = None ):
    """the rows of `DENSITY` for this case ( `n` within 1 %; the gaussians' `sigma`, the image's `size` )"""
    return [ r for r in DENSITY if r[ "case" ] == case and abs( r[ "n" ] - n ) <= 0.01 * r[ "n" ]
             and ( r.get( "sigma" ) is None or sigma is None or abs( r[ "sigma" ] - sigma ) < 1e-12 )
             and ( r.get( "size" ) is None or size is None or r[ "size" ] == size ) ]


# -- THE LINEAR SOLVER ----------------------------------------------------------------------------------
# totals of a whole Newton solve ( or of one factorization, as said ), n = 1e5, seconds

LINEAR = [
    dict( what = "2D lines Newton, total", solver = "Cholesky ( Eigen LDLT )", seconds = 11.9, source = "README l.166" ),
    dict( what = "2D lines Newton, total", solver = "AMG Ruge-Stuben + GS",      seconds = 15.2, source = "README l.166" ),
    dict( what = "2D lines Newton, total", solver = "AMG aggregation + spai0",   seconds = 17.9, source = "README l.166-167" ),
    dict( what = "2D lines, KMT trials, total", solver = "Cholesky",  seconds = 11.2, iterations = 26, diagrams = 117, source = "README l.344" ),
    dict( what = "2D lines, KMT trials, total", solver = "AMG Ruge-Stuben + GS", seconds = 14.0, iterations = 27, diagrams = 119, source = "README l.345" ),
    # one factorization + solve ( symbolic / numeric + solve / total )
    dict( what = "2D uniform, one factorization", solver = "simplicial ( Eigen LDLT )", symbolic = 0.55, numeric_solve = 1.10, seconds = 2.14, source = "README l.6833" ),
    dict( what = "2D uniform, one factorization", solver = "supernodal CHOLMOD, 1 BLAS thread", symbolic = 0.50, numeric_solve = 0.98, seconds = 1.97, source = "README l.6834" ),
    dict( what = "2D uniform, one factorization", solver = "AMGCL", symbolic = 0.25, numeric_solve = 0.56, seconds = 1.20, source = "README l.6837" ),
    dict( what = "2D lines, one factorization", solver = "simplicial", symbolic = 1.00, numeric_solve = 0.85, seconds = 3.75, source = "README l.6841" ),
    dict( what = "2D lines, one factorization", solver = "AMGCL", symbolic = 0.86, numeric_solve = 1.77, seconds = 4.29, source = "README l.6843" ),
    dict( what = "3D planes, one factorization", solver = "simplicial", symbolic = 3.39, numeric_solve = 806, seconds = 816, source = "README l.6859" ),
    dict( what = "3D planes, one factorization", solver = "supernodal CHOLMOD, 8 BLAS threads", symbolic = 8.91, numeric_solve = 6.17, seconds = 21.3, source = "README l.6861" ),
    dict( what = "3D planes, one factorization", solver = "in-house multigrid ( mg )", symbolic = 1.19, numeric_solve = 1.09, seconds = 8.46, source = "README l.6863" ),
]

# NOTES -- where the first summary of this file's numbers did not match the README, and what was kept
#  * 2D uniform diagram after § 19.11: Voronoi 146 ns/seed ( l.4344 ) -- 137 / 145 in l.4132 is the table of § 19.5
#    ( before / after the fp32 repairs of § 19.4, i.e. BEFORE § 19.11 ), not a range after it.
#  * 2D Laguerre after § 19.11: 167 ns/seed ( l.4345 ) in double; the "175" of l.4434 is the `float` kernel
#    ( 'sommets resolus' row of the fp32-vs-fp64 mass table ), l.4345 gives 179 for float.
#  * 3D uniform 1e6 diagram: 2.768 s ( l.131 ) first; then 1.88 s without memory / 1.17 s with it ( l.1634-1635 );
#    1880 / 1921 ns/seed after § 19.11 ( l.4346-4347 ) -- the 1.88-1.96 us/seed range mixes the Voronoi 1946 / 1960 of l.4132
#    and the 1880 of l.4346.
#  * 2D lines equal, "7.50 s, 53 diagrams" is the BEST variant ( limits + log + switch, 13 iterations ), not KMT:
#    the default KMT is 14.21 s, 24 it, 113 diag ( l.5798 ). 3D planes Voronoi "7.5 s" is KMT + log + switch ( 9 it / 17 diag );
#    the default KMT is 10.48 s ( 13 it / 27 diag ). 3D uniform "5.2 s ( 6 it / 9 diag )" is the first numbers ( l.145 ).
