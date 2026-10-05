"""The headline numbers of the old GPU campaign ( `nsdot/gpu_des_familles` ), to compare the python
`sdot` on the card with.

Conditions of ALL of them ( `doc/03-chiffres.md` l.3-6, `doc/07-methode.md` l.3-5 ): the machine `lmo`,
RTX 2080 Ti ( Turing sm_75, 68 SM, FP64 at 1/32 ), CUDA 13.3, `nvcc -O3 -lineinfo`, no fast-math; the
machine alone ( `errand -x` ); 300 ms of kernel in a loop before the first timing, then the MINIMUM of
10 runs; the time is the KERNEL ALONE ( CUDA events around the kernel(s) of the cells ), the upload of
the tree and the download of the measures are reported apart. The tree is built on the CPU and handed
over: it is not in the time.

`source` is relative to `nsdot/gpu_des_familles/`; every number was re-read there when this file was
written. `variant` is the old kernel the number belongs to: the old campaign kept one kernel per idea,
so the BEST of a case is not always the same kernel ( see `best` below ).

    gpu_ref( "uniform2d", 10**6, "float" )             -> the best row ( lowest ns/seed )
    gpu_rows( "uniform2d", 10**6, "float" )            -> all of them

This file declares no work ( it does not import errand ).
"""

CONDITIONS = "lmo RTX 2080 Ti sm_75, CUDA 13.3, kernel only ( CUDA events ), 300 ms warm-up, min of 10, machine alone"

#: ns / seed, ONE diagram, the tree given. `accuracy`: | m_float - m_double | / m_double per cell, against
#: the double CPU witness, when the campaign measured it for that kernel ( median, p99.99, max ).
GPU = [
    # -- 2D, float
    dict( case = "uniform2d", n = 1_000_000, kernel = "float", variant = "filnrm8", ns_per_seed = 7.6,
          regs = 74, blocks_per_sm = 6, occupancy = 0.75, local_bytes = 192,
          accuracy = dict( median = 3.2e-5, p9999 = 7.6e-4, max = 4.8e-3 ),
          source = "doc/05-profils.md l.87 ( 7.6, 74 regs, 6 blocks/SM, 75 % ), doc/03-chiffres.md l.326; accuracy doc/04-echelle.md l.187; local doc/03-chiffres.md l.414 ( 192 B, the stack )" ),
    dict( case = "uniform2d", n = 1_000_000, kernel = "float", variant = "filmsk8f", ns_per_seed = 8.2,
          regs = 63, blocks_per_sm = 8, occupancy = 1.00,
          accuracy = dict( median = 6.8e-6, p9999 = 2.4e-4, max = 1.4e-3 ),
          source = "doc/05-profils.md l.88 ( 8.2, 63 regs, 8 blocks/SM, 100 % ), README.md l.41; accuracy doc/04-echelle.md l.189" ),
    dict( case = "uniform2d", n = 1_000_000, kernel = "float", variant = "filmsk8m", ns_per_seed = 10.6,
          regs = 94,
          accuracy = dict( median = 4.9e-14, p9999 = 4.9e-12, max = 4.6e-8 ),
          source = "doc/04-echelle.md l.385 ( 10.6 ns, 94 regs, median 4.9e-14 / p99.99 4.9e-12 / max 4.6e-08 ); the Newton default, l.374" ),
    dict( case = "lines_voronoi", n = 100_000, kernel = "float", variant = "filnrm8", ns_per_seed = 18.5,
          accuracy = dict( median = 3.4e-6, max = 2.4e-3 ),
          source = "doc/03-chiffres.md l.326; accuracy doc/04-echelle.md l.199" ),
    dict( case = "lines_equal", n = 100_000, kernel = "float", variant = "filnrm8", ns_per_seed = 45.2,
          accuracy = dict( median = 2.0e-3, max = 2.7 ),
          source = "doc/03-chiffres.md l.326 ( 45.2; 47 in the table l.15, filmix6 45.5 l.15 ); accuracy doc/04-echelle.md l.199" ),
    # -- 2D, double
    dict( case = "uniform2d", n = 1_000_000, kernel = "double", variant = "filmsk8g", ns_per_seed = 96.7,
          regs = 96,
          source = "doc/04-echelle.md l.169 ( 96.7 ), l.172 ( 96 registers against 128 ), README.md l.42" ),
    dict( case = "lines_voronoi", n = 100_000, kernel = "double", variant = "filnrm8", ns_per_seed = 132.0,
          regs = 128, blocks_per_sm = 4, occupancy = 0.50, local_bytes = 192,
          source = "doc/03-chiffres.md l.420 ( filnrm8 double: 128 regs, 4 blocks/SM, 50 %, 192 B local )" ),
    dict( case = "lines_equal", n = 100_000, kernel = "double", variant = "filmix6", ns_per_seed = 385,
          source = "doc/03-chiffres.md l.501" ),
    # -- 3D ( one mapping only: `voies`, a cell on the whole warp ( doc/02-mappages.md l.70 ), two passes, `__launch_bounds__( 128, 4 )` )
    dict( case = "uniform3d", n = 1_000_000, kernel = "float", variant = "voies", ns_per_seed = 229,
          regs = 128, blocks_per_sm = 4, occupancy = 0.50,
          source = "doc/03-chiffres.md l.16; doc/05-profils.md l.66-68 ( 128 registers, four blocks per SM; 102 regs spill: 251 )" ),
    dict( case = "planes_voronoi", n = 100_000, kernel = "float", variant = "voies", ns_per_seed = 231,
          source = "doc/03-chiffres.md l.17" ),
    dict( case = "planes_equal", n = 100_000, kernel = "float", variant = "voies", ns_per_seed = 405,
          source = "doc/03-chiffres.md l.18" ),
    dict( case = "uniform3d", n = 1_000_000, kernel = "double", variant = "voies", ns_per_seed = 1103,
          source = "doc/03-chiffres.md l.502" ),
    dict( case = "planes_voronoi", n = 100_000, kernel = "double", variant = "voies", ns_per_seed = 1066,
          source = "doc/03-chiffres.md l.503" ),
    dict( case = "planes_equal", n = 100_000, kernel = "double", variant = "voies", ns_per_seed = 2926,
          source = "doc/03-chiffres.md l.504" ),
]

#: the CPU witness of the same campaign ( 8 pinned threads, same clouds ), ns / seed ( doc/03-chiffres.md l.13-18 float, l.498-504 double )
CPU_WITNESS = {
    ( "uniform2d", "float" ): 145, ( "lines_voronoi", "float" ): 141, ( "lines_equal", "float" ): 696,
    ( "uniform3d", "float" ): 1831, ( "planes_voronoi", "float" ): 1774, ( "planes_equal", "float" ): 3054,
    ( "uniform2d", "double" ): 138, ( "lines_voronoi", "double" ): 137, ( "lines_equal", "double" ): 688,
    ( "uniform3d", "double" ): 1975, ( "planes_voronoi", "double" ): 1908, ( "planes_equal", "double" ): 3302,
}

# NOTES
#  * 2D uniform float: 7.6 ( doc/05-profils.md l.87, the profiled run ) and 7.7 ( doc/03-chiffres.md l.13, l.383, l.416 ) are the
#    same kernel in two sessions; the README keeps 7.6.
#  * 2D lines Voronoi double 132.0 is `filnrm8` ( doc/03-chiffres.md l.420 ); `filmix6` gives 112 in the double table ( l.500 ),
#    which this file does NOT keep as the reference because the brief of this work names 132 -- `gpu_rows` has both.
#  * 2D uniform double: `filmix6` 93 ( doc/03-chiffres.md l.499 ) is below `filmsk8g` 96.7 but is an older table ( before the
#    300 ms warm-up and the noise floor of doc/07-methode.md ); `filph8c` 86.2 ( doc/04-echelle.md l.12 ) is from a `--reps 1 --reps-gpu 5`
#    scaling run. The reference is 96.7, the number the README gives.
#  * The float accuracy of the old engine depends on the variant by 9 orders of magnitude ( filnrm8 3.2e-5 -> filmsk8m 4.9e-14 ): a new
#    float kernel is to be read against BOTH the fast row ( 7.6 ) and the accurate one ( 10.6 ).
GPU += [
    dict( case = "lines_voronoi", n = 100_000, kernel = "double", variant = "filmix6", ns_per_seed = 112,
          source = "doc/03-chiffres.md l.500 ( older double table )", secondary = True ),
    dict( case = "uniform2d", n = 1_000_000, kernel = "double", variant = "filmix6", ns_per_seed = 93,
          source = "doc/03-chiffres.md l.499 ( older double table )", secondary = True ),
]


#: THE DENSITIES on the old card ( `doc/08-densites.md` ): the synthetic image WITHOUT hole ( 90:1, `cases.synthetic_image(
#: hole = False )` ), double, a uniform cloud. One diagram ( measures + facets, kernel only, ns / seed ) and whole Newton solves
#: ( the CONTRAST continuation `( 1 - s ) + s rho` in K uniform steps, CG, trials from the previous step doubled ).
GPU_DENSITY = [
    dict( what = "diagram", case = "uniform2d", n = 1_000_000, image = 0,   ns_per_seed = 96.6,  source = "doc/08-densites.md l.76 ( no image )" ),
    dict( what = "diagram", case = "uniform2d", n = 1_000_000, image = 512, ns_per_seed = 103.9, source = "doc/08-densites.md l.76 ( +7 % )" ),
    dict( what = "diagram", case = "uniform2d", n = 1_000_000, image = 2048, ns_per_seed = 101.6, source = "doc/08-densites.md l.76" ),
    dict( what = "newton", case = "image", n = 200_000, image = 512, steps = 1, iterations = 1341, seconds = 318, source = "doc/08-densites.md l.278 ( rho, K = 1 )" ),
    dict( what = "newton", case = "image", n = 200_000, image = 512, steps = 8, iterations = 662,  seconds = 164, source = "doc/08-densites.md l.278 ( rho, K = 8 )" ),
    dict( what = "newton", case = "lebesgue", n = 200_000, image = 0, steps = 1, iterations = 7,   seconds = 1.0, source = "doc/08-densites.md l.275" ),
]
# ( doc/06-ce-qui-reste.md l.10 repeats the +7 % of the 512^2 image at 1e6; the old card had no gaussian density. )


def gpu_rows( case, n, kernel ):
    """every row for this case / kernel at THIS n ( within 1 %: the deduplicated lines cloud has 99 944 seeds )"""
    return [ r for r in GPU if r[ "case" ] == case and r[ "kernel" ] == kernel and abs( r[ "n" ] - n ) <= 0.01 * r[ "n" ] ]


def gpu_ref( case, n, kernel ):
    """THE reference: the fastest primary row ( `None` if the campaign has none at this n )"""
    rows = [ r for r in gpu_rows( case, n, kernel ) if not r.get( "secondary" ) ]
    return min( rows, key = lambda r: r[ "ns_per_seed" ] ) if rows else None


def gpu_accurate_ref( case, n, kernel ):
    """the accurate float row when there is one ( `filmsk8m` ), else `gpu_ref`"""
    rows = [ r for r in gpu_rows( case, n, kernel ) if r[ "variant" ] == "filmsk8m" ]
    return rows[ 0 ] if rows else gpu_ref( case, n, kernel )
