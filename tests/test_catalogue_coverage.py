from errand import test

# The precompiled kernels are built for the sizes of `sdot.catalogue.provoke` (a few hundred seeds). A user's sizes are other: the catalogue
# must serve them all. On a card the layout of a buffer is padded to the alignment of the device, and its extents and byte strides used to be
# LITERALS of the generated source -- every size was another kernel, and a size the catalogue was not built with had to be compiled.
#
# This test asks for sizes `provoke` never used, with `LOOM_KERNELS=catalogue` (a kernel that is not there is an error, nothing compiles).
# It skips when the machine has no catalogue for its architecture (a checkout that has not built one).

if test( "sizes_the_catalogue_was_not_built_with_are_served_by_it" ):
    import os
    import numpy as np
    import loom
    from loom.compilation import catalogue as loom_catalogue
    from errand import skip
    import sdot
    from sdot import PowerDiagram, box_half_spaces, set_kernel_dtype

    tag = loom.resolved_device().current_catalogue_tag()
    if tag not in loom_catalogue.registered_tags():
        skip( f"no catalogue for { tag } here", hint = "python -m sdot.catalogue builds one" )

    previous = os.environ.get( "LOOM_KERNELS" )
    os.environ[ "LOOM_KERNELS" ] = "catalogue"
    try:
        rng = np.random.default_rng( 7 )
        for d in ( 2, 3 ):
            box = box_half_spaces( [ 0 ] * d, [ 1 ] * d )
            for n in ( 77, 1234 ):
                pos = rng.uniform( 0.05, 0.95, size = ( n, d ) )
                for accelerator in ( "plain", None ):
                    for kernel in ( "FP32", "FP64" ):
                        pd = PowerDiagram( pos, boundaries = box, accelerator = accelerator, kernel_dtype = kernel )
                        mass = np.asarray( loom.to_numpy( pd.measures.value ) )
                        assert abs( mass.sum() - 1.0 ) < 1e-3, f"d={ d } n={ n } { accelerator } { kernel }: the cells do not tile the box ({ mass.sum() })"
                        pd.moments
    finally:
        if previous is None:
            os.environ.pop( "LOOM_KERNELS", None )
        else:
            os.environ[ "LOOM_KERNELS" ] = previous
