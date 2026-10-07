from errand import test

if test( "basic" ):
    # ds = SumOfDiracs( [ 1, 2, 3, 4 ] )
    # op = SdotPlanNd1D( ds, di )
    from sdot import Image # SdotPlanNd1D, SumOfDiracs,
    from loom.testing import check_grad
    di = Image( values = [[ 1, 0, 1 ]] )
    assert float( di.mass ) == 2
