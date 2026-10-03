from .PowerDiagram import PowerDiagram


def Voronoi( positions, **kwargs ):
    """The Euclidean Voronoi diagram of `positions`: a `PowerDiagram` WITHOUT weights.

    This is not a special case that was hard-wired, it is the same object said differently. In the
    plane that separates two seeds, weights enter only through their DIFFERENCE (see
    `cell/Plane.h::bisector`): weights that are all equal move no plane, so "all
    equal" and "no weights" designate the same diagram. Better to carry nothing -- `weights`
    stays `Unbound`, arrives as `NoneTensor` on the C++ side, and the weight term disappears from the kernel at
    COMPILE time. An `[ n ]` of zeros would give exactly the same result while making it read them.

    A function and not a subclass: the subclasses of `PowerDiagram` are its STORAGES
    (`PowerDiagram_Plain`, `PowerDiagram_Bsp`), each naming its C++ structure, and a Voronoi
    is not one more storage. All the arguments of `PowerDiagram` go through, except `weights`
    -- asking for them amounts to asking for a power diagram, and that already has a name.
    """
    if "weights" in kwargs:
        raise TypeError( "a Voronoi diagram carries no weights -- use `PowerDiagram` for that" )
    return PowerDiagram( positions, **kwargs )
