import numpy

from .Image import Image


class Box:
    """A CONSTANT density on a box -- a parallelepiped -- zero elsewhere.

        rho( x ) = density      if  x = origin + frame^T t  with  t in [ 0, 1 ]^d
                 = 0            otherwise

    The rows of `frame` are the edges of the box ( `frame` can also be a vector of lengths: an axis aligned box ).
    `origin` is `0` and `frame` the identity by default -- the unit cube -- and a default needs the DIMENSION of the space,
    which a `Box()` does not know: it is given where it is known, at the normalization ( `normalized_version( nb_dims )`,
    called by `OtProblem` with the dimension of its source and by `PowerDiagram` with that of its seeds ). Giving `origin` or
    `frame` fixes it earlier, and `nb_dims` says it at once.

    `density`: the value on the box -- meaningless alone, as the distributions are normalized.

    Once the dimension is known a box IS an `Image` of one pixel -- `normalized_version` returns it -- so that everything an
    image enjoys ( the card's solve, the width continuation, 1D ) is kept. Before that it is only a description, and what is
    asked of it that needs a dimension ( `mass`, `bounding_half_spaces`, ... ) is answered by that image."""

    def __init__( self, origin = None, frame = None, density = 1.0, nb_dims = None ):
        self.origin, self.frame, self.density = origin, frame, density
        d = nb_dims
        for a in ( origin, frame ):
            if a is not None:
                shape = numpy.asarray( a ).shape
                if d is not None and int( d ) != int( shape[ 0 ] ):
                    raise ValueError( f"Box : `nb_dims` = { d } but a parameter is of shape { shape }" )
                d = shape[ 0 ]
        self._nb_dims = None if d is None else int( d )
        self._images = {}

    @property
    def nb_dims_or_none( self ):
        """the dimension, or `None` while the box has no default to make up"""
        return self._nb_dims

    def image( self, nb_dims = None ):
        """the `Image` of one pixel that this box is, in dimension `nb_dims` ( or its own )"""
        d = self._nb_dims if self._nb_dims is not None else nb_dims
        if d is None:
            raise ValueError( "Box : the dimension of the space is not known yet ( give `origin`, `frame` or `nb_dims`, "
                              "or use the box where it is known: an `OtProblem`, a `PowerDiagram` )" )
        if self._nb_dims is not None and nb_dims is not None and int( nb_dims ) != self._nb_dims:
            raise ValueError( f"Box : a box of dimension { self._nb_dims } used in dimension { nb_dims }" )
        d = int( d )
        if d not in self._images:
            origin = numpy.zeros( d ) if self.origin is None else numpy.asarray( self.origin, dtype = float ).reshape( d )
            if self.frame is None:
                frame = numpy.eye( d )
            else:
                frame = numpy.asarray( self.frame, dtype = float )
                frame = numpy.diag( frame ) if frame.ndim == 1 else frame.reshape( d, d )
            self._images[ d ] = Image( values = numpy.full( ( 1, ) * d, float( self.density ) ), origin = origin, frame = frame )
        return self._images[ d ]

    def normalized_version( self, nb_dims = None ):
        return self.image( nb_dims ).normalized_version()

    def __getattr__( self, name ):
        # what only an image can answer ( `mass`, `bounding_half_spaces`, `nb_dims`, ... ), once the dimension is known
        if name.startswith( "_" ):
            raise AttributeError( name )
        return getattr( self.image(), name )
