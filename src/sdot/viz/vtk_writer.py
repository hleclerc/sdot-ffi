"""ParaView output of a `Visualizer`: compressed binary VTK XML.

Three choices that cannot be guessed, and that are here:

- FORMAT. `.vtu` (unstructured grid) in appended binary (`AppendedData encoding="raw"`) and
  zlib-compressed -- what ParaView prefers, and much lighter than ASCII. A `.vtu` accepts
  the three kinds of cell at once: polygons, segments, isolated points.

- TIME. One frame per file, gathered by a `.pvd` that carries the abscissas. This is what
  ParaView expects for a time series whose geometry changes completely from one step to
  the next -- which is our case: neither the vertices nor the connectivity correspond.

- BEYOND 3D. ParaView is 3D. The three chosen dimensions are written as GEOMETRY and the
  other coordinates as POINT DATA (`x3`, `x4`, ...): ParaView can then slice and
  threshold on them itself, which is better than freezing a slice at write time.

A polytope given as half-spaces has no vertices: it is enumerated here, in Python
(`polytope.polytope_mesh`).
"""
import struct
import zlib
from pathlib import Path

import numpy as np

from .polytope import polytope_mesh


VTK_VERTEX, VTK_LINE, VTK_POLYGON = 1, 3, 7

_VTK_DTYPE = {
    "float32": "Float32", "float64": "Float64",
    "int32": "Int32", "int64": "Int64", "uint8": "UInt8",
}


class _Appended:
    """The arrays of the file, laid end to end in the final binary block.

    `add` returns the OFFSET to write in the `DataArray` attribute; `data` returns the whole block.
    Each array is compressed in chunks, with the header that VTK expects:
    `[ nb_chunks, chunk_size, last_size, compressed_size_of_each ]`, all in
    UInt64 (hence the file's `header_type`).
    """
    BLOCK = 32768

    def __init__( self ):
        self._chunks = []
        self._size   = 0

    def add( self, arr ):
        raw = np.ascontiguousarray( arr ).tobytes()
        blob = self._encode( raw )
        offset = self._size
        self._chunks.append( blob )
        self._size += len( blob )
        return offset

    def _encode( self, raw ):
        if not raw:
            return struct.pack( "<Q", 0 )                  # empty array: zero chunks
        parts = [ raw[ i : i + self.BLOCK ] for i in range( 0, len( raw ), self.BLOCK ) ]
        comp  = [ zlib.compress( part, 6 ) for part in parts ]
        head  = struct.pack( f"<{ 3 + len( parts ) }Q", len( parts ), self.BLOCK,
                             len( parts[ -1 ] ), *( len( c ) for c in comp ) )
        return head + b"".join( comp )

    def data( self ):
        return b"".join( self._chunks )


def _array_tag( name, arr, offset, nb_components = 1 ):
    kind = _VTK_DTYPE[ arr.dtype.name ]
    name = f' Name="{ name }"' if name else ""
    return ( f'<DataArray type="{ kind }"{ name } NumberOfComponents="{ nb_components }" '
             f'format="appended" offset="{ offset }"/>' )


def _write_vtu( path, coords, extra, cells, colors, extra_names, radii ):
    """A `.vtu` file.

    `coords`: `[n, 3]` the geometry. `extra`: `[n, k]` the coordinates beyond 3D, named
    by `extra_names`. `cells`: `( types [m], connectivity, offsets )`. `colors`: `[m, 4]` uint8.
    `radii`: `[m]` the WORLD radius of each cell -- that of the points (`add_points( radius )`),
    0 for everything else: enough to make a sphere `Glyph` at their true size in ParaView
    (`Scale Array = radius`, factor 1), where the HTML page has its slider.
    """
    types, conn, offs = cells
    app = _Appended()
    o_pts  = app.add( coords.astype( np.float32 ) )
    o_conn = app.add( conn.astype( np.int64 ) )
    o_offs = app.add( offs.astype( np.int64 ) )
    o_type = app.add( types.astype( np.uint8 ) )
    o_col  = app.add( colors.astype( np.uint8 ) )
    o_rad  = app.add( radii.astype( np.float32 ) )
    o_extra = [ app.add( np.ascontiguousarray( extra[ :, k ], np.float32 ) )
                for k in range( extra.shape[ 1 ] ) ]

    xml = [
        '<?xml version="1.0"?>',
        '<VTKFile type="UnstructuredGrid" version="1.0" byte_order="LittleEndian" '
        'header_type="UInt64" compressor="vtkZLibDataCompressor">',
        '  <UnstructuredGrid>',
        f'    <Piece NumberOfPoints="{ len( coords ) }" NumberOfCells="{ len( types ) }">',
        '      <Points>',
        '        ' + _array_tag( None, coords.astype( np.float32 ), o_pts, 3 ),
        '      </Points>',
        '      <Cells>',
        '        ' + _array_tag( "connectivity", conn.astype( np.int64 ), o_conn ),
        '        ' + _array_tag( "offsets", offs.astype( np.int64 ), o_offs ),
        '        ' + _array_tag( "types", types.astype( np.uint8 ), o_type ),
        '      </Cells>',
        '      <CellData Scalars="RGBA">',
        '        ' + _array_tag( "RGBA", colors.astype( np.uint8 ), o_col, 4 ),
        '        ' + _array_tag( "radius", radii.astype( np.float32 ), o_rad ),
        '      </CellData>',
    ]
    if extra_names:
        xml.append( f'      <PointData Scalars="{ extra_names[ 0 ] }">' )
        for name, off in zip( extra_names, o_extra ):
            xml.append( '        ' + _array_tag( name, np.zeros( 0, np.float32 ), off ) )
        xml.append( '      </PointData>' )
    xml += [
        '    </Piece>',
        '  </UnstructuredGrid>',
        '  <AppendedData encoding="raw">',
        '   _',
    ]
    head = ( "\n".join( xml ) ).encode( "ascii" )
    tail = b"\n  </AppendedData>\n</VTKFile>\n"
    path.write_bytes( head + app.data() + tail )
    return path


def _frame_mesh( viz, index, axes ):
    """Frame `index`, flattened: vertices, VTK cells, colors, radii.

    The vertices are RENUMBERED: the pool carries the whole scene, a file must only contain
    what its frame uses. The POINTS go through no Python loop: a frame of a
    reconstruction carries hundreds of thousands of them, and there is one per step.
    """
    fr   = viz.frame( index )
    pool = viz.positions
    d    = viz.nb_dims
    colors = np.array( viz.colors, np.float64 ).reshape( -1, 4 )

    polys = [ list( f ) for f in fr[ "polygons" ] ]
    edges = np.asarray( fr[ "edges" ] ).reshape( -1, 2 )
    pts   = np.asarray( fr[ "points" ] ).reshape( -1 )

    used = np.unique( np.concatenate( [
        np.array( [ i for f in polys for i in f ], np.int64 ),
        edges.reshape( -1 ).astype( np.int64 ),
        pts.astype( np.int64 ) ] ) ) if ( polys or len( edges ) or len( pts ) ) \
        else np.zeros( 0, np.int64 )
    remap = { int( v ): k for k, v in enumerate( used ) }

    verts = [ pool[ used ].reshape( -1, d ) ]
    cells, rgba = [], []                      # ( type, [ indices ] ) and the color of the cell

    for f, ci in zip( polys, fr[ "polygon_colors" ] ):
        cells.append( ( VTK_POLYGON, [ remap[ int( i ) ] for i in f ] ) )
        rgba.append( colors[ int( ci ) ] )
    for ( a, b ), ci in zip( edges, fr[ "edge_colors" ] ):
        cells.append( ( VTK_LINE, [ remap[ int( a ) ], remap[ int( b ) ] ] ) )
        rgba.append( colors[ int( ci ) ] )
    # the points, vectorized: `used` is sorted, so the rank of a vertex is found by bisection
    pnt_ids = np.searchsorted( used, pts.astype( np.int64 ) ) if len( pts ) else np.zeros( 0, np.int64 )
    pnt_rgba = colors[ np.asarray( fr[ "point_colors" ], np.int64 ) ].reshape( -1, 4 )
    pnt_radii = np.asarray( fr[ "point_radii" ], np.float32 ).reshape( -1 )

    # polytopes: they have no vertices, we enumerate them (the scene box bounds them, without
    # which an open polytope would have nothing to show).
    bounds = viz.bounds()
    for dirs, offs, col, with_edges in fr[ "polytopes" ]:
        pv, pe, pf = polytope_mesh( dirs, offs, bounds = bounds )
        if len( pv ) == 0:
            continue
        base = sum( len( v ) for v in verts )
        verts.append( pv )
        for f in pf:
            cells.append( ( VTK_POLYGON, [ base + i for i in f ] ) )
            rgba.append( np.asarray( col, np.float64 ) )
        for a, b in ( pe if with_edges else [] ):
            cells.append( ( VTK_LINE, [ base + int( a ), base + int( b ) ] ) )
            # the edge takes the face's tint, darkened, and always OPAQUE (the opacity
            # of a face makes no sense for a line)
            rgba.append( np.array( [ 0.72 * col[ 0 ], 0.72 * col[ 1 ], 0.72 * col[ 2 ], 1.0 ] ) )

    allv = np.concatenate( verts, axis = 0 ) if verts else np.zeros( ( 0, d ) )

    # the free-connectivity cells ( polygons, edges ), then the points as a block
    conn  = np.array( [ i for _, ids in cells for i in ids ], np.int64 )
    sizes = np.array( [ len( ids ) for _, ids in cells ], np.int64 )
    types = np.array( [ t for t, _ in cells ], np.uint8 )
    conn  = np.concatenate( [ conn, pnt_ids ] )
    offs  = np.cumsum( np.concatenate( [ sizes, np.ones( len( pnt_ids ), np.int64 ) ] ), dtype = np.int64 )
    types = np.concatenate( [ types, np.full( len( pnt_ids ), VTK_VERTEX, np.uint8 ) ] )
    rgba  = np.concatenate( [ np.array( rgba, np.float64 ).reshape( -1, 4 ), pnt_rgba ], axis = 0 )
    radii = np.concatenate( [ np.zeros( len( sizes ), np.float32 ), pnt_radii ] )
    return allv, ( types, conn, offs ), ( rgba * 255 ).astype( np.uint8 ), radii


def write_vtk( viz, filename, axes = ( 0, 1, 2 ), pvd = False ):
    """See `Visualizer.write_vtk`."""
    path = Path( filename )
    if viz.nb_dims is None:
        raise ValueError( "write_vtk: nothing to write (no primitive added)" )

    d = viz.nb_dims
    axes = tuple( a for a in axes if a < d )
    rest = [ k for k in range( d ) if k not in axes ]
    extra_names = [ f"x{ k }" for k in rest ]

    def one( index, out ):
        allv, cells, rgba, radii = _frame_mesh( viz, index, axes )
        coords = np.zeros( ( len( allv ), 3 ), np.float32 )
        for j, a in enumerate( axes ):
            coords[ :, j ] = allv[ :, a ]
        extra = ( allv[ :, rest ] if rest else np.zeros( ( len( allv ), 0 ) ) ).astype( np.float32 )
        return _write_vtu( out, coords, extra, cells, rgba, extra_names, radii )

    if viz.nb_frames == 1 and not pvd:
        return one( 0, path.with_suffix( ".vtu" ) )

    stem = path.with_suffix( "" )
    stem.parent.mkdir( parents = True, exist_ok = True )
    entries = []
    for i in range( viz.nb_frames ):
        out = Path( "%s_%04d.vtu" % ( stem, i ) )
        one( i, out )
        entries.append( ( viz.frame( i )[ "value" ], out.name ) )

    lines = [ '<?xml version="1.0"?>',
            '<VTKFile type="Collection" version="0.1" byte_order="LittleEndian">',
            '  <Collection>' ]
    lines += [ f'    <DataSet timestep="{ t }" group="" part="0" file="{ name }"/>'
             for t, name in entries ]
    lines += [ '  </Collection>', '</VTKFile>', '' ]
    out = stem.with_suffix( ".pvd" )
    out.write_text( "\n".join( lines ) )
    return out
