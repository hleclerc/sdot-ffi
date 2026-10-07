"""The quick-display methods, shared by everything that knows how to draw itself.

A class that exposes `add_to_viz( viz, **kwargs )` ( `Cell`, `PowerDiagram`, `SdotPlanNd`, ... ) inherits
from this mixin and gets, without more code, the ways of looking at itself:

    obj.write_html( "diagram.html" )     # a self-contained page ( WebGL, opens as `file://` )
    obj.write_pvd ( "diagram.pvd"  )     # ParaView: a `.pvd` ( + its `.vtu` ), even for a single frame
    obj.write_vtk ( "diagram.vtu"  )     # ParaView: a lone `.vtu`, a `.pvd` only when there are frames
    obj.show()                           # writes a temporary page and opens it in the browser
    viz = obj.to_viz()                   # the `Visualizer` itself, to add other things to it

`kwargs` go to `add_to_viz`, so that what is specific to the object ( `opacity`, `seeds`, ... ) is
asked for at the call. To draw several things in ONE scene -- or the successive states of a
descent, with `viz.new_frame` -- build the `Visualizer` and use `viz.add( obj, ... )`.
"""

from pathlib import Path


class Displayable:
    def to_viz( self, viz = None, title = None, **kwargs ):
        """A `Visualizer` holding this object ( `viz`, when given: it is then completed ). `title`
        names a NEW scene; `kwargs` are those of `add_to_viz`."""
        from .Visualizer import Visualizer

        if viz is None:
            viz = Visualizer( title = title or type( self ).__name__ )
        viz.add( self, **kwargs )
        return viz

    def write_html( self, filename, title = None, **kwargs ):
        """Writes the self-contained HTML page of this object. Returns the path written."""
        return self.to_viz( title = title, **kwargs ).write_html( filename )

    def write_vtk( self, filename, title = None, axes = ( 0, 1, 2 ), **kwargs ):
        """Writes this object for ParaView, as a `.vtu`. Returns the path written."""
        return self.to_viz( title = title, **kwargs ).write_vtk( filename, axes = axes )

    def write_pvd( self, filename, title = None, axes = ( 0, 1, 2 ), **kwargs ):
        """Writes this object for ParaView as a `.pvd` collection ( with its `.vtu` beside it ),
        the entry point one opens. Returns the `.pvd` path."""
        return self.to_viz( title = title, **kwargs ).write_vtk( filename, axes = axes, pvd = True )

    def show( self, filename = None, title = None, **kwargs ):
        """Writes the HTML page ( `filename`, or a temporary file ) and opens it in the browser.
        Returns the path written."""
        import tempfile
        import webbrowser

        if filename is None:
            filename = Path( tempfile.mkdtemp( prefix = "sdot_viz_" ) ) / f"{ type( self ).__name__ }.html"
        path = Path( self.write_html( filename, title = title, **kwargs ) )
        webbrowser.open( path.resolve().as_uri() )
        return path
