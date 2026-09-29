"""Les germes tels qu'ils sont venus : `positions [ n, d ]`, `weights [ n ]`, et aucune
accélération -- chaque cellule est coupée par les `n - 1` bissectrices, dans l'ordre.

Le plancher contre lequel `PowerDiagram_Bsp` se mesure ( `O( n² )` ), et ce qui reste quand les
positions sont un traceur, sur lequel aucun arbre ne se bâtit. Côté noyau : `PowerDiagram_Plain.h`.
"""

import loom
from loom.tensor import RealTensor

from .PowerDiagram import PowerDiagram


class PowerDiagram_Plain( PowerDiagram ):
    positions : RealTensor[ "num_point", "dim" ]

    # FACULTATIF, comme le domaine : absent ( `Unbound`, `NoneTensor` côté C++ ), le terme de poids
    # disparaît du plan À LA COMPILATION et le diagramme est l'euclidien. « Pas de poids » est un
    # ÉTAT, pas un tableau de zéros.
    weights   : RealTensor[ "num_point" ]

    def _init_seeds( self, positions, weights, accelerator ):
        return { "positions": positions, **( {} if weights is None else { "weights": weights } ) }

    def _ranks_of_items( self ):
        import numpy as np
        return np.arange( int( self.nb_points.value ) )

    def _grad_seeds_expr( self ):
        return "grad_of_inputs.power_diagram.positions, grad_of_inputs.power_diagram.weights"

    # ---- ce que le solveur de `OtPlan` ecrit ( voir `PowerDiagram_Bsp` ) ----------------------------

    def _solver_weights_call( self ):
        w = RealTensor[ self.num_point ]()
        return ( "inputs.power_diagram.with_weights( outputs.weights_out )",
                 dict( weights_out = loom.out( w ) ), dict( weights_out = w ) )

    def _solver_weights_after( self, produced ):
        self.weights = produced[ "weights_out" ].raw
