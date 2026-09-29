"""Le tenseur de travail des kernels de cellules.

Les tableaux d'une cellule dans la forme du noyau ( `cell/Local*.h` : SoA, dans le flottant du
noyau ) et les temporaires de la coupe vivent dans UN tenseur de mots par work-item, que le C++
découpe ( `cell/Scratch.h` ). Sa taille est décidée par l'hôte, avec la formule que le C++ partage
( `Local*::words_for` / `Cell_*.scratch_words` ) : exactement ce qu'il faut pour une opération de
`Cell_*` -- on sait borner ce qu'une coupe produit --, une supposition que loom fait grossir sur
débordement pour un diagramme entier.
"""

import loom
from loom.tensor import Axis, CtShapeVar, IntTensor, ShapeVar
from loom.util import Aggregate


def fp_size( dtype ):
    """32 ou 64, depuis `"FP32"` / `"FP64"` / `float` / un dtype numpy"""
    s = str( dtype ).lower()
    return 64 if "64" in s or s in ( "float", "double" ) else 32


def words_of( itemsize, n ):
    """`n` éléments de `itemsize` octets, arrondis à l'alignement de 32 octets, en mots de 32 bits
    -- `words_of<T>( n )` de `cell/Scratch.h`."""
    return -( -n * itemsize // 32 ) * 8


class CellScratch( Aggregate ):
    """Une ligne de mots par work-item ( ou par item, quand l'appel est batché sur des cellules ),
    et le flottant du noyau en constante de compilation.

    `nb_words` est une SORTIE avec une capacité : c'est par là qu'un kernel dit qu'il a manqué de
    place ( `cell/Ops.h::ask_more` ), et par là que `driver.call` le sait et relance en doublant.
    """
    words          : IntTensor[ "num_thread", "num_word", dict( size = 32 ) ]

    num_thread     : Axis[ "nb_threads" ]
    num_word       : Axis[ "nb_words" ]
    nb_threads     : ShapeVar
    nb_words       : ShapeVar
    kernel_fp_size : CtShapeVar

    @classmethod
    def for_call( cls, nb_words, kernel_dtype, nb_threads = 1, batch_axes = None ):
        """Le scratch d'un appel, DÉJÀ MARQUÉ : `nb_words` mots par ligne, `nb_threads` lignes
        ( batché sur `batch_axes` si l'appel l'est ), prêt à être passé sous le nom que le noyau
        lui donne -- `scratch = CellScratch.for_call( ... )`.

        Il n'y a plus de `name` à répéter : le rôle est porté par la VALEUR, donc le nom de
        l'argument est le nom, et rien ne peut plus se désaccorder entre les deux. C'est aussi ce
        qui a fait disparaître `merge_call`, dont l'unique métier était de fondre ces listes de
        chemins dans celles de l'appel."""
        sc = cls( kernel_fp_size = fp_size( kernel_dtype ), nb_threads = int( nb_threads ), batch_axes = batch_axes )
        return loom.scratch( sc, capacities = { "nb_words": int( nb_words ) } )

    @classmethod
    def for_flat_call( cls, name, nb_words, kernel_dtype, nb_threads = 1, batch_axes = None ):
        """LA FORME PLATE, le temps de la migration : `( scratch, kwargs de l'appel )`. Elle
        disparaît avec le dernier `driver.call` de sdot."""
        sc = cls( kernel_fp_size = fp_size( kernel_dtype ), nb_threads = int( nb_threads ), batch_axes = batch_axes )
        return sc, dict(
            output_capacities = { f"{ name }.nb_words": int( nb_words ) },
            output_attributes = [ name ],
            scratch_attributes = [ name ],
        )


def merge_call( base, extra ):
    """fusionne deux jeux de kwargs de `driver.call` ( capacités, listes de sorties ).

    Ne sert plus qu'aux sites encore en forme plate ; le vocabulaire de marqueurs le rend inutile,
    puisqu'un rôle porté par la valeur n'a rien à fondre dans quoi que ce soit."""
    res = dict( base )
    for k, v in extra.items():
        if isinstance( v, dict ):
            res[ k ] = { **res.get( k, {} ), **v }
        else:
            res[ k ] = list( res.get( k, [] ) ) + list( v )
    return res
