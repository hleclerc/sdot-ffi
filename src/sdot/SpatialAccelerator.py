from loom.util import Aggregate


class SpatialAccelerator( Aggregate ):
    """What answers "WHICH seeds are worth trying, and in what order".

    An accelerator does not know what a cell is. It knows how the seeds are spread out in
    space, and it derives an ENUMERATION from that: instead of the `n - 1` bisectors that
    `PowerDiagram_Plain` tries one by one, it proposes the nearby seeds first and stops
    exploring a region as soon as the caller tells it that it can no longer cut anything. This is the
    only place where the `O(n²)` is decided -- the geometry itself does not change one iota.

    An accelerator is also a storage ORDER: `PowerDiagram_Bsp` arranges the seeds the way
    the tree groups them (`seed_indices`), and it is this arrangement, as much as the pruning, that makes
    the speed (a leaf is read in one go).

    = The contract, C++ side

    The CELL drives (`cell/Engine.h`): it asks a PROVIDER ( `Provider` ) for a half-space, cuts,
    asks again. On the kernel side, an accelerator is thus a supplier -- an object whose method

        template<class State> bool next( const State &e, Local &l, Plane<TK,D> &p );

    fills in the next plane and returns `true`, or `false` when it has nothing left; `e` is the
    cell as it has BECOME (its vertices, in registers or in memory, see `cell/State.h`)
    and `l` a state that the engine ( `Engine` ) allocates per cell (the stack of a tree descent, for example).
    This is where the pruning and the candidate order live; the engine itself has no policy.

    `AaBsp` is the only accelerator today, and its supplier is
    `cell/Providers.h::ProviderBsp`, which reads its tensors (`node_box`, `node_begin` /
    `node_end`, `seed_indices`, the affine upper bound of the weights) and prunes through `cell/Pruning.h` --
    exactly: a box is rejected only if NO vertex of the cell can be cut by a
    seed lying in it. Another accelerator would need its own supplier AND its own
    storage (`PowerDiagram_Xxx.py` / `.h`, modeled on `PowerDiagram_Bsp`), which
    `provider<TK>( k0 )` returns.

    = The contract, Python side

    `nb_seeds()`, so that the diagram checks that the accelerator indexes ITS seeds.
    """

    def nb_seeds( self ):
        """How many seeds it was built on -- or `None` if it does not know.

        An accelerator INDEXES the caller's seeds: built on another cloud, its indices
        designate something else and the answer is wrong without anything saying so. This count
        makes it possible to check this for free, on the caller's side.
        """
        return None

