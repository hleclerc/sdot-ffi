from loom.util import Aggregate
from loom.tensor import RealTensor


class Distribution( Aggregate ):
    """Base class for probability distributions.

    Subclasses should override `measure` (property) to return the total mass.
    Supports automatic normalization via `normalized_version()` when `target_mass` is set.

    = Integrating a CELL against a distribution -- the kernel-side contract

    A distribution can serve as a MEASURE for `PowerDiagram.measures` : instead of the volume of the
    cell, we then want the integral of the density over it. The split is the same as for the spatial
    accelerators (see `SpatialAccelerator.py`) : the distribution knows how to CUT, the
    diagram knows what is computed ON a piece -- and this is what lets an image, a sum
    of gaussians, or nothing at all, plug in at the same place without a second integrator.

    The C++ class of the same name must therefore offer :

        void for_each_piece( const auto &cell, auto &&ws, auto &&func ) const;

    which cuts `cell` into pieces on which the density is SIMPLE and calls, for each one :

        func( piece, density )

      * `piece`   -- a convex polytope (a `Cell`), the piece. Its cuts keep the `cut_id`
                     they had in the cell (hence the index of the seed they face) ;
                     those added by the cutting carry `BOUNDARY`, i.e. "not a seed".
                     This property alone is what lets the adjoint treat a piece
                     exactly like a cell (`PowerDiagram::integrate_bwd_into`).
      * `density` -- HOW this piece is integrated. Two forms, distinguished at COMPILE TIME
                     by `density.is_constant` :

    1. `is_constant == true` : `density.value` is the density, constant over the piece, and
       `density.add_value_grad( grad_dist, g )` says where to put `g = d(output)/d(this value)` -- the
       mass being linear in it, the integrator passes the VOLUME of the piece, and only the distribution
       knows which slot of its parameters it falls into. This is `ConstantDensity`, what
       an image returns (one slot of `values` per box) and the unit density (no slot).
       The integration is then exact and costs nothing : `value * measure`.

    2. `is_constant == false` : the density INTEGRATES ITSELF, over a simplex --

           TF   integrate_over_simplex    ( const auto &pts ) const;
           void integrate_over_simplex_bwd( const auto &pts, TF g, auto &&grad_pts,
                                            auto &&grad_dist ) const;

       `pts` are the `d + 1` vertices, `grad_pts` the cotangent to accumulate into (the integrator
       glues it back onto the vertices of the piece, then `scatter_cell_grad` goes up to the seeds
       as usual), `grad_dist` that of the density parameters.

    THE INTEGRATOR WRITES NO INTEGRATION RULE. It brings the geometric cutting into
    simplices (`Cell::for_each_simplex`, in any dimension) and nothing else. A density that has a
    closed formula on a simplex -- or a reduction to a special function -- supplies it, and it is
    exact. One that is only a BLACK BOX (a function written elsewhere, a gaussian whose
    formula has not been derived) is wrapped in `PointwiseDensity`, which implements the same contract
    by quadrature from `value_at` / `gradient_at` / `add_value_grad_at` :

        void for_each_piece( const auto &cell, auto &&, auto &&func ) const {
            func( cell, PointwiseDensity{ *this } );
        }

    Quadrature is therefore not a regime of the integrator : it is ONE implementation of the contract,
    next to the exact formulas, and the choice belongs to the density.

    "No distribution" is an ordinary case of the same code, not an absence of code : it is
    `UnitDensity` (density 1, a single piece, the cell), built C++ side by
    `PowerDiagram::unit_density()` -- as `EverySeed` is for the accelerators.

    = The SCRATCH, and why it does not go through here

    Cutting needs room : `Cell::cut` writes into a SEPARATE cell, so a sequence of
    cuts shuttles between two buffers. These two buffers (plus the compaction table of the
    d > 2 regime) are supplied by the CALLER, in `ws` -- a `PieceWorkspace` (see
    `PieceWorkspace.h`), which `PowerDiagram.measures` allocates per work-item like everything else.

    A distribution therefore does not ask for memory : it only says, FROM PYTHON,
    how many more cuts than a cell one of its pieces can carry
    (`extra_cuts_per_piece`), and those cells are sized accordingly. A buffer size
    is not a kernel decision : it is what the platform must know BEFORE
    allocating, so it is said from here.
    """

    def bounding_half_spaces( self ):
        """The SUPPORT of the distribution, as half-spaces `direction . x <= offset`, or `None`.

        A compact-support density (an image) bounds cells for nothing : everything that
        goes beyond its support brings no mass, so cutting it does NOT change the result --
        it is an identity, not an approximation. `PowerDiagram` therefore adds these half-spaces to
        its own (see its `__init__`), and gains three things : the border cells stop
        being infinite, the pruning test of an accelerator becomes usable again
        (`cell_may_be_cut` has nothing to bite on an infinite cell), and the cutting no longer has to
        sweep the whole grid for lack of a bounding box (`Image::_for_each_piece`).

        `None` (the default) says "unbounded support", which is the case of a sum of gaussians :
        truncating it would lose mass, so we do not do it behind its back -- it is then up to the
        caller to give a `box` if they want one.

        Also returns `None` when the geometry is not readable host side (under `jit`) : bounding
        is an OPTIMIZATION, and an optimization that cannot be done must not break
        the call."""
        return None

    def extra_cuts_per_piece( self, nb_dims ):
        """How many more cuts than a cell a PIECE can carry.

        `0` (the default) says "the piece IS the cell" : no cutting, hence no spare cells
        to allocate. An image cuts by the 2d planes of a box of its grid, hence
        `2 * nb_dims`. This is the ONLY sizing information that the contract asks for ; see
        the class docstring for why it goes through Python and not through the kernel."""
        return 0

    # current_mass   : Tensor...
    target_mass      : RealTensor
    @property
    def mass( self ):
        """Total mass/measure of this distribution. Implemented by subclasses."""
        if self.current_mass.is_undefined:
            self._update_current_mass()
        return self.current_mass

    def normalized_version( self ):
        """Return a version of this distribution normalized to target_mass, if specified.

        If target_mass is not set, returns self unchanged.
        If target_mass is set, returns a copy with values scaled so that measure == target_mass.
        """
        return self

    def _update_current_mass( self ):
        """  """
        raise NotImplementedError

    def raw_1d_diracs( self ):
        """For a 1D dirac-source distribution (`_is_dirac_source`): `( weights, batched_extra,
        project_fn )`, letting a target distribution's `try_update_sdotplan1d` read plain,
        differentiable backend arrays and bypass `driver.call` entirely (ordinary autodiff
        differentiates straight through). `None` when this distribution cannot supply this
        cheaply (default: unsupported) -- the caller then falls back to the general
        driver.call/C++ path.

        - `weights`: `[ nb_diracs ]` (shared across the batch) or `[ *batch, nb_diracs ]`.
        - `batched_extra`: a dict of this distribution's OWN per-batch-element leaves needed to
          compute positions (e.g. a per-angle projection normal) -- EMPTY if positions do not
          depend on the batch. The caller merges these into whatever it maps/vmaps over, so
          they are sliced one batch element at a time, not materialized in full.
        - `project_fn( extra )`: given one slice of `batched_extra` (same keys, each now
          unbatched -- `{}` if `batched_extra` is empty), returns this distribution's `[ n ]`
          1D positions for that one batch element. Deferred like this (a function, not a
          materialized array) so a projection that DOES depend on the batch (e.g.
          `ProjectedSumOfDiracs`'s `points·normal`) is computed LAZILY, one angle at a time,
          inside the caller's `lax.map` -- eagerly materializing it for every batch element
          upfront would defeat the whole point of mapping instead of vmapping (see
          `Image.try_update_sdotplan1d`)."""
        return None

    def try_update_sdotplan1d( self, plan ):
        """Attempt to solve `plan` (an `SdotPlan1d` with `self` as one of its two
        distributions) without going through `driver.call` -- e.g. a closed-form, pure-JAX
        computation. On success: update `plan`'s output fields (at least `plan.cost`) and
        return True. On failure (unsupported combination): change nothing and return False,
        so the caller uses the general driver.call/C++ path instead. Default: always decline
        (default: unsupported)."""
        return False

    def batch_slice( self, index ):
        """An UNBATCHED version of `self` for one element (`index`, a traced int) of its
        (single) batch axis -- lets `SdotPlan1d` loop over the batch with `jax.lax.map` (one
        instance, and so one `driver.call`, per iteration) instead of a single call handling
        every batch element's memory at once. This is what lets the driver.call/C++ path scale
        to a large batch count the same way `Image.try_update_sdotplan1d`'s own `lax.map` already
        does for the pure-JAX path: peak memory bounded by ONE batch element, not the total
        count (see `SdotPlan1d._update_outputs_via_angle_loop`). `None` when unsupported (no
        batch axis, or this distribution type does not know how to slice itself) -- the caller
        then falls back to its previous, single-call batched behavior."""
        return None
