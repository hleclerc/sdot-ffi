"""The working tensor of the cell kernels.

A cell's arrays in the kernel's form ( `cell/Local*.h`: SoA, in the kernel's float ) and the
temporaries of the cut live in ONE tensor of words per work-item, which the C++ slices up
( `cell/Scratch.h` ). Its size is decided by the host, with the formula the C++ shares
( `Local*::words_for` / `Cell_*.scratch_words` ): exactly what a `Cell_*` operation needs -- we
can bound what a cut produces --, a guess that loom grows on overflow for a whole diagram.
"""

import loom
from loom.tensor import Axis, CtShapeVar, IntTensor, ShapeVar
from loom.util import Aggregate


def fp_size( dtype ):
    """32 or 64, from `"FP32"` / `"FP64"` / `float` / a numpy dtype"""
    s = str( dtype ).lower()
    return 64 if "64" in s or s in ( "float", "double" ) else 32


def words_of( itemsize, n ):
    """`n` elements of `itemsize` bytes, rounded up to the 32-byte alignment, in 32-bit words
    -- `words_of<T>( n )` of `cell/Scratch.h`."""
    return -( -n * itemsize // 32 ) * 8


class CellScratch( Aggregate ):
    """One row of words per work-item ( or per item, when the call is batched over cells ),
    and the kernel's float as a compile-time constant.

    `nb_words` is an OUTPUT with a capacity: it is how a kernel says it ran out of room
    ( `cell/Ops.h::ask_more` ), and how `loom.ffi_call` finds out and retries with double the size.
    """
    words          : IntTensor[ "num_thread", "num_word", dict( size = 32 ) ]

    num_thread     : Axis[ "nb_threads" ]
    num_word       : Axis[ "nb_words" ]
    nb_threads     : ShapeVar
    nb_words       : ShapeVar
    kernel_fp_size : CtShapeVar

    @classmethod
    def for_call( cls, nb_words, kernel_dtype, nb_threads = 1, batch_axes = None ):
        """The scratch of a call, ALREADY MARKED: `nb_words` words per row, `nb_threads` rows
        ( batched over `batch_axes` if the call is ), ready to be passed under the name the kernel
        gives it -- `scratch = CellScratch.for_call( ... )`.

        There is no `name` left to repeat: the role is carried by the VALUE, so the argument's
        name is the name, and nothing can get out of sync between the two. This is also what made
        `merge_call` disappear, whose only job was to merge these path lists into the call's."""
        sc = cls( kernel_fp_size = fp_size( kernel_dtype ), nb_threads = int( nb_threads ), batch_axes = batch_axes )
        return loom.scratch( sc, capacities = { "nb_words": int( nb_words ) } )
