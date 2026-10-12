#pragma once

#include "Cells.h"
#include "../cell/Engine.h"

// The definition of `engine_run_device` ( see `Cells.h` ): cut `cell` by the planes of the seeds until it is complete, on the card.
// `run<false>` is the cell kept in memory -- the register kernel is written in asimd and is for the CPU.
//
// A PROTOTYPE, off by default ( `SDOT_ENGINE_DEVICE_NOINLINE` / `_EXTERN` ): called across units, the engine measured +14 to +69 % on the
// forward of `measures` in 3D and in 2D up to 20 000 seeds, whatever the register cap given to it, and -26 % in 2D FP32 with 100 000 seeds.

namespace sdot {
namespace engine {

template<class Local,class Seeds>
__device__ __attribute__(( noinline )) int engine_run_device( Local &cell, const Seeds &seeds, SI k0 ) {
    using TK = typename Local::TKernel;
    auto f = seeds.template provider<TK>( k0 );
    return run<false>( cell, f );
}

} // namespace engine
} // namespace sdot
