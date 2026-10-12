#pragma once

#include <loom/support/common_macros.h> // HD
#include <type_traits>
#include "Seeds.h"
#include "../cell/Local1.h"
#include "../cell/Local2.h"
#include "../cell/LocalN.h"

// =====================================================================================
// THE CELL ENGINE, AS A UNIT: `engine_run( cell, seeds, k0 )` cuts the cell `cell` ( loaded with the domain ) by
// the planes of the seeds, until it is complete. It is DECLARED here and DEFINED by the `sdot_engine` module (`PowerDiagram._engine_unit`), which a
// kernel links as a shared library ( `FfiCode( sources = ... )`, see `PowerDiagram.ffi_units` ): the engine is not
// instantiated again in every kernel that builds cells -- measures, moments, cells, hessian rows, their adjoints.
//
// The cost is a call per cell, with the cell by reference ( it lives in the work-item's scratch ): measured at
// -4 to +5 % on the forward of `measures` and `moments`, in 2D and 3D, FP32 and FP64, on arm64 and x86
// ( `bench/noinline.py` ).
// =====================================================================================

namespace sdot {
namespace engine {

/// the cell type the engine works on, for a dimension and a kernel float -- the one `Cell_{1,2,N}::Local` gives
template<int D,class TK>
using LocalFor = std::conditional_t<D == 1, Local1<TK>, std::conditional_t<D == 2, Local2<TK>, LocalN<TK,D>>>;

/// the dimensions whose engine is a unit; `LOOM_..` : see `PowerDiagram.ffi_units`, which must say the same
constexpr bool has_unit( int D ) { return D == 2 || D == 3; }

template<class Local,class Seeds>
__attribute__(( visibility( "default" ) ))
int engine_run( Local &cell, const Seeds &seeds, SI k0 );

#if defined( __CUDACC__ ) && ( defined( SDOT_ENGINE_DEVICE_NOINLINE ) || defined( SDOT_ENGINE_DEVICE_EXTERN ) )
// THE SAME ENGINE ON THE CARD ( prototype ). A kernel calls it as a function instead of inlining it:
//   SDOT_ENGINE_DEVICE_NOINLINE   defined here, in the kernel's own unit, but not inlined ( to measure what a call costs )
//   SDOT_ENGINE_DEVICE_EXTERN     only declared: another unit ( `engine/CellsDevice.h`, compiled with relocatable device code ) defines it,
//                                 and the two are linked ( at load time, with nvJitLink )
template<class Local,class Seeds>
__device__ int engine_run_device( Local &cell, const Seeds &seeds, SI k0 );
#endif

} // namespace engine
} // namespace sdot

#if defined( __CUDACC__ ) && defined( SDOT_ENGINE_DEVICE_NOINLINE )
#include "CellsDevice.h"
#endif
