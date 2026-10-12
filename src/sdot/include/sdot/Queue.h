#pragma once

#include <loom/support/kernels/CpuQueue.h> // IWYU pragma: keep -- the default queue

namespace sdot {

/// The queue on which a kernel runs : the execution context of a `driver.call`.
///
/// The choice of device is a TYPEDEF, not a runtime test : the memory zone in which a
/// pointer lives is part of its type (see `Ptr.h`), so it is the device that decides the type of the
/// views the kernel manipulates. The generated source includes the header of ITS queue and sets `SDOT_QUEUE`
/// before including this one (see `Device.cpp_queue_include` / `cpp_queue_type` on the python side) ;
/// by default, the CPU -- what a hand-compiled source that defines nothing gets.
///
/// `run_parallel` accepts this queue alone, or a list of queues when there is a context to
/// choose from (it then takes the cheapest, transfers included).
#ifndef SDOT_QUEUE
#   define SDOT_QUEUE CpuQueue
#endif

using Queue = SDOT_QUEUE;

} // namespace sdot
