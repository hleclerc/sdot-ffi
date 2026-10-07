# Getting Started

## Framework

SDOT can work with a lot of different frameworks. Here are the links for the official installation instructions for some of them (one is enough :) ):

- [JAX](https://docs.jax.dev/en/latest/installation.html)
- [PyTorch](https://pytorch.org/get-started/locally/)
- [CuPy](https://docs.cupy.dev/en/stable/install.html)
- [Dask](https://docs.dask.org/en/stable/install.html)
- [NumPy](https://numpy.org/install/)

## Installation using pip

```bash
pip install sdot
```


> **GPU support** — follow the GPU installation instructions of the chosen framework for your platform (CUDA, …). SDOT will automatically use the devices your tensors are on.

Internally, SDOT uses compiled code through dynamic libraries. The packages come bundled with the object files for the most common cases.

For more exotic cases, SDOT may have to generate and compile some code. In this case, one will need a compiler toolchain (SDOT uses `ninja` as build system).

## Installation from the sources

Alternatively, to participate or get the last version, one can use the sources
```
git clone git@github.com:sdot-team/sdot.git
cd sdot
pip install -e .
```

## Hello, World

```python
import numpy as np
import sdot

f = sdot.SumOfDiracs( np.random.rand( 100, 2 ) )  # 100 equal-weight diracs, 2D
g = sdot.Image( np.random.rand( 10, 10 ) )  # 10×10 image (auto-normalized)

plan = sdot.ot_solve( f, g )
plan.show()
```

<script setup>
import { withBase } from 'vitepress'
</script>

<iframe
  :src="withBase('/examples/SdotPlanNd-start.html') + '?wheel=0&touch=0'"
  title="SdotPlanNd interactive viewer"
  loading="lazy"
  allowfullscreen
  style="width: 100%; height: 20vh; min-height: 480px; border: 0; display: block; margin: 1.5rem 0;"
></iframe>

<p style="text-align: right; margin: 0"><a :href="withBase('/examples/SdotPlanNd-start.html')" target="_blank">Open in a separate page ↗</a></p>



## Backend

SDOT is based on [Loom](https://github.com/hleclerc/loom) to achieve backend independance. 

Loom is able to automatically guess the backend and the devices to be used, but it is also possible to specify them.

Loom starts by looking at the loaded modules (`jax`, `torch`, ...). If none of them are present, it looks at installed ones (order is specified in `loom.prefered_frameworks`).

It is possible to 

### Choice of the 

SDOT looks for backend in this order:

1. Modules already imported in the current session (`jax`, `torch`)
2. The `SDOT_FRAMEWORK` environment variable
3. First available library (`jax`, then `torch`)

### Configuration

It is also possible to "force" the parameters, notably during runtime

```python
import sdot

sdot.driver.framework = "jax"      # or "torch"
sdot.driver.dtype     = "FP64"     # FP32, FP64
sdot.driver.device    = "cuda:0"   # or "cpu", "mps", …
```

Or via environment variables:

```bash
export SDOT_DTYPE=FP32
export SDOT_DEVICE=cuda:0
```

All arrays returned by SDOT live in the same framework, so they slot directly into your JAX/PyTorch computation graph.


## What's Next

- [Distributions in depth →](/guide/distributions)
- [Differentiability →](/guide/differentiability)
- [Ground metrics →](/guide/ground-metrics)
- [OT plans in depth →](/guide/distributions)
- [Backend configuration →](/guide/backends)
- [Mathematical background →](/tutorials/ot-plan-intro)
- [Examples gallery →](/examples/)
