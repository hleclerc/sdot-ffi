# Getting Started

## Installation

```bash
pip install sdot
```

Alternatively, to get the latest version (or to participate), one can use the sources:

```
git clone git@github.com:sdot-team/sdot.git
cd sdot
pip install -e .
```

## Framework

SDOT can work with a lot of different frameworks. Here are the links for the official installation instructions for some of them (one is enough for most of the applications :) ):

- [JAX](https://docs.jax.dev/en/latest/installation.html)
- [PyTorch](https://pytorch.org/get-started/locally/)
- [CuPy](https://docs.cupy.dev/en/stable/install.html)
- [Dask](https://docs.dask.org/en/stable/install.html)
- [NumPy](https://numpy.org/install/)

> **GPU support** — follow the GPU installation instructions of the chosen framework for your platform (CUDA, …). SDOT will automatically use the devices your tensors are on, but some of the frameworks need specific installation instructions.

## Compilers

SDOT uses dynamic libraries that can be generated and compiled on demand. 

Most of them will come with the packages installed via `pip`, but for more "exotic" case, you may need a compiler toolchain. In such a cas, you will need to install `ninja`, a C++ compiler, and `nvcc` if you plan to compile for Nvidia GPUs.

## Hello, World

Here is a simple program to test if the installation works.

:::code-group

```python [NumPy]
import numpy as np
import sdot

f = sdot.SumOfDiracs( np.random.rand( 100, 2 ) )  # 100 equal-weight diracs, 2D
g = sdot.Image( np.random.rand( 10, 10 ) )  # 10×10 image (auto-normalized)

plan = sdot.ot_solve( f, g )
plan.show()
```

```python [JAX]
import jax
import sdot

f = sdot.SumOfDiracs( jax.random.uniform( jax.random.key( 0 ), ( 100, 2 ) ) )  # 100 equal-weight diracs, 2D
g = sdot.Image( jax.random.uniform( jax.random.key( 0 ), ( 10, 10 ) ) )  # 10×10 image (auto-normalized)

plan = sdot.ot_solve( f, g )
plan.show()
```

```python [PyTorch]
import torch
import sdot

f = sdot.SumOfDiracs( torch.rand( 100, 2 ) )  # 100 equal-weight diracs, 2D
g = sdot.Image( torch.rand( 10, 10 ) )  # 10×10 image (auto-normalized)

plan = sdot.ot_solve( f, g )
plan.show()
```

```python [CuPy]
import cupy as cp
import sdot

f = sdot.SumOfDiracs( cp.random.rand( 100, 2 ) )  # 100 equal-weight diracs, 2D
g = sdot.Image( cp.random.rand( 10, 10 ) )  # 10×10 image (auto-normalized)

plan = sdot.ot_solve( f, g )
plan.show()
```

:::

<script setup>
import { withBase } from 'vitepress'
</script>

<iframe
  :src="withBase('/examples/SdotPlanNd-start.html') + '?wheel=0&touch=0'"
  title="SdotPlanNd interactive viewer"
  loading="lazy"
  allowfullscreen
  style="width: 100%; height: 18vh; min-height: 380px; border: 0; display: block; margin: 1.5rem 0;"
></iframe>

<p style="text-align: right; margin: 0"><a :href="withBase('/examples/SdotPlanNd-start.html')" target="_blank">Open in a separate page ↗</a></p>



<!-- ## Backend

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

All arrays returned by SDOT live in the same framework, so they slot directly into your JAX/PyTorch computation graph. -->


## What's Next

- [Examples gallery →](/examples/)
- [Distributions in depth →](/guide/distributions)
- [Differentiability →](/guide/differentiability)
- [Ground metrics →](/guide/ground-metrics)
- [OT plans in depth →](/guide/distributions)
- [Backend configuration →](/guide/backends)
- [Mathematical background →](/tutorials/ot-plan-intro)
