---
layout: home

hero:
  name: SDOT
  text: Semi-Discrete Optimal Transport
  tagline: Fast · Differentiable · N-dimensional · Framework independant (JAX/PyTorch/Cupy/Dask...)
  image:
    src: /examples/img/patate.png
    alt: VitePress
  actions:
    - theme: brand
      text: Get Started
      link: /guide/getting-started
    - theme: alt
      text: Examples
      link: /examples/
    - theme: alt
      text: GitHub
      link: https://github.com/sdot-team/sdot

features:
  - icon: 📦
    title: Many target distributions
    details: Images, Meshes, SumOfGaussians... all supporting smoothing for faster convergence
  - icon: 🔗
    title: Differentiable end-to-end
    details: Use sdot as a differentiable layer inside JAX jit / grad or a PyTorch autograd graph.
  - icon: 📐
    title: Any number of dimensions
    details: 2D, 6D... — the same API and the same convergence behavior.
  - icon: 🔄
    title: Flexible ground metrics
    details: Euclidean (default), periodic, entropic regularization, and custom radial kernels.
  - icon: 🚀
    title: Highly optimized
    details: SIMD, CUDA, Metal... with batch variants for data-parallel workloads.
  - icon: ⚡
    title: O(n log n) — not O(n²)
    details: Fast newton solvers on power diagrams. Memory efficient. Parallel version (Dask)
---

## Hello, World

Here is a computation of the Wasserstein 2 distance between a sum of diracs and an image:

:::code-group

```python [PyTorch]
import torch
import sdot

f = sdot.SumOfDiracs( torch.rand( 100, 2 ) )  # 100 equal-weight diracs, 2D
g = sdot.Image( torch.rand( 10, 10 ) )  # 10×10 image (auto-normalized)

print( sdot.w2( f, g ) )
```

```python [JAX]
import jax
import sdot

f = sdot.SumOfDiracs( jax.random.uniform( jax.random.key( 0 ), ( 100, 2 ) ) )  # 100 equal-weight diracs, 2D
g = sdot.Image( jax.random.uniform( jax.random.key( 0 ), ( 10, 10 ) ) )  # 10×10 image (auto-normalized)

print( sdot.w2( f, g ) )
```

```python [NumPy]
import numpy as np
import sdot

f = sdot.SumOfDiracs( np.random.rand( 100, 2 ) )  # 100 equal-weight diracs, 2D
g = sdot.Image( np.random.rand( 10, 10 ) )  # 10×10 image (auto-normalized)

print( sdot.w2( f, g ) )
```

```python [CuPy]
import cupy as cp
import sdot

f = sdot.SumOfDiracs( cp.random.rand( 100, 2 ) )  # 100 equal-weight diracs, 2D
g = sdot.Image( cp.random.rand( 10, 10 ) )  # 10×10 image (auto-normalized)

print( sdot.w2( f, g ) )
```

:::

<!-- If not already loaded or specified, Sdot automatically picks the best available backend (JAX, PyTorch, ...) and device. [Configure it →](/guide/backends) -->

## How the transport plans are computed?

Given a discrete measure **f** (a weighted sum of Dirac masses) and a continuous density **g**, semi-discrete OT finds the _power diagram_ (Laguerre tessellation) that partitions space so that the mass of each cell matches the weight of the corresponding Dirac.

The solution is unique, and can be computed via a provably-convergent Newton algorithm in **O(n log n)** time — making it practical for very large number of points in 2D, 3D or more, and can easily achieves machine precision if required.

$$f = \sum_i m_i \, \delta_{y_i}, \qquad g = \text{continuous density on } \Omega$$

$$\text{Find } w \text{ such that } \int_{C_i(w)} g(x) \, dx = m_i \quad \forall i$$

> **Reference:** Kitagawa, Mérigot, Thibert —
> _Convergence of a Newton algorithm for semi-discrete optimal transport_, JEMS 2019.

## Performance

A Newton solve needs only a handful of iterations whatever the number of diracs. Each iteration is a power diagram plus a
multigrid solve, and both are almost linear in `n`. On an 8-core CPU, a 2D plan with 10⁵ diracs is solved to 10⁻⁶
relative accuracy in about **1 s** (3D: 2.3 s). On a GPU the same solve takes **0.07 s** (3D: 0.14 s), and 10⁶ diracs in 2D
take **0.4 s**. That is 10 to 17x faster than the CPU.

[Detailed benchmarks →](/guide/performance) (single diagram, CPU vs GPU, image and gaussian targets, scaling with the
number of diracs and with the dimension, comparison other libraries)

Note: 1D benchmarks still need to be added.

## Target densities

To define a density, you essentially need to be able to integrate over simplices, and to provide a way to smooth the density when it is irregular.

Sdot provides a number of predefined target densities:
- [images](/guide/distributions#polynomialgrid)
- [splines](/guide/distributions#splinegrid)
- [sums of Gaussians](/guide/distributions#sumofgaussians) _(coming soon)_
- [Lebesgue measure on polytopes](/guide/distributions#polynomialcells) _(coming soon)_
- ...

For more details, see the [Distributions guide](/guide/distributions).

## Differentiation

In principle, sdot supports backward differentiation of any variable with respect to any variable.

See the [Differentiability guide](/guide/differentiability) and the [Examples gallery](/examples/) to go further.

<!-- 
SDOT has been used for Quantization & sampling, Meshing, Machine learning, PDEs, Registration...

See the [Examples gallery →](/examples/) for more illustrations -->
