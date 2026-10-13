---
title: Gaussian Mixture Fit
description: Minimize Wasserstein distance by LBFGS on Gaussian parameters. Demonstrates the use of gradients.
tags: [gradient, jax, torch, learning]
order: 2
tutorialLink: /tutorials/gaussian-mixture-fit
---

<script setup>
import { withBase } from 'vitepress'
</script>

# Gaussian Mixture Fit

200 points are drawn from a mixture of 4 Gaussians. We look for the mixture back: the `SumOfGaussians` $\rho_\theta$ that is closest to the point cloud $\mu = \frac{1}{n} \sum_i \delta_{x_i}$ in the Wasserstein sense,

$$
\min_\theta \; W_2^2\big( \mu, \; \rho_\theta \big), \qquad \rho_\theta = \frac{1}{\sum_k a_k} \sum_k a_k \, \mathcal{N}( c_k, s_k^2 I ).
$$

The parameters $\theta$ are the centers $c_k$, the sigmas $s_k$ and the masses $a_k$ of the Gaussians. `sdot.w2` is differentiable with respect to all of them, with JAX or PyTorch, and W2 is smooth in $\theta$: a quasi-Newton method fits well, **L-BFGS**.

Unlike a maximum likelihood fit, nothing here can collapse. A Gaussian that shrinks onto a single point does not make the loss go to $-\infty$: it leaves the other points without mass nearby, which W2 charges for.

## The loss

```python
def loss( centers, sigmas, masses ):
    return sdot.w2( sdot.SumOfDiracs( points ), sdot.SumOfGaussians( centers, sigmas, weights = masses ) )
```

The parameters go in as they are. The masses need not sum to 1: `SumOfGaussians` normalizes them. The domain of the transport is the box that holds the Gaussians up to `support_sigmas = 6` standard deviations, recomputed at each evaluation.

## With JAX and with PyTorch

L-BFGS is not written the same way in both: `optax` for JAX, `torch.optim.LBFGS` for PyTorch.

::: code-group

```python [JAX]
import jax, jax.numpy as jnp, optax
jax.config.update( "jax_enable_x64", True )

x = jnp.asarray( points )

def loss( params ):
    centers, sigmas, masses = params
    return sdot.w2( sdot.SumOfDiracs( x ), sdot.SumOfGaussians( centers, sigmas, weights = masses ) )

params = ( centers, sigmas, masses )   # the starting point
opt = optax.lbfgs()
state = opt.init( params )
value_and_grad = optax.value_and_grad_from_state( loss )

@jax.jit   # the line search is a `while_loop`: compiled once, not traced at each iteration
def step( params, state ):
    value, grad = value_and_grad( params, state = state )
    updates, state = opt.update( grad, state, params, value = value, grad = grad, value_fn = loss )
    return optax.apply_updates( params, updates ), state, value

for it in range( 40 ):
    params, state, value = step( params, state )
```

```python [PyTorch]
import torch

x = torch.as_tensor( points )
params = [ torch.tensor( p, requires_grad = True ) for p in ( centers, sigmas, masses ) ]   # the starting point

def closure():
    opt.zero_grad()
    centers, sigmas, masses = params
    value = sdot.w2( sdot.SumOfDiracs( x ), sdot.SumOfGaussians( centers, sigmas, weights = masses ) )
    value.backward()
    return value

# one iteration per `step`, to see them; `max_eval`, since its default ( 5/4 of `max_iter` ) leaves
# the line search no room to shorten a step
opt = torch.optim.LBFGS( params, line_search_fn = "strong_wolfe", max_iter = 1, max_eval = 20 )
for it in range( 40 ):
    value = opt.step( closure )
```

:::

Starting from 4 wide Gaussians of equal masses in the middle, both find the mixture the points were drawn from, up to the sampling noise of 200 points, in about 25 iterations:

| | center | sigma | mass |
|---|---|---|---|
| drawn | ( -1.00, -0.60 ) | 0.35 | 0.35 |
| fitted | ( -0.97, -0.71 ) | 0.34 | 0.30 |
| drawn | ( 1.10, -0.40 ) | 0.25 | 0.25 |
| fitted | ( 1.13, -0.42 ) | 0.26 | 0.25 |
| drawn | ( 0.20, 1.00 ) | 0.30 | 0.25 |
| fitted | ( 0.20, 1.02 ) | 0.27 | 0.24 |
| drawn | ( -0.90, 0.90 ) | 0.20 | 0.15 |
| fitted | ( -0.95, 0.90 ) | 0.18 | 0.21 |

<iframe
  :src="withBase('/examples/gaussian-mixture-fit-torch.html') + '?wheel=0&touch=0'"
  title="Gaussian mixture fit by L-BFGS on W2"
  loading="lazy"
  allowfullscreen
  style="width: 100%; height: 60vh; min-height: 380px; border: 0; display: block; margin: 1.5rem 0 0.5rem;"
></iframe>
<p style="text-align: right; margin: 0"><a :href="withBase('/examples/gaussian-mixture-fit-torch.html')" target="_blank">Open in a separate page ↗</a></p>

The points in gray, and for each Gaussian its center, its circle at one sigma (solid) and at two sigmas (dashed). The opacity follows the mass. Use the slider at the bottom of the viewer (or the play button) to go through the L-BFGS iterations.

## Reproducing the pages

The viewer above is written by [`docs/examples/gaussian_mixture_fit.py`](https://github.com/sdot-team/sdot/blob/main/docs/examples/gaussian_mixture_fit.py), one page per framework:

```bash
python docs/examples/gaussian_mixture_fit.py              # JAX
python docs/examples/gaussian_mixture_fit.py jax torch    # or both
```
