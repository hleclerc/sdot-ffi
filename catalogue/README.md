The CATALOGUE of precompiled kernels of the `sdot` wheel -- empty in a checkout.

`scripts/build_catalogue.py compile` drops `<tag>/libsdot_kernels.so` + `<tag>/catalogue.json` in
it (one tag per variant: `cpu-x86-64-v3`, `cuda`, ...), from the `catalogue_record/` survey.
The wheel ships this directory as is (`sdot/_catalogue`), and `sdot/__init__.py` registers it at
import: what is in it is not compiled on the user's machine. See
`loom/src/loom/compilation/catalogue.py`.
