The CATALOGUE of precompiled kernels of the `sdot` wheel -- always present, empty in a checkout.

`python -m sdot.catalogue` ( `make kernels` ) runs the standard usage of sdot on every piece of hardware present, as
every architecture it can behave as, and drops the kernels it compiles into `<tag>/<key>.so` here ( one tag per
architecture: `cpu-x86-64-v3`, `cuda-sm_80`, ... ), next to the runtime they share and an `index.json`.
The wheel ships this directory as is ( it is `sdot/precompiled-kernels` once installed, and the installation may be read-only: nothing writes here at run time ), and `sdot/__init__.py` registers it at import: a kernel that
is in it is not compiled on the user's machine, the first tag the machine can load that holds the kernel's key serves it.
See `loom/src/loom/compilation/catalogue.py`.
